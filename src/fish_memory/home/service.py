from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast
from uuid import UUID

from fish_memory.answer.context import ChatMessage
from fish_memory.home.brightness import resolve_brightness_value
from fish_memory.home.catalog import (
    HomeEntityCatalog,
    limit_candidates_by_device,
    normalize_home_alias,
    requested_home_domains,
)
from fish_memory.home.detection import looks_like_home_request
from fish_memory.home.executor import HomeActionClient, HomeActionExecutor
from fish_memory.home.models import (
    ExcludedHomeTarget,
    HomeAction,
    HomeActionStatus,
    HomeCandidate,
    HomeControlPlan,
    HomeExecutionResult,
    HomeIntentType,
    HomeOperation,
    HomeQueryPlan,
    HomeRouteDecision,
    HomeRunOutcome,
    HomeScopePlan,
    HomeSelection,
    HomeSetCommand,
    ResolvedQueryScope,
)
from fish_memory.home.parser import LocalHomeCommandParser
from fish_memory.home.preparation import (
    CatalogControlEvaluator,
    PreparedHomeRequest,
    freeze_catalog,
    thaw_home_value,
)
from fish_memory.home.router import HomeRouter
from fish_memory.home.scope_planner import HomeScopePlanner
from fish_memory.home.scope_resolver import HomeScopeResolver
from fish_memory.home.selector import HomeDeviceSelector
from fish_memory.home.targeting import has_explicit_set_request
from fish_memory.models import Message, visible_message_content
from fish_memory.performance.tracker import MessagePerformanceTracker

logger = logging.getLogger(__name__)

_ROUTER_HISTORY_LIMIT = 6


class ActionStore(Protocol):
    async def create_run(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        source_message_id: UUID,
        action: Mapping[str, object] | None = None,
    ) -> Any: ...

    async def update_run(
        self,
        run_id: UUID,
        *,
        status: HomeActionStatus,
        action: Mapping[str, object] | None = None,
        entity_id: str | None = None,
        service_domain: str | None = None,
        service_name: str | None = None,
        accepted: bool = False,
        confirmed: bool = False,
        result: object | None = None,
        error: str | None = None,
    ) -> Any: ...


class StepStore(Protocol):
    async def insert_steps(self, message_id: UUID, steps: Any) -> None: ...


class HistoryStore(Protocol):
    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int | None = None,
    ) -> list[Message]: ...


@dataclass(frozen=True, slots=True)
class HomeRunHandle:
    _task: asyncio.Task[HomeRunOutcome]

    async def result(self) -> HomeRunOutcome:
        return await asyncio.shield(self._task)


class HomeAutomationService:
    def __init__(
        self,
        *,
        client: HomeActionClient,
        store: ActionStore,
        history_store: HistoryStore,
        router: HomeRouter,
        scope_planner: HomeScopePlanner | None = None,
        selector: HomeDeviceSelector | None = None,
        performance_store: StepStore | None,
        timeout_seconds: float = 5.0,
        concurrency: int = 8,
    ) -> None:
        self._client = client
        self._store = store
        self._history_store = history_store
        self._router = router
        self._scope_planner = scope_planner
        self._selector = selector
        self._performance_store = performance_store
        self._timeout = timeout_seconds
        self._semaphore = asyncio.Semaphore(concurrency)
        self._parser = LocalHomeCommandParser()
        self._executor = HomeActionExecutor(client)
        self._tasks: set[asyncio.Task[Any]] = set()

    def start(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        content: str,
        message_sequence: int,
        *,
        intent: HomeIntentType,
        origin_ns: int | None = None,
    ) -> HomeRunHandle:
        if self._scope_planner is None or self._selector is None:
            raise RuntimeError("scoped Home routing is not configured")
        task = asyncio.create_task(
            self._process_scoped(
                fish_id,
                session_id,
                message_id,
                content,
                message_sequence,
                intent=intent,
                origin_ns=origin_ns,
            ),
            name=f"home-scoped-{message_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return HomeRunHandle(task)

    async def prepare(
        self,
        content: str,
        *,
        intent: HomeIntentType,
        history: Sequence[ChatMessage],
        tracker: MessagePerformanceTracker,
    ) -> PreparedHomeRequest:
        """Plan against frozen local state without creating a run or acting."""
        if self._scope_planner is None or self._selector is None:
            raise RuntimeError("scoped Home routing is not configured")
        try:
            catalog = freeze_catalog(self._client.catalog)
            if not catalog.snapshot.ready:
                return _prepared_failure(
                    intent, "Home Assistant 未连接或设备目录尚未就绪"
                )
            frozen_history: tuple[ChatMessage, ...] = tuple(
                {"role": item["role"], "content": item["content"]} for item in history
            )
            async with self._semaphore:
                plan = await asyncio.wait_for(
                    self._scope_planner.plan(
                        content,
                        intent,
                        areas=catalog.area_facets(),
                        history=frozen_history,
                        tracker=tracker,
                    ),
                    timeout=self._timeout,
                )
                resolver = HomeScopeResolver(catalog)
                if intent is HomeIntentType.QUERY:
                    if not isinstance(plan, HomeQueryPlan):
                        raise ValueError("query intent requires a query scope plan")
                    return PreparedHomeRequest(
                        intent=intent,
                        plan=plan,
                        outcome=HomeRunOutcome(
                            intent=intent,
                            status=HomeActionStatus.SUCCEEDED,
                            query=resolver.resolve_query(plan),
                        ),
                    )
                if not isinstance(plan, HomeControlPlan):
                    raise ValueError("control intent requires a control scope plan")
                return await self._prepare_scoped_control(
                    content, frozen_history, plan, resolver, catalog, tracker
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception("Home request preparation failed")
            return _prepared_failure(intent, f"{type(error).__name__}: {error}")

    def start_prepared(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        prepared: PreparedHomeRequest,
        *,
        origin_ns: int | None = None,
    ) -> HomeRunHandle:
        """Commit an already prepared request without re-planning its targets."""
        task = asyncio.create_task(
            self._process_prepared(
                fish_id, session_id, message_id, prepared, origin_ns=origin_ns
            ),
            name=f"home-prepared-{message_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return HomeRunHandle(task)

    def notify_user_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        content: str,
        message_sequence: int,
        *,
        intent: HomeIntentType = HomeIntentType.CONTROL,
        origin_ns: int | None = None,
    ) -> None:
        task = asyncio.create_task(
            self._process(
                fish_id,
                session_id,
                message_id,
                content,
                message_sequence,
                intent=intent,
                origin_ns=origin_ns,
            ),
            name=f"home-action-{message_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def close(self) -> None:
        await self.drain()

    async def _process_scoped(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        content: str,
        message_sequence: int,
        *,
        intent: HomeIntentType,
        origin_ns: int | None,
    ) -> HomeRunOutcome:
        tracker = MessagePerformanceTracker(origin_ns=origin_ns)
        total_started = tracker.now_ns()
        try:
            try:
                history = await self._load_router_history(
                    fish_id, session_id, message_sequence
                )
                prepared = await self.prepare(
                    content, intent=intent, history=history, tracker=tracker
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.exception(
                    "Scoped Home preparation failed for message %s", message_id
                )
                prepared = _prepared_failure(intent, f"{type(error).__name__}: {error}")
            return await self._commit_prepared(
                fish_id, session_id, message_id, prepared
            )
        finally:
            await self._record_scoped_performance(message_id, tracker, total_started)

    async def _process_prepared(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        prepared: PreparedHomeRequest,
        *,
        origin_ns: int | None,
    ) -> HomeRunOutcome:
        tracker = MessagePerformanceTracker(origin_ns=origin_ns)
        total_started = tracker.now_ns()
        try:
            return await self._commit_prepared(
                fish_id, session_id, message_id, prepared
            )
        finally:
            await self._record_scoped_performance(message_id, tracker, total_started)

    async def _record_scoped_performance(
        self,
        message_id: UUID,
        tracker: MessagePerformanceTracker,
        total_started: int,
    ) -> None:
        if self._performance_store is not None:
            tracker.record(
                "home.total",
                started_ns=total_started,
                finished_ns=tracker.now_ns(),
            )
            await self._performance_store.insert_steps(message_id, tracker.snapshot())

    async def _prepare_scoped_control(
        self,
        content: str,
        history: tuple[ChatMessage, ...],
        plan: HomeControlPlan,
        resolver: HomeScopeResolver,
        catalog: HomeEntityCatalog,
        tracker: MessagePerformanceTracker,
    ) -> PreparedHomeRequest:
        resolved = resolver.resolve_control(plan, CatalogControlEvaluator(catalog))
        candidates = resolved.eligible
        if plan.scope.selection is HomeSelection.COUNT:
            assert plan.scope.count is not None
            if len(candidates) < plan.scope.count:
                return _prepared_failure(
                    HomeIntentType.CONTROL,
                    "可操作设备数量不足",
                    plan=plan,
                    excluded=resolved.excluded,
                    satisfied=resolved.satisfied,
                )
            candidates = candidates[: plan.scope.count]

        if plan.scope.selection is HomeSelection.SPECIFIC:
            if not candidates:
                return _prepared_failure(
                    HomeIntentType.CONTROL,
                    "范围内没有可操作设备",
                    plan=plan,
                    excluded=resolved.excluded,
                    satisfied=resolved.satisfied,
                )
            assert self._selector is not None
            actions = await asyncio.wait_for(
                self._selector.select(
                    content,
                    candidates,
                    operation=plan.operation,
                    value=plan.value,
                    history=history,
                    tracker=tracker,
                ),
                timeout=self._timeout,
            )
            if not actions:
                return _prepared_failure(
                    HomeIntentType.CONTROL,
                    "无法唯一确定设备",
                    plan=plan,
                    status=HomeActionStatus.AMBIGUOUS,
                    excluded=resolved.excluded,
                    satisfied=resolved.satisfied,
                )
            by_entity = {
                candidate.entity.entity_id: candidate for candidate in candidates
            }
            actions = tuple(
                _planned_action(by_entity[action.entity_id], plan) for action in actions
            )
        else:
            actions = tuple(
                _planned_action(candidate, plan) for candidate in candidates
            )

        if not actions:
            if resolved.satisfied and not resolved.excluded:
                return PreparedHomeRequest(
                    intent=HomeIntentType.CONTROL,
                    plan=plan,
                    satisfied=resolved.satisfied,
                    outcome=HomeRunOutcome(
                        intent=HomeIntentType.CONTROL,
                        status=HomeActionStatus.SUCCEEDED,
                        satisfied=resolved.satisfied,
                    ),
                )
            return _prepared_failure(
                HomeIntentType.CONTROL,
                "范围内没有可操作设备",
                plan=plan,
                excluded=resolved.excluded,
                satisfied=resolved.satisfied,
            )

        return PreparedHomeRequest(
            intent=HomeIntentType.CONTROL,
            plan=plan,
            actions=actions,
            excluded=resolved.excluded,
            satisfied=resolved.satisfied,
        )

    async def _commit_prepared(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        prepared: PreparedHomeRequest,
    ) -> HomeRunOutcome:
        run = await self._store.create_run(
            fish_id=fish_id,
            session_id=session_id,
            source_message_id=message_id,
            action={},
        )
        try:
            outcome = prepared.outcome
            if outcome is not None:
                if outcome.query is not None:
                    assert isinstance(prepared.plan, HomeQueryPlan)
                    await self._store.update_run(
                        run.id,
                        status=outcome.status,
                        action={
                            "operation": HomeOperation.QUERY.value,
                            "scope": _scope_payload(prepared.plan.scope),
                        },
                        accepted=True,
                        confirmed=True,
                        result=_query_result_payload(outcome.query),
                    )
                elif outcome.status is HomeActionStatus.SUCCEEDED:
                    await self._store.update_run(
                        run.id,
                        status=outcome.status,
                        action={"actions": []},
                        accepted=True,
                        confirmed=True,
                        result={"satisfied": _actions_payload(outcome.satisfied)},
                    )
                else:
                    await self._finish_scoped_failure(
                        run.id,
                        outcome.intent,
                        outcome.error or "Home Assistant 请求准备失败",
                        status=outcome.status,
                        excluded=outcome.excluded,
                        satisfied=outcome.satisfied,
                    )
                return outcome
            if not isinstance(prepared.plan, HomeControlPlan) or not prepared.actions:
                raise ValueError("prepared control requires a plan and actions")
            async with self._semaphore:
                return await self._execute_prepared_control(run.id, prepared)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.exception(
                "Prepared Home execution failed for message %s", message_id
            )
            return await self._finish_scoped_failure(
                run.id,
                prepared.intent,
                f"{type(error).__name__}: {error}",
                excluded=prepared.excluded,
                satisfied=prepared.satisfied,
            )

    async def _execute_prepared_control(
        self, run_id: UUID, prepared: PreparedHomeRequest
    ) -> HomeRunOutcome:
        assert isinstance(prepared.plan, HomeControlPlan)
        actions = prepared.actions
        action_payload = _actions_payload(actions)
        await self._store.update_run(
            run_id,
            status=HomeActionStatus.EXECUTING,
            action=action_payload,
            entity_id=actions[0].entity_id if len(actions) == 1 else None,
        )
        results = await _execute_actions(
            actions,
            executor=self._executor,
            timeout_seconds=self._timeout,
        )
        errors = [
            f"{action.entity_id}: {result.error}"
            for action, result in zip(actions, results, strict=True)
            if result.error is not None
        ]
        batch_exclusions = (
            prepared.excluded
            if prepared.plan.scope.selection is not HomeSelection.SPECIFIC
            else ()
        )
        if errors and len(errors) == len(results):
            status = HomeActionStatus.FAILED
        elif errors or batch_exclusions:
            status = HomeActionStatus.PARTIAL
        else:
            status = HomeActionStatus.SUCCEEDED
        result_payload: dict[str, object] = {
            "items": [
                {
                    "entityId": action.entity_id,
                    "operation": action.operation.value,
                    "accepted": result.accepted,
                    "confirmed": result.confirmed,
                    "state": result.state,
                    "response": result.response,
                    "error": result.error,
                }
                for action, result in zip(actions, results, strict=True)
            ],
            "excluded": _excluded_payload(prepared.excluded),
            "satisfied": _actions_payload(prepared.satisfied),
        }
        await self._store.update_run(
            run_id,
            status=status,
            action=action_payload,
            entity_id=actions[0].entity_id if len(actions) == 1 else None,
            accepted=all(result.accepted for result in results),
            confirmed=all(result.confirmed for result in results),
            result=result_payload,
            error="；".join(errors) if errors else None,
        )
        return HomeRunOutcome(
            intent=HomeIntentType.CONTROL,
            status=status,
            actions=actions,
            excluded=prepared.excluded,
            satisfied=prepared.satisfied,
            error="；".join(errors) if errors else None,
        )

    async def _finish_scoped_failure(
        self,
        run_id: UUID,
        intent: HomeIntentType,
        error: str,
        *,
        status: HomeActionStatus = HomeActionStatus.FAILED,
        excluded: tuple[ExcludedHomeTarget, ...] = (),
        satisfied: tuple[HomeAction, ...] = (),
    ) -> HomeRunOutcome:
        await self._store.update_run(
            run_id,
            status=status,
            action={},
            result={
                "excluded": _excluded_payload(excluded),
                "satisfied": _actions_payload(satisfied),
            },
            error=error,
        )
        return HomeRunOutcome(
            intent=intent,
            status=status,
            excluded=excluded,
            satisfied=satisfied,
            error=error,
        )

    async def _load_router_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_sequence: int,
    ) -> tuple[ChatMessage, ...]:
        messages = await self._history_store.load_history(
            fish_id,
            session_id,
            before_sequence=message_sequence,
            limit=_ROUTER_HISTORY_LIMIT,
        )
        return tuple(
            {
                "role": message.role.value,
                "content": visible,
            }
            for message in messages
            if (visible := visible_message_content(message))
        )

    async def _process(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        content: str,
        message_sequence: int,
        *,
        intent: HomeIntentType,
        origin_ns: int | None,
    ) -> None:
        tracker = MessagePerformanceTracker(origin_ns=origin_ns)
        del intent
        total_started = tracker.now_ns()
        run: Any = None
        try:
            if not looks_like_home_request(content):
                return
            if not self._client.catalog.snapshot.ready:
                run = await self._store.create_run(
                    fish_id=fish_id,
                    session_id=session_id,
                    source_message_id=message_id,
                    action={},
                )
                await self._store.update_run(
                    run.id,
                    status=HomeActionStatus.FAILED,
                    error="Home Assistant 未连接或设备目录尚未就绪",
                )
                return
            decision = self._parser.parse(content, self._client.catalog)
            if decision is None:
                # Complete commands keep the zero-I/O local fast path. History is
                # only needed when the Router must resolve a conversational follow-up.
                messages = await self._history_store.load_history(
                    fish_id,
                    session_id,
                    before_sequence=message_sequence,
                    limit=_ROUTER_HISTORY_LIMIT,
                )
                history: tuple[ChatMessage, ...] = tuple(
                    {
                        "role": message.role.value,
                        "content": visible,
                    }
                    for message in messages
                    if (visible := visible_message_content(message))
                )
                candidate_pool = self._client.catalog.search(
                    content,
                    limit=len(self._client.catalog.snapshot.entities),
                    available_only=True,
                )
                if not candidate_pool:
                    if has_explicit_set_request(content):
                        # Set commands are expanded from the complete catalog after
                        # routing. Do not let an unrelated history match narrow the
                        # Router's view before it decides the set filters.
                        candidate_pool = tuple(
                            HomeCandidate(entity=entity, score=0, matched_aliases=())
                            for entity in self._client.catalog.available_entities()
                        )
                    else:
                        # History resolves elliptical entity references only when
                        # the current message provides no candidate of its own.
                        search_text = "\n".join(
                            [*(message["content"] for message in history), content]
                        )
                        candidate_pool = self._client.catalog.search(
                            search_text,
                            limit=len(self._client.catalog.snapshot.entities),
                            available_only=True,
                        )
                action_hints = {
                    candidate.entity.entity_id: hint
                    for candidate in candidate_pool
                    if (hint := self._parser.infer_action(content, candidate.entity))
                    is not None
                }
                requested_domains = requested_home_domains(content)
                preferred_action_hints = {
                    entity_id: hint
                    for entity_id, hint in action_hints.items()
                    if (entity := self._client.catalog.get(entity_id)) is not None
                    and entity.domain in requested_domains
                    and self._executor.action_is_applicable(hint)
                }
                if preferred_action_hints:
                    preferred_device_ids = {
                        entity.device_id
                        for entity_id in preferred_action_hints
                        if (entity := self._client.catalog.get(entity_id)) is not None
                        and entity.device_id is not None
                    }
                    prefer_unidentified_standard_entity = any(
                        entity.device_id is None
                        for entity_id in preferred_action_hints
                        if (entity := self._client.catalog.get(entity_id)) is not None
                    )
                    # A standard entity may replace helpers for the same physical
                    # device, but another speaker must not hide a TV's power button.
                    candidate_pool = tuple(
                        candidate
                        for candidate in candidate_pool
                        if candidate.entity.domain in requested_domains
                        or (
                            candidate.entity.device_id not in preferred_device_ids
                            and not (
                                candidate.entity.device_id is None
                                and prefer_unidentified_standard_entity
                            )
                        )
                    )
                candidates = limit_candidates_by_device(
                    tuple(
                        candidate
                        for candidate in candidate_pool
                        if (
                            (hint := action_hints.get(candidate.entity.entity_id))
                            is None
                            or not self._executor.target_is_satisfied(hint)
                        )
                    ),
                    limit=self._router.device_limit,
                )
                if not candidates:
                    return
            run = await self._store.create_run(
                fish_id=fish_id,
                session_id=session_id,
                source_message_id=message_id,
                action={},
            )
            async with self._semaphore:
                if decision is None:
                    try:
                        decision = await asyncio.wait_for(
                            self._router.route(
                                content,
                                candidates,
                                history=history,
                                tracker=tracker,
                            ),
                            timeout=self._timeout,
                        )
                    except ValueError as error:
                        # Router validation rejects unsafe model output, such as
                        # expanding an ambiguous room-level request to every device.
                        logger.info(
                            "Rejected ambiguous Home Assistant route for "
                            "message %s: %s",
                            message_id,
                            error,
                        )
                        await self._store.update_run(
                            run.id,
                            status=HomeActionStatus.AMBIGUOUS,
                            error="无法唯一确定设备或操作",
                        )
                        return
                actions = _resolve_actions(
                    decision,
                    catalog=self._client.catalog,
                    executor=self._executor,
                )
                if not actions:
                    if decision.skipped_commands:
                        await self._store.update_run(
                            run.id,
                            status=HomeActionStatus.FAILED,
                            action={},
                            result={
                                "skipped": _skipped_actions_payload(
                                    decision.skipped_commands
                                )
                            },
                            error="所选设备均不支持该操作",
                        )
                        return
                    await self._store.update_run(
                        run.id,
                        status=HomeActionStatus.AMBIGUOUS,
                        error="无法唯一确定设备或操作",
                    )
                    return
                action_payload = _actions_payload(actions)
                selected_entity_id = actions[0].entity_id if len(actions) == 1 else None
                await self._store.update_run(
                    run.id,
                    status=HomeActionStatus.EXECUTING,
                    action=action_payload,
                    entity_id=selected_entity_id,
                )
                results = await _execute_actions(
                    actions,
                    executor=self._executor,
                    timeout_seconds=self._timeout,
                )
                result_payload: dict[str, object]
                if len(results) == 1:
                    result_payload = {
                        "accepted": results[0].accepted,
                        "confirmed": results[0].confirmed,
                        "state": results[0].state,
                        "response": results[0].response,
                        "error": results[0].error,
                    }
                else:
                    result_payload = {
                        "items": [
                            {
                                "entityId": action.entity_id,
                                "operation": action.operation.value,
                                "accepted": result.accepted,
                                "confirmed": result.confirmed,
                                "state": result.state,
                                "response": result.response,
                                "error": result.error,
                            }
                            for action, result in zip(actions, results, strict=True)
                        ]
                    }
                if decision.skipped_commands:
                    result_payload["skipped"] = _skipped_actions_payload(
                        decision.skipped_commands
                    )
                errors = [
                    f"{action.entity_id}: {result.error}"
                    for action, result in zip(actions, results, strict=True)
                    if result.error is not None
                ]
                if errors and len(errors) == len(results):
                    final_status = HomeActionStatus.FAILED
                elif errors or decision.skipped_commands:
                    final_status = HomeActionStatus.PARTIAL
                else:
                    final_status = HomeActionStatus.SUCCEEDED
                await self._store.update_run(
                    run.id,
                    status=final_status,
                    action=action_payload,
                    entity_id=selected_entity_id,
                    accepted=all(result.accepted for result in results),
                    confirmed=all(result.confirmed for result in results),
                    service_domain=(
                        results[0].service_domain if len(results) == 1 else None
                    ),
                    service_name=(
                        results[0].service_name if len(results) == 1 else None
                    ),
                    result=result_payload,
                    error="；".join(errors) if errors else None,
                )
        except Exception as error:
            logger.exception("Home Assistant routing failed for message %s", message_id)
            if run is not None:
                await self._store.update_run(
                    run.id,
                    status=HomeActionStatus.FAILED,
                    error=f"{type(error).__name__}: {error}",
                )
        finally:
            if self._performance_store is not None:
                tracker.record(
                    "home.total",
                    started_ns=total_started,
                    finished_ns=time.perf_counter_ns(),
                )
                await self._performance_store.insert_steps(
                    message_id, tracker.snapshot()
                )


def _prepared_failure(
    intent: HomeIntentType,
    error: str,
    *,
    plan: HomeControlPlan | HomeQueryPlan | None = None,
    status: HomeActionStatus = HomeActionStatus.FAILED,
    excluded: tuple[ExcludedHomeTarget, ...] = (),
    satisfied: tuple[HomeAction, ...] = (),
) -> PreparedHomeRequest:
    return PreparedHomeRequest(
        intent=intent,
        plan=plan,
        excluded=excluded,
        satisfied=satisfied,
        outcome=HomeRunOutcome(
            intent=intent,
            status=status,
            excluded=excluded,
            satisfied=satisfied,
            error=error,
        ),
    )


def _action_payload(action: HomeAction) -> Mapping[str, object]:
    return {
        "entityId": action.entity_id,
        "operation": action.operation.value,
        "value": action.value,
    }


def _planned_action(
    candidate: HomeCandidate,
    plan: HomeControlPlan,
) -> HomeAction:
    value = plan.value
    if plan.operation is HomeOperation.SET_BRIGHTNESS:
        value = cast(
            str | int | float | bool | None,
            resolve_brightness_value(value, candidate.entity),
        )
    return HomeAction(candidate.entity.entity_id, plan.operation, value)


def _scope_payload(scope: HomeScopePlan) -> Mapping[str, object]:
    return {
        "kind": scope.kind.value,
        "areaIndexes": list(scope.area_indexes),
        "domains": list(scope.domains),
        "selection": scope.selection.value,
        "count": scope.count,
    }


def _query_result_payload(result: ResolvedQueryScope) -> Mapping[str, object]:
    return {
        "items": [
            {
                "entityId": item.entity.entity_id,
                "name": item.entity.aliases[0],
                "area": item.entity.area_name,
                "domain": item.entity.domain,
                "state": item.entity.state,
                "category": item.category.value,
                "attributes": thaw_home_value(item.entity.attributes),
            }
            for item in result.entities
        ]
    }


def _excluded_payload(
    excluded: tuple[ExcludedHomeTarget, ...],
) -> list[Mapping[str, object]]:
    return [
        {
            "entityId": item.entity.entity_id,
            "reason": item.reason,
        }
        for item in excluded
    ]


def _actions_payload(actions: tuple[HomeAction, ...]) -> Mapping[str, object]:
    if len(actions) == 1:
        return _action_payload(actions[0])
    return {"actions": [_action_payload(action) for action in actions]}


def _skipped_actions_payload(
    actions: tuple[HomeAction, ...],
) -> list[Mapping[str, object]]:
    return [
        {**_action_payload(action), "reason": "operation_not_supported"}
        for action in actions
    ]


def _resolve_actions(
    decision: HomeRouteDecision,
    *,
    catalog: HomeEntityCatalog,
    executor: HomeActionExecutor,
) -> tuple[HomeAction, ...]:
    actions: list[HomeAction] = []
    for command in decision.commands:
        if isinstance(command, HomeAction):
            if executor.action_is_applicable(command):
                actions.append(command)
        else:
            actions.extend(
                _expand_set_command(command, catalog=catalog, executor=executor)
            )
    unique: list[HomeAction] = []
    seen: set[HomeAction] = set()
    for action in actions:
        if action not in seen:
            seen.add(action)
            unique.append(action)
    return tuple(unique)


def _expand_set_command(
    command: HomeSetCommand,
    *,
    catalog: HomeEntityCatalog,
    executor: HomeActionExecutor,
) -> tuple[HomeAction, ...]:
    target = command.target
    normalized_area = normalize_home_alias(target.area or "")
    actions: list[HomeAction] = []
    for entity in sorted(catalog.available_entities(), key=lambda item: item.entity_id):
        if target.domain is not None and entity.domain != target.domain:
            continue
        if normalized_area and (
            normalize_home_alias(entity.area_name or "") != normalized_area
        ):
            continue
        if command.operation not in entity.operations:
            continue
        unresolved_action = HomeAction(
            entity.entity_id,
            command.operation,
            command.value,
        )
        if not executor.action_is_applicable(unresolved_action):
            continue
        value = command.value
        if command.operation is HomeOperation.SET_BRIGHTNESS:
            value = cast(
                str | int | float | bool | None,
                resolve_brightness_value(value, entity),
            )
        action = HomeAction(entity.entity_id, command.operation, value)
        if not executor.target_is_satisfied(action):
            actions.append(action)
    if target.count is None:
        return tuple(actions)
    if len(actions) < target.count:
        raise ValueError("not enough actionable entities for requested set count")
    return tuple(actions[: target.count])


async def _execute_actions(
    actions: tuple[HomeAction, ...],
    *,
    executor: HomeActionExecutor,
    timeout_seconds: float,
) -> tuple[HomeExecutionResult, ...]:
    indexes_by_entity: dict[str, list[int]] = {}
    for index, action in enumerate(actions):
        indexes_by_entity.setdefault(action.entity_id, []).append(index)

    async def execute_entity(indexes: list[int]) -> dict[int, HomeExecutionResult]:
        results: dict[int, HomeExecutionResult] = {}
        for index in indexes:
            try:
                results[index] = await asyncio.wait_for(
                    executor.execute(actions[index]),
                    timeout=timeout_seconds,
                )
            except TimeoutError as error:
                # The write may already have reached HA. Do not retry a timed-out
                # control because that could duplicate it.
                raise TimeoutError("Home Assistant 操作超时，执行结果未知") from error
        return results

    grouped_results = await asyncio.gather(
        *(execute_entity(indexes) for indexes in indexes_by_entity.values())
    )
    results_by_index = {
        index: result for group in grouped_results for index, result in group.items()
    }
    return tuple(results_by_index[index] for index in range(len(actions)))
