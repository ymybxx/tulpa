"""Durable dispatch and reconciliation for external Agent runs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from fish_memory.agent.autonomous import AutonomousRunPolicy, autonomous_run_input
from fish_memory.agent.connector import (
    AgentConnector,
    AgentConnectorError,
    AgentHistoryMessage,
    AgentProviderEvent,
    AgentProviderEventKind,
    AgentRunRequest,
    AgentRunSnapshot,
    ConnectorRunState,
)
from fish_memory.agent.models import (
    AgentHistorySource,
    AgentInteraction,
    AgentInteractionWorkItem,
    AgentRun,
    AgentRunStatus,
    AgentRunWorkItem,
)
from fish_memory.agent.outcome import (
    AGENT_RUN_OUTCOME_INSTRUCTIONS,
    AgentOutcomeDisposition,
    normalize_agent_run_outcome,
)
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.agent.session_context import AgentSessionContextLoader
from fish_memory.answer.history_tokens import estimate_history_message_tokens

logger = logging.getLogger(__name__)

_MAX_HISTORY_MESSAGES = 64
_MAX_HISTORY_MESSAGE_CHARS = 8_000
_MAX_HISTORY_BYTES = 256_000
_MAX_CONTINUATION_HISTORY_MESSAGES = 16
_MAX_CONTINUATION_HISTORY_MESSAGE_CHARS = 4_000
_MAX_CONTINUATION_HISTORY_BYTES = 64_000
_MAX_CONTINUATION_HISTORY_TOKENS = 8_000
_MAX_RECONCILABLE = 1_000
_STALE_RUN_ERROR_CODE = "timeout"
_STALE_RUN_ERROR_MESSAGE = (
    "Agent run status remained unavailable beyond the configured timeout"
)


class AgentRunStore(Protocol):
    async def mark_unidentified_dispatches_unknown(self) -> int: ...

    async def mark_unidentified_approval_dispatches_unknown(self) -> int: ...

    async def mark_unidentified_steer_dispatches_unknown(self) -> int: ...

    async def list_reconcilable_run_work(
        self,
        limit: int,
        after: tuple[datetime, UUID] | None = None,
    ) -> Sequence[AgentRunWorkItem]: ...

    async def list_stale_run_work(
        self,
        cutoff: datetime,
        limit: int = _MAX_RECONCILABLE,
    ) -> Sequence[AgentRunWorkItem]: ...

    async def fail_stale_run(
        self, run_id: UUID, error_code: str, error_message: str
    ) -> AgentRun | None: ...

    async def list_reconcilable_approval_responses(
        self,
        limit: int = 1_000,
        after: tuple[datetime, UUID] | None = None,
    ) -> Sequence[AgentInteractionWorkItem]: ...

    async def claim_pending_approval_response(
        self,
    ) -> AgentInteractionWorkItem | None: ...

    async def claim_pending_steer(self) -> AgentRunWorkItem | None: ...

    async def peek_queued_run(self) -> AgentRunWorkItem | None: ...

    async def load_local_projection(
        self, run_id: UUID, limit: int = 64
    ) -> Sequence[tuple[str, str]]: ...

    async def load_user_message_content(self, message_id: UUID) -> str | None: ...

    async def fail_queued_run(
        self, run_id: UUID, error_code: str, error_message: str
    ) -> AgentRun | None: ...

    async def claim_prepared_run(
        self,
        run_id: UUID,
        history_source: AgentHistorySource | str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRunWorkItem | None: ...

    async def record_dispatch_accepted(
        self,
        run_id: UUID,
        external_run_id: str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun: ...

    async def update_active_run(
        self,
        run_id: UUID,
        status: AgentRunStatus | str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun | None: ...

    async def mark_run_unknown(
        self,
        run_id: UUID,
        error_code: str,
        error_message: str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun | None: ...

    async def finish_run(
        self,
        run_id: UUID,
        status: AgentRunStatus | str,
        output: str | None,
        error_code: str | None,
        error_message: str | None,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun: ...

    async def complete_run_requiring_input(
        self,
        run_id: UUID,
        output: str,
        provider_data: Mapping[str, object] | None,
    ) -> tuple[AgentRun, AgentInteraction]: ...

    async def open_approval_interaction(
        self, run_id: UUID, provider_data: Mapping[str, object] | None
    ) -> AgentInteraction: ...

    async def mark_approval_dispatch_unknown(
        self, interaction_id: UUID, error_code: str, error_message: str
    ) -> AgentInteraction | None: ...

    async def resolve_approval(
        self, interaction_id: UUID
    ) -> AgentInteraction | None: ...

    async def record_steer_accepted(self, run_id: UUID) -> AgentRun | None: ...

    async def mark_steer_dispatch_unknown(
        self, run_id: UUID, error_code: str, error_message: str
    ) -> AgentRun | None: ...


class AgentRunProcessor:
    def __init__(
        self,
        store: AgentRunStore,
        connectors: Mapping[str, AgentConnector],
        *,
        concurrency: int = 4,
        poll_interval_seconds: float = 2.0,
        reconcile_limit: int = _MAX_RECONCILABLE,
        stale_timeout_seconds: float = 300.0,
        session_context_loader: AgentSessionContextLoader | None = None,
        autonomous_policy: AutonomousRunPolicy | None = None,
        plugin_policy: AutonomousRunPolicy | None = None,
        runtime_state: AgentRuntimeState | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if reconcile_limit < 1:
            raise ValueError("reconcile_limit must be positive")
        if stale_timeout_seconds <= 0:
            raise ValueError("stale_timeout_seconds must be positive")
        self._store = store
        self._connectors = dict(connectors)
        self._concurrency = concurrency
        self._poll_interval_seconds = poll_interval_seconds
        self._reconcile_limit = reconcile_limit
        self._stale_timeout_seconds = stale_timeout_seconds
        self._session_context_loader = session_context_loader
        self._run_policies = {"proactive": autonomous_policy, "plugin": plugin_policy}
        self._runtime_state = runtime_state
        self._watchers: dict[UUID, asyncio.Task[None]] = {}
        self._recovered = False
        self._closed = False
        self._approval_reconcile_not_before: dict[UUID, float] = {}
        self._reconciled_unknown_ids: set[UUID] = set()

    def configure_scheduling(
        self, *, concurrency: int, poll_interval_seconds: float
    ) -> None:
        if concurrency < 1 or poll_interval_seconds <= 0:
            raise ValueError("Agent scheduling values must be positive")
        # Existing watchers keep running; a lower limit only delays new work.
        self._concurrency = concurrency
        self._poll_interval_seconds = poll_interval_seconds

    async def process_once(self) -> bool:
        if self._closed:
            return False
        if self._runtime_state is not None:
            await self._runtime_state.wait_enabled()
        worked = await self._recover_uncertain_local_dispatches()
        worked = await self._reconcile_stale_runs() or worked
        worked = await self._start_reconcilable_watchers() or worked
        worked = await self._reconcile_approval_responses() or worked

        approval = await self._store.claim_pending_approval_response()
        if approval is not None:
            await self._dispatch_approval_response(approval)
            worked = True

        steer = await self._store.claim_pending_steer()
        if steer is not None:
            await self._dispatch_steer(steer)
            worked = True

        if len(self._watchers) >= self._concurrency:
            return worked

        queued = await self._store.peek_queued_run()
        if queued is None or (
            self._runtime_state is not None and not self._runtime_state.enabled
        ):
            return worked
        await self._prepare_and_dispatch(queued)
        return True

    async def close(self) -> None:
        self._closed = True
        tasks = tuple(self._watchers.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._watchers.clear()

    async def _recover_uncertain_local_dispatches(self) -> bool:
        if self._recovered:
            return False
        self._recovered = True
        recovered = await self._store.mark_unidentified_dispatches_unknown()
        recovered += await self._store.mark_unidentified_approval_dispatches_unknown()
        recovered += await self._store.mark_unidentified_steer_dispatches_unknown()
        return recovered > 0

    async def _reconcile_stale_runs(self) -> bool:
        cutoff = datetime.now(UTC) - timedelta(seconds=self._stale_timeout_seconds)
        work_items = await self._store.list_stale_run_work(
            cutoff,
            limit=self._reconcile_limit,
        )
        for work in work_items:
            policy = self._policy(work)
            if work.run.origin in {"proactive", "plugin"}:
                if policy is not None:
                    await policy.stop(work)
                # An unconfirmed stop must retain the autonomous slot.
                continue
            await self._cancel_watcher(work.run.id)
            connector = self._connector(work)
            external_run_id = work.run.external_run_id
            if connector is None or external_run_id is None:
                await self._fail_stale_run(work.run.id)
                continue
            try:
                snapshot = await connector.get_run(work, external_run_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._fail_stale_run(work.run.id)
                continue
            if snapshot.state is ConnectorRunState.UNKNOWN:
                await self._fail_stale_run(work.run.id)
                continue
            terminal = await self._apply_snapshot(work, snapshot)
            if terminal:
                continue
            refreshed = replace(
                work,
                run=replace(
                    work.run,
                    status=(
                        AgentRunStatus.WAITING_USER
                        if snapshot.state is ConnectorRunState.WAITING_USER
                        else AgentRunStatus.RUNNING
                    ),
                ),
            )
            self._start_watcher(refreshed, connector)
        return bool(work_items)

    async def _fail_stale_run(self, run_id: UUID) -> None:
        await self._store.fail_stale_run(
            run_id,
            _STALE_RUN_ERROR_CODE,
            _STALE_RUN_ERROR_MESSAGE,
        )

    async def _cancel_watcher(self, run_id: UUID) -> None:
        task = self._watchers.pop(run_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _start_reconcilable_watchers(self) -> bool:
        started = False
        cursor: tuple[datetime, UUID] | None = None
        active_unknown_ids: set[UUID] = set()
        while True:
            work_items = await self._store.list_reconcilable_run_work(
                self._reconcile_limit,
                cursor,
            )
            for work in work_items:
                if work.run.status is AgentRunStatus.UNKNOWN:
                    active_unknown_ids.add(work.run.id)
                if work.run.id in self._watchers:
                    continue
                if (
                    work.run.status is AgentRunStatus.UNKNOWN
                    and work.run.id in self._reconciled_unknown_ids
                ):
                    continue
                connector = self._connector(work)
                if connector is None:
                    await self._store.mark_run_unknown(
                        work.run.id,
                        "unsupported_capability",
                        "connector type "
                        f"{work.endpoint.connector_type!s} is unavailable",
                        None,
                    )
                    self._reconciled_unknown_ids.add(work.run.id)
                    started = True
                    continue
                if work.run.external_run_id is None:
                    continue
                self._start_watcher(work, connector)
                started = True
            if len(work_items) < self._reconcile_limit:
                break
            last = work_items[-1].run
            cursor = (last.created_at, last.id)
        self._reconciled_unknown_ids.intersection_update(active_unknown_ids)
        return started

    async def _reconcile_approval_responses(self) -> bool:
        worked = False
        cursor: tuple[datetime, UUID] | None = None
        active_ids: set[UUID] = set()
        loop = asyncio.get_running_loop()
        while True:
            items = await self._store.list_reconcilable_approval_responses(
                self._reconcile_limit,
                cursor,
            )
            for item in items:
                interaction_id = item.interaction.id
                active_ids.add(interaction_id)
                if self._approval_reconcile_not_before.get(interaction_id, 0) > (
                    loop.time()
                ):
                    continue
                progressed = await self._reconcile_approval_response(item)
                if progressed:
                    self._approval_reconcile_not_before.pop(interaction_id, None)
                    worked = True
                else:
                    self._approval_reconcile_not_before[interaction_id] = (
                        loop.time() + self._poll_interval_seconds
                    )
            if len(items) < self._reconcile_limit:
                break
            last = items[-1].interaction
            cursor = (last.created_at, last.id)
        self._approval_reconcile_not_before = {
            interaction_id: not_before
            for interaction_id, not_before in (
                self._approval_reconcile_not_before.items()
            )
            if interaction_id in active_ids
        }
        return worked

    async def _reconcile_approval_response(
        self, item: AgentInteractionWorkItem
    ) -> bool:
        if item.run.status in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }:
            await self._store.resolve_approval(item.interaction.id)
            return True
        connector = self._connector(item)
        external_run_id = item.run.external_run_id
        if connector is None or external_run_id is None:
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id,
                "unsupported_capability",
                "Approval response cannot be reconciled",
            )
            return False
        try:
            snapshot = await connector.get_run(_run_work(item), external_run_id)
        except AgentConnectorError:
            return False
        if snapshot.state is ConnectorRunState.WAITING_USER:
            return False
        if snapshot.state is ConnectorRunState.UNKNOWN:
            if item.run.status is not AgentRunStatus.UNKNOWN:
                await self._apply_snapshot(item, snapshot, recover_approval=False)
            return False
        await self._store.resolve_approval(item.interaction.id)
        await self._apply_snapshot(item, snapshot, recover_approval=False)
        return True

    async def _dispatch_approval_response(self, item: AgentInteractionWorkItem) -> None:
        connector = self._connector(item)
        external_run_id = item.run.external_run_id
        if connector is None or external_run_id is None:
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id,
                "unsupported_capability",
                "Approval response cannot be dispatched",
            )
            return
        try:
            run_work = _run_work(item)
            before = await connector.get_run(run_work, external_run_id)
        except AgentConnectorError as error:
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id, error.code, str(error)
            )
            return
        if before.state is ConnectorRunState.UNKNOWN:
            await self._apply_snapshot(item, before, recover_approval=False)
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id,
                "dispatch_unknown",
                "Approval response acceptance is unknown",
            )
            self._defer_approval_reconciliation(item.interaction.id)
            return
        if before.state is not ConnectorRunState.WAITING_USER:
            await self._store.resolve_approval(item.interaction.id)
            await self._apply_snapshot(item, before, recover_approval=False)
            return
        response = _approval_response(item.interaction)
        if response is None:
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id,
                "invalid_request",
                "Approval response is malformed",
            )
            return
        choice, resolve_all = response
        try:
            await connector.respond_approval(
                run_work, external_run_id, choice, resolve_all
            )
            after = await connector.get_run(run_work, external_run_id)
        except AgentConnectorError as error:
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id, error.code, str(error)
            )
            return
        if after.state is ConnectorRunState.WAITING_USER:
            await self._apply_snapshot(item, after, recover_approval=False)
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id,
                "dispatch_unknown",
                "Approval response acceptance is not yet visible",
            )
            self._defer_approval_reconciliation(item.interaction.id)
            return
        if after.state is ConnectorRunState.UNKNOWN:
            await self._apply_snapshot(item, after, recover_approval=False)
            await self._store.mark_approval_dispatch_unknown(
                item.interaction.id,
                "dispatch_unknown",
                "Approval response acceptance is unknown",
            )
            self._defer_approval_reconciliation(item.interaction.id)
            return
        await self._store.resolve_approval(item.interaction.id)
        await self._apply_snapshot(item, after, recover_approval=False)

    def _defer_approval_reconciliation(self, interaction_id: UUID) -> None:
        self._approval_reconcile_not_before[interaction_id] = (
            asyncio.get_running_loop().time() + self._poll_interval_seconds
        )

    async def _dispatch_steer(self, work: AgentRunWorkItem) -> None:
        provider = work.run.provider_data or {}
        steer = provider.get("steer")
        source_value = (
            steer.get("source_message_id") if isinstance(steer, Mapping) else None
        )
        try:
            source_id = UUID(source_value) if isinstance(source_value, str) else None
        except ValueError:
            source_id = None
        connector = self._connector(work)
        external_run_id = work.run.external_run_id
        content = (
            await self._store.load_user_message_content(source_id)
            if source_id is not None
            else None
        )
        if connector is None or external_run_id is None or not content:
            await self._store.mark_steer_dispatch_unknown(
                work.run.id,
                "invalid_request",
                "Steer dispatch cannot be reconstructed",
            )
            return
        try:
            await connector.steer_run(work, external_run_id, content)
        except AgentConnectorError as error:
            await self._store.mark_steer_dispatch_unknown(
                work.run.id, error.code, str(error)
            )
            return
        await self._store.record_steer_accepted(work.run.id)

    async def _prepare_and_dispatch(self, work: AgentRunWorkItem) -> None:
        policy = self._policy(work)
        if work.run.origin in {"proactive", "plugin"} and (
            policy is None or policy.expired(work)
        ):
            if policy is not None:
                await policy.stop(work)
            else:
                await self._store.fail_queued_run(
                    work.run.id,
                    "autonomous_disabled",
                    "Autonomous Run policy is unavailable",
                )
            return
        connector = self._connector(work)
        if connector is None:
            await self._store.fail_queued_run(
                work.run.id,
                "unsupported_capability",
                f"connector type {work.endpoint.connector_type!s} is unavailable",
            )
            return
        if not work.target.enabled:
            await self._store.fail_queued_run(
                work.run.id, "invalid_request", "Agent target is disabled"
            )
            return
        try:
            work = connector.preflight_run(work)
        except AgentConnectorError as error:
            await self._store.fail_queued_run(work.run.id, error.code, str(error))
            return
        try:
            prepared = await self._prepare_history(work, connector)
        except AgentConnectorError as error:
            await self._store.fail_queued_run(work.run.id, error.code, str(error))
            return
        if prepared is None:
            await self._store.fail_queued_run(
                work.run.id,
                "protocol_error",
                "Continuation history is unavailable",
            )
            return
        history_source, history = prepared
        if self._runtime_state is not None and not self._runtime_state.enabled:
            return
        claimed = await self._store.claim_prepared_run(
            work.run.id, history_source, work.run.provider_data
        )
        if claimed is None:
            return
        claimed = replace(
            claimed,
            target=work.target,
            endpoint=work.endpoint,
        )
        request = AgentRunRequest(
            input=autonomous_run_input(work),
            instructions=AGENT_RUN_OUTCOME_INSTRUCTIONS
            + (
                "\n这是宠物自主探索。仅使用已获授权的能力；需要授权、登录或用户输入时停止并返回阻塞原因。"
                if work.run.origin in {"proactive", "plugin"}
                else ""
            ),
            conversation_history=history,
        )
        try:
            accepted = await connector.start_run(claimed, request)
        except AgentConnectorError as error:
            if error.dispatch_uncertain:
                await self._store.mark_run_unknown(
                    work.run.id,
                    error.code,
                    str(error),
                    work.run.provider_data,
                )
            else:
                await self._store.finish_run(
                    work.run.id,
                    AgentRunStatus.FAILED,
                    None,
                    error.code,
                    str(error),
                    work.run.provider_data,
                )
            return
        except Exception:
            await self._store.mark_run_unknown(
                work.run.id,
                "dispatch_unknown",
                "Agent dispatch ended without a normalized response",
                work.run.provider_data,
            )
            return
        stored = await self._store.record_dispatch_accepted(
            work.run.id, accepted.external_run_id, work.run.provider_data
        )
        watched = replace(
            claimed,
            run=replace(
                stored,
                external_run_id=accepted.external_run_id,
            ),
        )
        if watched.run.origin in {"proactive", "plugin"} and policy is not None:
            if (
                policy.expired(watched)
                or accepted.state == ConnectorRunState.WAITING_USER
            ):
                await policy.stop(watched)
                return
        if accepted.state is ConnectorRunState.UNKNOWN:
            await self._store.mark_run_unknown(
                work.run.id,
                "dispatch_unknown",
                "Provider accepted the run with an unknown state",
                work.run.provider_data,
            )
            watched = replace(
                watched, run=replace(watched.run, status=AgentRunStatus.UNKNOWN)
            )
        elif accepted.state is ConnectorRunState.WAITING_USER:
            await self._store.update_active_run(
                work.run.id, AgentRunStatus.WAITING_USER, work.run.provider_data
            )
            watched = replace(
                watched, run=replace(watched.run, status=AgentRunStatus.WAITING_USER)
            )
        self._start_watcher(watched, connector)

    async def _prepare_history(
        self, work: AgentRunWorkItem, connector: AgentConnector
    ) -> tuple[AgentHistorySource, tuple[AgentHistoryMessage, ...]] | None:
        source_history: tuple[AgentHistoryMessage, ...] = ()
        if self._session_context_loader is not None:
            loaded = await self._session_context_loader.load_for_run(work)
            if loaded is None:
                return None
            source_history = loaded
        if work.run.previous_run_id is None:
            return (
                AgentHistorySource.FISH_SESSION
                if source_history
                else AgentHistorySource.NONE,
                source_history,
            )
        continuation = await self._prepare_continuation_history(work, connector)
        if continuation is None:
            return None
        continuation_source, continuation_history = continuation
        if not source_history:
            return continuation_source, continuation_history
        combined_source = (
            AgentHistorySource.FISH_SESSION_HERMES_SESSION
            if continuation_source is AgentHistorySource.HERMES_SESSION
            else AgentHistorySource.FISH_SESSION_LOCAL_PROJECTION
        )
        return (
            combined_source,
            (*source_history, *continuation_history),
        )

    async def _prepare_continuation_history(
        self, work: AgentRunWorkItem, connector: AgentConnector
    ) -> tuple[AgentHistorySource, tuple[AgentHistoryMessage, ...]] | None:
        connector_history: Sequence[AgentHistoryMessage] = ()
        try:
            connector_history = await connector.get_session_history(work)
        except AgentConnectorError as error:
            if not (
                error.retryable
                or error.code
                in {
                    "endpoint_unavailable",
                    "timeout",
                    "protocol_error",
                    "invalid_request",
                }
            ):
                raise
        bounded = _bound_continuation_history(connector_history)
        if bounded:
            return AgentHistorySource.HERMES_SESSION, bounded
        local = await self._store.load_local_projection(
            work.run.id, _MAX_HISTORY_MESSAGES
        )
        projected: list[AgentHistoryMessage] = []
        for role, content in local:
            if role not in {"user", "assistant"} or not isinstance(content, str):
                return None
            projected.append(
                AgentHistoryMessage(
                    role=role,
                    content=content,
                )
            )
        bounded = _bound_continuation_history(projected)
        if not bounded:
            return None
        return AgentHistorySource.LOCAL_PROJECTION, bounded

    def _start_watcher(self, work: AgentRunWorkItem, connector: AgentConnector) -> None:
        if self._closed or work.run.id in self._watchers:
            return
        external_run_id = work.run.external_run_id
        if external_run_id is None:
            return
        task = asyncio.create_task(
            self._watch_run(work, connector, external_run_id),
            name=f"fish-memory-agent-run-{work.run.id}",
        )
        self._watchers[work.run.id] = task

        def consume(done: asyncio.Task[None]) -> None:
            if self._watchers.get(work.run.id) is done:
                self._watchers.pop(work.run.id, None)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.error("Agent run watcher failed with %s", type(error).__name__)

        task.add_done_callback(consume)

    async def _watch_run(
        self,
        work: AgentRunWorkItem,
        connector: AgentConnector,
        external_run_id: str,
    ) -> None:
        if work.run.status is AgentRunStatus.UNKNOWN:
            try:
                snapshot = await connector.get_run(work, external_run_id)
            except asyncio.CancelledError:
                raise
            except AgentConnectorError:
                self._reconciled_unknown_ids.add(work.run.id)
                return
            await self._apply_snapshot(work, snapshot)
            self._reconciled_unknown_ids.add(work.run.id)
            return
        try:
            async for event in connector.watch_run(work, external_run_id):
                terminal = await self._apply_event(work, event)
                if terminal:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        while True:
            try:
                snapshot = await connector.get_run(work, external_run_id)
            except asyncio.CancelledError:
                raise
            except AgentConnectorError as error:
                if not error.retryable:
                    await self._store.mark_run_unknown(
                        work.run.id, error.code, str(error), None
                    )
                    self._reconciled_unknown_ids.add(work.run.id)
                    return
                await asyncio.sleep(self._poll_interval_seconds)
                continue
            except Exception:
                await asyncio.sleep(self._poll_interval_seconds)
                continue
            terminal = await self._apply_snapshot(work, snapshot)
            if terminal or snapshot.state is ConnectorRunState.UNKNOWN:
                if snapshot.state is ConnectorRunState.UNKNOWN:
                    self._reconciled_unknown_ids.add(work.run.id)
                return
            await asyncio.sleep(self._poll_interval_seconds)

    async def _apply_event(
        self, work: AgentRunWorkItem, event: AgentProviderEvent
    ) -> bool:
        if event.kind is AgentProviderEventKind.APPROVAL:
            if work.run.origin in {"proactive", "plugin"}:
                await self._stop_autonomous(work)
                return True
            provider = dict(event.provider_data or {})
            provider["allowed_decisions"] = list(event.approval_choices)
            await self._store.open_approval_interaction(work.run.id, provider)
            return False
        if event.kind is AgentProviderEventKind.STATE and event.state.is_terminal:
            return False
        snapshot = AgentRunSnapshot(
            state=event.state,
            output=event.output,
            error_message=event.error_message,
            provider_data=event.provider_data,
        )
        return await self._apply_snapshot(
            work,
            snapshot,
            recover_approval=False,
        )

    async def _apply_snapshot(
        self,
        work: AgentRunWorkItem | AgentInteractionWorkItem,
        snapshot: AgentRunSnapshot,
        *,
        recover_approval: bool = True,
    ) -> bool:
        run = work.run
        policy = self._policy(work)
        if (
            run.origin in {"proactive", "plugin"}
            and not snapshot.state.is_terminal
            and (
                snapshot.state == ConnectorRunState.WAITING_USER
                or policy is None
                or policy.expired(work)
            )
        ):
            await self._stop_autonomous(work)
            return True
        if snapshot.state is ConnectorRunState.RUNNING:
            await self._store.update_active_run(
                run.id, AgentRunStatus.RUNNING, snapshot.provider_data
            )
            return False
        if snapshot.state is ConnectorRunState.WAITING_USER:
            if recover_approval:
                await self._store.open_approval_interaction(
                    run.id,
                    {
                        "allowed_decisions": ["deny"],
                        "recoveredWithoutDetails": True,
                    },
                )
            else:
                await self._store.update_active_run(
                    run.id, AgentRunStatus.WAITING_USER, snapshot.provider_data
                )
            return False
        if snapshot.state is ConnectorRunState.UNKNOWN:
            await self._store.mark_run_unknown(
                run.id,
                "provider_error",
                snapshot.error_message or "Provider run state is unknown",
                snapshot.provider_data,
            )
            return False
        if snapshot.state is ConnectorRunState.COMPLETED:
            outcome = normalize_agent_run_outcome(snapshot.output)
            if outcome.disposition is AgentOutcomeDisposition.INPUT_REQUIRED:
                if run.origin in {"proactive", "plugin"}:
                    await self._store.finish_run(
                        run.id,
                        AgentRunStatus.CANCELLED,
                        outcome.message,
                        "autonomous_blocked",
                        "Input required",
                        snapshot.provider_data,
                    )
                else:
                    await self._store.complete_run_requiring_input(
                        run.id, outcome.message, snapshot.provider_data
                    )
            else:
                await self._store.finish_run(
                    run.id,
                    AgentRunStatus.COMPLETED,
                    outcome.message,
                    None,
                    None,
                    snapshot.provider_data,
                )
            return True
        status = (
            AgentRunStatus.FAILED
            if snapshot.state is ConnectorRunState.FAILED
            else AgentRunStatus.CANCELLED
        )
        output = (
            normalize_agent_run_outcome(snapshot.output).message
            if snapshot.output is not None
            else None
        )
        await self._store.finish_run(
            run.id,
            status,
            output,
            "provider_error" if status is AgentRunStatus.FAILED else "cancelled",
            snapshot.error_message,
            snapshot.provider_data,
        )
        return True

    async def _stop_autonomous(
        self, work: AgentRunWorkItem | AgentInteractionWorkItem
    ) -> None:
        policy = self._policy(work)
        if policy is not None:
            await policy.stop(
                AgentRunWorkItem(
                    run=work.run,
                    task=work.task,
                    target=work.target,
                    endpoint=work.endpoint,
                )
            )
        else:
            await self._store.mark_run_unknown(
                work.run.id,
                "autonomous_disabled",
                "Autonomous policy unavailable; no approval response sent",
                None,
            )

    def _policy(
        self, work: AgentRunWorkItem | AgentInteractionWorkItem
    ) -> AutonomousRunPolicy | None:
        return self._run_policies.get(work.run.origin)

    def _connector(
        self, work: AgentRunWorkItem | AgentInteractionWorkItem
    ) -> AgentConnector | None:
        return self._connectors.get(str(work.endpoint.connector_type))


def _approval_response(interaction: AgentInteraction) -> tuple[str, bool] | None:
    provider = interaction.provider_data
    if not isinstance(provider, Mapping):
        return None
    raw = provider.get("approval_response")
    if not isinstance(raw, Mapping):
        return None
    choice = raw.get("choice")
    resolve_all = raw.get("resolve_all")
    if not isinstance(choice, str) or not isinstance(resolve_all, bool):
        return None
    return choice, resolve_all


def _run_work(item: AgentInteractionWorkItem) -> AgentRunWorkItem:
    return AgentRunWorkItem(
        run=item.run,
        task=item.task,
        target=item.target,
        endpoint=item.endpoint,
    )


def _bound_history(
    messages: Sequence[AgentHistoryMessage],
) -> tuple[AgentHistoryMessage, ...]:
    bounded: list[AgentHistoryMessage] = []
    remaining_bytes = _MAX_HISTORY_BYTES
    for message in reversed(messages[-_MAX_HISTORY_MESSAGES:]):
        if message.role not in {"user", "assistant"} or not isinstance(
            message.content, str
        ):
            continue
        content = message.content[:_MAX_HISTORY_MESSAGE_CHARS]
        if not content:
            continue
        content = _truncate_utf8(content, remaining_bytes)
        if not content:
            break
        bounded.append(AgentHistoryMessage(role=message.role, content=content))
        remaining_bytes -= len(content.encode("utf-8"))
        if remaining_bytes <= 0:
            break
    bounded.reverse()
    return tuple(bounded)


def _bound_continuation_history(
    messages: Sequence[AgentHistoryMessage],
) -> tuple[AgentHistoryMessage, ...]:
    bounded: list[AgentHistoryMessage] = []
    remaining_bytes = _MAX_CONTINUATION_HISTORY_BYTES
    remaining_tokens = _MAX_CONTINUATION_HISTORY_TOKENS
    for message in reversed(messages[-_MAX_CONTINUATION_HISTORY_MESSAGES:]):
        if message.role not in {"user", "assistant"} or not isinstance(
            message.content, str
        ):
            continue
        content = message.content[:_MAX_CONTINUATION_HISTORY_MESSAGE_CHARS]
        if not content:
            continue
        content = _truncate_utf8(content, remaining_bytes)
        if not content:
            break
        token_count = estimate_history_message_tokens(content)
        if token_count > remaining_tokens:
            break
        bounded.append(AgentHistoryMessage(role=message.role, content=content))
        remaining_bytes -= len(content.encode("utf-8"))
        remaining_tokens -= token_count
        if remaining_bytes <= 0 or remaining_tokens <= 0:
            break
    bounded.reverse()
    return tuple(bounded)


def _truncate_utf8(value: str, maximum_bytes: int) -> str:
    if maximum_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


__all__ = ["AgentRunProcessor", "AgentRunStore"]
