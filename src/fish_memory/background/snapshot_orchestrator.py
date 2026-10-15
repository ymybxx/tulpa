"""Shared LLM orchestration for normalized Snapshot Item mutations."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from fish_memory.answer.context import ChatMessage
from fish_memory.entities.models import SnapshotEntityContext
from fish_memory.entities.resolver import EntityResolutionResult
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.snapshots.confirmer import (
    SnapshotConfirmationResult,
    SnapshotEvidenceError,
)
from fish_memory.snapshots.models import (
    PreUpdateMarker,
    SnapshotFact,
    SnapshotItemMutation,
    SnapshotItemState,
    SnapshotPreUpdateResult,
)
from fish_memory.snapshots.operations import apply_confirmed_operations

_RECENT_TURN_LIMIT = 5


class SnapshotDerivationError(RuntimeError):
    """A retryable failure produced inside an LLM-derived Snapshot decision."""

    def __init__(self, *, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {cause}")


class SnapshotEvidenceStoreProtocol(Protocol):
    async def load_support_facts(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> tuple[SnapshotFact, ...]: ...

    async def load_recent_turns_for_fish(
        self,
        fish_id: UUID,
        through_message_id: UUID,
        limit: int = _RECENT_TURN_LIMIT,
    ) -> tuple[ChatMessage, ...]: ...


class SnapshotUpdaterProtocol(Protocol):
    async def preupdate(
        self,
        existing_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        current_time: datetime,
    ) -> SnapshotPreUpdateResult: ...


class SnapshotConfirmerProtocol(Protocol):
    async def confirm(
        self,
        *,
        item_index: int,
        item: SnapshotItemState,
        support_facts: Sequence[SnapshotFact],
        new_facts: Sequence[SnapshotFact],
        recent_turns: Sequence[ChatMessage],
    ) -> SnapshotConfirmationResult: ...


class EntityResolverProtocol(Protocol):
    async def resolve(
        self,
        existing_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        entity_context: SnapshotEntityContext,
    ) -> EntityResolutionResult: ...


@dataclass(frozen=True, slots=True)
class SnapshotOrchestrationStepNames:
    preupdate: str
    entity_resolution: str
    confirm: str
    aggregate: str


@dataclass(frozen=True, slots=True)
class SnapshotOrchestrationResult:
    mutations: tuple[SnapshotItemMutation, ...]
    entity_resolution: EntityResolutionResult


@dataclass(frozen=True, slots=True)
class _PreUpdateCallResult:
    result: SnapshotPreUpdateResult
    started_ns: int
    finished_ns: int


@dataclass(frozen=True, slots=True)
class _EntityCallResult:
    result: EntityResolutionResult
    started_ns: int
    finished_ns: int


@dataclass(frozen=True, slots=True)
class _ConfirmCallResult:
    result: SnapshotConfirmationResult
    started_ns: int
    finished_ns: int


class SnapshotMutationOrchestrator:
    """Run one pre-update/confirmation batch without merging or persisting it."""

    def __init__(
        self,
        evidence_store: SnapshotEvidenceStoreProtocol,
        updater: SnapshotUpdaterProtocol,
        confirmer: SnapshotConfirmerProtocol,
        resolver: EntityResolverProtocol,
        *,
        llm_semaphore: asyncio.Semaphore,
        allow_static_ignores: bool = False,
    ) -> None:
        self._evidence_store = evidence_store
        self._updater = updater
        self._confirmer = confirmer
        self._resolver = resolver
        self._llm_semaphore = llm_semaphore
        self._allow_static_ignores = allow_static_ignores

    async def orchestrate(
        self,
        *,
        fish_id: UUID,
        current_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        current_time: datetime,
        entity_context: SnapshotEntityContext,
        tracker: MessagePerformanceTracker,
        step_names: SnapshotOrchestrationStepNames,
    ) -> SnapshotOrchestrationResult:
        preupdate_call, entity_call = await self._run_initial_calls(
            current_items,
            new_facts,
            current_time,
            entity_context,
            tracker,
        )
        tracker.record(
            step_names.preupdate,
            started_ns=preupdate_call.started_ns,
            finished_ns=preupdate_call.finished_ns,
            input_payload=preupdate_call.result.request_payload,
            output_payload=preupdate_call.result.response_payload,
        )
        entity_resolution = entity_call.result
        if not entity_resolution.skipped:
            tracker.record(
                step_names.entity_resolution,
                started_ns=entity_call.started_ns,
                finished_ns=entity_call.finished_ns,
                input_payload=entity_resolution.request_payload,
                output_payload=entity_resolution.response_payload,
            )

        pre_update = preupdate_call.result.pre_update
        confirmations = await self._confirm_updates(
            fish_id=fish_id,
            current_items=current_items,
            new_facts=new_facts,
            markers=pre_update.updates,
            tracker=tracker,
            confirm_step_name=step_names.confirm,
        )
        confirmed_by_item = {
            marker.item_index: confirmation.result.operations
            for marker, confirmation in zip(
                pre_update.updates,
                confirmations,
                strict=True,
            )
        }
        aggregate_started_ns = tracker.now_ns()
        mutations = apply_confirmed_operations(
            items=current_items,
            new_facts=new_facts,
            pre_update=pre_update,
            confirmed_operations=confirmed_by_item,
            allow_static_ignores=self._allow_static_ignores,
        )
        tracker.record(
            step_names.aggregate,
            started_ns=aggregate_started_ns,
            finished_ns=tracker.now_ns(),
        )
        return SnapshotOrchestrationResult(
            mutations=mutations,
            entity_resolution=entity_resolution,
        )

    async def _run_initial_calls(
        self,
        current_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        current_time: datetime,
        entity_context: SnapshotEntityContext,
        tracker: MessagePerformanceTracker,
    ) -> tuple[_PreUpdateCallResult, _EntityCallResult]:
        async def run_preupdate() -> _PreUpdateCallResult:
            async with self._llm_semaphore:
                started_ns = tracker.now_ns()
                try:
                    result = await self._updater.preupdate(
                        current_items,
                        new_facts,
                        current_time,
                    )
                except ValueError as error:
                    raise SnapshotDerivationError(
                        stage="preupdate",
                        cause=error,
                    ) from error
                return _PreUpdateCallResult(
                    result=result,
                    started_ns=started_ns,
                    finished_ns=tracker.now_ns(),
                )

        async def run_entity_resolution() -> _EntityCallResult:
            async with self._llm_semaphore:
                started_ns = tracker.now_ns()
                try:
                    result = await self._resolver.resolve(
                        current_items,
                        new_facts,
                        entity_context,
                    )
                except ValueError as error:
                    raise SnapshotDerivationError(
                        stage="entity_resolution",
                        cause=error,
                    ) from error
                return _EntityCallResult(
                    result=result,
                    started_ns=started_ns,
                    finished_ns=tracker.now_ns(),
                )

        async with asyncio.TaskGroup() as task_group:
            preupdate_task = task_group.create_task(run_preupdate())
            entity_task = task_group.create_task(run_entity_resolution())
        return preupdate_task.result(), entity_task.result()

    async def _confirm_updates(
        self,
        *,
        fish_id: UUID,
        current_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        markers: Sequence[PreUpdateMarker],
        tracker: MessagePerformanceTracker,
        confirm_step_name: str,
    ) -> tuple[_ConfirmCallResult, ...]:
        if not markers:
            return ()

        support_ids = tuple(
            dict.fromkeys(
                fact_id
                for marker in markers
                for fact_id in current_items[marker.item_index].support_fact_ids
            )
        )
        anchors = tuple(
            _marker_anchor(marker, new_facts).source_message_id for marker in markers
        )
        unique_anchors = tuple(dict.fromkeys(anchors))
        async with asyncio.TaskGroup() as task_group:
            support_task = task_group.create_task(
                self._evidence_store.load_support_facts(fish_id, support_ids)
            )
            recent_tasks = {
                anchor: task_group.create_task(
                    self._evidence_store.load_recent_turns_for_fish(
                        fish_id,
                        anchor,
                        _RECENT_TURN_LIMIT,
                    )
                )
                for anchor in unique_anchors
            }
        support_by_id = {fact.id: fact for fact in support_task.result()}
        recent_by_anchor = {
            anchor: task.result() for anchor, task in recent_tasks.items()
        }

        async def run_confirmation(
            marker: PreUpdateMarker,
            anchor: UUID,
        ) -> _ConfirmCallResult:
            item = current_items[marker.item_index]
            support_facts = tuple(
                support_by_id[fact_id] for fact_id in item.support_fact_ids
            )
            assigned_facts = tuple(
                new_facts[index] for index in marker.source_fact_indexes
            )
            async with self._llm_semaphore:
                started_ns = tracker.now_ns()
                try:
                    result = await self._confirmer.confirm(
                        item_index=marker.item_index,
                        item=item,
                        support_facts=support_facts,
                        new_facts=assigned_facts,
                        recent_turns=recent_by_anchor[anchor],
                    )
                except SnapshotEvidenceError:
                    raise
                except ValueError as error:
                    raise SnapshotDerivationError(
                        stage="confirmation",
                        cause=error,
                    ) from error
                return _ConfirmCallResult(
                    result=result,
                    started_ns=started_ns,
                    finished_ns=tracker.now_ns(),
                )

        async with asyncio.TaskGroup() as task_group:
            tasks = tuple(
                task_group.create_task(run_confirmation(marker, anchor))
                for marker, anchor in zip(markers, anchors, strict=True)
            )
        results = tuple(task.result() for task in tasks)
        for result in results:
            tracker.record(
                confirm_step_name,
                started_ns=result.started_ns,
                finished_ns=result.finished_ns,
                input_payload=result.result.request_payload,
                output_payload=result.result.response_payload,
            )
        return results


def entity_context_fact_ids(
    items: Sequence[SnapshotItemState],
    new_facts: Sequence[SnapshotFact],
) -> tuple[UUID, ...]:
    return tuple(
        dict.fromkeys(
            (
                *(fact_id for item in items for fact_id in item.support_fact_ids),
                *(fact.id for fact in new_facts),
            )
        )
    )


def primary_orchestration_exception(exception: Exception) -> BaseException:
    leaves = _exception_leaves(exception)
    return next(
        (leaf for leaf in leaves if not isinstance(leaf, SnapshotDerivationError)),
        leaves[0],
    )


def _exception_leaves(exception: BaseException) -> tuple[BaseException, ...]:
    if not isinstance(exception, BaseExceptionGroup) or not exception.exceptions:
        return (exception,)
    return tuple(
        leaf for child in exception.exceptions for leaf in _exception_leaves(child)
    )


def _marker_anchor(
    marker: PreUpdateMarker,
    new_facts: Sequence[SnapshotFact],
) -> SnapshotFact:
    return max(
        (new_facts[index] for index in marker.source_fact_indexes),
        key=lambda fact: fact.sequence,
    )
