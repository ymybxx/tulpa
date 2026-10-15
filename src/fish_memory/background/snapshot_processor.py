import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from fish_memory.background.snapshot_orchestrator import (
    EntityResolverProtocol,
    SnapshotConfirmerProtocol,
    SnapshotDerivationError,
    SnapshotEvidenceStoreProtocol,
    SnapshotMutationOrchestrator,
    SnapshotOrchestrationStepNames,
    SnapshotUpdaterProtocol,
    entity_context_fact_ids,
    primary_orchestration_exception,
)
from fish_memory.entities.models import EntityMergeGroup, SnapshotEntityContext
from fish_memory.performance.models import PerformanceStep
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.snapshots.models import (
    PendingNormalizedSnapshotBatch,
    SnapshotItemMutation,
)

logger = logging.getLogger(__name__)

_CANDIDATE_LIMIT = 100
_DYNAMIC_STEP_NAMES = SnapshotOrchestrationStepNames(
    preupdate="llm.dynamic_snapshot.preupdate",
    entity_resolution="llm.dynamic_entity_resolution",
    confirm="llm.dynamic_snapshot.confirm",
    aggregate="snapshot.aggregate",
)


class SnapshotStoreProtocol(SnapshotEvidenceStoreProtocol, Protocol):
    async def list_pending_normalized_batches(
        self,
        limit: int,
        batch_size: int,
    ) -> list[PendingNormalizedSnapshotBatch]: ...

    async def persist_mutations(
        self,
        batch: PendingNormalizedSnapshotBatch,
        mutations: Sequence[SnapshotItemMutation],
    ) -> bool: ...

    async def record_error(
        self,
        fish_id: UUID,
        session_id: UUID,
        error: str,
    ) -> None: ...


class EntityStoreProtocol(Protocol):
    async def load_context(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> SnapshotEntityContext: ...

    async def merge_groups(
        self,
        fish_id: UUID,
        groups: Sequence[EntityMergeGroup],
    ) -> None: ...


class PerformanceStoreProtocol(Protocol):
    async def insert_steps(
        self,
        message_id: UUID,
        steps: Sequence[PerformanceStep],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _RetryState:
    failure_count: int
    retry_at: datetime | None


class DynamicSnapshotProcessor:
    def __init__(
        self,
        store: SnapshotStoreProtocol,
        updater: SnapshotUpdaterProtocol,
        confirmer: SnapshotConfirmerProtocol,
        resolver: EntityResolverProtocol,
        entity_store: EntityStoreProtocol,
        performance_store: PerformanceStoreProtocol,
        *,
        llm_semaphore: asyncio.Semaphore,
        max_retries: int,
        batch_size: int,
        clock: Callable[[], int] = time.perf_counter_ns,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        retry_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_retries < 1:
            raise ValueError("max retries must be at least 1")
        if batch_size < 1:
            raise ValueError("batch size must be at least 1")
        self._store = store
        self._orchestrator = SnapshotMutationOrchestrator(
            store,
            updater,
            confirmer,
            resolver,
            llm_semaphore=llm_semaphore,
        )
        self._entity_store = entity_store
        self._performance_store = performance_store
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._clock = clock
        self._now = now
        self._retry_now = retry_now
        self._retry_states: dict[tuple[UUID, UUID], _RetryState] = {}

    async def process_once(self) -> bool:
        candidates = await self._store.list_pending_normalized_batches(
            _CANDIDATE_LIMIT,
            self._batch_size,
        )
        for batch in candidates:
            retry_state = self._retry_states.get((batch.fish_id, batch.session_id))
            if retry_state is not None and (
                retry_state.retry_at is None or self._retry_now() < retry_state.retry_at
            ):
                continue
            await self._process(batch)
            return True
        return False

    async def _process(self, batch: PendingNormalizedSnapshotBatch) -> None:
        current_time = self._now()
        worker_started_ns = self._clock()
        queue_wait_us = max(
            0,
            int(
                (current_time - batch.facts[0].mentioned_at).total_seconds() * 1_000_000
            ),
        )
        origin_ns = worker_started_ns - queue_wait_us * 1_000
        tracker = MessagePerformanceTracker(clock=self._clock, origin_ns=origin_ns)
        tracker.record(
            "snapshot.queue_wait",
            started_ns=origin_ns,
            finished_ns=worker_started_ns,
        )

        try:
            load_started_ns = tracker.now_ns()
            # Expiration only controls Task 7 Working Memory rendering. Every
            # current Item must remain eligible for evolution.
            current_items = batch.snapshot.sorted_items()
            entity_context = await self._entity_store.load_context(
                batch.fish_id,
                entity_context_fact_ids(current_items, batch.facts),
            )
            tracker.record(
                "snapshot.load",
                started_ns=load_started_ns,
                finished_ns=tracker.now_ns(),
            )

            orchestration = await self._orchestrator.orchestrate(
                fish_id=batch.fish_id,
                current_items=current_items,
                new_facts=batch.facts,
                current_time=current_time,
                entity_context=entity_context,
                tracker=tracker,
                step_names=_DYNAMIC_STEP_NAMES,
            )
            if orchestration.entity_resolution.groups:
                await self._entity_store.merge_groups(
                    batch.fish_id,
                    orchestration.entity_resolution.groups,
                )

            persist_started_ns = tracker.now_ns()
            persisted = await self._store.persist_mutations(
                batch,
                orchestration.mutations,
            )
            persist_finished_ns = tracker.now_ns()
            self._retry_states.pop((batch.fish_id, batch.session_id), None)
            if not persisted:
                return
            tracker.record(
                "snapshot.persist",
                started_ns=persist_started_ns,
                finished_ns=persist_finished_ns,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            primary = primary_orchestration_exception(exception)
            if not self._record_failure(batch.fish_id, batch.session_id):
                return
            if isinstance(primary, SnapshotDerivationError):
                await self._skip_derivation(batch, tracker, origin_ns, primary)
                return
            await self._store.record_error(
                batch.fish_id,
                batch.session_id,
                str(primary),
            )
            return

        tracker.record(
            "snapshot.total",
            started_ns=origin_ns,
            finished_ns=tracker.now_ns(),
        )
        try:
            await self._performance_store.insert_steps(
                batch.trigger_message_id,
                tracker.snapshot(),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Failed to persist Dynamic Snapshot performance steps for %s",
                batch.trigger_message_id,
            )

    async def _skip_derivation(
        self,
        batch: PendingNormalizedSnapshotBatch,
        tracker: MessagePerformanceTracker,
        origin_ns: int,
        error: SnapshotDerivationError,
    ) -> None:
        skipped_started_ns = tracker.now_ns()
        try:
            persisted = await self._store.persist_mutations(batch, ())
        except asyncio.CancelledError:
            raise
        except Exception as persist_error:
            await self._store.record_error(
                batch.fish_id,
                batch.session_id,
                str(persist_error),
            )
            return
        self._retry_states.pop((batch.fish_id, batch.session_id), None)
        if not persisted:
            return
        skipped_finished_ns = tracker.now_ns()
        tracker.record(
            "snapshot.derivation_skipped",
            started_ns=skipped_started_ns,
            finished_ns=skipped_finished_ns,
            output_payload={"stage": error.stage},
            error=str(error),
        )
        tracker.record(
            "snapshot.total",
            started_ns=origin_ns,
            finished_ns=tracker.now_ns(),
            error="SnapshotDerivationSkipped",
        )
        try:
            await self._performance_store.insert_steps(
                batch.trigger_message_id,
                tracker.snapshot(),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Failed to persist skipped Dynamic Snapshot diagnostics for %s",
                batch.trigger_message_id,
            )

    def _record_failure(self, fish_id: UUID, session_id: UUID) -> bool:
        key = (fish_id, session_id)
        previous = self._retry_states.get(key)
        failure_count = 1 if previous is None else previous.failure_count + 1
        if failure_count > self._max_retries:
            self._retry_states[key] = _RetryState(
                failure_count=failure_count,
                retry_at=None,
            )
            return True

        delay_seconds = 2 ** min(failure_count - 1, self._max_retries - 1)
        self._retry_states[key] = _RetryState(
            failure_count=failure_count,
            retry_at=self._retry_now() + timedelta(seconds=delay_seconds),
        )
        return False
