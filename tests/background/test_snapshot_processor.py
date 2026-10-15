import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.background.snapshot_processor import DynamicSnapshotProcessor
from fish_memory.entities.models import EntityMergeGroup, SnapshotEntityContext
from fish_memory.entities.resolver import EntityResolutionResult
from fish_memory.models import Role
from fish_memory.performance.models import PerformanceStep
from fish_memory.snapshots.confirmer import (
    SnapshotConfirmationResult,
    SnapshotEvidenceError,
    SnapshotUpdateConfirmer,
)
from fish_memory.snapshots.models import (
    ConfirmedCreate,
    ConfirmedUpdate,
    NormalizedDynamicSnapshot,
    PendingNormalizedSnapshotBatch,
    PreCreateOperation,
    PreUpdateMarker,
    SnapshotFact,
    SnapshotItemMutation,
    SnapshotItemState,
    SnapshotPreUpdate,
    SnapshotPreUpdateResult,
)

NOW = datetime(2026, 8, 27, 12, tzinfo=UTC)
FISH_ONE = UUID(int=1)
SESSION_ONE = UUID(int=21)
SESSION_TWO = UUID(int=22)
MESSAGE_ONE = UUID(int=31)
MESSAGE_TWO = UUID(int=32)
MESSAGE_THREE = UUID(int=33)
OLD_FACT_ONE = UUID(int=101)
OLD_FACT_TWO = UUID(int=102)


def make_fact(
    fact_id: int,
    sequence: int,
    message_id: UUID,
    text: str,
    *,
    fish_id: UUID = FISH_ONE,
) -> SnapshotFact:
    return SnapshotFact(
        id=UUID(int=fact_id),
        sequence=sequence,
        fish_id=fish_id,
        source_message_id=message_id,
        source_role=Role.USER,
        text=text,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW + timedelta(minutes=sequence),
        entities=(text,),
    )


def make_item(
    item_id: int,
    summary: str,
    support_fact_id: UUID,
    *,
    days_ago: int,
) -> SnapshotItemState:
    return SnapshotItemState(
        id=item_id,
        summary=summary,
        support_fact_ids=(support_fact_id,),
        lineage_fact_ids=(support_fact_id,),
        last_supported_at=NOW - timedelta(days=days_ago),
        item_version=2,
    )


def make_batch(
    *,
    duplicate_new_fact_ids: bool = False,
) -> PendingNormalizedSnapshotBatch:
    facts = (
        make_fact(201, 5, MESSAGE_ONE, "更新旧主题"),
        make_fact(202, 6, MESSAGE_TWO, "继续最近主题"),
        make_fact(
            202 if duplicate_new_fact_ids else 203,
            7,
            MESSAGE_TWO,
            "拆分最近主题",
        ),
        make_fact(204, 8, MESSAGE_THREE, "直接新主题"),
    )
    # Intentionally reverse database order. The processor owns prompt ordering.
    items = (
        make_item(42, "最近主题", OLD_FACT_TWO, days_ago=1),
        make_item(41, "很久以前的主题", OLD_FACT_ONE, days_ago=30),
    )
    return PendingNormalizedSnapshotBatch(
        fish_id=FISH_ONE,
        session_id=SESSION_ONE,
        facts=facts,
        snapshot=NormalizedDynamicSnapshot(
            items=items,
            through_fact_sequence=4,
            version=3,
        ),
        target_fact_sequence=8,
        trigger_message_id=MESSAGE_THREE,
    )


def make_preupdate_result() -> SnapshotPreUpdateResult:
    return SnapshotPreUpdateResult(
        pre_update=SnapshotPreUpdate(
            creates=(
                PreCreateOperation(
                    summary="直接创建的主题",
                    source_fact_indexes=(3,),
                ),
            ),
            updates=(
                PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),
                PreUpdateMarker(item_index=1, source_fact_indexes=(1, 2)),
            ),
        ),
        request_payload={"prompt": "dynamic-preupdate"},
        response_payload={"content": "dynamic-markers"},
    )


def make_support_fact(fact_id: UUID, text: str) -> SnapshotFact:
    return SnapshotFact(
        id=fact_id,
        sequence=1,
        fish_id=FISH_ONE,
        source_message_id=UUID(int=fact_id.int + 500),
        source_role=Role.USER,
        text=text,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW - timedelta(days=40),
        entities=(text,),
    )


class CallGate:
    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.entered = 0
        self.all_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.entered += 1
        if self.entered == self.expected:
            self.all_entered.set()
        await self.release.wait()


class FakeStore:
    def __init__(self, batches: Sequence[PendingNormalizedSnapshotBatch]) -> None:
        self.batches = list(batches)
        self.requests: list[tuple[int, int]] = []
        self.support_calls: list[tuple[UUID, tuple[UUID, ...]]] = []
        self.recent_calls: list[tuple[UUID, UUID, int]] = []
        self.persisted: list[
            tuple[PendingNormalizedSnapshotBatch, tuple[SnapshotItemMutation, ...]]
        ] = []
        self.errors: list[tuple[UUID, UUID, str]] = []
        self.persist_result = True
        self.support_facts = {
            OLD_FACT_ONE: make_support_fact(OLD_FACT_ONE, "旧事实一"),
            OLD_FACT_TWO: make_support_fact(OLD_FACT_TWO, "旧事实二"),
        }
        self.turns = {
            MESSAGE_ONE: ({"role": "user", "content": "第一轮"},),
            MESSAGE_TWO: ({"role": "user", "content": "第二轮"},),
        }
        self.events: list[str] = []

    async def list_pending_normalized_batches(
        self,
        limit: int,
        batch_size: int,
    ) -> list[PendingNormalizedSnapshotBatch]:
        self.requests.append((limit, batch_size))
        return list(self.batches)

    async def load_support_facts(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> tuple[SnapshotFact, ...]:
        self.support_calls.append((fish_id, tuple(fact_ids)))
        return tuple(self.support_facts[fact_id] for fact_id in fact_ids)

    async def load_recent_turns_for_fish(
        self,
        fish_id: UUID,
        through_message_id: UUID,
        limit: int = 5,
    ) -> tuple[ChatMessage, ...]:
        self.recent_calls.append((fish_id, through_message_id, limit))
        return self.turns[through_message_id]  # type: ignore[return-value]

    async def persist_mutations(
        self,
        batch: PendingNormalizedSnapshotBatch,
        mutations: Sequence[SnapshotItemMutation],
    ) -> bool:
        self.events.append("persist")
        self.persisted.append((batch, tuple(mutations)))
        return self.persist_result

    async def record_error(
        self,
        fish_id: UUID,
        session_id: UUID,
        error: str,
    ) -> None:
        self.errors.append((fish_id, session_id, error))


class FakeUpdater:
    def __init__(
        self,
        semaphore: asyncio.Semaphore,
        result: SnapshotPreUpdateResult | None = None,
    ) -> None:
        self.semaphore = semaphore
        self.result = result or make_preupdate_result()
        self.calls: list[
            tuple[tuple[SnapshotItemState, ...], tuple[SnapshotFact, ...], datetime]
        ] = []
        self.gate: CallGate | None = None
        self.failure: BaseException | None = None

    async def preupdate(
        self,
        existing_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        current_time: datetime,
    ) -> SnapshotPreUpdateResult:
        self.calls.append((tuple(existing_items), tuple(new_facts), current_time))
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        return self.result


class FakeResolver:
    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self.semaphore = semaphore
        self.result = EntityResolutionResult(
            groups=(),
            dropped_groups=(),
            request_payload=None,
            response_payload=None,
            skipped=True,
        )
        self.calls: list[
            tuple[
                tuple[SnapshotItemState, ...],
                tuple[SnapshotFact, ...],
                SnapshotEntityContext,
            ]
        ] = []
        self.gate: CallGate | None = None
        self.failure: BaseException | None = None

    async def resolve(
        self,
        existing_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        entity_context: SnapshotEntityContext,
    ) -> EntityResolutionResult:
        self.calls.append((tuple(existing_items), tuple(new_facts), entity_context))
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        return self.result


@dataclass(frozen=True, slots=True)
class ConfirmCall:
    item_index: int
    item: SnapshotItemState
    support_facts: tuple[SnapshotFact, ...]
    new_facts: tuple[SnapshotFact, ...]
    recent_turns: tuple[ChatMessage, ...]


class FakeConfirmer:
    def __init__(self) -> None:
        self.calls: list[ConfirmCall] = []
        self.completion_order: list[int] = []
        self.delay_by_index: dict[int, float] = {}
        self.gate: CallGate | None = None
        self.failure: ValueError | None = None
        self.results = {
            0: SnapshotConfirmationResult(
                operations=(
                    ConfirmedUpdate(
                        basis="新事实替换旧事实",
                        item_index=0,
                        new_support_indexes=(0,),
                        remove_support_indexes=(0,),
                        summary="很久以前的主题已更新",
                    ),
                ),
                request_payload={"confirm": 0},
                response_payload={"content": "confirmed-0"},
            ),
            1: SnapshotConfirmationResult(
                operations=(
                    ConfirmedCreate(
                        basis="原主题应拆分",
                        new_support_indexes=(1,),
                        summary="确认阶段拆出的主题",
                    ),
                ),
                request_payload={"confirm": 1},
                response_payload={"content": "confirmed-1"},
            ),
        }

    async def confirm(
        self,
        *,
        item_index: int,
        item: SnapshotItemState,
        support_facts: Sequence[SnapshotFact],
        new_facts: Sequence[SnapshotFact],
        recent_turns: Sequence[ChatMessage],
    ) -> SnapshotConfirmationResult:
        self.calls.append(
            ConfirmCall(
                item_index=item_index,
                item=item,
                support_facts=tuple(support_facts),
                new_facts=tuple(new_facts),
                recent_turns=tuple(recent_turns),
            )
        )
        if self.failure is not None:
            raise self.failure
        if self.gate is not None:
            await self.gate.wait()
        await asyncio.sleep(self.delay_by_index.get(item_index, 0))
        self.completion_order.append(item_index)
        return self.results[item_index]


class FailingConfirmer(FakeConfirmer):
    def __init__(self) -> None:
        super().__init__()
        self.all_entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def confirm(
        self,
        *,
        item_index: int,
        item: SnapshotItemState,
        support_facts: Sequence[SnapshotFact],
        new_facts: Sequence[SnapshotFact],
        recent_turns: Sequence[ChatMessage],
    ) -> SnapshotConfirmationResult:
        self.calls.append(
            ConfirmCall(
                item_index,
                item,
                tuple(support_facts),
                tuple(new_facts),
                tuple(recent_turns),
            )
        )
        if len(self.calls) == 2:
            self.all_entered.set()
        await self.all_entered.wait()
        if item_index == 0:
            raise ValueError("malformed confirmation")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("unreachable")


class BlockingConfirmer(FakeConfirmer):
    def __init__(self) -> None:
        super().__init__()
        self.all_entered = asyncio.Event()
        self.cancelled_count = 0

    async def confirm(
        self,
        *,
        item_index: int,
        item: SnapshotItemState,
        support_facts: Sequence[SnapshotFact],
        new_facts: Sequence[SnapshotFact],
        recent_turns: Sequence[ChatMessage],
    ) -> SnapshotConfirmationResult:
        self.calls.append(
            ConfirmCall(
                item_index,
                item,
                tuple(support_facts),
                tuple(new_facts),
                tuple(recent_turns),
            )
        )
        if len(self.calls) == 2:
            self.all_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled_count += 1
            raise
        raise AssertionError("unreachable")


class FakeEntityStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.context = SnapshotEntityContext()
        self.load_calls: list[tuple[UUID, tuple[UUID, ...]]] = []
        self.merge_calls: list[tuple[UUID, tuple[EntityMergeGroup, ...]]] = []
        self.merge_failure: BaseException | None = None

    async def load_context(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> SnapshotEntityContext:
        self.load_calls.append((fish_id, tuple(fact_ids)))
        return self.context

    async def merge_groups(
        self,
        fish_id: UUID,
        groups: Sequence[EntityMergeGroup],
    ) -> None:
        self.events.append("merge")
        if self.merge_failure is not None:
            raise self.merge_failure
        self.merge_calls.append((fish_id, tuple(groups)))


class FakePerformanceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, tuple[PerformanceStep, ...]]] = []
        self.failure: BaseException | None = None

    async def insert_steps(
        self,
        message_id: UUID,
        steps: Sequence[PerformanceStep],
    ) -> None:
        if self.failure is not None:
            raise self.failure
        self.calls.append((message_id, tuple(steps)))


class IncrementingClock:
    def __init__(self) -> None:
        self.value = 10_000_000_000

    def __call__(self) -> int:
        value = self.value
        self.value += 100_000
        return value


@dataclass(frozen=True, slots=True)
class Harness:
    processor: DynamicSnapshotProcessor
    store: FakeStore
    updater: FakeUpdater
    resolver: FakeResolver
    confirmer: FakeConfirmer
    entities: FakeEntityStore
    performance: FakePerformanceStore
    semaphore: asyncio.Semaphore


def make_harness(
    batches: Sequence[PendingNormalizedSnapshotBatch],
    *,
    confirmer: FakeConfirmer | None = None,
    concurrency: int = 2,
    now: Callable[[], datetime] = lambda: NOW,
    retry_now: Callable[[], datetime] | None = None,
    max_retries: int = 3,
) -> Harness:
    semaphore = asyncio.Semaphore(concurrency)
    store = FakeStore(batches)
    updater = FakeUpdater(semaphore)
    resolver = FakeResolver(semaphore)
    chosen_confirmer = confirmer or FakeConfirmer()
    entities = FakeEntityStore(store.events)
    performance = FakePerformanceStore()
    processor = DynamicSnapshotProcessor(
        store,
        updater,
        cast_confirmer(chosen_confirmer),
        resolver,
        entities,
        performance,
        llm_semaphore=semaphore,
        max_retries=max_retries,
        batch_size=20,
        clock=IncrementingClock(),
        now=now,
        retry_now=now if retry_now is None else retry_now,
    )
    return Harness(
        processor,
        store,
        updater,
        resolver,
        chosen_confirmer,
        entities,
        performance,
        semaphore,
    )


def cast_confirmer(value: FakeConfirmer) -> SnapshotUpdateConfirmer:
    return value  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_no_candidate_returns_false_without_llm() -> None:
    state = make_harness(())

    assert await state.processor.process_once() is False
    assert state.store.requests == [(100, 20)]
    assert state.updater.calls == []


@pytest.mark.asyncio
async def test_orchestrates_parallel_confirmation_and_atomic_mutations() -> None:
    """Catches serial confirms, unstable ordering, or partial legacy persistence."""
    state = make_harness((make_batch(),))
    preupdate_gate = CallGate(2)
    confirm_gate = CallGate(2)
    state.updater.gate = preupdate_gate
    state.resolver.gate = preupdate_gate
    state.confirmer.gate = confirm_gate
    state.confirmer.delay_by_index[0] = 0.01
    group = EntityMergeGroup(("安安", "妮妮"))
    state.resolver.result = EntityResolutionResult(
        groups=(group,),
        dropped_groups=(),
        request_payload={"prompt": "entity"},
        response_payload={"content": "entity-groups"},
    )

    task = asyncio.create_task(state.processor.process_once())
    await asyncio.wait_for(preupdate_gate.all_entered.wait(), timeout=1)
    assert state.semaphore.locked()
    preupdate_gate.release.set()
    await asyncio.wait_for(confirm_gate.all_entered.wait(), timeout=1)
    assert state.semaphore.locked()
    confirm_gate.release.set()

    assert await asyncio.wait_for(task, timeout=1) is True
    # Dynamic pre-update sees all current items, including the 30-day-old one.
    assert [item.id for item in state.updater.calls[0][0]] == [41, 42]
    assert state.resolver.calls[0][0] == state.updater.calls[0][0]
    batch = state.store.persisted[0][0]
    assert state.entities.load_calls == [
        (
            FISH_ONE,
            (
                OLD_FACT_ONE,
                OLD_FACT_TWO,
                *(fact.id for fact in batch.facts),
            ),
        )
    ]
    assert state.store.support_calls == [(FISH_ONE, (OLD_FACT_ONE, OLD_FACT_TWO))]
    assert state.store.recent_calls == [
        (FISH_ONE, MESSAGE_ONE, 5),
        (FISH_ONE, MESSAGE_TWO, 5),
    ]
    assert [call.item_index for call in state.confirmer.calls] == [0, 1]
    assert state.confirmer.completion_order == [1, 0]
    assert [fact.id for fact in state.confirmer.calls[0].support_facts] == [
        OLD_FACT_ONE
    ]
    assert [fact.id for fact in state.confirmer.calls[1].support_facts] == [
        OLD_FACT_TWO
    ]
    assert state.confirmer.calls[0].recent_turns == (
        {"role": "user", "content": "第一轮"},
    )
    assert state.confirmer.calls[1].recent_turns == (
        {"role": "user", "content": "第二轮"},
    )

    mutations = state.store.persisted[0][1]
    assert [(mutation.action, mutation.summary) for mutation in mutations] == [
        ("update", "很久以前的主题已更新"),
        ("create", "直接创建的主题"),
        ("create", "确认阶段拆出的主题"),
    ]
    assert mutations[0].support_fact_ids == (batch.facts[0].id,)
    assert mutations[0].lineage_fact_ids == (OLD_FACT_ONE, batch.facts[0].id)
    assert mutations[2].support_fact_ids == (batch.facts[2].id,)
    assert state.store.events == ["merge", "persist"]
    assert state.entities.merge_calls == [(FISH_ONE, (group,))]

    _, steps = state.performance.calls[0]
    assert [step.name for step in steps] == [
        "snapshot.queue_wait",
        "snapshot.load",
        "llm.dynamic_snapshot.preupdate",
        "llm.dynamic_entity_resolution",
        "llm.dynamic_snapshot.confirm",
        "llm.dynamic_snapshot.confirm",
        "snapshot.aggregate",
        "snapshot.persist",
        "snapshot.total",
    ]
    preupdate_step = next(
        step for step in steps if step.name == "llm.dynamic_snapshot.preupdate"
    )
    assert preupdate_step.input_payload == {"prompt": "dynamic-preupdate"}
    assert preupdate_step.output_payload == {"content": "dynamic-markers"}
    confirm_steps = [
        step for step in steps if step.name == "llm.dynamic_snapshot.confirm"
    ]
    assert [step.index for step in confirm_steps] == [0, 1]
    assert [step.input_payload for step in confirm_steps] == [
        {"confirm": 0},
        {"confirm": 1},
    ]
    assert [step.output_payload for step in confirm_steps] == [
        {"content": "confirmed-0"},
        {"content": "confirmed-1"},
    ]
    aggregate_step = next(step for step in steps if step.name == "snapshot.aggregate")
    assert aggregate_step.index == 0
    assert aggregate_step.duration_us >= 0
    assert steps.index(aggregate_step) > steps.index(confirm_steps[-1])
    assert steps.index(aggregate_step) < next(
        index for index, step in enumerate(steps) if step.name == "snapshot.persist"
    )


@pytest.mark.asyncio
async def test_same_anchor_is_loaded_once_for_multiple_markers() -> None:
    """Catches redundant recent-turn queries that can also diverge in context."""
    batch = make_batch()
    facts = tuple(
        SnapshotFact(
            id=fact.id,
            sequence=fact.sequence,
            fish_id=fact.fish_id,
            source_message_id=MESSAGE_ONE,
            source_role=fact.source_role,
            text=fact.text,
            occurred_start=fact.occurred_start,
            occurred_end=fact.occurred_end,
            mentioned_at=fact.mentioned_at,
            entities=fact.entities,
        )
        for fact in batch.facts
    )
    state = make_harness(
        (
            PendingNormalizedSnapshotBatch(
                fish_id=batch.fish_id,
                session_id=batch.session_id,
                facts=facts,
                snapshot=batch.snapshot,
                target_fact_sequence=batch.target_fact_sequence,
                trigger_message_id=batch.trigger_message_id,
            ),
        )
    )

    assert await state.processor.process_once()

    assert state.store.recent_calls == [(FISH_ONE, MESSAGE_ONE, 5)]


@pytest.mark.asyncio
async def test_confirmation_failure_cancels_siblings_and_writes_nothing() -> None:
    """Catches gather-style leaked tasks or partial Entity/Snapshot commits."""
    confirmer = FailingConfirmer()
    state = make_harness((make_batch(),), confirmer=confirmer)

    assert await state.processor.process_once() is True

    assert confirmer.cancelled.is_set()
    assert state.entities.merge_calls == []
    assert state.store.persisted == []
    assert state.store.events == []
    assert state.performance.calls == []


@pytest.mark.asyncio
async def test_processor_cancellation_waits_for_all_confirmation_tasks() -> None:
    """Catches background confirms surviving their cancelled parent batch."""
    confirmer = BlockingConfirmer()
    state = make_harness((make_batch(),), confirmer=confirmer)
    task = asyncio.create_task(state.processor.process_once())
    await asyncio.wait_for(confirmer.all_entered.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert confirmer.cancelled_count == 2
    assert state.entities.merge_calls == []
    assert state.store.persisted == []


@pytest.mark.asyncio
async def test_global_validation_failure_writes_nothing() -> None:
    """Catches duplicate Fact IDs slipping through per-marker local validation."""
    state = make_harness((make_batch(duplicate_new_fact_ids=True),))

    assert await state.processor.process_once() is True

    assert len(state.confirmer.calls) == 2
    assert state.entities.merge_calls == []
    assert state.store.persisted == []
    assert state.store.events == []
    assert state.performance.calls == []


@pytest.mark.asyncio
async def test_entity_failure_prevents_confirm_and_snapshot_persistence() -> None:
    """Catches Entity failure being treated as optional after pre-update succeeds."""
    state = make_harness((make_batch(),))
    state.resolver.failure = RuntimeError("entity resolution failed")

    assert await state.processor.process_once() is True

    assert state.confirmer.calls == []
    assert state.entities.merge_calls == []
    assert state.store.persisted == []


@pytest.mark.asyncio
async def test_entity_merge_failure_prevents_snapshot_persistence() -> None:
    """Catches Snapshot progress advancing after an Entity merge dependency fails."""
    state = make_harness((make_batch(),))
    state.resolver.result = EntityResolutionResult(
        groups=(EntityMergeGroup(("安安", "妮妮")),),
        dropped_groups=(),
        request_payload=None,
        response_payload=None,
    )
    state.entities.merge_failure = RuntimeError("entity merge failed")

    assert await state.processor.process_once() is True

    assert state.store.persisted == []
    assert state.performance.calls == []


@pytest.mark.asyncio
async def test_snapshot_cas_conflict_may_leave_idempotent_entity_merge() -> None:
    """Catches an attempted rollback of an already successful Entity merge."""
    state = make_harness((make_batch(),))
    group = EntityMergeGroup(("安安", "妮妮"))
    state.resolver.result = EntityResolutionResult(
        groups=(group,),
        dropped_groups=(),
        request_payload=None,
        response_payload=None,
    )
    state.store.persist_result = False

    assert await state.processor.process_once() is True

    assert state.entities.merge_calls == [(FISH_ONE, (group,))]
    assert len(state.store.persisted) == 1
    assert state.performance.calls == []


@pytest.mark.asyncio
async def test_snapshot_cas_conflict_clears_previous_batch_retry_state() -> None:
    """Catches a stale batch's failures exhausting a fresh batch's retry budget."""
    retry_clock = MutableNow()
    state = make_harness(
        (make_batch(),),
        retry_now=retry_clock,
        max_retries=1,
    )
    state.updater.failure = ValueError("old batch provider output")
    assert await state.processor.process_once() is True

    retry_clock.advance(1)
    state.updater.failure = None
    state.store.persist_result = False
    assert await state.processor.process_once() is True

    state.store.persist_result = True
    state.updater.failure = ValueError("fresh batch provider output")
    assert await state.processor.process_once() is True

    assert len(state.store.persisted) == 1
    assert state.store.persisted[0][1]
    assert state.store.errors == []


@pytest.mark.asyncio
async def test_retry_backoff_skips_failed_session_and_records_terminal_error() -> None:
    """Catches retries spinning immediately or never publishing terminal failure."""
    retry_clock = MutableNow()
    first = make_batch()
    second = replace(first, session_id=SESSION_TWO)
    state = make_harness(
        (first, second),
        retry_now=retry_clock,
        max_retries=1,
    )
    state.updater.failure = RuntimeError("preupdate unavailable")

    assert await state.processor.process_once() is True
    assert state.store.errors == []
    # The first Session is backing off, so another Session can still advance.
    state.updater.failure = None
    assert await state.processor.process_once() is True
    assert state.store.persisted[0][0].session_id == SESSION_TWO

    state.updater.failure = RuntimeError("preupdate unavailable")
    retry_clock.advance(1)
    assert await state.processor.process_once() is True
    assert state.store.errors == [(FISH_ONE, SESSION_ONE, "preupdate unavailable")]


@pytest.mark.asyncio
async def test_terminal_derivation_failure_skips_batch_without_poisoning_fish() -> None:
    """Catches malformed provider output permanently blocking later Snapshot work."""
    retry_clock = MutableNow()
    state = make_harness(
        (make_batch(),),
        retry_now=retry_clock,
        max_retries=1,
    )
    state.updater.failure = ValueError("malformed provider output")

    assert await state.processor.process_once() is True
    assert state.store.persisted == []
    retry_clock.advance(1)
    assert await state.processor.process_once() is True

    assert state.store.persisted == [(make_batch(), ())]
    assert state.store.errors == []
    steps = state.performance.calls[0][1]
    skipped = next(step for step in steps if step.name == "snapshot.derivation_skipped")
    assert skipped.error == "preupdate: malformed provider output"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ("entity_resolution", "malformed entity output"),
        ("confirmation", "malformed confirmation output"),
    ],
)
async def test_all_provider_validation_stages_are_safe_to_skip(
    stage: str,
    message: str,
) -> None:
    """Catches Entity or confirmation JSON poisoning later Snapshot batches."""
    retry_clock = MutableNow()
    state = make_harness(
        (make_batch(),),
        retry_now=retry_clock,
        max_retries=1,
    )
    if stage == "entity_resolution":
        state.resolver.failure = ValueError(message)
    else:
        state.confirmer.failure = ValueError(message)

    assert await state.processor.process_once() is True
    retry_clock.advance(1)
    assert await state.processor.process_once() is True

    assert state.store.persisted == [(make_batch(), ())]
    assert state.store.errors == []
    steps = state.performance.calls[0][1]
    skipped = next(step for step in steps if step.name == "snapshot.derivation_skipped")
    assert skipped.error == f"{stage}: {message}"


@pytest.mark.asyncio
async def test_snapshot_evidence_consistency_error_remains_blocking() -> None:
    """Catches a database evidence mismatch being mislabeled as provider noise."""
    retry_clock = MutableNow()
    state = make_harness(
        (make_batch(),),
        retry_now=retry_clock,
        max_retries=1,
    )
    state.confirmer.failure = SnapshotEvidenceError("support facts do not match")

    assert await state.processor.process_once() is True
    retry_clock.advance(1)
    assert await state.processor.process_once() is True

    assert state.store.persisted == []
    assert state.store.errors == [(FISH_ONE, SESSION_ONE, "support facts do not match")]


@pytest.mark.asyncio
async def test_performance_failure_does_not_rollback_persisted_mutations() -> None:
    """Catches diagnostics storage being placed inside the Snapshot transaction."""
    state = make_harness((make_batch(),))
    state.performance.failure = RuntimeError("performance unavailable")

    assert await state.processor.process_once() is True

    assert len(state.store.persisted) == 1
    assert state.store.errors == []


class MutableNow:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)
