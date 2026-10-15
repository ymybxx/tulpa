import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.background.snapshot_orchestrator import SnapshotConfirmerProtocol
from fish_memory.background.static_profile_processor import StaticProfileProcessor
from fish_memory.entities.models import EntityMergeGroup, SnapshotEntityContext
from fish_memory.entities.resolver import EntityResolutionResult
from fish_memory.models import Role
from fish_memory.performance.models import PerformanceStep
from fish_memory.snapshots.confirmer import SnapshotConfirmationResult
from fish_memory.snapshots.models import (
    ConfirmedCreate,
    ConfirmedIgnore,
    ConfirmedUpdate,
    NormalizedStaticProfile,
    PendingNormalizedStaticProfileBatch,
    PreCreateOperation,
    PreUpdateMarker,
    SnapshotFact,
    SnapshotItemMutation,
    SnapshotItemState,
    SnapshotPreUpdate,
    SnapshotPreUpdateResult,
)

NOW = datetime(2026, 8, 27, 14, tzinfo=UTC)
FISH_ID = UUID(int=501)
MESSAGE_SESSION_A = UUID(int=511)
MESSAGE_SESSION_B = UUID(int=512)
OLD_FACT_A = UUID(int=521)
OLD_FACT_B = UUID(int=522)


def fact(
    fact_id: int,
    sequence: int,
    source_message_id: UUID,
    text: str,
) -> SnapshotFact:
    return SnapshotFact(
        id=UUID(int=fact_id),
        sequence=sequence,
        fish_id=FISH_ID,
        source_message_id=source_message_id,
        source_role=Role.USER,
        text=text,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW + timedelta(minutes=sequence),
        entities=(text,),
    )


def item(item_id: int, summary: str, support_id: UUID) -> SnapshotItemState:
    return SnapshotItemState(
        id=item_id,
        summary=summary,
        support_fact_ids=(support_id,),
        lineage_fact_ids=(support_id,),
        last_supported_at=NOW - timedelta(days=100),
        item_version=3,
    )


def batch() -> PendingNormalizedStaticProfileBatch:
    return PendingNormalizedStaticProfileBatch(
        fish_id=FISH_ID,
        facts=(
            fact(531, 9, MESSAGE_SESSION_A, "用户现在每天跑步"),
            fact(532, 10, MESSAGE_SESSION_B, "助手现在叫安安"),
            fact(533, 11, MESSAGE_SESSION_B, "用户长期喜欢爵士乐"),
        ),
        profile=NormalizedStaticProfile(
            items=(
                item(71, "用户每周跑步", OLD_FACT_A),
                item(72, "助手叫小美", OLD_FACT_B),
            ),
            through_fact_sequence=8,
            version=4,
        ),
        target_fact_sequence=11,
        trigger_message_id=MESSAGE_SESSION_B,
    )


def preupdate_result() -> SnapshotPreUpdateResult:
    return SnapshotPreUpdateResult(
        pre_update=SnapshotPreUpdate(
            creates=(
                PreCreateOperation(
                    summary="用户长期喜欢爵士乐",
                    source_fact_indexes=(2,),
                ),
            ),
            updates=(
                PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),
                PreUpdateMarker(item_index=1, source_fact_indexes=(1,)),
            ),
        ),
        request_payload={"prompt": "static-preupdate"},
        response_payload={"content": "static-markers"},
    )


def support_fact(fact_id: UUID, text: str) -> SnapshotFact:
    return SnapshotFact(
        id=fact_id,
        sequence=1,
        fish_id=FISH_ID,
        source_message_id=UUID(int=fact_id.int + 100),
        source_role=Role.USER,
        text=text,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW - timedelta(days=200),
        entities=(text,),
    )


class Gate:
    def __init__(self) -> None:
        self.entered = 0
        self.all_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.entered += 1
        if self.entered == 2:
            self.all_entered.set()
        await self.release.wait()


class FakeStore:
    def __init__(self, batches: Sequence[PendingNormalizedStaticProfileBatch]) -> None:
        self.batches = list(batches)
        self.requests: list[tuple[int, int]] = []
        self.support_calls: list[tuple[UUID, tuple[UUID, ...]]] = []
        self.recent_calls: list[tuple[UUID, UUID, int]] = []
        self.persisted: list[
            tuple[
                PendingNormalizedStaticProfileBatch,
                tuple[SnapshotItemMutation, ...],
            ]
        ] = []
        self.errors: list[tuple[UUID, str]] = []
        self.persist_result = True
        self.events: list[str] = []
        self.support = {
            OLD_FACT_A: support_fact(OLD_FACT_A, "旧跑步事实"),
            OLD_FACT_B: support_fact(OLD_FACT_B, "旧名字事实"),
        }
        self.turns = {
            MESSAGE_SESSION_A: ({"role": "user", "content": "会话A"},),
            MESSAGE_SESSION_B: ({"role": "user", "content": "会话B"},),
        }

    async def list_pending_normalized_static_batches(
        self,
        limit: int,
        batch_size: int,
    ) -> list[PendingNormalizedStaticProfileBatch]:
        self.requests.append((limit, batch_size))
        return list(self.batches)

    async def load_support_facts(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> tuple[SnapshotFact, ...]:
        self.support_calls.append((fish_id, tuple(fact_ids)))
        return tuple(self.support[fact_id] for fact_id in fact_ids)

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
        pending: PendingNormalizedStaticProfileBatch,
        mutations: Sequence[SnapshotItemMutation],
    ) -> bool:
        self.events.append("persist")
        self.persisted.append((pending, tuple(mutations)))
        return self.persist_result

    async def record_static_error(self, fish_id: UUID, error: str) -> None:
        self.errors.append((fish_id, error))


class FakeUpdater:
    def __init__(self) -> None:
        self.calls: list[tuple[SnapshotItemState, ...]] = []
        self.failure: BaseException | None = None
        self.result = preupdate_result()

    async def preupdate(
        self,
        existing_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        current_time: datetime,
    ) -> SnapshotPreUpdateResult:
        del new_facts, current_time
        self.calls.append(tuple(existing_items))
        if self.failure is not None:
            raise self.failure
        return self.result


class FakeResolver:
    def __init__(self) -> None:
        self.calls: list[tuple[SnapshotItemState, ...]] = []
        self.failure: BaseException | None = None
        self.result = EntityResolutionResult(
            groups=(),
            dropped_groups=(),
            request_payload=None,
            response_payload=None,
            skipped=True,
        )

    async def resolve(
        self,
        existing_items: Sequence[SnapshotItemState],
        new_facts: Sequence[SnapshotFact],
        entity_context: SnapshotEntityContext,
    ) -> EntityResolutionResult:
        del new_facts, entity_context
        self.calls.append(tuple(existing_items))
        if self.failure is not None:
            raise self.failure
        return self.result


@dataclass(frozen=True, slots=True)
class ConfirmCall:
    item_index: int
    recent_turns: tuple[ChatMessage, ...]


class FakeConfirmer:
    def __init__(self) -> None:
        self.calls: list[ConfirmCall] = []
        self.gate: Gate | None = None
        self.results = {
            0: SnapshotConfirmationResult(
                operations=(
                    ConfirmedUpdate(
                        basis="跑步习惯频率更新",
                        item_index=0,
                        new_support_indexes=(0,),
                        remove_support_indexes=(0,),
                        summary="用户每天跑步",
                    ),
                ),
                request_payload={"confirm": "run"},
                response_payload={"content": "run-updated"},
            ),
            1: SnapshotConfirmationResult(
                operations=(
                    ConfirmedCreate(
                        basis="名字不是原条目的同一主体",
                        new_support_indexes=(0,),
                        summary="助手现在叫安安",
                    ),
                ),
                request_payload={"confirm": "name"},
                response_payload={"content": "name-created"},
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
        del item, support_facts, new_facts
        self.calls.append(ConfirmCall(item_index, tuple(recent_turns)))
        if self.gate is not None:
            await self.gate.wait()
        return self.results[item_index]


class FakeEntityStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.load_calls: list[tuple[UUID, tuple[UUID, ...]]] = []
        self.merge_calls: list[tuple[UUID, tuple[EntityMergeGroup, ...]]] = []

    async def load_context(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> SnapshotEntityContext:
        self.load_calls.append((fish_id, tuple(fact_ids)))
        return SnapshotEntityContext()

    async def merge_groups(
        self,
        fish_id: UUID,
        groups: Sequence[EntityMergeGroup],
    ) -> None:
        self.events.append("merge")
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


class Clock:
    def __init__(self) -> None:
        self.value = 20_000_000_000

    def __call__(self) -> int:
        value = self.value
        self.value += 100_000
        return value


@dataclass(frozen=True, slots=True)
class Harness:
    processor: StaticProfileProcessor
    store: FakeStore
    updater: FakeUpdater
    resolver: FakeResolver
    confirmer: FakeConfirmer
    entities: FakeEntityStore
    performance: FakePerformanceStore
    semaphore: asyncio.Semaphore


def harness(
    batches: Sequence[PendingNormalizedStaticProfileBatch],
    *,
    retry_now: Callable[[], datetime] = lambda: NOW,
    max_retries: int = 3,
) -> Harness:
    semaphore = asyncio.Semaphore(2)
    store = FakeStore(batches)
    updater = FakeUpdater()
    resolver = FakeResolver()
    confirmer = FakeConfirmer()
    entities = FakeEntityStore(store.events)
    performance = FakePerformanceStore()
    processor = StaticProfileProcessor(
        store,
        updater,
        cast_confirmer(confirmer),
        resolver,
        entities,
        performance,
        llm_semaphore=semaphore,
        max_retries=max_retries,
        batch_size=20,
        clock=Clock(),
        now=lambda: NOW,
        retry_now=retry_now,
    )
    return Harness(
        processor,
        store,
        updater,
        resolver,
        confirmer,
        entities,
        performance,
        semaphore,
    )


def cast_confirmer(value: FakeConfirmer) -> SnapshotConfirmerProtocol:
    return value  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_no_candidate_returns_false() -> None:
    state = harness(())

    assert await state.processor.process_once() is False
    assert state.store.requests == [(100, 20)]


@pytest.mark.asyncio
async def test_static_markers_confirm_concurrently_with_anchor_session_context() -> (
    None
):
    """Catches Static updates reusing one Session or serializing confirmation calls."""
    state = harness((batch(),))
    gate = Gate()
    state.confirmer.gate = gate

    task = asyncio.create_task(state.processor.process_once())
    await asyncio.wait_for(gate.all_entered.wait(), timeout=1)
    assert state.semaphore.locked()
    gate.release.set()

    assert await asyncio.wait_for(task, timeout=1) is True
    assert state.store.support_calls == [(FISH_ID, (OLD_FACT_A, OLD_FACT_B))]
    assert state.store.recent_calls == [
        (FISH_ID, MESSAGE_SESSION_A, 5),
        (FISH_ID, MESSAGE_SESSION_B, 5),
    ]
    assert state.confirmer.calls == [
        ConfirmCall(0, ({"role": "user", "content": "会话A"},)),
        ConfirmCall(1, ({"role": "user", "content": "会话B"},)),
    ]
    mutations = state.store.persisted[0][1]
    assert [(mutation.action, mutation.summary) for mutation in mutations] == [
        ("update", "用户每天跑步"),
        ("create", "用户长期喜欢爵士乐"),
        ("create", "助手现在叫安安"),
    ]
    assert state.store.events == ["persist"]
    _, steps = state.performance.calls[0]
    assert [step.name for step in steps] == [
        "static.queue_wait",
        "static.load",
        "llm.static_profile.preupdate",
        "llm.static_profile.confirm",
        "llm.static_profile.confirm",
        "static.aggregate",
        "static.persist",
        "static.total",
    ]
    preupdate = next(
        step for step in steps if step.name == "llm.static_profile.preupdate"
    )
    assert preupdate.input_payload == {"prompt": "static-preupdate"}
    confirms = [step for step in steps if step.name == "llm.static_profile.confirm"]
    assert [step.index for step in confirms] == [0, 1]
    assert [step.input_payload for step in confirms] == [
        {"confirm": "run"},
        {"confirm": "name"},
    ]
    aggregate = next(step for step in steps if step.name == "static.aggregate")
    assert aggregate.index == 0
    assert aggregate.duration_us >= 0
    assert steps.index(aggregate) > steps.index(confirms[-1])
    assert steps.index(aggregate) < next(
        index for index, step in enumerate(steps) if step.name == "static.persist"
    )


@pytest.mark.asyncio
async def test_static_confirmation_can_ignore_a_misrouted_short_term_fact() -> None:
    temporary = fact(534, 9, MESSAGE_SESSION_A, "用户这次要求使用英文")
    pending = replace(
        batch(),
        facts=(temporary,),
        profile=NormalizedStaticProfile(
            items=(item(71, "用户长期使用中文", OLD_FACT_A),),
            through_fact_sequence=8,
            version=4,
        ),
        target_fact_sequence=9,
        trigger_message_id=MESSAGE_SESSION_A,
    )
    state = harness((pending,))
    state.updater.result = SnapshotPreUpdateResult(
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
        ),
        request_payload={"prompt": "static-preupdate"},
        response_payload={"content": "misrouted-marker"},
    )
    state.confirmer.results = {
        0: SnapshotConfirmationResult(
            operations=(
                ConfirmedIgnore(
                    basis="单次要求不改变长期偏好。",
                    new_support_indexes=(0,),
                ),
            ),
            request_payload={"confirm": "temporary-language"},
            response_payload={"content": "ignored"},
        )
    }

    assert await state.processor.process_once() is True

    assert state.store.persisted == [(pending, ())]
    assert state.store.errors == []
    assert state.confirmer.calls == [
        ConfirmCall(0, ({"role": "user", "content": "会话A"},))
    ]
    _, steps = state.performance.calls[0]
    assert [step.name for step in steps].count("llm.static_profile.confirm") == 1
    assert "static.persist" in [step.name for step in steps]


@pytest.mark.asyncio
async def test_static_preupdate_can_reject_all_facts_and_advance_progress() -> None:
    temporary = fact(535, 9, MESSAGE_SESSION_A, "用户询问今天的天气")
    pending = replace(
        batch(),
        facts=(temporary,),
        profile=NormalizedStaticProfile(
            items=(),
            through_fact_sequence=8,
            version=4,
        ),
        target_fact_sequence=9,
        trigger_message_id=MESSAGE_SESSION_A,
    )
    state = harness((pending,))
    state.updater.result = SnapshotPreUpdateResult(
        pre_update=SnapshotPreUpdate(creates=(), updates=()),
        request_payload={"prompt": "static-preupdate"},
        response_payload={"content": "all-short-term-facts-rejected"},
    )

    assert await state.processor.process_once() is True

    assert state.confirmer.calls == []
    assert state.store.persisted == [(pending, ())]
    assert state.store.errors == []
    _, steps = state.performance.calls[0]
    assert [step.name for step in steps].count("llm.static_profile.confirm") == 0
    assert "static.persist" in [step.name for step in steps]


@pytest.mark.asyncio
async def test_static_entity_failure_writes_no_snapshot_or_progress() -> None:
    state = harness((batch(),))
    state.resolver.failure = RuntimeError("entity failed")

    assert await state.processor.process_once() is True

    assert state.confirmer.calls == []
    assert state.entities.merge_calls == []
    assert state.store.persisted == []
    assert state.performance.calls == []


@pytest.mark.asyncio
async def test_static_aggregate_failure_does_not_persist() -> None:
    """Catches a failed global Fact validation being timed as a successful write."""
    pending = batch()
    duplicate = SnapshotFact(
        id=pending.facts[0].id,
        sequence=pending.facts[1].sequence,
        fish_id=pending.facts[1].fish_id,
        source_message_id=pending.facts[1].source_message_id,
        source_role=pending.facts[1].source_role,
        text=pending.facts[1].text,
        occurred_start=pending.facts[1].occurred_start,
        occurred_end=pending.facts[1].occurred_end,
        mentioned_at=pending.facts[1].mentioned_at,
        entities=pending.facts[1].entities,
    )
    state = harness(
        (replace(pending, facts=(pending.facts[0], duplicate, pending.facts[2])),)
    )

    assert await state.processor.process_once() is True

    assert len(state.confirmer.calls) == 2
    assert state.store.events == []
    assert state.store.persisted == []
    assert state.performance.calls == []


@pytest.mark.asyncio
async def test_static_cas_conflict_records_no_performance() -> None:
    state = harness((batch(),))
    state.store.persist_result = False

    assert await state.processor.process_once() is True

    assert len(state.store.persisted) == 1
    assert state.performance.calls == []


@pytest.mark.asyncio
async def test_static_cas_conflict_clears_previous_batch_retry_state() -> None:
    """Catches an old Static retry count leaking across a fresh CAS reload."""
    retry_clock = AdvancingRetryNow()
    state = harness(
        (batch(),),
        retry_now=retry_clock,
        max_retries=1,
    )
    state.updater.failure = ValueError("old batch provider output")
    assert await state.processor.process_once() is True

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
async def test_static_performance_failure_does_not_rollback_snapshot() -> None:
    state = harness((batch(),))
    state.performance.failure = RuntimeError("performance failed")

    assert await state.processor.process_once() is True

    assert len(state.store.persisted) == 1
    assert state.store.errors == []


class AdvancingRetryNow:
    def __init__(self) -> None:
        self.value = NOW

    def __call__(self) -> datetime:
        value = self.value
        self.value += timedelta(seconds=2)
        return value


@pytest.mark.asyncio
async def test_static_terminal_derivation_failure_advances_with_empty_mutation() -> (
    None
):
    """Catches one malformed Static response poisoning the Fish indefinitely."""
    retry_clock = AdvancingRetryNow()
    state = harness(
        (batch(),),
        retry_now=retry_clock,
        max_retries=1,
    )
    state.updater.failure = ValueError("ambiguous provider output")

    assert await state.processor.process_once() is True
    assert state.store.persisted == []
    assert await state.processor.process_once() is True

    assert state.store.persisted == [(batch(), ())]
    assert state.store.errors == []
    steps = state.performance.calls[0][1]
    skipped = next(step for step in steps if step.name == "static.derivation_skipped")
    assert skipped.error == "preupdate: ambiguous provider output"
