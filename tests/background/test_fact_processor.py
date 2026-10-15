import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fish_memory.background.fact_processor import FactProcessor
from fish_memory.db.embedding_spaces import EmbeddingSpaceMismatchError
from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.facts.extractor import FactExtractionResult
from fish_memory.facts.models import (
    ExtractedFact,
    FactExtractionContext,
    PendingFactMessage,
)
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.models import PerformanceStep

NOW = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)
FISH_ONE = UUID("00000000-0000-0000-0000-000000000001")
FISH_TWO = UUID("00000000-0000-0000-0000-000000000002")
MESSAGE_ONE = UUID("00000000-0000-0000-0000-000000000011")
MESSAGE_TWO = UUID("00000000-0000-0000-0000-000000000012")
SESSION_ONE = UUID("00000000-0000-0000-0000-000000000021")
SESSION_TWO = UUID("00000000-0000-0000-0000-000000000022")
EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model="text-embedding-v4",
    dimensions=1024,
)


@dataclass(frozen=True, slots=True)
class ExtractionCall:
    message: Message
    content: str
    context: FactExtractionContext | None
    llm_semaphore_locked: bool
    embedding_semaphore_locked: bool


@dataclass(frozen=True, slots=True)
class PersistCall:
    pending: PendingFactMessage
    facts: tuple[ExtractedFact, ...]
    vectors: tuple[tuple[float, ...], ...]
    embedding_space: EmbeddingSpace
    llm_semaphore_locked: bool
    embedding_semaphore_locked: bool


@dataclass(frozen=True, slots=True)
class FailureCall:
    pending: PendingFactMessage
    error: str
    terminal: bool


@dataclass(frozen=True, slots=True)
class PerformanceCall:
    message_id: UUID
    steps: tuple[PerformanceStep, ...]


@dataclass(frozen=True, slots=True)
class ProcessorHarness:
    processor: FactProcessor
    store: "FakeFactStore"
    extractor: "FakeExtractor"
    embedder: "FakeEmbedder"
    performance: "FakePerformanceStore"
    space_store: "FakeEmbeddingSpaceStore"
    llm_semaphore: asyncio.Semaphore
    embedding_semaphore: asyncio.Semaphore


class FakeFactStore:
    def __init__(
        self,
        candidates: Sequence[PendingFactMessage],
        llm_semaphore: asyncio.Semaphore,
        embedding_semaphore: asyncio.Semaphore,
    ) -> None:
        self.candidates = list(candidates)
        self.llm_semaphore = llm_semaphore
        self.embedding_semaphore = embedding_semaphore
        self.requested_limits: list[tuple[int, int]] = []
        self.persisted: list[PersistCall] = []
        self.failures: list[FailureCall] = []
        self.persist_failures: dict[UUID, BaseException] = {}

    async def list_pending_messages(
        self,
        limit: int,
        batch_size: int,
    ) -> list[PendingFactMessage]:
        self.requested_limits.append((limit, batch_size))
        return list(self.candidates)

    async def persist_extraction(
        self,
        pending: PendingFactMessage,
        facts: Sequence[ExtractedFact],
        vectors: Sequence[Sequence[float]],
        embedding_space: EmbeddingSpace,
    ) -> None:
        failure = self.persist_failures.get(pending.message.fish_id)
        if failure is not None:
            raise failure
        self.persisted.append(
            PersistCall(
                pending=pending,
                facts=tuple(facts),
                vectors=tuple(tuple(vector) for vector in vectors),
                embedding_space=embedding_space,
                llm_semaphore_locked=self.llm_semaphore.locked(),
                embedding_semaphore_locked=self.embedding_semaphore.locked(),
            )
        )

    async def record_failure(
        self,
        pending: PendingFactMessage,
        error: str,
        *,
        terminal: bool,
    ) -> None:
        self.failures.append(
            FailureCall(pending=pending, error=error, terminal=terminal)
        )


class FakeExtractor:
    def __init__(
        self,
        result: FactExtractionResult,
        llm_semaphore: asyncio.Semaphore,
        embedding_semaphore: asyncio.Semaphore,
    ) -> None:
        self.result = result
        self.llm_semaphore = llm_semaphore
        self.embedding_semaphore = embedding_semaphore
        self.calls: list[ExtractionCall] = []
        self.failures: dict[UUID, BaseException] = {}
        self.message_failures: dict[UUID, BaseException] = {}
        self.blocked_messages: set[UUID] = set()
        self.block_event = asyncio.Event()

    async def extract(
        self,
        message: Message,
        content: str,
        context: FactExtractionContext | None = None,
    ) -> FactExtractionResult:
        self.calls.append(
            ExtractionCall(
                message=message,
                content=content,
                context=context,
                llm_semaphore_locked=self.llm_semaphore.locked(),
                embedding_semaphore_locked=self.embedding_semaphore.locked(),
            )
        )
        if message.id in self.blocked_messages:
            await self.block_event.wait()
        failure = self.message_failures.get(message.id) or self.failures.get(
            message.fish_id
        )
        if failure is not None:
            raise failure
        return self.result


@dataclass(frozen=True, slots=True)
class EmbeddingCall:
    texts: tuple[str, ...]
    llm_semaphore_locked: bool
    embedding_semaphore_locked: bool


class FakeEmbedder:
    def __init__(
        self,
        llm_semaphore: asyncio.Semaphore,
        embedding_semaphore: asyncio.Semaphore,
    ) -> None:
        self.llm_semaphore = llm_semaphore
        self.embedding_semaphore = embedding_semaphore
        self.calls: list[EmbeddingCall] = []
        self.failure: BaseException | None = None

    @property
    def space(self) -> EmbeddingSpace:
        return EMBEDDING_SPACE

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(
            EmbeddingCall(
                texts=tuple(texts),
                llm_semaphore_locked=self.llm_semaphore.locked(),
                embedding_semaphore_locked=self.embedding_semaphore.locked(),
            )
        )
        if self.failure is not None:
            raise self.failure
        return EmbeddingResult(
            vectors=tuple((float(index + 1),) * 1024 for index in range(len(texts))),
            input_tokens=len(texts) * 3,
        )


class FakePerformanceStore:
    def __init__(self) -> None:
        self.calls: list[PerformanceCall] = []
        self.failure: BaseException | None = None

    async def insert_steps(
        self,
        message_id: UUID,
        steps: Sequence[PerformanceStep],
    ) -> None:
        self.calls.append(PerformanceCall(message_id=message_id, steps=tuple(steps)))
        if self.failure is not None:
            raise self.failure


class FakeEmbeddingSpaceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, EmbeddingSpace]] = []
        self.failure: BaseException | None = None

    async def ensure_compatible(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
    ) -> None:
        self.calls.append((fish_id, space))
        if self.failure is not None:
            raise self.failure


class IncrementingClock:
    def __init__(self, value: int = 10_000_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        value = self.value
        self.value += 100_000
        return value


class MutableNow:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def make_pending(
    *,
    fish_id: UUID = FISH_ONE,
    session_id: UUID = SESSION_ONE,
    message_id: UUID = MESSAGE_ONE,
    sequence: int = 4,
    role: Role = Role.USER,
    content: str = "完整用户内容",
    status: MessageStatus = MessageStatus.INITIALIZED,
    consumed_offset: int | None = None,
    previous_opposite_message: Message | None = None,
) -> PendingFactMessage:
    return PendingFactMessage(
        message=Message(
            id=message_id,
            fish_id=fish_id,
            session_id=session_id,
            sequence=sequence,
            role=role,
            content=content,
            status=status,
            created_at=NOW - timedelta(seconds=2),
            consumed_offset=consumed_offset,
        ),
        target_sequence=sequence,
        previous_opposite_message=previous_opposite_message,
    )


def make_extraction() -> FactExtractionResult:
    return FactExtractionResult(
        facts=(
            ExtractedFact(
                text="该用户每天打篮球",
                occurred_start=None,
                occurred_end=None,
                entities=("篮球",),
                what="每天打篮球",
                who=("该用户",),
            ),
        ),
        request_payload={
            "model": "qwen-plus",
            "messages": [{"role": "user", "content": "完整 prompt"}],
        },
        response_payload={
            "content": '{"facts":[{"text":"该用户每天打篮球"}]}',
            "input_tokens": 21,
            "output_tokens": 9,
        },
    )


def make_processor(
    candidates: Sequence[PendingFactMessage],
    *,
    now: Callable[[], datetime] = lambda: NOW,
    retry_now: Callable[[], datetime] | None = None,
    max_retries: int = 3,
    batch_size: int = 8,
    fact_llm_timeout_seconds: float = 120,
) -> ProcessorHarness:
    llm_semaphore = asyncio.Semaphore(1)
    embedding_semaphore = asyncio.Semaphore(1)
    store = FakeFactStore(candidates, llm_semaphore, embedding_semaphore)
    extractor = FakeExtractor(make_extraction(), llm_semaphore, embedding_semaphore)
    embedder = FakeEmbedder(llm_semaphore, embedding_semaphore)
    performance = FakePerformanceStore()
    space_store = FakeEmbeddingSpaceStore()
    processor = FactProcessor(
        store,
        extractor,
        embedder,
        performance,
        embedding_space_store=space_store,
        llm_semaphore=llm_semaphore,
        embedding_semaphore=embedding_semaphore,
        max_retries=max_retries,
        batch_size=batch_size,
        llm_timeout_seconds=fact_llm_timeout_seconds,
        clock=IncrementingClock(),
        now=now,
        retry_now=now if retry_now is None else retry_now,
    )
    return ProcessorHarness(
        processor=processor,
        store=store,
        extractor=extractor,
        embedder=embedder,
        performance=performance,
        space_store=space_store,
        llm_semaphore=llm_semaphore,
        embedding_semaphore=embedding_semaphore,
    )


@pytest.mark.asyncio
async def test_records_failure_when_fact_extraction_exceeds_timeout() -> None:
    pending = make_pending()
    harness = make_processor([pending], fact_llm_timeout_seconds=0.001)
    harness.extractor.blocked_messages.add(pending.message.id)

    processed = await harness.processor.process_once()

    assert processed is True
    assert harness.store.persisted == []
    assert len(harness.store.failures) == 1
    assert harness.store.failures[0].error == "Fact LLM request exceeded 0.001 seconds"


@pytest.mark.asyncio
async def test_processes_batch_for_one_fish_in_sequence_order() -> None:
    first = make_pending(content="用户原始内容")
    second = make_pending(
        message_id=MESSAGE_TWO,
        sequence=5,
        content="第二条消息",
    )
    harness = make_processor([first, second], batch_size=2)

    processed = await harness.processor.process_once()

    assert processed is True
    assert harness.store.requested_limits == [(100, 2)]
    assert [call.message for call in harness.extractor.calls] == [
        first.message,
        second.message,
    ]
    assert [call.content for call in harness.extractor.calls] == [
        "用户原始内容",
        "第二条消息",
    ]
    assert [call.context for call in harness.extractor.calls] == [None, None]
    assert [call.pending for call in harness.store.persisted] == [first, second]
    assert [call.message_id for call in harness.performance.calls] == [
        first.message.id,
        second.message.id,
    ]


@pytest.mark.asyncio
async def test_assistant_extraction_uses_only_visible_content() -> None:
    pending = make_pending(
        role=Role.ASSISTANT,
        status=MessageStatus.INTERRUPTED,
        content="说到这里，未消费",
        consumed_offset=4,
    )
    harness = make_processor([pending])

    processed = await harness.processor.process_once()

    assert processed is True
    assert [call.content for call in harness.extractor.calls] == ["说到这里"]


@pytest.mark.asyncio
async def test_passes_previous_opposite_role_message_as_extraction_context() -> None:
    previous = make_pending(
        role=Role.ASSISTANT,
        content="Target 是刚才提到的地点",
        sequence=3,
    ).message
    pending = make_pending(
        role=Role.USER,
        content="我实际上兑换了一张优惠券",
        sequence=4,
        previous_opposite_message=previous,
    )
    harness = make_processor([pending])

    await harness.processor.process_once()

    context = harness.extractor.calls[0].context
    assert context is not None
    assert context.role is Role.ASSISTANT
    assert context.content == "Target 是刚才提到的地点"


@pytest.mark.asyncio
async def test_assistant_extraction_uses_previous_user_message_as_context() -> None:
    previous = make_pending(
        role=Role.USER,
        content="你刚才在哪里兑换了优惠券？",
        sequence=3,
    ).message
    pending = make_pending(
        role=Role.ASSISTANT,
        content="在 Target。",
        sequence=4,
        previous_opposite_message=previous,
    )
    harness = make_processor([pending])

    await harness.processor.process_once()

    context = harness.extractor.calls[0].context
    assert context is not None
    assert context.role is Role.USER
    assert context.content == "你刚才在哪里兑换了优惠券？"


@pytest.mark.asyncio
async def test_empty_visible_content_advances_without_extraction() -> None:
    pending = make_pending(
        role=Role.ASSISTANT,
        status=MessageStatus.INTERRUPTED,
        content="未消费",
        consumed_offset=0,
    )
    harness = make_processor([pending])

    processed = await harness.processor.process_once()

    assert processed is True
    assert harness.extractor.calls == []
    assert harness.embedder.calls == []
    assert harness.store.persisted[0].facts == ()
    assert [step.name for step in harness.performance.calls[0].steps] == [
        "fact.queue_wait",
        "fact.persist",
        "fact.total",
    ]


@pytest.mark.asyncio
async def test_separates_llm_and_embedding_semaphores_from_persistence() -> None:
    harness = make_processor([make_pending()])

    await harness.processor.process_once()

    assert harness.extractor.calls[0].llm_semaphore_locked is True
    assert harness.extractor.calls[0].embedding_semaphore_locked is False
    assert harness.embedder.calls == [
        EmbeddingCall(
            texts=("每天打篮球",),
            llm_semaphore_locked=False,
            embedding_semaphore_locked=True,
        )
    ]
    assert harness.store.persisted[0].llm_semaphore_locked is False
    assert harness.store.persisted[0].embedding_semaphore_locked is False
    assert harness.llm_semaphore.locked() is False
    assert harness.embedding_semaphore.locked() is False


@pytest.mark.asyncio
async def test_embedding_failure_does_not_persist_or_record_terminal_error() -> None:
    harness = make_processor([make_pending()])
    harness.embedder.failure = RuntimeError("embedding unavailable")

    assert await harness.processor.process_once() is True

    assert harness.store.persisted == []
    assert harness.store.failures == [
        FailureCall(
            pending=make_pending(),
            error="embedding unavailable",
            terminal=False,
        )
    ]
    assert harness.performance.calls == []


@pytest.mark.asyncio
async def test_persists_vectors_and_records_embedding_performance() -> None:
    harness = make_processor([make_pending()])

    await harness.processor.process_once()

    persisted = harness.store.persisted[0]
    assert persisted.embedding_space == EMBEDDING_SPACE
    assert len(persisted.vectors) == 1
    assert persisted.vectors[0] == (1.0,) * 1024
    by_name = {step.name: step for step in harness.performance.calls[0].steps}
    assert by_name["embedding.fact"].input_payload == {
        "provider": "openai-compatible",
        "model": "text-embedding-v4",
        "dimensions": 1024,
        "texts": ["每天打篮球"],
    }
    assert by_name["embedding.fact"].output_payload == {
        "count": 1,
        "dimensions": 1024,
        "input_tokens": 3,
    }


@pytest.mark.asyncio
async def test_embedding_space_mismatch_is_terminal_without_calling_models() -> None:
    pending = make_pending()
    harness = make_processor([pending])
    harness.space_store.failure = EmbeddingSpaceMismatchError(
        FISH_ONE,
        EmbeddingSpace("openai-compatible", "text-embedding-v4", 1024),
        EmbeddingSpace("local", "intfloat/multilingual-e5-small", 384),
    )

    assert await harness.processor.process_once() is True

    assert harness.extractor.calls == []
    assert harness.embedder.calls == []
    assert harness.store.failures[0].terminal is True
    assert "1024" in harness.store.failures[0].error
    assert "384" in harness.store.failures[0].error


@pytest.mark.asyncio
async def test_skips_generating_assistant_and_processes_other_fish() -> None:
    generating = make_pending(
        role=Role.ASSISTANT,
        status=MessageStatus.GENERATING,
    )
    eligible = make_pending(
        fish_id=FISH_TWO,
        message_id=MESSAGE_TWO,
        content="另一条可处理消息",
    )
    harness = make_processor([generating, eligible])

    processed = await harness.processor.process_once()

    assert processed is True
    assert [call.message for call in harness.extractor.calls] == [eligible.message]
    assert [call.pending for call in harness.store.persisted] == [eligible]
    assert harness.store.failures == []

    harness.store.candidates = [generating]
    assert await harness.processor.process_once() is False
    assert [call.pending for call in harness.store.persisted] == [eligible]


@pytest.mark.asyncio
async def test_blocked_session_does_not_block_another_session_of_same_fish() -> None:
    generating = make_pending(
        role=Role.ASSISTANT,
        status=MessageStatus.GENERATING,
    )
    eligible = make_pending(
        session_id=SESSION_TWO,
        message_id=MESSAGE_TWO,
        content="另一个会话的稳定消息",
    )
    harness = make_processor([generating, eligible])

    assert await harness.processor.process_once() is True
    assert [call.pending for call in harness.store.persisted] == [eligible]


@pytest.mark.asyncio
async def test_extraction_failure_backs_off_only_that_fish() -> None:
    failed = make_pending()
    eligible = make_pending(
        fish_id=FISH_TWO,
        message_id=MESSAGE_TWO,
        content="另一条可处理消息",
    )
    harness = make_processor([failed, eligible])
    harness.extractor.failures[FISH_ONE] = RuntimeError("LLM unavailable")

    assert await harness.processor.process_once() is True

    assert harness.store.persisted == []
    assert harness.store.failures == [
        FailureCall(
            pending=failed,
            error="LLM unavailable",
            terminal=False,
        )
    ]
    assert harness.performance.calls == []

    assert await harness.processor.process_once() is True
    assert [call.message for call in harness.extractor.calls] == [
        failed.message,
        eligible.message,
    ]
    assert [call.pending for call in harness.store.persisted] == [eligible]


@pytest.mark.asyncio
async def test_persistence_failure_backs_off_only_that_fish() -> None:
    failed = make_pending()
    eligible = make_pending(
        fish_id=FISH_TWO,
        message_id=MESSAGE_TWO,
        content="另一条可处理消息",
    )
    harness = make_processor([failed, eligible])
    harness.store.persist_failures[FISH_ONE] = RuntimeError("database unavailable")

    assert await harness.processor.process_once() is True

    assert harness.store.persisted == []
    assert harness.store.failures == [
        FailureCall(
            pending=failed,
            error="database unavailable",
            terminal=False,
        )
    ]
    assert harness.performance.calls == []

    assert await harness.processor.process_once() is True
    assert [call.message for call in harness.extractor.calls] == [
        failed.message,
        eligible.message,
    ]
    assert [call.pending for call in harness.store.persisted] == [eligible]


@pytest.mark.asyncio
async def test_records_fact_performance_timeline_and_full_llm_payloads() -> None:
    pending = make_pending()
    harness = make_processor([pending])

    processed = await harness.processor.process_once()

    assert processed is True
    assert harness.performance.calls[0].message_id == pending.message.id
    steps = harness.performance.calls[0].steps
    assert [step.name for step in steps] == [
        "fact.queue_wait",
        "llm.fact_extraction",
        "embedding.fact",
        "fact.persist",
        "fact.total",
    ]
    assert steps[0].started_offset_us == 0
    assert steps[0].duration_us == 2_000_000
    assert steps[1].input_payload == make_extraction().request_payload
    assert steps[1].output_payload == make_extraction().response_payload


@pytest.mark.asyncio
async def test_performance_failure_does_not_change_successful_fact_result() -> None:
    pending = make_pending()
    harness = make_processor([pending])
    harness.performance.failure = RuntimeError("metrics unavailable")

    processed = await harness.processor.process_once()

    assert processed is True
    assert [call.pending for call in harness.store.persisted] == [pending]
    assert harness.store.failures == []
    assert len(harness.performance.calls) == 1


@pytest.mark.asyncio
async def test_terminal_fact_failure_advances_to_later_message() -> None:
    now = MutableNow()
    failed = make_pending()
    later = make_pending(
        message_id=MESSAGE_TWO,
        sequence=5,
        content="后续消息仍应处理",
    )
    harness = make_processor(
        [failed, later],
        now=now,
        max_retries=1,
        batch_size=2,
    )
    harness.extractor.message_failures[MESSAGE_ONE] = RuntimeError("retry")

    assert await harness.processor.process_once() is True
    assert harness.store.failures == [
        FailureCall(pending=failed, error="retry", terminal=False)
    ]
    assert await harness.processor.process_once() is False
    now.advance(1)
    assert await harness.processor.process_once() is True
    assert harness.store.failures == [
        FailureCall(pending=failed, error="retry", terminal=False),
        FailureCall(pending=failed, error="retry", terminal=True),
    ]
    assert [call.pending for call in harness.store.persisted] == [later]


@pytest.mark.asyncio
async def test_success_clears_retry_state() -> None:
    now = MutableNow()
    harness = make_processor([make_pending()], now=now)
    harness.extractor.failures[FISH_ONE] = RuntimeError("first failure")

    assert await harness.processor.process_once() is True
    now.advance(1)
    harness.extractor.failures.clear()
    assert await harness.processor.process_once() is True

    harness.extractor.failures[FISH_ONE] = RuntimeError("failure after success")
    assert await harness.processor.process_once() is True
    assert await harness.processor.process_once() is False
    now.advance(1)
    assert await harness.processor.process_once() is True


@pytest.mark.asyncio
async def test_retries_when_business_clock_is_frozen() -> None:
    retry_now = MutableNow()
    harness = make_processor(
        [make_pending()],
        now=lambda: NOW,
        retry_now=retry_now,
        max_retries=1,
    )
    harness.extractor.failures[FISH_ONE] = RuntimeError("invalid response")

    assert await harness.processor.process_once() is True
    assert await harness.processor.process_once() is False

    retry_now.advance(1)

    assert await harness.processor.process_once() is True
    assert len(harness.extractor.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["extract", "persist"])
async def test_cancelled_error_is_not_swallowed(failure_stage: str) -> None:
    harness = make_processor([make_pending()])
    if failure_stage == "extract":
        harness.extractor.failures[FISH_ONE] = asyncio.CancelledError()
    else:
        harness.store.persist_failures[FISH_ONE] = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await harness.processor.process_once()

    assert harness.store.failures == []
    assert harness.performance.calls == []
