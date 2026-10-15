import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast
from uuid import UUID

from fish_memory.db.embedding_spaces import EmbeddingSpaceMismatchError
from fish_memory.embeddings.client import (
    EmbeddingResult,
    EmbeddingSpace,
)
from fish_memory.facts.extractor import FactExtractionResult
from fish_memory.facts.models import (
    ExtractedFact,
    FactExtractionContext,
    PendingFactMessage,
)
from fish_memory.models import Message, MessageStatus, Role, visible_message_content
from fish_memory.performance.models import PerformanceStep
from fish_memory.performance.tracker import MessagePerformanceTracker

logger = logging.getLogger(__name__)

# Each candidate represents one fish, so a bounded batch can scan past fish
# temporarily blocked by generation or retry backoff.
_CANDIDATE_LIMIT = 100


class FactStoreProtocol(Protocol):
    async def list_pending_messages(
        self,
        limit: int,
        batch_size: int,
    ) -> list[PendingFactMessage]: ...

    async def persist_extraction(
        self,
        pending: PendingFactMessage,
        facts: Sequence[ExtractedFact],
        vectors: Sequence[Sequence[float]],
        embedding_space: EmbeddingSpace,
    ) -> None: ...

    async def record_failure(
        self,
        pending: PendingFactMessage,
        error: str,
        *,
        terminal: bool,
    ) -> None: ...


class FactExtractorProtocol(Protocol):
    async def extract(
        self,
        message: Message,
        content: str,
        context: FactExtractionContext | None = None,
    ) -> FactExtractionResult: ...


class EmbeddingClientProtocol(Protocol):
    @property
    def space(self) -> EmbeddingSpace: ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


class EmbeddingSpaceStoreProtocol(Protocol):
    async def ensure_compatible(
        self,
        fish_id: UUID,
        requested: EmbeddingSpace,
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


@dataclass(frozen=True, slots=True)
class _PreparedExtraction:
    pending: PendingFactMessage
    facts: Sequence[ExtractedFact]
    vectors: Sequence[Sequence[float]]
    tracker: MessagePerformanceTracker
    origin_ns: int


class FactProcessor:
    def __init__(
        self,
        store: FactStoreProtocol,
        extractor: FactExtractorProtocol,
        embedder: EmbeddingClientProtocol,
        performance_store: PerformanceStoreProtocol,
        *,
        embedding_space_store: EmbeddingSpaceStoreProtocol,
        llm_semaphore: asyncio.Semaphore,
        embedding_semaphore: asyncio.Semaphore,
        max_retries: int,
        batch_size: int,
        llm_timeout_seconds: float = 120,
        clock: Callable[[], int] = time.perf_counter_ns,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        retry_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be positive")
        self._store = store
        self._extractor = extractor
        self._embedder = embedder
        self._performance_store = performance_store
        self._embedding_space_store = embedding_space_store
        self._embedding_space = embedder.space
        self._llm_semaphore = llm_semaphore
        self._embedding_semaphore = embedding_semaphore
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._llm_timeout_seconds = llm_timeout_seconds
        self._clock = clock
        self._now = now
        self._retry_now = retry_now
        self._retry_states: dict[UUID, _RetryState] = {}

    async def process_once(self) -> bool:
        candidates = await self._store.list_pending_messages(
            limit=_CANDIDATE_LIMIT,
            batch_size=self._batch_size,
        )
        for batch in _group_pending_by_session(candidates):
            first_message = batch[0].message
            retry_state = self._retry_states.get(first_message.id)
            if retry_state is not None and (
                retry_state.retry_at is None or self._retry_now() < retry_state.retry_at
            ):
                continue
            eligible_batch = _eligible_prefix(batch)
            if not eligible_batch:
                continue

            await self._process_batch(eligible_batch)
            return True
        return False

    async def _process_batch(self, batch: Sequence[PendingFactMessage]) -> None:
        try:
            await self._embedding_space_store.ensure_compatible(
                batch[0].message.fish_id,
                self._embedding_space,
            )
        except asyncio.CancelledError:
            raise
        except EmbeddingSpaceMismatchError as exception:
            for pending in batch:
                await self._store.record_failure(
                    pending,
                    str(exception),
                    terminal=True,
                )
            return
        except Exception as exception:
            await self._record_failure(batch[0], str(exception))
            return

        tasks = [
            asyncio.create_task(self._prepare_extraction(pending)) for pending in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Persist in message order so the fact_extraction watermark never
        # advances past a failed earlier message in the same fish batch.
        for pending, result in zip(batch, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                if await self._record_failure(pending, str(result)):
                    continue
                return
            prepared = cast(_PreparedExtraction, result)
            try:
                await self._persist_prepared(prepared)
            except asyncio.CancelledError:
                raise
            except Exception as exception:
                if await self._record_failure(pending, str(exception)):
                    continue
                return

    async def _prepare_extraction(
        self,
        pending: PendingFactMessage,
    ) -> _PreparedExtraction:
        message = pending.message
        worker_started_ns = self._clock()
        queue_wait_us = max(
            0,
            int((self._now() - message.created_at).total_seconds() * 1_000_000),
        )
        origin_ns = worker_started_ns - queue_wait_us * 1_000
        tracker = MessagePerformanceTracker(
            clock=self._clock,
            origin_ns=origin_ns,
        )
        tracker.record(
            "fact.queue_wait",
            started_ns=origin_ns,
            finished_ns=worker_started_ns,
        )

        try:
            content = visible_message_content(message)
            facts: Sequence[ExtractedFact] = ()
            vectors: Sequence[Sequence[float]] = ()
            if content:
                context = _extraction_context(pending)
                async with self._llm_semaphore:
                    extraction_started_ns = tracker.now_ns()
                    try:
                        async with asyncio.timeout(self._llm_timeout_seconds):
                            extraction = await self._extractor.extract(
                                message,
                                content,
                                context,
                            )
                    except TimeoutError as exception:
                        raise TimeoutError(
                            "Fact LLM request exceeded "
                            f"{self._llm_timeout_seconds:g} seconds"
                        ) from exception
                    extraction_finished_ns = tracker.now_ns()
                facts = extraction.facts
                embedding_texts = [fact.embedding_text for fact in facts]
                async with self._embedding_semaphore:
                    embedding_started_ns = tracker.now_ns()
                    embedding = await self._embedder.embed(embedding_texts)
                    embedding_finished_ns = tracker.now_ns()
                tracker.record(
                    "llm.fact_extraction",
                    started_ns=extraction_started_ns,
                    finished_ns=extraction_finished_ns,
                    input_payload=extraction.request_payload,
                    output_payload=extraction.response_payload,
                )
                tracker.record(
                    "embedding.fact",
                    started_ns=embedding_started_ns,
                    finished_ns=embedding_finished_ns,
                    input_payload={
                        "provider": self._embedding_space.provider,
                        "model": self._embedding_space.model,
                        "dimensions": self._embedding_space.dimensions,
                        "texts": embedding_texts,
                    },
                    output_payload={
                        "count": len(embedding.vectors),
                        "dimensions": (
                            len(embedding.vectors[0]) if embedding.vectors else 0
                        ),
                        "input_tokens": embedding.input_tokens,
                    },
                )
                vectors = embedding.vectors

            return _PreparedExtraction(
                pending=pending,
                facts=facts,
                vectors=vectors,
                tracker=tracker,
                origin_ns=origin_ns,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise

    async def _persist_prepared(self, prepared: _PreparedExtraction) -> None:
        message = prepared.pending.message
        tracker = prepared.tracker
        persist_started_ns = tracker.now_ns()
        await self._store.persist_extraction(
            prepared.pending,
            prepared.facts,
            prepared.vectors,
            self._embedding_space,
        )
        persist_finished_ns = tracker.now_ns()
        tracker.record(
            "fact.persist",
            started_ns=persist_started_ns,
            finished_ns=persist_finished_ns,
        )

        self._retry_states.pop(message.id, None)
        tracker.record(
            "fact.total",
            started_ns=prepared.origin_ns,
            finished_ns=tracker.now_ns(),
        )
        try:
            await self._performance_store.insert_steps(
                message.id,
                tracker.snapshot(),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Failed to persist Fact performance steps for message %s",
                message.id,
            )

    async def _record_failure(
        self,
        pending: PendingFactMessage,
        error: str,
    ) -> bool:
        message = pending.message
        previous = self._retry_states.get(message.id)
        failure_count = (
            max(
                pending.fact_extraction_attempt_count,
                0 if previous is None else previous.failure_count,
            )
            + 1
        )
        terminal = failure_count > self._max_retries
        await self._store.record_failure(
            pending,
            error,
            terminal=terminal,
        )
        if terminal:
            self._retry_states.pop(message.id, None)
            return True

        # Keep the task watermark behind a retriable message. The message-level
        # failure is already persisted for observability, but downstream tasks
        # must wait until this message succeeds or reaches the retry ceiling.
        delay_seconds = 2 ** min(
            failure_count - 1,
            self._max_retries - 1,
        )
        self._retry_states[message.id] = _RetryState(
            failure_count=failure_count,
            retry_at=self._retry_now() + timedelta(seconds=delay_seconds),
        )
        return False


def _group_pending_by_session(
    candidates: Sequence[PendingFactMessage],
) -> list[list[PendingFactMessage]]:
    groups: list[list[PendingFactMessage]] = []
    groups_by_session: dict[tuple[UUID, UUID], list[PendingFactMessage]] = {}
    for pending in candidates:
        key = (pending.message.fish_id, pending.message.session_id)
        group = groups_by_session.get(key)
        if group is None:
            group = []
            groups_by_session[key] = group
            groups.append(group)
        group.append(pending)
    return groups


def _eligible_prefix(
    batch: Sequence[PendingFactMessage],
) -> list[PendingFactMessage]:
    eligible: list[PendingFactMessage] = []
    for pending in batch:
        message = pending.message
        if (
            message.role is Role.ASSISTANT
            and message.status is MessageStatus.GENERATING
        ):
            break
        eligible.append(pending)
    return eligible


def _extraction_context(
    pending: PendingFactMessage,
) -> FactExtractionContext | None:
    if pending.external_event is not None:
        # An external audience event is not the previous local user's utterance.
        return FactExtractionContext(
            role=pending.message.role, content="", external_event=pending.external_event
        )
    previous = pending.previous_opposite_message
    if previous is None or previous.role is pending.message.role:
        return None
    content = visible_message_content(previous)
    if not content:
        return None
    return FactExtractionContext(role=previous.role, content=content)
