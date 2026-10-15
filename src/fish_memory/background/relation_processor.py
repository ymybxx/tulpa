import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.relations.models import (
    PendingSnapshotItemProjectionBatch,
    SnapshotItemEmbeddingProjection,
    SnapshotItemEmbeddingState,
)
from fish_memory.snapshots.models import SnapshotItemState

logger = logging.getLogger(__name__)

_CANDIDATE_LIMIT = 100


class SnapshotItemRelationStoreProtocol(Protocol):
    async def list_pending_batches(
        self,
        limit: int,
        dimensions: int,
    ) -> list[PendingSnapshotItemProjectionBatch]: ...

    async def persist_batch(
        self,
        batch: PendingSnapshotItemProjectionBatch,
        projections: Sequence[SnapshotItemEmbeddingProjection],
        embedding_space: EmbeddingSpace,
    ) -> bool: ...

    async def record_error(
        self,
        batch: PendingSnapshotItemProjectionBatch,
        error: str,
    ) -> None: ...


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


class SnapshotItemRelationProcessor:
    def __init__(
        self,
        store: SnapshotItemRelationStoreProtocol,
        embedder: EmbeddingClientProtocol,
        *,
        embedding_space_store: EmbeddingSpaceStoreProtocol,
        embedding_semaphore: asyncio.Semaphore,
        snapshot_type: str,
        max_retries: int,
        batch_size: int,
        retry_now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if snapshot_type not in {"dynamic", "static"}:
            raise ValueError("invalid relation snapshot type")
        if max_retries < 1:
            raise ValueError("max retries must be at least 1")
        if batch_size < 1:
            raise ValueError("batch size must be at least 1")
        self._store = store
        self._embedder = embedder
        self._embedding_space_store = embedding_space_store
        self._embedding_semaphore = embedding_semaphore
        self._embedding_space = embedder.space
        self._snapshot_type = snapshot_type
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._retry_now = retry_now
        self._retry_states: dict[int, _RetryState] = {}

    async def process_once(self) -> bool:
        candidates = await self._store.list_pending_batches(
            _CANDIDATE_LIMIT,
            self._embedding_space.dimensions,
        )
        for batch in candidates:
            retry_state = self._retry_states.get(batch.snapshot_id)
            if retry_state is not None and (
                retry_state.retry_at is None or self._retry_now() < retry_state.retry_at
            ):
                continue
            await self._process(batch)
            return True
        return False

    async def _process(self, batch: PendingSnapshotItemProjectionBatch) -> None:
        try:
            if batch.snapshot_type != self._snapshot_type:
                raise ValueError(
                    "relation batch snapshot type does not match processor"
                )
            await self._embedding_space_store.ensure_compatible(
                batch.fish_id,
                self._embedding_space,
            )
            projections = await self._embedding_projections(batch)
            persisted = await self._store.persist_batch(
                batch,
                projections,
                self._embedding_space,
            )
            if not persisted:
                return
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            if self._record_failure(batch.snapshot_id):
                await self._store.record_error(batch, str(exception))
            return
        self._retry_states.pop(batch.snapshot_id, None)

    async def _embedding_projections(
        self,
        batch: PendingSnapshotItemProjectionBatch,
    ) -> tuple[SnapshotItemEmbeddingProjection, ...]:
        state_by_item_id = {state.item_id: state for state in batch.embedding_states}
        pending: list[
            tuple[SnapshotItemState, str, SnapshotItemEmbeddingState | None]
        ] = []
        for item in batch.items:
            summary = item.summary.strip()
            summary_hash = hashlib.sha256(summary.encode()).hexdigest()
            state = state_by_item_id.get(item.id)
            if state is None or not _projection_is_current(
                item,
                state,
                summary_hash,
                self._embedding_space.dimensions,
            ):
                pending.append((item, summary_hash, state))
        selected = pending[: self._batch_size]

        changed_indexes: list[int] = []
        changed_texts: list[str] = []
        for index, (item, summary_hash, state) in enumerate(selected):
            if state is None or not _vector_is_reusable(
                state,
                summary_hash,
                self._embedding_space.dimensions,
            ):
                changed_indexes.append(index)
                changed_texts.append(item.summary.strip())

        vectors_by_index: dict[int, tuple[float, ...]] = {}
        if changed_texts:
            async with self._embedding_semaphore:
                embedded = await self._embedder.embed(changed_texts)
            if len(embedded.vectors) != len(changed_texts):
                raise ValueError("relation embedding count does not match input count")
            vectors_by_index = dict(zip(changed_indexes, embedded.vectors, strict=True))

        return tuple(
            SnapshotItemEmbeddingProjection(
                item_id=item.id,
                item_version=item.item_version,
                session_id=batch.session_id,
                snapshot_type=batch.snapshot_type,
                snapshot_text=item.summary.strip(),
                text_hash=summary_hash,
                last_supported_at=item.last_supported_at,
                vector=vectors_by_index.get(index),
            )
            for index, (item, summary_hash, _state) in enumerate(selected)
        )

    def _record_failure(self, snapshot_id: int) -> bool:
        previous = self._retry_states.get(snapshot_id)
        failure_count = 1 if previous is None else previous.failure_count + 1
        if failure_count > self._max_retries:
            self._retry_states[snapshot_id] = _RetryState(
                failure_count=failure_count,
                retry_at=None,
            )
            return True
        delay_seconds = 2 ** min(
            failure_count - 1,
            self._max_retries - 1,
        )
        self._retry_states[snapshot_id] = _RetryState(
            failure_count=failure_count,
            retry_at=self._retry_now() + timedelta(seconds=delay_seconds),
        )
        return False


@dataclass(frozen=True, slots=True)
class _RetryState:
    failure_count: int
    retry_at: datetime | None


def _projection_is_current(
    item: SnapshotItemState,
    state: SnapshotItemEmbeddingState,
    summary_hash: str,
    dimensions: int,
) -> bool:
    return (
        state.item_version == item.item_version
        and state.text_hash == summary_hash
        and state.dimensions == dimensions
    )


def _vector_is_reusable(
    state: SnapshotItemEmbeddingState,
    summary_hash: str,
    dimensions: int,
) -> bool:
    return state.text_hash == summary_hash and state.dimensions == dimensions
