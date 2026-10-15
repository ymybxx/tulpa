import asyncio
import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fish_memory.background import relation_processor as processor_module
from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.relations.models import (
    PendingSnapshotItemProjectionBatch,
    SnapshotItemEmbeddingProjection,
    SnapshotItemEmbeddingState,
)
from fish_memory.snapshots.models import SnapshotItemState

FISH_ID = UUID(int=1)
SESSION_ID = UUID(int=2)
NOW = datetime(2026, 8, 27, tzinfo=UTC)
SPACE = EmbeddingSpace("local", "test-model", 3)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def item(
    item_id: int,
    summary: str,
    *,
    item_version: int = 1,
) -> SnapshotItemState:
    fact_id = UUID(int=100 + item_id)
    return SnapshotItemState(
        id=item_id,
        summary=summary,
        support_fact_ids=(fact_id,),
        lineage_fact_ids=(fact_id,),
        last_supported_at=NOW,
        item_version=item_version,
    )


def batch(
    *items: SnapshotItemState,
    states: tuple[SnapshotItemEmbeddingState, ...] = (),
    snapshot_id: int = 11,
) -> PendingSnapshotItemProjectionBatch:
    return PendingSnapshotItemProjectionBatch(
        fish_id=FISH_ID,
        snapshot_id=snapshot_id,
        session_id=SESSION_ID,
        snapshot_type="dynamic",
        snapshot_version=4,
        items=items,
        embedding_states=states,
    )


class FakeEmbedder:
    space = SPACE

    def __init__(self, *, fail: bool = False, missing_vector: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._fail = fail
        self._missing_vector = missing_vector

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        if self._fail:
            raise RuntimeError("embedding unavailable")
        vectors = tuple((float(index + 1), 0.0, 0.0) for index in range(len(texts)))
        if self._missing_vector:
            vectors = vectors[:-1]
        return EmbeddingResult(vectors=vectors, input_tokens=len(texts))


class FakeEmbeddingSpaceStore:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, EmbeddingSpace]] = []

    async def ensure_compatible(
        self,
        fish_id: UUID,
        requested: EmbeddingSpace,
    ) -> None:
        self.calls.append((fish_id, requested))


class FakeStore:
    def __init__(
        self,
        batches: Sequence[PendingSnapshotItemProjectionBatch],
    ) -> None:
        self.batches = list(batches)
        self.list_calls: list[tuple[int, int]] = []
        self.persisted: list[
            tuple[
                PendingSnapshotItemProjectionBatch,
                tuple[SnapshotItemEmbeddingProjection, ...],
                EmbeddingSpace,
            ]
        ] = []
        self.errors: list[tuple[PendingSnapshotItemProjectionBatch, str]] = []

    async def list_pending_batches(
        self,
        limit: int,
        dimensions: int,
    ) -> list[PendingSnapshotItemProjectionBatch]:
        self.list_calls.append((limit, dimensions))
        return list(self.batches)

    async def persist_batch(
        self,
        pending: PendingSnapshotItemProjectionBatch,
        projections: Sequence[SnapshotItemEmbeddingProjection],
        embedding_space: EmbeddingSpace,
    ) -> bool:
        self.persisted.append((pending, tuple(projections), embedding_space))
        return True

    async def record_error(
        self,
        pending: PendingSnapshotItemProjectionBatch,
        error: str,
    ) -> None:
        self.errors.append((pending, error))


def processor(
    store: FakeStore,
    embedder: FakeEmbedder,
    *,
    batch_size: int = 20,
    max_retries: int = 3,
    retry_now: Callable[[], datetime] = lambda: NOW,
) -> object:
    return processor_module.SnapshotItemRelationProcessor(
        store,
        embedder,
        embedding_space_store=FakeEmbeddingSpaceStore(),
        embedding_semaphore=asyncio.Semaphore(1),
        snapshot_type="dynamic",
        max_retries=max_retries,
        batch_size=batch_size,
        retry_now=retry_now,
    )


@pytest.mark.asyncio
async def test_processes_at_most_batch_size_stale_items() -> None:
    pending = batch(item(7, "第一项"), item(8, "第二项"))
    store = FakeStore((pending,))
    embedder = FakeEmbedder()
    worker = processor(store, embedder, batch_size=1)

    assert await worker.process_once() is True

    assert store.list_calls == [(100, 3)]
    assert embedder.calls == [("第一项",)]
    projections = store.persisted[0][1]
    assert [projection.item_id for projection in projections] == [7]
    assert projections[0].item_version == 1
    assert projections[0].vector == (1.0, 0.0, 0.0)


@pytest.mark.asyncio
async def test_support_only_update_reuses_vector_and_ignores_diagnostic_model() -> None:
    current = item(7, "西装待取", item_version=2)
    state = SnapshotItemEmbeddingState(
        item_id=7,
        item_version=1,
        text_hash=text_hash(current.summary),
        provider="another-provider",
        model="another-model",
        dimensions=SPACE.dimensions,
    )
    store = FakeStore((batch(current, states=(state,)),))
    embedder = FakeEmbedder()
    worker = processor(store, embedder)

    assert await worker.process_once() is True

    assert embedder.calls == []
    projection = store.persisted[0][1][0]
    assert projection.item_id == 7
    assert projection.item_version == 2
    assert projection.vector is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state_hash", "state_dimensions"),
    (("a" * 64, 3), (text_hash("当前摘要"), 2)),
)
async def test_summary_or_dimension_change_reembeds(
    state_hash: str,
    state_dimensions: int,
) -> None:
    current = item(7, "当前摘要", item_version=2)
    state = SnapshotItemEmbeddingState(
        item_id=7,
        item_version=1,
        text_hash=state_hash,
        provider=SPACE.provider,
        model=SPACE.model,
        dimensions=state_dimensions,
    )
    store = FakeStore((batch(current, states=(state,)),))
    embedder = FakeEmbedder()
    worker = processor(store, embedder)

    assert await worker.process_once() is True

    assert embedder.calls == [("当前摘要",)]
    assert store.persisted[0][1][0].vector == (1.0, 0.0, 0.0)


@pytest.mark.asyncio
async def test_progress_only_candidate_persists_without_embedding() -> None:
    pending = batch()
    store = FakeStore((pending,))
    embedder = FakeEmbedder()
    worker = processor(store, embedder)

    assert await worker.process_once() is True

    assert embedder.calls == []
    assert store.persisted[0][1] == ()


@pytest.mark.asyncio
async def test_no_candidate_returns_false() -> None:
    store = FakeStore(())
    worker = processor(store, FakeEmbedder())

    assert await worker.process_once() is False
    assert store.persisted == []


@pytest.mark.asyncio
async def test_embedding_count_mismatch_is_retried_without_persistence() -> None:
    pending = batch(item(7, "当前摘要"))
    store = FakeStore((pending,))
    retry_times = iter((NOW, NOW + timedelta(seconds=2)))
    worker = processor(
        store,
        FakeEmbedder(missing_vector=True),
        max_retries=1,
        retry_now=lambda: next(retry_times),
    )

    assert await worker.process_once() is True
    assert await worker.process_once() is True

    assert store.persisted == []
    assert store.errors == [
        (pending, "relation embedding count does not match input count")
    ]


@pytest.mark.asyncio
async def test_retry_backoff_is_isolated_by_snapshot_id() -> None:
    first = batch(item(7, "第一项"), snapshot_id=11)
    second = batch(item(8, "第二项"), snapshot_id=12)
    store = FakeStore((first, second))
    embedder = FakeEmbedder(fail=True)
    worker = processor(store, embedder)

    assert await worker.process_once() is True

    embedder._fail = False
    assert await worker.process_once() is True

    assert store.persisted[0][0].snapshot_id == 12
