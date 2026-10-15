import hashlib
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.relations import SnapshotItemRelationStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.relations.models import (
    PendingSnapshotItemProjectionBatch,
    SnapshotItemEmbeddingProjection,
    SnapshotItemEmbeddingState,
)
from fish_memory.snapshots.models import SnapshotItemState

FISH_ID = UUID(int=1)
SESSION_ID = UUID(int=2)
FACT_A = UUID(int=10)
FACT_B = UUID(int=11)
NOW = datetime(2026, 8, 27, tzinfo=UTC)
SPACE = EmbeddingSpace("local", "test-model", 3)


class FakeContext:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.header: dict[str, object] | None = {"id": 11, "version": 4}
        self.locked_items: list[dict[str, object]] = [{"id": 7, "item_version": 2}]
        self.complete = True
        self.update_result = "UPDATE 1"

    def transaction(self) -> FakeContext:
        return FakeContext(None)

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.calls.append((" ".join(query.split()), args))
        return self.header

    async def fetch(
        self,
        query: str,
        *args: object,
    ) -> list[dict[str, object]]:
        self.calls.append((" ".join(query.split()), args))
        return self.locked_items

    async def fetchval(self, query: str, *args: object) -> object:
        self.calls.append((" ".join(query.split()), args))
        return self.complete

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((" ".join(query.split()), args))
        if "UPDATE memory_snapshot_item_embeddings" in query:
            return self.update_result
        return "OK"


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.rows: list[dict[str, object]] = []

    def acquire(self) -> FakeContext:
        return FakeContext(self.connection)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.connection.calls.append((" ".join(query.split()), args))
        return self.rows


def store_with(
    connection: FakeConnection,
    *,
    snapshot_type: str = "dynamic",
    background_owner: UUID | None = None,
) -> tuple[SnapshotItemRelationStore, FakePool]:
    pool = FakePool(connection)
    return (
        SnapshotItemRelationStore(
            cast(asyncpg.Pool, pool),
            snapshot_type=snapshot_type,
            background_owner=background_owner,
        ),
        pool,
    )


def current_item(
    item_id: int = 7,
    *,
    item_version: int = 2,
    summary: str = "当前关系摘要",
) -> SnapshotItemState:
    return SnapshotItemState(
        id=item_id,
        summary=summary,
        support_fact_ids=(FACT_A,),
        lineage_fact_ids=(FACT_A, FACT_B),
        last_supported_at=NOW,
        item_version=item_version,
    )


def batch(*items: SnapshotItemState) -> PendingSnapshotItemProjectionBatch:
    return PendingSnapshotItemProjectionBatch(
        fish_id=FISH_ID,
        snapshot_id=11,
        session_id=SESSION_ID,
        snapshot_type="dynamic",
        snapshot_version=4,
        items=items,
        embedding_states=(),
    )


def projection(*, vector: tuple[float, ...] | None) -> SnapshotItemEmbeddingProjection:
    return SnapshotItemEmbeddingProjection(
        item_id=7,
        item_version=2,
        session_id=SESSION_ID,
        snapshot_type="dynamic",
        snapshot_text="当前关系摘要",
        text_hash=hashlib.sha256("当前关系摘要".encode()).hexdigest(),
        last_supported_at=NOW,
        vector=vector,
    )


@pytest.mark.asyncio
async def test_lists_dynamic_candidate_items_and_projection_states() -> None:
    connection = FakeConnection()
    store, pool = store_with(connection, background_owner=UUID(int=99))
    pool.rows = [
        {
            "fish_id": FISH_ID,
            "snapshot_id": 11,
            "session_id": SESSION_ID,
            "snapshot_type": "dynamic",
            "snapshot_version": 4,
            "item_id": 7,
            "summary": "当前关系摘要",
            "support_fact_ids": [FACT_A],
            "lineage_fact_ids": [FACT_A, FACT_B],
            "last_supported_at": NOW,
            "item_version": 2,
            "embedding_item_id": 7,
            "embedding_item_version": 1,
            "embedding_text_hash": "a" * 64,
            "embedding_provider": "diagnostic-provider",
            "embedding_model": "diagnostic-model",
            "embedding_dimensions": 3,
        }
    ]

    pending = await store.list_pending_batches(10, SPACE.dimensions)

    assert pending == [
        PendingSnapshotItemProjectionBatch(
            fish_id=FISH_ID,
            snapshot_id=11,
            session_id=SESSION_ID,
            snapshot_type="dynamic",
            snapshot_version=4,
            items=(current_item(),),
            embedding_states=(
                SnapshotItemEmbeddingState(
                    item_id=7,
                    item_version=1,
                    text_hash="a" * 64,
                    provider="diagnostic-provider",
                    model="diagnostic-model",
                    dimensions=3,
                ),
            ),
        )
    ]
    query, args = connection.calls[0]
    assert "fish.snapshot_storage_version = 2" in query
    assert "fish.deleted_at IS NULL" in query
    assert "fish.background_owner IS NOT DISTINCT FROM $4" in query
    assert "session_background_task_progress" in query
    assert "COALESCE(progress.through_sequence, 0) < snapshot.version" in query
    assert "embedding.snapshot_item_id IS NULL" in query
    assert "embedding.item_version <> item.item_version" in query
    assert "embedding.dimensions <> $5" in query
    assert "LEFT JOIN memory_snapshot_item_embeddings" in query
    assert "embedding.fish_id = snapshot.fish_id" in query
    assert "embedding.session_id IS NOT DISTINCT FROM snapshot.session_id" in query
    assert "embedding.snapshot_type = snapshot.snapshot_type" in query
    assert args == (10, "dynamic_relations", "dynamic", UUID(int=99), 3)


@pytest.mark.asyncio
async def test_static_candidates_use_fish_progress_and_static_scope() -> None:
    connection = FakeConnection()
    store, _pool = store_with(connection, snapshot_type="static")

    assert await store.list_pending_batches(10, 3) == []

    query, args = connection.calls[0]
    assert "background_task_progress" in query
    assert "session_background_task_progress" not in query
    assert "snapshot.session_id IS NULL" in query
    assert args == (10, "static_relations", "static", None, 3)


@pytest.mark.asyncio
async def test_reuses_vector_without_deleting_masks_and_advances_when_complete() -> (
    None
):
    connection = FakeConnection()
    store, _pool = store_with(connection)

    persisted = await store.persist_batch(
        batch(current_item()),
        (projection(vector=None),),
        SPACE,
    )

    assert persisted is True
    metadata_update = next(
        call
        for call in connection.calls
        if "UPDATE memory_snapshot_item_embeddings" in call[0]
    )
    assert "embedding =" not in metadata_update[0]
    assert "provider =" not in metadata_update[0]
    assert "model =" not in metadata_update[0]
    assert metadata_update[1][0:3] == (7, 2, FISH_ID)
    assert not any(
        "DELETE FROM memory_snapshot_item_mask_embeddings" in query
        for query, _args in connection.calls
    )
    assert any(
        "session_background_task_progress" in query for query, _args in connection.calls
    )
    completeness_query, completeness_args = next(
        call for call in connection.calls if "SELECT NOT EXISTS" in call[0]
    )
    assert "embedding.fish_id = $3" in completeness_query
    assert "embedding.session_id IS NOT DISTINCT FROM $4" in completeness_query
    assert "embedding.snapshot_type = $5" in completeness_query
    assert completeness_args == (11, 3, FISH_ID, SESSION_ID, "dynamic")


@pytest.mark.asyncio
async def test_reembedding_replaces_full_vector_and_deletes_item_masks() -> None:
    connection = FakeConnection()
    store, _pool = store_with(connection)

    assert await store.persist_batch(
        batch(current_item()),
        (projection(vector=(1.0, 0.0, 0.0)),),
        SPACE,
    )

    delete = next(
        call
        for call in connection.calls
        if "DELETE FROM memory_snapshot_item_mask_embeddings" in call[0]
    )
    assert delete[1] == (7,)
    upsert = next(
        call
        for call in connection.calls
        if "INSERT INTO memory_snapshot_item_embeddings" in call[0]
    )
    assert upsert[1][0:7] == (
        7,
        2,
        FISH_ID,
        SESSION_ID,
        "dynamic",
        "当前关系摘要",
        hashlib.sha256("当前关系摘要".encode()).hexdigest(),
    )
    assert cast(str, upsert[1][-1]) == "[1.0,0.0,0.0]"


@pytest.mark.asyncio
async def test_partial_projection_commit_does_not_advance_progress() -> None:
    connection = FakeConnection()
    connection.complete = False
    store, _pool = store_with(connection)

    assert await store.persist_batch(
        batch(current_item()),
        (projection(vector=(1.0, 0.0, 0.0)),),
        SPACE,
    )

    assert not any(
        "background_task_progress" in query for query, _args in connection.calls
    )


@pytest.mark.asyncio
async def test_header_version_conflict_writes_nothing() -> None:
    connection = FakeConnection()
    connection.header = None
    store, _pool = store_with(connection)

    assert not await store.persist_batch(
        batch(current_item()),
        (projection(vector=(1.0, 0.0, 0.0)),),
        SPACE,
    )

    assert not any(
        "INSERT INTO memory_snapshot_item_embeddings" in query
        or "UPDATE memory_snapshot_item_embeddings" in query
        or "background_task_progress" in query
        for query, _args in connection.calls
    )


@pytest.mark.asyncio
async def test_any_current_item_version_conflict_writes_nothing() -> None:
    connection = FakeConnection()
    connection.locked_items = [{"id": 7, "item_version": 3}]
    store, _pool = store_with(connection)

    assert not await store.persist_batch(
        batch(current_item()),
        (projection(vector=(1.0, 0.0, 0.0)),),
        SPACE,
    )

    assert not any(
        "INSERT INTO memory_snapshot_item_embeddings" in query
        or "UPDATE memory_snapshot_item_embeddings" in query
        or "background_task_progress" in query
        for query, _args in connection.calls
    )


@pytest.mark.asyncio
async def test_projection_must_be_a_unique_subset_of_current_items() -> None:
    connection = FakeConnection()
    store, _pool = store_with(connection)
    duplicate = projection(vector=(1.0, 0.0, 0.0))

    assert not await store.persist_batch(
        batch(current_item()),
        (duplicate, duplicate),
        SPACE,
    )
    assert connection.calls == []
