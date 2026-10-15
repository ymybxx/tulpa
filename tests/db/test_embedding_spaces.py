from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.embedding_spaces import (
    EmbeddingSpaceMismatchError,
    FishEmbeddingSpaceStore,
    ensure_embedding_hnsw_index,
)
from fish_memory.embeddings.client import EmbeddingSpace

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
LOCAL_SPACE = EmbeddingSpace(
    provider="local",
    model="intfloat/multilingual-e5-small",
    dimensions=384,
)


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.fetchrow_results: list[dict[str, object] | None] = []
        self.fetch_results: list[list[dict[str, object]]] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, object] | None:
        self.calls.append((" ".join(query.split()), args))
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        self.calls.append((" ".join(query.split()), args))
        return self.fetch_results.pop(0)

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((" ".join(query.split()), args))
        return "OK"

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *args: object) -> None:
        return None


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


def store_with(connection: FakeConnection) -> FishEmbeddingSpaceStore:
    return FishEmbeddingSpaceStore(cast(asyncpg.Pool, FakePool(connection)))


@pytest.mark.asyncio
async def test_claims_unbound_fish_with_current_embedding_space() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "embedding_provider": None,
            "embedding_model": None,
            "embedding_dimensions": None,
        }
    )
    connection.fetch_results.append([])

    await store_with(connection).ensure_compatible(FISH_ID, LOCAL_SPACE)

    update_query, update_args = connection.calls[2]
    assert "UPDATE fishes" in update_query
    assert update_args == (
        FISH_ID,
        "local",
        "intfloat/multilingual-e5-small",
        384,
    )


@pytest.mark.asyncio
async def test_rejects_dimension_change_for_existing_fish() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "embedding_provider": "openai-compatible",
            "embedding_model": "text-embedding-v4",
            "embedding_dimensions": 1024,
        }
    )

    with pytest.raises(EmbeddingSpaceMismatchError) as exc_info:
        await store_with(connection).ensure_compatible(FISH_ID, LOCAL_SPACE)

    message = str(exc_info.value)
    assert str(FISH_ID) in message
    assert "1024" in message
    assert "384" in message


@pytest.mark.asyncio
async def test_rejects_model_change_even_when_dimensions_match() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "embedding_provider": "local",
            "embedding_model": "another-384-model",
            "embedding_dimensions": 384,
        }
    )

    with pytest.raises(EmbeddingSpaceMismatchError, match="another-384-model"):
        await store_with(connection).ensure_compatible(FISH_ID, LOCAL_SPACE)


@pytest.mark.asyncio
async def test_unbound_legacy_fish_inherits_existing_vector_space() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "embedding_provider": None,
            "embedding_model": None,
            "embedding_dimensions": None,
        }
    )
    connection.fetch_results.append(
        [
            {
                "provider": "openai-compatible",
                "model": "text-embedding-v4",
                "dimensions": 1024,
            }
        ]
    )

    with pytest.raises(EmbeddingSpaceMismatchError, match="1024"):
        await store_with(connection).ensure_compatible(FISH_ID, LOCAL_SPACE)

    assert any("UPDATE fishes" in query for query, _ in connection.calls)


@pytest.mark.asyncio
async def test_creates_dimension_specific_hnsw_index() -> None:
    connection = FakeConnection()

    await ensure_embedding_hnsw_index(
        cast(asyncpg.Pool, FakePool(connection)),
        768,
    )

    query, args = connection.calls[0]
    assert "idx_fact_embeddings_hnsw_768" in query
    assert "embedding::vector(768)" in query
    assert "WHERE dimensions = 768" in query
    assert args == ()
    relation_query, relation_args = connection.calls[1]
    assert "idx_memory_snapshot_item_embeddings_hnsw_768" in relation_query
    assert "ON memory_snapshot_item_embeddings" in relation_query
    assert "embedding::vector(768)" in relation_query
    assert "WHERE dimensions = 768" in relation_query
    assert "fact_relation_group_embeddings" not in relation_query
    assert relation_args == ()
    assert len(connection.calls) == 2


@pytest.mark.asyncio
async def test_rejects_hnsw_dimensions_outside_pgvector_limit() -> None:
    connection = FakeConnection()

    with pytest.raises(ValueError, match="between 1 and 2000"):
        await ensure_embedding_hnsw_index(
            cast(asyncpg.Pool, FakePool(connection)),
            2001,
        )

    assert connection.calls == []
