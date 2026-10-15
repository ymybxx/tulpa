from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.recall import RecallStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.recall.masking import EntityAliases, MaskEmbedding, SingleMaskSource

FISH_ID = UUID(int=1)
FACT_ID = UUID(int=2)
ENTITY_ID = UUID(int=3)
SPACE = EmbeddingSpace("local", "test", 3)


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
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []

    async def fetch(
        self,
        query: str,
        *args: object,
    ) -> list[dict[str, object]]:
        self.calls.append((" ".join(query.split()), args))
        return self.rows

    async def executemany(
        self,
        query: str,
        args: list[tuple[object, ...]],
    ) -> None:
        self.executemany_calls.append((" ".join(query.split()), args))


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeContext:
        return FakeContext(self.connection)

    async def fetch(
        self,
        query: str,
        *args: object,
    ) -> list[dict[str, object]]:
        return await self.connection.fetch(query, *args)

    async def executemany(
        self,
        query: str,
        args: list[tuple[object, ...]],
    ) -> None:
        await self.connection.executemany(query, args)


def store_with(connection: FakeConnection) -> RecallStore:
    return RecallStore(cast(asyncpg.Pool, FakePool(connection)))


def mask(object_id: UUID | int) -> MaskEmbedding:
    return MaskEmbedding(
        object_id=object_id,
        entity_ids=(ENTITY_ID,),
        signature_hash="a" * 64,
        source_text_hash="b" * 64,
        masked_text="以前使用过银剑",
        vector=(1.0, 0.0, 0.0),
    )


@pytest.mark.asyncio
async def test_loads_fact_masks_by_object_and_dimensions_only() -> None:
    connection = FakeConnection(
        [
            {
                "object_id": FACT_ID,
                "entity_ids": [ENTITY_ID],
                "signature_hash": "a" * 64,
                "source_text_hash": "b" * 64,
                "masked_text": "以前使用过银剑",
                "vector_text": "[1,0,0]",
            }
        ]
    )

    rows = await store_with(connection).load_fact_mask_embeddings(
        FISH_ID,
        3,
        [FACT_ID],
    )

    assert rows == [mask(FACT_ID)]
    query, args = connection.calls[0]
    assert "FROM fact_mask_embeddings" in query
    assert "fish_id = $1" in query
    assert "dimensions = $2" in query
    assert "fact_id = ANY($3::uuid[])" in query
    assert "provider" not in query
    assert "model" not in query
    assert args == (FISH_ID, 3, [FACT_ID])


@pytest.mark.asyncio
async def test_bulk_upserts_fact_and_relation_key_masks() -> None:
    connection = FakeConnection([])
    store = store_with(connection)

    await store.upsert_fact_mask_embeddings(FISH_ID, SPACE, [mask(FACT_ID)])
    await store.upsert_relation_mask_embeddings(FISH_ID, SPACE, [mask(7)])

    fact_query, fact_rows = connection.executemany_calls[0]
    relation_query, relation_rows = connection.executemany_calls[1]
    assert "INSERT INTO fact_mask_embeddings" in fact_query
    assert "ON CONFLICT (fact_id, signature_hash, dimensions)" in fact_query
    assert "INSERT INTO memory_snapshot_item_mask_embeddings" in relation_query
    assert (
        "ON CONFLICT (snapshot_item_id, signature_hash, dimensions)" in relation_query
    )
    assert (
        "FROM memory_snapshot_item_embeddings AS relation_embedding" in relation_query
    )
    assert "JOIN memory_snapshot_items AS item" in relation_query
    assert "relation_embedding.item_version = item.item_version" in relation_query
    assert "relation_embedding.text_hash = $5" in relation_query
    assert "FOR SHARE OF relation_embedding" in relation_query
    assert "FOR SHARE OF item" not in relation_query
    assert fact_rows[0][0:3] == (FACT_ID, FISH_ID, [ENTITY_ID])
    assert relation_rows[0][0:3] == (7, FISH_ID, [ENTITY_ID])
    assert fact_rows[0][-1] == "[1.0,0.0,0.0]"
    assert relation_rows[0][-1] == "[1.0,0.0,0.0]"


@pytest.mark.asyncio
async def test_resolves_all_aliases_for_entities_mentioned_in_query() -> None:
    connection = FakeConnection(
        [
            {
                "query_index": 0,
                "entity_id": ENTITY_ID,
                "aliases": ["geralt", "杰洛特", "白狼"],
            }
        ]
    )

    entities = await store_with(connection).resolve_query_entities(
        FISH_ID,
        "白狼以前用过什么剑",
    )

    assert entities == [EntityAliases(ENTITY_ID, ("geralt", "杰洛特", "白狼"))]
    query, args = connection.calls[0]
    assert "FROM unnest($2::text[]) WITH ORDINALITY" in query
    assert "strpos(input.query, alias.alias) > 0" in query
    assert "array_agg(alias.alias ORDER BY alias.alias)" in query
    assert args == (FISH_ID, ["白狼以前用过什么剑"])


@pytest.mark.asyncio
async def test_resolves_batch_query_entities_in_input_order_with_one_sql_call() -> None:
    connection = FakeConnection(
        [
            {
                "query_index": 0,
                "entity_id": ENTITY_ID,
                "aliases": ["geralt", "杰洛特", "白狼"],
            },
            {
                "query_index": 1,
                "entity_id": ENTITY_ID,
                "aliases": ["geralt", "杰洛特", "白狼"],
            },
        ]
    )

    batches = await store_with(connection).resolve_query_entities_batch(
        FISH_ID,
        ["白狼以前用过什么剑", "杰洛特以前用过什么剑", ""],
    )

    same_entity = EntityAliases(ENTITY_ID, ("geralt", "杰洛特", "白狼"))
    assert batches == ((same_entity,), (same_entity,), ())
    assert len(connection.calls) == 1
    query, args = connection.calls[0]
    assert "unnest($2::text[]) WITH ORDINALITY" in query
    assert "query_index" in query
    assert args == (
        FISH_ID,
        ["白狼以前用过什么剑", "杰洛特以前用过什么剑", ""],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "object_id", "method_name", "table_name"),
    [
        ("fact", FACT_ID, "list_fact_single_mask_sources", "fact_embeddings"),
        (
            "relation",
            7,
            "list_relation_single_mask_sources",
            "memory_snapshot_item_embeddings",
        ),
    ],
)
async def test_lists_only_missing_single_entity_mask_sources(
    kind: str,
    object_id: UUID | int,
    method_name: str,
    table_name: str,
) -> None:
    connection = FakeConnection(
        [
            {
                "object_id": object_id,
                "fish_id": FISH_ID,
                "source_text": "杰洛特以前使用过银剑",
                "source_text_hash": "b" * 64,
                "entity_id": ENTITY_ID,
                "aliases": ["geralt", "杰洛特", "白狼"],
            }
        ]
    )
    method = getattr(store_with(connection), method_name)

    sources = await method(3, 100, offset=25)

    assert sources == [
        SingleMaskSource(
            kind=kind,
            object_id=object_id,
            fish_id=FISH_ID,
            source_text="杰洛特以前使用过银剑",
            source_text_hash="b" * 64,
            entity=EntityAliases(ENTITY_ID, ("geralt", "杰洛特", "白狼")),
        )
    ]
    query, args = connection.calls[0]
    assert f"FROM {table_name}" in query
    assert "NOT EXISTS" in query
    assert "source_text_hash" in query
    assert "entity_ids = ARRAY" in query
    assert "fish.background_owner IS NOT DISTINCT FROM $3" in query
    assert "fish.deleted_at IS NULL" in query
    assert "fish.snapshot_storage_version = 2" in query
    assert "OFFSET $4" in query
    if kind == "relation":
        assert "JOIN memory_snapshot_items AS item" in query
        assert "link.fact_id = ANY(item.support_fact_ids)" in query
        assert "relation_embedding.item_version = item.item_version" in query
        assert "existing.snapshot_item_id" in query
    assert args == (3, 100, None, 25)


@pytest.mark.asyncio
async def test_loads_only_masks_for_current_item_projection() -> None:
    connection = FakeConnection(
        [
            {
                "object_id": 7,
                "entity_ids": [ENTITY_ID],
                "signature_hash": "a" * 64,
                "source_text_hash": "b" * 64,
                "masked_text": "以前使用过银剑",
                "vector_text": "[1,0,0]",
            }
        ]
    )

    rows = await store_with(connection).load_relation_mask_embeddings(
        FISH_ID,
        3,
        [7],
    )

    assert rows == [mask(7)]
    query, args = connection.calls[0]
    assert "FROM memory_snapshot_item_mask_embeddings AS mask" in query
    assert "JOIN memory_snapshot_item_embeddings AS relation_embedding" in query
    assert "JOIN memory_snapshot_items AS item" in query
    assert "JOIN memory_snapshots AS snapshot" in query
    assert "mask.source_text_hash = relation_embedding.text_hash" in query
    assert "relation_embedding.item_version = item.item_version" in query
    assert (
        "relation_embedding.session_id IS NOT DISTINCT FROM snapshot.session_id"
        in query
    )
    assert "relation_embedding.snapshot_type = snapshot.snapshot_type" in query
    assert args == (FISH_ID, 3, [7])
