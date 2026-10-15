from __future__ import annotations

import json
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.recall import RecallStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.models import Role
from fish_memory.recall.models import (
    RecallOrder,
    RecallSourceRole,
    RecallTimeField,
    RecallTimeFilter,
    RecallTimeRange,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
FACT_ID = UUID("00000000-0000-0000-0000-000000000002")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000003")
NOW = datetime(2026, 7, 26, tzinfo=UTC)
EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model="text-embedding-v4",
    dimensions=1024,
)


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

    def transaction(self) -> FakeContext:
        return FakeContext(None)

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((" ".join(query.split()), args))
        return "OK"

    async def fetch(
        self,
        query: str,
        *args: object,
    ) -> list[dict[str, object]]:
        self.calls.append((" ".join(query.split()), args))
        return self.rows


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


def fact_record(
    *,
    semantic_score: float | None = None,
    invalidated_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "fact_id": FACT_ID,
        "source_message_id": MESSAGE_ID,
        "source_role": "user",
        "text": "该用户不想继续玩游戏",
        "occurred_start": None,
        "occurred_end": None,
        "mentioned_at": NOW,
        "entities": json.dumps(["游戏"], ensure_ascii=False),
        "semantic_score": semantic_score,
        "invalidated_at": invalidated_at,
    }


def store_with(connection: FakeConnection) -> RecallStore:
    return RecallStore(cast(asyncpg.Pool, FakePool(connection)))


@pytest.mark.asyncio
async def test_semantic_search_filters_fish_and_model_and_uses_hnsw() -> None:
    connection = FakeConnection([fact_record(semantic_score=0.82)])
    store = store_with(connection)
    vector = (0.25,) * 1024

    results = await store.semantic_search(
        FISH_ID,
        EMBEDDING_SPACE,
        vector,
        time_filter=None,
        order=RecallOrder.RELEVANCE,
        limit=20,
    )

    assert len(results) == 1
    assert results[0].id == FACT_ID
    assert results[0].source_message_id == MESSAGE_ID
    assert results[0].source_role is Role.USER
    assert results[0].entities == ("游戏",)
    assert results[0].semantic_score == pytest.approx(0.82)
    assert results[0].invalidated_at is None
    set_query, set_args = connection.calls[0]
    assert set_query == "SET LOCAL hnsw.iterative_scan = strict_order"
    assert set_args == ()
    query, args = connection.calls[1]
    assert "JOIN facts AS fact ON fact.id = embedding.fact_id" in query
    assert "JOIN messages AS source ON source.id = fact.source_message_id" in query
    assert "embedding.fish_id = $1" in query
    assert "embedding.provider = $2" in query
    assert "embedding.model = $3" in query
    assert "embedding.dimensions = $4" in query
    assert "embedding.embedding::vector(1024) <=> $5::vector(1024)" in query
    assert "fact.invalidated_at" in query
    assert "ORDER BY embedding.embedding::vector(1024) <=> $5::vector(1024)" in query
    assert "LIMIT $6" in query
    assert args[0:4] == (
        FISH_ID,
        "openai-compatible",
        "text-embedding-v4",
        1024,
    )
    assert cast(str, args[4]).startswith("[0.25,0.25")
    assert args[5] == 20


@pytest.mark.asyncio
async def test_entity_search_uses_normalized_links_and_ranks_by_query_vector() -> None:
    connection = FakeConnection([fact_record(semantic_score=0.91)])
    vector = (0.25,) * 1024

    results = await store_with(connection).entity_search(
        FISH_ID,
        ("杰洛特",),
        EMBEDDING_SPACE,
        vector,
        limit=20,
    )

    assert results[0].id == FACT_ID
    assert results[0].semantic_score == pytest.approx(0.91)
    query, args = connection.calls[0]
    assert "FROM entity_aliases" in query
    assert "JOIN fact_entities" in query
    assert "JOIN facts AS fact" in query
    assert "alias.fish_id = $1" in query
    assert "alias.alias = ANY($2::text[])" in query
    assert "embedding.provider = $3" in query
    assert "embedding.model = $4" in query
    assert "embedding.dimensions = $5" in query
    assert "embedding.embedding::vector(1024) <=> $6::vector(1024)" in query
    assert "ORDER BY semantic_score DESC NULLS LAST" in query
    assert args[0:5] == (
        FISH_ID,
        ["杰洛特"],
        "openai-compatible",
        "text-embedding-v4",
        1024,
    )
    assert cast(str, args[5]).startswith("[0.25,0.25")
    assert args[6] == 20


@pytest.mark.asyncio
async def test_entity_search_applies_planned_time_filter_and_order() -> None:
    connection = FakeConnection([fact_record(semantic_score=0.91)])
    start = datetime.fromisoformat("2026-07-01T00:00:00+08:00")
    end = datetime.fromisoformat("2026-07-31T23:59:59+08:00")

    await store_with(connection).entity_search(
        FISH_ID,
        ("杰洛特",),
        EMBEDDING_SPACE,
        (0.25,) * 1024,
        time_filter=RecallTimeFilter(
            field=RecallTimeField.MENTIONED,
            start=start,
            end=end,
        ),
        order=RecallOrder.LATEST,
        limit=20,
    )

    query, args = connection.calls[0]
    assert "fact.mentioned_at >= $7" in query
    assert "fact.mentioned_at <= $8" in query
    assert "ORDER BY fact.mentioned_at DESC" in query
    assert args[6:] == (start, end, 20)


@pytest.mark.asyncio
async def test_occurred_semantic_search_does_not_fallback_to_mentioned() -> None:
    connection = FakeConnection([fact_record(semantic_score=0.72)])
    store = store_with(connection)
    start = datetime.fromisoformat("2025-01-01T00:00:00+08:00")
    end = datetime.fromisoformat("2025-12-31T23:59:59+08:00")
    vector = (0.25,) * 1024

    results = await store.semantic_search(
        FISH_ID,
        EMBEDDING_SPACE,
        vector,
        time_filter=RecallTimeFilter(
            field=RecallTimeField.OCCURRED,
            start=start,
            end=end,
        ),
        order=RecallOrder.RELEVANCE,
        limit=30,
    )

    assert len(results) == 1
    query, args = connection.calls[1]
    assert "fact.occurred_start IS NOT NULL" in query
    assert "COALESCE(fact.occurred_end, fact.occurred_start) >= $6" in query
    assert "fact.occurred_start <= $7" in query
    assert "COALESCE(fact.occurred_start, fact.mentioned_at)" not in query
    assert args[0:4] == (
        FISH_ID,
        "openai-compatible",
        "text-embedding-v4",
        1024,
    )
    assert args[5:] == (start, end, 30)


@pytest.mark.asyncio
async def test_mentioned_semantic_search_uses_requested_order() -> None:
    connection = FakeConnection([fact_record(semantic_score=0.72)])
    store = store_with(connection)
    start = datetime.fromisoformat("2026-07-01T00:00:00+08:00")
    vector = (0.25,) * 1024

    await store.semantic_search(
        FISH_ID,
        EMBEDDING_SPACE,
        vector,
        time_filter=RecallTimeFilter(
            field=RecallTimeField.MENTIONED,
            start=start,
            end=None,
        ),
        order=RecallOrder.EARLIEST,
        limit=10,
    )

    query, args = connection.calls[1]
    assert "fact.mentioned_at >= $6" in query
    assert "WITH semantic_candidates" not in query
    assert "ORDER BY fact.mentioned_at ASC" in query
    assert "embedding.embedding::vector(1024) <=> $5::vector(1024)" in query
    assert args[0:4] == (
        FISH_ID,
        "openai-compatible",
        "text-embedding-v4",
        1024,
    )
    assert args[5:] == (start, 10)


@pytest.mark.asyncio
async def test_time_search_filters_before_selecting_latest_boundary() -> None:
    connection = FakeConnection([fact_record(semantic_score=None)])
    start = datetime.fromisoformat("2026-07-01T00:00:00+08:00")
    end = datetime.fromisoformat("2026-07-31T23:59:59+08:00")
    time_filter = RecallTimeFilter(
        field=RecallTimeField.OCCURRED,
        start=start,
        end=end,
    )

    results = await store_with(connection).time_search(
        FISH_ID,
        time_filter,
        RecallOrder.LATEST,
        limit=50,
    )

    assert len(results) == 1
    query, args = connection.calls[0]
    assert "fact.fish_id = $1" in query
    assert "fact.occurred_start IS NOT NULL" in query
    assert "COALESCE(fact.occurred_end, fact.occurred_start) >= $2" in query
    assert "fact.occurred_start <= $3" in query
    assert "ORDER BY fact.occurred_start DESC" in query
    assert "fact_embeddings" not in query
    assert args == (FISH_ID, start, end, 50)


@pytest.mark.asyncio
async def test_mentioned_time_search_without_bounds_has_valid_where_clause() -> None:
    connection = FakeConnection([fact_record(semantic_score=None)])

    await store_with(connection).time_search(
        FISH_ID,
        RecallTimeFilter(
            field=RecallTimeField.MENTIONED,
            start=None,
            end=None,
        ),
        RecallOrder.EARLIEST,
        limit=50,
    )

    query, args = connection.calls[0]
    assert "WHERE fact.fish_id = $1 ORDER BY" in query
    assert "AND ORDER BY" not in query
    assert args == (FISH_ID, 50)


@pytest.mark.asyncio
async def test_fact_head_hnsw_maps_projection_ready_items_in_the_same_query() -> None:
    connection = FakeConnection(
        [
            {
                **fact_record(semantic_score=0.88),
                "embedding_text": "不再玩游戏",
                "text_hash": "a" * 64,
                "relation_item_ids": [7, 9],
            }
        ]
    )

    head = await store_with(connection).search_fact_head(
        FISH_ID,
        EMBEDDING_SPACE.dimensions,
        (0.25,) * EMBEDDING_SPACE.dimensions,
        time_range=None,
        limit=500,
        use_hnsw=True,
    )

    assert head[0].fact.id == FACT_ID
    assert head[0].relation_item_ids == (7, 9)
    assert head[0].embedding_text == "不再玩游戏"
    set_query, _ = connection.calls[0]
    assert set_query == "SET LOCAL hnsw.iterative_scan = strict_order"
    query, args = connection.calls[1]
    assert "WITH fact_head AS" in query
    assert "fact.fish_id = $1" in query
    assert "fact.invalidated_at IS NULL" in query
    assert "relation_item_ids" in query
    assert "item.support_fact_ids @> ARRAY[fact.id]::uuid[]" in query
    assert "FROM memory_snapshot_items AS item" in query
    assert "JOIN memory_snapshots AS snapshot" in query
    assert "JOIN memory_snapshot_item_embeddings AS relation_embedding" in query
    assert "relation_embedding.item_version = item.item_version" in query
    assert (
        "relation_embedding.session_id IS NOT DISTINCT FROM snapshot.session_id"
        in query
    )
    assert "relation_embedding.snapshot_type = snapshot.snapshot_type" in query
    assert "embedding.dimensions = $2" in query
    assert "embedding.provider" not in query
    assert "embedding.model" not in query
    assert "LIMIT $4" in query
    assert args[0:2] == (FISH_ID, 1024)
    assert cast(str, args[2]).startswith("[0.25,0.25")
    assert args[3] == 500


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_role", "expected_role"),
    [
        (RecallSourceRole.USER, "user"),
        (RecallSourceRole.ASSISTANT, "assistant"),
    ],
)
async def test_fact_head_filters_by_persisted_source_role(
    source_role: RecallSourceRole,
    expected_role: str,
) -> None:
    """Catches head candidates crossing the Planner's source-role boundary."""
    connection = FakeConnection(
        [
            {
                **fact_record(semantic_score=0.88),
                "embedding_text": "不再玩游戏",
                "text_hash": "a" * 64,
                "relation_item_ids": [],
            }
        ]
    )

    await store_with(connection).search_fact_head(
        FISH_ID,
        EMBEDDING_SPACE.dimensions,
        (0.25,) * EMBEDDING_SPACE.dimensions,
        source_role=source_role,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )

    query, args = connection.calls[1]
    assert "source.role = $4" in query
    assert args[3:] == (expected_role, 50)


@pytest.mark.asyncio
async def test_either_fact_head_omits_source_role_predicate() -> None:
    """Catches either scope becoming an unintended third persisted role."""
    connection = FakeConnection(
        [
            {
                **fact_record(semantic_score=0.88),
                "embedding_text": "不再玩游戏",
                "text_hash": "a" * 64,
                "relation_item_ids": [],
            }
        ]
    )

    await store_with(connection).search_fact_head(
        FISH_ID,
        EMBEDDING_SPACE.dimensions,
        (0.25,) * EMBEDDING_SPACE.dimensions,
        source_role=RecallSourceRole.EITHER,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )

    query, args = connection.calls[1]
    assert "source.role =" not in query
    assert args[3:] == (50,)


@pytest.mark.asyncio
async def test_fact_head_without_range_has_no_time_predicate() -> None:
    """Catches an absent production range accidentally adding a time scope."""
    connection = FakeConnection(
        [
            {
                **fact_record(semantic_score=0.88),
                "embedding_text": "不再玩游戏",
                "text_hash": "a" * 64,
                "relation_item_ids": [],
            }
        ]
    )

    await store_with(connection).search_fact_head(
        FISH_ID,
        EMBEDDING_SPACE.dimensions,
        (0.25,) * EMBEDDING_SPACE.dimensions,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )

    query, args = connection.calls[1]
    assert "OR ()" not in query
    assert "fact.occurred_start IS NOT NULL" not in query
    assert "fact.mentioned_at >=" not in query
    assert "fact.mentioned_at <=" not in query
    assert "LIMIT $4" in query
    assert args[0:2] == (FISH_ID, 1024)
    assert cast(str, args[2]).startswith("[0.25,0.25")
    assert args[3] == 50


@pytest.mark.asyncio
async def test_fact_head_bounded_range_uses_one_parenthesized_or() -> None:
    """Catches either legal production time branch being narrowed away."""
    connection = FakeConnection(
        [
            {
                **fact_record(semantic_score=0.88),
                "embedding_text": "用户购买奢华晚礼服花费800美元",
                "text_hash": "a" * 64,
                "relation_item_ids": [],
            }
        ]
    )
    start = datetime.fromisoformat("2025-01-01T00:00:00+00:00")
    end = datetime.fromisoformat("2025-12-31T23:59:59+00:00")

    await store_with(connection).search_fact_head(
        FISH_ID,
        EMBEDDING_SPACE.dimensions,
        (0.25,) * EMBEDDING_SPACE.dimensions,
        time_range=RecallTimeRange(start=start, end=end),
        limit=50,
        use_hnsw=False,
    )

    query, args = connection.calls[1]
    assert (
        "((fact.occurred_start IS NOT NULL AND "
        "COALESCE(fact.occurred_end, fact.occurred_start) >= $4 AND "
        "fact.occurred_start <= $6) OR (fact.mentioned_at >= $5 AND "
        "fact.mentioned_at <= $7))"
    ) in query
    assert "LIMIT $8" in query
    assert args[3:] == (start, start, end, end, 50)


@pytest.mark.asyncio
async def test_fact_head_end_only_range_bounds_both_or_branches() -> None:
    """Catches an end-only range omitting one legal production time branch."""
    connection = FakeConnection(
        [
            {
                **fact_record(semantic_score=0.88),
                "embedding_text": "Ava 和 Lily 出生于四月",
                "text_hash": "a" * 64,
                "relation_item_ids": [],
            }
        ]
    )

    await store_with(connection).search_fact_head(
        FISH_ID,
        EMBEDDING_SPACE.dimensions,
        (0.25,) * EMBEDDING_SPACE.dimensions,
        time_range=RecallTimeRange(start=None, end=NOW),
        limit=50,
        use_hnsw=False,
    )

    query, args = connection.calls[1]
    assert (
        "((fact.occurred_start IS NOT NULL AND fact.occurred_start <= $4) "
        "OR (fact.mentioned_at <= $5))"
    ) in query
    assert "LIMIT $6" in query
    assert args[3:] == (NOW, NOW, 50)


@pytest.mark.asyncio
async def test_time_scoped_fact_count_stops_at_exact_limit_plus_one() -> None:
    connection = FakeConnection([{"fact_count": 200_001}])
    start = datetime.fromisoformat("2025-01-01T00:00:00+08:00")
    end = datetime.fromisoformat("2025-12-31T23:59:59+08:00")

    count = await store_with(connection).count_time_scoped_facts(
        FISH_ID,
        1024,
        RecallTimeRange(start=start, end=end),
        stop_after=200_000,
    )

    assert count == 200_001
    query, args = connection.calls[0]
    assert "LIMIT $7" in query
    assert "fact.invalidated_at IS NULL" in query
    assert "embedding.fish_id = $1" in query
    assert "embedding.dimensions = $2" in query
    assert "fact.occurred_start IS NOT NULL" in query
    assert "COALESCE(fact.occurred_end, fact.occurred_start) >= $3" in query
    assert "fact.occurred_start <= $5" in query
    assert "fact.mentioned_at >= $4" in query
    assert "fact.mentioned_at <= $6" in query
    assert args == (FISH_ID, 1024, start, start, end, end, 200_001)


@pytest.mark.asyncio
async def test_time_scoped_fact_count_filters_by_persisted_source_role() -> None:
    """Catches Exact/HNSW selection counting a broader role set than the head."""
    connection = FakeConnection([{"fact_count": 1}])
    start = datetime.fromisoformat("2025-01-01T00:00:00+08:00")

    count = await store_with(connection).count_time_scoped_facts(
        FISH_ID,
        1024,
        RecallTimeRange(start=start, end=None),
        source_role=RecallSourceRole.USER,
        stop_after=200_000,
    )

    assert count == 1
    query, args = connection.calls[0]
    assert "JOIN messages AS source ON source.id = fact.source_message_id" in query
    assert "COALESCE(fact.occurred_end, fact.occurred_start) >= $3" in query
    assert "fact.mentioned_at >= $4" in query
    assert "source.role = $5" in query
    assert args == (FISH_ID, 1024, start, start, "user", 200_001)


@pytest.mark.asyncio
async def test_unbounded_count_range_uses_no_time_predicate_or_args() -> None:
    """Catches an unbounded production range being rendered as empty SQL."""
    connection = FakeConnection([{"fact_count": 7}])

    count = await store_with(connection).count_time_scoped_facts(
        FISH_ID,
        1024,
        RecallTimeRange(start=None, end=None),
        stop_after=200_000,
    )

    assert count == 7
    query, args = connection.calls[0]
    assert "OR ()" not in query
    assert "fact.occurred_start IS NOT NULL" not in query
    assert "fact.mentioned_at >=" not in query
    assert "fact.mentioned_at <=" not in query
    assert "LIMIT $3" in query
    assert args == (FISH_ID, 1024, 200_001)


@pytest.mark.asyncio
async def test_bounded_count_range_renders_both_time_branches() -> None:
    """Catches the production OR range reusing placeholders across branches."""
    connection = FakeConnection([{"fact_count": 5}])
    start = datetime.fromisoformat("2025-01-01T00:00:00+00:00")
    end = datetime.fromisoformat("2026-12-31T23:59:59+00:00")

    count = await store_with(connection).count_time_scoped_facts(
        FISH_ID,
        1024,
        RecallTimeRange(start=start, end=end),
        stop_after=200_000,
    )

    assert count == 5
    query, args = connection.calls[0]
    assert "fact.occurred_start IS NOT NULL" in query
    assert "COALESCE(fact.occurred_end, fact.occurred_start) >= $3" in query
    assert "fact.occurred_start <= $5" in query
    assert "fact.mentioned_at >= $4" in query
    assert "fact.mentioned_at <= $6" in query
    assert "OR" in query
    assert "LIMIT $7" in query
    assert args == (FISH_ID, 1024, start, start, end, end, 200_001)


@pytest.mark.asyncio
async def test_relation_head_exact_filters_time_through_member_facts() -> None:
    connection = FakeConnection(
        [
            {
                "snapshot_item_id": 7,
                "snapshot_type": "dynamic",
                "snapshot_version": 4,
                "snapshot_text": "西装待取、新靴待取、旧靴待退",
                "text_hash": "b" * 64,
                "last_supported_at": NOW,
                "support_fact_ids": [FACT_ID, UUID(int=4)],
                "support_count": 2,
                "semantic_score": 0.93,
            }
        ]
    )
    start = datetime.fromisoformat("2025-01-01T00:00:00+08:00")
    end = datetime.fromisoformat("2025-12-31T23:59:59+08:00")

    head = await store_with(connection).search_relation_head(
        FISH_ID,
        EMBEDDING_SPACE.dimensions,
        (0.25,) * EMBEDDING_SPACE.dimensions,
        source_role=RecallSourceRole.USER,
        time_range=RecallTimeRange(start=start, end=end),
        limit=50,
        use_hnsw=False,
    )

    assert head[0].id == 7
    assert head[0].support_fact_ids == (FACT_ID, UUID(int=4))
    assert head[0].semantic_score == pytest.approx(0.93)
    set_query, _ = connection.calls[0]
    assert set_query == "SET LOCAL enable_indexscan = off"
    query, args = connection.calls[1]
    assert "SELECT count(DISTINCT member.id)::integer" in query
    assert "= item.support_count" in query
    assert "JOIN fact_embeddings AS member_embedding" in query
    assert "member_embedding.fish_id = $1" in query
    assert "member_embedding.dimensions = $2" in query
    assert "member.id = ANY(item.support_fact_ids)" in query
    assert "member.invalidated_at IS NULL" in query
    assert "member.occurred_start IS NOT NULL" in query
    assert "COALESCE(member.occurred_end, member.occurred_start) >= $4" in query
    assert "member.occurred_start <= $6" in query
    assert "member.mentioned_at >= $5" in query
    assert "member.mentioned_at <= $7" in query
    assert "FROM facts AS source_member" in query
    assert "JOIN messages AS source_message" in query
    assert "source_member.id = ANY(item.support_fact_ids)" in query
    assert "source_member.fish_id = $1" in query
    assert "source_member.invalidated_at IS NULL" in query
    assert "source_message.role = $8" in query
    completeness_query = query.split(") = item.support_count", maxsplit=1)[0]
    assert "source_message.role" not in completeness_query
    assert "relation_embedding.fish_id = $1" in query
    assert "relation_embedding.dimensions = $2" in query
    assert (
        "relation_embedding.session_id IS NOT DISTINCT FROM snapshot.session_id"
        in query
    )
    assert "relation_embedding.snapshot_type = snapshot.snapshot_type" in query
    assert "relation_embedding.item_version = item.item_version" in query
    assert "audit.item_version = item.item_version" in query
    assert "relation_embedding.provider" not in query
    assert "relation_embedding.model" not in query
    assert args[0:2] == (FISH_ID, 1024)
    assert cast(str, args[2]).startswith("[0.25,0.25")
    assert args[3:] == (start, start, end, end, "user", 50)


@pytest.mark.asyncio
async def test_load_fact_keys_hydrates_valid_time_scoped_relation_members() -> None:
    entity_id = UUID(int=55)
    connection = FakeConnection(
        [
            {
                **fact_record(semantic_score=0.84),
                "embedding_text": "白狼使用银剑",
                "text_hash": "c" * 64,
                "entity_ids": [entity_id],
            }
        ]
    )
    start = datetime.fromisoformat("2025-01-01T00:00:00+08:00")

    keys = await store_with(connection).load_fact_keys(
        FISH_ID,
        [FACT_ID],
        1024,
        (0.25,) * 1024,
        time_range=RecallTimeRange(start=start, end=None),
    )

    assert keys[0].fact.id == FACT_ID
    assert keys[0].embedding_text == "白狼使用银剑"
    assert keys[0].entity_ids == (entity_id,)
    query, args = connection.calls[0]
    assert "fact.id = ANY($2::uuid[])" in query
    assert "fact.invalidated_at IS NULL" in query
    assert "embedding.fish_id = $1" in query
    assert "embedding.dimensions = $3" in query
    assert "COALESCE(fact.occurred_end, fact.occurred_start) >= $5" in query
    assert "fact.mentioned_at >= $6" in query
    assert "embedding.provider" not in query
    assert "embedding.model" not in query
    assert args[0:3] == (FISH_ID, [FACT_ID], 1024)
    assert cast(str, args[3]).startswith("[0.25,0.25")
    assert args[4:] == (start, start)


@pytest.mark.asyncio
async def test_load_relation_item_keys_scores_fact_anchored_items_in_one_batch() -> (
    None
):
    connection = FakeConnection(
        [
            {
                "snapshot_item_id": 7,
                "snapshot_type": "dynamic",
                "snapshot_version": 4,
                "snapshot_text": "西装待取、新靴待取、旧靴待退",
                "text_hash": "b" * 64,
                "last_supported_at": NOW,
                "support_fact_ids": [FACT_ID, UUID(int=4)],
                "support_count": 2,
                "semantic_score": 0.81,
            }
        ]
    )

    start = datetime.fromisoformat("2025-01-01T00:00:00+00:00")

    keys = await store_with(connection).load_relation_item_keys(
        FISH_ID,
        [7],
        1024,
        (0.25,) * 1024,
        time_range=RecallTimeRange(start=start, end=NOW),
    )

    assert keys[0].id == 7
    assert keys[0].semantic_score == pytest.approx(0.81)
    query, args = connection.calls[0]
    assert "item.id = ANY($2::bigint[])" in query
    assert "relation_embedding.dimensions = $3" in query
    assert "member.invalidated_at IS NULL" in query
    assert "member.occurred_start IS NOT NULL" in query
    assert "COALESCE(member.occurred_end, member.occurred_start) >= $5" in query
    assert "member.occurred_start <= $7" in query
    assert "member.mentioned_at >= $6" in query
    assert "member.mentioned_at <= $8" in query
    assert "relation_embedding.item_version = item.item_version" in query
    assert "audit.item_version = item.item_version" in query
    assert "relation_embedding.provider" not in query
    assert "relation_embedding.model" not in query
    assert args[0:3] == (FISH_ID, [7], 1024)
    assert cast(str, args[3]).startswith("[0.25,0.25")
    assert args[4:] == (start, start, NOW, NOW)
