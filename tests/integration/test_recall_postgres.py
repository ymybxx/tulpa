import asyncio
import hashlib
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.embedding_spaces import ensure_embedding_hnsw_index
from fish_memory.db.facts import FactStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.recall import RecallStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.facts.models import ExtractedFact
from fish_memory.models import MessageStatus, Role
from fish_memory.recall.masking import MaskEmbedding
from fish_memory.recall.models import (
    RecallOrder,
    RecallSourceRole,
    RecallTimeField,
    RecallTimeFilter,
    RecallTimeRange,
)

FIRST_FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SECOND_FISH_ID = UUID("00000000-0000-0000-0000-000000000002")
MODEL = "text-embedding-v4"
EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model=MODEL,
    dimensions=1024,
)


def vector_at(index: int) -> tuple[float, ...]:
    values = [0.0] * 1024
    values[index] = 1.0
    return tuple(values)


def fact(
    text: str,
    *,
    occurred_start: datetime | None = None,
) -> ExtractedFact:
    return ExtractedFact(
        text=text,
        occurred_start=occurred_start,
        occurred_end=occurred_start,
        entities=("测试",),
    )


async def insert_dynamic_item_projection(
    pool: asyncpg.Pool,
    *,
    fish_id: UUID,
    session_id: UUID,
    support_fact_ids: tuple[UUID, ...],
    summary: str,
    vector: tuple[float, ...],
    snapshot_version: int = 1,
) -> tuple[int, int]:
    snapshot_id = cast(
        int,
        await pool.fetchval(
            """
            INSERT INTO memory_snapshots (
                fish_id, session_id, snapshot_type, content,
                through_fact_sequence, version
            )
            VALUES ($1, $2, 'dynamic', NULL, $3, $4)
            RETURNING id
            """,
            fish_id,
            session_id,
            len(support_fact_ids),
            snapshot_version,
        ),
    )
    revision_id = cast(
        int,
        await pool.fetchval(
            """
            INSERT INTO memory_snapshot_revisions (
                snapshot_id, version, through_fact_sequence
            )
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            snapshot_id,
            snapshot_version,
            len(support_fact_ids),
        ),
    )
    item_id = cast(
        int,
        await pool.fetchval(
            """
            INSERT INTO memory_snapshot_items (
                snapshot_id, summary, support_fact_ids,
                lineage_fact_ids, last_supported_at
            )
            VALUES ($1, $2, $3::uuid[], $3::uuid[], now())
            RETURNING id
            """,
            snapshot_id,
            summary,
            list(support_fact_ids),
        ),
    )
    await pool.execute(
        """
        INSERT INTO memory_snapshot_item_versions (
            snapshot_item_id, snapshot_revision_id, snapshot_version,
            item_version, action, basis, summary, support_fact_ids,
            lineage_fact_ids, new_support_fact_ids,
            removed_support_fact_ids, last_supported_at
        )
        SELECT
            item.id, $2, $3, item.item_version, 'create', NULL,
            item.summary, item.support_fact_ids, item.lineage_fact_ids,
            item.support_fact_ids, '{}'::uuid[], item.last_supported_at
        FROM memory_snapshot_items AS item
        WHERE item.id = $1
        """,
        item_id,
        revision_id,
        snapshot_version,
    )
    await pool.execute(
        """
        INSERT INTO memory_snapshot_item_embeddings (
            snapshot_item_id, item_version, fish_id, session_id,
            snapshot_type, snapshot_text, text_hash, last_supported_at,
            provider, model, dimensions, embedding
        )
        VALUES (
            $1, 1, $2, $3, 'dynamic', $4, $5, now(),
            'diagnostic-provider', 'diagnostic-model', 1024, $6::vector
        )
        """,
        item_id,
        fish_id,
        session_id,
        summary,
        hashlib.sha256(summary.encode()).hexdigest(),
        "[" + ",".join(str(value) for value in vector) + "]",
    )
    return snapshot_id, item_id


async def insert_recall_facts(
    pool: asyncpg.Pool,
    *,
    fish_id: UUID,
    rows: tuple[tuple[str, tuple[float, ...]], ...],
) -> tuple[UUID, tuple[UUID, ...]]:
    messages = MessagesStore(pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(pool)
    sources = [
        await messages.insert_user_message(fish_id, text) for text, _vector in rows
    ]
    await messages.insert_user_message(fish_id, "sentinel")
    for source, (text, vector) in zip(sources, rows, strict=True):
        pending = next(
            item
            for item in await facts.list_pending_messages(limit=20)
            if item.message.id == source.id
        )
        await facts.persist_extraction(
            pending,
            [fact(text)],
            [vector],
            EMBEDDING_SPACE,
        )
    records = await pool.fetch(
        """
        SELECT id, session_id
        FROM facts
        WHERE fish_id = $1
        ORDER BY sequence
        """,
        fish_id,
    )
    return (
        cast(UUID, records[0]["session_id"]),
        tuple(cast(UUID, record["id"]) for record in records),
    )


@pytest.mark.asyncio
async def test_resolves_batch_query_entities_by_shared_entity_id(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    await messages.insert_user_message(FIRST_FISH_ID, "seed")
    recall = RecallStore(postgres_pool)
    wolf_id = UUID(int=71)
    await postgres_pool.executemany(
        """
        INSERT INTO entity_aliases (fish_id, alias, entity_id)
        VALUES ($1, $2, $3)
        """,
        [
            (FIRST_FISH_ID, "白狼", wolf_id),
            (FIRST_FISH_ID, "杰洛特", wolf_id),
        ],
    )

    entities = await recall.resolve_query_entities_batch(
        FIRST_FISH_ID,
        ("白狼以前用过什么剑", "杰洛特以前用过什么剑", ""),
    )

    assert len(entities) == 3
    for resolved in entities[:2]:
        assert len(resolved) == 1
        assert resolved[0].entity_id == wolf_id
        assert set(resolved[0].aliases) == {"白狼", "杰洛特"}
    assert entities[2] == ()


@pytest.mark.asyncio
async def test_hnsw_semantic_search_orders_and_isolates_fish(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    recall = RecallStore(postgres_pool)

    first_source = await messages.insert_user_message(FIRST_FISH_ID, "first")
    second_source = await messages.insert_user_message(FIRST_FISH_ID, "second")
    await messages.insert_user_message(FIRST_FISH_ID, "latest")
    other_source = await messages.insert_user_message(SECOND_FISH_ID, "other")
    await messages.insert_user_message(SECOND_FISH_ID, "other latest")

    first_pending = next(
        item
        for item in await facts.list_pending_messages(limit=10)
        if item.message.id == first_source.id
    )
    await facts.persist_extraction(
        first_pending,
        [fact("最相似事实")],
        [vector_at(0)],
        EMBEDDING_SPACE,
    )
    second_pending = next(
        item
        for item in await facts.list_pending_messages(limit=10)
        if item.message.id == second_source.id
    )
    await facts.persist_extraction(
        second_pending,
        [
            fact(
                "较远事实",
                occurred_start=datetime(2025, 8, 3, tzinfo=UTC),
            )
        ],
        [vector_at(1)],
        EMBEDDING_SPACE,
    )
    other_pending = next(
        item
        for item in await facts.list_pending_messages(limit=10)
        if item.message.id == other_source.id
    )
    await facts.persist_extraction(
        other_pending,
        [fact("其他 fish 的同向量事实")],
        [vector_at(0)],
        EMBEDDING_SPACE,
    )

    results = await recall.semantic_search(
        FIRST_FISH_ID,
        EMBEDDING_SPACE,
        vector_at(0),
        limit=20,
    )

    assert [item.text for item in results] == ["最相似事实", "较远事实"]
    assert results[0].semantic_score == pytest.approx(1.0)
    assert all("其他 fish" not in item.text for item in results)

    time_results = await recall.time_search(
        FIRST_FISH_ID,
        RecallTimeFilter(
            field=RecallTimeField.OCCURRED,
            start=datetime(2025, 8, 1, tzinfo=UTC),
            end=datetime(2025, 8, 31, 23, 59, 59, tzinfo=UTC),
        ),
        RecallOrder.EARLIEST,
        limit=20,
    )
    assert [item.text for item in time_results] == ["较远事实"]


@pytest.mark.asyncio
async def test_time_search_keeps_low_similarity_earliest_fact(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    recall = RecallStore(postgres_pool)

    first_sources = [
        await messages.insert_user_message(FIRST_FISH_ID, content)
        for content in ("early", "middle", "late")
    ]
    await messages.insert_user_message(FIRST_FISH_ID, "latest sentinel")
    other_source = await messages.insert_user_message(SECOND_FISH_ID, "other early")
    await messages.insert_user_message(SECOND_FISH_ID, "other sentinel")

    times = (
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
        datetime(2022, 1, 1, tzinfo=UTC),
    )
    vectors = (vector_at(3), vector_at(0), vector_at(0))
    for source, occurred_at, vector in zip(
        first_sources,
        times,
        vectors,
        strict=True,
    ):
        pending = next(
            item
            for item in await facts.list_pending_messages(limit=10)
            if item.message.id == source.id
        )
        await facts.persist_extraction(
            pending,
            [fact(source.content, occurred_start=occurred_at)],
            [vector],
            EMBEDDING_SPACE,
        )

    other_pending = next(
        item
        for item in await facts.list_pending_messages(limit=10)
        if item.message.id == other_source.id
    )
    await facts.persist_extraction(
        other_pending,
        [fact("other early", occurred_start=datetime(2019, 1, 1, tzinfo=UTC))],
        [vector_at(0)],
        EMBEDDING_SPACE,
    )

    results = await recall.time_search(
        FIRST_FISH_ID,
        RecallTimeFilter(
            field=RecallTimeField.OCCURRED,
            start=None,
            end=None,
        ),
        RecallOrder.EARLIEST,
        limit=10,
    )

    assert [item.text for item in results] == ["early", "middle", "late"]
    assert all(item.text != "other early" for item in results)


@pytest.mark.asyncio
async def test_relation_first_heads_map_facts_and_persist_masks(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await ensure_embedding_hnsw_index(postgres_pool, 1024)
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    recall = RecallStore(postgres_pool)

    first = await messages.insert_user_message(FIRST_FISH_ID, "白狼使用过银剑")
    second = await messages.insert_user_message(FIRST_FISH_ID, "银剑放在仓库")
    await messages.insert_user_message(FIRST_FISH_ID, "sentinel")
    for source, extracted, vector in (
        (first, fact("杰洛特以前使用过银剑"), vector_at(0)),
        (second, fact("银剑放在仓库"), vector_at(1)),
    ):
        pending = next(
            item
            for item in await facts.list_pending_messages(limit=10)
            if item.message.id == source.id
        )
        await facts.persist_extraction(
            pending,
            [extracted],
            [vector],
            EMBEDDING_SPACE,
        )

    rows = await postgres_pool.fetch(
        "SELECT id, session_id, text FROM facts WHERE fish_id = $1 ORDER BY sequence",
        FIRST_FISH_ID,
    )
    fact_ids = tuple(cast(UUID, row["id"]) for row in rows)
    session_id = cast(UUID, rows[0]["session_id"])
    summary = "白狼的银剑使用与存放信息"
    _snapshot_id, item_id = await insert_dynamic_item_projection(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        session_id=session_id,
        support_fact_ids=fact_ids,
        summary=summary,
        vector=vector_at(2),
        snapshot_version=4,
    )

    fact_head = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    assert fact_head[0].fact.id == fact_ids[0]
    assert fact_head[0].relation_item_ids == (item_id,)
    assert fact_head[0].embedding_text == "杰洛特以前使用过银剑 测试"

    relation_head = await recall.search_relation_head(
        FIRST_FISH_ID,
        1024,
        vector_at(2),
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    assert relation_head[0].id == item_id
    assert relation_head[0].support_fact_ids == fact_ids
    assert relation_head[0].support_count == 2
    assert relation_head[0].snapshot_version == 4

    mask = MaskEmbedding(
        object_id=fact_ids[0],
        entity_ids=(UUID(int=99),),
        signature_hash="a" * 64,
        source_text_hash=cast(str, fact_head[0].text_hash),
        masked_text="以前使用过银剑",
        vector=vector_at(3),
    )
    await recall.upsert_fact_mask_embeddings(
        FIRST_FISH_ID,
        EMBEDDING_SPACE,
        [mask],
    )
    assert await recall.load_fact_mask_embeddings(
        FIRST_FISH_ID,
        1024,
        [fact_ids[0]],
    ) == [mask]


@pytest.mark.asyncio
async def test_heads_filter_source_role_without_trimming_relation_members(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await ensure_embedding_hnsw_index(postgres_pool, 1024)
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    user_source = await messages.insert_user_message(FIRST_FISH_ID, "用户原文")
    assistant_source = await messages.insert_generating_answer(
        FIRST_FISH_ID,
        user_source.session_id,
        user_source.sequence,
    )
    await messages.finish_answer(
        assistant_source.id,
        FIRST_FISH_ID,
        content="助手原文",
        requested_status=MessageStatus.COMPLETED,
    )
    await messages.insert_user_message(FIRST_FISH_ID, "sentinel")
    for text, vector in (("用户事实", vector_at(0)), ("助手事实", vector_at(1))):
        pending = (await facts.list_pending_messages(limit=10))[0]
        await facts.persist_extraction(
            pending,
            [fact(text)],
            [vector],
            EMBEDDING_SPACE,
        )
    rows = await postgres_pool.fetch(
        "SELECT id FROM facts WHERE fish_id = $1 ORDER BY sequence",
        FIRST_FISH_ID,
    )
    fact_ids = tuple(cast(UUID, row["id"]) for row in rows)
    _snapshot_id, mixed_relation = await insert_dynamic_item_projection(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        session_id=user_source.session_id,
        support_fact_ids=fact_ids,
        summary="混合来源关系",
        vector=vector_at(2),
    )
    recall = RecallStore(postgres_pool)

    user_head = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        source_role=RecallSourceRole.USER,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    assistant_head = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(1),
        source_role=RecallSourceRole.ASSISTANT,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    either_head = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        source_role=RecallSourceRole.EITHER,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )

    assert [row.fact.source_role for row in user_head] == [Role.USER]
    assert [row.fact.source_role for row in assistant_head] == [Role.ASSISTANT]
    assert {row.fact.source_role for row in either_head} == {
        Role.USER,
        Role.ASSISTANT,
    }

    user_relation_head = await recall.search_relation_head(
        FIRST_FISH_ID,
        1024,
        vector_at(2),
        source_role=RecallSourceRole.USER,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    assistant_relation_head = await recall.search_relation_head(
        FIRST_FISH_ID,
        1024,
        vector_at(2),
        source_role=RecallSourceRole.ASSISTANT,
        time_range=None,
        limit=50,
        use_hnsw=True,
    )

    assert mixed_relation in {row.id for row in user_relation_head}
    assert mixed_relation in {row.id for row in assistant_relation_head}

    members = await recall.load_fact_keys(
        FIRST_FISH_ID,
        fact_ids,
        1024,
        vector_at(2),
        time_range=None,
    )
    assert {row.fact.source_role for row in members} == {Role.USER, Role.ASSISTANT}


@pytest.mark.asyncio
async def test_production_range_uses_mentioned_or_occurred_for_facts_and_relations(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches either legal time branch or Relation completeness being narrowed."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    await ensure_embedding_hnsw_index(postgres_pool, 1024)
    session_id, fact_ids = await insert_recall_facts(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        rows=(
            ("用户购买奢华晚礼服花费800美元", vector_at(0)),
            ("Ava 和 Lily 出生于四月", vector_at(1)),
            ("范围内发生、范围外提及", vector_at(2)),
            ("过去范围外事实", vector_at(3)),
            ("未来范围外事实", vector_at(4)),
        ),
    )
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    context_sequence = cast(
        int,
        await postgres_pool.fetchval(
            "SELECT max(sequence) FROM messages WHERE session_id = $1",
            session_id,
        ),
    )
    assistant_source = await messages.insert_generating_answer(
        FIRST_FISH_ID,
        session_id,
        context_sequence,
    )
    await messages.finish_answer(
        assistant_source.id,
        FIRST_FISH_ID,
        content="助手在范围内提及",
        requested_status=MessageStatus.COMPLETED,
    )
    await messages.insert_user_message(
        FIRST_FISH_ID,
        "assistant sentinel",
        session_id=session_id,
    )
    old_sentinel = (await facts.list_pending_messages(limit=20))[0]
    assert old_sentinel.message.content == "sentinel"
    await facts.persist_extraction(old_sentinel, [], [], EMBEDDING_SPACE)
    assistant_pending = next(
        item
        for item in await facts.list_pending_messages(limit=20)
        if item.message.id == assistant_source.id
    )
    await facts.persist_extraction(
        assistant_pending,
        [fact("助手在范围内提及")],
        [vector_at(5)],
        EMBEDDING_SPACE,
    )
    assistant_fact_id = cast(
        UUID,
        await postgres_pool.fetchval(
            "SELECT id FROM facts WHERE source_message_id = $1",
            assistant_source.id,
        ),
    )
    fact_ids = (*fact_ids, assistant_fact_id)
    inside_start = datetime(2026, 4, 1, tzinfo=UTC)
    inside_end = datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC)
    inside_may = datetime(2026, 5, 1, tzinfo=UTC)
    past = datetime(2024, 1, 1, tzinfo=UTC)
    future = datetime(2027, 1, 1, tzinfo=UTC)
    await postgres_pool.executemany(
        """
        UPDATE facts
        SET mentioned_at = $2,
            occurred_start = $3,
            occurred_end = $3
        WHERE id = $1
        """,
        [
            (fact_ids[0], inside_may, None),
            (fact_ids[1], datetime(2026, 5, 2, tzinfo=UTC), None),
            (fact_ids[2], past, datetime(2026, 5, 10, tzinfo=UTC)),
            (fact_ids[3], past, past),
            (fact_ids[4], future, future),
            (fact_ids[5], datetime(2026, 5, 3, tzinfo=UTC), None),
        ],
    )
    _complete_snapshot, complete_relation = await insert_dynamic_item_projection(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        session_id=session_id,
        support_fact_ids=(fact_ids[0], fact_ids[2]),
        summary="提及或发生均在范围内",
        vector=vector_at(10),
    )
    recall = RecallStore(postgres_pool)
    bounded = RecallTimeRange(start=inside_start, end=inside_end)

    bounded_head = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        source_role=RecallSourceRole.USER,
        time_range=bounded,
        limit=50,
        use_hnsw=False,
    )
    bounded_ids = {row.fact.id for row in bounded_head}
    assert bounded_ids == {fact_ids[0], fact_ids[1], fact_ids[2]}
    assert fact_ids[5] not in bounded_ids
    assert (
        await recall.count_time_scoped_facts(
            FIRST_FISH_ID,
            1024,
            bounded,
            source_role=RecallSourceRole.USER,
            stop_after=50,
        )
        == 3
    )

    start_only = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        source_role=RecallSourceRole.USER,
        time_range=RecallTimeRange(start=inside_start, end=None),
        limit=50,
        use_hnsw=False,
    )
    assert fact_ids[0] in {row.fact.id for row in start_only}
    assert fact_ids[2] in {row.fact.id for row in start_only}
    assert fact_ids[3] not in {row.fact.id for row in start_only}

    end_only = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        source_role=RecallSourceRole.USER,
        time_range=RecallTimeRange(start=None, end=inside_end),
        limit=50,
        use_hnsw=False,
    )
    assert fact_ids[0] in {row.fact.id for row in end_only}
    assert fact_ids[2] in {row.fact.id for row in end_only}
    assert fact_ids[4] not in {row.fact.id for row in end_only}

    relation_head = await recall.search_relation_head(
        FIRST_FISH_ID,
        1024,
        vector_at(10),
        source_role=RecallSourceRole.USER,
        time_range=bounded,
        limit=50,
        use_hnsw=False,
    )
    relation_ids = {row.id for row in relation_head}
    assert complete_relation in relation_ids

    await postgres_pool.execute(
        """
        UPDATE facts
        SET occurred_start = $2, occurred_end = $2
        WHERE id = $1
        """,
        fact_ids[2],
        past,
    )
    partial_relation_head = await recall.search_relation_head(
        FIRST_FISH_ID,
        1024,
        vector_at(10),
        source_role=RecallSourceRole.USER,
        time_range=bounded,
        limit=50,
        use_hnsw=False,
    )
    assert complete_relation not in {row.id for row in partial_relation_head}

    anchored = await recall.load_relation_item_keys(
        FIRST_FISH_ID,
        [complete_relation],
        1024,
        vector_at(10),
        time_range=bounded,
    )
    assert [row.id for row in anchored] == [complete_relation]
    partial_members = await recall.load_fact_keys(
        FIRST_FISH_ID,
        [fact_ids[0], fact_ids[2]],
        1024,
        vector_at(10),
        time_range=bounded,
    )
    assert [row.fact.id for row in partial_members] == [fact_ids[0]]


@pytest.mark.asyncio
async def test_stale_item_projection_and_removed_support_are_not_expanded(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await ensure_embedding_hnsw_index(postgres_pool, 1024)
    session_id, fact_ids = await insert_recall_facts(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        rows=(("第一事实", vector_at(0)), ("第二事实", vector_at(1))),
    )
    snapshot_id, item_id = await insert_dynamic_item_projection(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        session_id=session_id,
        support_fact_ids=fact_ids,
        summary="两条事实的当前摘要",
        vector=vector_at(2),
    )
    recall = RecallStore(postgres_pool)

    initial = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(1),
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    assert next(
        row for row in initial if row.fact.id == fact_ids[1]
    ).relation_item_ids == (item_id,)

    await postgres_pool.execute(
        "UPDATE facts SET mentioned_at = $2 WHERE id = $1",
        fact_ids[1],
        datetime(2020, 1, 1, tzinfo=UTC),
    )
    partial_scope_head = await recall.search_relation_head(
        FIRST_FISH_ID,
        1024,
        vector_at(2),
        time_range=RecallTimeRange(
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=None,
        ),
        limit=50,
        use_hnsw=False,
    )
    assert partial_scope_head == []

    await postgres_pool.execute(
        "UPDATE memory_snapshots SET version = 2 WHERE id = $1",
        snapshot_id,
    )
    revision_id = cast(
        int,
        await postgres_pool.fetchval(
            """
            INSERT INTO memory_snapshot_revisions (
                snapshot_id, version, through_fact_sequence
            )
            VALUES ($1, 2, 2)
            RETURNING id
            """,
            snapshot_id,
        ),
    )
    await postgres_pool.execute(
        """
        UPDATE memory_snapshot_items
        SET support_fact_ids = ARRAY[$2]::uuid[],
            item_version = 2,
            updated_at = date_trunc('second', now())
        WHERE id = $1
        """,
        item_id,
        fact_ids[0],
    )
    await postgres_pool.execute(
        """
        INSERT INTO memory_snapshot_item_versions (
            snapshot_item_id, snapshot_revision_id, snapshot_version,
            item_version, action, basis, summary, support_fact_ids,
            lineage_fact_ids, new_support_fact_ids,
            removed_support_fact_ids, last_supported_at
        )
        SELECT
            item.id, $2, 2, item.item_version, 'update', '移除旧支持',
            item.summary, item.support_fact_ids, item.lineage_fact_ids,
            '{}'::uuid[], ARRAY[$3]::uuid[], item.last_supported_at
        FROM memory_snapshot_items AS item
        WHERE item.id = $1
        """,
        item_id,
        revision_id,
        fact_ids[1],
    )

    stale_fact_head = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    assert all(not row.relation_item_ids for row in stale_fact_head)
    assert (
        await recall.search_relation_head(
            FIRST_FISH_ID,
            1024,
            vector_at(2),
            time_range=None,
            limit=50,
            use_hnsw=True,
        )
        == []
    )

    await postgres_pool.execute(
        """
        UPDATE memory_snapshot_item_embeddings
        SET item_version = 2
        WHERE snapshot_item_id = $1
        """,
        item_id,
    )
    current_fact_head = await recall.search_fact_head(
        FIRST_FISH_ID,
        1024,
        vector_at(0),
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    by_id = {row.fact.id: row for row in current_fact_head}
    assert by_id[fact_ids[0]].relation_item_ids == (item_id,)
    assert by_id[fact_ids[1]].relation_item_ids == ()

    relation_head = await recall.search_relation_head(
        FIRST_FISH_ID,
        1024,
        vector_at(2),
        time_range=None,
        limit=50,
        use_hnsw=True,
    )
    assert relation_head[0].support_fact_ids == (fact_ids[0],)
    assert relation_head[0].support_count == 1
    assert relation_head[0].snapshot_version == 2

    removed_fact = await recall.load_fact_keys(
        FIRST_FISH_ID,
        [fact_ids[1]],
        1024,
        vector_at(1),
        time_range=None,
    )
    assert [row.fact.id for row in removed_fact] == [fact_ids[1]]
    assert removed_fact[0].relation_item_ids == ()


@pytest.mark.asyncio
async def test_stale_relation_mask_cannot_reappear_after_full_update(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    session_id, fact_ids = await insert_recall_facts(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        rows=(("杰洛特以前使用过银剑", vector_at(0)),),
    )
    old_summary = "杰洛特以前使用过银剑"
    snapshot_id, item_id = await insert_dynamic_item_projection(
        postgres_pool,
        fish_id=FIRST_FISH_ID,
        session_id=session_id,
        support_fact_ids=fact_ids,
        summary=old_summary,
        vector=vector_at(1),
    )
    recall = RecallStore(postgres_pool)
    old_mask = MaskEmbedding(
        object_id=item_id,
        entity_ids=(UUID(int=99),),
        signature_hash="c" * 64,
        source_text_hash=hashlib.sha256(old_summary.encode()).hexdigest(),
        masked_text="以前使用过银剑",
        vector=vector_at(2),
    )

    async with postgres_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        await connection.fetchval(
            "SELECT id FROM memory_snapshot_items WHERE id = $1 FOR UPDATE",
            item_id,
        )
        await connection.execute(
            "UPDATE memory_snapshots SET version = 2 WHERE id = $1",
            snapshot_id,
        )
        revision_id = cast(
            int,
            await connection.fetchval(
                """
                INSERT INTO memory_snapshot_revisions (
                    snapshot_id, version, through_fact_sequence
                )
                VALUES ($1, 2, 1)
                RETURNING id
                """,
                snapshot_id,
            ),
        )
        new_summary = "银剑已经归还"
        await connection.execute(
            """
            UPDATE memory_snapshot_items
            SET summary = $2, item_version = 2
            WHERE id = $1
            """,
            item_id,
            new_summary,
        )
        await connection.execute(
            """
            INSERT INTO memory_snapshot_item_versions (
                snapshot_item_id, snapshot_revision_id, snapshot_version,
                item_version, action, basis, summary, support_fact_ids,
                lineage_fact_ids, new_support_fact_ids,
                removed_support_fact_ids, last_supported_at
            )
            SELECT
                item.id, $2, 2, item.item_version, 'update', '摘要更新',
                item.summary, item.support_fact_ids, item.lineage_fact_ids,
                '{}'::uuid[], '{}'::uuid[], item.last_supported_at
            FROM memory_snapshot_items AS item
            WHERE item.id = $1
            """,
            item_id,
            revision_id,
        )
        await connection.execute(
            """
            UPDATE memory_snapshot_item_embeddings
            SET item_version = 2,
                snapshot_text = $2,
                text_hash = $3,
                embedding = $4::vector
            WHERE snapshot_item_id = $1
            """,
            item_id,
            new_summary,
            hashlib.sha256(new_summary.encode()).hexdigest(),
            "[" + ",".join(str(value) for value in vector_at(3)) + "]",
        )
        stale_upsert = asyncio.create_task(
            recall.upsert_relation_mask_embeddings(
                FIRST_FISH_ID,
                EMBEDDING_SPACE,
                [old_mask],
            )
        )
        await asyncio.sleep(0.05)
        assert not stale_upsert.done()
        await transaction.commit()

    await stale_upsert
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*)
            FROM memory_snapshot_item_mask_embeddings
            WHERE snapshot_item_id = $1
            """,
            item_id,
        )
        == 0
    )

    current_hash = hashlib.sha256("银剑已经归还".encode()).hexdigest()
    current_mask = MaskEmbedding(
        object_id=item_id,
        entity_ids=(UUID(int=99),),
        signature_hash="c" * 64,
        source_text_hash=current_hash,
        masked_text="已经归还",
        vector=vector_at(3),
    )
    await recall.upsert_relation_mask_embeddings(
        FIRST_FISH_ID,
        EMBEDDING_SPACE,
        [current_mask],
    )
    assert await recall.load_relation_mask_embeddings(
        FIRST_FISH_ID,
        1024,
        [item_id],
    ) == [current_mask]
