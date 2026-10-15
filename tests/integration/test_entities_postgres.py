import hashlib
from typing import cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.entities import EntityStore
from fish_memory.db.facts import FactStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.recall import RecallStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.entities.models import EntityMergeGroup
from fish_memory.facts.models import ExtractedFact
from fish_memory.recall.models import RecallOrder

FISH_ID = UUID(int=501)
VECTOR = (0.25,) * 1024
EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model="text-embedding-v4",
    dimensions=1024,
)


@pytest.mark.asyncio
async def test_entity_merge_unifies_aliases_and_existing_fact_links(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    entities = EntityStore(postgres_pool)
    await messages.insert_user_message(FISH_ID, "杰洛特")
    await messages.insert_user_message(FISH_ID, "大呆")
    await messages.insert_user_message(FISH_ID, "latest")

    first = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        first,
        [ExtractedFact("助手叫杰洛特", None, None, ("杰洛特",))],
        [VECTOR],
        EMBEDDING_SPACE,
    )
    second = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        second,
        [ExtractedFact("助手改名为大呆", None, None, ("大呆",))],
        [VECTOR],
        EMBEDDING_SPACE,
    )
    fact_ids = tuple(
        row["id"]
        for row in await postgres_pool.fetch(
            "SELECT id FROM facts WHERE fish_id = $1 ORDER BY sequence",
            FISH_ID,
        )
    )

    before = await entities.load_context(FISH_ID, fact_ids)
    assert len(before.candidates) == 2

    await entities.merge_groups(
        FISH_ID,
        (EntityMergeGroup(("杰洛特", "大呆")),),
    )

    after = await entities.load_context(FISH_ID, fact_ids)
    assert len(after.candidates) == 1
    assert set(after.candidates[0].aliases) == {"杰洛特", "大呆"}
    assert after.fact_entity_indexes == {
        fact_ids[0]: (0,),
        fact_ids[1]: (0,),
    }

    summaries = await entities.list_summaries(FISH_ID)
    assert summaries[0].entity_id == after.candidates[0].id
    assert set(summaries[0].aliases) == {"杰洛特", "大呆"}
    assert summaries[0].fact_count == 2
    page = await entities.list_facts(FISH_ID, summaries[0].entity_id, limit=1)
    assert len(page.items) == 1
    assert page.has_more is True

    recalled = await RecallStore(postgres_pool).entity_search(
        FISH_ID,
        ("杰洛特",),
        EMBEDDING_SPACE,
        VECTOR,
        order=RecallOrder.RELEVANCE,
        limit=10,
    )
    assert {fact.id for fact in recalled} == set(fact_ids)


@pytest.mark.asyncio
async def test_load_context_keeps_entity_links_for_invalidated_facts(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    entities = EntityStore(postgres_pool)

    await messages.insert_user_message(FISH_ID, "弱弱")
    await messages.insert_user_message(FISH_ID, "latest")
    pending = (await facts.list_pending_messages(limit=10))[0]
    await facts.persist_extraction(
        pending,
        [ExtractedFact("用户曾将助手称为弱弱", None, None, ("弱弱",))],
        [VECTOR],
        EMBEDDING_SPACE,
    )
    fact_id = cast(
        UUID,
        await postgres_pool.fetchval(
            "SELECT id FROM facts WHERE fish_id = $1",
            FISH_ID,
        ),
    )
    await postgres_pool.execute(
        "UPDATE facts SET invalidated_at = now() WHERE id = $1",
        fact_id,
    )

    context = await entities.load_context(FISH_ID, (fact_id,))

    assert context.candidates[0].aliases == ("弱弱",)
    assert context.fact_entity_indexes == {fact_id: (0,)}


@pytest.mark.asyncio
async def test_relation_single_masks_use_only_current_support_fact_entities(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    facts = FactStore(postgres_pool)
    first = await messages.insert_user_message(FISH_ID, "杰洛特")
    second = await messages.insert_user_message(FISH_ID, "希尔")
    await messages.insert_user_message(FISH_ID, "latest")
    for source, extracted in (
        (
            first,
            ExtractedFact("杰洛特使用过银剑", None, None, ("杰洛特",)),
        ),
        (
            second,
            ExtractedFact("希尔使用过银剑", None, None, ("希尔",)),
        ),
    ):
        pending = next(
            item
            for item in await facts.list_pending_messages(limit=10)
            if item.message.id == source.id
        )
        await facts.persist_extraction(
            pending,
            [extracted],
            [VECTOR],
            EMBEDDING_SPACE,
        )
    rows = await postgres_pool.fetch(
        "SELECT id, session_id FROM facts WHERE fish_id = $1 ORDER BY sequence",
        FISH_ID,
    )
    current_fact_id = cast(UUID, rows[0]["id"])
    removed_fact_id = cast(UUID, rows[1]["id"])
    session_id = cast(UUID, rows[0]["session_id"])
    snapshot_id = cast(
        int,
        await postgres_pool.fetchval(
            """
            INSERT INTO memory_snapshots (
                fish_id, session_id, snapshot_type, content,
                through_fact_sequence, version
            )
            VALUES ($1, $2, 'dynamic', NULL, 2, 1)
            RETURNING id
            """,
            FISH_ID,
            session_id,
        ),
    )
    revision_id = cast(
        int,
        await postgres_pool.fetchval(
            """
            INSERT INTO memory_snapshot_revisions (
                snapshot_id, version, through_fact_sequence
            )
            VALUES ($1, 1, 2)
            RETURNING id
            """,
            snapshot_id,
        ),
    )
    summary = "杰洛特和希尔都使用过银剑"
    item_id = cast(
        int,
        await postgres_pool.fetchval(
            """
            INSERT INTO memory_snapshot_items (
                snapshot_id, summary, support_fact_ids,
                lineage_fact_ids, last_supported_at
            )
            VALUES ($1, $2, ARRAY[$3]::uuid[], ARRAY[$3, $4]::uuid[], now())
            RETURNING id
            """,
            snapshot_id,
            summary,
            current_fact_id,
            removed_fact_id,
        ),
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
            item.id, $2, 1, 1, 'create', NULL, item.summary,
            item.support_fact_ids, item.lineage_fact_ids,
            item.support_fact_ids, '{}'::uuid[], item.last_supported_at
        FROM memory_snapshot_items AS item
        WHERE item.id = $1
        """,
        item_id,
        revision_id,
    )
    await postgres_pool.execute(
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
        FISH_ID,
        session_id,
        summary,
        hashlib.sha256(summary.encode()).hexdigest(),
        "[" + ",".join(str(value) for value in VECTOR) + "]",
    )

    sources = await RecallStore(postgres_pool).list_relation_single_mask_sources(
        1024,
        100,
    )

    assert [(source.object_id, source.entity.aliases) for source in sources] == [
        (item_id, ("杰洛特",))
    ]
