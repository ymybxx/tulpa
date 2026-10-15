from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.db.fishes import FishStore
from fish_memory.db.memory_forget import MemoryForgetStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.embeddings.client import EmbeddingSpace

EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model="text-embedding-v4",
    dimensions=1024,
)


async def _seed(pool: asyncpg.Pool) -> tuple[UUID, dict[str, UUID], dict[str, int]]:
    await run_migrations(cast(MigrationPool, pool))
    fish_id = await FishStore(pool, embedding_space=EMBEDDING_SPACE).create(
        name="forget"
    )
    source = await MessagesStore(
        pool, embedding_space=EMBEDDING_SPACE
    ).insert_user_message(fish_id, "我不喝咖啡了，晚上写代码")
    facts = {name: uuid4() for name in ("coffee", "night", "code")}
    for index, (name, fact_id) in enumerate(facts.items()):
        await pool.execute(
            """
            INSERT INTO facts (
                id, fish_id, session_id, source_message_id,
                fact_index, text, mentioned_at, entities
            )
            VALUES ($1, $2, $3, $4, $5, $6, now(), '[]'::jsonb)
            """,
            fact_id,
            fish_id,
            source.session_id,
            source.id,
            index,
            name,
        )
    snapshot_id = await pool.fetchval(
        """
        INSERT INTO memory_snapshots (
            fish_id, session_id, snapshot_type, content, through_fact_sequence
        )
        VALUES ($1, NULL, 'static', NULL, 3)
        RETURNING id
        """,
        fish_id,
    )
    items: dict[str, int] = {}
    for name, support, lineage in (
        ("likes-coffee", ["coffee"], ["coffee"]),
        ("night-coder", ["night", "code"], ["night", "code", "coffee"]),
        ("codes", ["code"], ["code"]),
    ):
        items[name] = await pool.fetchval(
            """
            INSERT INTO memory_snapshot_items (
                snapshot_id, summary, support_fact_ids,
                lineage_fact_ids, last_supported_at
            )
            VALUES ($1, $2, $3::uuid[], $4::uuid[], now())
            RETURNING id
            """,
            snapshot_id,
            name,
            [facts[fact] for fact in support],
            [facts[fact] for fact in lineage],
        )
    await pool.execute(
        """
        INSERT INTO fact_relation_groups (
            fish_id, snapshot_type, fact_ids, snapshot_version
        )
        VALUES ($1, 'static', $2::uuid[], 1), ($1, 'static', $3::uuid[], 1)
        """,
        fish_id,
        [facts["coffee"], facts["night"]],
        [facts["code"]],
    )
    return fish_id, facts, items


@pytest.mark.asyncio
async def test_forgetting_an_impression_removes_its_facts_and_everything_built_on_them(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches a forgotten memory surviving in a sibling summary or a relation group."""
    fish_id, facts, items = await _seed(postgres_pool)
    store = MemoryForgetStore(postgres_pool)
    version = await postgres_pool.fetchval(
        "SELECT version FROM memory_snapshots WHERE fish_id = $1", fish_id
    )

    preview = await store.preview(fish_id, item_id=items["likes-coffee"])
    plan = await store.forget(fish_id, item_id=items["likes-coffee"])

    assert preview == plan
    assert plan is not None and (plan.fact_count, plan.item_count) == (1, 2)
    remaining_facts = {
        row["text"] for row in await postgres_pool.fetch("SELECT text FROM facts")
    }
    remaining_items = {
        row["summary"]
        for row in await postgres_pool.fetch(
            "SELECT summary FROM memory_snapshot_items"
        )
    }
    groups = await postgres_pool.fetch("SELECT fact_ids FROM fact_relation_groups")
    assert remaining_facts == {"night", "code"}
    assert remaining_items == {"codes"}
    assert [list(row["fact_ids"]) for row in groups] == [[facts["code"]]]
    assert (
        await postgres_pool.fetchval(
            "SELECT version FROM memory_snapshots WHERE fish_id = $1", fish_id
        )
        == version + 1
    )


@pytest.mark.asyncio
async def test_forgetting_is_scoped_to_the_fish_and_reports_missing_memories(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches one character's request deleting another character's memory."""
    fish_id, facts, items = await _seed(postgres_pool)
    other = await FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE).create(
        name="other"
    )
    store = MemoryForgetStore(postgres_pool)

    assert await store.forget(other, item_id=items["codes"]) is None
    assert await store.forget(other, fact_id=facts["code"]) is None
    plan = await store.forget(fish_id, fact_id=facts["code"])

    assert plan is not None and (plan.fact_count, plan.item_count) == (1, 2)
    assert await store.forget(fish_id, fact_id=facts["code"]) is None
    with pytest.raises(ValueError):
        await store.preview(fish_id)


@pytest.mark.asyncio
async def test_fish_keeps_background_and_its_own_wake_words(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches the new character fields being dropped on save or read."""
    await run_migrations(cast(MigrationPool, postgres_pool))
    fishes = FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    fish_id = await fishes.create(name="阿澄", background="住在电脑里")

    profile = await fishes.get_profile(fish_id)
    assert profile is not None and profile.background == "住在电脑里"
    assert await fishes.get_wake_words(fish_id) is None
    assert await fishes.update_wake_words(fish_id, ["阿澄", "小澄"])
    assert await fishes.get_wake_words(fish_id) == ("阿澄", "小澄")
    with pytest.raises(asyncpg.CheckViolationError):
        await fishes.update_wake_words(fish_id, ["a", "b", "c", "d", "e"])
