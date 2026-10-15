import hashlib
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.db.facts import FactStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.facts.models import ExtractedFact, PendingFactMessage
from fish_memory.models import MessageStatus, visible_message_content

FIRST_FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SECOND_FISH_ID = UUID("00000000-0000-0000-0000-000000000002")
EVALUATION_OWNER = UUID("00000000-0000-0000-0000-000000000003")
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model=EMBEDDING_MODEL,
    dimensions=1024,
)
VECTOR = (0.25,) * 1024


async def migrated_stores(
    pool: asyncpg.Pool,
) -> tuple[MessagesStore, FactStore]:
    await run_migrations(cast(MigrationPool, pool))
    return MessagesStore(pool, embedding_space=EMBEDDING_SPACE), FactStore(pool)


def extracted_fact(
    text: str,
    *,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
) -> ExtractedFact:
    return ExtractedFact(
        text=text,
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        entities=("主人", "上海"),
    )


async def persist_extraction(
    store: FactStore,
    pending: PendingFactMessage,
    extracted: list[ExtractedFact] | tuple[ExtractedFact, ...],
) -> None:
    await store.persist_extraction(
        pending,
        extracted,
        [VECTOR] * len(extracted),
        EMBEDDING_SPACE,
    )


@pytest.mark.asyncio
async def test_persisted_facts_register_normalized_entity_links(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    await messages.insert_user_message(FIRST_FISH_ID, "stable")
    await messages.insert_user_message(FIRST_FISH_ID, "latest")
    pending = (await facts.list_pending_messages(limit=10))[0]

    await persist_extraction(
        facts,
        pending,
        [
            ExtractedFact(
                text="用户提到了杰洛特",
                occurred_start=None,
                occurred_end=None,
                entities=("  ＧerAlt  ", "杰洛特"),
            )
        ],
    )

    rows = await postgres_pool.fetch(
        """
        SELECT alias.alias, link.fact_id
        FROM entity_aliases AS alias
        JOIN fact_entities AS link ON link.entity_id = alias.entity_id
        WHERE alias.fish_id = $1
        ORDER BY alias.alias
        """,
        FIRST_FISH_ID,
    )
    # PostgreSQL collations can order Chinese and Latin aliases differently.
    assert {(row["alias"], row["fact_id"]) for row in rows} == {
        ("geralt", rows[0]["fact_id"]),
        ("杰洛特", rows[0]["fact_id"]),
    }


@pytest.mark.asyncio
async def test_fact_batch_persists_distinct_embedding_sources_per_fact(
    postgres_pool: asyncpg.Pool,
) -> None:
    """Catches PostgreSQL rows receiving the final Fact's key text as a batch."""
    messages, facts = await migrated_stores(postgres_pool)
    await messages.insert_user_message(FIRST_FISH_ID, "two distinct facts")
    await messages.insert_user_message(FIRST_FISH_ID, "latest")
    pending = (await facts.list_pending_messages(limit=10))[0]
    first = ExtractedFact(
        text="用户仍需领取第一件物品",
        occurred_start=None,
        occurred_end=None,
        entities=("第一件物品",),
        what="仍需领取第一件物品",
    )
    second = ExtractedFact(
        text="用户曾在会议上使用第二件物品",
        occurred_start=None,
        occurred_end=None,
        entities=("第二件物品",),
        what="曾在会议上使用第二件物品",
    )

    await facts.persist_extraction(
        pending,
        [first, second],
        [VECTOR, VECTOR],
        EMBEDDING_SPACE,
    )

    rows = await postgres_pool.fetch(
        """
        SELECT fact.fact_index, embedding.embedding_text, embedding.text_hash
        FROM facts AS fact
        JOIN fact_embeddings AS embedding ON embedding.fact_id = fact.id
        WHERE fact.source_message_id = $1
        ORDER BY fact.fact_index
        """,
        pending.message.id,
    )
    expected_texts = ["仍需领取第一件物品", "曾在会议上使用第二件物品"]
    assert [row["embedding_text"] for row in rows] == expected_texts
    assert [row["text_hash"] for row in rows] == [
        hashlib.sha256(text.encode()).hexdigest() for text in expected_texts
    ]


@pytest.mark.asyncio
async def test_candidates_are_one_non_latest_message_per_active_fish(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    first_oldest = await messages.insert_user_message(FIRST_FISH_ID, "first oldest")
    first_latest = await messages.insert_user_message(FIRST_FISH_ID, "first latest")
    second_oldest = await messages.insert_user_message(
        SECOND_FISH_ID,
        "second oldest",
    )
    second_latest = await messages.insert_user_message(
        SECOND_FISH_ID,
        "second latest",
    )

    candidates = await facts.list_pending_messages(limit=10)

    assert {candidate.message.fish_id for candidate in candidates} == {
        FIRST_FISH_ID,
        SECOND_FISH_ID,
    }
    assert {candidate.message.id for candidate in candidates} == {
        first_oldest.id,
        second_oldest.id,
    }
    assert first_latest.id not in {item.message.id for item in candidates}
    assert second_latest.id not in {item.message.id for item in candidates}

    first_candidate = next(
        item for item in candidates if item.message.fish_id == FIRST_FISH_ID
    )
    assert first_candidate.target_sequence == first_oldest.sequence
    await persist_extraction(facts, first_candidate, ())
    first_newest = await messages.insert_user_message(FIRST_FISH_ID, "first newest")
    await messages.soft_delete_fish(SECOND_FISH_ID)

    candidates = await facts.list_pending_messages(limit=10)

    assert [candidate.message.id for candidate in candidates] == [first_latest.id]
    assert candidates[0].target_sequence == first_latest.sequence
    assert first_newest.id not in {item.message.id for item in candidates}
    assert all(candidate.message.fish_id != SECOND_FISH_ID for candidate in candidates)


@pytest.mark.asyncio
async def test_fact_candidates_are_isolated_by_background_owner(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    normal_facts = FactStore(postgres_pool)
    evaluation_facts = FactStore(
        postgres_pool,
        background_owner=EVALUATION_OWNER,
    )
    normal_source = await messages.insert_user_message(FIRST_FISH_ID, "normal stable")
    await messages.insert_user_message(FIRST_FISH_ID, "normal latest")
    evaluation_source = await messages.insert_user_message(
        SECOND_FISH_ID,
        "evaluation stable",
    )
    await messages.insert_user_message(SECOND_FISH_ID, "evaluation latest")
    await postgres_pool.execute(
        "UPDATE fishes SET background_owner = $2 WHERE id = $1",
        SECOND_FISH_ID,
        EVALUATION_OWNER,
    )
    await postgres_pool.executemany(
        """
        INSERT INTO facts (
            id, fish_id, session_id, source_message_id, fact_index,
            text, mentioned_at, entities
        )
        VALUES ($1, $2, $3, $4, 0, $5, now(), '[]'::jsonb)
        """,
        [
            (
                uuid4(),
                FIRST_FISH_ID,
                normal_source.session_id,
                normal_source.id,
                "normal fact",
            ),
            (
                uuid4(),
                SECOND_FISH_ID,
                evaluation_source.session_id,
                evaluation_source.id,
                "evaluation fact",
            ),
        ],
    )

    normal_candidates = await normal_facts.list_pending_messages(limit=10)
    evaluation_candidates = await evaluation_facts.list_pending_messages(limit=10)

    assert [item.message.fish_id for item in normal_candidates] == [FIRST_FISH_ID]
    assert [item.message.fish_id for item in evaluation_candidates] == [SECOND_FISH_ID]


@pytest.mark.asyncio
async def test_fact_candidates_exclude_legacy_fish(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    current_fish_id = uuid4()
    legacy_fish_id = uuid4()
    current_source = await messages.insert_user_message(current_fish_id, "current")
    await messages.insert_user_message(current_fish_id, "latest")
    legacy_source = await messages.insert_user_message(legacy_fish_id, "legacy")
    await messages.insert_user_message(legacy_fish_id, "latest")
    await postgres_pool.executemany(
        """
        INSERT INTO facts (
            id, fish_id, session_id, source_message_id, fact_index,
            text, mentioned_at, entities
        )
        VALUES ($1, $2, $3, $4, 0, $5, now(), '[]'::jsonb)
        """,
        [
            (
                uuid4(),
                current_fish_id,
                current_source.session_id,
                current_source.id,
                "current fact",
            ),
            (
                uuid4(),
                legacy_fish_id,
                legacy_source.session_id,
                legacy_source.id,
                "legacy fact",
            ),
        ],
    )
    await postgres_pool.execute(
        "UPDATE fishes SET snapshot_storage_version = 1 WHERE id = $1",
        legacy_fish_id,
    )

    pending = await facts.list_pending_messages(limit=10)

    assert [item.message.fish_id for item in pending] == [current_fish_id]


@pytest.mark.asyncio
async def test_store_returns_full_generating_assistant_for_processor_visibility(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    first_user = await messages.insert_user_message(FIRST_FISH_ID, "question")
    assistant = await messages.insert_generating_answer(
        FIRST_FISH_ID,
        first_user.session_id,
        first_user.sequence,
    )
    await messages.insert_user_message(FIRST_FISH_ID, "latest")

    first_candidate = (await facts.list_pending_messages(limit=10))[0]
    await persist_extraction(facts, first_candidate, ())
    generating_candidate = (await facts.list_pending_messages(limit=10))[0]

    assert generating_candidate.message.id == assistant.id
    assert generating_candidate.message.status is MessageStatus.GENERATING

    await messages.update_consumed_offset(assistant.id, FIRST_FISH_ID, 7)
    await messages.finish_answer(
        assistant.id,
        FIRST_FISH_ID,
        content="visible hidden",
        requested_status=MessageStatus.INTERRUPTED,
    )
    assistant_candidate = (await facts.list_pending_messages(limit=10))[0]

    assert assistant_candidate.message.id == assistant.id
    assert assistant_candidate.message.content == "visible hidden"
    assert assistant_candidate.message.consumed_offset == 7
    assert visible_message_content(assistant_candidate.message) == "visible"


@pytest.mark.asyncio
async def test_rerunning_source_preserves_stable_identity_and_removes_tail(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    source = await messages.insert_user_message(FIRST_FISH_ID, "source")
    await messages.insert_user_message(FIRST_FISH_ID, "latest")
    pending = (await facts.list_pending_messages(limit=10))[0]
    assert pending.message.id == source.id

    await persist_extraction(
        facts,
        pending,
        [extracted_fact("first"), extracted_fact("removed")],
    )
    original_rows = await postgres_pool.fetch(
        """
        SELECT id, sequence, fact_index, text
        FROM facts
        WHERE source_message_id = $1
        ORDER BY fact_index
        """,
        source.id,
    )
    await persist_extraction(facts, pending, [extracted_fact("replacement")])

    rows = await postgres_pool.fetch(
        """
        SELECT id, sequence, fact_index, text
        FROM facts
        WHERE source_message_id = $1
        ORDER BY fact_index
        """,
        source.id,
    )
    assert len(original_rows) == 2
    assert len(rows) == 1
    assert rows[0]["id"] == original_rows[0]["id"]
    assert rows[0]["sequence"] == original_rows[0]["sequence"]
    assert [(row["fact_index"], row["text"]) for row in rows] == [(0, "replacement")]


@pytest.mark.asyncio
async def test_fact_rows_and_watermark_roll_back_together(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    first = await messages.insert_user_message(FIRST_FISH_ID, "first")
    source = await messages.insert_user_message(FIRST_FISH_ID, "source")
    await messages.insert_user_message(FIRST_FISH_ID, "latest")

    first_pending = (await facts.list_pending_messages(limit=10))[0]
    await persist_extraction(facts, first_pending, ())
    source_pending = (await facts.list_pending_messages(limit=10))[0]
    await persist_extraction(
        facts,
        source_pending,
        [extracted_fact("preserved")],
    )
    await postgres_pool.execute(
        """
        UPDATE session_background_task_progress
        SET through_sequence = $2
        WHERE session_id = $1 AND task_name = 'fact_extraction'
        """,
        source.session_id,
        first.sequence,
    )

    with pytest.raises(asyncpg.CheckViolationError):
        await persist_extraction(
            facts,
            source_pending,
            [
                extracted_fact(
                    "invalid",
                    occurred_end=datetime(2026, 7, 23, tzinfo=UTC),
                )
            ],
        )

    stored_facts = await postgres_pool.fetch(
        """
        SELECT fact_index, text
        FROM facts
        WHERE source_message_id = $1
        """,
        source.id,
    )
    through_sequence = await postgres_pool.fetchval(
        """
        SELECT through_sequence
        FROM session_background_task_progress
        WHERE session_id = $1 AND task_name = 'fact_extraction'
        """,
        source.session_id,
    )
    assert [(row["fact_index"], row["text"]) for row in stored_facts] == [
        (0, "preserved")
    ]
    assert through_sequence == first.sequence


@pytest.mark.asyncio
async def test_progress_target_and_pending_exclude_current_latest(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    first = await messages.insert_user_message(FIRST_FISH_ID, "first")
    target = await messages.insert_user_message(FIRST_FISH_ID, "target")
    latest = await messages.insert_user_message(FIRST_FISH_ID, "latest")
    first_pending = (await facts.list_pending_messages(limit=10))[0]
    await persist_extraction(facts, first_pending, ())

    progress = await facts.get_progress(FIRST_FISH_ID, first.session_id)

    assert progress.through_sequence == first.sequence
    assert progress.target_sequence == target.sequence
    assert progress.pending == 1
    assert progress.target_sequence < latest.sequence


@pytest.mark.asyncio
async def test_terminal_message_failure_advances_watermark_and_keeps_later_work(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, facts = await migrated_stores(postgres_pool)
    failed = await messages.insert_user_message(FIRST_FISH_ID, "will fail")
    later = await messages.insert_user_message(FIRST_FISH_ID, "must continue")
    await messages.insert_user_message(FIRST_FISH_ID, "latest remains unstable")

    first_pending = (await facts.list_pending_messages(limit=10))[0]
    assert first_pending.message.id == failed.id

    await facts.record_failure(first_pending, "LLM response invalid", terminal=True)

    progress = await facts.get_progress(FIRST_FISH_ID, failed.session_id)
    next_pending = (await facts.list_pending_messages(limit=10))[0]
    failed_row = await postgres_pool.fetchrow(
        """
        SELECT
            fact_extraction_status,
            fact_extraction_attempt_count,
            fact_extraction_error
        FROM messages
        WHERE id = $1
        """,
        failed.id,
    )

    assert progress.through_sequence == failed.sequence
    assert progress.pending == 1
    assert next_pending.message.id == later.id
    assert dict(failed_row) == {
        "fact_extraction_status": "failed",
        "fact_extraction_attempt_count": 1,
        "fact_extraction_error": "LLM response invalid",
    }
