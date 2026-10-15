import asyncio
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.background.relation_processor import SnapshotItemRelationProcessor
from fish_memory.db.embedding_spaces import FishEmbeddingSpaceStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.relations import SnapshotItemRelationStore
from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.relations.models import SnapshotItemEmbeddingProjection

SPACE = EmbeddingSpace("local", "relation-test", 3)
SUPPORTED_AT = datetime(2026, 8, 27, 8, 0, tzinfo=UTC)


class FakeEmbedder:
    space = SPACE

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            vectors=tuple((1.0, 0.0, 0.0) for _text in texts),
            input_tokens=len(texts),
        )


def worker(
    pool: asyncpg.Pool,
    embedder: FakeEmbedder,
    *,
    snapshot_type: str,
    batch_size: int,
    background_owner: UUID | None = None,
) -> SnapshotItemRelationProcessor:
    return SnapshotItemRelationProcessor(
        SnapshotItemRelationStore(
            pool,
            snapshot_type=snapshot_type,
            background_owner=background_owner,
        ),
        embedder,
        embedding_space_store=FishEmbeddingSpaceStore(pool),
        embedding_semaphore=asyncio.Semaphore(1),
        snapshot_type=snapshot_type,
        max_retries=3,
        batch_size=batch_size,
    )


async def insert_dynamic_snapshot(
    pool: asyncpg.Pool,
    *,
    fish_id: UUID,
    session_id: UUID,
    summaries: Sequence[str],
    version: int = 4,
) -> tuple[int, tuple[int, ...]]:
    await pool.execute("INSERT INTO fishes (id) VALUES ($1)", fish_id)
    await pool.execute(
        "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2)",
        session_id,
        fish_id,
    )
    snapshot_id = cast(
        int,
        await pool.fetchval(
            """
            INSERT INTO memory_snapshots (
                fish_id, session_id, snapshot_type, content,
                through_fact_sequence, version
            )
            VALUES ($1, $2, 'dynamic', NULL, 10, $3)
            RETURNING id
            """,
            fish_id,
            session_id,
            version,
        ),
    )
    item_ids: list[int] = []
    for index, summary in enumerate(summaries):
        fact_id = UUID(int=1000 + index)
        item_ids.append(
            cast(
                int,
                await pool.fetchval(
                    """
                    INSERT INTO memory_snapshot_items (
                        snapshot_id, summary, support_fact_ids,
                        lineage_fact_ids, last_supported_at
                    )
                    VALUES ($1, $2, $3::uuid[], $3::uuid[], $4)
                    RETURNING id
                    """,
                    snapshot_id,
                    summary,
                    [fact_id],
                    SUPPORTED_AT,
                ),
            )
        )
    return snapshot_id, tuple(item_ids)


@pytest.mark.asyncio
async def test_item_relation_processor_repairs_partial_and_evolving_projection(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = uuid4()
    session_id = uuid4()
    other_session_id = uuid4()
    _snapshot_id, (first_item_id, second_item_id) = await insert_dynamic_snapshot(
        postgres_pool,
        fish_id=fish_id,
        session_id=session_id,
        summaries=("第一项", "第二项"),
    )
    embedder = FakeEmbedder()
    processor = worker(
        postgres_pool,
        embedder,
        snapshot_type="dynamic",
        batch_size=1,
    )

    assert await processor.process_once() is True
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_item_embeddings"
        )
        == 1
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT through_sequence
            FROM session_background_task_progress
            WHERE session_id = $1 AND task_name = 'dynamic_relations'
            """,
            session_id,
        )
        is None
    )

    assert await processor.process_once() is True
    assert embedder.calls == [("第一项",), ("第二项",)]
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_item_embeddings"
        )
        == 2
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT through_sequence
            FROM session_background_task_progress
            WHERE session_id = $1 AND task_name = 'dynamic_relations'
            """,
            session_id,
        )
        == 4
    )

    first_hash = hashlib.sha256("第一项".encode()).hexdigest()
    mask_signature = "b" * 64
    await postgres_pool.execute(
        """
        INSERT INTO memory_snapshot_item_mask_embeddings (
            snapshot_item_id, fish_id, entity_ids, signature_hash,
            source_text_hash, masked_text, provider, model,
            dimensions, embedding
        )
        VALUES (
            $1, $2, $3::uuid[], $4, $5, '掩码摘要',
            'diagnostic-provider', 'diagnostic-model', 3,
            '[0,1,0]'::vector
        )
        """,
        first_item_id,
        fish_id,
        [uuid4()],
        mask_signature,
        first_hash,
    )
    added_fact_id = uuid4()
    await postgres_pool.execute(
        "UPDATE memory_snapshots SET version = 5 WHERE id = $1",
        _snapshot_id,
    )
    await postgres_pool.execute(
        """
        UPDATE memory_snapshot_items
        SET support_fact_ids = support_fact_ids || $2::uuid[],
            lineage_fact_ids = lineage_fact_ids || $2::uuid[],
            item_version = 2,
            last_supported_at = $3
        WHERE id = $1
        """,
        first_item_id,
        [added_fact_id],
        SUPPORTED_AT + timedelta(hours=1),
    )

    assert await processor.process_once() is True
    assert embedder.calls == [("第一项",), ("第二项",)]
    support_only = await postgres_pool.fetchrow(
        """
        SELECT item_version, session_id, embedding::text AS vector_text
        FROM memory_snapshot_item_embeddings
        WHERE snapshot_item_id = $1
        """,
        first_item_id,
    )
    assert support_only is not None
    assert support_only["item_version"] == 2
    assert support_only["session_id"] == session_id
    assert support_only["vector_text"] == "[1,0,0]"
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*) FROM memory_snapshot_item_mask_embeddings
            WHERE snapshot_item_id = $1
            """,
            first_item_id,
        )
        == 1
    )

    await postgres_pool.execute(
        "UPDATE memory_snapshots SET version = 6 WHERE id = $1",
        _snapshot_id,
    )
    await postgres_pool.execute(
        """
        UPDATE memory_snapshot_items
        SET summary = '第一项已完成', item_version = 3
        WHERE id = $1
        """,
        first_item_id,
    )

    assert await processor.process_once() is True
    assert embedder.calls[-1] == ("第一项已完成",)
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*) FROM memory_snapshot_item_mask_embeddings
            WHERE snapshot_item_id = $1
            """,
            first_item_id,
        )
        == 0
    )
    updated_full = await postgres_pool.fetchrow(
        """
        SELECT snapshot_item_id, item_version, snapshot_text
        FROM memory_snapshot_item_embeddings
        WHERE snapshot_item_id = $1
        """,
        first_item_id,
    )
    assert updated_full is not None
    assert tuple(updated_full.values()) == (first_item_id, 3, "第一项已完成")

    await postgres_pool.execute(
        "DELETE FROM memory_snapshot_item_embeddings WHERE snapshot_item_id = $1",
        second_item_id,
    )
    assert await processor.process_once() is True
    assert embedder.calls[-1] == ("第二项",)
    assert (
        await postgres_pool.fetchval(
            """
            SELECT through_sequence
            FROM session_background_task_progress
            WHERE session_id = $1 AND task_name = 'dynamic_relations'
            """,
            session_id,
        )
        == 6
    )

    await postgres_pool.execute(
        "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2)",
        other_session_id,
        fish_id,
    )
    await postgres_pool.execute(
        """
        UPDATE memory_snapshot_item_embeddings
        SET session_id = $2
        WHERE snapshot_item_id = $1
        """,
        second_item_id,
        other_session_id,
    )

    assert await processor.process_once() is True
    repaired_scope = await postgres_pool.fetchrow(
        """
        SELECT session_id, item_version, dimensions
        FROM memory_snapshot_item_embeddings
        WHERE snapshot_item_id = $1
        """,
        second_item_id,
    )
    assert repaired_scope is not None
    assert tuple(repaired_scope.values()) == (session_id, 1, 3)
    assert embedder.calls[-1] == ("第二项",)

    await postgres_pool.execute(
        "DELETE FROM memory_snapshot_items WHERE id = $1",
        first_item_id,
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*) FROM memory_snapshot_item_embeddings
            WHERE snapshot_item_id = $1
            """,
            first_item_id,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_static_projection_candidates_isolate_storage_owner_and_deletion(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    owner = uuid4()
    selected_fish = uuid4()
    other_owner_fish = uuid4()
    legacy_fish = uuid4()
    deleted_fish = uuid4()
    for fish_id, background_owner, storage_version, deleted_at in (
        (selected_fish, owner, 2, None),
        (other_owner_fish, uuid4(), 2, None),
        (legacy_fish, owner, 1, None),
        (deleted_fish, owner, 2, SUPPORTED_AT),
    ):
        await postgres_pool.execute(
            """
            INSERT INTO fishes (
                id, background_owner, snapshot_storage_version, deleted_at
            )
            VALUES ($1, $2, $3, $4)
            """,
            fish_id,
            background_owner,
            storage_version,
            deleted_at,
        )
        await postgres_pool.execute(
            """
            INSERT INTO memory_snapshots (
                fish_id, session_id, snapshot_type, content,
                through_fact_sequence, version
            )
            VALUES ($1, NULL, 'static', NULL, 0, 1)
            """,
            fish_id,
        )

    store = SnapshotItemRelationStore(
        postgres_pool,
        snapshot_type="static",
        background_owner=owner,
    )
    pending = await store.list_pending_batches(10, SPACE.dimensions)

    assert [(batch.fish_id, batch.items) for batch in pending] == [(selected_fish, ())]

    embedder = FakeEmbedder()
    processor = worker(
        postgres_pool,
        embedder,
        snapshot_type="static",
        batch_size=1,
        background_owner=owner,
    )
    assert await processor.process_once() is True
    assert embedder.calls == []
    assert (
        await postgres_pool.fetchval(
            """
            SELECT through_sequence
            FROM background_task_progress
            WHERE fish_id = $1 AND task_name = 'static_relations'
            """,
            selected_fish,
        )
        == 1
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*)
            FROM background_task_progress
            WHERE task_name = 'static_relations' AND fish_id <> $1
            """,
            selected_fish,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_projection_conflicts_roll_back_full_rows_masks_and_progress(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = uuid4()
    session_id = uuid4()
    snapshot_id, (first_item_id, second_item_id) = await insert_dynamic_snapshot(
        postgres_pool,
        fish_id=fish_id,
        session_id=session_id,
        summaries=("第一项", "第二项"),
    )
    store = SnapshotItemRelationStore(postgres_pool, snapshot_type="dynamic")
    pending = (await store.list_pending_batches(10, SPACE.dimensions))[0]
    await postgres_pool.execute(
        """
        UPDATE memory_snapshot_items
        SET item_version = item_version + 1
        WHERE id = $1
        """,
        second_item_id,
    )
    stale_projections = tuple(
        SnapshotItemEmbeddingProjection(
            item_id=item.id,
            item_version=item.item_version,
            session_id=session_id,
            snapshot_type="dynamic",
            snapshot_text=item.summary,
            text_hash=hashlib.sha256(item.summary.encode()).hexdigest(),
            last_supported_at=item.last_supported_at,
            vector=(1.0, 0.0, 0.0),
        )
        for item in pending.items
    )

    assert not await store.persist_batch(pending, stale_projections, SPACE)
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_item_embeddings"
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*) FROM session_background_task_progress
            WHERE session_id = $1 AND task_name = 'dynamic_relations'
            """,
            session_id,
        )
        == 0
    )

    await postgres_pool.execute(
        "UPDATE memory_snapshot_items SET item_version = 1 WHERE id = $1",
        second_item_id,
    )
    fresh = (await store.list_pending_batches(10, SPACE.dimensions))[0]
    first, second = fresh.items
    first_projection = SnapshotItemEmbeddingProjection(
        item_id=first_item_id,
        item_version=first.item_version,
        session_id=session_id,
        snapshot_type="dynamic",
        snapshot_text=first.summary,
        text_hash=hashlib.sha256(first.summary.encode()).hexdigest(),
        last_supported_at=first.last_supported_at,
        vector=(1.0, 0.0, 0.0),
    )
    missing_reuse = SnapshotItemEmbeddingProjection(
        item_id=second_item_id,
        item_version=second.item_version,
        session_id=session_id,
        snapshot_type="dynamic",
        snapshot_text=second.summary,
        text_hash=hashlib.sha256(second.summary.encode()).hexdigest(),
        last_supported_at=second.last_supported_at,
        vector=None,
    )

    assert not await store.persist_batch(
        fresh,
        (first_projection, missing_reuse),
        SPACE,
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM memory_snapshot_item_embeddings"
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*) FROM memory_snapshot_item_mask_embeddings
            """
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*) FROM session_background_task_progress
            WHERE session_id = $1 AND task_name = 'dynamic_relations'
            """,
            session_id,
        )
        == 0
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT version FROM memory_snapshots WHERE id = $1",
            snapshot_id,
        )
        == 4
    )
