from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.db.migrations import (
    CREATE_MIGRATIONS_TABLE_SQL,
    MIGRATIONS,
    MigrationPool,
    run_migrations,
)
from fish_memory.observations.routes import _observation_response
from fish_memory.observations.store import ObservationStore


async def _insert_observation(
    pool: asyncpg.Pool,
    fish_id: UUID,
    summary: str,
    *,
    source: str = "environment_history",
    observation_type: str = "semantic_snapshot_300s",
) -> UUID:
    observation_id = uuid4()
    await pool.execute(
        """
        INSERT INTO observations (
            id, fish_id, kind, source, observation_type, scope_key,
            semantic_key, summary, started_at, ended_at
        ) VALUES ($1, $2, 'state', $3, $4, 'capture:test',
                  $1::uuid::text, $5, now(), now())
        """,
        observation_id,
        fish_id,
        source,
        observation_type,
        summary,
    )
    return observation_id


@pytest.mark.asyncio
async def test_upgrade_preserves_existing_rows_and_accepts_full_semantic_summaries(
    postgres_pool: asyncpg.Pool,
) -> None:
    await postgres_pool.execute(CREATE_MIGRATIONS_TABLE_SQL)
    for migration in MIGRATIONS:
        if migration.version >= 39:
            continue
        await postgres_pool.execute(migration.sql)
        await postgres_pool.execute(
            "INSERT INTO schema_migrations (version) VALUES ($1)", migration.version
        )

    fish_id = uuid4()
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", fish_id)
    old_summary = "旧" * 2000
    old_id = await _insert_observation(postgres_pool, fish_id, old_summary)
    with pytest.raises(asyncpg.CheckViolationError):
        await _insert_observation(postgres_pool, fish_id, "新" * 2001)

    await run_migrations(cast(MigrationPool, postgres_pool))
    await run_migrations(cast(MigrationPool, postgres_pool))
    assert (
        await postgres_pool.fetchval(
            "SELECT summary FROM observations WHERE id=$1", old_id
        )
        == old_summary
    )
    full_summary = "影音内容：\n" + "完整剧情。" * 500 + "\n电脑活动：\n切到编辑器。"
    new_id = await _insert_observation(postgres_pool, fish_id, full_summary)
    stored = await ObservationStore(postgres_pool).get_observation(fish_id, new_id)
    assert stored is not None
    assert stored.summary == full_summary
    assert _observation_response(stored).summary == full_summary
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM schema_migrations WHERE version=39"
        )
        == 1
    )


@pytest.mark.asyncio
async def test_long_summary_limit_remains_scoped_and_bounded(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = uuid4()
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", fish_id)
    summary = "长" * 31000
    observation_id = await _insert_observation(
        postgres_pool,
        fish_id,
        summary,
        observation_type="semantic_snapshot_7200s",
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT summary FROM observations WHERE id=$1", observation_id
        )
        == summary
    )
    for source, observation_type, text in (
        ("environment_history", "semantic_snapshot_300s", "长" * 31001),
        ("environment_history", "semantic_snapshot_300s", ""),
        ("computer_history", "semantic_snapshot_300s", "长" * 2001),
        ("environment_history", "activity_summary_10min", "长" * 2001),
        ("environment_history", "semantic_snapshot_0s", "长" * 2001),
    ):
        with pytest.raises(asyncpg.CheckViolationError):
            await _insert_observation(
                postgres_pool,
                fish_id,
                text,
                source=source,
                observation_type=observation_type,
            )
