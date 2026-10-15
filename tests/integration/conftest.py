import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio
from pg0 import Pg0

from fish_memory.db.embedded import ResolvedDatabase, resolve_database
from fish_memory.db.pool import create_pool


@pytest.fixture
def test_database_url() -> str | None:
    return os.getenv("FISH_MEMORY_TEST_DATABASE_URL")


@pytest_asyncio.fixture
async def postgres_pool(
    test_database_url: str | None,
) -> AsyncIterator[asyncpg.Pool]:
    resolved: ResolvedDatabase | None = None
    instance_name: str | None = None
    if test_database_url is None:
        if os.getenv("FISH_MEMORY_RUN_PG0_TESTS") != "1":
            pytest.skip("PostgreSQL test database is not configured")
        instance_name = f"fish-memory-integration-{uuid4().hex}"
        resolved = await resolve_database(f"pg0://{instance_name}")
        test_database_url = resolved.database_url

    pool = await create_pool(test_database_url)
    async with pool.acquire() as connection:
        await connection.execute("DROP TABLE IF EXISTS message_attachments")
        await connection.execute("DROP TABLE IF EXISTS session_context_progress")
        await connection.execute("DROP TABLE IF EXISTS session_history_pending")
        await connection.execute("DROP TABLE IF EXISTS session_history_units")
        await connection.execute("DROP TABLE IF EXISTS session_history_chunks")
        await connection.execute("DROP TABLE IF EXISTS session_history_contexts")
        await connection.execute(
            "DROP FUNCTION IF EXISTS enqueue_session_history() CASCADE"
        )
        await connection.execute(
            "DROP FUNCTION IF EXISTS "
            "history_observation_session(uuid,timestamptz,jsonb)"
        )
        await connection.execute("DROP TABLE IF EXISTS llm_call_logs")
        await connection.execute("DROP TABLE IF EXISTS background_assistant_deliveries")
        await connection.execute("DROP TABLE IF EXISTS message_input_modes")
        await connection.execute("DROP TABLE IF EXISTS mind_temperaments")
        await connection.execute("DROP TABLE IF EXISTS mind_trace")
        await connection.execute("DROP TABLE IF EXISTS mind_commitments")
        await connection.execute("DROP TABLE IF EXISTS mind_intentions")
        await connection.execute("DROP TABLE IF EXISTS message_unsaid")
        await connection.execute(
            "DROP TABLE IF EXISTS computer_history_summary_checkpoints"
        )
        await connection.execute("DROP TABLE IF EXISTS observations")
        await connection.execute("DROP TABLE IF EXISTS observation_app_policies")
        await connection.execute("DROP TABLE IF EXISTS agent_interactions")
        await connection.execute("DROP TABLE IF EXISTS agent_runs")
        await connection.execute("DROP TABLE IF EXISTS agent_tasks")
        await connection.execute(
            "ALTER TABLE IF EXISTS fishes DROP COLUMN IF EXISTS agent_target_id"
        )
        await connection.execute("DROP TABLE IF EXISTS agent_targets")
        await connection.execute("DROP TABLE IF EXISTS agent_endpoints")
        await connection.execute("DROP TABLE IF EXISTS home_action_runs")
        await connection.execute(
            "DROP SEQUENCE IF EXISTS home_action_run_events_sequence_seq"
        )
        await connection.execute("DROP TABLE IF EXISTS evaluation_questions")
        await connection.execute(
            "DROP TABLE IF EXISTS memory_snapshot_item_mask_embeddings"
        )
        await connection.execute("DROP TABLE IF EXISTS memory_snapshot_item_embeddings")
        await connection.execute("DROP TABLE IF EXISTS memory_snapshot_item_versions")
        await connection.execute("DROP TABLE IF EXISTS memory_snapshot_items")
        await connection.execute("DROP TABLE IF EXISTS memory_snapshot_revisions")
        await connection.execute(
            "DROP TABLE IF EXISTS fact_relation_group_mask_embeddings"
        )
        await connection.execute("DROP TABLE IF EXISTS fact_mask_embeddings")
        await connection.execute("DROP TABLE IF EXISTS fact_entities")
        await connection.execute("DROP TABLE IF EXISTS entity_aliases")
        await connection.execute("DROP TABLE IF EXISTS fact_relation_group_embeddings")
        await connection.execute("DROP TABLE IF EXISTS fact_relation_groups")
        await connection.execute("DROP TABLE IF EXISTS memory_snapshot_versions")
        await connection.execute("DROP TABLE IF EXISTS memory_snapshots")
        await connection.execute("DROP TABLE IF EXISTS fact_embeddings")
        await connection.execute("DROP TABLE IF EXISTS facts")
        await connection.execute(
            "DROP TABLE IF EXISTS session_background_task_progress"
        )
        await connection.execute("DROP TABLE IF EXISTS background_task_progress")
        await connection.execute("DROP TABLE IF EXISTS message_performance_steps")
        await connection.execute("DROP TABLE IF EXISTS messages")
        await connection.execute("DROP TABLE IF EXISTS fish_sessions")
        await connection.execute("DROP TABLE IF EXISTS fishes")
        await connection.execute("DROP TABLE IF EXISTS schema_migrations")
    try:
        yield pool
    finally:
        async with pool.acquire() as connection:
            await connection.execute("DROP TABLE IF EXISTS message_attachments")
            await connection.execute("DROP TABLE IF EXISTS session_context_progress")
            await connection.execute("DROP TABLE IF EXISTS session_history_pending")
            await connection.execute("DROP TABLE IF EXISTS session_history_units")
            await connection.execute("DROP TABLE IF EXISTS session_history_chunks")
            await connection.execute("DROP TABLE IF EXISTS session_history_contexts")
            await connection.execute(
                "DROP FUNCTION IF EXISTS enqueue_session_history() CASCADE"
            )
            await connection.execute(
                "DROP FUNCTION IF EXISTS "
                "history_observation_session(uuid,timestamptz,jsonb)"
            )
            await connection.execute("DROP TABLE IF EXISTS llm_call_logs")
            await connection.execute(
                "DROP TABLE IF EXISTS background_assistant_deliveries"
            )
            await connection.execute("DROP TABLE IF EXISTS message_input_modes")
            await connection.execute("DROP TABLE IF EXISTS mind_temperaments")
            await connection.execute("DROP TABLE IF EXISTS mind_trace")
            await connection.execute("DROP TABLE IF EXISTS mind_commitments")
            await connection.execute("DROP TABLE IF EXISTS mind_intentions")
            await connection.execute("DROP TABLE IF EXISTS message_unsaid")
            await connection.execute(
                "DROP TABLE IF EXISTS computer_history_summary_checkpoints"
            )
            await connection.execute("DROP TABLE IF EXISTS observations")
            await connection.execute("DROP TABLE IF EXISTS observation_app_policies")
            await connection.execute("DROP TABLE IF EXISTS agent_interactions")
            await connection.execute("DROP TABLE IF EXISTS agent_runs")
            await connection.execute("DROP TABLE IF EXISTS agent_tasks")
            await connection.execute(
                "ALTER TABLE IF EXISTS fishes DROP COLUMN IF EXISTS agent_target_id"
            )
            await connection.execute("DROP TABLE IF EXISTS agent_targets")
            await connection.execute("DROP TABLE IF EXISTS agent_endpoints")
            await connection.execute("DROP TABLE IF EXISTS home_action_runs")
            await connection.execute(
                "DROP SEQUENCE IF EXISTS home_action_run_events_sequence_seq"
            )
            await connection.execute("DROP TABLE IF EXISTS evaluation_questions")
            await connection.execute(
                "DROP TABLE IF EXISTS memory_snapshot_item_mask_embeddings"
            )
            await connection.execute(
                "DROP TABLE IF EXISTS memory_snapshot_item_embeddings"
            )
            await connection.execute(
                "DROP TABLE IF EXISTS memory_snapshot_item_versions"
            )
            await connection.execute("DROP TABLE IF EXISTS memory_snapshot_items")
            await connection.execute("DROP TABLE IF EXISTS memory_snapshot_revisions")
            await connection.execute(
                "DROP TABLE IF EXISTS fact_relation_group_mask_embeddings"
            )
            await connection.execute("DROP TABLE IF EXISTS fact_mask_embeddings")
            await connection.execute("DROP TABLE IF EXISTS fact_entities")
            await connection.execute("DROP TABLE IF EXISTS entity_aliases")
            await connection.execute(
                "DROP TABLE IF EXISTS fact_relation_group_embeddings"
            )
            await connection.execute("DROP TABLE IF EXISTS fact_relation_groups")
            await connection.execute("DROP TABLE IF EXISTS memory_snapshot_versions")
            await connection.execute("DROP TABLE IF EXISTS memory_snapshots")
            await connection.execute("DROP TABLE IF EXISTS fact_embeddings")
            await connection.execute("DROP TABLE IF EXISTS facts")
            await connection.execute(
                "DROP TABLE IF EXISTS session_background_task_progress"
            )
            await connection.execute("DROP TABLE IF EXISTS background_task_progress")
            await connection.execute("DROP TABLE IF EXISTS message_performance_steps")
            await connection.execute("DROP TABLE IF EXISTS messages")
            await connection.execute("DROP TABLE IF EXISTS fish_sessions")
            await connection.execute("DROP TABLE IF EXISTS fishes")
            await connection.execute("DROP TABLE IF EXISTS schema_migrations")
        await pool.close()
        if resolved is not None:
            await resolved.close()
        if instance_name is not None:
            await asyncio.to_thread(Pg0(name=instance_name).drop)
            instance_dir = Path.home() / ".pg0" / "instances" / instance_name
            await asyncio.to_thread(shutil.rmtree, instance_dir, True)
