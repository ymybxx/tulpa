from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import asyncpg
import pytest

from fish_memory.db.migrations import (
    CREATE_MIGRATIONS_TABLE_SQL,
    MIGRATIONS,
    MigrationPool,
    run_migrations,
)


@pytest.mark.asyncio
async def test_session_migration_backfills_legacy_conversation_ownership(
    postgres_pool: asyncpg.Pool,
) -> None:
    fish_id = uuid4()
    user_message_id = uuid4()
    assistant_message_id = uuid4()
    fact_id = uuid4()
    created_at = datetime(2026, 8, 9, 8, 0, tzinfo=UTC)

    async with postgres_pool.acquire() as connection:
        await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
        for migration in MIGRATIONS:
            if migration.version >= 20:
                continue
            await connection.execute(migration.sql)
            await connection.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                migration.version,
            )

        await connection.execute(
            """
            INSERT INTO fishes (
                id,
                created_at,
                history_window_start_sequence,
                history_window_token_count
            )
            VALUES ($1, $2, 11, 1200)
            """,
            fish_id,
            created_at,
        )
        await connection.execute(
            """
            INSERT INTO messages (
                id, fish_id, role, content, status, created_at
            )
            VALUES
                ($1, $3, 'user', 'hello', 'initialized', $4),
                ($2, $3, 'assistant', 'hi', 'completed', $4 + interval '5 minutes')
            """,
            user_message_id,
            assistant_message_id,
            fish_id,
            created_at,
        )
        await connection.execute(
            """
            INSERT INTO facts (
                id,
                fish_id,
                source_message_id,
                fact_index,
                text,
                mentioned_at
            )
            VALUES ($1, $2, $3, 0, 'legacy fact', $4)
            """,
            fact_id,
            fish_id,
            user_message_id,
            created_at,
        )
        await connection.executemany(
            """
            INSERT INTO memory_snapshots (
                fish_id,
                snapshot_type,
                content,
                through_fact_sequence,
                version
            )
            VALUES ($1, $2, '{"items": []}'::jsonb, 1, 2)
            """,
            [(fish_id, "dynamic"), (fish_id, "static")],
        )
        await connection.executemany(
            """
            INSERT INTO memory_snapshot_versions (
                fish_id,
                snapshot_type,
                version,
                content,
                through_fact_sequence
            )
            VALUES ($1, $2, 2, '{"items": []}'::jsonb, 1)
            """,
            [(fish_id, "dynamic"), (fish_id, "static")],
        )
        await connection.executemany(
            """
            INSERT INTO background_task_progress (
                fish_id, task_name, through_sequence
            )
            VALUES ($1, $2, $3)
            """,
            [
                (fish_id, "fact_extraction", 2),
                (fish_id, "dynamic_snapshot", 1),
                (fish_id, "dynamic_relations", 2),
                (fish_id, "static_profile", 1),
            ],
        )

    await run_migrations(cast(MigrationPool, postgres_pool))

    assert await postgres_pool.fetchval("SELECT to_regclass('fish_sessions')") == (
        "fish_sessions"
    )

    session = await postgres_pool.fetchrow(
        """
        SELECT *
        FROM fish_sessions
        WHERE fish_id = $1
        """,
        fish_id,
    )
    assert session is not None
    session_id = session["id"]
    assert session["created_at"] == created_at
    assert session["last_message_at"] == datetime(2026, 8, 9, 8, 5, tzinfo=UTC)
    assert session["history_window_start_sequence"] == 11
    assert session["history_window_token_count"] == 1200

    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE session_id = $1",
            session_id,
        )
        == 2
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT session_id FROM facts WHERE id = $1",
            fact_id,
        )
        == session_id
    )

    snapshot_scopes = {
        row["snapshot_type"]: row["session_id"]
        for row in await postgres_pool.fetch(
            "SELECT snapshot_type, session_id FROM memory_snapshots"
        )
    }
    version_scopes = {
        row["snapshot_type"]: row["session_id"]
        for row in await postgres_pool.fetch(
            "SELECT snapshot_type, session_id FROM memory_snapshot_versions"
        )
    }
    assert snapshot_scopes == {"dynamic": session_id, "static": None}
    assert version_scopes == {"dynamic": session_id, "static": None}

    session_tasks = {
        row["task_name"]: row["through_sequence"]
        for row in await postgres_pool.fetch(
            """
            SELECT task_name, through_sequence
            FROM session_background_task_progress
            WHERE session_id = $1
            """,
            session_id,
        )
    }
    fish_tasks = {
        row["task_name"]: row["through_sequence"]
        for row in await postgres_pool.fetch(
            """
            SELECT task_name, through_sequence
            FROM background_task_progress
            WHERE fish_id = $1
            """,
            fish_id,
        )
    }
    assert session_tasks == {
        "fact_extraction": 2,
        "dynamic_snapshot": 1,
        "dynamic_relations": 2,
    }
    assert fish_tasks == {"static_profile": 1}

    assert (
        await postgres_pool.fetchval(
            """
        SELECT count(*)
        FROM information_schema.columns
        WHERE table_name = 'fishes'
          AND column_name IN (
              'history_window_start_sequence',
              'history_window_token_count'
          )
        """
        )
        == 0
    )


@pytest.mark.asyncio
async def test_migrations_are_idempotent_on_postgres(
    postgres_pool: asyncpg.Pool,
) -> None:
    pool = cast(MigrationPool, postgres_pool)

    await run_migrations(pool)
    await run_migrations(pool)

    async with postgres_pool.acquire() as connection:
        versions = await connection.fetch(
            "SELECT version, count(*) FROM schema_migrations GROUP BY version"
        )
        indexes = await connection.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename IN (
                'messages',
                'facts',
                'fact_embeddings',
                'fact_relation_group_embeddings',
                'fact_mask_embeddings',
                'fact_relation_group_mask_embeddings',
                'memory_snapshot_item_embeddings',
                'memory_snapshot_item_mask_embeddings',
                'entity_aliases',
                'fact_entities',
                'fish_sessions'
            )
            """
        )
        tables = await connection.fetch(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            """
        )
        history_columns = await connection.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name IN ('messages', 'fishes', 'fish_sessions')
              AND column_name LIKE 'history_%'
            """
        )
        fact_failure_columns = await connection.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'messages'
              AND column_name LIKE 'fact_extraction_%'
            """
        )
        timestamp_precisions = await connection.fetch(
            """
            SELECT table_name, column_name, datetime_precision
            FROM information_schema.columns
            WHERE data_type = 'timestamp with time zone'
              AND table_name IN (
                  'schema_migrations',
                  'messages',
                  'message_performance_steps',
                  'fishes',
                  'facts',
                  'background_task_progress',
                  'session_background_task_progress',
                  'fish_sessions',
                  'memory_snapshots',
                  'fact_embeddings',
                  'evaluation_questions',
                  'memory_snapshot_versions',
                  'fact_relation_groups',
                  'fact_relation_group_embeddings',
                  'fact_mask_embeddings',
                  'fact_relation_group_mask_embeddings',
                  'memory_snapshot_item_embeddings',
                  'memory_snapshot_item_mask_embeddings',
                  'entity_aliases'
              )
            """
        )

    assert {record["version"]: record["count"] for record in versions} == {
        migration.version: 1 for migration in MIGRATIONS
    }
    assert {record["indexname"] for record in indexes} >= {
        "idx_messages_fish_sequence",
        "idx_messages_one_generating_answer_per_session",
        "idx_messages_session_sequence",
        "idx_facts_fish_sequence",
        "idx_facts_session_sequence",
        "idx_facts_fish_occurred_start",
        "idx_facts_fish_mentioned_at",
        "idx_facts_entities_gin",
        "idx_fact_embeddings_fish",
        "idx_fact_relation_group_embeddings_static_scope",
        "idx_fact_relation_group_embeddings_dynamic_scope",
        "idx_fact_relation_group_embeddings_head_scope",
        "idx_fact_mask_embeddings_fish_dimensions",
        "idx_fact_relation_group_mask_embeddings_fish_dimensions",
        "idx_memory_snapshot_item_embeddings_static_scope",
        "idx_memory_snapshot_item_embeddings_dynamic_scope",
        "idx_memory_snapshot_item_mask_embeddings_fish_dimensions",
        "idx_entity_aliases_entity",
        "idx_fact_entities_fact",
        "idx_fish_sessions_fish_created",
        "idx_fish_sessions_fish_last_message",
    }
    assert {record["tablename"] for record in tables} >= {
        "facts",
        "background_task_progress",
        "session_background_task_progress",
        "fish_sessions",
        "fact_embeddings",
        "fact_relation_group_embeddings",
        "fact_mask_embeddings",
        "fact_relation_group_mask_embeddings",
        "memory_snapshot_item_embeddings",
        "memory_snapshot_item_mask_embeddings",
        "evaluation_questions",
        "entity_aliases",
        "fact_entities",
    }
    assert {record["column_name"] for record in history_columns} == {
        "history_token_count",
        "history_window_start_sequence",
        "history_window_token_count",
    }
    assert {record["column_name"] for record in fact_failure_columns} == {
        "fact_extraction_status",
        "fact_extraction_attempt_count",
        "fact_extraction_error",
    }
    assert {record["datetime_precision"] for record in timestamp_precisions} == {0}


@pytest.mark.asyncio
async def test_migration_repairs_relation_embedding_member_counts(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish_id = uuid4()
    fact_ids = [uuid4(), uuid4()]
    await postgres_pool.execute("INSERT INTO fishes (id) VALUES ($1)", fish_id)
    relation_group_id = await postgres_pool.fetchval(
        """
        INSERT INTO fact_relation_groups (
            fish_id, snapshot_type, fact_ids, snapshot_version
        )
        VALUES ($1, 'static', $2::uuid[], 1)
        RETURNING id
        """,
        fish_id,
        fact_ids,
    )
    await postgres_pool.execute(
        """
        INSERT INTO fact_relation_group_embeddings (
            relation_group_id,
            fish_id,
            session_id,
            snapshot_type,
            snapshot_version,
            snapshot_text,
            text_hash,
            last_supported_at,
            provider,
            model,
            dimensions,
            member_count,
            embedding
        )
        VALUES (
            $1, $2, NULL, 'static', 1, '两个成员', $3,
            '2026-08-25T08:00:00Z', 'local', 'test', 3, 1,
            '[1,0,0]'::vector
        )
        """,
        relation_group_id,
        fish_id,
        "a" * 64,
    )
    await postgres_pool.execute("DELETE FROM schema_migrations WHERE version = 29")
    vector_before = await postgres_pool.fetchval(
        """
        SELECT embedding::text
        FROM fact_relation_group_embeddings
        WHERE relation_group_id = $1
        """,
        relation_group_id,
    )

    await run_migrations(cast(MigrationPool, postgres_pool))

    repaired = await postgres_pool.fetchrow(
        """
        SELECT member_count, embedding::text AS vector_text
        FROM fact_relation_group_embeddings
        WHERE relation_group_id = $1
        """,
        relation_group_id,
    )
    assert repaired is not None
    assert repaired["member_count"] == 2
    assert repaired["vector_text"] == vector_before == "[1,0,0]"


@pytest.mark.asyncio
async def test_snapshot_item_cutover_leaves_legacy_fish_untouched(
    postgres_pool: asyncpg.Pool,
) -> None:
    old_fish_id = uuid4()
    new_fish_id = uuid4()
    relation_group_id: int

    async with postgres_pool.acquire() as connection:
        await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
        for migration in MIGRATIONS:
            if migration.version >= 30:
                continue
            await connection.execute(migration.sql)
            await connection.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                migration.version,
            )

        await connection.execute("INSERT INTO fishes (id) VALUES ($1)", old_fish_id)
        await connection.execute(
            """
            INSERT INTO memory_snapshots (
                fish_id,
                session_id,
                snapshot_type,
                content,
                through_fact_sequence,
                version
            )
            VALUES ($1, NULL, 'static', $2::jsonb, 3, 2)
            """,
            old_fish_id,
            '{"items":[{"summary":"legacy item","sourceFactIds":["legacy"]}]}',
        )
        await connection.execute(
            """
            INSERT INTO memory_snapshot_versions (
                fish_id,
                session_id,
                snapshot_type,
                version,
                content,
                through_fact_sequence
            )
            VALUES ($1, NULL, 'static', 2, $2::jsonb, 3)
            """,
            old_fish_id,
            '{"items":[{"summary":"legacy item","sourceFactIds":["legacy"]}]}',
        )
        relation_group_id = await connection.fetchval(
            """
            INSERT INTO fact_relation_groups (
                fish_id, snapshot_type, fact_ids, snapshot_version
            )
            VALUES ($1, 'static', $2::uuid[], 2)
            RETURNING id
            """,
            old_fish_id,
            [uuid4()],
        )
        await connection.execute(
            """
            INSERT INTO fact_relation_group_embeddings (
                relation_group_id,
                fish_id,
                session_id,
                snapshot_type,
                snapshot_version,
                snapshot_text,
                text_hash,
                last_supported_at,
                provider,
                model,
                dimensions,
                member_count,
                embedding
            )
            VALUES (
                $1, $2, NULL, 'static', 2, 'legacy item', $3,
                '2026-08-27T08:00:00Z', 'local', 'test', 3, 1,
                '[1,0,0]'::vector
            )
            """,
            relation_group_id,
            old_fish_id,
            "a" * 64,
        )
        legacy_content_before = await connection.fetchval(
            """
            SELECT content::text
            FROM memory_snapshots
            WHERE fish_id = $1 AND snapshot_type = 'static'
            """,
            old_fish_id,
        )
        legacy_vector_before = await connection.fetchval(
            """
            SELECT embedding::text
            FROM fact_relation_group_embeddings
            WHERE relation_group_id = $1
            """,
            relation_group_id,
        )

        migration = next(item for item in MIGRATIONS if item.version == 30)
        await connection.execute(migration.sql)
        await connection.execute("INSERT INTO fishes (id) VALUES ($1)", new_fish_id)

        old_fish = await connection.fetchrow(
            "SELECT snapshot_storage_version FROM fishes WHERE id = $1", old_fish_id
        )
        new_fish = await connection.fetchrow(
            "SELECT snapshot_storage_version FROM fishes WHERE id = $1", new_fish_id
        )
        normalized_item_count = await connection.fetchval(
            "SELECT count(*) FROM memory_snapshot_items"
        )
        legacy_content_after = await connection.fetchval(
            """
            SELECT content::text
            FROM memory_snapshots
            WHERE fish_id = $1 AND snapshot_type = 'static'
            """,
            old_fish_id,
        )
        legacy_vector_after = await connection.fetchval(
            """
            SELECT embedding::text
            FROM fact_relation_group_embeddings
            WHERE relation_group_id = $1
            """,
            relation_group_id,
        )

    assert old_fish is not None
    assert new_fish is not None
    assert old_fish["snapshot_storage_version"] == 1
    assert new_fish["snapshot_storage_version"] == 2
    assert normalized_item_count == 0
    assert legacy_content_after == legacy_content_before
    assert legacy_vector_after == legacy_vector_before


@pytest.mark.asyncio
async def test_item_embedding_migration_does_not_backfill_or_touch_legacy_rows(
    postgres_pool: asyncpg.Pool,
) -> None:
    fish_id = uuid4()
    legacy_fact_id = uuid4()
    legacy_entity_id = uuid4()

    async with postgres_pool.acquire() as connection:
        await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
        for migration in MIGRATIONS:
            if migration.version >= 31:
                continue
            await connection.execute(migration.sql)
            await connection.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                migration.version,
            )

        await connection.execute("INSERT INTO fishes (id) VALUES ($1)", fish_id)
        snapshot_id = await connection.fetchval(
            """
            INSERT INTO memory_snapshots (
                fish_id, session_id, snapshot_type, content,
                through_fact_sequence, version
            )
            VALUES ($1, NULL, 'static', NULL, 1, 1)
            RETURNING id
            """,
            fish_id,
        )
        await connection.execute(
            """
            INSERT INTO memory_snapshot_items (
                snapshot_id, summary, support_fact_ids,
                lineage_fact_ids, last_supported_at
            )
            VALUES ($1, '规范化条目', $2::uuid[], $2::uuid[], now())
            """,
            snapshot_id,
            [legacy_fact_id],
        )
        relation_group_id = await connection.fetchval(
            """
            INSERT INTO fact_relation_groups (
                fish_id, snapshot_type, fact_ids, snapshot_version
            )
            VALUES ($1, 'static', $2::uuid[], 1)
            RETURNING id
            """,
            fish_id,
            [legacy_fact_id],
        )
        await connection.execute(
            """
            INSERT INTO fact_relation_group_embeddings (
                relation_group_id, fish_id, session_id, snapshot_type,
                snapshot_version, snapshot_text, text_hash,
                last_supported_at, provider, model, dimensions,
                member_count, embedding
            )
            VALUES (
                $1, $2, NULL, 'static', 1, 'legacy relation', $3,
                now(), 'legacy-provider', 'legacy-model', 3, 1,
                '[1,0,0]'::vector
            )
            """,
            relation_group_id,
            fish_id,
            "a" * 64,
        )
        await connection.execute(
            """
            INSERT INTO fact_relation_group_mask_embeddings (
                relation_group_id, fish_id, entity_ids, signature_hash,
                source_text_hash, masked_text, provider, model,
                dimensions, embedding
            )
            VALUES (
                $1, $2, $3::uuid[], $4, $5, 'legacy masked',
                'legacy-provider', 'legacy-model', 3, '[0,1,0]'::vector
            )
            """,
            relation_group_id,
            fish_id,
            [legacy_entity_id],
            "b" * 64,
            "a" * 64,
        )
        legacy_before = await connection.fetchrow(
            """
            SELECT
                relation_embedding.embedding::text AS full_vector,
                mask.embedding::text AS mask_vector,
                relation_embedding.snapshot_text,
                mask.masked_text
            FROM fact_relation_group_embeddings AS relation_embedding
            JOIN fact_relation_group_mask_embeddings AS mask
              ON mask.relation_group_id = relation_embedding.relation_group_id
            WHERE relation_embedding.relation_group_id = $1
            """,
            relation_group_id,
        )

        migration = next(item for item in MIGRATIONS if item.version == 31)
        await connection.execute(migration.sql)

        full_count = await connection.fetchval(
            "SELECT count(*) FROM memory_snapshot_item_embeddings"
        )
        mask_count = await connection.fetchval(
            "SELECT count(*) FROM memory_snapshot_item_mask_embeddings"
        )
        legacy_after = await connection.fetchrow(
            """
            SELECT
                relation_embedding.embedding::text AS full_vector,
                mask.embedding::text AS mask_vector,
                relation_embedding.snapshot_text,
                mask.masked_text
            FROM fact_relation_group_embeddings AS relation_embedding
            JOIN fact_relation_group_mask_embeddings AS mask
              ON mask.relation_group_id = relation_embedding.relation_group_id
            WHERE relation_embedding.relation_group_id = $1
            """,
            relation_group_id,
        )

    assert full_count == 0
    assert mask_count == 0
    assert legacy_before is not None
    assert legacy_after is not None
    assert tuple(legacy_after.values()) == tuple(legacy_before.values())


@pytest.mark.asyncio
async def test_agent_task_migration_creates_constrained_provider_neutral_schema(
    postgres_pool: asyncpg.Pool,
) -> None:
    fish_id = uuid4()
    endpoint_id = uuid4()
    target_id = uuid4()

    async with postgres_pool.acquire() as connection:
        await run_migrations(postgres_pool)

        table_names = await connection.fetch(
            """
            SELECT relname
            FROM pg_class
            WHERE relname = ANY($1::text[])
            """,
            [
                "agent_endpoints",
                "agent_targets",
                "agent_tasks",
                "agent_runs",
                "agent_interactions",
            ],
        )
        await connection.execute("INSERT INTO fishes (id) VALUES ($1)", fish_id)
        await connection.execute(
            """
            INSERT INTO agent_endpoints (id, connector_type, base_url)
            VALUES ($1, 'example.connector', 'http://localhost:9100')
            """,
            endpoint_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_targets (
                id, endpoint_id, external_key, route_prefix, connector_config,
                display_name, enabled
            )
            VALUES ($1, $2, 'default', '', '{}'::jsonb, 'Example', true)
            """,
            target_id,
            endpoint_id,
        )
        await connection.execute(
            "UPDATE fishes SET agent_target_id = $2 WHERE id = $1",
            fish_id,
            target_id,
        )

        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute(
                """
                INSERT INTO agent_endpoints (id, connector_type, base_url)
                VALUES ($1, 'Not-a-slug', 'http://localhost:9200')
                """,
                uuid4(),
            )
        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute(
                """
                INSERT INTO agent_targets (
                    id, endpoint_id, external_key, route_prefix, connector_config,
                    display_name, enabled
                )
                VALUES ($1, $2, 'other', '', '[]'::jsonb, 'Other', true)
                """,
                uuid4(),
                endpoint_id,
            )

        fact_status_constraint = await connection.fetchval(
            """
            SELECT pg_get_constraintdef(oid)
            FROM pg_constraint
            WHERE conname = 'messages_fact_extraction_status_check'
            """
        )
        lineage_index = await connection.fetchval(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE indexname = 'idx_agent_runs_one_child_per_previous'
            """
        )

    assert {row["relname"] for row in table_names} == {
        "agent_endpoints",
        "agent_targets",
        "agent_tasks",
        "agent_runs",
        "agent_interactions",
    }
    assert fact_status_constraint is not None
    assert "waiting_consumption" in fact_status_constraint
    assert lineage_index is not None
    assert "UNIQUE INDEX" in lineage_index
    assert "WHERE (previous_run_id IS NOT NULL)" in lineage_index


@pytest.mark.asyncio
async def test_agent_task_session_scope_migration_backfills_and_enforces_message_scope(
    postgres_pool: asyncpg.Pool,
) -> None:
    fish_id = uuid4()
    session_id = uuid4()
    other_session_id = uuid4()
    endpoint_id = uuid4()
    target_id = uuid4()
    task_id = uuid4()
    run_id = uuid4()
    source_id = uuid4()
    confirmation_id = uuid4()
    other_message_id = uuid4()

    async with postgres_pool.acquire() as connection:
        await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
        for migration in MIGRATIONS:
            if migration.version >= 33:
                break
            await connection.execute(migration.sql)
            await connection.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                migration.version,
            )
        await connection.execute("INSERT INTO fishes (id) VALUES ($1)", fish_id)
        await connection.execute(
            "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2), ($3, $2)",
            session_id,
            fish_id,
            other_session_id,
        )
        await connection.execute(
            """
            INSERT INTO messages (id, fish_id, session_id, role, content, status)
            VALUES
                ($1, $2, $3, 'user', 'start', 'initialized'),
                ($4, $2, $3, 'assistant', 'accepted', 'completed'),
                ($5, $2, $6, 'user', 'other session', 'initialized')
            """,
            source_id,
            fish_id,
            session_id,
            confirmation_id,
            other_message_id,
            other_session_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_endpoints (id, connector_type, base_url)
            VALUES ($1, 'hermes', 'http://localhost:9000')
            """,
            endpoint_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_targets (
                id, endpoint_id, external_key, display_name
            ) VALUES ($1, $2, 'default', 'Hermes')
            """,
            target_id,
            endpoint_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_tasks (
                id, fish_id, target_id, external_session_id, title
            ) VALUES ($1, $2, $3, 'external-session', 'Scoped task')
            """,
            task_id,
            fish_id,
            target_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_runs (
                id, task_id, target_id, source_message_id,
                confirmation_message_id, input, status
            ) VALUES ($1, $2, $3, $4, $5, 'start', 'queued')
            """,
            run_id,
            task_id,
            target_id,
            source_id,
            confirmation_id,
        )

        migration = next(item for item in MIGRATIONS if item.version == 33)
        await connection.execute(migration.sql)

        scoped = await connection.fetchrow(
            """
            SELECT task.session_id AS task_session_id,
                   run.fish_id AS run_fish_id,
                   run.session_id AS run_session_id
            FROM agent_tasks AS task
            JOIN agent_runs AS run ON run.task_id = task.id
            WHERE task.id = $1
            """,
            task_id,
        )
        assert scoped is not None
        assert dict(scoped) == {
            "task_session_id": session_id,
            "run_fish_id": fish_id,
            "run_session_id": session_id,
        }
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await connection.execute(
                "UPDATE agent_runs SET result_message_id = $2 WHERE id = $1",
                run_id,
                other_message_id,
            )


@pytest.mark.asyncio
async def test_agent_task_session_scope_migration_rejects_ambiguous_legacy_task(
    postgres_pool: asyncpg.Pool,
) -> None:
    fish_id = uuid4()
    first_session_id = uuid4()
    second_session_id = uuid4()
    endpoint_id = uuid4()
    target_id = uuid4()
    task_id = uuid4()

    async with postgres_pool.acquire() as connection:
        await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
        for migration in MIGRATIONS:
            if migration.version >= 33:
                break
            await connection.execute(migration.sql)
            await connection.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1)",
                migration.version,
            )
        await connection.execute("INSERT INTO fishes (id) VALUES ($1)", fish_id)
        await connection.execute(
            "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $3), ($2, $3)",
            first_session_id,
            second_session_id,
            fish_id,
        )
        source_ids = (uuid4(), uuid4())
        await connection.execute(
            """
            INSERT INTO messages (id, fish_id, session_id, role, content, status)
            VALUES
                ($1, $3, $4, 'user', 'first', 'initialized'),
                ($2, $3, $5, 'user', 'second', 'initialized')
            """,
            source_ids[0],
            source_ids[1],
            fish_id,
            first_session_id,
            second_session_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_endpoints (id, connector_type, base_url)
            VALUES ($1, 'hermes', 'http://localhost:9000')
            """,
            endpoint_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_targets (id, endpoint_id, external_key, display_name)
            VALUES ($1, $2, 'default', 'Hermes')
            """,
            target_id,
            endpoint_id,
        )
        await connection.execute(
            """
            INSERT INTO agent_tasks (
                id, fish_id, target_id, external_session_id, title
            ) VALUES ($1, $2, $3, 'external', 'ambiguous')
            """,
            task_id,
            fish_id,
            target_id,
        )
        for source_id in source_ids:
            await connection.execute(
                """
                INSERT INTO agent_runs (
                    id, task_id, target_id, source_message_id, input, status
                )
                VALUES ($1, $2, $3, $4, 'start', 'completed')
                """,
                uuid4(),
                task_id,
                target_id,
                source_id,
            )

        migration = next(item for item in MIGRATIONS if item.version == 33)
        with pytest.raises(asyncpg.RaiseError, match="cannot migrate Agent Tasks"):
            await connection.execute(migration.sql)
