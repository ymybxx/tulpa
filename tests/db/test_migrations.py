from types import TracebackType
from typing import Any

import pytest

from fish_memory.db import migrations
from fish_memory.db.migrations import Migration, run_migrations


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeConnection:
    def __init__(
        self,
        applied_versions: set[int] | None = None,
        fail_on: str | None = None,
    ) -> None:
        self.applied_versions = set(applied_versions or set())
        self.fail_on = fail_on
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        normalized = " ".join(query.split())
        self.executed.append((normalized, args))
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("migration failed")
        if normalized.startswith("INSERT INTO schema_migrations"):
            self.applied_versions.add(args[0])
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, int]]:
        del args
        self.executed.append((" ".join(query.split()), ()))
        return [{"version": version} for version in sorted(self.applied_versions)]

    def transaction(self) -> FakeTransaction:
        return FakeTransaction()


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


@pytest.mark.asyncio
async def test_runs_unapplied_migrations_in_version_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection()
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (Migration(2, "migration two"), Migration(1, "migration one")),
    )

    await run_migrations(FakePool(connection))

    statements = [query for query, _ in connection.executed]
    assert statements.index("migration one") < statements.index("migration two")
    assert connection.applied_versions == {1, 2}
    assert "pg_advisory_lock" in statements[0]
    assert "pg_advisory_unlock" in statements[-1]


@pytest.mark.asyncio
async def test_skips_versions_already_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(applied_versions={1})
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (Migration(1, "already applied"), Migration(2, "new migration")),
    )

    await run_migrations(FakePool(connection))

    statements = [query for query, _ in connection.executed]
    assert "already applied" not in statements
    assert "new migration" in statements


@pytest.mark.asyncio
async def test_releases_advisory_lock_when_migration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(fail_on="broken migration")
    monkeypatch.setattr(
        migrations,
        "MIGRATIONS",
        (Migration(1, "broken migration"),),
    )

    with pytest.raises(RuntimeError, match="migration failed"):
        await run_migrations(FakePool(connection))

    assert "pg_advisory_unlock" in connection.executed[-1][0]


def test_initial_schema_contains_required_constraints_and_indexes() -> None:
    schema_sql = migrations.MIGRATIONS[0].sql

    assert "CREATE TABLE messages" in schema_sql
    assert "GENERATED ALWAYS AS IDENTITY" in schema_sql
    assert "idx_messages_fish_sequence" in schema_sql
    assert "fish_id, sequence DESC" in schema_sql
    assert "idx_messages_one_generating_answer_per_fish" in schema_sql
    assert "WHERE role = 'assistant' AND status = 'generating'" in schema_sql
    assert "consumed_offset" in schema_sql
    assert "context_through_sequence" in schema_sql


def test_second_migration_creates_message_performance_steps() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 2)

    assert "CREATE TABLE message_performance_steps" in migration.sql
    assert (
        "message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE"
        in migration.sql
    )
    assert "input_payload JSONB" in migration.sql
    assert "output_payload JSONB" in migration.sql
    assert "status" not in migration.sql
    assert "started_offset_us >= 0 AND duration_us >= 0" in migration.sql
    assert "idx_message_performance_steps_message" in migration.sql
    assert "idx_message_performance_steps_name_duration" in migration.sql


def test_third_migration_creates_and_backfills_fishes() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 3)

    assert "CREATE TABLE fishes" in migration.sql
    assert "deleted_at TIMESTAMPTZ" in migration.sql
    assert "INSERT INTO fishes (id, created_at)" in migration.sql
    assert "SELECT fish_id, min(created_at)" in migration.sql
    assert "FOREIGN KEY (fish_id) REFERENCES fishes(id)" in migration.sql
    assert migration.sql.index("INSERT INTO fishes") < migration.sql.index(
        "ADD CONSTRAINT"
    )
    assert "ON DELETE CASCADE" not in migration.sql


def test_fourth_migration_creates_facts_and_task_progress() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 4)

    assert "CREATE TABLE facts" in migration.sql
    assert "sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE" in migration.sql
    assert "source_message_id UUID NOT NULL REFERENCES messages(id)" in migration.sql
    assert "entities JSONB NOT NULL DEFAULT '[]'::jsonb" in migration.sql
    assert "UNIQUE (source_message_id, fact_index)" in migration.sql
    assert "occurred_end IS NULL OR occurred_start IS NOT NULL" in migration.sql
    assert "occurred_end >= occurred_start" in migration.sql
    assert "CREATE TABLE background_task_progress" in migration.sql
    assert "PRIMARY KEY (fish_id, task_name)" in migration.sql
    assert "through_sequence BIGINT NOT NULL DEFAULT 0" in migration.sql


def test_twelfth_migration_adds_fact_structured_semantics() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 12)

    assert "ADD COLUMN structured JSONB NOT NULL DEFAULT '{}'::jsonb" in migration.sql
    assert "jsonb_typeof(structured) = 'object'" in migration.sql
    assert "jsonb_build_object" in migration.sql
    assert "'what', text" in migration.sql


def test_fifth_migration_creates_memory_snapshots() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 5)

    assert "CREATE TABLE memory_snapshots" in migration.sql
    assert "fish_id UUID NOT NULL REFERENCES fishes(id)" in migration.sql
    assert "snapshot_type TEXT NOT NULL" in migration.sql
    assert "content JSONB NOT NULL" in migration.sql
    assert "through_fact_sequence BIGINT NOT NULL DEFAULT 0" in migration.sql
    assert "version BIGINT NOT NULL DEFAULT 1" in migration.sql
    assert "PRIMARY KEY (fish_id, snapshot_type)" in migration.sql
    assert "jsonb_typeof(content) = 'object'" in migration.sql


def test_thirteenth_migration_creates_snapshot_versions() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 13)

    assert "CREATE TABLE memory_snapshot_versions" in migration.sql
    assert "snapshot_type TEXT NOT NULL" in migration.sql
    assert "content JSONB NOT NULL" in migration.sql
    assert "PRIMARY KEY (fish_id, snapshot_type, version)" in migration.sql
    assert "INSERT INTO memory_snapshot_versions" in migration.sql


def test_fourteenth_migration_adds_fact_invalidation_and_relation_groups() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 14)

    assert "ALTER TABLE facts" in migration.sql
    assert "ADD COLUMN invalidated_at TIMESTAMPTZ" in migration.sql
    assert "CREATE TABLE fact_relation_groups" in migration.sql
    assert "fact_ids UUID[] NOT NULL" in migration.sql
    assert "USING GIN (fact_ids)" in migration.sql


def test_twenty_fourth_migration_creates_home_action_runs() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 24)

    assert "CREATE TABLE home_action_runs" in migration.sql
    assert "sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE" in migration.sql
    assert "source_message_id UUID NOT NULL REFERENCES messages(id)" in migration.sql
    assert (
        "status IN ('routing', 'executing', 'succeeded', 'failed', 'ambiguous')"
        in migration.sql
    )
    assert "idx_home_action_runs_scope_sequence" in migration.sql
    assert "idx_home_action_runs_source_message" in migration.sql


def test_twenty_fifth_migration_adds_home_action_status_event_cursor() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 25)

    assert "CREATE SEQUENCE home_action_run_events_sequence_seq" in migration.sql
    assert "ADD COLUMN event_sequence BIGINT" in migration.sql
    assert "SET event_sequence = nextval" in migration.sql
    assert "idx_home_action_runs_scope_event_sequence" in migration.sql


def test_twenty_sixth_migration_adds_partial_home_action_status() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 26)

    assert "DROP CONSTRAINT home_action_runs_status_check" in migration.sql
    assert "'partial'" in migration.sql


def test_migration_27_creates_relation_embeddings_without_backfill() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 27)

    assert "CREATE TABLE fact_relation_group_embeddings" in migration.sql
    assert "relation_group_id BIGINT PRIMARY KEY" in migration.sql
    assert "REFERENCES fact_relation_groups(id) ON DELETE CASCADE" in migration.sql
    assert (
        "fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE" in migration.sql
    )
    assert (
        "session_id UUID REFERENCES fish_sessions(id) ON DELETE CASCADE"
        in migration.sql
    )
    assert "snapshot_text TEXT NOT NULL" in migration.sql
    assert "text_hash CHAR(64) NOT NULL" in migration.sql
    assert "last_supported_at TIMESTAMPTZ(0) NOT NULL" in migration.sql
    assert "vector_dims(embedding) = dimensions" in migration.sql
    assert "idx_fact_relation_group_embeddings_static_scope" in migration.sql
    assert "idx_fact_relation_group_embeddings_dynamic_scope" in migration.sql
    assert "INSERT INTO fact_relation_group_embeddings" not in migration.sql


def test_migration_28_adds_relation_head_and_mask_embedding_metadata() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 28)

    assert "ADD COLUMN member_count INTEGER" in migration.sql
    assert "UPDATE fact_relation_group_embeddings AS embedding" in migration.sql
    assert "cardinality(relation_group.fact_ids)" in migration.sql
    assert "fact_embeddings" in migration.sql
    assert "ADD COLUMN embedding_text TEXT" in migration.sql
    assert "ADD COLUMN text_hash CHAR(64)" in migration.sql
    assert "CREATE TABLE fact_mask_embeddings" in migration.sql
    assert "CREATE TABLE fact_relation_group_mask_embeddings" in migration.sql
    assert "entity_ids UUID[] NOT NULL" in migration.sql
    assert "signature_hash CHAR(64) NOT NULL" in migration.sql
    assert "source_text_hash CHAR(64) NOT NULL" in migration.sql
    assert "dimensions INTEGER NOT NULL" in migration.sql
    assert "vector_dims(embedding) = dimensions" in migration.sql
    assert "USING hnsw" not in migration.sql
    assert "INSERT INTO fact_mask_embeddings" not in migration.sql
    assert "INSERT INTO fact_relation_group_mask_embeddings" not in migration.sql


def test_thirtieth_migration_creates_normalized_snapshot_items() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 30)

    assert "snapshot_storage_version SMALLINT NOT NULL DEFAULT 1" in migration.sql
    assert "ALTER COLUMN snapshot_storage_version SET DEFAULT 2" in migration.sql
    assert "snapshot_storage_version IN (1, 2)" in migration.sql
    assert "ALTER COLUMN content DROP NOT NULL" in migration.sql
    assert "CREATE TABLE memory_snapshot_revisions" in migration.sql
    assert "CREATE TABLE memory_snapshot_items" in migration.sql
    assert "support_count INTEGER GENERATED ALWAYS" in migration.sql
    assert "support_fact_ids <@ lineage_fact_ids" in migration.sql
    assert "USING GIN (support_fact_ids)" in migration.sql
    assert "CREATE TABLE memory_snapshot_item_versions" in migration.sql
    assert "action IN ('create', 'update')" in migration.sql
    assert "char_length(basis) <= 100" in migration.sql
    assert "category" not in migration.sql
    assert "status" not in migration.sql
    assert "deprecated" not in migration.sql


def test_thirty_first_migration_creates_empty_item_relation_projections() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 31)

    assert "CREATE TABLE memory_snapshot_item_embeddings" in migration.sql
    assert "snapshot_item_id BIGINT PRIMARY KEY" in migration.sql
    assert "REFERENCES memory_snapshot_items(id) ON DELETE CASCADE" in migration.sql
    assert "item_version BIGINT NOT NULL" in migration.sql
    assert "fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE" in (
        migration.sql
    )
    assert "session_id UUID REFERENCES fish_sessions(id) ON DELETE CASCADE" in (
        migration.sql
    )
    assert "snapshot_type TEXT NOT NULL" in migration.sql
    assert "snapshot_text TEXT NOT NULL" in migration.sql
    assert "text_hash CHAR(64) NOT NULL" in migration.sql
    assert "last_supported_at TIMESTAMPTZ(0) NOT NULL" in migration.sql
    assert "provider TEXT NOT NULL" in migration.sql
    assert "model TEXT NOT NULL" in migration.sql
    assert "dimensions INTEGER NOT NULL" in migration.sql
    assert "embedding VECTOR NOT NULL" in migration.sql
    assert "item_version >= 1" in migration.sql
    assert "text_hash ~ '^[0-9a-f]{64}$'" in migration.sql
    assert "vector_dims(embedding) = dimensions" in migration.sql
    assert "snapshot_type = 'static' AND session_id IS NULL" in migration.sql
    assert "snapshot_type = 'dynamic' AND session_id IS NOT NULL" in migration.sql

    assert "CREATE TABLE memory_snapshot_item_mask_embeddings" in migration.sql
    assert (
        "REFERENCES memory_snapshot_item_embeddings(snapshot_item_id)" in migration.sql
    )
    assert "PRIMARY KEY (snapshot_item_id, signature_hash, dimensions)" in (
        migration.sql
    )
    assert "ON DELETE CASCADE" in migration.sql
    assert "USING hnsw" not in migration.sql

    assert "support_count" not in migration.sql
    assert "support_fact_ids" not in migration.sql
    assert "lineage_fact_ids" not in migration.sql
    assert "snapshot_version" not in migration.sql
    assert "fact_relation_groups" not in migration.sql
    assert "fact_relation_group_embeddings" not in migration.sql
    assert "fact_relation_group_mask_embeddings" not in migration.sql
    assert "memory_snapshot_versions" not in migration.sql
    assert "INSERT INTO" not in migration.sql
    assert "UPDATE " not in migration.sql


def test_thirty_second_migration_creates_provider_neutral_agent_task_schema() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 32)

    for table in (
        "agent_endpoints",
        "agent_targets",
        "agent_tasks",
        "agent_runs",
        "agent_interactions",
    ):
        assert f"CREATE TABLE {table}" in migration.sql

    assert "ADD COLUMN agent_target_id UUID" in migration.sql
    assert "waiting_consumption" in migration.sql
    assert "DROP CONSTRAINT messages_fact_extraction_status_check" in migration.sql
    assert "^[a-z][a-z0-9._-]{0,63}$" in migration.sql
    assert "UNIQUE (connector_type, base_url)" in migration.sql
    assert "UNIQUE (endpoint_id, external_key, route_prefix)" in migration.sql
    assert "jsonb_typeof(connector_config) = 'object'" in migration.sql
    assert "UNIQUE (id, target_id)" in migration.sql
    assert "FOREIGN KEY (task_id, target_id)" in migration.sql
    assert "FOREIGN KEY (previous_run_id, task_id)" in migration.sql
    assert "idx_agent_runs_one_active_per_task" in migration.sql
    assert "idx_agent_runs_one_child_per_previous" in migration.sql
    assert "WHERE previous_run_id IS NOT NULL" in migration.sql
    assert "idx_agent_interactions_one_open_per_run" in migration.sql
    assert "resolved_at IS NULL AND closed_at IS NULL" in migration.sql
    assert "UNIQUE (target_id, external_run_id)" in migration.sql
    assert "request_message_id UUID UNIQUE" in migration.sql
    assert "response_message_id UUID UNIQUE" in migration.sql
    assert "next_run_id UUID UNIQUE" in migration.sql

    for absent in (
        "agent_task_notifications",
        "agent_task_events",
        "agent_outbox",
        "agent_task_message_links",
        "review_state",
        "announced_at",
    ):
        assert absent not in migration.sql


def test_thirty_third_migration_binds_agent_hierarchy_to_one_fish_session() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 33)

    assert "ALTER TABLE agent_tasks" in migration.sql
    assert "ADD COLUMN session_id UUID" in migration.sql
    assert "cannot migrate Agent Tasks" in migration.sql
    assert "FOREIGN KEY (fish_id, session_id)" in migration.sql
    assert "FOREIGN KEY (source_message_id, fish_id, session_id)" in migration.sql
    assert "FOREIGN KEY (confirmation_message_id, fish_id, session_id)" in migration.sql
    assert "FOREIGN KEY (result_message_id, fish_id, session_id)" in migration.sql
    assert "FOREIGN KEY (request_message_id, fish_id, session_id)" in migration.sql
    assert "FOREIGN KEY (response_message_id, fish_id, session_id)" in migration.sql
    assert "FOREIGN KEY (next_run_id, fish_id, session_id)" in migration.sql


def test_thirty_fourth_migration_records_fish_session_history_source() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 34)

    assert "DROP CONSTRAINT agent_runs_history_source_check" in migration.sql
    assert "'fish_session'" in migration.sql


def test_thirty_fifth_migration_records_combined_history_sources() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 35)

    assert "'fish_session_hermes_session'" in migration.sql
    assert "'fish_session_local_projection'" in migration.sql


def test_thirty_sixth_migration_creates_isolated_observation_storage() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 36)

    assert "CREATE TABLE observations" in migration.sql
    assert "CREATE TABLE observation_app_policies" in migration.sql
    assert "REFERENCES fishes(id)" in migration.sql
    assert "kind IN ('state', 'event')" in migration.sql
    assert "idx_observations_state_stream" in migration.sql
    for forbidden in ("session_id", "message_id", "fact_id", "snapshot_id"):
        assert forbidden not in migration.sql


def test_thirty_seventh_migration_adds_background_assistant_delivery_metadata() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 37)

    assert "CREATE TABLE background_assistant_deliveries" in migration.sql
    assert "message_id UUID PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE" in (
        migration.sql
    )
    assert "source IN ('agent', 'proactive_observation')" in migration.sql
    assert "interrupt_policy IN ('resume', 'drop')" in migration.sql
    assert (
        "source_observation_id UUID REFERENCES observations(id) ON DELETE SET NULL"
        in (migration.sql)
    )
    assert "INSERT INTO background_assistant_deliveries" in migration.sql
    assert "SELECT message.id, 'agent', 'neutral', 'resume'" in migration.sql
    assert "idx_background_assistant_deliveries_one_per_observation" in migration.sql


def test_thirty_eighth_migration_adds_temporary_computer_history_checkpoints() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 38)

    assert "CREATE TABLE computer_history_summary_checkpoints" in migration.sql
    assert "PRIMARY KEY (fish_id, segment_id)" in migration.sql
    assert "through_event_count" in migration.sql
    assert "delta_chain_digest" in migration.sql
    assert "delta_capsules JSONB" in migration.sql
    assert "status IN ('active', 'finalized')" in migration.sql
    assert (
        "observation_id UUID REFERENCES observations(id) ON DELETE CASCADE"
        in migration.sql
    )
    for forbidden in ("message_id", "fact_id", "snapshot_id"):
        assert forbidden not in migration.sql


def test_fifteenth_migration_creates_entity_alias_and_fact_indexes() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 15)

    assert "CREATE TABLE entity_aliases" in migration.sql
    assert "PRIMARY KEY (fish_id, alias)" in migration.sql
    assert "entity_id UUID NOT NULL" in migration.sql
    assert "idx_entity_aliases_entity" in migration.sql
    assert "CREATE TABLE fact_entities" in migration.sql
    assert "fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE" in (
        migration.sql
    )
    assert "PRIMARY KEY (entity_id, fact_id)" in migration.sql
    assert "idx_fact_entities_fact" in migration.sql
    assert "jsonb_array_elements_text(fact.entities)" in migration.sql
    assert "INSERT INTO entity_aliases" in migration.sql
    assert "INSERT INTO fact_entities" in migration.sql


def test_sixteenth_migration_adds_optional_fish_soul() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 16)

    assert "ALTER TABLE fishes" in migration.sql
    assert "ADD COLUMN soul TEXT" in migration.sql


def test_seventeenth_migration_adds_structured_fish_role_settings() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 17)

    assert "ALTER TABLE fishes" in migration.sql
    assert "ADD COLUMN role TEXT" in migration.sql
    assert "ADD COLUMN setting TEXT" in migration.sql
    assert "ADD COLUMN name TEXT" in migration.sql


def test_eighteenth_migration_normalizes_database_timestamps_to_seconds() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 18)

    assert "TIMESTAMPTZ(0)" in migration.sql
    assert "USING date_trunc('second', created_at)" in migration.sql
    assert "ALTER TABLE messages" in migration.sql
    assert "ALTER TABLE facts" in migration.sql
    assert "ALTER TABLE memory_snapshots" in migration.sql
    assert "ALTER TABLE memory_snapshot_versions" in migration.sql
    assert "ALTER TABLE evaluation_questions" in migration.sql


def test_nineteenth_migration_adds_single_assistant_fish_binding() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 19)

    assert "ADD COLUMN is_assistant BOOLEAN NOT NULL DEFAULT FALSE" in migration.sql
    assert "idx_fishes_one_assistant" in migration.sql
    assert "WHERE is_assistant AND deleted_at IS NULL" in migration.sql


def test_session_scope_migration_is_registered_after_existing_schema() -> None:
    assert any(migration.version == 20 for migration in migrations.MIGRATIONS)


def test_twenty_first_migration_indexes_mentioned_time_recall() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 21)

    assert "idx_facts_fish_mentioned_at" in migration.sql
    assert "ON facts (fish_id, mentioned_at)" in migration.sql


def test_twenty_second_migration_adds_per_fish_embedding_spaces() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 22)

    assert "ADD COLUMN embedding_provider TEXT" in migration.sql
    assert "ADD COLUMN embedding_model TEXT" in migration.sql
    assert "ADD COLUMN embedding_dimensions INTEGER" in migration.sql
    assert "ALTER COLUMN embedding TYPE VECTOR" in migration.sql
    assert "ADD COLUMN provider TEXT" in migration.sql
    assert "ADD COLUMN dimensions INTEGER" in migration.sql
    assert "vector_dims(embedding) = dimensions" in migration.sql
    assert "DROP INDEX idx_fact_embeddings_hnsw" in migration.sql
    assert "CREATE INDEX idx_fact_embeddings_hnsw_384" not in migration.sql
    assert "CREATE INDEX idx_fact_embeddings_hnsw_1024" not in migration.sql
    assert "openai-compatible" in migration.sql


def test_sixth_migration_simplifies_snapshot_categories() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 6)

    assert "WHEN 'recent_event' THEN 'event'" in migration.sql
    assert "'state'" in migration.sql
    assert "jsonb_array_elements" in migration.sql
    assert "jsonb_agg" in migration.sql
    assert "(entry.item ->> 'lastSupportedAt')::timestamptz" in migration.sql


def test_seventh_migration_creates_hnsw_fact_embeddings() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 7)

    assert "CREATE EXTENSION IF NOT EXISTS vector" in migration.sql
    assert "CREATE TABLE fact_embeddings" in migration.sql
    assert "fact_id UUID PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE" in (
        migration.sql
    )
    assert "fish_id UUID NOT NULL REFERENCES fishes(id)" in migration.sql
    assert "embedding VECTOR(1024) NOT NULL" in migration.sql
    assert "idx_fact_embeddings_fish" in migration.sql
    assert "USING hnsw (embedding vector_cosine_ops)" in migration.sql


def test_tenth_migration_persists_incremental_history_token_window() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 10)

    assert "ADD COLUMN history_token_count INTEGER NOT NULL DEFAULT 0" in migration.sql
    assert (
        "ADD COLUMN history_window_start_sequence BIGINT NOT NULL DEFAULT 0"
        in migration.sql
    )
    assert (
        "ADD COLUMN history_window_token_count BIGINT NOT NULL DEFAULT 0"
        in migration.sql
    )
    assert "history_token_count >= 0" in migration.sql
    assert "history_window_token_count >= 0" in migration.sql


def test_eleventh_migration_persists_message_level_fact_failures() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 11)

    assert (
        "ADD COLUMN fact_extraction_status TEXT NOT NULL DEFAULT 'pending'"
        in migration.sql
    )
    assert (
        "ADD COLUMN fact_extraction_attempt_count INTEGER NOT NULL DEFAULT 0"
        in migration.sql
    )
    assert "ADD COLUMN fact_extraction_error TEXT" in migration.sql
    assert (
        "fact_extraction_status IN ('pending', 'completed', 'failed')" in migration.sql
    )
    assert "fact_extraction_attempt_count >= 0" in migration.sql


def test_eighth_migration_creates_evaluation_questions() -> None:
    migration = next(item for item in migrations.MIGRATIONS if item.version == 8)

    assert "CREATE TABLE evaluation_questions" in migration.sql
    assert "question_id TEXT PRIMARY KEY" in migration.sql
    assert "gold_answer JSONB NOT NULL" in migration.sql
    assert "item JSONB NOT NULL" in migration.sql
    assert "result JSONB" in migration.sql
