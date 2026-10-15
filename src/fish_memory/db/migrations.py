from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, cast

from fish_memory.history.schema import (
    SESSION_CONTEXT_PROGRESS_SQL,
    SESSION_HISTORY_SQL,
    SESSION_HISTORY_VISIBILITY_SQL,
)

ADVISORY_LOCK_ID = 1_572_077_401

CREATE_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

INITIAL_SCHEMA_SQL = """
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    fish_id UUID NOT NULL,
    sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    status VARCHAR(16) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    context_through_sequence BIGINT,
    consumed_offset INTEGER,
    finished_at TIMESTAMPTZ,
    CONSTRAINT messages_role_status_check CHECK (
        (role = 'user' AND status = 'initialized')
        OR
        (role = 'assistant' AND status IN (
            'generating', 'completed', 'interrupted', 'failed'
        ))
    ),
    CONSTRAINT messages_consumed_offset_check CHECK (
        consumed_offset IS NULL OR consumed_offset >= -1
    )
);

CREATE INDEX idx_messages_fish_sequence
ON messages (fish_id, sequence DESC);

CREATE UNIQUE INDEX idx_messages_one_generating_answer_per_fish
ON messages (fish_id)
WHERE role = 'assistant' AND status = 'generating';
"""

PERFORMANCE_STEPS_SQL = """
CREATE TABLE message_performance_steps (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    step_index INTEGER NOT NULL DEFAULT 0,
    started_offset_us BIGINT NOT NULL,
    duration_us BIGINT NOT NULL,
    input_payload JSONB,
    output_payload JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT message_performance_offset_check CHECK (
        started_offset_us >= 0 AND duration_us >= 0
    )
);

CREATE INDEX idx_message_performance_steps_message
ON message_performance_steps (message_id, started_offset_us, id);

CREATE INDEX idx_message_performance_steps_name_duration
ON message_performance_steps (step_name, duration_us);
"""

FISH_LIFECYCLE_SQL = """
CREATE TABLE fishes (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ
);

INSERT INTO fishes (id, created_at)
SELECT fish_id, min(created_at)
FROM messages
GROUP BY fish_id;

ALTER TABLE messages
ADD CONSTRAINT messages_fish_id_fkey
FOREIGN KEY (fish_id) REFERENCES fishes(id);
"""

FACTS_AND_BACKGROUND_PROGRESS_SQL = """
CREATE TABLE facts (
    id UUID PRIMARY KEY,
    sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    fish_id UUID NOT NULL REFERENCES fishes(id),
    source_message_id UUID NOT NULL REFERENCES messages(id),
    fact_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    occurred_start TIMESTAMPTZ,
    occurred_end TIMESTAMPTZ,
    mentioned_at TIMESTAMPTZ NOT NULL,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_message_id, fact_index),
    CHECK (fact_index >= 0),
    CHECK (jsonb_typeof(entities) = 'array'),
    CHECK (occurred_end IS NULL OR occurred_start IS NOT NULL),
    CHECK (
        occurred_start IS NULL
        OR occurred_end IS NULL
        OR occurred_end >= occurred_start
    )
);

CREATE INDEX idx_facts_fish_sequence ON facts (fish_id, sequence);
CREATE INDEX idx_facts_fish_occurred_start
ON facts (fish_id, occurred_start)
WHERE occurred_start IS NOT NULL;
CREATE INDEX idx_facts_entities_gin ON facts USING GIN (entities);

CREATE TABLE background_task_progress (
    fish_id UUID NOT NULL REFERENCES fishes(id),
    task_name TEXT NOT NULL,
    through_sequence BIGINT NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fish_id, task_name),
    CHECK (through_sequence >= 0)
);
"""

DYNAMIC_SNAPSHOT_SQL = """
CREATE TABLE memory_snapshots (
    fish_id UUID NOT NULL REFERENCES fishes(id),
    snapshot_type TEXT NOT NULL,
    content JSONB NOT NULL,
    through_fact_sequence BIGINT NOT NULL DEFAULT 0,
    version BIGINT NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fish_id, snapshot_type),
    CHECK (through_fact_sequence >= 0),
    CHECK (version >= 1),
    CHECK (jsonb_typeof(content) = 'object')
);
"""

SIMPLIFY_SNAPSHOT_CATEGORIES_SQL = """
UPDATE memory_snapshots
SET content = jsonb_set(
    content,
    '{items}',
    COALESCE(
        (
            SELECT jsonb_agg(
                jsonb_set(
                    entry.item,
                    '{category}',
                    to_jsonb(
                        CASE entry.item ->> 'category'
                            WHEN 'recent_event' THEN 'event'
                            ELSE 'state'
                        END
                    )
                )
                ORDER BY
                    (entry.item ->> 'lastSupportedAt')::timestamptz,
                    entry.position
            )
            FROM jsonb_array_elements(content -> 'items')
                WITH ORDINALITY AS entry(item, position)
        ),
        '[]'::jsonb
    )
)
WHERE snapshot_type = 'dynamic'
  AND jsonb_typeof(content -> 'items') = 'array'
"""

FACT_EMBEDDINGS_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE fact_embeddings (
    fact_id UUID PRIMARY KEY REFERENCES facts(id) ON DELETE CASCADE,
    fish_id UUID NOT NULL REFERENCES fishes(id),
    model TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_fact_embeddings_fish
ON fact_embeddings (fish_id);

CREATE INDEX idx_fact_embeddings_hnsw
ON fact_embeddings
USING hnsw (embedding vector_cosine_ops);
"""

EVALUATION_QUESTIONS_SQL = """
CREATE TABLE evaluation_questions (
    question_id TEXT PRIMARY KEY,
    question_type TEXT NOT NULL,
    question TEXT NOT NULL,
    gold_answer JSONB NOT NULL,
    question_date TIMESTAMPTZ NOT NULL,
    fish_id UUID NOT NULL,
    item JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'imported',
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_evaluation_questions_updated
ON evaluation_questions (updated_at DESC, question_id);
"""

BACKGROUND_OWNER_SQL = """
ALTER TABLE fishes
ADD COLUMN background_owner UUID;

CREATE INDEX idx_fishes_background_owner
ON fishes (background_owner)
WHERE deleted_at IS NULL;
"""

HISTORY_TOKEN_WINDOW_SQL = """
ALTER TABLE messages
ADD COLUMN history_token_count INTEGER NOT NULL DEFAULT 0
CHECK (history_token_count >= 0);

ALTER TABLE fishes
ADD COLUMN history_window_start_sequence BIGINT NOT NULL DEFAULT 0
CHECK (history_window_start_sequence >= 0),
ADD COLUMN history_window_token_count BIGINT NOT NULL DEFAULT 0
CHECK (history_window_token_count >= 0);
"""

FACT_EXTRACTION_FAILURES_SQL = """
ALTER TABLE messages
ADD COLUMN fact_extraction_status TEXT NOT NULL DEFAULT 'pending'
CHECK (fact_extraction_status IN ('pending', 'completed', 'failed')),
ADD COLUMN fact_extraction_attempt_count INTEGER NOT NULL DEFAULT 0
CHECK (fact_extraction_attempt_count >= 0),
ADD COLUMN fact_extraction_error TEXT;
"""

FACT_STRUCTURED_SEMANTICS_SQL = """
ALTER TABLE facts
ADD COLUMN structured JSONB NOT NULL DEFAULT '{}'::jsonb;

UPDATE facts
SET structured = jsonb_build_object(
    'what', text,
    'where', '[]'::jsonb,
    'who', '[]'::jsonb
);

ALTER TABLE facts
ADD CONSTRAINT facts_structured_is_object
CHECK (jsonb_typeof(structured) = 'object');
"""

SNAPSHOT_VERSIONS_SQL = """
CREATE TABLE memory_snapshot_versions (
    fish_id UUID NOT NULL REFERENCES fishes(id),
    snapshot_type TEXT NOT NULL,
    version BIGINT NOT NULL,
    content JSONB NOT NULL,
    through_fact_sequence BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fish_id, snapshot_type, version),
    CHECK (version >= 1),
    CHECK (through_fact_sequence >= 0),
    CHECK (jsonb_typeof(content) = 'object')
);

CREATE INDEX idx_memory_snapshot_versions_lookup
ON memory_snapshot_versions (fish_id, snapshot_type, version DESC);

INSERT INTO memory_snapshot_versions (
    fish_id,
    snapshot_type,
    version,
    content,
    through_fact_sequence,
    created_at
)
SELECT
    fish_id,
    snapshot_type,
    version,
    content,
    through_fact_sequence,
    updated_at
FROM memory_snapshots;
"""

FACT_RELATIONS_SQL = """
ALTER TABLE facts
ADD COLUMN invalidated_at TIMESTAMPTZ;

CREATE TABLE fact_relation_groups (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fish_id UUID NOT NULL REFERENCES fishes(id),
    snapshot_type TEXT NOT NULL,
    fact_ids UUID[] NOT NULL,
    snapshot_version BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (snapshot_type IN ('dynamic', 'static')),
    CHECK (cardinality(fact_ids) > 0)
);

CREATE INDEX idx_fact_relation_groups_facts
ON fact_relation_groups USING GIN (fact_ids);

CREATE INDEX idx_fact_relation_groups_fish_type
ON fact_relation_groups (fish_id, snapshot_type);
"""

ENTITY_INDEXES_SQL = """
CREATE TABLE entity_aliases (
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    entity_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fish_id, alias),
    CHECK (btrim(alias) <> '')
);

CREATE INDEX idx_entity_aliases_entity
ON entity_aliases (fish_id, entity_id, alias);

CREATE TABLE fact_entities (
    entity_id UUID NOT NULL,
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, fact_id)
);

CREATE INDEX idx_fact_entities_fact
ON fact_entities (fact_id, entity_id);

WITH existing_aliases AS (
    SELECT DISTINCT
        fact.fish_id,
        lower(regexp_replace(btrim(raw.alias), '\\s+', ' ', 'g')) AS alias
    FROM facts AS fact
    CROSS JOIN LATERAL jsonb_array_elements_text(fact.entities)
        AS raw(alias)
    WHERE btrim(raw.alias) <> ''
)
INSERT INTO entity_aliases (fish_id, alias, entity_id)
SELECT fish_id, alias, gen_random_uuid()
FROM existing_aliases
ON CONFLICT DO NOTHING;

INSERT INTO fact_entities (entity_id, fact_id)
SELECT DISTINCT alias.entity_id, fact.id
FROM facts AS fact
CROSS JOIN LATERAL jsonb_array_elements_text(fact.entities)
    AS raw(alias)
JOIN entity_aliases AS alias
  ON alias.fish_id = fact.fish_id
 AND alias.alias = lower(
        regexp_replace(btrim(raw.alias), '\\s+', ' ', 'g')
    )
WHERE btrim(raw.alias) <> ''
ON CONFLICT DO NOTHING;
"""

FISH_SOUL_SQL = """
ALTER TABLE fishes
ADD COLUMN soul TEXT;
"""

FISH_SOUL_PROFILE_SQL = """
ALTER TABLE fishes
ADD COLUMN role TEXT,
ADD COLUMN setting TEXT,
ADD COLUMN name TEXT;
"""

SECOND_PRECISION_TIMESTAMPS_SQL = """
ALTER TABLE schema_migrations
ALTER COLUMN applied_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', applied_at);

ALTER TABLE messages
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at),
ALTER COLUMN finished_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', finished_at);

ALTER TABLE message_performance_steps
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at);

ALTER TABLE fishes
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at),
ALTER COLUMN deleted_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', deleted_at);

ALTER TABLE facts
ALTER COLUMN occurred_start TYPE TIMESTAMPTZ(0)
USING date_trunc('second', occurred_start),
ALTER COLUMN occurred_end TYPE TIMESTAMPTZ(0)
USING date_trunc('second', occurred_end),
ALTER COLUMN mentioned_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', mentioned_at),
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at),
ALTER COLUMN invalidated_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', invalidated_at);

ALTER TABLE background_task_progress
ALTER COLUMN updated_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', updated_at);

ALTER TABLE memory_snapshots
ALTER COLUMN updated_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', updated_at);

ALTER TABLE fact_embeddings
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at);

ALTER TABLE evaluation_questions
ALTER COLUMN question_date TYPE TIMESTAMPTZ(0)
USING date_trunc('second', question_date),
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at),
ALTER COLUMN updated_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', updated_at);

ALTER TABLE memory_snapshot_versions
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at);

ALTER TABLE fact_relation_groups
ALTER COLUMN updated_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', updated_at);

ALTER TABLE entity_aliases
ALTER COLUMN created_at TYPE TIMESTAMPTZ(0)
USING date_trunc('second', created_at);
"""

ASSISTANT_FISH_SQL = """
ALTER TABLE fishes
ADD COLUMN is_assistant BOOLEAN NOT NULL DEFAULT FALSE;

CREATE UNIQUE INDEX idx_fishes_one_assistant
ON fishes (is_assistant)
WHERE is_assistant AND deleted_at IS NULL;
"""

SESSION_SCOPING_SQL = """
CREATE TABLE fish_sessions (
    id UUID PRIMARY KEY,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    last_message_at TIMESTAMPTZ(0),
    history_window_start_sequence BIGINT NOT NULL DEFAULT 0,
    history_window_token_count BIGINT NOT NULL DEFAULT 0,
    CHECK (history_window_start_sequence >= 0),
    CHECK (history_window_token_count >= 0)
);

CREATE INDEX idx_fish_sessions_fish_created
ON fish_sessions (fish_id, created_at DESC, id DESC);

CREATE INDEX idx_fish_sessions_fish_last_message
ON fish_sessions (fish_id, last_message_at DESC, id DESC);

INSERT INTO fish_sessions (
    id,
    fish_id,
    created_at,
    last_message_at,
    history_window_start_sequence,
    history_window_token_count
)
SELECT
    gen_random_uuid(),
    fish.id,
    min(message.created_at),
    max(message.created_at),
    fish.history_window_start_sequence,
    fish.history_window_token_count
FROM fishes AS fish
JOIN messages AS message ON message.fish_id = fish.id
GROUP BY
    fish.id,
    fish.history_window_start_sequence,
    fish.history_window_token_count;

ALTER TABLE messages
ADD COLUMN session_id UUID;

UPDATE messages AS message
SET session_id = session.id
FROM fish_sessions AS session
WHERE session.fish_id = message.fish_id;

ALTER TABLE messages
ALTER COLUMN session_id SET NOT NULL,
ADD CONSTRAINT messages_session_id_fkey
FOREIGN KEY (session_id) REFERENCES fish_sessions(id);

DROP INDEX idx_messages_one_generating_answer_per_fish;

CREATE UNIQUE INDEX idx_messages_one_generating_answer_per_session
ON messages (session_id)
WHERE role = 'assistant' AND status = 'generating';

CREATE INDEX idx_messages_session_sequence
ON messages (session_id, sequence DESC);

ALTER TABLE facts
ADD COLUMN session_id UUID;

UPDATE facts AS fact
SET session_id = message.session_id
FROM messages AS message
WHERE message.id = fact.source_message_id;

ALTER TABLE facts
ALTER COLUMN session_id SET NOT NULL,
ADD CONSTRAINT facts_session_id_fkey
FOREIGN KEY (session_id) REFERENCES fish_sessions(id);

CREATE INDEX idx_facts_session_sequence
ON facts (session_id, sequence);

CREATE TABLE session_background_task_progress (
    session_id UUID NOT NULL REFERENCES fish_sessions(id) ON DELETE CASCADE,
    task_name TEXT NOT NULL,
    through_sequence BIGINT NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    PRIMARY KEY (session_id, task_name),
    CHECK (through_sequence >= 0)
);

INSERT INTO session_background_task_progress (
    session_id,
    task_name,
    through_sequence,
    last_error,
    updated_at
)
SELECT
    session.id,
    progress.task_name,
    progress.through_sequence,
    progress.last_error,
    progress.updated_at
FROM background_task_progress AS progress
JOIN fish_sessions AS session ON session.fish_id = progress.fish_id
WHERE progress.task_name IN (
    'fact_extraction',
    'dynamic_snapshot',
    'dynamic_relations'
);

DELETE FROM background_task_progress
WHERE task_name IN (
    'fact_extraction',
    'dynamic_snapshot',
    'dynamic_relations'
);

ALTER TABLE memory_snapshots
ADD COLUMN id BIGINT GENERATED ALWAYS AS IDENTITY,
ADD COLUMN session_id UUID REFERENCES fish_sessions(id);

UPDATE memory_snapshots AS snapshot
SET session_id = session.id
FROM fish_sessions AS session
WHERE snapshot.snapshot_type = 'dynamic'
  AND session.fish_id = snapshot.fish_id;

ALTER TABLE memory_snapshots
DROP CONSTRAINT memory_snapshots_pkey,
ADD PRIMARY KEY (id),
ADD CONSTRAINT memory_snapshots_scope_check CHECK (
    (snapshot_type = 'dynamic' AND session_id IS NOT NULL)
    OR (snapshot_type = 'static' AND session_id IS NULL)
);

CREATE UNIQUE INDEX idx_memory_snapshots_dynamic_session
ON memory_snapshots (session_id, snapshot_type)
WHERE session_id IS NOT NULL;

CREATE UNIQUE INDEX idx_memory_snapshots_static_fish
ON memory_snapshots (fish_id, snapshot_type)
WHERE session_id IS NULL;

ALTER TABLE memory_snapshot_versions
ADD COLUMN id BIGINT GENERATED ALWAYS AS IDENTITY,
ADD COLUMN session_id UUID REFERENCES fish_sessions(id);

UPDATE memory_snapshot_versions AS snapshot
SET session_id = session.id
FROM fish_sessions AS session
WHERE snapshot.snapshot_type = 'dynamic'
  AND session.fish_id = snapshot.fish_id;

ALTER TABLE memory_snapshot_versions
DROP CONSTRAINT memory_snapshot_versions_pkey,
ADD PRIMARY KEY (id),
ADD CONSTRAINT memory_snapshot_versions_scope_check CHECK (
    (snapshot_type = 'dynamic' AND session_id IS NOT NULL)
    OR (snapshot_type = 'static' AND session_id IS NULL)
);

CREATE UNIQUE INDEX idx_memory_snapshot_versions_dynamic_session
ON memory_snapshot_versions (session_id, snapshot_type, version)
WHERE session_id IS NOT NULL;

CREATE UNIQUE INDEX idx_memory_snapshot_versions_static_fish
ON memory_snapshot_versions (fish_id, snapshot_type, version)
WHERE session_id IS NULL;

CREATE INDEX idx_memory_snapshot_versions_session_lookup
ON memory_snapshot_versions (session_id, snapshot_type, version DESC)
WHERE session_id IS NOT NULL;

ALTER TABLE fishes
DROP COLUMN history_window_start_sequence,
DROP COLUMN history_window_token_count;
"""

TEMPORAL_RECALL_INDEX_SQL = """
CREATE INDEX idx_facts_fish_mentioned_at
ON facts (fish_id, mentioned_at);
"""

FISH_EMBEDDING_SPACES_SQL = """
ALTER TABLE fishes
ADD COLUMN embedding_provider TEXT,
ADD COLUMN embedding_model TEXT,
ADD COLUMN embedding_dimensions INTEGER,
ADD CONSTRAINT fishes_embedding_space_check CHECK (
    (
        embedding_provider IS NULL
        AND embedding_model IS NULL
        AND embedding_dimensions IS NULL
    )
    OR
    (
        length(trim(embedding_provider)) > 0
        AND length(trim(embedding_model)) > 0
        AND embedding_dimensions BETWEEN 1 AND 2000
    )
);

ALTER TABLE fact_embeddings
ADD COLUMN provider TEXT,
ADD COLUMN dimensions INTEGER;

UPDATE fact_embeddings
SET provider = 'openai-compatible',
    dimensions = vector_dims(embedding);

ALTER TABLE fact_embeddings
ALTER COLUMN provider SET NOT NULL,
ALTER COLUMN dimensions SET NOT NULL;

DROP INDEX idx_fact_embeddings_hnsw;

ALTER TABLE fact_embeddings
ALTER COLUMN embedding TYPE VECTOR
USING embedding::vector;

ALTER TABLE fact_embeddings
ADD CONSTRAINT fact_embeddings_provider_check CHECK (
    length(trim(provider)) > 0
),
ADD CONSTRAINT fact_embeddings_model_check CHECK (
    length(trim(model)) > 0
),
ADD CONSTRAINT fact_embeddings_dimensions_check CHECK (
    dimensions BETWEEN 1 AND 2000
    AND vector_dims(embedding) = dimensions
);

UPDATE fishes AS fish
SET embedding_provider = source.provider,
    embedding_model = source.model,
    embedding_dimensions = source.dimensions
FROM (
    SELECT
        fish_id,
        min(provider) AS provider,
        min(model) AS model,
        min(dimensions) AS dimensions
    FROM fact_embeddings
    GROUP BY fish_id
    HAVING count(DISTINCT provider) = 1
       AND count(DISTINCT model) = 1
       AND count(DISTINCT dimensions) = 1
) AS source
WHERE fish.id = source.fish_id;

-- HNSW is dimension-specific and expensive for populated databases. The app
-- creates only the index for its active embedding space after migrations.
"""

FISH_VOICE_CONFIG_SQL = """
ALTER TABLE fishes
ADD COLUMN voice_config JSONB,
ADD CONSTRAINT fishes_voice_config_check CHECK (
    voice_config IS NULL
    OR (
        jsonb_typeof(voice_config) = 'object'
        AND voice_config ?& ARRAY['model', 'voice', 'kind']
        AND voice_config ->> 'kind' IN ('preset', 'custom', 'local')
    )
);
"""

HOME_ACTION_RUNS_SQL = """
CREATE TABLE home_action_runs (
    id UUID PRIMARY KEY,
    sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES fish_sessions(id) ON DELETE CASCADE,
    source_message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    action JSONB NOT NULL DEFAULT '{}'::jsonb,
    entity_id TEXT,
    service_domain TEXT,
    service_name TEXT,
    status TEXT NOT NULL,
    accepted BOOLEAN NOT NULL DEFAULT false,
    confirmed BOOLEAN NOT NULL DEFAULT false,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT date_trunc('second', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT date_trunc('second', now()),
    finished_at TIMESTAMPTZ,
    CONSTRAINT home_action_runs_status_check CHECK (
        status IN ('routing', 'executing', 'succeeded', 'failed', 'ambiguous')
    ),
    CONSTRAINT home_action_runs_action_check CHECK (
        jsonb_typeof(action) = 'object'
    )
);

CREATE INDEX idx_home_action_runs_scope_sequence
ON home_action_runs (fish_id, session_id, sequence);

CREATE INDEX idx_home_action_runs_source_message
ON home_action_runs (source_message_id, sequence);
"""

HOME_ACTION_EVENT_SEQUENCE_SQL = """
CREATE SEQUENCE home_action_run_events_sequence_seq;

ALTER TABLE home_action_runs
ADD COLUMN event_sequence BIGINT;

UPDATE home_action_runs
SET event_sequence = nextval('home_action_run_events_sequence_seq');

ALTER TABLE home_action_runs
ALTER COLUMN event_sequence SET DEFAULT nextval('home_action_run_events_sequence_seq'),
ALTER COLUMN event_sequence SET NOT NULL;

CREATE UNIQUE INDEX idx_home_action_runs_event_sequence
ON home_action_runs (event_sequence);

CREATE INDEX idx_home_action_runs_scope_event_sequence
ON home_action_runs (fish_id, session_id, event_sequence);
"""

HOME_ACTION_PARTIAL_STATUS_SQL = """
ALTER TABLE home_action_runs
DROP CONSTRAINT home_action_runs_status_check;

ALTER TABLE home_action_runs
ADD CONSTRAINT home_action_runs_status_check CHECK (
    status IN ('routing', 'executing', 'succeeded', 'partial', 'failed', 'ambiguous')
);
"""

RELATION_GROUP_EMBEDDINGS_SQL = """
CREATE TABLE fact_relation_group_embeddings (
    relation_group_id BIGINT PRIMARY KEY
        REFERENCES fact_relation_groups(id) ON DELETE CASCADE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    session_id UUID REFERENCES fish_sessions(id) ON DELETE CASCADE,
    snapshot_type TEXT NOT NULL,
    snapshot_version BIGINT NOT NULL,
    snapshot_text TEXT NOT NULL,
    text_hash CHAR(64) NOT NULL,
    last_supported_at TIMESTAMPTZ(0) NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding VECTOR NOT NULL,
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    CONSTRAINT fact_relation_group_embeddings_scope_check CHECK (
        (snapshot_type = 'static' AND session_id IS NULL)
        OR (snapshot_type = 'dynamic' AND session_id IS NOT NULL)
    ),
    CONSTRAINT fact_relation_group_embeddings_version_check CHECK (
        snapshot_version >= 1
    ),
    CONSTRAINT fact_relation_group_embeddings_text_check CHECK (
        btrim(snapshot_text) <> ''
        AND text_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT fact_relation_group_embeddings_space_check CHECK (
        btrim(provider) <> ''
        AND btrim(model) <> ''
        AND dimensions BETWEEN 1 AND 2000
        AND vector_dims(embedding) = dimensions
    )
);

CREATE INDEX idx_fact_relation_group_embeddings_static_scope
ON fact_relation_group_embeddings (
    fish_id, provider, model, dimensions
)
WHERE snapshot_type = 'static';

CREATE INDEX idx_fact_relation_group_embeddings_dynamic_scope
ON fact_relation_group_embeddings (
    fish_id, session_id, provider, model, dimensions
)
WHERE snapshot_type = 'dynamic';
"""

RECALL_MASK_EMBEDDINGS_SQL = """
ALTER TABLE fact_relation_group_embeddings
ADD COLUMN member_count INTEGER NOT NULL DEFAULT 1,
ADD CONSTRAINT fact_relation_group_embeddings_member_count_check CHECK (
    member_count >= 1
);

UPDATE fact_relation_group_embeddings AS embedding
SET member_count = cardinality(relation_group.fact_ids)
FROM fact_relation_groups AS relation_group
WHERE relation_group.id = embedding.relation_group_id;

CREATE INDEX idx_fact_relation_group_embeddings_head_scope
ON fact_relation_group_embeddings (fish_id, dimensions, member_count DESC);

ALTER TABLE fact_embeddings
ADD COLUMN embedding_text TEXT,
ADD COLUMN text_hash CHAR(64),
ADD CONSTRAINT fact_embeddings_source_text_check CHECK (
    (
        embedding_text IS NULL
        AND text_hash IS NULL
    )
    OR (
        btrim(embedding_text) <> ''
        AND text_hash ~ '^[0-9a-f]{64}$'
    )
);

CREATE TABLE fact_mask_embeddings (
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    entity_ids UUID[] NOT NULL,
    signature_hash CHAR(64) NOT NULL,
    source_text_hash CHAR(64) NOT NULL,
    masked_text TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding VECTOR NOT NULL,
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    PRIMARY KEY (fact_id, signature_hash, dimensions),
    CONSTRAINT fact_mask_embeddings_signature_check CHECK (
        cardinality(entity_ids) > 0
        AND signature_hash ~ '^[0-9a-f]{64}$'
        AND source_text_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT fact_mask_embeddings_text_check CHECK (
        btrim(masked_text) <> ''
    ),
    CONSTRAINT fact_mask_embeddings_space_check CHECK (
        btrim(provider) <> ''
        AND btrim(model) <> ''
        AND dimensions BETWEEN 1 AND 2000
        AND vector_dims(embedding) = dimensions
    )
);

CREATE INDEX idx_fact_mask_embeddings_fish_dimensions
ON fact_mask_embeddings (fish_id, dimensions);

CREATE TABLE fact_relation_group_mask_embeddings (
    relation_group_id BIGINT NOT NULL
        REFERENCES fact_relation_group_embeddings(relation_group_id)
        ON DELETE CASCADE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    entity_ids UUID[] NOT NULL,
    signature_hash CHAR(64) NOT NULL,
    source_text_hash CHAR(64) NOT NULL,
    masked_text TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding VECTOR NOT NULL,
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    PRIMARY KEY (relation_group_id, signature_hash, dimensions),
    CONSTRAINT fact_relation_group_mask_embeddings_signature_check CHECK (
        cardinality(entity_ids) > 0
        AND signature_hash ~ '^[0-9a-f]{64}$'
        AND source_text_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT fact_relation_group_mask_embeddings_text_check CHECK (
        btrim(masked_text) <> ''
    ),
    CONSTRAINT fact_relation_group_mask_embeddings_space_check CHECK (
        btrim(provider) <> ''
        AND btrim(model) <> ''
        AND dimensions BETWEEN 1 AND 2000
        AND vector_dims(embedding) = dimensions
    )
);

CREATE INDEX idx_fact_relation_group_mask_embeddings_fish_dimensions
ON fact_relation_group_mask_embeddings (fish_id, dimensions);
"""

REPAIR_RELATION_MEMBER_COUNTS_SQL = """
UPDATE fact_relation_group_embeddings AS embedding
SET member_count = cardinality(relation_group.fact_ids),
    updated_at = date_trunc('second', now())
FROM fact_relation_groups AS relation_group
WHERE relation_group.id = embedding.relation_group_id
  AND embedding.member_count <> cardinality(relation_group.fact_ids);
"""

NORMALIZED_SNAPSHOT_ITEMS_SQL = """
ALTER TABLE fishes
ADD COLUMN snapshot_storage_version SMALLINT NOT NULL DEFAULT 1;

ALTER TABLE fishes
ALTER COLUMN snapshot_storage_version SET DEFAULT 2;

ALTER TABLE fishes
ADD CONSTRAINT fishes_snapshot_storage_version_check
CHECK (snapshot_storage_version IN (1, 2));

ALTER TABLE memory_snapshots
ALTER COLUMN content DROP NOT NULL;

CREATE TABLE memory_snapshot_revisions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    snapshot_id BIGINT NOT NULL
        REFERENCES memory_snapshots(id) ON DELETE CASCADE,
    version BIGINT NOT NULL,
    through_fact_sequence BIGINT NOT NULL,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    UNIQUE (snapshot_id, version),
    CHECK (version >= 1),
    CHECK (through_fact_sequence >= 0)
);

CREATE TABLE memory_snapshot_items (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    snapshot_id BIGINT NOT NULL
        REFERENCES memory_snapshots(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    support_fact_ids UUID[] NOT NULL,
    lineage_fact_ids UUID[] NOT NULL,
    support_count INTEGER GENERATED ALWAYS AS (
        cardinality(support_fact_ids)
    ) STORED,
    last_supported_at TIMESTAMPTZ(0) NOT NULL,
    item_version BIGINT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    CHECK (btrim(summary) <> ''),
    CHECK (cardinality(support_fact_ids) > 0),
    CHECK (cardinality(lineage_fact_ids) > 0),
    CHECK (support_fact_ids <@ lineage_fact_ids),
    CHECK (item_version >= 1)
);

CREATE INDEX idx_memory_snapshot_items_snapshot_time
ON memory_snapshot_items (snapshot_id, last_supported_at, id);

CREATE INDEX idx_memory_snapshot_items_support
ON memory_snapshot_items USING GIN (support_fact_ids);

CREATE TABLE memory_snapshot_item_versions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    snapshot_item_id BIGINT NOT NULL
        REFERENCES memory_snapshot_items(id) ON DELETE CASCADE,
    snapshot_revision_id BIGINT NOT NULL
        REFERENCES memory_snapshot_revisions(id) ON DELETE CASCADE,
    snapshot_version BIGINT NOT NULL,
    item_version BIGINT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('create', 'update')),
    basis TEXT,
    summary TEXT NOT NULL,
    support_fact_ids UUID[] NOT NULL,
    lineage_fact_ids UUID[] NOT NULL,
    new_support_fact_ids UUID[] NOT NULL,
    removed_support_fact_ids UUID[] NOT NULL,
    last_supported_at TIMESTAMPTZ(0) NOT NULL,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    UNIQUE (snapshot_item_id, item_version),
    UNIQUE (snapshot_item_id, snapshot_version),
    CHECK (basis IS NULL OR char_length(basis) <= 100),
    CHECK (btrim(summary) <> ''),
    CHECK (cardinality(support_fact_ids) > 0),
    CHECK (support_fact_ids <@ lineage_fact_ids)
);
"""

SNAPSHOT_ITEM_RELATION_EMBEDDINGS_SQL = """
CREATE TABLE memory_snapshot_item_embeddings (
    snapshot_item_id BIGINT PRIMARY KEY
        REFERENCES memory_snapshot_items(id) ON DELETE CASCADE,
    item_version BIGINT NOT NULL,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    session_id UUID REFERENCES fish_sessions(id) ON DELETE CASCADE,
    snapshot_type TEXT NOT NULL,
    snapshot_text TEXT NOT NULL,
    text_hash CHAR(64) NOT NULL,
    last_supported_at TIMESTAMPTZ(0) NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding VECTOR NOT NULL,
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    CONSTRAINT memory_snapshot_item_embeddings_scope_check CHECK (
        (snapshot_type = 'static' AND session_id IS NULL)
        OR (snapshot_type = 'dynamic' AND session_id IS NOT NULL)
    ),
    CONSTRAINT memory_snapshot_item_embeddings_version_check CHECK (
        item_version >= 1
    ),
    CONSTRAINT memory_snapshot_item_embeddings_text_check CHECK (
        btrim(snapshot_text) <> ''
        AND text_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT memory_snapshot_item_embeddings_space_check CHECK (
        btrim(provider) <> ''
        AND btrim(model) <> ''
        AND dimensions BETWEEN 1 AND 2000
        AND vector_dims(embedding) = dimensions
    )
);

CREATE INDEX idx_memory_snapshot_item_embeddings_static_scope
ON memory_snapshot_item_embeddings (fish_id, dimensions)
WHERE snapshot_type = 'static';

CREATE INDEX idx_memory_snapshot_item_embeddings_dynamic_scope
ON memory_snapshot_item_embeddings (fish_id, session_id, dimensions)
WHERE snapshot_type = 'dynamic';

CREATE TABLE memory_snapshot_item_mask_embeddings (
    snapshot_item_id BIGINT NOT NULL
        REFERENCES memory_snapshot_item_embeddings(snapshot_item_id)
        ON DELETE CASCADE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    entity_ids UUID[] NOT NULL,
    signature_hash CHAR(64) NOT NULL,
    source_text_hash CHAR(64) NOT NULL,
    masked_text TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    embedding VECTOR NOT NULL,
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    PRIMARY KEY (snapshot_item_id, signature_hash, dimensions),
    CONSTRAINT memory_snapshot_item_mask_embeddings_signature_check CHECK (
        cardinality(entity_ids) > 0
        AND signature_hash ~ '^[0-9a-f]{64}$'
        AND source_text_hash ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT memory_snapshot_item_mask_embeddings_text_check CHECK (
        btrim(masked_text) <> ''
    ),
    CONSTRAINT memory_snapshot_item_mask_embeddings_space_check CHECK (
        btrim(provider) <> ''
        AND btrim(model) <> ''
        AND dimensions BETWEEN 1 AND 2000
        AND vector_dims(embedding) = dimensions
    )
);

CREATE INDEX idx_memory_snapshot_item_mask_embeddings_fish_dimensions
ON memory_snapshot_item_mask_embeddings (fish_id, dimensions);
"""

AGENT_TASK_CHANNEL_SQL = """
CREATE TABLE agent_endpoints (
    id UUID PRIMARY KEY,
    connector_type TEXT NOT NULL,
    base_url TEXT NOT NULL,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    CONSTRAINT agent_endpoints_connector_type_check CHECK (
        connector_type ~ '^[a-z][a-z0-9._-]{0,63}$'
    ),
    UNIQUE (connector_type, base_url)
);

CREATE TABLE agent_targets (
    id UUID PRIMARY KEY,
    endpoint_id UUID NOT NULL REFERENCES agent_endpoints(id),
    external_key TEXT NOT NULL,
    route_prefix TEXT NOT NULL DEFAULT '',
    credential_ref TEXT,
    connector_config JSONB,
    display_name TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    CONSTRAINT agent_targets_connector_config_check CHECK (
        connector_config IS NULL
        OR jsonb_typeof(connector_config) = 'object'
    ),
    UNIQUE (endpoint_id, external_key, route_prefix)
);

COMMENT ON COLUMN agent_targets.connector_config IS
    'Bounded non-secret connector configuration; credentials use credential_ref.';

ALTER TABLE fishes
ADD COLUMN agent_target_id UUID REFERENCES agent_targets(id) ON DELETE SET NULL;

CREATE INDEX idx_fishes_agent_target_id
ON fishes (agent_target_id)
WHERE agent_target_id IS NOT NULL;

CREATE TABLE agent_tasks (
    id UUID PRIMARY KEY,
    fish_id UUID NOT NULL REFERENCES fishes(id),
    target_id UUID NOT NULL REFERENCES agent_targets(id),
    external_session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    UNIQUE (target_id, external_session_id),
    UNIQUE (id, target_id)
);

CREATE INDEX idx_agent_tasks_fish_updated
ON agent_tasks (fish_id, updated_at DESC, id DESC);

CREATE TABLE agent_runs (
    id UUID PRIMARY KEY,
    task_id UUID NOT NULL,
    target_id UUID NOT NULL,
    previous_run_id UUID,
    source_message_id UUID NOT NULL REFERENCES messages(id),
    confirmation_message_id UUID REFERENCES messages(id),
    input TEXT NOT NULL,
    external_run_id TEXT,
    history_source TEXT NOT NULL DEFAULT 'none',
    status TEXT NOT NULL,
    output TEXT,
    result_message_id UUID REFERENCES messages(id),
    error_code TEXT,
    error_message TEXT,
    provider_data JSONB,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    started_at TIMESTAMPTZ(0),
    updated_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    finished_at TIMESTAMPTZ(0),
    CONSTRAINT agent_runs_task_target_fkey FOREIGN KEY (task_id, target_id)
        REFERENCES agent_tasks(id, target_id),
    CONSTRAINT agent_runs_previous_task_fkey FOREIGN KEY (previous_run_id, task_id)
        REFERENCES agent_runs(id, task_id),
    CONSTRAINT agent_runs_history_source_check CHECK (
        history_source IN ('none', 'hermes_session', 'local_projection')
    ),
    CONSTRAINT agent_runs_status_check CHECK (
        status IN (
            'queued', 'dispatching', 'running', 'waiting_user', 'completed',
            'failed', 'cancelled', 'unknown', 'abandoned'
        )
    ),
    CONSTRAINT agent_runs_provider_data_check CHECK (
        provider_data IS NULL OR jsonb_typeof(provider_data) = 'object'
    ),
    UNIQUE (id, task_id),
    UNIQUE (target_id, external_run_id),
    UNIQUE (confirmation_message_id),
    UNIQUE (result_message_id)
);

CREATE INDEX idx_agent_runs_task_created
ON agent_runs (task_id, created_at, id);

CREATE UNIQUE INDEX idx_agent_runs_one_child_per_previous
ON agent_runs (previous_run_id)
WHERE previous_run_id IS NOT NULL;

CREATE UNIQUE INDEX idx_agent_runs_one_active_per_task
ON agent_runs (task_id)
WHERE status IN ('queued', 'dispatching', 'running', 'waiting_user', 'unknown');

CREATE TABLE agent_interactions (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id),
    kind TEXT NOT NULL,
    request_message_id UUID UNIQUE REFERENCES messages(id),
    response_message_id UUID UNIQUE REFERENCES messages(id),
    next_run_id UUID UNIQUE REFERENCES agent_runs(id),
    provider_data JSONB,
    created_at TIMESTAMPTZ(0) NOT NULL DEFAULT date_trunc('second', now()),
    resolved_at TIMESTAMPTZ(0),
    closed_at TIMESTAMPTZ(0),
    CONSTRAINT agent_interactions_kind_check CHECK (
        kind IN ('approval', 'input')
    ),
    CONSTRAINT agent_interactions_provider_data_check CHECK (
        provider_data IS NULL OR jsonb_typeof(provider_data) = 'object'
    ),
    CONSTRAINT agent_interactions_resolution_check CHECK (
        NOT (resolved_at IS NOT NULL AND closed_at IS NOT NULL)
        AND (response_message_id IS NULL OR request_message_id IS NOT NULL)
        AND (resolved_at IS NULL OR response_message_id IS NOT NULL)
        AND (kind <> 'approval' OR next_run_id IS NULL)
        AND (
            next_run_id IS NULL
            OR (
                kind = 'input'
                AND resolved_at IS NOT NULL
                AND closed_at IS NULL
            )
        )
        AND (
            kind <> 'input'
            OR resolved_at IS NULL
            OR next_run_id IS NOT NULL
        )
    )
);

CREATE UNIQUE INDEX idx_agent_interactions_one_open_per_run
ON agent_interactions (run_id)
WHERE resolved_at IS NULL AND closed_at IS NULL;

ALTER TABLE messages
DROP CONSTRAINT messages_fact_extraction_status_check;

ALTER TABLE messages
ADD CONSTRAINT messages_fact_extraction_status_check CHECK (
    fact_extraction_status IN (
        'pending', 'waiting_consumption', 'completed', 'failed'
    )
);
"""

AGENT_TASK_LOCAL_SESSION_SCOPE_SQL = """
-- Migration 32 predates the local Fish-session identity for Agent Tasks.  Do
-- not choose an arbitrary source Message when repairing existing data: that
-- would silently merge two conversations.  A Task is migratable only when all
-- of its already-bound Messages agree on both Fish and Session.
ALTER TABLE fish_sessions
ADD CONSTRAINT fish_sessions_fish_id_id_key UNIQUE (fish_id, id);

ALTER TABLE messages
ADD CONSTRAINT messages_id_fish_id_session_id_key UNIQUE (id, fish_id, session_id);

ALTER TABLE agent_tasks
ADD COLUMN session_id UUID;

WITH task_message_ids AS (
    SELECT run.task_id, run.source_message_id AS message_id
    FROM agent_runs AS run
    UNION ALL
    SELECT run.task_id, run.confirmation_message_id
    FROM agent_runs AS run
    WHERE run.confirmation_message_id IS NOT NULL
    UNION ALL
    SELECT run.task_id, run.result_message_id
    FROM agent_runs AS run
    WHERE run.result_message_id IS NOT NULL
    UNION ALL
    SELECT run.task_id, interaction.request_message_id
    FROM agent_interactions AS interaction
    JOIN agent_runs AS run ON run.id = interaction.run_id
    WHERE interaction.request_message_id IS NOT NULL
    UNION ALL
    SELECT run.task_id, interaction.response_message_id
    FROM agent_interactions AS interaction
    JOIN agent_runs AS run ON run.id = interaction.run_id
    WHERE interaction.response_message_id IS NOT NULL
),
valid_task_sessions AS (
    SELECT task.id AS task_id,
           (array_agg(DISTINCT message.session_id))[1] AS session_id
    FROM agent_tasks AS task
    JOIN task_message_ids AS link ON link.task_id = task.id
    JOIN messages AS message ON message.id = link.message_id
    GROUP BY task.id
    HAVING count(DISTINCT message.session_id) = 1
       AND bool_and(message.fish_id = task.fish_id)
)
UPDATE agent_tasks AS task
SET session_id = valid.session_id
FROM valid_task_sessions AS valid
WHERE task.id = valid.task_id;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM agent_tasks
        WHERE session_id IS NULL
    ) THEN
        RAISE EXCEPTION
            'cannot migrate Agent Tasks with missing, cross-Fish, '
            'or cross-Session Message bindings';
    END IF;
END $$;

ALTER TABLE agent_tasks
ALTER COLUMN session_id SET NOT NULL,
ADD CONSTRAINT agent_tasks_fish_session_fkey
    FOREIGN KEY (fish_id, session_id) REFERENCES fish_sessions (fish_id, id),
ADD CONSTRAINT agent_tasks_id_fish_session_key
    UNIQUE (id, fish_id, session_id);

ALTER TABLE agent_runs
ADD COLUMN fish_id UUID,
ADD COLUMN session_id UUID;

UPDATE agent_runs AS run
SET fish_id = task.fish_id,
    session_id = task.session_id
FROM agent_tasks AS task
WHERE task.id = run.task_id;

ALTER TABLE agent_runs
ALTER COLUMN fish_id SET NOT NULL,
ALTER COLUMN session_id SET NOT NULL,
ADD CONSTRAINT agent_runs_task_fish_session_fkey
    FOREIGN KEY (task_id, fish_id, session_id)
    REFERENCES agent_tasks (id, fish_id, session_id),
ADD CONSTRAINT agent_runs_source_message_fish_session_fkey
    FOREIGN KEY (source_message_id, fish_id, session_id)
    REFERENCES messages (id, fish_id, session_id),
ADD CONSTRAINT agent_runs_confirmation_message_fish_session_fkey
    FOREIGN KEY (confirmation_message_id, fish_id, session_id)
    REFERENCES messages (id, fish_id, session_id),
ADD CONSTRAINT agent_runs_result_message_fish_session_fkey
    FOREIGN KEY (result_message_id, fish_id, session_id)
    REFERENCES messages (id, fish_id, session_id),
ADD CONSTRAINT agent_runs_id_fish_session_key
    UNIQUE (id, fish_id, session_id);

ALTER TABLE agent_interactions
ADD COLUMN fish_id UUID,
ADD COLUMN session_id UUID;

UPDATE agent_interactions AS interaction
SET fish_id = run.fish_id,
    session_id = run.session_id
FROM agent_runs AS run
WHERE run.id = interaction.run_id;

ALTER TABLE agent_interactions
ALTER COLUMN fish_id SET NOT NULL,
ALTER COLUMN session_id SET NOT NULL,
ADD CONSTRAINT agent_interactions_run_fish_session_fkey
    FOREIGN KEY (run_id, fish_id, session_id)
    REFERENCES agent_runs (id, fish_id, session_id),
ADD CONSTRAINT agent_interactions_request_message_fish_session_fkey
    FOREIGN KEY (request_message_id, fish_id, session_id)
    REFERENCES messages (id, fish_id, session_id),
ADD CONSTRAINT agent_interactions_response_message_fish_session_fkey
    FOREIGN KEY (response_message_id, fish_id, session_id)
    REFERENCES messages (id, fish_id, session_id),
ADD CONSTRAINT agent_interactions_next_run_fish_session_fkey
    FOREIGN KEY (next_run_id, fish_id, session_id)
    REFERENCES agent_runs (id, fish_id, session_id);
"""

AGENT_RUN_FISH_SESSION_HISTORY_SOURCE_SQL = """
ALTER TABLE agent_runs
DROP CONSTRAINT agent_runs_history_source_check;

ALTER TABLE agent_runs
ADD CONSTRAINT agent_runs_history_source_check CHECK (
    history_source IN (
        'none', 'fish_session', 'hermes_session', 'local_projection'
    )
);
"""

AGENT_RUN_COMBINED_HISTORY_SOURCE_SQL = """
ALTER TABLE agent_runs
DROP CONSTRAINT agent_runs_history_source_check;

ALTER TABLE agent_runs
ADD CONSTRAINT agent_runs_history_source_check CHECK (
    history_source IN (
        'none', 'fish_session', 'hermes_session', 'local_projection',
        'fish_session_hermes_session', 'fish_session_local_projection'
    )
);
"""

OBSERVATIONS_SQL = """
CREATE TABLE observation_app_policies (
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    app_bundle_id TEXT NOT NULL,
    app_name TEXT,
    recording_enabled BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fish_id, app_bundle_id),
    CHECK (length(app_bundle_id) BETWEEN 1 AND 255),
    CHECK (app_name IS NULL OR length(app_name) BETWEEN 1 AND 200)
);

CREATE TABLE observations (
    id UUID PRIMARY KEY,
    sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('state', 'event')),
    source TEXT NOT NULL,
    observation_type TEXT NOT NULL,
    scope_key TEXT,
    semantic_key TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    app_bundle_id TEXT,
    app_name TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (length(source) BETWEEN 1 AND 64),
    CHECK (length(observation_type) BETWEEN 1 AND 64),
    CHECK (scope_key IS NULL OR length(scope_key) BETWEEN 1 AND 200),
    CHECK (length(semantic_key) BETWEEN 1 AND 512),
    CHECK (length(summary) BETWEEN 1 AND 2000),
    CHECK (jsonb_typeof(payload) = 'object'),
    CHECK (app_bundle_id IS NULL OR length(app_bundle_id) BETWEEN 1 AND 255),
    CHECK (app_name IS NULL OR length(app_name) BETWEEN 1 AND 200),
    CHECK (ended_at >= started_at),
    CHECK (
        (kind = 'state' AND scope_key IS NOT NULL)
        OR (kind = 'event' AND scope_key IS NULL AND ended_at = started_at)
    )
);

CREATE INDEX idx_observations_fish_time
ON observations (fish_id, ended_at DESC, started_at DESC);

CREATE INDEX idx_observations_fish_app_time
ON observations (fish_id, app_bundle_id, ended_at DESC)
WHERE app_bundle_id IS NOT NULL;

CREATE INDEX idx_observations_fish_type_time
ON observations (fish_id, observation_type, ended_at DESC);

CREATE INDEX idx_observations_state_stream
ON observations (
    fish_id,
    source,
    observation_type,
    scope_key,
    started_at DESC,
    sequence DESC
)
WHERE kind = 'state';
"""

BACKGROUND_ASSISTANT_DELIVERIES_SQL = """
CREATE TABLE background_assistant_deliveries (
    message_id UUID PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (source IN ('agent', 'proactive_observation')),
    emotion TEXT NOT NULL CHECK (
        emotion IN ('neutral', 'angry', 'happy', 'shy', 'sad')
    ),
    interrupt_policy TEXT NOT NULL CHECK (interrupt_policy IN ('resume', 'drop')),
    expires_at TIMESTAMPTZ,
    supersession_key TEXT,
    source_observation_id UUID REFERENCES observations(id) ON DELETE SET NULL,
    cancelled_at TIMESTAMPTZ,
    CHECK (
        (source = 'agent'
         AND interrupt_policy = 'resume'
         AND expires_at IS NULL
         AND supersession_key IS NULL
         AND source_observation_id IS NULL)
        OR
        (source = 'proactive_observation'
         AND interrupt_policy = 'drop'
         AND expires_at IS NOT NULL
         AND supersession_key IS NOT NULL
         AND length(supersession_key) BETWEEN 1 AND 200)
    )
);

CREATE INDEX idx_background_assistant_deliveries_active_expiry
ON background_assistant_deliveries (expires_at, message_id)
WHERE source = 'proactive_observation' AND cancelled_at IS NULL;

CREATE INDEX idx_background_assistant_deliveries_observation
ON background_assistant_deliveries (source_observation_id)
WHERE source_observation_id IS NOT NULL;

CREATE UNIQUE INDEX idx_background_assistant_deliveries_one_per_observation
ON background_assistant_deliveries (source_observation_id)
WHERE source = 'proactive_observation' AND source_observation_id IS NOT NULL;

INSERT INTO background_assistant_deliveries (
    message_id,
    source,
    emotion,
    interrupt_policy
)
SELECT message.id, 'agent', 'neutral', 'resume'
FROM messages AS message
LEFT JOIN agent_runs AS result_run ON result_run.result_message_id = message.id
LEFT JOIN agent_interactions AS interaction
  ON interaction.request_message_id = message.id
WHERE message.role = 'assistant'
  AND ((result_run.id IS NOT NULL)::int + (interaction.id IS NOT NULL)::int) = 1;
"""

COMPUTER_HISTORY_SUMMARY_CHECKPOINTS_SQL = """
CREATE TABLE computer_history_summary_checkpoints (
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    segment_id UUID NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    revision INTEGER NOT NULL,
    through_event_count INTEGER NOT NULL,
    through_byte_offset BIGINT NOT NULL,
    through_suppressed_event_count INTEGER NOT NULL,
    delta_chain_digest TEXT NOT NULL,
    full_summary JSONB,
    applications JSONB NOT NULL DEFAULT '[]'::jsonb,
    inherited_observation_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    delta_capsules JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_request_id UUID,
    last_delta_digest TEXT,
    last_successful_summary_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active',
    finalized_at TIMESTAMPTZ,
    observation_id UUID REFERENCES observations(id) ON DELETE CASCADE,
    final_event_digest TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fish_id, segment_id),
    CHECK (ended_at = started_at + INTERVAL '10 minutes'),
    CHECK (revision >= 1),
    CHECK (through_event_count >= 0),
    CHECK (through_byte_offset >= 0),
    CHECK (through_suppressed_event_count >= 0),
    CHECK (length(delta_chain_digest) = 64),
    CHECK (full_summary IS NULL OR jsonb_typeof(full_summary) = 'object'),
    CHECK (jsonb_typeof(applications) = 'array'),
    CHECK (jsonb_typeof(inherited_observation_ids) = 'array'),
    CHECK (jsonb_typeof(delta_capsules) = 'array'),
    CHECK (last_delta_digest IS NULL OR length(last_delta_digest) = 64),
    CHECK (status IN ('active', 'finalized')),
    CHECK (final_event_digest IS NULL OR length(final_event_digest) = 64),
    CHECK (
        (status = 'active'
         AND finalized_at IS NULL
         AND observation_id IS NULL
         AND final_event_digest IS NULL)
        OR
        (status = 'finalized'
         AND finalized_at IS NOT NULL
         AND observation_id IS NOT NULL
         AND final_event_digest IS NOT NULL)
    )
);

CREATE INDEX idx_computer_history_checkpoints_fish_recent
ON computer_history_summary_checkpoints (
    fish_id,
    started_at DESC,
    updated_at DESC
);

CREATE UNIQUE INDEX idx_computer_history_checkpoints_observation
ON computer_history_summary_checkpoints (observation_id)
WHERE observation_id IS NOT NULL;
"""

ENVIRONMENT_HISTORY_SUMMARY_LENGTH_SQL = """
-- Migration 36 gave this single-column check its PostgreSQL-generated name.
-- Keep the full semantic text in both summary and document payload. Other
-- observations retain their existing limit; new hierarchy durations need no
-- separate database migration.
ALTER TABLE observations
DROP CONSTRAINT observations_summary_check,
ADD CONSTRAINT observations_summary_length_check CHECK (
    length(summary) BETWEEN 1 AND CASE
        WHEN source = 'environment_history'
             AND observation_type ~ '^semantic_snapshot_[1-9][0-9]*s$'
        THEN 31000
        ELSE 2000
    END
);
"""


AUTONOMOUS_AGENT_RUN_SQL = """
ALTER TABLE agent_runs ALTER COLUMN source_message_id DROP NOT NULL;
ALTER TABLE agent_runs ADD COLUMN origin TEXT NOT NULL DEFAULT 'user';
ALTER TABLE agent_runs ADD COLUMN origin_context JSONB;
ALTER TABLE agent_runs ADD CONSTRAINT agent_runs_origin_check CHECK (
    (origin = 'user' AND source_message_id IS NOT NULL)
    OR (origin = 'proactive' AND source_message_id IS NULL
        AND confirmation_message_id IS NULL AND previous_run_id IS NULL
        AND origin_context IS NOT NULL AND jsonb_typeof(origin_context) = 'object')
);
CREATE UNIQUE INDEX idx_agent_runs_one_active_autonomous
ON agent_runs ((1)) WHERE origin = 'proactive'
AND status IN ('queued', 'dispatching', 'running', 'waiting_user', 'unknown');
CREATE INDEX idx_agent_runs_autonomous_created ON agent_runs (created_at DESC)
WHERE origin = 'proactive';
"""


PROACTIVE_STREAM_LIFECYCLE_SQL = """
ALTER TABLE background_assistant_deliveries
DROP CONSTRAINT background_assistant_deliveries_check;
ALTER TABLE background_assistant_deliveries
ADD CONSTRAINT background_assistant_deliveries_check CHECK (
    (source = 'agent'
     AND interrupt_policy = 'resume'
     AND expires_at IS NULL
     AND supersession_key IS NULL
     AND source_observation_id IS NULL)
    OR
    (source = 'proactive_observation'
     AND interrupt_policy = 'drop'
     AND supersession_key IS NOT NULL
     AND length(supersession_key) BETWEEN 1 AND 200)
);
"""


PLUGIN_EVENTS_SQL = """
ALTER TABLE background_assistant_deliveries
DROP CONSTRAINT background_assistant_deliveries_source_check;
ALTER TABLE background_assistant_deliveries
ADD CONSTRAINT background_assistant_deliveries_source_check
CHECK (source IN ('agent','proactive_observation','plugin'));
ALTER TABLE background_assistant_deliveries
DROP CONSTRAINT background_assistant_deliveries_check;
ALTER TABLE background_assistant_deliveries
ADD CONSTRAINT background_assistant_deliveries_check CHECK (
    (source='agent' AND interrupt_policy='resume' AND expires_at IS NULL
     AND supersession_key IS NULL AND source_observation_id IS NULL)
    OR
    (source IN ('proactive_observation','plugin') AND interrupt_policy='drop'
     AND supersession_key IS NOT NULL AND length(supersession_key) BETWEEN 1 AND 200)
);
CREATE UNIQUE INDEX idx_background_plugin_event
ON background_assistant_deliveries (source_observation_id)
WHERE source='plugin' AND source_observation_id IS NOT NULL;
CREATE INDEX idx_observations_plugin_session
ON observations (fish_id, (payload->>'session_id'), sequence DESC)
WHERE source='plugin' AND observation_type='plugin.event';
"""


PLUGIN_TASKS_SQL = """
ALTER TABLE agent_runs DROP CONSTRAINT agent_runs_origin_check;
ALTER TABLE agent_runs ADD CONSTRAINT agent_runs_origin_check CHECK (
    (origin = 'user' AND source_message_id IS NOT NULL)
    OR (origin IN ('proactive','plugin') AND source_message_id IS NULL
        AND confirmation_message_id IS NULL AND previous_run_id IS NULL
        AND origin_context IS NOT NULL AND jsonb_typeof(origin_context) = 'object'
        AND (origin <> 'plugin' OR (
            origin_context ?& ARRAY['pluginId','instanceId','generation','requestId','requestDigest','contextKey']
            AND length(origin_context->>'instanceId') > 0
            AND length(origin_context->>'generation') > 0
            AND length(origin_context->>'requestId') > 0
        )))
);
CREATE UNIQUE INDEX idx_agent_runs_plugin_request ON agent_runs
(fish_id, (origin_context->>'instanceId'), (origin_context->>'generation'), (origin_context->>'requestId'))
WHERE origin='plugin';
CREATE UNIQUE INDEX idx_agent_runs_plugin_active ON agent_runs
(fish_id, (origin_context->>'instanceId')) WHERE origin='plugin'
AND status IN ('queued','dispatching','running','waiting_user','unknown');

CREATE OR REPLACE FUNCTION notify_fish_plugin_task() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.origin = 'plugin' AND (TG_OP = 'INSERT'
       OR NEW.status IS DISTINCT FROM OLD.status
       OR NEW.output IS DISTINCT FROM OLD.output
       OR NEW.origin_context IS DISTINCT FROM OLD.origin_context) THEN
        PERFORM pg_notify('fish_plugin_tasks', NEW.id::text);
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER fish_plugin_task_changed AFTER INSERT OR UPDATE ON agent_runs
FOR EACH ROW EXECUTE FUNCTION notify_fish_plugin_task();
"""


LLM_CALL_LOGS_SQL = """
CREATE TABLE llm_call_logs (
    id UUID PRIMARY KEY,
    sequence BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    call_type TEXT NOT NULL,
    model TEXT NOT NULL,
    request_url TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running','completed','failed','cancelled')),
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    duration_ms BIGINT,
    http_status INTEGER,
    error TEXT,
    source TEXT NOT NULL,
    request JSONB NOT NULL,
    response JSONB
);
CREATE INDEX idx_llm_call_logs_type_sequence ON llm_call_logs(call_type,sequence DESC);
CREATE INDEX idx_llm_call_logs_status_sequence ON llm_call_logs(status,sequence DESC);
"""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    sql: str


MIND_SQL = """
CREATE TABLE mind_commitments (
    id UUID PRIMARY KEY,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    session_id UUID REFERENCES fish_sessions(id) ON DELETE CASCADE,
    goal TEXT NOT NULL,
    steps JSONB NOT NULL DEFAULT '[]'::jsonb,
    end_condition TEXT NOT NULL DEFAULT '',
    until TIMESTAMPTZ,
    needs_feedback BOOLEAN NOT NULL DEFAULT TRUE,
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'suspended', 'done')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_feedback_at TIMESTAMPTZ
);
CREATE INDEX idx_mind_commitments_active ON mind_commitments (fish_id, status);
CREATE TABLE mind_trace (
    id BIGSERIAL PRIMARY KEY,
    fish_id UUID NOT NULL,
    session_id UUID,
    at TIMESTAMPTZ NOT NULL,
    drives JSONB NOT NULL,
    body JSONB NOT NULL,
    focus TEXT
);
CREATE INDEX idx_mind_trace_fish_at ON mind_trace (fish_id, at);
"""

MIND_COMMITMENT_STEP_SQL = """
ALTER TABLE mind_commitments ADD COLUMN step INTEGER NOT NULL DEFAULT 0;
"""

# A commitment carries a living plan instead of fixed steps; the old columns
# stay, unread.
MIND_COMMITMENT_PLAN_SQL = """
ALTER TABLE mind_commitments ADD COLUMN plan TEXT NOT NULL DEFAULT '';
"""

MIND_COMMITMENT_MATERIAL_SQL = """
ALTER TABLE mind_commitments
    ADD COLUMN needs_material BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN looked_up_at TIMESTAMPTZ;
"""

MIND_TEMPERAMENT_SQL = """
CREATE TABLE mind_temperaments (
    fish_id UUID PRIMARY KEY REFERENCES fishes(id) ON DELETE CASCADE,
    digest TEXT NOT NULL,
    reading JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# What TA has on its hands; replaces mind_commitments, which stays unread.
MIND_INTENTIONS_SQL = """
CREATE TABLE mind_intentions (
    id UUID PRIMARY KEY,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    session_id UUID REFERENCES fish_sessions(id) ON DELETE CASCADE,
    goal TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('user', 'said', 'wanted')),
    end_condition TEXT NOT NULL DEFAULT '',
    plan TEXT NOT NULL DEFAULT '',
    next TEXT NOT NULL CHECK (next IN ('now', 'soon', 'event')),
    status TEXT NOT NULL CHECK (status IN ('active', 'done', 'dropped')),
    created_at TIMESTAMPTZ NOT NULL,
    touched_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_mind_intentions_active ON mind_intentions (fish_id, status);
"""

MIND_INTENTION_INFO_SQL = """
ALTER TABLE mind_intentions
    ADD COLUMN needs_info BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN looked_up_at TIMESTAMPTZ;
"""

MIND_COMMITMENT_PACE_SQL = """
ALTER TABLE mind_commitments
ADD COLUMN pace TEXT NOT NULL DEFAULT 'occasional'
    CHECK (pace IN ('continuous', 'occasional', 'silent'));
"""


FISH_CHARACTER_DETAILS_SQL = """
ALTER TABLE fishes
ADD COLUMN background TEXT,
ADD COLUMN wake_words JSONB,
ADD CONSTRAINT fishes_background_length_check CHECK (
    background IS NULL OR char_length(background) <= 1000
),
ADD CONSTRAINT fishes_wake_words_check CHECK (
    wake_words IS NULL
    OR (
        jsonb_typeof(wake_words) = 'array'
        AND jsonb_array_length(wake_words) BETWEEN 1 AND 4
    )
);
"""


MESSAGE_INPUT_MODES_SQL = """
CREATE TABLE message_input_modes (
    message_id UUID PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK (mode IN ('typed', 'spoken'))
);
"""


MESSAGE_UNSAID_SQL = """
CREATE TABLE message_unsaid (
    message_id UUID PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    content TEXT NOT NULL CHECK (char_length(content) BETWEEN 1 AND 32000)
);
"""

# Pictures and text files sent along with a chat message. An upload waits
# with no message until the message that carries it is saved.
LLM_CALL_USAGE_SQL = """
ALTER TABLE llm_call_logs
    ADD COLUMN input_tokens INTEGER,
    ADD COLUMN output_tokens INTEGER,
    ADD COLUMN purpose TEXT;
CREATE INDEX idx_llm_call_logs_started_at ON llm_call_logs(started_at);
UPDATE llm_call_logs SET
    input_tokens = COALESCE(
        (response->'usage'->>'prompt_tokens')::integer,
        (response->'usage'->>'input_tokens')::integer,
        (response->'body'->'usage'->>'prompt_tokens')::integer,
        (response->'body'->'usage'->>'input_tokens')::integer),
    output_tokens = COALESCE(
        (response->'usage'->>'completion_tokens')::integer,
        (response->'usage'->>'output_tokens')::integer,
        (response->'body'->'usage'->>'completion_tokens')::integer,
        (response->'body'->'usage'->>'output_tokens')::integer)
WHERE started_at > now() - interval '8 days'
  AND jsonb_typeof(response) = 'object';
"""

MESSAGE_FAILURE_SQL = """
ALTER TABLE messages
    ADD COLUMN failure VARCHAR(32);
"""

MESSAGE_ATTACHMENTS_SQL = """
CREATE TABLE message_attachments (
    id UUID PRIMARY KEY,
    fish_id UUID NOT NULL,
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    kind VARCHAR(8) NOT NULL CHECK (kind IN ('image', 'text')),
    name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    data BYTEA,
    text_content TEXT,
    width INTEGER,
    height INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((kind = 'image') = (data IS NOT NULL)),
    CHECK ((kind = 'text') = (text_content IS NOT NULL))
);
CREATE INDEX message_attachments_message_idx ON message_attachments (message_id);
"""


MIGRATIONS = (
    Migration(version=1, sql=INITIAL_SCHEMA_SQL),
    Migration(version=2, sql=PERFORMANCE_STEPS_SQL),
    Migration(version=3, sql=FISH_LIFECYCLE_SQL),
    Migration(version=4, sql=FACTS_AND_BACKGROUND_PROGRESS_SQL),
    Migration(version=5, sql=DYNAMIC_SNAPSHOT_SQL),
    Migration(version=6, sql=SIMPLIFY_SNAPSHOT_CATEGORIES_SQL),
    Migration(version=7, sql=FACT_EMBEDDINGS_SQL),
    Migration(version=8, sql=EVALUATION_QUESTIONS_SQL),
    Migration(version=9, sql=BACKGROUND_OWNER_SQL),
    Migration(version=10, sql=HISTORY_TOKEN_WINDOW_SQL),
    Migration(version=11, sql=FACT_EXTRACTION_FAILURES_SQL),
    Migration(version=12, sql=FACT_STRUCTURED_SEMANTICS_SQL),
    Migration(version=13, sql=SNAPSHOT_VERSIONS_SQL),
    Migration(version=14, sql=FACT_RELATIONS_SQL),
    Migration(version=15, sql=ENTITY_INDEXES_SQL),
    Migration(version=16, sql=FISH_SOUL_SQL),
    Migration(version=17, sql=FISH_SOUL_PROFILE_SQL),
    Migration(version=18, sql=SECOND_PRECISION_TIMESTAMPS_SQL),
    Migration(version=19, sql=ASSISTANT_FISH_SQL),
    Migration(version=20, sql=SESSION_SCOPING_SQL),
    Migration(version=21, sql=TEMPORAL_RECALL_INDEX_SQL),
    Migration(version=22, sql=FISH_EMBEDDING_SPACES_SQL),
    Migration(version=23, sql=FISH_VOICE_CONFIG_SQL),
    Migration(version=24, sql=HOME_ACTION_RUNS_SQL),
    Migration(version=25, sql=HOME_ACTION_EVENT_SEQUENCE_SQL),
    Migration(version=26, sql=HOME_ACTION_PARTIAL_STATUS_SQL),
    Migration(version=27, sql=RELATION_GROUP_EMBEDDINGS_SQL),
    Migration(version=28, sql=RECALL_MASK_EMBEDDINGS_SQL),
    Migration(version=29, sql=REPAIR_RELATION_MEMBER_COUNTS_SQL),
    Migration(version=30, sql=NORMALIZED_SNAPSHOT_ITEMS_SQL),
    Migration(version=31, sql=SNAPSHOT_ITEM_RELATION_EMBEDDINGS_SQL),
    Migration(version=32, sql=AGENT_TASK_CHANNEL_SQL),
    Migration(version=33, sql=AGENT_TASK_LOCAL_SESSION_SCOPE_SQL),
    Migration(version=34, sql=AGENT_RUN_FISH_SESSION_HISTORY_SOURCE_SQL),
    Migration(version=35, sql=AGENT_RUN_COMBINED_HISTORY_SOURCE_SQL),
    Migration(version=36, sql=OBSERVATIONS_SQL),
    Migration(version=37, sql=BACKGROUND_ASSISTANT_DELIVERIES_SQL),
    Migration(version=38, sql=COMPUTER_HISTORY_SUMMARY_CHECKPOINTS_SQL),
    Migration(version=39, sql=ENVIRONMENT_HISTORY_SUMMARY_LENGTH_SQL),
    Migration(version=40, sql=AUTONOMOUS_AGENT_RUN_SQL),
    Migration(version=41, sql=PROACTIVE_STREAM_LIFECYCLE_SQL),
    Migration(version=42, sql=PLUGIN_EVENTS_SQL),
    Migration(version=43, sql=PLUGIN_TASKS_SQL),
    Migration(version=44, sql=LLM_CALL_LOGS_SQL),
    Migration(version=45, sql=SESSION_HISTORY_SQL),
    Migration(version=46, sql=SESSION_CONTEXT_PROGRESS_SQL),
    Migration(version=47, sql=SESSION_HISTORY_VISIBILITY_SQL),
    Migration(version=48, sql=FISH_CHARACTER_DETAILS_SQL),
    Migration(version=49, sql=MESSAGE_INPUT_MODES_SQL),
    Migration(version=50, sql=MESSAGE_UNSAID_SQL),
    Migration(version=51, sql=MIND_SQL),
    Migration(version=52, sql=MIND_COMMITMENT_PACE_SQL),
    Migration(version=53, sql=MIND_TEMPERAMENT_SQL),
    Migration(version=54, sql=MIND_COMMITMENT_STEP_SQL),
    Migration(version=55, sql=MIND_COMMITMENT_PLAN_SQL),
    Migration(version=56, sql=MIND_COMMITMENT_MATERIAL_SQL),
    Migration(version=57, sql=MIND_INTENTIONS_SQL),
    Migration(version=58, sql=MIND_INTENTION_INFO_SQL),
    Migration(version=59, sql=MESSAGE_ATTACHMENTS_SQL),
    Migration(version=60, sql=MESSAGE_FAILURE_SQL),
    Migration(version=61, sql=LLM_CALL_USAGE_SQL),
)


class Transaction(Protocol):
    async def __aenter__(self) -> Any: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Any: ...


class MigrationConnection(Protocol):
    async def execute(self, query: str, *args: Any) -> str: ...

    async def fetch(
        self,
        query: str,
        *args: Any,
    ) -> Sequence[Mapping[str, object]]: ...

    def transaction(self) -> Transaction: ...


class ConnectionAcquisition(Protocol):
    async def __aenter__(self) -> MigrationConnection: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Any: ...


class MigrationPool(Protocol):
    def acquire(self) -> ConnectionAcquisition: ...


async def run_migrations(pool: MigrationPool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            "SELECT pg_advisory_lock($1)",
            ADVISORY_LOCK_ID,
        )
        try:
            await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
            records = await connection.fetch(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
            applied_versions = {cast(int, record["version"]) for record in records}

            for migration in sorted(MIGRATIONS, key=lambda item: item.version):
                if migration.version in applied_versions:
                    continue
                async with connection.transaction():
                    await connection.execute(migration.sql)
                    await connection.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)",
                        migration.version,
                    )
        finally:
            await connection.execute(
                "SELECT pg_advisory_unlock($1)",
                ADVISORY_LOCK_ID,
            )
