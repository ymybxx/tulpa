"""A durable change inbox keeps context updates independent of request lifetimes."""

SESSION_HISTORY_SQL = """
CREATE TABLE session_history_contexts (
    session_id UUID PRIMARY KEY REFERENCES fish_sessions(id) ON DELETE CASCADE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    initialized BOOLEAN NOT NULL DEFAULT false
);
CREATE TABLE session_history_chunks (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES fish_sessions(id) ON DELETE CASCADE,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    revision BIGINT NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    content_revision BIGINT NOT NULL DEFAULT 0,
    CHECK (ended_at >= started_at)
);
CREATE INDEX session_history_chunks_time
ON session_history_chunks(session_id, started_at);
CREATE TABLE session_history_units (
    source_kind TEXT NOT NULL,
    source_id UUID NOT NULL,
    chunk_id UUID NOT NULL REFERENCES session_history_chunks(id) ON DELETE CASCADE,
    label TEXT NOT NULL CHECK (label IN ('u','a','as','event')),
    content TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ NOT NULL,
    source_sequence BIGINT NOT NULL,
    insertion_revision BIGINT NOT NULL DEFAULT 0,
    summarized BOOLEAN NOT NULL DEFAULT false,
    PRIMARY KEY (source_kind, source_id)
);
CREATE INDEX session_history_units_chunk ON session_history_units(chunk_id);
CREATE TABLE session_history_pending (
    source_kind TEXT NOT NULL,
    source_id UUID NOT NULL,
    fish_id UUID NOT NULL REFERENCES fishes(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES fish_sessions(id) ON DELETE CASCADE,
    queued_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (source_kind, source_id)
);
CREATE INDEX session_history_pending_session ON session_history_pending(session_id);

CREATE FUNCTION history_observation_session(fish UUID, occurred TIMESTAMPTZ, data JSONB)
RETURNS UUID LANGUAGE SQL STABLE AS $$
    SELECT id FROM fish_sessions
    WHERE fish_id=fish AND (
        CASE WHEN COALESCE(data->>'session_id',data->>'sessionId','')
            ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        THEN id=COALESCE(data->>'session_id',data->>'sessionId')::uuid
        ELSE created_at<=occurred END
    ) ORDER BY created_at DESC,id DESC LIMIT 1;
$$;

CREATE FUNCTION enqueue_session_history() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    item RECORD;
    target UUID;
    source_type TEXT;
BEGIN
    IF TG_OP='DELETE' THEN item:=OLD; ELSE item:=NEW; END IF;
    IF TG_TABLE_NAME='messages' THEN
        target:=item.session_id;
        source_type:='message';
        IF TG_OP<>'DELETE' AND NEW.role<>'user'
            AND NEW.status NOT IN ('completed','interrupted')
        THEN RETURN NULL; END IF;
        IF TG_OP='UPDATE' AND NEW.content IS NOT DISTINCT FROM OLD.content
            AND NEW.status IS NOT DISTINCT FROM OLD.status
            AND NEW.consumed_offset IS NOT DISTINCT FROM OLD.consumed_offset
            AND (NEW.fact_extraction_status='waiting_consumption')
                IS NOT DISTINCT FROM (OLD.fact_extraction_status='waiting_consumption')
        THEN RETURN NEW; END IF;
    ELSE
        IF NOT (item.kind='event' AND item.observation_type<>'plugin.definition'
            OR item.source='environment_history'
                AND item.observation_type IN (
                    'semantic_snapshot_5s','semantic_snapshot_10s'))
        THEN RETURN NULL; END IF;
        target:=history_observation_session(item.fish_id,item.started_at,item.payload);
        source_type:='observation';
    END IF;
    IF target IS NOT NULL AND EXISTS(SELECT 1 FROM fish_sessions WHERE id=target) THEN
        INSERT INTO session_history_pending(source_kind,source_id,fish_id,session_id)
        VALUES(source_type,item.id,item.fish_id,target)
        ON CONFLICT(source_kind,source_id) DO UPDATE
        SET queued_at=clock_timestamp(),session_id=EXCLUDED.session_id;
    END IF;
    RETURN NULL;
END;
$$;
CREATE TRIGGER session_history_message_changed
AFTER INSERT OR UPDATE OR DELETE ON messages
FOR EACH ROW EXECUTE FUNCTION enqueue_session_history();
CREATE TRIGGER session_history_observation_changed
AFTER INSERT OR UPDATE OR DELETE ON observations
FOR EACH ROW EXECUTE FUNCTION enqueue_session_history();
"""

# Source writes advance this fence before their transaction commits, including
# changes still waiting in the projection inbox. Model calls never hold the lock.
SESSION_CONTEXT_PROGRESS_SQL = """
CREATE TABLE session_context_progress (
    session_id UUID PRIMARY KEY REFERENCES fish_sessions(id) ON DELETE CASCADE,
    revision BIGINT NOT NULL DEFAULT 0
);
INSERT INTO session_context_progress(session_id) SELECT id FROM fish_sessions;

CREATE OR REPLACE FUNCTION initialize_history_progress()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO session_context_progress(session_id) VALUES(NEW.id);
    RETURN NULL;
END;
$$;
CREATE TRIGGER session_history_progress_created
AFTER INSERT ON fish_sessions
FOR EACH ROW EXECUTE FUNCTION initialize_history_progress();

CREATE OR REPLACE FUNCTION advance_history_context(fish UUID, session UUID)
RETURNS void LANGUAGE SQL AS $$
    INSERT INTO session_context_progress(session_id,revision)
    SELECT id,1 FROM fish_sessions WHERE id=session AND fish_id=fish
    ON CONFLICT(session_id) DO UPDATE
    SET revision=session_context_progress.revision+1;
$$;

CREATE OR REPLACE FUNCTION advance_pending_history_context()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM advance_history_context(NEW.fish_id,NEW.session_id);
    IF TG_OP='UPDATE' AND OLD.session_id IS DISTINCT FROM NEW.session_id THEN
        PERFORM advance_history_context(OLD.fish_id,OLD.session_id);
    END IF;
    RETURN NULL;
END;
$$;
CREATE TRIGGER session_history_pending_progress
AFTER INSERT OR UPDATE ON session_history_pending
FOR EACH ROW EXECUTE FUNCTION advance_pending_history_context();

CREATE OR REPLACE FUNCTION advance_profile_history_context()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE session_context_progress p SET revision=p.revision+1
    FROM fish_sessions s WHERE s.id=p.session_id AND s.fish_id=NEW.id;
    RETURN NULL;
END;
$$;
CREATE TRIGGER session_history_profile_progress
AFTER UPDATE OF role,setting,name,deleted_at,snapshot_storage_version ON fishes
FOR EACH ROW WHEN (
    (OLD.role,OLD.setting,OLD.name,OLD.deleted_at,OLD.snapshot_storage_version)
    IS DISTINCT FROM
    (NEW.role,NEW.setting,NEW.name,NEW.deleted_at,NEW.snapshot_storage_version)
) EXECUTE FUNCTION advance_profile_history_context();

CREATE OR REPLACE FUNCTION advance_static_history_context()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE item RECORD; owner UUID;
BEGIN
    IF TG_OP='DELETE' THEN item:=OLD; ELSE item:=NEW; END IF;
    IF TG_TABLE_NAME='memory_snapshots' THEN
        IF item.snapshot_type<>'static' THEN RETURN NULL; END IF;
        owner:=item.fish_id;
    ELSE
        SELECT fish_id INTO owner FROM memory_snapshots
        WHERE id=item.snapshot_id AND snapshot_type='static';
    END IF;
    UPDATE session_context_progress p SET revision=p.revision+1
    FROM fish_sessions s WHERE s.id=p.session_id AND s.fish_id=owner;
    RETURN NULL;
END;
$$;
CREATE TRIGGER session_history_static_progress
AFTER INSERT OR UPDATE OR DELETE ON memory_snapshots
FOR EACH ROW EXECUTE FUNCTION advance_static_history_context();
CREATE TRIGGER session_history_static_item_progress
AFTER INSERT OR UPDATE OR DELETE ON memory_snapshot_items
FOR EACH ROW EXECUTE FUNCTION advance_static_history_context();
"""

# Hidden capture receipts remain durable, but do not change model context.
# Visibility transitions still enqueue removals/additions before projection.
SESSION_HISTORY_VISIBILITY_SQL = """
CREATE OR REPLACE FUNCTION enqueue_session_history() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    item RECORD;
    target UUID;
    source_type TEXT;
BEGIN
    IF TG_OP='DELETE' THEN item:=OLD; ELSE item:=NEW; END IF;
    IF TG_TABLE_NAME='messages' THEN
        target:=item.session_id;
        source_type:='message';
        IF TG_OP<>'DELETE' AND NEW.role<>'user'
            AND NEW.status NOT IN ('completed','interrupted')
        THEN RETURN NULL; END IF;
        IF TG_OP='UPDATE' AND NEW.content IS NOT DISTINCT FROM OLD.content
            AND NEW.status IS NOT DISTINCT FROM OLD.status
            AND NEW.consumed_offset IS NOT DISTINCT FROM OLD.consumed_offset
            AND (NEW.fact_extraction_status='waiting_consumption')
                IS NOT DISTINCT FROM (OLD.fact_extraction_status='waiting_consumption')
        THEN RETURN NEW; END IF;
    ELSE
        IF NOT (item.kind='event' AND item.observation_type<>'plugin.definition'
            OR item.source='environment_history'
                AND item.observation_type IN (
                    'semantic_snapshot_5s','semantic_snapshot_10s'))
        THEN RETURN NULL; END IF;
        IF TG_OP='UPDATE' THEN
            IF OLD.payload->'include_in_context'='false'::jsonb
                AND NEW.payload->'include_in_context'='false'::jsonb
            THEN RETURN NULL; END IF;
        ELSIF item.payload->'include_in_context'='false'::jsonb THEN
            RETURN NULL;
        END IF;
        target:=history_observation_session(item.fish_id,item.started_at,item.payload);
        source_type:='observation';
    END IF;
    IF target IS NOT NULL AND EXISTS(SELECT 1 FROM fish_sessions WHERE id=target) THEN
        INSERT INTO session_history_pending(source_kind,source_id,fish_id,session_id)
        VALUES(source_type,item.id,item.fish_id,target)
        ON CONFLICT(source_kind,source_id) DO UPDATE
        SET queued_at=clock_timestamp(),session_id=EXCLUDED.session_id;
    END IF;
    RETURN NULL;
END;
$$;
"""
