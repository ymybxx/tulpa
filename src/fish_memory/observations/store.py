from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.db.json_values import canonical_json_object
from fish_memory.observations.computer_history import (
    COMPUTER_HISTORY_OBSERVATION_TYPE,
    COMPUTER_HISTORY_SCOPE_KEY,
    COMPUTER_HISTORY_SOURCE,
    ComputerHistorySegmentConflictError,
    ComputerHistorySummaryCheckpoint,
)
from fish_memory.observations.models import (
    AppRecordingPolicy,
    ImportantObservationEvent,
    Observation,
    ObservationKind,
    ObservationQuery,
    StateObservation,
    normalize_app_bundle_id,
    redact_sensitive_observation_text,
)

_MAX_PAYLOAD_BYTES = 32_768


@dataclass(frozen=True, slots=True)
class ObservationContextGuard:
    session_id: UUID
    revision: int
    is_current: Callable[[], bool]


class _ContextChanged(Exception):
    """Roll back an event whose in-memory owner changed during its DB write."""


_OBSERVATION_COLUMNS = """
id,
sequence,
fish_id,
kind,
source,
observation_type,
scope_key,
semantic_key,
summary,
payload,
app_bundle_id,
app_name,
started_at,
ended_at,
created_at,
updated_at
"""

_COMPUTER_HISTORY_CHECKPOINT_COLUMNS = """
fish_id,
segment_id,
started_at,
ended_at,
revision,
through_event_count,
through_byte_offset,
through_suppressed_event_count,
delta_chain_digest,
full_summary,
applications,
inherited_observation_ids,
delta_capsules,
last_request_id,
last_delta_digest,
last_successful_summary_at,
status,
finalized_at,
observation_id,
final_event_digest,
created_at,
updated_at
"""


class ObservationFishNotFoundError(LookupError):
    pass


class StaleStateObservationError(ValueError):
    pass


class ObservationStore:
    """Persistence isolated from Message, Fact, Snapshot, and Recall storage."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def record_state(
        self,
        fish_id: UUID,
        state: StateObservation,
    ) -> Observation | None:
        payload = _payload_json(state.payload)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await _lock_active_fish(connection, fish_id)
                if not await _recording_enabled(
                    connection,
                    fish_id,
                    state.app_bundle_id,
                ):
                    return None
                latest = await connection.fetchrow(
                    f"""
                    SELECT {_OBSERVATION_COLUMNS}
                    FROM observations
                    WHERE fish_id = $1
                      AND kind = 'state'
                      AND source = $2
                      AND observation_type = $3
                      AND scope_key = $4
                    ORDER BY started_at DESC, sequence DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    fish_id,
                    state.source,
                    state.observation_type,
                    state.scope_key,
                )
                if latest is not None and state.observed_at < cast(
                    datetime,
                    latest["ended_at"],
                ):
                    raise StaleStateObservationError(
                        "state observation predates the latest value in its stream"
                    )
                if (
                    latest is not None
                    and latest["semantic_key"] == state.semantic_key
                    and latest["app_bundle_id"] == state.app_bundle_id
                ):
                    row = await connection.fetchrow(
                        f"""
                        UPDATE observations
                        SET ended_at = $2,
                            summary = $3,
                            payload = $4::jsonb,
                            app_name = $5,
                            updated_at = now()
                        WHERE id = $1
                        RETURNING {_OBSERVATION_COLUMNS}
                        """,
                        latest["id"],
                        state.observed_at,
                        state.summary,
                        payload,
                        state.app_name,
                    )
                else:
                    row = await connection.fetchrow(
                        f"""
                        INSERT INTO observations (
                            id,
                            fish_id,
                            kind,
                            source,
                            observation_type,
                            scope_key,
                            semantic_key,
                            summary,
                            payload,
                            app_bundle_id,
                            app_name,
                            started_at,
                            ended_at
                        )
                        VALUES (
                            $1, $2, 'state', $3, $4, $5, $6, $7,
                            $8::jsonb, $9, $10, $11, $11
                        )
                        RETURNING {_OBSERVATION_COLUMNS}
                        """,
                        uuid4(),
                        fish_id,
                        state.source,
                        state.observation_type,
                        state.scope_key,
                        state.semantic_key,
                        state.summary,
                        payload,
                        state.app_bundle_id,
                        state.app_name,
                        state.observed_at,
                    )
        return _required_observation(row)

    async def record_event(
        self,
        fish_id: UUID,
        event: ImportantObservationEvent,
        *,
        guard: ObservationContextGuard | None = None,
    ) -> Observation | None:
        payload = _payload_json(event.payload)
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                await _lock_active_fish(connection, fish_id)
                if not await _recording_enabled(
                    connection,
                    fish_id,
                    event.app_bundle_id,
                ):
                    return None
                if guard is not None:
                    revision = await connection.fetchval(
                        "SELECT p.revision FROM session_context_progress p "
                        "JOIN fish_sessions s ON s.id=p.session_id "
                        "WHERE s.fish_id=$1 AND s.id=$2 FOR UPDATE OF p",
                        fish_id,
                        guard.session_id,
                    )
                    if revision != guard.revision or not guard.is_current():
                        return None
                row = await connection.fetchrow(
                    f"""
                    INSERT INTO observations (
                        id,
                        fish_id,
                        kind,
                        source,
                        observation_type,
                        scope_key,
                        semantic_key,
                        summary,
                        payload,
                        app_bundle_id,
                        app_name,
                        started_at,
                        ended_at
                    )
                    VALUES (
                        $1, $2, 'event', $3, $4, NULL, $5, $6,
                        $7::jsonb, $8, $9, $10, $10
                    )
                    RETURNING {_OBSERVATION_COLUMNS}
                    """,
                    uuid4(),
                    fish_id,
                    event.source,
                    event.observation_type,
                    event.semantic_key,
                    event.summary,
                    payload,
                    event.app_bundle_id,
                    event.app_name,
                    event.occurred_at,
                )
                if guard is not None and not guard.is_current():
                    raise _ContextChanged
        except _ContextChanged:
            return None
        return _required_observation(row)

    async def require_active_fish(self, fish_id: UUID) -> None:
        row = await self._pool.fetchrow(
            """
            SELECT 1
            FROM fishes
            WHERE id = $1 AND deleted_at IS NULL
            """,
            fish_id,
        )
        if row is None:
            raise ObservationFishNotFoundError(str(fish_id))

    async def get_computer_history_segment(
        self,
        fish_id: UUID,
        segment_id: UUID,
    ) -> Observation | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_OBSERVATION_COLUMNS}
            FROM observations
            WHERE fish_id = $1
              AND kind = 'state'
              AND source = $2
              AND observation_type = $3
              AND scope_key = $4
              AND semantic_key = $5
            ORDER BY sequence DESC
            LIMIT 1
            """,
            fish_id,
            COMPUTER_HISTORY_SOURCE,
            COMPUTER_HISTORY_OBSERVATION_TYPE,
            COMPUTER_HISTORY_SCOPE_KEY,
            f"segment:{segment_id}",
        )
        return None if row is None else _observation_from_record(row)

    async def list_recent_computer_history_summaries(
        self,
        fish_id: UUID,
        *,
        before: datetime,
        limit: int,
    ) -> tuple[Observation, ...]:
        if limit < 1 or limit > 3:
            raise ValueError("Computer History inherited summary limit must be 1..3")
        rows = await self._pool.fetch(
            f"""
            SELECT {_OBSERVATION_COLUMNS}
            FROM observations
            WHERE fish_id = $1
              AND kind = 'state'
              AND source = $2
              AND observation_type = $3
              AND scope_key = $4
              AND ended_at <= $5
            ORDER BY ended_at DESC, sequence DESC
            LIMIT $6
            """,
            fish_id,
            COMPUTER_HISTORY_SOURCE,
            COMPUTER_HISTORY_OBSERVATION_TYPE,
            COMPUTER_HISTORY_SCOPE_KEY,
            before,
            limit,
        )
        return tuple(_observation_from_record(row) for row in rows)

    async def get_computer_history_checkpoint(
        self,
        fish_id: UUID,
        segment_id: UUID,
    ) -> ComputerHistorySummaryCheckpoint | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
            FROM computer_history_summary_checkpoints
            WHERE fish_id = $1 AND segment_id = $2
            """,
            fish_id,
            segment_id,
        )
        return None if row is None else _checkpoint_from_record(row)

    async def get_latest_computer_history_checkpoint(
        self,
        fish_id: UUID,
    ) -> ComputerHistorySummaryCheckpoint | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
            FROM computer_history_summary_checkpoints
            WHERE fish_id = $1
              AND status = 'active'
              AND full_summary IS NOT NULL
            ORDER BY started_at DESC, updated_at DESC
            LIMIT 1
            """,
            fish_id,
        )
        return None if row is None else _checkpoint_from_record(row)

    async def delete_stale_active_computer_history_checkpoints(
        self,
        fish_id: UUID,
        *,
        updated_before: datetime,
    ) -> int:
        rows = await self._pool.fetch(
            """
            DELETE FROM computer_history_summary_checkpoints
            WHERE fish_id = $1
              AND status = 'active'
              AND updated_at < $2
            RETURNING segment_id
            """,
            fish_id,
            updated_before,
        )
        return len(rows)

    async def delete_stale_finalized_computer_history_checkpoints(
        self,
        fish_id: UUID,
        *,
        updated_before: datetime,
    ) -> int:
        rows = await self._pool.fetch(
            """
            DELETE FROM computer_history_summary_checkpoints
            WHERE fish_id = $1
              AND status = 'finalized'
              AND updated_at < $2
            RETURNING segment_id
            """,
            fish_id,
            updated_before,
        )
        return len(rows)

    async def acknowledge_finalized_computer_history_checkpoint(
        self,
        fish_id: UUID,
        segment_id: UUID,
        *,
        observation_id: UUID,
        final_event_digest: str,
    ) -> bool:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT status, observation_id, final_event_digest
                    FROM computer_history_summary_checkpoints
                    WHERE fish_id = $1 AND segment_id = $2
                    FOR UPDATE
                    """,
                    fish_id,
                    segment_id,
                )
                if row is None:
                    return False
                if (
                    row["status"] != "finalized"
                    or row["observation_id"] != observation_id
                    or row["final_event_digest"] != final_event_digest
                ):
                    raise ComputerHistorySegmentConflictError(
                        "checkpoint acknowledgement does not match finalized data"
                    )
                deleted = await connection.fetchrow(
                    """
                    DELETE FROM computer_history_summary_checkpoints
                    WHERE fish_id = $1 AND segment_id = $2
                    RETURNING segment_id
                    """,
                    fish_id,
                    segment_id,
                )
        return deleted is not None

    async def save_computer_history_checkpoint(
        self,
        checkpoint: ComputerHistorySummaryCheckpoint,
        *,
        expected_revision: int,
    ) -> ComputerHistorySummaryCheckpoint:
        values = _checkpoint_json_values(checkpoint)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await _lock_active_fish(connection, checkpoint.fish_id)
                existing = await connection.fetchrow(
                    f"""
                    SELECT {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
                    FROM computer_history_summary_checkpoints
                    WHERE fish_id = $1 AND segment_id = $2
                    FOR UPDATE
                    """,
                    checkpoint.fish_id,
                    checkpoint.segment_id,
                )
                if existing is None:
                    if expected_revision != 0 or checkpoint.revision != 1:
                        raise ComputerHistorySegmentConflictError(
                            "checkpoint baseRevision is stale"
                        )
                    row = await connection.fetchrow(
                        f"""
                        INSERT INTO computer_history_summary_checkpoints (
                            fish_id,
                            segment_id,
                            started_at,
                            ended_at,
                            revision,
                            through_event_count,
                            through_byte_offset,
                            through_suppressed_event_count,
                            delta_chain_digest,
                            full_summary,
                            applications,
                            inherited_observation_ids,
                            delta_capsules,
                            last_request_id,
                            last_delta_digest,
                            last_successful_summary_at,
                            status,
                            finalized_at,
                            observation_id,
                            final_event_digest,
                            created_at,
                            updated_at
                        )
                        VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9,
                            $10::jsonb, $11::jsonb, $12::jsonb, $13::jsonb,
                            $14, $15, $16, $17, $18, $19, $20, $21, $22
                        )
                        RETURNING {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
                        """,
                        *_checkpoint_query_args(checkpoint, values),
                    )
                else:
                    if (
                        cast(int, existing["revision"]) != expected_revision
                        or checkpoint.revision != expected_revision + 1
                        or cast(str, existing["status"]) != "active"
                    ):
                        raise ComputerHistorySegmentConflictError(
                            "checkpoint baseRevision is stale"
                        )
                    if (
                        existing["started_at"] != checkpoint.started_at
                        or existing["ended_at"] != checkpoint.ended_at
                    ):
                        raise ComputerHistorySegmentConflictError(
                            "checkpoint interval does not match"
                        )
                    row = await connection.fetchrow(
                        f"""
                        UPDATE computer_history_summary_checkpoints
                        SET revision = $3,
                            through_event_count = $4,
                            through_byte_offset = $5,
                            through_suppressed_event_count = $6,
                            delta_chain_digest = $7,
                            full_summary = $8::jsonb,
                            applications = $9::jsonb,
                            inherited_observation_ids = $10::jsonb,
                            delta_capsules = $11::jsonb,
                            last_request_id = $12,
                            last_delta_digest = $13,
                            last_successful_summary_at = $14,
                            updated_at = $15
                        WHERE fish_id = $1 AND segment_id = $2
                        RETURNING {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
                        """,
                        checkpoint.fish_id,
                        checkpoint.segment_id,
                        checkpoint.revision,
                        checkpoint.through_event_count,
                        checkpoint.through_byte_offset,
                        checkpoint.through_suppressed_event_count,
                        checkpoint.delta_chain_digest,
                        values["full_summary"],
                        values["applications"],
                        values["inherited_observation_ids"],
                        values["delta_capsules"],
                        checkpoint.last_request_id,
                        checkpoint.last_delta_digest,
                        checkpoint.last_successful_summary_at,
                        checkpoint.updated_at,
                    )
        if row is None:
            raise RuntimeError("database did not return Computer History checkpoint")
        return _checkpoint_from_record(row)

    async def finalize_computer_history_checkpoint(
        self,
        checkpoint: ComputerHistorySummaryCheckpoint,
        *,
        expected_revision: int,
        event_digest: str,
        summary: str,
        payload: Mapping[str, object],
    ) -> tuple[ComputerHistorySummaryCheckpoint, Observation]:
        if checkpoint.status != "finalized" or checkpoint.observation_id is None:
            raise ValueError("final checkpoint metadata is incomplete")
        sanitized_payload = _sanitize_computer_history_payload(payload)
        encoded_payload = _payload_json(sanitized_payload)
        summary = redact_sensitive_observation_text(summary)
        values = _checkpoint_json_values(checkpoint)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await _lock_active_fish(connection, checkpoint.fish_id)
                existing_checkpoint = await connection.fetchrow(
                    f"""
                    SELECT {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
                    FROM computer_history_summary_checkpoints
                    WHERE fish_id = $1 AND segment_id = $2
                    FOR UPDATE
                    """,
                    checkpoint.fish_id,
                    checkpoint.segment_id,
                )
                if existing_checkpoint is None:
                    if expected_revision != 0 or checkpoint.revision != 1:
                        raise ComputerHistorySegmentConflictError(
                            "checkpoint baseRevision is stale"
                        )
                elif (
                    cast(str, existing_checkpoint["status"]) == "finalized"
                    and existing_checkpoint["final_event_digest"] == event_digest
                    and existing_checkpoint["through_event_count"]
                    == checkpoint.through_event_count
                    and existing_checkpoint["through_suppressed_event_count"]
                    == checkpoint.through_suppressed_event_count
                ):
                    stored = _checkpoint_from_record(existing_checkpoint)
                    observation = await _get_computer_history_segment_on_connection(
                        connection,
                        checkpoint.fish_id,
                        checkpoint.segment_id,
                    )
                    if observation is None:
                        raise RuntimeError(
                            "finalized checkpoint has no Computer History Observation"
                        )
                    return stored, observation
                elif (
                    cast(int, existing_checkpoint["revision"]) != expected_revision
                    or cast(str, existing_checkpoint["status"]) != "active"
                    or checkpoint.revision != expected_revision + 1
                    or existing_checkpoint["started_at"] != checkpoint.started_at
                    or existing_checkpoint["ended_at"] != checkpoint.ended_at
                ):
                    raise ComputerHistorySegmentConflictError(
                        "checkpoint baseRevision is stale"
                    )

                observation = await _record_computer_history_segment_on_connection(
                    connection,
                    checkpoint.fish_id,
                    observation_id=checkpoint.observation_id,
                    segment_id=checkpoint.segment_id,
                    started_at=checkpoint.started_at,
                    ended_at=checkpoint.ended_at,
                    event_digest=event_digest,
                    summary=summary,
                    sanitized_payload=sanitized_payload,
                    encoded_payload=encoded_payload,
                )
                if observation.id != checkpoint.observation_id:
                    # A legacy full-segment request may have committed the same
                    # Observation immediately before checkpoint finalization.
                    # Reuse that verified row instead of storing a dangling ID.
                    checkpoint = checkpoint.model_copy(
                        update={"observation_id": observation.id}
                    )
                if existing_checkpoint is None:
                    row = await connection.fetchrow(
                        f"""
                        INSERT INTO computer_history_summary_checkpoints (
                            fish_id, segment_id, started_at, ended_at, revision,
                            through_event_count, through_byte_offset,
                            through_suppressed_event_count, delta_chain_digest,
                            full_summary, applications, inherited_observation_ids,
                            delta_capsules, last_request_id, last_delta_digest,
                            last_successful_summary_at, status, finalized_at,
                            observation_id, final_event_digest, created_at, updated_at
                        )
                        VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9,
                            $10::jsonb, $11::jsonb, $12::jsonb, $13::jsonb,
                            $14, $15, $16, $17, $18, $19, $20, $21, $22
                        )
                        RETURNING {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
                        """,
                        *_checkpoint_query_args(checkpoint, values),
                    )
                else:
                    row = await connection.fetchrow(
                        f"""
                        UPDATE computer_history_summary_checkpoints
                        SET revision = $3,
                            through_event_count = $4,
                            through_byte_offset = $5,
                            through_suppressed_event_count = $6,
                            delta_chain_digest = $7,
                            full_summary = $8::jsonb,
                            applications = $9::jsonb,
                            inherited_observation_ids = $10::jsonb,
                            delta_capsules = $11::jsonb,
                            last_request_id = $12,
                            last_delta_digest = $13,
                            last_successful_summary_at = $14,
                            status = 'finalized',
                            finalized_at = $15,
                            observation_id = $16,
                            final_event_digest = $17,
                            updated_at = $18
                        WHERE fish_id = $1 AND segment_id = $2
                        RETURNING {_COMPUTER_HISTORY_CHECKPOINT_COLUMNS}
                        """,
                        checkpoint.fish_id,
                        checkpoint.segment_id,
                        checkpoint.revision,
                        checkpoint.through_event_count,
                        checkpoint.through_byte_offset,
                        checkpoint.through_suppressed_event_count,
                        checkpoint.delta_chain_digest,
                        values["full_summary"],
                        values["applications"],
                        values["inherited_observation_ids"],
                        values["delta_capsules"],
                        checkpoint.last_request_id,
                        checkpoint.last_delta_digest,
                        checkpoint.last_successful_summary_at,
                        checkpoint.finalized_at,
                        checkpoint.observation_id,
                        checkpoint.final_event_digest,
                        checkpoint.updated_at,
                    )
        if row is None:
            raise RuntimeError("database did not return finalized checkpoint")
        return _checkpoint_from_record(row), observation

    async def record_computer_history_segment(
        self,
        fish_id: UUID,
        *,
        segment_id: UUID,
        started_at: datetime,
        ended_at: datetime,
        event_digest: str,
        summary: str,
        payload: Mapping[str, object],
    ) -> Observation:
        sanitized_payload = _sanitize_computer_history_payload(payload)
        encoded_payload = _payload_json(sanitized_payload)
        summary = redact_sensitive_observation_text(summary)
        semantic_key = f"segment:{segment_id}"
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                # The Fish lock serializes competing first writes even without a
                # Computer-History-specific table or conversation-side effects.
                await _lock_active_fish(connection, fish_id)
                existing = await connection.fetchrow(
                    f"""
                    SELECT {_OBSERVATION_COLUMNS}
                    FROM observations
                    WHERE fish_id = $1
                      AND kind = 'state'
                      AND source = $2
                      AND observation_type = $3
                      AND scope_key = $4
                      AND semantic_key = $5
                    ORDER BY sequence DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    fish_id,
                    COMPUTER_HISTORY_SOURCE,
                    COMPUTER_HISTORY_OBSERVATION_TYPE,
                    COMPUTER_HISTORY_SCOPE_KEY,
                    semantic_key,
                )
                if existing is not None:
                    observation = _observation_from_record(existing)
                    if (
                        existing["started_at"] != started_at
                        or existing["ended_at"] != ended_at
                        or observation.payload.get("segmentId") != str(segment_id)
                        or observation.payload.get("eventDigest") != event_digest
                        or observation.payload.get("eventCount")
                        != sanitized_payload.get("eventCount")
                        or observation.payload.get("suppressedEventCount")
                        != sanitized_payload.get("suppressedEventCount")
                    ):
                        raise ComputerHistorySegmentConflictError(
                            "segmentId is already associated with different data"
                        )
                    return observation

                row = await connection.fetchrow(
                    f"""
                    INSERT INTO observations (
                        id,
                        fish_id,
                        kind,
                        source,
                        observation_type,
                        scope_key,
                        semantic_key,
                        summary,
                        payload,
                        app_bundle_id,
                        app_name,
                        started_at,
                        ended_at
                    )
                    VALUES (
                        $1, $2, 'state', $3, $4, $5, $6, $7,
                        $8::jsonb, NULL, NULL, $9, $10
                    )
                    RETURNING {_OBSERVATION_COLUMNS}
                    """,
                    uuid4(),
                    fish_id,
                    COMPUTER_HISTORY_SOURCE,
                    COMPUTER_HISTORY_OBSERVATION_TYPE,
                    COMPUTER_HISTORY_SCOPE_KEY,
                    semantic_key,
                    summary,
                    encoded_payload,
                    started_at,
                    ended_at,
                )
        return _required_observation(row)

    async def get_observation(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> Observation | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_OBSERVATION_COLUMNS}
            FROM observations
            WHERE fish_id = $1 AND id = $2
            """,
            fish_id,
            observation_id,
        )
        return None if row is None else _observation_from_record(row)

    async def get_latest_state_observation(
        self,
        fish_id: UUID,
        *,
        source: str,
        observation_type: str,
        scope_key: str,
    ) -> Observation | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_OBSERVATION_COLUMNS}
            FROM observations
            WHERE fish_id = $1
              AND kind = 'state'
              AND source = $2
              AND observation_type = $3
              AND scope_key = $4
            ORDER BY started_at DESC, sequence DESC
            LIMIT 1
            """,
            fish_id,
            source,
            observation_type,
            scope_key,
        )
        return None if row is None else _observation_from_record(row)

    async def query_observations(
        self,
        fish_id: UUID,
        query: ObservationQuery,
    ) -> tuple[Observation, ...]:
        rows = await self._pool.fetch(
            f"""
            SELECT {_OBSERVATION_COLUMNS}
            FROM observations
            WHERE fish_id = $1
              AND ($2::timestamptz IS NULL OR ended_at >= $2)
              AND ($3::timestamptz IS NULL OR started_at <= $3)
              AND ($4::text IS NULL OR app_bundle_id = $4)
              AND ($5::text IS NULL OR observation_type = $5)
              AND ($6::text IS NULL OR kind = $6)
            ORDER BY ended_at DESC, sequence DESC
            LIMIT $7
            """,
            fish_id,
            query.start_at,
            query.end_at,
            query.app_bundle_id,
            query.observation_type,
            None if query.kind is None else str(query.kind),
            query.limit,
        )
        return tuple(_observation_from_record(row) for row in rows)

    async def delete_observation(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> bool:
        row = await self._pool.fetchrow(
            """
            DELETE FROM observations
            WHERE fish_id = $1 AND id = $2
            RETURNING id
            """,
            fish_id,
            observation_id,
        )
        return row is not None

    async def set_app_recording_policy(
        self,
        fish_id: UUID,
        app_bundle_id: str,
        *,
        recording_enabled: bool,
        app_name: str | None = None,
    ) -> AppRecordingPolicy:
        normalized_bundle_id = normalize_app_bundle_id(app_bundle_id)
        if normalized_bundle_id is None:
            raise ValueError("app_bundle_id must not be empty")
        if app_name is not None:
            app_name = app_name.strip()
            if not app_name or len(app_name) > 200:
                raise ValueError("app_name must contain between 1 and 200 characters")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await _lock_active_fish(connection, fish_id)
                row = await connection.fetchrow(
                    """
                    INSERT INTO observation_app_policies (
                        fish_id,
                        app_bundle_id,
                        app_name,
                        recording_enabled
                    )
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (fish_id, app_bundle_id) DO UPDATE
                    SET app_name = COALESCE(
                            EXCLUDED.app_name,
                            observation_app_policies.app_name
                        ),
                        recording_enabled = EXCLUDED.recording_enabled,
                        updated_at = now()
                    RETURNING *
                    """,
                    fish_id,
                    normalized_bundle_id,
                    app_name,
                    recording_enabled,
                )
        return _required_policy(row)

    async def get_app_recording_policy(
        self,
        fish_id: UUID,
        app_bundle_id: str,
    ) -> AppRecordingPolicy | None:
        normalized_bundle_id = normalize_app_bundle_id(app_bundle_id)
        if normalized_bundle_id is None:
            raise ValueError("app_bundle_id must not be empty")
        row = await self._pool.fetchrow(
            """
            SELECT *
            FROM observation_app_policies
            WHERE fish_id = $1 AND app_bundle_id = $2
            """,
            fish_id,
            normalized_bundle_id,
        )
        return None if row is None else _policy_from_record(row)

    async def list_app_recording_policies(
        self,
        fish_id: UUID,
    ) -> tuple[AppRecordingPolicy, ...]:
        rows = await self._pool.fetch(
            """
            SELECT *
            FROM observation_app_policies
            WHERE fish_id = $1
            ORDER BY app_bundle_id
            """,
            fish_id,
        )
        return tuple(_policy_from_record(row) for row in rows)

    async def is_app_recording_enabled(
        self,
        fish_id: UUID,
        app_bundle_id: str,
    ) -> bool:
        normalized_bundle_id = normalize_app_bundle_id(app_bundle_id)
        if normalized_bundle_id is None:
            raise ValueError("app_bundle_id must not be empty")
        row = await self._pool.fetchrow(
            """
            SELECT COALESCE(policy.recording_enabled, true) AS recording_enabled
            FROM fishes AS fish
            LEFT JOIN observation_app_policies AS policy
              ON policy.fish_id = fish.id
             AND policy.app_bundle_id = $2
            WHERE fish.id = $1 AND fish.deleted_at IS NULL
            """,
            fish_id,
            normalized_bundle_id,
        )
        if row is None:
            raise ObservationFishNotFoundError(str(fish_id))
        return cast(bool, row["recording_enabled"])


async def _lock_active_fish(
    connection: asyncpg.Connection,
    fish_id: UUID,
) -> None:
    row = await connection.fetchrow(
        """
        SELECT id
        FROM fishes
        WHERE id = $1 AND deleted_at IS NULL
        FOR UPDATE
        """,
        fish_id,
    )
    if row is None:
        raise ObservationFishNotFoundError(str(fish_id))


async def _recording_enabled(
    connection: asyncpg.Connection,
    fish_id: UUID,
    app_bundle_id: str | None,
) -> bool:
    if app_bundle_id is None:
        return True
    value = await connection.fetchval(
        """
        SELECT recording_enabled
        FROM observation_app_policies
        WHERE fish_id = $1 AND app_bundle_id = $2
        """,
        fish_id,
        app_bundle_id,
    )
    return True if value is None else cast(bool, value)


async def _get_computer_history_segment_on_connection(
    connection: asyncpg.Connection,
    fish_id: UUID,
    segment_id: UUID,
) -> Observation | None:
    row = await connection.fetchrow(
        f"""
        SELECT {_OBSERVATION_COLUMNS}
        FROM observations
        WHERE fish_id = $1
          AND kind = 'state'
          AND source = $2
          AND observation_type = $3
          AND scope_key = $4
          AND semantic_key = $5
        ORDER BY sequence DESC
        LIMIT 1
        FOR UPDATE
        """,
        fish_id,
        COMPUTER_HISTORY_SOURCE,
        COMPUTER_HISTORY_OBSERVATION_TYPE,
        COMPUTER_HISTORY_SCOPE_KEY,
        f"segment:{segment_id}",
    )
    return None if row is None else _observation_from_record(row)


async def _record_computer_history_segment_on_connection(
    connection: asyncpg.Connection,
    fish_id: UUID,
    *,
    observation_id: UUID,
    segment_id: UUID,
    started_at: datetime,
    ended_at: datetime,
    event_digest: str,
    summary: str,
    sanitized_payload: Mapping[str, object],
    encoded_payload: str,
) -> Observation:
    existing = await _get_computer_history_segment_on_connection(
        connection,
        fish_id,
        segment_id,
    )
    if existing is not None:
        if (
            existing.started_at != started_at
            or existing.ended_at != ended_at
            or existing.payload.get("segmentId") != str(segment_id)
            or existing.payload.get("eventDigest") != event_digest
            or existing.payload.get("eventCount") != sanitized_payload.get("eventCount")
            or existing.payload.get("suppressedEventCount")
            != sanitized_payload.get("suppressedEventCount")
        ):
            raise ComputerHistorySegmentConflictError(
                "segmentId is already associated with different data"
            )
        row = await connection.fetchrow(
            f"""
            UPDATE observations
            SET summary = $2,
                payload = $3::jsonb,
                updated_at = now()
            WHERE id = $1
            RETURNING {_OBSERVATION_COLUMNS}
            """,
            existing.id,
            summary,
            encoded_payload,
        )
        return _required_observation(row)

    row = await connection.fetchrow(
        f"""
        INSERT INTO observations (
            id,
            fish_id,
            kind,
            source,
            observation_type,
            scope_key,
            semantic_key,
            summary,
            payload,
            app_bundle_id,
            app_name,
            started_at,
            ended_at
        )
        VALUES (
            $1, $2, 'state', $3, $4, $5, $6, $7,
            $8::jsonb, NULL, NULL, $9, $10
        )
        RETURNING {_OBSERVATION_COLUMNS}
        """,
        observation_id,
        fish_id,
        COMPUTER_HISTORY_SOURCE,
        COMPUTER_HISTORY_OBSERVATION_TYPE,
        COMPUTER_HISTORY_SCOPE_KEY,
        f"segment:{segment_id}",
        summary,
        encoded_payload,
        started_at,
        ended_at,
    )
    return _required_observation(row)


def _checkpoint_json_values(
    checkpoint: ComputerHistorySummaryCheckpoint,
) -> dict[str, str | None]:
    full_summary = (
        None
        if checkpoint.full_summary is None
        else json.dumps(
            checkpoint.full_summary.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    applications = json.dumps(
        checkpoint.applications,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    inherited = json.dumps(
        [str(value) for value in checkpoint.inherited_observation_ids],
        separators=(",", ":"),
    )
    capsules = json.dumps(
        [
            value.model_dump(mode="json", by_alias=True)
            for value in checkpoint.delta_capsules
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(capsules.encode("utf-8")) > 512 * 1_024:
        raise ValueError("Computer History checkpoint capsules exceed 512 KiB")
    return {
        "full_summary": full_summary,
        "applications": applications,
        "inherited_observation_ids": inherited,
        "delta_capsules": capsules,
    }


def _checkpoint_query_args(
    checkpoint: ComputerHistorySummaryCheckpoint,
    values: Mapping[str, str | None],
) -> tuple[object, ...]:
    return (
        checkpoint.fish_id,
        checkpoint.segment_id,
        checkpoint.started_at,
        checkpoint.ended_at,
        checkpoint.revision,
        checkpoint.through_event_count,
        checkpoint.through_byte_offset,
        checkpoint.through_suppressed_event_count,
        checkpoint.delta_chain_digest,
        values["full_summary"],
        values["applications"],
        values["inherited_observation_ids"],
        values["delta_capsules"],
        checkpoint.last_request_id,
        checkpoint.last_delta_digest,
        checkpoint.last_successful_summary_at,
        checkpoint.status,
        checkpoint.finalized_at,
        checkpoint.observation_id,
        checkpoint.final_event_digest,
        checkpoint.created_at,
        checkpoint.updated_at,
    )


def _checkpoint_from_record(
    record: Mapping[str, object],
) -> ComputerHistorySummaryCheckpoint:
    return ComputerHistorySummaryCheckpoint.model_validate(
        {
            "schemaVersion": 1,
            "fishId": str(record["fish_id"]),
            "segmentId": str(record["segment_id"]),
            "startedAt": cast(datetime, record["started_at"]).isoformat(),
            "endedAt": cast(datetime, record["ended_at"]).isoformat(),
            "revision": record["revision"],
            "throughEventCount": record["through_event_count"],
            "throughByteOffset": record["through_byte_offset"],
            "throughSuppressedEventCount": record["through_suppressed_event_count"],
            "deltaChainDigest": record["delta_chain_digest"],
            "fullSummary": _json_value(record["full_summary"]),
            "applications": _json_value(record["applications"]),
            "inheritedObservationIds": _json_value(record["inherited_observation_ids"]),
            "deltaCapsules": _json_value(record["delta_capsules"]),
            "lastRequestId": (
                None
                if record["last_request_id"] is None
                else str(record["last_request_id"])
            ),
            "lastDeltaDigest": record["last_delta_digest"],
            "lastSuccessfulSummaryAt": (
                None
                if record["last_successful_summary_at"] is None
                else cast(datetime, record["last_successful_summary_at"]).isoformat()
            ),
            "status": record["status"],
            "finalizedAt": (
                None
                if record["finalized_at"] is None
                else cast(datetime, record["finalized_at"]).isoformat()
            ),
            "observationId": (
                None
                if record["observation_id"] is None
                else str(record["observation_id"])
            ),
            "finalEventDigest": record["final_event_digest"],
            "createdAt": cast(datetime, record["created_at"]).isoformat(),
            "updatedAt": cast(datetime, record["updated_at"]).isoformat(),
        }
    )


def _json_value(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


def _payload_json(payload: Mapping[str, object]) -> str:
    encoded = canonical_json_object(
        payload,
        label="observation payload",
        maximum_bytes=_MAX_PAYLOAD_BYTES,
        none_as_empty=True,
    )
    if encoded is None:
        raise RuntimeError("observation payload normalization returned no JSON")
    return encoded


def _sanitize_computer_history_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    sanitized = dict(payload)
    digest = payload.get("eventDigest")
    protected_digest = digest if isinstance(digest, str) else None

    def redact(value: str) -> str:
        marker = "FISH_MEMORY_EVENT_DIGEST"
        protected = (
            value.replace(protected_digest, marker)
            if protected_digest is not None
            else value
        )
        redacted = redact_sensitive_observation_text(protected)
        return (
            redacted.replace(marker, protected_digest)
            if protected_digest is not None
            else redacted
        )

    for key in ("documentMarkdown", "title", "description"):
        value = sanitized.get(key)
        if isinstance(value, str):
            sanitized[key] = redact(value)

    applications = sanitized.get("applications")
    if isinstance(applications, list):
        sanitized["applications"] = [
            redact(value) if isinstance(value, str) else value for value in applications
        ]

    suggestion = sanitized.get("suggestion")
    if isinstance(suggestion, Mapping):
        sanitized_suggestion = dict(suggestion)
        for key in ("name", "description"):
            value = sanitized_suggestion.get(key)
            if isinstance(value, str):
                sanitized_suggestion[key] = redact(value)
        sanitized["suggestion"] = sanitized_suggestion
    return sanitized


def _required_observation(
    record: Mapping[str, object] | None,
) -> Observation:
    if record is None:
        raise RuntimeError("database did not return the observation")
    return _observation_from_record(record)


def _observation_from_record(record: Mapping[str, object]) -> Observation:
    payload = _json_object(record["payload"])
    return Observation(
        id=cast(UUID, record["id"]),
        sequence=cast(int, record["sequence"]),
        fish_id=cast(UUID, record["fish_id"]),
        kind=ObservationKind(cast(str, record["kind"])),
        source=cast(str, record["source"]),
        observation_type=cast(str, record["observation_type"]),
        scope_key=cast(str | None, record["scope_key"]),
        semantic_key=cast(str, record["semantic_key"]),
        summary=cast(str, record["summary"]),
        payload=payload,
        app_bundle_id=cast(str | None, record["app_bundle_id"]),
        app_name=cast(str | None, record["app_name"]),
        started_at=cast(datetime, record["started_at"]),
        ended_at=cast(datetime, record["ended_at"]),
        created_at=cast(datetime, record["created_at"]),
        updated_at=cast(datetime, record["updated_at"]),
    )


def _required_policy(
    record: Mapping[str, object] | None,
) -> AppRecordingPolicy:
    if record is None:
        raise RuntimeError("database did not return the observation app policy")
    return _policy_from_record(record)


def _policy_from_record(record: Mapping[str, object]) -> AppRecordingPolicy:
    return AppRecordingPolicy(
        fish_id=cast(UUID, record["fish_id"]),
        app_bundle_id=cast(str, record["app_bundle_id"]),
        app_name=cast(str | None, record["app_name"]),
        recording_enabled=cast(bool, record["recording_enabled"]),
        created_at=cast(datetime, record["created_at"]),
        updated_at=cast(datetime, record["updated_at"]),
    )


def _json_object(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        decoded = json.loads(value)
    else:
        decoded = value
    if not isinstance(decoded, Mapping):
        raise RuntimeError("database observation payload is not a JSON object")
    return cast(Mapping[str, object], decoded)
