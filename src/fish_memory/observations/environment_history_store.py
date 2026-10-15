"""Semantic history uses Observation storage; raw frames and PCM never enter SQL."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from fish_memory.observations.environment_history_audio import system_audio_words
from fish_memory.observations.environment_history_hierarchy import (
    level_for_period,
    parent_level_for_period,
    snapshot_levels,
)
from fish_memory.observations.environment_history_models import (
    HISTORY_SOURCE,
    MINUTE_TYPE,
    WINDOW_TYPE,
    SemanticDocument,
    SemanticSnapshot,
    observation_type_for_period,
)
from fish_memory.observations.models import redact_sensitive_observation_text


class EnvironmentHistoryConflict(ValueError):
    pass


_INDEX_SECONDS = 10
_TIMESTAMP_TOLERANCE_SECONDS = 0.001


def _usable_window_sql(alias: str) -> str:
    # Missing metadata alone does not prove that a legacy observation lacked
    # media. Only the explicit zero-sample evidence plus no frame times does.
    return f"""NOT COALESCE(
        COALESCE({alias}.payload #> '{{evidence,frameTimes}}', '[]'::jsonb)
            IN ('[]'::jsonb, 'null'::jsonb)
        AND {alias}.payload #> '{{evidence,audioCapturedSamples}}' = '0'::jsonb,
        false
    )"""


def _usable_minute_sql(alias: str) -> str:
    # Old minute documents have no direct evidence. Their immutable source IDs
    # identify whether they incorporated a known empty window. Do not let such
    # a minute hide the valid ten-second observations that preceded it.
    return f"""{_usable_window_sql(alias)} AND NOT EXISTS (
        SELECT 1
        FROM jsonb_array_elements_text(
            CASE
                WHEN jsonb_typeof({alias}.payload->'sourceSnapshotIds')='array'
                THEN {alias}.payload->'sourceSnapshotIds'
                ELSE '[]'::jsonb
            END
        ) AS source_id(value)
        JOIN observations source_window
          ON source_window.id=source_id.value::uuid
         AND source_window.fish_id={alias}.fish_id
        WHERE source_window.source={alias}.source
          AND source_window.observation_type='{WINDOW_TYPE}'
          AND NOT ({_usable_window_sql("source_window")})
    )"""


class EnvironmentHistoryStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get(self, fish_id: UUID, snapshot_id: UUID) -> SemanticSnapshot | None:
        # Exact identity lookups retain old records for audit and idempotency.
        # Context and merge readers below exclude proven no-media observations.
        row = await self._pool.fetchrow(
            "SELECT * FROM observations WHERE fish_id=$1 AND id=$2 AND source=$3",
            fish_id,
            snapshot_id,
            HISTORY_SOURCE,
        )
        return _snapshot(row) if row is not None else None

    async def save(
        self,
        value: SemanticSnapshot,
        *,
        evidence: Mapping[str, object] | None = None,
    ) -> SemanticSnapshot:
        observation_type = observation_type_for_period(value.period_seconds)
        document = _redacted_document(value.document)
        payload = {
            "session_id": str(value.session_id)
            if value.session_id
            else (evidence or {}).get("sessionId"),
            "captureSessionId": str(value.capture_session_id),
            "windowIndex": value.window_index,
            "periodSeconds": value.period_seconds,
            "inputDigest": value.input_digest,
            "document": document.model_dump(),
            "sourceSnapshotIds": [str(item) for item in value.source_ids],
            "evidence": dict(evidence or {}),
        }
        if not document.history_summary:
            # Keep the capture receipt for retries without advancing visible
            # context or invalidating an in-flight background thought.
            payload["include_in_context"] = False
        encoded = _encode_snapshot_payload(payload)
        await self._pool.execute(
            """
            INSERT INTO observations (
                id,fish_id,kind,source,observation_type,scope_key,semantic_key,
                summary,payload,started_at,ended_at
            ) SELECT $1,$2,'state',$3,$4,$5,$6,$7,$8::jsonb,$9,$10
              WHERE EXISTS(SELECT 1 FROM fishes WHERE id=$2 AND deleted_at IS NULL)
            ON CONFLICT(id) DO NOTHING
            """,
            value.id,
            value.fish_id,
            HISTORY_SOURCE,
            observation_type,
            f"capture:{value.capture_session_id}:{value.period_seconds}",
            f"window:{value.window_index}",
            document.summary or "已处理的影音时间段",
            encoded,
            value.started_at,
            value.ended_at,
        )
        stored = await self.get(value.fish_id, value.id)
        if stored is None:
            raise ValueError("snapshot Fish is no longer active")
        if stored.input_digest != value.input_digest:
            raise EnvironmentHistoryConflict("snapshot input identity changed")
        return stored

    async def revise_attribution(
        self,
        value: SemanticSnapshot,
        *,
        through_index: int,
        segments: list[dict[str, object]] | None = None,
        _attempt: int = 0,
    ) -> SemanticSnapshot | None:
        """Persist bounded label corrections; words and inferred stories stay fixed.

        A compare-and-swap plus a monotonically increasing capture index keeps
        late asynchronous writes from restoring an older speaker assignment.
        The original leaf attribution is retained once for diagnosis.
        """
        if value.period_seconds > 300 or not (
            value.window_index <= through_index <= value.window_index + 60
        ):
            raise ValueError("attribution revision outside its bounded capture range")
        row = await self._pool.fetchrow(
            "SELECT * FROM observations WHERE fish_id=$1 AND id=$2 AND source=$3",
            value.fish_id,
            value.id,
            HISTORY_SOURCE,
        )
        if row is None:
            return None
        current = _snapshot(row)
        if any(
            getattr(current, name) != getattr(value, name)
            for name in (
                "input_digest",
                "capture_session_id",
                "window_index",
                "period_seconds",
                "started_at",
                "ended_at",
                "source_ids",
            )
        ):
            raise EnvironmentHistoryConflict("attribution revision input changed")
        document = _redacted_document(value.document)
        if _attribution_content(current.document) != _attribution_content(document):
            raise EnvironmentHistoryConflict(
                "attribution revision changed words or story"
            )
        if current.document == document:
            return current
        payload = row["payload"]
        payload = json.loads(payload) if isinstance(payload, str) else dict(payload)
        original_payload = json.dumps(payload, ensure_ascii=False)
        evidence = dict(payload.get("evidence") or {})
        if int(evidence.get("speakerRevisionWindow", -1)) >= through_index:
            return current
        if value.period_seconds == 10 and "originalAttributionDigest" not in evidence:
            evidence["originalAttributionDigest"] = hashlib.sha256(
                current.document.audio.encode()
            ).hexdigest()
            evidence["originalAttribution"] = current.document.audio
        evidence["speakerRevisionWindow"] = through_index
        evidence["speakerAttributionRevised"] = True
        if segments is not None:
            evidence["audioSegments"] = segments
        payload = {**payload, "document": document.model_dump(), "evidence": evidence}
        try:
            encoded = _encode_snapshot_payload(payload)
        except ValueError:
            payload["evidence"].pop("originalAttribution", None)
            payload["evidence"]["originalAttributionOmitted"] = "payload_budget"
            encoded = _encode_snapshot_payload(payload)
        changed = await self._pool.execute(
            """UPDATE observations SET summary=$1,payload=$2::jsonb
               WHERE fish_id=$3 AND id=$4 AND source=$5 AND payload=$6::jsonb""",
            document.summary or "已处理的影音时间段",
            encoded,
            value.fish_id,
            value.id,
            HISTORY_SOURCE,
            original_payload,
        )
        if changed == "UPDATE 0":
            if _attempt >= 2:
                raise EnvironmentHistoryConflict("concurrent attribution revision")
            return await self.revise_attribution(
                value,
                through_index=through_index,
                segments=segments,
                _attempt=_attempt + 1,
            )
        return await self.get(value.fish_id, value.id)

    async def child_snapshots(
        self,
        fish_id: UUID,
        capture_session_id: UUID,
        first_index: int,
        *,
        child_period_seconds: int,
        parent_period_seconds: int,
    ) -> tuple[SemanticSnapshot, ...]:
        """Read one complete direct-child range without bridging gaps/sessions."""
        if (
            level_for_period(parent_period_seconds).child_period_seconds
            != child_period_seconds
        ):
            raise ValueError("snapshot periods do not form a direct hierarchy edge")
        if first_index < 0 or first_index % (parent_period_seconds // _INDEX_SECONDS):
            raise ValueError("first_index is not aligned to the parent period")

        child_count = parent_period_seconds // child_period_seconds
        index_step = child_period_seconds // _INDEX_SECONDS
        expected_indices = tuple(
            first_index + offset * index_step for offset in range(child_count)
        )
        child_type = observation_type_for_period(child_period_seconds)
        rows = await self._pool.fetch(
            f"""
            SELECT o.* FROM observations o
            WHERE o.fish_id=$1 AND o.source=$2 AND o.observation_type=$3
              AND o.scope_key=$4
              AND (o.payload->>'windowIndex')::bigint = ANY($5::bigint[])
              AND {_usable_snapshot_sql("o", child_period_seconds)}
            ORDER BY (o.payload->>'windowIndex')::bigint, o.sequence
            """,
            fish_id,
            HISTORY_SOURCE,
            child_type,
            f"capture:{capture_session_id}:{child_period_seconds}",
            expected_indices,
        )
        snapshots = tuple(_snapshot(row) for row in rows)
        if not _is_complete_child_range(
            snapshots,
            fish_id=fish_id,
            capture_session_id=capture_session_id,
            child_period_seconds=child_period_seconds,
            expected_indices=expected_indices,
        ):
            return ()
        return snapshots

    async def minute_windows(
        self, fish_id: UUID, capture_session_id: UUID, first_index: int
    ) -> tuple[SemanticSnapshot, ...]:
        level = parent_level_for_period(_INDEX_SECONDS)
        if level is None:
            return ()
        rows = await self._pool.fetch(
            f"""
            SELECT o.* FROM observations o
            WHERE o.fish_id=$1 AND o.source=$2 AND o.observation_type=$3
              AND o.scope_key=$4
              AND (o.payload->>'windowIndex')::bigint BETWEEN $5 AND $6
              AND {_usable_window_sql("o")}
            ORDER BY o.started_at
            """,
            fish_id,
            HISTORY_SOURCE,
            WINDOW_TYPE,
            f"capture:{capture_session_id}:10",
            first_index,
            first_index + level.child_count - 1,
        )
        return tuple(_snapshot(row) for row in rows)

    async def context_snapshots(
        self, fish_id: UUID
    ) -> tuple[SemanticSnapshot | None, tuple[SemanticSnapshot, ...]]:
        # One SQL statement sees a consistent snapshot while a merge completes.
        rows = await self._pool.fetch(
            f"""
            WITH minute AS (
                SELECT m.* FROM observations m
                WHERE m.fish_id=$1 AND m.source=$2 AND m.observation_type=$3
                  AND {_usable_minute_sql("m")}
                ORDER BY m.ended_at DESC, m.sequence DESC LIMIT 1
            ), tail AS (
                SELECT o.* FROM observations o
                WHERE o.fish_id=$1 AND o.source=$2 AND o.observation_type=$4
                  AND {_usable_window_sql("o")}
                  AND o.ended_at > COALESCE((SELECT ended_at FROM minute),'-infinity')
                ORDER BY o.ended_at DESC, o.sequence DESC LIMIT 12
            )
            SELECT * FROM minute UNION ALL SELECT * FROM tail
            """,
            fish_id,
            HISTORY_SOURCE,
            MINUTE_TYPE,
            WINDOW_TYPE,
        )
        values = tuple(_snapshot(row) for row in rows)
        minute = next((item for item in values if item.period_seconds == 60), None)
        tail = tuple(
            sorted(
                (item for item in values if item.period_seconds == 10),
                key=lambda item: item.started_at,
            )
        )
        return minute, tail

    async def recent_hour_snapshots(
        self,
        fish_id: UUID,
        *,
        since: datetime,
        through: datetime,
    ) -> tuple[SemanticSnapshot, ...]:
        """Return completed nodes, retaining the whole left-boundary parent."""
        _validate_aware_interval(since, through)
        level_predicates = " OR ".join(
            f"""(o.observation_type='{level.observation_type}'
                 AND (o.payload->>'periodSeconds')::integer={level.period_seconds}
                 AND {_usable_snapshot_sql("o", level.period_seconds)})"""
            for level in snapshot_levels()
        )
        rows = await self._pool.fetch(
            f"""
            SELECT o.* FROM observations o
            WHERE o.fish_id=$1 AND o.source=$2
              AND o.ended_at >= $3 AND o.ended_at <= $4
              AND ({level_predicates})
            ORDER BY o.started_at,
                     (o.payload->>'periodSeconds')::integer DESC,
                     o.sequence
            """,
            fish_id,
            HISTORY_SOURCE,
            since,
            through,
        )
        return tuple(_snapshot(row) for row in rows)

    async def delete_compacted_windows(
        self,
        fish_id: UUID,
        *,
        ended_before: datetime,
    ) -> int:
        """Delete old detail nodes only after their direct parent is durable."""
        _validate_aware_timestamp(ended_before, name="ended_before")
        edges = tuple(
            level
            for level in snapshot_levels()
            if level.child_period_seconds is not None
        )
        if not edges:
            return 0
        parent_predicates = " OR ".join(
            f"""(child.observation_type='{observation_type_for_period(level.child_period_seconds)}'
                 AND (child.payload->>'periodSeconds')::integer=
                     {level.child_period_seconds}
                 AND parent.observation_type='{level.observation_type}'
                 AND (parent.payload->>'periodSeconds')::integer={level.period_seconds}
                 AND jsonb_array_length(
                     CASE WHEN jsonb_typeof(parent.payload->'sourceSnapshotIds')='array'
                          THEN parent.payload->'sourceSnapshotIds' ELSE '[]'::jsonb END
                 )={level.child_count}
                 AND {_usable_snapshot_sql("parent", level.period_seconds)})"""
            for level in edges
            if level.child_period_seconds is not None
        )
        child_types = [
            observation_type_for_period(level.child_period_seconds)
            for level in edges
            if level.child_period_seconds is not None
        ]
        deleted = await self._pool.fetchval(
            f"""
            WITH candidates AS (
                SELECT child.id FROM observations child
                WHERE child.fish_id=$1 AND child.source=$2
                  AND child.observation_type = ANY($3::text[])
                  AND child.ended_at < $4
                  AND EXISTS (
                    SELECT 1 FROM observations parent
                    WHERE parent.fish_id=child.fish_id
                      AND parent.source=child.source
                      AND parent.payload->>'captureSessionId' =
                          child.payload->>'captureSessionId'
                      AND parent.started_at <= child.started_at
                      AND parent.ended_at >= child.ended_at
                      AND (parent.payload->'sourceSnapshotIds') ? child.id::text
                      AND ({parent_predicates})
                  )
                ORDER BY child.ended_at, child.sequence
                LIMIT 500
            ), deleted AS (
                DELETE FROM observations child
                USING candidates
                WHERE child.id=candidates.id
                RETURNING child.id
            )
            SELECT count(*) FROM deleted
            """,
            fish_id,
            HISTORY_SOURCE,
            child_types,
            ended_before,
        )
        return int(deleted or 0)

    async def pending_minute_windows(
        self,
        fish_id: UUID,
        limit: int = 4,
        *,
        since: datetime | None = None,
        through: datetime | None = None,
    ) -> tuple[SemanticSnapshot, ...]:
        """Return one semantic representative per complete, unmerged group."""
        if limit < 1:
            return ()
        if since is not None:
            _validate_aware_timestamp(since, name="since")
        if through is not None:
            _validate_aware_timestamp(through, name="through")
        if since is not None and through is not None and through < since:
            raise ValueError("through must not precede since")
        level = parent_level_for_period(_INDEX_SECONDS)
        if level is None:
            return ()
        child_count = level.child_count
        parent_period = level.period_seconds
        rows = await self._pool.fetch(
            f"""
            WITH windows AS (
                SELECT o.*,
                       (o.payload->>'windowIndex')::bigint AS window_index,
                       ((o.payload->>'windowIndex')::bigint / {child_count})
                           * {child_count} AS first_index
                FROM observations o
                WHERE o.fish_id=$1 AND o.source=$2 AND o.observation_type=$3
                  AND {_usable_window_sql("o")}
                  AND ($5::timestamptz IS NULL OR o.started_at >= $5)
                  AND ($6::timestamptz IS NULL OR o.ended_at <= $6)
                  AND NOT EXISTS (
                    SELECT 1 FROM observations m
                    WHERE m.fish_id=o.fish_id AND m.source=$2
                      AND m.observation_type=$4
                      AND {_usable_minute_sql("m")}
                      AND m.scope_key='capture:' ||
                          (o.payload->>'captureSessionId') || ':{parent_period}'
                      AND m.semantic_key='window:' ||
                          (((o.payload->>'windowIndex')::bigint / {child_count})
                           * {child_count})::text
                  )
            ), ordered AS (
                SELECT windows.*,
                       lag(ended_at) OVER (
                           PARTITION BY scope_key, first_index ORDER BY window_index
                       ) AS previous_end
                FROM windows
            ), complete AS (
                SELECT scope_key, first_index FROM ordered
                GROUP BY scope_key, first_index
                HAVING count(*)={child_count}
                   AND count(DISTINCT window_index)={child_count}
                   AND bool_and(abs(extract(epoch FROM ended_at-started_at)
                                    - {_INDEX_SECONDS})
                                <= 0.001)
                   AND bool_and(previous_end IS NULL OR
                                abs(extract(epoch FROM started_at-previous_end))
                                <= 0.001)
            )
            SELECT w.* FROM ordered w
            JOIN complete c USING(scope_key, first_index)
            WHERE w.window_index=w.first_index+{child_count - 1}
            ORDER BY w.ended_at, w.sequence LIMIT $7
            """,
            fish_id,
            HISTORY_SOURCE,
            WINDOW_TYPE,
            level.observation_type,
            since,
            through,
            min(limit, 4),
        )
        return tuple(_snapshot(row) for row in rows)


def _usable_snapshot_sql(alias: str, period_seconds: int) -> str:
    level = level_for_period(period_seconds)
    if level.child_period_seconds is None:
        return _usable_window_sql(alias)
    if level_for_period(level.child_period_seconds).child_period_seconds is None:
        return _usable_minute_sql(alias)
    return "TRUE"


def _is_complete_child_range(
    snapshots: tuple[SemanticSnapshot, ...],
    *,
    fish_id: UUID,
    capture_session_id: UUID,
    child_period_seconds: int,
    expected_indices: tuple[int, ...],
) -> bool:
    if len(snapshots) != len(expected_indices):
        return False
    previous_end: datetime | None = None
    for snapshot, expected_index in zip(snapshots, expected_indices, strict=True):
        if (
            snapshot.fish_id != fish_id
            or snapshot.capture_session_id != capture_session_id
            or snapshot.period_seconds != child_period_seconds
            or snapshot.window_index != expected_index
        ):
            return False
        duration = (snapshot.ended_at - snapshot.started_at).total_seconds()
        if abs(duration - child_period_seconds) > _TIMESTAMP_TOLERANCE_SECONDS:
            return False
        if previous_end is not None:
            gap = (snapshot.started_at - previous_end).total_seconds()
            if abs(gap) > _TIMESTAMP_TOLERANCE_SECONDS:
                return False
        previous_end = snapshot.ended_at
    return True


def _validate_aware_timestamp(value: datetime, *, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _validate_aware_interval(since: datetime, through: datetime) -> None:
    _validate_aware_timestamp(since, name="since")
    _validate_aware_timestamp(through, name="through")
    if through < since:
        raise ValueError("through must not precede since")


def _attribution_content(document: SemanticDocument) -> dict[str, object]:
    payload = document.model_dump()
    payload["audio"] = system_audio_words(document.audio)
    if document.audiovisual_summary == f"听到的话：\n{document.audio}":
        payload["audiovisualSummary"] = f"听到的话：{payload['audio']}"
    if document.narrative_sources:
        payload["narrativeSources"] = {
            key: words
            if key.endswith((":media", ":summary"))
            else system_audio_words(words)
            for key, words in document.narrative_sources.items()
        }
    return payload


def _redacted_document(document: SemanticDocument) -> SemanticDocument:
    """Redact language fields without converting versions or provenance IDs."""
    payload = document.model_dump()
    for name in (
        "audiovisualSummary",
        "activitySummary",
        "summary",
        "audio",
        "visualSummary",
    ):
        value = payload.get(name)
        if isinstance(value, str):
            payload[name] = redact_sensitive_observation_text(value)
    events = payload.get("narrativeEvents", [])
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            for name in ("subject", "action", "object", "narration"):
                value = event.get(name)
                if isinstance(value, str):
                    event[name] = redact_sensitive_observation_text(value)
            for evidence in event.get("evidence", []):
                if isinstance(evidence, dict) and isinstance(
                    evidence.get("quote"), str
                ):
                    evidence["quote"] = redact_sensitive_observation_text(
                        evidence["quote"]
                    )
    paragraphs = payload.get("narrativeParagraphs", [])
    if isinstance(paragraphs, list):
        for paragraph in paragraphs:
            if isinstance(paragraph, dict) and isinstance(paragraph.get("text"), str):
                paragraph["text"] = redact_sensitive_observation_text(paragraph["text"])
    sources = payload.get("narrativeSources", {})
    if isinstance(sources, dict):
        for key, words in sources.items():
            if isinstance(words, str):
                sources[key] = redact_sensitive_observation_text(words)
    return SemanticDocument.model_validate(payload)


def _encode_snapshot_payload(payload: dict[str, Any]) -> str:
    """Optional per-turn scores must never evict an otherwise valid narrative."""

    def encode() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    encoded = encode()
    if len(encoded.encode()) <= 32_768:
        return encoded
    evidence = payload.get("evidence", {})
    segments = evidence.get("audioSegments")
    if isinstance(segments, list):
        evidence = dict(evidence)
        payload["evidence"] = evidence
        evidence["speakerDiagnosticsOmitted"] = "payload_budget"
        evidence["audioSegments"] = [
            {
                key: value
                for key, value in segment.items()
                if key
                in {
                    "startSeconds",
                    "endSeconds",
                    "speakerId",
                    "overlap",
                    "reason",
                    "trackId",
                    "trackReason",
                    "alignmentReason",
                }
            }
            for segment in segments
            if isinstance(segment, dict)
        ]
        encoded = encode()
        if len(encoded.encode()) <= 32_768:
            return encoded
        # The same time ranges and voice labels are already in document.audio.
        # Keep its full text and explicit omission metadata over repeated debug
        # intervals. Other evidence and the document remain unchanged.
        evidence.pop("audioSegments")
        evidence["audioSegmentCount"] = len(segments)
        encoded = encode()
    if len(encoded.encode()) > 32_768:
        raise ValueError("semantic snapshot exceeds Observation payload limit")
    return encoded


def _snapshot(row: Mapping[str, Any]) -> SemanticSnapshot:
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    document_payload = payload["document"]
    if isinstance(document_payload, Mapping):
        # Older rows retain richer presentation/diagnostic fields and a sibling
        # documentMarkdown payload. Only the compact semantic contract is read.
        document_payload = dict(document_payload)
        for retired_field in ("visual", "uncertainties"):
            document_payload.pop(retired_field, None)
    return SemanticSnapshot(
        id=UUID(str(row["id"])),
        fish_id=UUID(str(row["fish_id"])),
        capture_session_id=UUID(payload["captureSessionId"]),
        window_index=int(payload["windowIndex"]),
        period_seconds=int(payload["periodSeconds"]),
        started_at=row["started_at"],
        ended_at=row["ended_at"],
        created_at=row["created_at"],
        document=SemanticDocument.model_validate(document_payload),
        input_digest=payload["inputDigest"],
        source_ids=tuple(UUID(value) for value in payload["sourceSnapshotIds"]),
        session_id=UUID(payload["session_id"]) if payload.get("session_id") else None,
        audio_unavailable=(
            isinstance(payload.get("evidence"), Mapping)
            and payload["evidence"].get("transcriptionTextStatus") == "unavailable"
        ),
    )
