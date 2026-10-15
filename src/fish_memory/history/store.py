from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from fish_memory.answer.context import INTERRUPTION_MARKER
from fish_memory.extensions.event_context import event_data_view
from fish_memory.extensions.models import EventDefinition
from fish_memory.history.models import HistoryChunk, HistoryUnit, chunk_bounds
from fish_memory.observations.environment_history_models import SemanticDocument


def _json(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value or {})


class SessionHistoryStore:
    """Materialize source changes; original Messages/Observations remain untouched.

    Units beneath a summary stay here until their chunk expires. That permits a
    late source correction to invalidate a summary without losing other units.
    Model calls never hold a database lock.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool
        self.on_unit: Callable[[UUID, UUID, HistoryUnit, datetime], None] | None = None

    async def context_revision(self, fish: UUID, session: UUID) -> int | None:
        return await self.pool.fetchval(
            "SELECT p.revision FROM session_context_progress p "
            "JOIN fish_sessions s ON s.id=p.session_id "
            "WHERE s.fish_id=$1 AND s.id=$2",
            fish,
            session,
        )

    async def has_thought_since_last_speech(
        self, fish: UUID, session: UUID, now: datetime
    ) -> bool:
        """Compare thought origins, never their later model completion times.

        Legacy thoughts have no start marker and cannot establish this ordering.
        Durable sources preserve it across projection delays and compaction.
        """
        return bool(
            await self.pool.fetchval(
                """WITH last_speech AS (
                       SELECT created_at FROM messages
                       WHERE fish_id=$1 AND session_id=$2 AND role='assistant'
                         AND status IN ('completed','interrupted')
                         AND fact_extraction_status<>'waiting_consumption'
                         AND created_at<=$3
                         AND (CASE WHEN consumed_offset IS NULL OR consumed_offset<0
                              THEN content ELSE left(content,consumed_offset) END)
                             ~ '[^[:space:]]'
                       ORDER BY created_at DESC,sequence DESC LIMIT 1
                   )
                   SELECT EXISTS (
                       SELECT 1 FROM observations o
                       JOIN fish_sessions s ON s.id=$2 AND s.fish_id=$1
                       WHERE o.fish_id=$1 AND o.source='experimental'
                         AND o.observation_type='experimental.thought'
                         AND o.payload->>'session_id'=($2::uuid)::text
                         AND o.payload->'include_in_context'
                             IS DISTINCT FROM 'false'::jsonb
                         AND COALESCE(o.payload->>'thought_started_at','')<>''
                         AND o.started_at<=$3
                         AND o.started_at>COALESCE(
                             (SELECT created_at FROM last_speech),
                             '-infinity'::timestamptz)
                   )""",
                fish,
                session,
                now,
            )
        )

    async def pending_sessions(self) -> list[tuple[UUID, UUID]]:
        rows = await self.pool.fetch(
            "SELECT fish_id,session_id FROM session_history_pending "
            "GROUP BY fish_id,session_id ORDER BY min(queued_at) LIMIT 16"
        )
        return [(r["fish_id"], r["session_id"]) for r in rows]

    async def _lock(self, connection: asyncpg.Connection, session: UUID) -> None:
        await connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
            f"session-history:{session}",
        )

    async def _initialize(
        self, connection: asyncpg.Connection, fish: UUID, session: UUID
    ) -> None:
        exists = await connection.fetchval(
            "SELECT id FROM fish_sessions WHERE id=$1 AND fish_id=$2", session, fish
        )
        if exists is None:
            raise LookupError("history session does not belong to this pet")
        await connection.execute(
            "INSERT INTO session_history_contexts(session_id,fish_id) VALUES($1,$2) "
            "ON CONFLICT DO NOTHING",
            session,
            fish,
        )
        if await connection.fetchval(
            "SELECT initialized FROM session_history_contexts WHERE session_id=$1",
            session,
        ):
            return
        await connection.execute(
            """INSERT INTO session_history_pending
               (source_kind,source_id,fish_id,session_id)
               SELECT 'message',id,fish_id,session_id FROM messages WHERE session_id=$1
               ON CONFLICT DO NOTHING""",
            session,
        )
        await connection.execute(
            """INSERT INTO session_history_pending
               (source_kind,source_id,fish_id,session_id)
               SELECT 'observation',id,fish_id,$1 FROM observations
               WHERE fish_id=$2 AND (
                   kind='event' AND observation_type<>'plugin.definition'
                   OR source='environment_history'
                      AND observation_type IN (
                          'semantic_snapshot_5s','semantic_snapshot_10s')
               ) AND history_observation_session(fish_id,started_at,payload)=$1
               AND payload->'include_in_context' IS DISTINCT FROM 'false'::jsonb
               ON CONFLICT DO NOTHING""",
            session,
            fish,
        )
        await connection.execute(
            "UPDATE session_history_contexts SET initialized=true WHERE session_id=$1",
            session,
        )

    async def flush(self, fish: UUID, session: UUID) -> tuple[datetime, ...]:
        boundaries: list[datetime] = []
        while True:
            materialized: list[tuple[HistoryUnit, datetime]] = []
            async with self.pool.acquire() as connection, connection.transaction():
                await self._lock(connection, session)
                await self._initialize(connection, fish, session)
                pending = await connection.fetch(
                    """SELECT * FROM session_history_pending
                       WHERE fish_id=$1 AND session_id=$2 ORDER BY queued_at
                       LIMIT 256 FOR UPDATE""",
                    fish,
                    session,
                )
                for item in pending:
                    kind, identifier = item["source_kind"], item["source_id"]
                    unit = await self._project(connection, kind, identifier)
                    changed = await self._upsert(
                        connection, fish, session, kind, identifier, unit
                    )
                    if (
                        changed
                        and unit is not None
                        and unit.label != "as"
                        and unit.interrupts_observation
                    ):
                        boundaries.append(unit.started_at)
                    if changed and unit is not None:
                        materialized.append((unit, item["queued_at"]))
                    await connection.execute(
                        "DELETE FROM session_history_pending "
                        "WHERE source_kind=$1 AND source_id=$2",
                        kind,
                        identifier,
                    )
            if self.on_unit is not None:
                for unit, received_at in materialized:
                    self.on_unit(fish, session, unit, received_at)
            if len(pending) < 256:
                return tuple(boundaries)

    async def _project(
        self, connection: asyncpg.Connection, kind: str, identifier: UUID
    ) -> HistoryUnit | None:
        if kind == "message":
            row = await connection.fetchrow(
                "SELECT * FROM messages WHERE id=$1", identifier
            )
            if row is None or (
                row["role"] != "user"
                and (
                    row["status"] not in ("completed", "interrupted")
                    or row["fact_extraction_status"] == "waiting_consumption"
                )
            ):
                return None
            content = row["content"]
            offset = row["consumed_offset"]
            if row["role"] != "user" and offset is not None and offset >= 0:
                content = content[:offset]
            if not content.strip():
                return None
            if row["role"] != "user" and (
                row["status"] == "interrupted"
                or (offset is not None and 0 <= offset < len(row["content"]))
            ):
                content += "\n" + INTERRUPTION_MARKER
            return HistoryUnit(
                identifier,
                "u" if row["role"] == "user" else "a",
                content,
                row["created_at"],
                row["created_at"],
                row["sequence"],
            )
        row = await connection.fetchrow(
            "SELECT * FROM observations WHERE id=$1", identifier
        )
        if row is None:
            return None
        payload = _json(row["payload"])
        if payload.get("include_in_context") is False:
            return None
        if row["source"] == "environment_history":
            document = SemanticDocument.model_validate(payload.get("document", {}))
            content = document.history_summary
            if not content:
                return None
            label = "as"
        elif row["observation_type"] == "plugin.event" and not payload.get(
            "context_text"
        ):
            data = payload.get("data", {})
            definition_id = payload.get("definition_id")
            if definition_id:
                definition = await connection.fetchval(
                    "SELECT payload FROM observations WHERE id=$1 AND fish_id=$2",
                    UUID(definition_id),
                    row["fish_id"],
                )
                if definition:
                    fields = _json(definition)
                    fields = {
                        key: value
                        for key, value in fields.items()
                        if key not in {"kind", "plugin_id", "revision"}
                    }
                    data = event_data_view(data, EventDefinition.model_validate(fields))
            content = f"{payload.get('type', row['summary'])}: " + json.dumps(
                data, ensure_ascii=False, separators=(",", ":")
            )
            label = "event"
        else:
            content, label = payload.get("context_text") or row["summary"], "event"
        if not content.strip():
            return None
        return HistoryUnit(
            identifier,
            label,
            content,
            row["started_at"],
            row["ended_at"],
            row["sequence"],
            # Idle material reaches speech only through TA's own thoughts: as a
            # new event it woke TA, who answered it like something just said
            # ("哇！想起来啦").
            triggers_decision=payload.get("wakes") is not False
            and not (
                row["source"] == "experimental"
                and row["observation_type"]
                in {
                    "experimental.started",
                    "experimental.thought",
                    "experimental.material",
                }
            ),
            # TA's own inner material does not split what TA is watching.
            interrupts_observation=row["observation_type"]
            not in {"experimental.thought", "experimental.material"},
            reports_work=payload.get("reports_work") is True
            or (
                row["source"] == "experimental"
                and row["observation_type"] == "experimental.result"
            ),
        )

    async def _upsert(
        self,
        connection: asyncpg.Connection,
        fish: UUID,
        session: UUID,
        kind: str,
        identifier: UUID,
        unit: HistoryUnit | None,
    ) -> bool:
        existing = await connection.fetchrow(
            "SELECT * FROM session_history_units WHERE source_kind=$1 AND source_id=$2",
            kind,
            identifier,
        )
        if (
            unit is not None
            and existing is not None
            and (
                existing["content"],
                existing["label"],
                existing["started_at"],
                existing["ended_at"],
            )
            == (unit.content, unit.label, unit.started_at, unit.ended_at)
        ):
            return False
        if existing is not None:
            await connection.execute(
                "DELETE FROM session_history_units "
                "WHERE source_kind=$1 AND source_id=$2",
                kind,
                identifier,
            )
            await connection.execute(
                "UPDATE session_history_chunks SET summary='',revision=revision+1, "
                "content_revision=content_revision+1 WHERE id=$1",
                existing["chunk_id"],
            )
            await connection.execute(
                "UPDATE session_history_units SET summarized=false WHERE chunk_id=$1",
                existing["chunk_id"],
            )
        if unit is not None:
            chunk_id = await connection.fetchval(
                """SELECT id FROM session_history_chunks WHERE session_id=$1
                   AND started_at<=$2 AND ended_at>$2 ORDER BY started_at LIMIT 1""",
                session,
                unit.started_at,
            )
            if chunk_id is None:
                start, end = chunk_bounds(unit.started_at)
                chunk_id = uuid4()
                await connection.execute(
                    """INSERT INTO session_history_chunks
                       (id,session_id,fish_id,started_at,ended_at)
                       VALUES($1,$2,$3,$4,$5)""",
                    chunk_id,
                    session,
                    fish,
                    start,
                    end,
                )
            else:
                later = await connection.fetchrow(
                    """SELECT count(*)>0 AS has_units,
                              COALESCE(bool_or(summarized),false) AS has_summary
                       FROM session_history_units WHERE chunk_id=$1
                         AND (started_at,source_sequence,source_id)>($2,$3,$4)""",
                    chunk_id,
                    unit.started_at,
                    unit.sequence,
                    unit.source_id,
                )
                if later is not None and later["has_units"]:
                    # A late insertion is not a chronological append: a pending
                    # compaction must reread it. Restore retained sources if the
                    # existing summary would otherwise precede this older unit.
                    await connection.execute(
                        "UPDATE session_history_chunks "
                        "SET content_revision=content_revision+1, "
                        "summary=CASE WHEN $2 THEN '' ELSE summary END WHERE id=$1",
                        chunk_id,
                        later["has_summary"],
                    )
                    if later["has_summary"]:
                        await connection.execute(
                            "UPDATE session_history_units SET summarized=false "
                            "WHERE chunk_id=$1",
                            chunk_id,
                        )
            revision = await connection.fetchval(
                "UPDATE session_history_chunks SET revision=revision+1 "
                "WHERE id=$1 RETURNING revision",
                chunk_id,
            )
            await connection.execute(
                """INSERT INTO session_history_units
                   (source_kind,source_id,chunk_id,label,content,started_at,
                    ended_at,source_sequence,insertion_revision)
                   VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                kind,
                identifier,
                chunk_id,
                unit.label,
                unit.content,
                unit.started_at,
                unit.ended_at,
                unit.sequence,
                revision,
            )
        await connection.execute(
            """DELETE FROM session_history_chunks c WHERE session_id=$1
               AND NOT EXISTS(
                   SELECT 1 FROM session_history_units u WHERE u.chunk_id=c.id)""",
            session,
        )
        return True

    async def read(self, fish: UUID, session: UUID) -> tuple[HistoryChunk, ...]:
        async with (
            self.pool.acquire() as connection,
            connection.transaction(isolation="repeatable_read"),
        ):
            rows = await connection.fetch(
                """SELECT c.*,count(u.source_id) AS unit_count
                   FROM session_history_chunks c
                   LEFT JOIN session_history_units u ON u.chunk_id=c.id
                   WHERE c.session_id=$1 AND c.fish_id=$2
                   GROUP BY c.id ORDER BY c.started_at,c.id""",
                session,
                fish,
            )
            units = await connection.fetch(
                """SELECT u.* FROM session_history_units u
                   JOIN session_history_chunks c ON c.id=u.chunk_id
                   WHERE c.session_id=$1 AND c.fish_id=$2 AND NOT u.summarized
                   ORDER BY u.started_at,u.source_sequence,u.source_id""",
                session,
                fish,
            )
        grouped: dict[UUID, list[HistoryUnit]] = {}
        for unit in units:
            grouped.setdefault(unit["chunk_id"], []).append(_unit(unit))
        return tuple(
            HistoryChunk(
                row["id"],
                row["started_at"],
                row["ended_at"],
                row["revision"],
                row["summary"],
                tuple(grouped.get(row["id"], ())),
                row["unit_count"],
                row["content_revision"],
            )
            for row in rows
        )

    async def expire(self, fish: UUID, session: UUID, before: datetime) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._lock(connection, session)
            deleted = await connection.execute(
                "DELETE FROM session_history_chunks "
                "WHERE fish_id=$1 AND session_id=$2 AND ended_at<$3",
                fish,
                session,
                before,
            )
            if deleted != "DELETE 0":
                await connection.execute(
                    "SELECT advance_history_context($1,$2)", fish, session
                )

    async def replace_prefix(
        self,
        fish: UUID,
        session: UUID,
        selected: tuple[HistoryChunk, ...],
        summary: str,
    ) -> bool:
        if not selected:
            return False
        ids = [chunk.id for chunk in selected]
        async with self.pool.acquire() as connection, connection.transaction():
            await self._lock(connection, session)
            rows = await connection.fetch(
                """SELECT id,revision,content_revision FROM session_history_chunks
                   WHERE fish_id=$1 AND session_id=$2
                     AND started_at<$3 FOR UPDATE""",
                fish,
                session,
                max(c.ended_at for c in selected),
            )
            if {row["id"]: row["content_revision"] for row in rows} != {
                c.id: c.content_revision for c in selected
            }:
                return False
            # Only appends to the last selected chunk can follow the combined
            # summary. Appends to earlier chunks belong inside its chronology.
            earlier = {c.id: c.revision for c in selected[:-1]}
            if any(
                row["id"] in earlier and row["revision"] != earlier[row["id"]]
                for row in rows
            ):
                return False
            identifier = uuid4()
            await connection.execute(
                """INSERT INTO session_history_chunks
                   (id,fish_id,session_id,started_at,ended_at,summary)
                   VALUES($1,$2,$3,$4,$5,$6)""",
                identifier,
                fish,
                session,
                selected[0].started_at,
                max(c.ended_at for c in selected),
                summary,
            )
            for chunk in selected:
                await connection.execute(
                    "UPDATE session_history_units SET summarized=true "
                    "WHERE chunk_id=$1 AND insertion_revision<=$2",
                    chunk.id,
                    chunk.revision,
                )
            await connection.execute(
                "UPDATE session_history_units SET chunk_id=$1,insertion_revision=0 "
                "WHERE chunk_id=ANY($2::uuid[])",
                identifier,
                ids,
            )
            await connection.execute(
                "DELETE FROM session_history_chunks WHERE id=ANY($1::uuid[])", ids
            )
            # Last write: do not hold the progress fence while materializing
            # chunks, whose FK locks can contend with source observation writes.
            await connection.execute(
                "SELECT advance_history_context($1,$2)", fish, session
            )
        return True


def _unit(row: Mapping[str, Any]) -> HistoryUnit:
    return HistoryUnit(
        row["source_id"],
        row["label"],
        row["content"],
        row["started_at"],
        row["ended_at"],
        row["source_sequence"],
    )
