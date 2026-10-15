"""Plugin definitions/events use the existing observations table."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import asyncpg

from fish_memory.extensions.event_context import event_history_text
from fish_memory.extensions.models import EventDefinition, ExtensionError, Instance


class ExtensionStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def receive_host_event(
        self,
        fish_id: UUID,
        session_id: UUID,
        event_id: UUID,
        event_type: str,
        data: dict[str, Any],
        *,
        wakes: bool = True,
        reports_work: bool = False,
    ) -> None:
        """Keep a host event in TA's context. Unless `wakes` is False it wakes
        TA while 主动陪伴 is on; with `reports_work` TA answers it even while
        主动陪伴 is off, like the result of work TA started."""
        identifier = uuid5(NAMESPACE_URL, f"fish-host:{fish_id}:{event_id}")
        fields = {
            key: value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False)
            for key, value in data.items()
        }
        text = (
            event_type
            + "\n"
            + "\n".join(f"{key}：{value}" for key, value in fields.items())
        )
        await self.pool.execute(
            """INSERT INTO observations
            (id,fish_id,kind,source,observation_type,semantic_key,
             summary,payload,started_at,ended_at)
            VALUES($1,$2,'event','host','host.event',$3,$3,$4::jsonb,now(),now())
            ON CONFLICT(id) DO NOTHING""",
            identifier,
            fish_id,
            event_type,
            json.dumps(
                {
                    "session_id": str(session_id),
                    "event_id": str(event_id),
                    "context_text": text,
                    **({} if wakes else {"wakes": False}),
                    **({"reports_work": True} if reports_work else {}),
                },
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def observation_id(instance: Instance, event_id: UUID) -> UUID:
        return uuid5(
            NAMESPACE_URL, f"fish-plugin:{instance.fish_id}:{instance.id}:{event_id}"
        )

    @staticmethod
    def definition_id(instance: Instance, definition: EventDefinition) -> UUID:
        return uuid5(
            NAMESPACE_URL,
            f"fish-plugin-definition:{instance.fish_id}:{instance.plugin_id}:{definition.revision}",
        )

    async def definition(self, instance: Instance, definition: EventDefinition) -> UUID:
        identifier = self.definition_id(instance, definition)
        payload = {
            "kind": "plugin_definition",
            "plugin_id": instance.plugin_id,
            "revision": definition.revision,
            **definition.model_dump(by_alias=True),
        }
        await self.pool.execute(
            """INSERT INTO observations
            (id, fish_id, kind, source, observation_type,
             semantic_key, summary, payload, started_at, ended_at)
            VALUES ($1,$2,'event','plugin','plugin.definition',
                    $3,$3,$4::jsonb,now(),now())
            ON CONFLICT (id) DO NOTHING""",
            identifier,
            instance.fish_id,
            definition.type,
            json.dumps(payload),
        )
        return identifier

    async def receive(
        self,
        instance: Instance,
        event_id: UUID,
        definition: EventDefinition,
        data: dict[str, Any],
        disposition: str,
        reason: str | None,
    ) -> tuple[dict[str, Any], bool]:
        identifier = self.observation_id(instance, event_id)
        definition_id = self.definition_id(instance, definition)
        payload = {
            "kind": "plugin_event",
            "instance_id": str(instance.id),
            "run_id": str(instance.generation),
            "plugin_id": instance.plugin_id,
            "session_id": str(instance.session_id),
            "event_id": str(event_id),
            "type": definition.type,
            "definition_id": str(definition_id),
            "definition_revision": definition.revision,
            "include_in_context": definition.include_in_context,
            "data": data,
            "disposition": disposition,
            "reason": reason,
            "status": (
                "pending"
                if disposition == "responding"
                else "queued"
                if disposition == "queued"
                else "stored_only"
            ),
        }
        if (text := event_history_text(data, definition)) is not None:
            payload["context_text"] = text
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                # Session ownership is checked on every write, including after deletion.
                exists = await connection.fetchval(
                    """SELECT s.id FROM fish_sessions s JOIN fishes f ON f.id=s.fish_id
                    WHERE s.id=$1 AND s.fish_id=$2 AND f.deleted_at IS NULL
                    FOR KEY SHARE OF s, f""",
                    instance.session_id,
                    instance.fish_id,
                )
                if exists is None:
                    raise ExtensionError(
                        "not_found", "Pet session is no longer available"
                    )
                row = await connection.fetchrow(
                    """INSERT INTO observations
                    (id,fish_id,kind,source,observation_type,semantic_key,summary,payload,started_at,ended_at)
                    VALUES ($1,$2,'event','plugin','plugin.event',
                            $3,$3,$4::jsonb,now(),now())
                    ON CONFLICT (id) DO NOTHING RETURNING payload""",
                    identifier,
                    instance.fish_id,
                    definition.type,
                    json.dumps(payload),
                )
                created = row is not None
                if row is None:
                    row = await connection.fetchrow(
                        "SELECT payload FROM observations WHERE id=$1 AND fish_id=$2",
                        identifier,
                        instance.fish_id,
                    )
                saved = _payload(row["payload"])
                if saved.get("type") != definition.type or saved.get("data") != data:
                    raise ExtensionError(
                        "id_conflict", "event_id already has different data"
                    )
        return {**saved, "observation_id": str(identifier)}, created

    async def update(self, observation_id: UUID, **changes: Any) -> dict[str, Any]:
        row = await self.pool.fetchrow(
            """UPDATE observations SET payload=payload || $2::jsonb
            WHERE id=$1 AND source='plugin' AND observation_type='plugin.event'
              AND NOT (payload->>'status' IN ('completed','interrupted')
                AND COALESCE($2::jsonb->>'status','')
                    NOT IN ('completed','interrupted'))
            RETURNING payload""",
            observation_id,
            json.dumps(changes),
        )
        return _payload(row["payload"]) if row else {}

    async def result(self, instance: Instance, event_id: UUID) -> dict[str, Any]:
        row = await self.pool.fetchrow(
            "SELECT payload FROM observations "
            "WHERE id=$1 AND fish_id=$2 AND source='plugin'",
            self.observation_id(instance, event_id),
            instance.fish_id,
        )
        if row is None:
            raise ExtensionError("not_found", "Event not found")
        return _payload(row["payload"])

    async def update_many(self, observation_ids: list[UUID], **changes: Any) -> None:
        await self.pool.execute(
            """UPDATE observations SET payload=payload || $2::jsonb
            WHERE id=ANY($1::uuid[]) AND source='plugin'
              AND observation_type='plugin.event'
              AND NOT (payload->>'status' IN ('completed','interrupted')
                AND COALESCE($2::jsonb->>'status','')
                    NOT IN ('completed','interrupted'))""",
            observation_ids,
            json.dumps(changes),
        )

    async def recent(
        self, fish_id: UUID, session_id: UUID, limit: int = 20
    ) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            """SELECT payload,started_at FROM observations
            WHERE fish_id=$1 AND source='plugin' AND observation_type='plugin.event'
              AND payload->>'session_id'=$2
              AND COALESCE(payload->>'include_in_context','true') <> 'false'
            ORDER BY sequence DESC LIMIT $3""",
            fish_id,
            str(session_id),
            limit,
        )
        return [
            {
                **_payload(row["payload"]),
                "received_at": row["started_at"].isoformat(timespec="seconds"),
            }
            for row in reversed(rows)
        ]

    async def recover(self) -> None:
        await self.pool.execute(
            """UPDATE observations SET payload=payload ||
              '{"status":"interrupted","reason":"host_restarted"}'::jsonb
            WHERE source='plugin' AND observation_type='plugin.event'
              AND payload->>'status' IN
                  ('queued','pending','generating','playing','generated','cancelling')"""
        )

    async def record_reasoning(
        self, instance: Instance, identifier: UUID, **record: Any
    ) -> None:
        payload = {
            "kind": "plugin_reasoning",
            "plugin_id": instance.plugin_id,
            "instance_id": str(instance.id),
            "run_id": str(instance.generation),
            "session_id": str(instance.session_id),
            **record,
        }
        await self.pool.execute(
            """INSERT INTO observations
            (id,fish_id,kind,source,observation_type,semantic_key,summary,payload,started_at,ended_at)
            VALUES ($1,$2,'event','plugin','plugin.reasoning','plugin.reasoning',
              '插件辅助推理',$3::jsonb,now(),now())""",
            identifier,
            instance.fish_id,
            json.dumps(payload, ensure_ascii=False),
        )

    async def review_window(
        self,
        instance_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        run_id: UUID,
        after: datetime,
        through: datetime,
    ) -> dict[str, Any]:
        """Bounded evidence from this run; generation text is never spoken evidence."""
        rows = await self.pool.fetch(
            """WITH selected AS (
                SELECT payload, started_at, sequence, count(*) OVER() AS total
                FROM observations
                WHERE fish_id=$1 AND source='plugin'
                  AND observation_type='plugin.event'
                  AND payload->>'session_id'=$2
                  AND payload->>'instance_id'=$3 AND payload->>'run_id'=$4
                  AND COALESCE((payload->>'finished_at')::timestamptz,
                    CASE WHEN payload->>'status' IN ('failed','silent')
                      THEN (payload->>'generated_at')::timestamptz END, started_at)>$5
                  AND COALESCE((payload->>'finished_at')::timestamptz,
                    CASE WHEN payload->>'status' IN ('failed','silent')
                      THEN (payload->>'generated_at')::timestamptz END, started_at)<=$6
                ORDER BY sequence DESC LIMIT 12
            )
            SELECT selected.*, p.input_payload->>'model' AS model,
              p.input_payload->'messages'->-1->>'content' AS input_context,
              p.input_payload->'messages'->-1->'environmentImage' AS screen_frame,
              p.input_payload->'extensionContext' AS context_audit,
              p.duration_us AS generation_us,
              d.payload->'schema'->>'description' AS event_description,
              d.payload->'schema' AS event_schema
            FROM selected LEFT JOIN observations d
              ON d.id=NULLIF(selected.payload->>'definition_id','')::uuid
            LEFT JOIN LATERAL (
                SELECT input_payload, duration_us FROM message_performance_steps
                WHERE message_id=NULLIF(selected.payload->>'message_id','')::uuid
                  AND step_name='llm.answer.plugin'
                ORDER BY id DESC LIMIT 1
            ) p ON true ORDER BY selected.sequence""",
            fish_id,
            str(session_id),
            str(instance_id),
            str(run_id),
            after,
            through,
        )
        events = []
        definitions: dict[str, dict[str, Any]] = {}
        for row in rows:
            value = _payload(row["payload"])
            revision = value.get("definition_revision", "")
            if revision and revision not in definitions and len(definitions) < 8:
                definitions[revision] = {
                    "type": value["type"],
                    "revision": revision,
                    "description": str(row["event_description"] or "")[:1500],
                }
            perception = None
            try:
                captured = json.loads(row["input_context"] or "{}")
                perception = captured.get("current_context", {}).get("perception")
            except (ValueError, AttributeError):
                pass
            audit = _payload(row["context_audit"]) if row["context_audit"] else None
            if audit and audit.get("context_version", 0) >= 5:
                perception = audit.get("perception")
            references = event_references(
                value.get("data", {}),
                _payload(row["event_schema"]) if row["event_schema"] else {},
            )
            events.append(
                {
                    **value,
                    "references": references,
                    "received_at": row["started_at"].isoformat(timespec="milliseconds"),
                    "model": row["model"],
                    "generation_us": row["generation_us"],
                    "perception": perception,
                    "screen_frame": _payload(row["screen_frame"])
                    if row["screen_frame"]
                    else None,
                    "context_audit": _payload(row["context_audit"])
                    if row["context_audit"]
                    else None,
                }
            )
        conversation = await self.pool.fetch(
            """SELECT m.id,m.role,m.content,m.status,m.consumed_offset,m.created_at,
                u.id AS reply_to_message_id
            FROM messages m
            LEFT JOIN background_assistant_deliveries d ON d.message_id=m.id
            LEFT JOIN messages u ON u.sequence=m.context_through_sequence
              AND u.role='user' AND u.fish_id=m.fish_id AND u.session_id=m.session_id
            WHERE m.fish_id=$1 AND m.session_id=$2 AND m.role IN ('user','assistant')
              AND d.source IS DISTINCT FROM 'plugin'
              AND COALESCE(m.finished_at,m.created_at)>$3
              AND COALESCE(m.finished_at,m.created_at)<=$4
            ORDER BY m.sequence DESC LIMIT 20""",
            fish_id,
            session_id,
            after,
            through,
        )
        return {
            "events": events,
            "definitions": list(definitions.values()),
            "total_events": rows[0]["total"] if rows else 0,
            "conversation_messages": [
                {
                    "message_id": str(r["id"]),
                    "role": r["role"],
                    "text": r["content"],
                    "status": r["status"],
                    "consumed_offset": r["consumed_offset"],
                    "reply_to_message_id": str(r["reply_to_message_id"])
                    if r["reply_to_message_id"]
                    else None,
                    "at": r["created_at"].isoformat(timespec="seconds"),
                }
                for r in reversed(conversation)
            ],
        }


def _payload(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value)


def event_references(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Declared reference fields inform review, never speech assembly or scoring."""
    references = {}
    for key, definition in schema.get("properties", {}).items():
        if (
            isinstance(definition, dict)
            and definition.get("x-fish-evidence") == "reference"
            and key in data
        ):
            references[key] = data[key]
    if len(json.dumps(references, ensure_ascii=False).encode()) > 8192:
        return {"omitted": "Reference fields exceed review budget"}
    return references
