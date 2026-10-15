"""Instance-scoped admission to the existing Task/Run tables."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from fish_memory.agent.models import AgentRunWorkItem
from fish_memory.db.agent_tasks import _WORK_COLUMNS, _work_item_from_record

_WORK_QUERY = f"""
SELECT {_WORK_COLUMNS} FROM agent_runs AS run
JOIN agent_tasks AS task ON task.id = run.task_id
JOIN agent_targets AS target ON target.id = run.target_id
JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
WHERE run.origin = 'plugin'
"""
_ACTIVE = "('queued','dispatching','running','waiting_user','unknown')"


class PluginTaskConflict(ValueError):
    pass


class PluginTaskStore:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        target_id: UUID,
        purpose: str,
        origin_context: dict[str, Any],
    ) -> AgentRunWorkItem:
        context = json.dumps(origin_context, ensure_ascii=False, allow_nan=False)
        if not purpose.strip() or len(purpose) > 4000 or len(context.encode()) > 24_000:
            raise ValueError("Plugin Task input exceeds budget")
        instance = origin_context["instanceId"]
        generation = origin_context["generation"]
        request = origin_context["requestId"]
        async with self.pool.acquire() as connection, connection.transaction():
            # Serialize across generations as well: an unconfirmed old stop
            # retains this instance's slot after a restart.
            await connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1,0))",
                f"plugin-task:{fish_id}:{instance}",
            )
            existing = await connection.fetchrow(
                _WORK_QUERY
                + """ AND run.fish_id=$1 AND run.session_id=$2
                AND run.origin_context->>'instanceId'=$3
                AND run.origin_context->>'generation'=$4
                AND run.origin_context->>'requestId'=$5""",
                fish_id,
                session_id,
                instance,
                generation,
                request,
            )
            if existing is not None:
                work = _work_item_from_record(existing)
                if (work.run.origin_context or {}).get(
                    "requestDigest"
                ) != origin_context["requestDigest"]:
                    raise PluginTaskConflict("request_id was used with different input")
                return work
            if await connection.fetchval(
                f"""SELECT EXISTS(SELECT 1 FROM agent_runs WHERE origin='plugin'
                AND fish_id=$1 AND origin_context->>'instanceId'=$2
                AND status IN {_ACTIVE})""",
                fish_id,
                instance,
            ):
                raise PluginTaskConflict("This plugin already has an active Task")
            owned = await connection.fetchval(
                """SELECT session.id FROM fish_sessions AS session
                JOIN fishes AS fish ON fish.id=session.fish_id
                JOIN agent_targets AS target ON target.id=$3 AND target.enabled
                WHERE session.id=$1 AND session.fish_id=$2 AND fish.deleted_at IS NULL
                FOR SHARE OF session, fish, target""",
                session_id,
                fish_id,
                target_id,
            )
            if owned is None:
                raise ValueError("Pet session or Agent target is unavailable")
            task_id, run_id = uuid4(), uuid4()
            await connection.execute(
                """INSERT INTO agent_tasks
                (id,fish_id,session_id,target_id,external_session_id,title)
                VALUES($1,$2,$3,$4,$5,$6)""",
                task_id,
                fish_id,
                session_id,
                target_id,
                f"fm-task-{task_id}",
                purpose[:160],
            )
            await connection.execute(
                """INSERT INTO agent_runs(id,task_id,target_id,fish_id,session_id,
                input,status,origin,origin_context)
                VALUES($1,$2,$3,$4,$5,$6,'queued','plugin',$7::jsonb)""",
                run_id,
                task_id,
                target_id,
                fish_id,
                session_id,
                purpose,
                context,
            )
            row = await connection.fetchrow(_WORK_QUERY + " AND run.id=$1", run_id)
            assert row is not None
            return _work_item_from_record(row)

    async def get(self, run_id: UUID) -> AgentRunWorkItem | None:
        row = await self.pool.fetchrow(_WORK_QUERY + " AND run.id=$1", run_id)
        return _work_item_from_record(row) if row else None

    async def by_request(
        self,
        fish: UUID,
        session: UUID,
        instance: UUID,
        generation: UUID,
        request: UUID,
    ) -> AgentRunWorkItem | None:
        row = await self.pool.fetchrow(
            _WORK_QUERY
            + """ AND run.fish_id=$1 AND run.session_id=$2
            AND run.origin_context->>'instanceId'=$3
            AND run.origin_context->>'generation'=$4
            AND run.origin_context->>'requestId'=$5""",
            fish,
            session,
            str(instance),
            str(generation),
            str(request),
        )
        return _work_item_from_record(row) if row else None

    async def active(self) -> tuple[AgentRunWorkItem, ...]:
        rows = await self.pool.fetch(_WORK_QUERY + f" AND run.status IN {_ACTIVE}")
        return tuple(_work_item_from_record(row) for row in rows)

    async def request_stop(self, run_id: UUID) -> AgentRunWorkItem | None:
        await self.pool.execute(
            """UPDATE agent_runs SET
            origin_context=origin_context || '{"stopRequested":true}'::jsonb,
            status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
            finished_at=CASE WHEN status='queued' THEN now() ELSE finished_at END,
            updated_at=now()
            WHERE id=$1 AND origin='plugin'""",
            run_id,
        )
        return await self.get(run_id)
