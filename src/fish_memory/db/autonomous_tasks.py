"""Autonomous admission into the same durable Task / Run lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import asyncpg

from fish_memory.agent.models import AgentRunWorkItem
from fish_memory.db.agent_tasks import _WORK_COLUMNS, _work_item_from_record

_WORK_QUERY = f"""
SELECT {_WORK_COLUMNS} FROM agent_runs AS run
JOIN agent_tasks AS task ON task.id = run.task_id
JOIN agent_targets AS target ON target.id = run.target_id
JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
WHERE run.origin = 'proactive'
"""


class AutonomousTaskStore:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def available(self, cooldown_seconds: float, starts_per_hour: int) -> bool:
        async with self._pool.acquire() as connection:
            return await self._available(connection, cooldown_seconds, starts_per_hour)

    @staticmethod
    async def _available(
        connection: asyncpg.Connection, cooldown: float, limit: int
    ) -> bool:
        now = datetime.now(UTC)
        return bool(
            await connection.fetchval(
                """
            SELECT NOT EXISTS (
                SELECT 1 FROM agent_runs WHERE origin = 'proactive'
                AND (status IN ('queued','dispatching','running',
                               'waiting_user','unknown')
                     OR created_at > $1)
            ) AND (SELECT count(*) FROM agent_runs WHERE origin = 'proactive'
                   AND created_at > $2) < $3
            """,
                now - timedelta(seconds=cooldown),
                now - timedelta(hours=1),
                limit,
            )
        )

    async def create(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        target_id: UUID,
        purpose: str,
        origin_context: Mapping[str, object],
        cooldown_seconds: float,
        starts_per_hour: int,
    ) -> AgentRunWorkItem | None:
        context = json.dumps(dict(origin_context), ensure_ascii=False)
        if not purpose.strip() or len(purpose) > 1000 or len(context) > 24_000:
            raise ValueError("autonomous task input exceeds budget")
        task_id, run_id = uuid4(), uuid4()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock("
                    "hashtext('fish-memory-autonomous-task'))"
                )
                if not await self._available(
                    connection, cooldown_seconds, starts_per_hour
                ):
                    return None
                owned = await connection.fetchval(
                    """
                    SELECT session.id FROM fish_sessions AS session
                    JOIN fishes AS fish ON fish.id = session.fish_id
                    JOIN agent_targets AS target ON target.id = $3 AND target.enabled
                    WHERE session.id = $1 AND session.fish_id = $2
                      AND fish.deleted_at IS NULL
                    FOR SHARE OF session, fish, target
                    """,
                    session_id,
                    fish_id,
                    target_id,
                )
                if owned is None:
                    return None
                await connection.execute(
                    """
                    INSERT INTO agent_tasks(id,fish_id,session_id,target_id,
                                            external_session_id,title)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    """,
                    task_id,
                    fish_id,
                    session_id,
                    target_id,
                    f"fm-task-{task_id}",
                    purpose[:160],
                )
                await connection.execute(
                    """
                    INSERT INTO agent_runs(id,task_id,target_id,fish_id,
                                          session_id,input,status,
                                           origin,origin_context)
                    VALUES ($1,$2,$3,$4,$5,$6,'queued','proactive',$7::jsonb)
                    """,
                    run_id,
                    task_id,
                    target_id,
                    fish_id,
                    session_id,
                    purpose,
                    context,
                )
                row = await connection.fetchrow(
                    _WORK_QUERY + " AND run.id = $1", run_id
                )
                return _work_item_from_record(row) if row else None

    async def get(self, run_id: UUID) -> AgentRunWorkItem | None:
        row = await self._pool.fetchrow(_WORK_QUERY + " AND run.id = $1", run_id)
        return _work_item_from_record(row) if row else None

    async def active(self) -> tuple[AgentRunWorkItem, ...]:
        rows = await self._pool.fetch(
            _WORK_QUERY
            + """
            AND run.status IN ('queued','dispatching','running',
                               'waiting_user','unknown')
            """
        )
        return tuple(_work_item_from_record(row) for row in rows)

    async def request_stop(self, run_id: UUID) -> AgentRunWorkItem | None:
        await self._pool.execute(
            """
            UPDATE agent_runs SET
                origin_context = origin_context || '{"stopRequested":true}'::jsonb,
                status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                finished_at = CASE WHEN status = 'queued'
                    THEN now() ELSE finished_at END
            WHERE id = $1 AND origin = 'proactive'
            """,
            run_id,
        )
        return await self.get(run_id)
