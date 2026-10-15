from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.home.models import HomeActionStatus


@dataclass(frozen=True, slots=True)
class HomeActionRun:
    id: UUID
    # The row sequence identifies an action; the event sequence changes with status.
    sequence: int
    event_sequence: int
    fish_id: UUID
    session_id: UUID
    source_message_id: UUID
    action: Mapping[str, object]
    status: HomeActionStatus
    entity_id: str | None
    service_domain: str | None
    service_name: str | None
    accepted: bool
    confirmed: bool
    result: object | None
    error: str | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


class HomeActionStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_run(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        source_message_id: UUID,
        action: Mapping[str, object] | None = None,
    ) -> HomeActionRun:
        row = await self._pool.fetchrow(
            """
            INSERT INTO home_action_runs (
                id, fish_id, session_id, source_message_id, action, status
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, 'routing')
            RETURNING *
            """,
            uuid4(),
            fish_id,
            session_id,
            source_message_id,
            _dump(action or {}),
        )
        return _from_row(cast(Mapping[str, Any], row))

    async def update_run(
        self,
        run_id: UUID,
        *,
        status: HomeActionStatus,
        action: Mapping[str, object] | None = None,
        entity_id: str | None = None,
        service_domain: str | None = None,
        service_name: str | None = None,
        accepted: bool = False,
        confirmed: bool = False,
        result: object | None = None,
        error: str | None = None,
    ) -> HomeActionRun:
        finished = status in {
            HomeActionStatus.SUCCEEDED,
            HomeActionStatus.PARTIAL,
            HomeActionStatus.FAILED,
            HomeActionStatus.AMBIGUOUS,
        }
        row = await self._pool.fetchrow(
            """
            UPDATE home_action_runs
            SET status = $2,
                event_sequence = nextval('home_action_run_events_sequence_seq'),
                action = COALESCE($3::jsonb, action),
                entity_id = COALESCE($4, entity_id),
                service_domain = COALESCE($5, service_domain),
                service_name = COALESCE($6, service_name),
                accepted = $7,
                confirmed = $8,
                result = $9::jsonb,
                error = $10,
                updated_at = date_trunc('second', now()),
                finished_at = CASE
                    WHEN $11 THEN date_trunc('second', now())
                    ELSE NULL
                END
            WHERE id = $1
            RETURNING *
            """,
            run_id,
            status.value,
            None if action is None else _dump(action),
            entity_id,
            service_domain,
            service_name,
            accepted,
            confirmed,
            None if result is None else _dump(result),
            error,
            finished,
        )
        if row is None:
            raise LookupError(str(run_id))
        return _from_row(cast(Mapping[str, Any], row))

    async def list_runs(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[HomeActionRun]:
        rows = await self._pool.fetch(
            """
            SELECT * FROM home_action_runs
            WHERE fish_id = $1 AND session_id = $2 AND sequence > $3
            ORDER BY sequence
            LIMIT $4
            """,
            fish_id,
            session_id,
            after_sequence,
            limit,
        )
        return [_from_row(row) for row in rows]

    async def list_events(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        after_event_sequence: int = 0,
        limit: int = 100,
    ) -> list[HomeActionRun]:
        rows = await self._pool.fetch(
            """
            SELECT * FROM home_action_runs
            WHERE fish_id = $1 AND session_id = $2 AND event_sequence > $3
            ORDER BY event_sequence
            LIMIT $4
            """,
            fish_id,
            session_id,
            after_event_sequence,
            limit,
        )
        return [_from_row(row) for row in rows]

    async def list_message_runs(
        self,
        fish_id: UUID,
        source_message_id: UUID,
    ) -> list[HomeActionRun]:
        rows = await self._pool.fetch(
            """
            SELECT * FROM home_action_runs
            WHERE fish_id = $1 AND source_message_id = $2
            ORDER BY sequence
            """,
            fish_id,
            source_message_id,
        )
        return [_from_row(row) for row in rows]


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _from_row(row: Mapping[str, Any]) -> HomeActionRun:
    # asyncpg returns jsonb as text unless the application installs a codec.
    action = _json_value(row["action"])
    if not isinstance(action, Mapping):
        raise ValueError("home action must be a JSON object")
    return HomeActionRun(
        id=cast(UUID, row["id"]),
        sequence=cast(int, row["sequence"]),
        event_sequence=cast(int, row["event_sequence"]),
        fish_id=cast(UUID, row["fish_id"]),
        session_id=cast(UUID, row["session_id"]),
        source_message_id=cast(UUID, row["source_message_id"]),
        action=cast(Mapping[str, object], action),
        status=HomeActionStatus(cast(str, row["status"])),
        entity_id=cast(str | None, row["entity_id"]),
        service_domain=cast(str | None, row["service_domain"]),
        service_name=cast(str | None, row["service_name"]),
        accepted=cast(bool, row["accepted"]),
        confirmed=cast(bool, row["confirmed"]),
        result=_json_value(row["result"]),
        error=cast(str | None, row["error"]),
        created_at=cast(datetime, row["created_at"]),
        updated_at=cast(datetime, row["updated_at"]),
        finished_at=cast(datetime | None, row["finished_at"]),
    )


def _json_value(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value
