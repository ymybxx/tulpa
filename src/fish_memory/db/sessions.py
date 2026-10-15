from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.models import Session, SessionSummary

SESSION_COLUMNS = """
id,
fish_id,
created_at,
last_message_at,
history_window_start_sequence,
history_window_token_count
"""


class SessionNotFoundError(LookupError):
    pass


class SessionStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._pool = pool
        self._now = now

    async def create(
        self,
        fish_id: UUID,
        *,
        created_at: datetime | None = None,
    ) -> Session:
        effective_created_at = created_at or self._now()
        if effective_created_at.tzinfo is None:
            raise ValueError("session created_at must be timezone-aware")
        row = await self._pool.fetchrow(
            f"""
            INSERT INTO fish_sessions (id, fish_id, created_at)
            SELECT $1, $2, $3
            FROM fishes
            WHERE id = $2
              AND deleted_at IS NULL
            RETURNING {SESSION_COLUMNS}
            """,
            uuid4(),
            fish_id,
            effective_created_at,
        )
        if row is None:
            raise SessionNotFoundError(str(fish_id))
        return _session_from_record(row)

    async def get(self, session_id: UUID) -> Session | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {SESSION_COLUMNS}
            FROM fish_sessions
            WHERE id = $1
            """,
            session_id,
        )
        return None if row is None else _session_from_record(row)

    async def get_latest(self, fish_id: UUID) -> Session | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {SESSION_COLUMNS}
            FROM fish_sessions
            WHERE fish_id = $1
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            fish_id,
        )
        return None if row is None else _session_from_record(row)

    async def list(self, fish_id: UUID) -> list[SessionSummary]:
        rows = await self._pool.fetch(
            f"""
            SELECT {SESSION_COLUMNS}
            FROM fish_sessions
            WHERE fish_id = $1
            ORDER BY created_at DESC, id DESC
            """,
            fish_id,
        )
        return [_summary_from_record(row) for row in rows]

    async def require_owned(self, fish_id: UUID, session_id: UUID) -> Session:
        row = await self._pool.fetchrow(
            f"""
            SELECT {SESSION_COLUMNS}
            FROM fish_sessions
            WHERE id = $1
              AND fish_id = $2
            """,
            session_id,
            fish_id,
        )
        if row is None:
            raise SessionNotFoundError(str(session_id))
        return _session_from_record(row)


def _session_from_record(record: Mapping[str, object]) -> Session:
    return Session(
        id=cast(UUID, record["id"]),
        fish_id=cast(UUID, record["fish_id"]),
        created_at=cast(datetime, record["created_at"]),
        last_message_at=cast(datetime | None, record["last_message_at"]),
        history_window_start_sequence=cast(
            int, record["history_window_start_sequence"]
        ),
        history_window_token_count=cast(int, record["history_window_token_count"]),
    )


def _summary_from_record(record: Mapping[str, object]) -> SessionSummary:
    session = _session_from_record(record)
    return SessionSummary(
        id=session.id,
        fish_id=session.fish_id,
        created_at=session.created_at,
        last_message_at=session.last_message_at,
        history_window_start_sequence=session.history_window_start_sequence,
        history_window_token_count=session.history_window_token_count,
    )
