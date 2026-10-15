import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.answer.history_tokens import (
    estimate_history_message_tokens,
    retain_recent_message_window,
)
from fish_memory.db.embedding_spaces import EmbeddingSpaceMismatchError
from fish_memory.db.sessions import SessionNotFoundError
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.models import FishSummary, Message, MessageStatus, Role

DEFAULT_EMBEDDING_SPACE = EmbeddingSpace(
    provider="local",
    model="intfloat/multilingual-e5-small",
    dimensions=384,
)

MESSAGE_COLUMNS = """
id,
fish_id,
session_id,
sequence,
role,
content,
status,
created_at,
context_through_sequence,
consumed_offset,
finished_at
,
history_token_count,
failure
"""

TERMINAL_STATUSES = {
    MessageStatus.COMPLETED,
    MessageStatus.INTERRUPTED,
    MessageStatus.FAILED,
}


@dataclass(frozen=True, slots=True)
class MessageOrigin:
    """Why an assistant message was sent without the user asking.

    `source` is the background message source; the rest comes from the
    Observation that triggered it, when there was one.
    """

    source: str
    observation_source: str | None = None
    observation_type: str | None = None
    app_name: str | None = None


@dataclass(frozen=True, slots=True)
class MessagePage:
    items: list[Message]
    has_before: bool
    has_after: bool
    min_sequence: int | None
    max_sequence: int | None


class MessageNotFoundError(LookupError):
    pass


class FishDeletedError(RuntimeError):
    pass


class LegacyFishReadOnlyError(RuntimeError):
    pass


class MessageIdentityConflictError(RuntimeError):
    pass


class MessagesStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        embedding_space: EmbeddingSpace = DEFAULT_EMBEDDING_SPACE,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        history_token_limit: int = 50_000,
        history_token_retain: int = 25_000,
    ) -> None:
        if history_token_retain >= history_token_limit:
            raise ValueError(
                "history_token_retain must be smaller than history_token_limit"
            )
        self._pool = pool
        self._embedding_space = embedding_space
        self._now = now
        self._history_token_limit = history_token_limit
        self._history_token_retain = history_token_retain

    async def insert_user_message(
        self,
        fish_id: UUID,
        content: str,
        *,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
        message_id: UUID | None = None,
    ) -> Message:
        message_id = message_id or uuid4()
        effective_created_at = created_at or self._now()
        history_token_count = estimate_history_message_tokens(content)
        if effective_created_at.tzinfo is None:
            raise ValueError("message created_at must be timezone-aware")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO fishes (
                        id,
                        embedding_provider,
                        embedding_model,
                        embedding_dimensions
                    )
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT DO NOTHING
                    """,
                    fish_id,
                    self._embedding_space.provider,
                    self._embedding_space.model,
                    self._embedding_space.dimensions,
                )
                fish = await connection.fetchrow(
                    """
                    SELECT
                        deleted_at,
                        snapshot_storage_version,
                        embedding_provider,
                        embedding_model,
                        embedding_dimensions
                    FROM fishes
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    fish_id,
                )
                if fish is None:
                    raise RuntimeError("database did not return the fish")
                if fish["deleted_at"] is not None:
                    raise FishDeletedError(str(fish_id))
                if fish["snapshot_storage_version"] != 2:
                    raise LegacyFishReadOnlyError(str(fish_id))
                await self._ensure_embedding_space(connection, fish_id, fish)
                if session_id is None:
                    session = await connection.fetchrow(
                        """
                        SELECT id
                        FROM fish_sessions
                        WHERE fish_id = $1
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                        """,
                        fish_id,
                    )
                    if session is None:
                        session = await connection.fetchrow(
                            """
                            INSERT INTO fish_sessions (id, fish_id, created_at)
                            VALUES ($1, $2, $3)
                            RETURNING *
                            """,
                            uuid4(),
                            fish_id,
                            effective_created_at,
                        )
                    if session is None:
                        raise RuntimeError("database did not return the session")
                    effective_session_id = cast(UUID, session["id"])
                else:
                    session = await connection.fetchrow(
                        """
                        SELECT id
                        FROM fish_sessions
                        WHERE id = $1
                          AND fish_id = $2
                        """,
                        session_id,
                        fish_id,
                    )
                    if session is None:
                        raise SessionNotFoundError(str(session_id))
                    effective_session_id = session_id
                row = await connection.fetchrow(
                    f"""
                    INSERT INTO messages (
                        id,
                        fish_id,
                        session_id,
                        role,
                        content,
                        status,
                        created_at,
                        history_token_count
                    )
                    VALUES ($1, $2, $3, 'user', $4, 'initialized', $5, $6)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING {MESSAGE_COLUMNS}
                    """,
                    message_id,
                    fish_id,
                    effective_session_id,
                    content,
                    effective_created_at,
                    history_token_count,
                )
                inserted = row is not None
                if row is None:
                    row = await connection.fetchrow(
                        f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = $1",
                        message_id,
                    )
                message = _required_message(row)
                _require_matching_message_identity(
                    message,
                    fish_id=fish_id,
                    session_id=effective_session_id,
                    role=Role.USER,
                    content=content,
                    status=MessageStatus.INITIALIZED,
                    context_through_sequence=None,
                )
                if inserted:
                    await connection.execute(
                        """
                        UPDATE fish_sessions
                        SET last_message_at = CASE
                            WHEN last_message_at IS NULL OR last_message_at < $2 THEN $2
                            ELSE last_message_at
                        END
                        WHERE id = $1
                        """,
                        effective_session_id,
                        effective_created_at,
                    )
                    await self._record_history_message(connection, message)
        return message

    async def insert_user_message_on_connection(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        created_at: datetime | None = None,
        message_id: UUID | None = None,
    ) -> Message:
        """Insert an idempotent User Message inside a caller-owned transaction."""
        message_id = message_id or uuid4()
        effective_created_at = created_at or self._now()
        if effective_created_at.tzinfo is None:
            raise ValueError("message created_at must be timezone-aware")
        history_token_count = estimate_history_message_tokens(content)
        session = await connection.fetchrow(
            """
            SELECT session.id
            FROM fish_sessions AS session
            JOIN fishes AS fish ON fish.id = session.fish_id
            WHERE session.id = $1
              AND session.fish_id = $2
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
            """,
            session_id,
            fish_id,
        )
        if session is None:
            raise SessionNotFoundError(str(session_id))
        row = await connection.fetchrow(
            f"""
            INSERT INTO messages (
                id, fish_id, session_id, role, content, status, created_at,
                history_token_count
            )
            VALUES ($1, $2, $3, 'user', $4, 'initialized', $5, $6)
            ON CONFLICT (id) DO NOTHING
            RETURNING {MESSAGE_COLUMNS}
            """,
            message_id,
            fish_id,
            session_id,
            content,
            effective_created_at,
            history_token_count,
        )
        inserted = row is not None
        if row is None:
            row = await connection.fetchrow(
                f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = $1",
                message_id,
            )
        message = _required_message(row)
        _require_matching_message_identity(
            message,
            fish_id=fish_id,
            session_id=session_id,
            role=Role.USER,
            content=content,
            status=MessageStatus.INITIALIZED,
            context_through_sequence=None,
        )
        if inserted:
            await connection.execute(
                """
                UPDATE fish_sessions
                SET last_message_at = CASE
                    WHEN last_message_at IS NULL OR last_message_at < $2 THEN $2
                    ELSE last_message_at
                END
                WHERE id = $1
                """,
                session_id,
                effective_created_at,
            )
            await self._record_history_message(connection, message)
        return message

    async def insert_background_message_on_connection(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID,
        session_id: UUID,
        context_through_sequence: int,
        content: str,
        *,
        created_at: datetime | None = None,
        message_id: UUID | None = None,
    ) -> Message:
        """Insert a completed Assistant Message behind the consumption barrier."""
        if not content.strip():
            raise ValueError("background Assistant message content must not be blank")
        message_id = message_id or uuid4()
        effective_created_at = created_at or self._now()
        if effective_created_at.tzinfo is None:
            raise ValueError("message created_at must be timezone-aware")
        row = await connection.fetchrow(
            f"""
            INSERT INTO messages (
                id,
                fish_id,
                session_id,
                role,
                content,
                status,
                created_at,
                context_through_sequence,
                consumed_offset,
                finished_at,
                history_token_count,
                fact_extraction_status
            )
            SELECT
                $1, $2, $3, 'assistant', $4, 'completed', $5, $6, 0, $5,
                0, 'waiting_consumption'
            FROM fish_sessions AS session
            JOIN fishes AS fish ON fish.id = session.fish_id
            WHERE session.id = $3
              AND session.fish_id = $2
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
            ON CONFLICT (id) DO NOTHING
            RETURNING {MESSAGE_COLUMNS}
            """,
            message_id,
            fish_id,
            session_id,
            content,
            effective_created_at,
            context_through_sequence,
        )
        inserted = row is not None
        if row is None:
            row = await connection.fetchrow(
                f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = $1",
                message_id,
            )
            if row is None:
                raise SessionNotFoundError(str(session_id))
        message = _required_message(row)
        _require_matching_message_identity(
            message,
            fish_id=fish_id,
            session_id=session_id,
            role=Role.ASSISTANT,
            content=content,
            status=MessageStatus.COMPLETED,
            context_through_sequence=context_through_sequence,
        )
        if message.consumed_offset != 0 or message.history_token_count != 0:
            raise MessageIdentityConflictError(str(message.id))
        if inserted:
            await connection.execute(
                """
                UPDATE fish_sessions
                SET last_message_at = CASE
                    WHEN last_message_at IS NULL OR last_message_at < $2 THEN $2
                    ELSE last_message_at
                END
                WHERE id = $1
                """,
                session_id,
                effective_created_at,
            )
        return message

    async def insert_agent_message_on_connection(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID,
        session_id: UUID,
        context_through_sequence: int,
        content: str,
        *,
        created_at: datetime | None = None,
        message_id: UUID | None = None,
    ) -> Message:
        """Compatibility alias for the generic background Assistant barrier."""
        return await self.insert_background_message_on_connection(
            connection,
            fish_id,
            session_id,
            context_through_sequence,
            content,
            created_at=created_at,
            message_id=message_id,
        )

    async def finalize_background_message_consumption(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> Message | None:
        if consumed_offset < -1:
            raise ValueError("consumed offset must be -1 or non-negative")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                return await self.finalize_background_message_consumption_on_connection(
                    connection,
                    message_id,
                    fish_id,
                    consumed_offset,
                    interrupted,
                )

    async def finalize_background_message_consumption_on_connection(
        self,
        connection: asyncpg.Connection,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> Message | None:
        if consumed_offset < -1:
            raise ValueError("consumed offset must be -1 or non-negative")
        row = await self._lock_answer_for_progress(connection, message_id, fish_id)
        if row is None:
            return None
        if row["status"] not in {
            MessageStatus.COMPLETED,
            MessageStatus.INTERRUPTED,
        }:
            return None
        if row["fact_extraction_status"] != "waiting_consumption":
            return _message_from_record(row)
        return await self._apply_background_consumption_locked(
            connection,
            row,
            consumed_offset,
            interrupted,
        )

    async def finalize_agent_message_consumption(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> Message | None:
        """Compatibility alias for the generic background Assistant barrier."""
        return await self.finalize_background_message_consumption(
            message_id,
            fish_id,
            consumed_offset,
            interrupted,
        )

    async def get_message_on_connection(
        self,
        connection: asyncpg.Connection,
        message_id: UUID,
    ) -> Message | None:
        row = await connection.fetchrow(
            f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = $1",
            message_id,
        )
        return None if row is None else _message_from_record(row)

    async def _lock_answer_for_progress(
        self,
        connection: asyncpg.Connection,
        message_id: UUID,
        fish_id: UUID,
    ) -> Mapping[str, object] | None:
        return await connection.fetchrow(
            f"""
            SELECT {MESSAGE_COLUMNS}, fact_extraction_status
            FROM messages AS message
            WHERE message.id = $1
              AND message.fish_id = $2
              AND message.role = 'assistant'
              AND message.status IN (
                  'generating', 'completed', 'interrupted', 'failed'
              )
              AND EXISTS (
                  SELECT 1
                  FROM fishes AS fish
                  WHERE fish.id = message.fish_id
                    AND fish.deleted_at IS NULL
              )
            FOR UPDATE
            """,
            message_id,
            fish_id,
        )

    async def _apply_background_consumption_locked(
        self,
        connection: asyncpg.Connection,
        row: Mapping[str, object],
        consumed_offset: int,
        interrupted: bool,
    ) -> Message:
        content = cast(str, row["content"])
        full = consumed_offset == -1 or consumed_offset >= len(content)
        if not full and not interrupted:
            updated = await connection.fetchrow(
                f"""
                UPDATE messages
                SET consumed_offset = GREATEST(consumed_offset, $3)
                WHERE id = $1
                  AND fish_id = $2
                  AND fact_extraction_status = 'waiting_consumption'
                RETURNING {MESSAGE_COLUMNS}
                """,
                row["id"],
                row["fish_id"],
                consumed_offset,
            )
            return _required_message(updated)

        effective_offset = (
            consumed_offset
            if consumed_offset == -1
            else min(consumed_offset, len(content))
        )
        visible_content = (
            content[:effective_offset]
            if interrupted and effective_offset != -1
            else content
        )
        status = MessageStatus.INTERRUPTED if interrupted else MessageStatus.COMPLETED
        token_count = _history_token_count(
            Role.ASSISTANT,
            visible_content,
            status,
            effective_offset,
        )
        updated = await connection.fetchrow(
            f"""
            UPDATE messages
            SET content = $3,
                status = $4,
                consumed_offset = $5,
                finished_at = COALESCE(finished_at, now()),
                history_token_count = $6,
                fact_extraction_status = 'pending',
                fact_extraction_error = NULL
            WHERE id = $1
              AND fish_id = $2
              AND fact_extraction_status = 'waiting_consumption'
            RETURNING {MESSAGE_COLUMNS}
            """,
            row["id"],
            row["fish_id"],
            visible_content,
            status.value,
            effective_offset,
            token_count,
        )
        if updated is None:
            updated = await connection.fetchrow(
                f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = $1",
                row["id"],
            )
        await self._rebuild_history_window(
            connection,
            cast(UUID, row["session_id"]),
        )
        return _required_message(updated)

    async def _ensure_embedding_space(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID,
        fish: Mapping[str, object],
    ) -> None:
        provider = fish["embedding_provider"]
        model = fish["embedding_model"]
        dimensions = fish["embedding_dimensions"]
        if provider is None and model is None and dimensions is None:
            await connection.execute(
                """
                UPDATE fishes
                SET embedding_provider = $2,
                    embedding_model = $3,
                    embedding_dimensions = $4
                WHERE id = $1
                """,
                fish_id,
                self._embedding_space.provider,
                self._embedding_space.model,
                self._embedding_space.dimensions,
            )
            return
        if not isinstance(provider, str) or not isinstance(model, str):
            raise RuntimeError("Fish has an incomplete embedding space")
        if not isinstance(dimensions, int):
            raise RuntimeError("Fish has an incomplete embedding space")
        existing = EmbeddingSpace(provider, model, dimensions)
        if existing != self._embedding_space:
            raise EmbeddingSpaceMismatchError(
                fish_id,
                existing,
                self._embedding_space,
            )

    async def soft_delete_fish(self, fish_id: UUID) -> bool:
        row = await self._pool.fetchrow(
            """
            UPDATE fishes
            SET deleted_at = now(),
                is_assistant = FALSE
            WHERE id = $1
              AND deleted_at IS NULL
            RETURNING id
            """,
            fish_id,
        )
        return row is not None

    async def insert_generating_answer(
        self,
        fish_id: UUID,
        session_id: UUID,
        context_through_sequence: int,
        *,
        message_id: UUID | None = None,
        consumed_offset: int = -1,
    ) -> Message:
        message_id = message_id or uuid4()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    INSERT INTO messages (
                        id,
                        fish_id,
                        session_id,
                        role,
                        content,
                        status,
                        context_through_sequence,
                        consumed_offset
                    )
                    SELECT $1, $2, $3, 'assistant', '', 'generating', $4, $5
                    FROM fish_sessions
                    WHERE id = $3
                      AND fish_id = $2
                    ON CONFLICT (id) DO NOTHING
                    RETURNING {MESSAGE_COLUMNS}
                    """,
                    message_id,
                    fish_id,
                    session_id,
                    context_through_sequence,
                    consumed_offset,
                )
                inserted = row is not None
                if row is None:
                    row = await connection.fetchrow(
                        f"SELECT {MESSAGE_COLUMNS} FROM messages WHERE id = $1",
                        message_id,
                    )
                    if row is None:
                        raise SessionNotFoundError(str(session_id))
                message = _required_message(row)
                _require_matching_message_identity(
                    message,
                    fish_id=fish_id,
                    session_id=session_id,
                    role=Role.ASSISTANT,
                    content="",
                    status=MessageStatus.GENERATING,
                    context_through_sequence=context_through_sequence,
                )
                if inserted:
                    await connection.execute(
                        """
                        UPDATE fish_sessions
                        SET last_message_at = CASE
                            WHEN last_message_at IS NULL OR last_message_at < $2 THEN $2
                            ELSE last_message_at
                        END
                        WHERE id = $1
                        """,
                        session_id,
                        message.created_at,
                    )
        return message

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int | None = None,
    ) -> list[Message]:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                window_start, _initialized = await self._ensure_history_window(
                    connection, session_id
                )
                limit_sql = "" if limit is None else "LIMIT $5"
                arguments: tuple[object, ...] = (
                    (session_id, fish_id, before_sequence, window_start)
                    if limit is None
                    else (session_id, fish_id, before_sequence, window_start, limit)
                )
                rows = await connection.fetch(
                    f"""
            SELECT {MESSAGE_COLUMNS}
            FROM messages
            WHERE session_id = $1
              AND fish_id = $2
              AND sequence < $3
              AND sequence >= $4
              AND EXISTS (
                  SELECT 1
                  FROM fishes
                  WHERE id = messages.fish_id
                    AND deleted_at IS NULL
              )
              AND (
                  role = 'user'
                  OR (
                      status IN ('completed', 'interrupted')
                      AND fact_extraction_status <> 'waiting_consumption'
                  )
            )
            ORDER BY sequence DESC
            {limit_sql}
            """,
                    *arguments,
                )
        return [_message_from_record(row) for row in reversed(rows)]

    async def load_history_before_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        *,
        limit: int,
    ) -> list[Message] | None:
        """Load eligible history before a verified user source Message.

        The source identity check keeps external-Agent context strictly inside the
        FishSession that created the Run; it deliberately does not fall back to
        a Fish-wide or latest-message query.
        """
        if limit < 1:
            raise ValueError("limit must be positive")
        source = await self._pool.fetchrow(
            """
            SELECT sequence
            FROM messages
            WHERE id = $1
              AND fish_id = $2
              AND session_id = $3
              AND role = 'user'
            """,
            message_id,
            fish_id,
            session_id,
        )
        if source is None:
            return None
        return await self.load_history(
            fish_id,
            session_id,
            before_sequence=cast(int, source["sequence"]),
            limit=limit,
        )

    async def load_history_before_task_root_source(
        self,
        fish_id: UUID,
        session_id: UUID,
        task_id: UUID,
        *,
        limit: int,
    ) -> list[Message] | None:
        """Load context preceding the unique root Run source of one Task."""
        if limit < 1:
            raise ValueError("limit must be positive")
        source = await self._pool.fetchrow(
            """
            SELECT source.sequence
            FROM agent_runs AS root
            JOIN agent_tasks AS task ON task.id = root.task_id
            JOIN messages AS source ON source.id = root.source_message_id
            WHERE root.task_id = $1
              AND root.previous_run_id IS NULL
              AND root.fish_id = $2
              AND root.session_id = $3
              AND task.fish_id = $2
              AND task.session_id = $3
              AND source.fish_id = $2
              AND source.session_id = $3
              AND source.role = 'user'
              AND NOT EXISTS (
                  SELECT 1
                  FROM agent_runs AS another_root
                  WHERE another_root.task_id = root.task_id
                    AND another_root.previous_run_id IS NULL
                    AND another_root.id <> root.id
              )
            """,
            task_id,
            fish_id,
            session_id,
        )
        if source is None:
            return None
        return await self.load_history(
            fish_id,
            session_id,
            before_sequence=cast(int, source["sequence"]),
            limit=limit,
        )

    async def load_history_readonly(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
        appended_history_token_count: int | None = None,
    ) -> list[Message]:
        """Load history without writes, optionally simulating pending user tokens."""
        async with self._pool.acquire() as connection:
            state = await connection.fetchrow(
                """
                SELECT history_window_start_sequence, history_window_token_count
                FROM fish_sessions
                WHERE id = $1
                """,
                session_id,
            )
            if state is None:
                raise SessionNotFoundError(str(session_id))
            window_start = cast(int, state["history_window_start_sequence"])
            window_total = cast(int, state["history_window_token_count"])
            requires_rebuild = appended_history_token_count is not None and (
                window_start == 0
                or window_total + appended_history_token_count
                > self._history_token_limit
            )
            if window_start > 0 and not requires_rebuild:
                rows = await connection.fetch(
                    f"""
                    SELECT {MESSAGE_COLUMNS}
                    FROM messages
                    WHERE session_id = $1
                      AND fish_id = $2
                      AND sequence < $3
                      AND sequence >= $4
                      AND EXISTS (
                          SELECT 1
                          FROM fishes
                          WHERE id = messages.fish_id
                            AND deleted_at IS NULL
                      )
                      AND (
                          role = 'user'
                          OR (
                              status IN ('completed', 'interrupted')
                              AND fact_extraction_status <> 'waiting_consumption'
                          )
                      )
                    ORDER BY sequence DESC
                    LIMIT $5
                    """,
                    session_id,
                    fish_id,
                    before_sequence,
                    window_start,
                    limit,
                )
                return [_message_from_record(row) for row in reversed(rows)]

            rows = await connection.fetch(
                f"""
                SELECT {MESSAGE_COLUMNS}
                FROM messages
                WHERE session_id = $1
                  AND fish_id = $2
                  AND EXISTS (
                      SELECT 1
                      FROM fishes
                      WHERE id = messages.fish_id
                        AND deleted_at IS NULL
                  )
                  AND (
                      role = 'user'
                      OR (
                          status IN ('completed', 'interrupted')
                          AND fact_extraction_status <> 'waiting_consumption'
                      )
                  )
                ORDER BY sequence
                """,
                session_id,
                fish_id,
            )

        messages = [_message_from_record(row) for row in rows]
        window_messages = [
            (
                message.sequence,
                message.history_token_count
                or _history_token_count(
                    message.role,
                    message.content,
                    message.status,
                    message.consumed_offset,
                ),
            )
            for message in messages
        ]
        if appended_history_token_count is not None:
            window_messages.append(
                (
                    max((message.sequence for message in messages), default=0) + 1,
                    appended_history_token_count,
                )
            )
        effective_window_start, _ = retain_recent_message_window(
            window_messages,
            retain_limit=self._history_token_retain,
        )
        retained_messages = [
            message
            for message in messages
            if effective_window_start <= message.sequence < before_sequence
        ]
        return retained_messages[-limit:] if limit else []

    async def get_latest_user_sequence(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> int | None:
        value = await self._pool.fetchval(
            """
            SELECT max(sequence)
            FROM messages
            WHERE fish_id = $1
              AND session_id = $2
              AND role = 'user'
              AND EXISTS (
                  SELECT 1
                  FROM fishes
                  WHERE id = messages.fish_id
                    AND deleted_at IS NULL
              )
            """,
            fish_id,
            session_id,
        )
        return cast(int | None, value)

    async def list_fishes(self) -> list[FishSummary]:
        rows = await self._pool.fetch(
            """
            WITH visible AS (
                SELECT
                    message.fish_id,
                    message.content,
                    message.created_at,
                    count(*) OVER (
                        PARTITION BY message.fish_id
                    ) AS message_count,
                    row_number() OVER (
                        PARTITION BY message.fish_id
                        ORDER BY message.sequence DESC
                    ) AS recent_rank
                FROM messages AS message
                JOIN fishes AS fish ON fish.id = message.fish_id
                WHERE fish.deleted_at IS NULL
                  AND (
                      message.role = 'user'
                      OR message.status IN (
                          'completed', 'interrupted', 'failed'
                      )
                  )
            )
            , summaries AS (
                SELECT
                    fish_id,
                    content AS last_message,
                    created_at AS last_active_at,
                    message_count
                FROM visible
                WHERE recent_rank = 1
            )
            SELECT
                fish.id AS fish_id,
                COALESCE(summary.last_message, '') AS last_message,
                summary.last_active_at,
                COALESCE(summary.message_count, 0)::integer AS message_count,
                fish.is_assistant,
                COALESCE(fish.name, '') AS name,
                COALESCE(fish.role, '') AS role,
                fish.background_owner IS NOT NULL AS evaluation
            FROM fishes AS fish
            LEFT JOIN summaries AS summary ON summary.fish_id = fish.id
            WHERE fish.deleted_at IS NULL
            ORDER BY summary.last_active_at DESC NULLS LAST, fish.id
            """
        )
        return [_fish_summary_from_record(row) for row in rows]

    async def list_messages(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        limit: int,
    ) -> list[Message]:
        rows = await self._pool.fetch(
            f"""
            SELECT *
            FROM (
                SELECT {MESSAGE_COLUMNS}
                FROM messages
                WHERE fish_id = $1
                  AND session_id = $2
                  AND EXISTS (
                      SELECT 1 FROM fishes
                      WHERE id = messages.fish_id AND deleted_at IS NULL
                  )
                  AND (
                      role = 'user'
                      OR status IN ('completed', 'interrupted', 'failed')
                  )
                ORDER BY sequence DESC
                LIMIT $3
            ) AS recent
            ORDER BY sequence
            """,
            fish_id,
            session_id,
            limit,
        )
        return [_message_from_record(row) for row in rows]

    async def list_message_page(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        limit: int,
        before: int | None = None,
        after: int | None = None,
    ) -> MessagePage:
        if before is not None and after is not None:
            raise ValueError("before and after cannot be used together")
        comparison = (
            "< $3"
            if before is not None
            else "> $3"
            if after is not None
            else "IS NOT NULL"
        )
        order = (
            "DESC"
            if before is not None or (before is None and after is None)
            else "ASC"
        )
        cursor = before if before is not None else after
        if cursor is None:
            rows = await self._pool.fetch(
                f"""
                SELECT {MESSAGE_COLUMNS}
                FROM messages
                WHERE fish_id = $1
                  AND session_id = $2
                  AND EXISTS (
                      SELECT 1 FROM fishes
                      WHERE id = messages.fish_id AND deleted_at IS NULL
                  )
                  AND (
                      role = 'user'
                      OR status IN ('completed', 'interrupted', 'failed')
                  )
                ORDER BY sequence DESC
                LIMIT $3
                """,
                fish_id,
                session_id,
                limit + 1,
            )
            has_before = len(rows) > limit
            selected = rows[:limit]
            selected.reverse()
        else:
            rows = await self._pool.fetch(
                f"""
                SELECT {MESSAGE_COLUMNS}
                FROM messages
                WHERE fish_id = $1
                  AND session_id = $2
                  AND sequence {comparison}
                  AND EXISTS (
                      SELECT 1 FROM fishes
                      WHERE id = messages.fish_id AND deleted_at IS NULL
                  )
                  AND (
                      role = 'user'
                      OR status IN ('completed', 'interrupted', 'failed')
                  )
                ORDER BY sequence {order}
                LIMIT $4
                """,
                fish_id,
                session_id,
                cursor,
                limit + 1,
            )
            has_before = before is not None and len(rows) > limit
            has_after = after is not None and len(rows) > limit
            selected = rows[:limit]
            if before is not None:
                selected.reverse()

        items = [_message_from_record(row) for row in selected]
        if not items:
            bounds = await self._pool.fetchrow(
                """
                SELECT min(sequence) AS min_sequence, max(sequence) AS max_sequence
                FROM messages
                WHERE fish_id = $1
                  AND session_id = $2
                """,
                fish_id,
                session_id,
            )
            return MessagePage(
                items=[],
                has_before=False,
                has_after=False,
                min_sequence=(
                    None if bounds is None else cast(int | None, bounds["min_sequence"])
                ),
                max_sequence=(
                    None if bounds is None else cast(int | None, bounds["max_sequence"])
                ),
            )
        if before is None and after is None:
            has_after = False
        return MessagePage(
            items=items,
            has_before=has_before,
            has_after=has_after,
            min_sequence=items[0].sequence,
            max_sequence=items[-1].sequence,
        )

    async def list_timeline_page(
        self,
        fish_id: UUID,
        *,
        limit: int,
        before: int | None = None,
        after: int | None = None,
    ) -> MessagePage:
        """One conversation across sessions, oldest first, as the chat page shows it.

        Unlike a session page it keeps an answer still being generated, so the
        page can show TA thinking and pick the text up once it is done.
        """
        if before is not None and after is not None:
            raise ValueError("before and after cannot be used together")
        comparison = (
            "< $2"
            if before is not None
            else "> $2"
            if after is not None
            else "IS NOT NULL"
        )
        order = "ASC" if after is not None else "DESC"
        rows = await self._pool.fetch(
            f"""
            SELECT {MESSAGE_COLUMNS}
            FROM messages
            WHERE fish_id = $1
              AND ($2::bigint IS NULL OR sequence {comparison})
              AND EXISTS (
                  SELECT 1 FROM fishes
                  WHERE id = messages.fish_id AND deleted_at IS NULL
              )
            ORDER BY sequence {order}
            LIMIT $3
            """,
            fish_id,
            before if before is not None else after,
            limit + 1,
        )
        more = len(rows) > limit
        selected = rows[:limit]
        if after is None:
            selected.reverse()
        items = [_message_from_record(row) for row in selected]
        return MessagePage(
            items=items,
            has_before=more and after is None,
            has_after=more and after is not None,
            min_sequence=items[0].sequence if items else None,
            max_sequence=items[-1].sequence if items else None,
        )

    async def search_messages(
        self,
        fish_id: UUID,
        query: str,
        *,
        limit: int,
    ) -> list[Message]:
        """Finished messages containing `query` literally, newest first."""
        pattern = "%" + re.sub(r"([\\%_])", r"\\\1", query) + "%"
        rows = await self._pool.fetch(
            f"""
            SELECT {MESSAGE_COLUMNS}
            FROM messages
            WHERE fish_id = $1
              AND content ILIKE $2 ESCAPE '\\'
              AND (role = 'user' OR status IN ('completed', 'interrupted'))
              AND EXISTS (
                  SELECT 1 FROM fishes
                  WHERE id = messages.fish_id AND deleted_at IS NULL
              )
            ORDER BY sequence DESC
            LIMIT $3
            """,
            fish_id,
            pattern,
            limit,
        )
        return [_message_from_record(row) for row in rows]

    async def record_input_mode(self, message_id: UUID, mode: str) -> None:
        """Whether the user typed or said this message; the first record wins."""
        await self._pool.execute(
            "INSERT INTO message_input_modes (message_id, mode) VALUES ($1, $2) "
            "ON CONFLICT (message_id) DO NOTHING",
            message_id,
            mode,
        )

    async def input_modes(
        self, fish_id: UUID, message_ids: Sequence[UUID]
    ) -> dict[UUID, Literal["typed", "spoken"]]:
        # The table's CHECK constraint admits only these two modes.
        if not message_ids:
            return {}
        rows = await self._pool.fetch(
            """
            SELECT mode_row.message_id, mode_row.mode
            FROM message_input_modes AS mode_row
            JOIN messages ON messages.id = mode_row.message_id
            WHERE messages.fish_id = $1 AND mode_row.message_id = ANY($2::uuid[])
            """,
            fish_id,
            list(message_ids),
        )
        return {row["message_id"]: row["mode"] for row in rows}

    async def message_origins(
        self, fish_id: UUID, message_ids: Sequence[UUID]
    ) -> dict[UUID, MessageOrigin]:
        """Origins of the given messages that TA sent on TA's own."""
        if not message_ids:
            return {}
        rows = await self._pool.fetch(
            """
            SELECT background.message_id, background.source,
                   observation.source AS observation_source,
                   observation.observation_type, observation.app_name
            FROM background_assistant_deliveries AS background
            JOIN messages ON messages.id = background.message_id
            LEFT JOIN observations AS observation
              ON observation.id = background.source_observation_id
            WHERE messages.fish_id = $1
              AND background.message_id = ANY($2::uuid[])
            """,
            fish_id,
            list(message_ids),
        )
        return {
            row["message_id"]: MessageOrigin(
                source=row["source"],
                observation_source=row["observation_source"],
                observation_type=row["observation_type"],
                app_name=row["app_name"],
            )
            for row in rows
        }

    async def finish_answer(
        self,
        message_id: UUID,
        fish_id: UUID,
        *,
        content: str,
        requested_status: MessageStatus,
        failure: str | None = None,
    ) -> MessageStatus:
        if requested_status not in TERMINAL_STATUSES:
            raise ValueError("requested status must be terminal")

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT context_through_sequence
                         , consumed_offset
                         , sequence
                         , session_id
                    FROM messages
                    WHERE id = $1
                      AND fish_id = $2
                      AND role = 'assistant'
                      AND status = 'generating'
                    FOR UPDATE
                    """,
                    message_id,
                    fish_id,
                )
                if row is None:
                    raise MessageNotFoundError(str(message_id))

                effective_status = requested_status
                context_sequence = cast(int, row["context_through_sequence"])
                if requested_status is MessageStatus.COMPLETED:
                    has_new_user_message = await connection.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM messages
                            WHERE session_id = $1
                              AND role = 'user'
                              AND sequence > $2
                        )
                        """,
                        row["session_id"],
                        context_sequence,
                    )
                    if has_new_user_message:
                        effective_status = MessageStatus.INTERRUPTED

                history_token_count = _history_token_count(
                    Role.ASSISTANT,
                    content,
                    effective_status,
                    cast(int | None, row["consumed_offset"]),
                )
                await connection.execute(
                    """
                    UPDATE messages
                    SET content = $3,
                        status = $4,
                        finished_at = now(),
                        history_token_count = $5,
                        failure = $6
                    WHERE id = $1 AND fish_id = $2
                    """,
                    message_id,
                    fish_id,
                    content,
                    effective_status,
                    history_token_count,
                    failure,
                )
                if effective_status in {
                    MessageStatus.COMPLETED,
                    MessageStatus.INTERRUPTED,
                }:
                    await self._record_history_message(
                        connection,
                        Message(
                            id=message_id,
                            fish_id=fish_id,
                            session_id=cast(UUID, row["session_id"]),
                            sequence=cast(int, row["sequence"]),
                            role=Role.ASSISTANT,
                            content=content,
                            status=effective_status,
                            created_at=self._now(),
                            consumed_offset=cast(int | None, row["consumed_offset"]),
                            history_token_count=history_token_count,
                        ),
                    )
                return effective_status

    async def record_failure(
        self, message_id: UUID, fish_id: UUID, failure: str
    ) -> bool:
        """Note why a finished reply could not be spoken; the first reason stays."""
        async with self._pool.acquire() as connection:
            result = await connection.execute(
                """
                UPDATE messages SET failure = $3
                WHERE id = $1 AND fish_id = $2
                  AND role = 'assistant' AND failure IS NULL
                """,
                message_id,
                fish_id,
                failure,
            )
        return result.endswith(" 1")

    async def update_consumed_offset(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
    ) -> int | None:
        if consumed_offset < -1:
            raise ValueError("consumed offset must be -1 or non-negative")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await self._lock_answer_for_progress(
                    connection, message_id, fish_id
                )
                if row is None:
                    return None
                status = MessageStatus(cast(str, row["status"]))
                current_offset = cast(int | None, row["consumed_offset"])
                if status is MessageStatus.INTERRUPTED:
                    return current_offset
                if row["fact_extraction_status"] == "waiting_consumption":
                    message = await self._apply_background_consumption_locked(
                        connection,
                        row,
                        consumed_offset,
                        False,
                    )
                    return message.consumed_offset

                content = cast(str, row["content"])
                if status is not MessageStatus.GENERATING and consumed_offset > len(
                    content
                ):
                    raise ValueError("consumed offset exceeds answer content length")
                updated = await connection.fetchrow(
                    f"""
                    UPDATE messages
                    SET consumed_offset = GREATEST(consumed_offset, $3)
                    WHERE id = $1 AND fish_id = $2
                    RETURNING {MESSAGE_COLUMNS}
                    """,
                    message_id,
                    fish_id,
                    consumed_offset,
                )
                return _required_message(updated).consumed_offset

    async def interrupt_terminal_answer(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
    ) -> int | None:
        """Atomically replace a completed answer with its actually consumed prefix."""
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT message.content,
                           message.consumed_offset,
                           message.sequence,
                           message.status,
                           message.session_id
                    FROM messages AS message
                    JOIN fishes AS fish ON fish.id = message.fish_id
                    WHERE message.id = $1
                      AND message.fish_id = $2
                      AND fish.deleted_at IS NULL
                      AND message.role = 'assistant'
                      AND message.status IN ('completed', 'interrupted')
                    FOR UPDATE
                    """,
                    message_id,
                    fish_id,
                )
                if row is None:
                    return None
                if row["status"] == MessageStatus.INTERRUPTED:
                    return cast(int, row["consumed_offset"])

                content = cast(str, row["content"])
                if consumed_offset > len(content):
                    raise ValueError("consumed offset exceeds answer content length")
                visible_content = content[:consumed_offset]
                history_token_count = _history_token_count(
                    Role.ASSISTANT,
                    visible_content,
                    MessageStatus.INTERRUPTED,
                    consumed_offset,
                )
                await connection.execute(
                    """
                    UPDATE messages
                    SET content = $4,
                        consumed_offset = $3,
                        status = 'interrupted',
                        history_token_count = $5
                    WHERE id = $1 AND fish_id = $2
                    """,
                    message_id,
                    fish_id,
                    consumed_offset,
                    visible_content,
                    history_token_count,
                )
                await _keep_unsaid(connection, message_id, content[consumed_offset:])
                await self._rebuild_history_window(
                    connection,
                    cast(UUID, row["session_id"]),
                )
                return consumed_offset

    async def record_unsaid(self, message_id: UUID, content: str) -> None:
        """Keep what an interrupted answer never said, apart from what it said."""
        async with self._pool.acquire() as connection:
            await _keep_unsaid(connection, message_id, content)

    async def unsaid_texts(
        self, fish_id: UUID, message_ids: Sequence[UUID]
    ) -> dict[UUID, str]:
        if not message_ids:
            return {}
        rows = await self._pool.fetch(
            """
            SELECT unsaid.message_id, unsaid.content
            FROM message_unsaid AS unsaid
            JOIN messages ON messages.id = unsaid.message_id
            WHERE messages.fish_id = $1 AND unsaid.message_id = ANY($2::uuid[])
            """,
            fish_id,
            list(message_ids),
        )
        return {row["message_id"]: row["content"] for row in rows}

    async def fail_stale_generating_answers(self) -> int:
        value = await self._pool.fetchval(
            """
            WITH updated AS (
                UPDATE messages
                SET status = 'failed', finished_at = now()
                WHERE role = 'assistant' AND status = 'generating'
                RETURNING 1
            )
            SELECT count(*) FROM updated
            """
        )
        return cast(int, value)

    async def _record_history_message(
        self,
        connection: asyncpg.Connection,
        message: Message,
    ) -> None:
        if not _is_history_message(message.role, message.status):
            return
        window_start, initialized = await self._ensure_history_window(
            connection, message.session_id
        )
        if not initialized:
            return

        state = await connection.fetchrow(
            """
            SELECT history_window_start_sequence, history_window_token_count
            FROM fish_sessions
            WHERE id = $1
            FOR UPDATE
            """,
            message.session_id,
        )
        if state is None:
            raise RuntimeError("database did not return the session token window")
        total = cast(int, state["history_window_token_count"])
        next_total = total + message.history_token_count
        if next_total <= self._history_token_limit:
            await connection.execute(
                """
                UPDATE fish_sessions
                SET history_window_start_sequence = $2,
                    history_window_token_count = $3
                WHERE id = $1
                """,
                message.session_id,
                window_start or message.sequence,
                next_total,
            )
            return
        await self._rebuild_history_window(connection, message.session_id)

    async def _ensure_history_window(
        self,
        connection: asyncpg.Connection,
        session_id: UUID,
    ) -> tuple[int, bool]:
        state = await connection.fetchrow(
            """
            SELECT history_window_start_sequence
            FROM fish_sessions
            WHERE id = $1
            FOR UPDATE
            """,
            session_id,
        )
        if state is None:
            raise SessionNotFoundError(str(session_id))
        start_sequence = cast(int, state["history_window_start_sequence"])
        if start_sequence > 0:
            return start_sequence, True
        await self._rebuild_history_window(connection, session_id)
        state = await connection.fetchrow(
            """
            SELECT history_window_start_sequence
            FROM fish_sessions
            WHERE id = $1
            """,
            session_id,
        )
        if state is None:
            raise SessionNotFoundError(str(session_id))
        return cast(int, state["history_window_start_sequence"]), False

    async def _rebuild_history_window(
        self,
        connection: asyncpg.Connection,
        session_id: UUID,
    ) -> None:
        rows = await connection.fetch(
            f"""
            SELECT {MESSAGE_COLUMNS}
            FROM messages
            WHERE session_id = $1
              AND (
                  role = 'user'
                  OR (
                      status IN ('completed', 'interrupted')
                      AND fact_extraction_status <> 'waiting_consumption'
                  )
              )
            ORDER BY sequence
            """,
            session_id,
        )
        messages = [_message_from_record(row) for row in rows]
        updates = [
            (
                message.id,
                _history_token_count(
                    message.role,
                    message.content,
                    message.status,
                    message.consumed_offset,
                ),
            )
            for message in messages
            if message.history_token_count == 0
        ]
        if updates:
            await connection.executemany(
                """
                UPDATE messages
                SET history_token_count = $2
                WHERE id = $1
                """,
                updates,
            )
            update_values = dict(updates)
            messages = [
                Message(
                    id=message.id,
                    fish_id=message.fish_id,
                    session_id=message.session_id,
                    sequence=message.sequence,
                    role=message.role,
                    content=message.content,
                    status=message.status,
                    created_at=message.created_at,
                    context_through_sequence=message.context_through_sequence,
                    consumed_offset=message.consumed_offset,
                    finished_at=message.finished_at,
                    history_token_count=update_values.get(
                        message.id, message.history_token_count
                    ),
                )
                for message in messages
            ]
        start_sequence, total = retain_recent_message_window(
            [(message.sequence, message.history_token_count) for message in messages],
            retain_limit=self._history_token_retain,
        )
        await connection.execute(
            """
            UPDATE fish_sessions
            SET history_window_start_sequence = $2,
                history_window_token_count = $3
            WHERE id = $1
            """,
            session_id,
            start_sequence,
            total,
        )


def _is_history_message(role: Role, status: MessageStatus) -> bool:
    return role is Role.USER or status in {
        MessageStatus.COMPLETED,
        MessageStatus.INTERRUPTED,
    }


def _history_token_count(
    role: Role,
    content: str,
    status: MessageStatus,
    consumed_offset: int | None,
) -> int:
    if not _is_history_message(role, status):
        return 0
    if role is Role.ASSISTANT and consumed_offset not in (None, -1):
        content = content[:consumed_offset]
    return estimate_history_message_tokens(content)


def _required_message(record: Mapping[str, object] | None) -> Message:
    if record is None:
        raise RuntimeError("database did not return the inserted message")
    return _message_from_record(record)


def _require_matching_message_identity(
    message: Message,
    *,
    fish_id: UUID,
    session_id: UUID,
    role: Role,
    content: str,
    status: MessageStatus,
    context_through_sequence: int | None,
) -> None:
    if (
        message.fish_id != fish_id
        or message.session_id != session_id
        or message.role is not role
        or message.content != content
        or message.status is not status
        or message.context_through_sequence != context_through_sequence
    ):
        raise MessageIdentityConflictError(str(message.id))


async def _keep_unsaid(connection: object, message_id: UUID, content: str) -> None:
    # The answer itself keeps only what was said, so TA never remembers the rest
    # as said. This copy lets the chat page show it struck through, and lets TA
    # later pick it up as something left unfinished.
    if not content.strip():
        return
    await cast(asyncpg.Connection, connection).execute(
        "INSERT INTO message_unsaid (message_id, content) VALUES ($1, $2) "
        "ON CONFLICT (message_id) DO NOTHING",
        message_id,
        content[:32000],
    )


def _message_from_record(record: Mapping[str, object]) -> Message:
    return Message(
        id=cast(UUID, record["id"]),
        fish_id=cast(UUID, record["fish_id"]),
        session_id=cast(UUID, record["session_id"]),
        sequence=cast(int, record["sequence"]),
        role=Role(cast(str, record["role"])),
        content=cast(str, record["content"]),
        status=MessageStatus(cast(str, record["status"])),
        created_at=cast(datetime, record["created_at"]),
        context_through_sequence=cast(
            int | None,
            record["context_through_sequence"],
        ),
        consumed_offset=cast(int | None, record["consumed_offset"]),
        finished_at=cast(datetime | None, record["finished_at"]),
        history_token_count=cast(int, record["history_token_count"]),
        failure=cast(str | None, record.get("failure")),
    )


def _fish_summary_from_record(record: Mapping[str, object]) -> FishSummary:
    return FishSummary(
        fish_id=cast(UUID, record["fish_id"]),
        last_message=cast(str, record["last_message"]),
        last_active_at=cast(datetime | None, record["last_active_at"]),
        message_count=cast(int, record["message_count"]),
        is_assistant=cast(bool, record["is_assistant"]),
        name=cast(str, record.get("name", "")),
        role=cast(str, record.get("role", "")),
        evaluation=cast(bool, record.get("evaluation", False)),
    )
