from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.agent.models import (
    AgentActivationPolicy,
    AgentInteractionKind,
    AgentMessageSubjectKind,
)
from fish_memory.background_messages.models import (
    BackgroundAssistantMessage,
    BackgroundMessageInterruptPolicy,
    BackgroundMessageReplyContext,
    BackgroundMessageSource,
)
from fish_memory.db.messages import MessagesStore
from fish_memory.db.sessions import SessionNotFoundError
from fish_memory.models import Message
from fish_memory.pet.types import PetEmotion

_EMOTIONS = frozenset({"neutral", "angry", "happy", "shy", "sad"})
_MAX_SUPERSESSION_KEY_CHARS = 200
# A connected Companion settles the exact played prefix at expiry. Delayed
# cleanup is only the offline safety net; deleting at expires_at could race an
# active TTS and lose text the user actually heard.
_EXPIRY_CLEANUP_GRACE = timedelta(minutes=5)


class BackgroundMessageConflictError(RuntimeError):
    pass


class ObservationNotFoundError(LookupError):
    pass


class BackgroundMessageStore:
    """Persistence for Assistant messages delivered outside foreground turns.

    The delivery record is metadata only. The associated ``messages`` row remains
    behind the existing ``waiting_consumption`` barrier until playback reports a
    final offset.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        messages_store: MessagesStore | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._pool = pool
        self._messages = messages_store or MessagesStore(pool)
        self._now = now

    async def insert_agent_delivery_on_connection(
        self,
        connection: asyncpg.Connection,
        message_id: UUID,
        *,
        emotion: str | PetEmotion = "neutral",
    ) -> None:
        normalized_emotion = _emotion(emotion)
        row = await connection.fetchrow(
            """
            INSERT INTO background_assistant_deliveries (
                message_id,
                source,
                emotion,
                interrupt_policy
            )
            VALUES ($1, 'agent', $2, 'resume')
            ON CONFLICT (message_id) DO NOTHING
            RETURNING message_id
            """,
            message_id,
            normalized_emotion,
        )
        if row is not None:
            return
        existing = await connection.fetchrow(
            """
            SELECT source, emotion, interrupt_policy, expires_at,
                   supersession_key, source_observation_id
            FROM background_assistant_deliveries
            WHERE message_id = $1
            """,
            message_id,
        )
        if (
            existing is None
            or existing["source"] != BackgroundMessageSource.AGENT.value
            or existing["emotion"] != normalized_emotion
            or existing["interrupt_policy"]
            != BackgroundMessageInterruptPolicy.RESUME.value
            or existing["expires_at"] is not None
            or existing["supersession_key"] is not None
            or existing["source_observation_id"] is not None
        ):
            raise BackgroundMessageConflictError(
                "message already has incompatible background delivery metadata"
            )

    async def has_delivery(self, message_id: UUID) -> bool:
        """A persisted receipt for idempotent external notifications."""
        async with self._pool.acquire() as connection:
            return bool(
                await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM background_assistant_deliveries "
                    "WHERE message_id=$1)",
                    message_id,
                )
            )

    async def materialize_proactive_observation(
        self,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        emotion: str | PetEmotion,
        expires_at: datetime | None,
        supersession_key: str,
        source_observation_id: UUID | None = None,
        created_at: datetime | None = None,
        message_id: UUID | None = None,
        source: BackgroundMessageSource = BackgroundMessageSource.PROACTIVE_OBSERVATION,
    ) -> Message:
        if source not in {
            BackgroundMessageSource.PROACTIVE_OBSERVATION,
            BackgroundMessageSource.PLUGIN,
        }:
            raise ValueError("unsupported live background source")
        normalized_emotion = _emotion(emotion)
        normalized_key = _supersession_key(supersession_key)
        # A live expression is owned by its runtime session, not a wall-clock TTL.
        # Dated observations retain their existing explicit expiry contract.
        if expires_at is not None and expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        effective_created_at = created_at or self._now()
        if effective_created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if expires_at is not None and expires_at <= effective_created_at:
            raise ValueError("expires_at must be later than created_at")

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                session = await connection.fetchrow(
                    """
                    SELECT session.id
                    FROM fish_sessions AS session
                    JOIN fishes AS fish ON fish.id = session.fish_id
                    WHERE session.id = $1
                      AND session.fish_id = $2
                      AND fish.deleted_at IS NULL
                    FOR UPDATE OF session
                    """,
                    session_id,
                    fish_id,
                )
                if session is None:
                    raise SessionNotFoundError(str(session_id))
                if message_id is not None and normalized_key.startswith(
                    "codex-completion:"
                ):
                    existing = await self._messages.get_message_on_connection(
                        connection, message_id
                    )
                    if existing is not None:
                        if (existing.fish_id, existing.session_id) != (
                            fish_id,
                            session_id,
                        ):
                            raise BackgroundMessageConflictError(
                                "notification recipient changed"
                            )
                        return existing
                if source_observation_id is not None:
                    observation = await connection.fetchrow(
                        """
                        SELECT id
                        FROM observations
                        WHERE id = $1 AND fish_id = $2
                        FOR UPDATE
                        """,
                        source_observation_id,
                        fish_id,
                    )
                    if observation is None:
                        raise ObservationNotFoundError(str(source_observation_id))
                    existing_id = await connection.fetchval(
                        """
                        SELECT delivery.message_id
                        FROM background_assistant_deliveries AS delivery
                        JOIN messages AS message
                          ON message.id = delivery.message_id
                        WHERE delivery.source = $3
                          AND delivery.source_observation_id = $1
                          AND message.fish_id = $2
                        """,
                        source_observation_id,
                        fish_id,
                        source.value,
                    )
                    if existing_id is not None:
                        existing = await self._messages.get_message_on_connection(
                            connection,
                            cast(UUID, existing_id),
                        )
                        if existing is None:
                            raise BackgroundMessageConflictError(
                                "proactive delivery lost its Message"
                            )
                        return existing

                context_through_sequence = cast(
                    int,
                    await connection.fetchval(
                        """
                        SELECT COALESCE(max(sequence), 0)
                        FROM messages
                        WHERE fish_id = $1
                          AND session_id = $2
                          AND fact_extraction_status <> 'waiting_consumption'
                        """,
                        fish_id,
                        session_id,
                    ),
                )
                message = await self._messages.insert_background_message_on_connection(
                    connection,
                    fish_id,
                    session_id,
                    context_through_sequence,
                    content,
                    created_at=effective_created_at,
                    message_id=message_id or uuid4(),
                )
                await connection.execute(
                    """
                    INSERT INTO background_assistant_deliveries (
                        message_id,
                        source,
                        emotion,
                        interrupt_policy,
                        expires_at,
                        supersession_key,
                        source_observation_id
                    )
                    VALUES (
                        $1, $6, $2, 'drop', $3, $4, $5
                    )
                    """,
                    message.id,
                    normalized_emotion,
                    expires_at,
                    normalized_key,
                    source_observation_id,
                    source.value,
                )
                return message

    async def materialize_proactive(
        self,
        fish_id: UUID,
        session_id: UUID,
        observation_id: UUID,
        content: str,
        emotion: str | PetEmotion,
        *,
        expires_at: datetime,
        supersession_key: str,
    ) -> Message | None:
        """Coordinator-facing alias with one idempotency key per Observation."""
        return await self.materialize_proactive_observation(
            fish_id,
            session_id,
            content,
            emotion=emotion,
            expires_at=expires_at,
            supersession_key=supersession_key,
            source_observation_id=observation_id,
        )

    async def cancel_proactive_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        *,
        source: BackgroundMessageSource = BackgroundMessageSource.PROACTIVE_OBSERVATION,
    ) -> Message | None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                rows = await self._lock_cancellable(
                    connection,
                    fish_id,
                    session_id,
                    message_id=message_id,
                    source=source.value,
                )
                if not rows:
                    return None
                return await self._cancel_locked(connection, rows[0], self._now())

    async def cancel_superseded(
        self,
        fish_id: UUID,
        session_id: UUID,
        supersession_key: str,
    ) -> int:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                return await self._cancel_matching_locked(
                    connection,
                    fish_id,
                    session_id,
                    supersession_key=_supersession_key(supersession_key),
                    cancelled_at=self._now(),
                )

    async def cancel_proactive_for_fish(self, fish_id: UUID) -> int:
        """Cancel pending deliveries after that Fish's consumer is quiesced."""
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                rows = await self._lock_cancellable(connection, fish_id, None)
                cancelled_at = self._now()
                for row in rows:
                    await self._cancel_locked(connection, row, cancelled_at)
                return len(rows)

    async def expire_proactive_messages(
        self,
        *,
        fish_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> int:
        now = self._now()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                rows = await self._lock_cancellable(
                    connection,
                    fish_id,
                    session_id,
                    expired_through=now - _EXPIRY_CLEANUP_GRACE,
                )
                for row in rows:
                    await self._cancel_locked(connection, row, now)
                return len(rows)

    async def cancel_orphaned_proactive_streams(
        self,
        *,
        source: BackgroundMessageSource = BackgroundMessageSource.PROACTIVE_OBSERVATION,
    ) -> int:
        """Startup only: stale live streams and Codex notices must not resume."""
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                rows = await self._lock_cancellable(
                    connection,
                    None,
                    None,
                    runtime_owned_only=True,
                    source=source.value,
                )
                now = self._now()
                for row in rows:
                    await self._cancel_locked(connection, row, now)
                return len(rows)

    async def update_proactive_content(
        self, message_id: UUID, fish_id: UUID, session_id: UUID, content: str
    ) -> bool:
        """Append the generated prefix only while playback owns consumption."""
        result = await self._pool.execute(
            """
            UPDATE messages AS message SET content = $4
            FROM background_assistant_deliveries AS delivery
            WHERE message.id = $1 AND message.fish_id = $2
              AND message.session_id = $3 AND delivery.message_id = message.id
              AND delivery.source IN ('proactive_observation', 'plugin')
              AND delivery.cancelled_at IS NULL
              AND message.fact_extraction_status = 'waiting_consumption'
              AND left($4, char_length(message.content)) = message.content
            """,
            message_id,
            fish_id,
            session_id,
            content,
        )
        return result == "UPDATE 1"

    async def finalize_consumption(
        self,
        message_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> Message | None:
        if consumed_offset < -1:
            raise ValueError("consumed offset must be -1 or non-negative")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT delivery.cancelled_at,
                           message.fish_id,
                           message.session_id
                    FROM background_assistant_deliveries AS delivery
                    JOIN messages AS message ON message.id = delivery.message_id
                    WHERE delivery.message_id = $1
                    FOR UPDATE OF delivery, message
                    """,
                    message_id,
                )
                if (
                    row is None
                    or row["fish_id"] != fish_id
                    or row["session_id"] != session_id
                ):
                    return None
                if row["cancelled_at"] is not None:
                    return await self._messages.get_message_on_connection(
                        connection,
                        message_id,
                    )
                finalize = (
                    self._messages.finalize_background_message_consumption_on_connection
                )
                return await finalize(
                    connection,
                    message_id,
                    fish_id,
                    consumed_offset,
                    interrupted,
                )

    async def list_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        include_incomplete: bool,
        limit: int,
        *,
        incomplete_after_sequence: int = 0,
    ) -> Sequence[BackgroundAssistantMessage]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if incomplete_after_sequence < 0:
            raise ValueError("incomplete_after_sequence must be non-negative")
        if limit < 1:
            raise ValueError("limit must be positive")
        await self.expire_proactive_messages(fish_id=fish_id, session_id=session_id)
        rows = await self._pool.fetch(
            _DELIVERY_QUERY
            + """
              AND (
                  message.sequence > $3
                  OR (
                      $4
                      AND message.sequence > $5
                      AND message.sequence <= $3
                      AND (
                          message.fact_extraction_status = 'waiting_consumption'
                          OR (
                              message.consumed_offset <> -1
                              AND message.consumed_offset
                                  < char_length(message.content)
                          )
                      )
                  )
              )
            ORDER BY CASE
                         WHEN $4
                          AND message.sequence > $5
                          AND message.sequence <= $3
                          AND (
                              message.fact_extraction_status = 'waiting_consumption'
                              OR (
                                  message.consumed_offset <> -1
                                  AND message.consumed_offset
                                      < char_length(message.content)
                              )
                          )
                         THEN 0
                         ELSE 1
                     END,
                     message.sequence
            LIMIT $6
            """,
            fish_id,
            session_id,
            after_sequence,
            include_incomplete,
            incomplete_after_sequence,
            limit,
        )
        return tuple(_message_from_record(row) for row in rows)

    async def list_incomplete_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        through_sequence: int,
        limit: int,
    ) -> Sequence[BackgroundAssistantMessage]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if through_sequence < 0:
            raise ValueError("through_sequence must be non-negative")
        if after_sequence > through_sequence:
            raise ValueError("after_sequence must not exceed through_sequence")
        if limit < 1:
            raise ValueError("limit must be positive")
        await self.expire_proactive_messages(fish_id=fish_id, session_id=session_id)
        rows = await self._pool.fetch(
            _DELIVERY_QUERY
            + """
              AND message.sequence > $3
              AND message.sequence <= $4
              AND (
                  message.fact_extraction_status = 'waiting_consumption'
                  OR (
                      message.consumed_offset <> -1
                      AND message.consumed_offset < char_length(message.content)
                  )
              )
            ORDER BY message.sequence
            LIMIT $5
            """,
            fish_id,
            session_id,
            after_sequence,
            through_sequence,
            limit,
        )
        return tuple(_message_from_record(row) for row in rows)

    async def _cancel_matching_locked(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID,
        session_id: UUID,
        *,
        supersession_key: str,
        cancelled_at: datetime,
    ) -> int:
        rows = await self._lock_cancellable(
            connection,
            fish_id,
            session_id,
            supersession_key=supersession_key,
        )
        for row in rows:
            await self._cancel_locked(connection, row, cancelled_at)
        return len(rows)

    async def _lock_cancellable(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID | None,
        session_id: UUID | None,
        *,
        message_id: UUID | None = None,
        supersession_key: str | None = None,
        expired_through: datetime | None = None,
        runtime_owned_only: bool = False,
        source: str = "proactive_observation",
    ) -> Sequence[Mapping[str, object]]:
        return await connection.fetch(
            """
            SELECT delivery.message_id,
                   message.fish_id,
                   message.session_id,
                   message.consumed_offset
            FROM background_assistant_deliveries AS delivery
            JOIN messages AS message ON message.id = delivery.message_id
            WHERE delivery.source = $7
              AND delivery.cancelled_at IS NULL
              AND message.fact_extraction_status = 'waiting_consumption'
              AND ($1::uuid IS NULL OR message.fish_id = $1)
              AND ($2::uuid IS NULL OR message.session_id = $2)
              AND ($3::uuid IS NULL OR delivery.message_id = $3)
              AND ($4::text IS NULL OR delivery.supersession_key = $4)
              AND ($5::timestamptz IS NULL OR delivery.expires_at <= $5)
              AND (NOT $6::boolean OR delivery.expires_at IS NULL
                   OR delivery.supersession_key LIKE 'codex-completion:%')
            ORDER BY message.sequence
            FOR UPDATE OF delivery, message
            """,
            fish_id,
            session_id,
            message_id,
            supersession_key,
            expired_through,
            runtime_owned_only,
            source,
        )

    async def _cancel_locked(
        self,
        connection: asyncpg.Connection,
        row: Mapping[str, object],
        cancelled_at: datetime,
    ) -> Message | None:
        consumed_offset = max(0, cast(int, row["consumed_offset"]))
        if consumed_offset == 0:
            deleted = await connection.fetchrow(
                """
                DELETE FROM messages
                WHERE id = $1
                  AND fish_id = $2
                  AND consumed_offset = 0
                  AND fact_extraction_status = 'waiting_consumption'
                RETURNING session_id
                """,
                row["message_id"],
                row["fish_id"],
            )
            if deleted is None:
                raise BackgroundMessageConflictError(
                    "unplayed proactive Message could not be cancelled"
                )
            await connection.execute(
                """
                UPDATE fish_sessions AS session
                SET last_message_at = (
                    SELECT max(message.created_at)
                    FROM messages AS message
                    WHERE message.session_id = session.id
                )
                WHERE session.id = $1
                """,
                deleted["session_id"],
            )
            return None
        await connection.execute(
            """
            UPDATE background_assistant_deliveries
            SET cancelled_at = $2
            WHERE message_id = $1 AND cancelled_at IS NULL
            """,
            row["message_id"],
            cancelled_at,
        )
        message = (
            await self._messages.finalize_background_message_consumption_on_connection(
                connection,
                cast(UUID, row["message_id"]),
                cast(UUID, row["fish_id"]),
                consumed_offset,
                True,
            )
        )
        if message is None:
            raise BackgroundMessageConflictError(
                "locked background Assistant message disappeared"
            )
        return message


_DELIVERY_QUERY = """
            SELECT delivery.source,
                   delivery.emotion,
                   delivery.interrupt_policy,
                   delivery.expires_at,
                   delivery.supersession_key,
                   message.id AS message_id,
                   message.sequence,
                   message.session_id,
                   message.content,
                   message.consumed_offset,
                   result_run.id AS result_run_id,
                   interaction.id AS interaction_id,
                   interaction.kind AS interaction_kind
            FROM background_assistant_deliveries AS delivery
            JOIN messages AS message ON message.id = delivery.message_id
            LEFT JOIN agent_runs AS result_run
              ON result_run.result_message_id = message.id
            LEFT JOIN agent_interactions AS interaction
              ON interaction.request_message_id = message.id
            WHERE message.fish_id = $1
              AND message.session_id = $2
              AND message.role = 'assistant'
              AND delivery.cancelled_at IS NULL
              AND (delivery.expires_at IS NULL OR delivery.expires_at > now())
              AND char_length(message.content) > 0
              AND (
                  (
                      delivery.source = 'agent'
                      AND ((result_run.id IS NOT NULL)::int
                           + (interaction.id IS NOT NULL)::int) = 1
                  )
                  OR (
                      delivery.source IN ('proactive_observation', 'plugin')
                      AND result_run.id IS NULL
                      AND interaction.id IS NULL
                  )
              )
"""


def _message_from_record(record: Mapping[str, object]) -> BackgroundAssistantMessage:
    source = BackgroundMessageSource(cast(str, record["source"]))
    reply_context: BackgroundMessageReplyContext | None = None
    if source is BackgroundMessageSource.AGENT:
        interaction_id = cast(UUID | None, record["interaction_id"])
        if interaction_id is not None:
            activation_policy = (
                AgentActivationPolicy.PLAYBACK_COMPLETE
                if record["interaction_kind"] == AgentInteractionKind.APPROVAL
                else AgentActivationPolicy.PLAYBACK_START
            )
            reply_context = BackgroundMessageReplyContext(
                AgentMessageSubjectKind.INTERACTION,
                interaction_id,
                activation_policy,
            )
        else:
            reply_context = BackgroundMessageReplyContext(
                AgentMessageSubjectKind.RUN_RESULT,
                cast(UUID, record["result_run_id"]),
                AgentActivationPolicy.PLAYBACK_START,
            )
    return BackgroundAssistantMessage(
        source=source,
        message_id=cast(UUID, record["message_id"]),
        sequence=cast(int, record["sequence"]),
        session_id=cast(UUID, record["session_id"]),
        content=cast(str, record["content"]),
        consumed_offset=cast(int, record["consumed_offset"]),
        emotion=cast(str, record["emotion"]),
        interrupt_policy=BackgroundMessageInterruptPolicy(
            cast(str, record["interrupt_policy"])
        ),
        expires_at=cast(datetime | None, record["expires_at"]),
        supersession_key=cast(str | None, record["supersession_key"]),
        reply_context=reply_context,
    )


def _emotion(value: str | PetEmotion) -> str:
    normalized = value.value if isinstance(value, PetEmotion) else value
    if normalized not in _EMOTIONS:
        raise ValueError("emotion is unsupported")
    return normalized


def _supersession_key(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > _MAX_SUPERSESSION_KEY_CHARS:
        raise ValueError("supersession_key must contain 1 to 200 characters")
    if "\x00" in normalized:
        raise ValueError("supersession_key contains a null byte")
    return normalized


__all__ = [
    "BackgroundMessageConflictError",
    "BackgroundMessageStore",
    "ObservationNotFoundError",
]
