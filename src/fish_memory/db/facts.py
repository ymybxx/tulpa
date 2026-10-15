import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.db.entities import lock_entity_index, replace_fact_entities
from fish_memory.db.fact_frontier import fact_frontier_ctes
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.facts.embedding import build_fact_embedding_text
from fish_memory.facts.models import (
    BackgroundTaskProgress,
    ExtractedFact,
    PendingFactMessage,
)
from fish_memory.models import Message, MessageStatus, Role


class FactStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        background_owner: UUID | None = None,
    ) -> None:
        self._pool = pool
        self._background_owner = background_owner

    async def list_pending_messages(
        self,
        limit: int,
        batch_size: int = 1,
    ) -> list[PendingFactMessage]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        rows = await self._pool.fetch(
            f"""
            WITH {fact_frontier_ctes(owner_parameter="$3")},
            candidate_session AS (
                SELECT
                    bounds.session_id,
                    bounds.fish_id,
                    bounds.target_sequence,
                    COALESCE(progress.through_sequence, 0) AS through_sequence,
                    progress.updated_at AS progress_updated_at
                FROM bounds
                LEFT JOIN session_background_task_progress AS progress
                    ON progress.session_id = bounds.session_id
                   AND progress.task_name = 'fact_extraction'
                WHERE EXISTS (
                    SELECT 1
                    FROM messages AS pending_message
                    WHERE pending_message.session_id = bounds.session_id
                      AND pending_message.sequence
                          > COALESCE(progress.through_sequence, 0)
                      AND pending_message.sequence <= bounds.target_sequence
                      AND pending_message.fact_extraction_status = 'pending'
                )
                ORDER BY progress.updated_at NULLS FIRST, bounds.session_id
                LIMIT $1
            )
            SELECT
                candidate.id,
                candidate.fish_id,
                candidate.session_id,
                candidate.sequence,
                candidate.role,
                candidate.content,
                candidate.status,
                candidate.created_at,
                candidate.context_through_sequence,
                candidate.consumed_offset,
                candidate.finished_at,
                candidate.fact_extraction_attempt_count,
                context_message.id AS context_id,
                context_message.fish_id AS context_fish_id,
                context_message.session_id AS context_session_id,
                context_message.sequence AS context_sequence,
                context_message.role AS context_role,
                context_message.content AS context_content,
                context_message.status AS context_status,
                context_message.created_at AS context_created_at,
                context_message.context_through_sequence
                    AS context_context_through_sequence,
                context_message.consumed_offset AS context_consumed_offset,
                context_message.finished_at AS context_finished_at,
                plugin_event.payload AS external_event,
                candidate_session.target_sequence AS target_sequence
            FROM candidate_session
            JOIN LATERAL (
                SELECT
                    message.id,
                    message.fish_id,
                    message.session_id,
                    message.sequence,
                    message.role,
                    message.content,
                    message.status,
                    message.created_at,
                    message.context_through_sequence,
                    message.consumed_offset,
                    message.finished_at,
                    message.fact_extraction_attempt_count
                FROM messages AS message
                WHERE message.session_id = candidate_session.session_id
                  AND message.sequence
                      > candidate_session.through_sequence
                  AND message.sequence <= candidate_session.target_sequence
                  AND message.fact_extraction_status = 'pending'
                ORDER BY message.sequence
                LIMIT $2
            ) AS candidate ON TRUE
            LEFT JOIN LATERAL (
                SELECT
                    message.id,
                    message.fish_id,
                    message.session_id,
                    message.sequence,
                    message.role,
                    message.content,
                    message.status,
                    message.created_at,
                    message.context_through_sequence,
                    message.consumed_offset,
                    message.finished_at
                FROM messages AS message
                WHERE message.session_id = candidate.session_id
                  AND message.sequence < candidate.sequence
                  AND message.role <> candidate.role
                ORDER BY message.sequence DESC
                LIMIT 1
            ) AS context_message ON TRUE
            LEFT JOIN background_assistant_deliveries AS plugin_delivery
              ON plugin_delivery.message_id = candidate.id
             AND plugin_delivery.source = 'plugin'
            LEFT JOIN observations AS plugin_event
              ON plugin_event.id = plugin_delivery.source_observation_id
            ORDER BY
                candidate_session.progress_updated_at NULLS FIRST,
                candidate_session.session_id,
                candidate.sequence
            """,
            limit,
            batch_size,
            self._background_owner,
        )
        return [_pending_message_from_record(row) for row in rows]

    async def persist_extraction(
        self,
        pending: PendingFactMessage,
        facts: Sequence[ExtractedFact],
        vectors: Sequence[Sequence[float]],
        embedding_space: EmbeddingSpace,
    ) -> None:
        if len(facts) != len(vectors):
            raise ValueError("fact and embedding count must match")
        message = pending.message
        rows = [
            (
                uuid4(),
                message.fish_id,
                message.session_id,
                message.id,
                fact_index,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                message.created_at,
                json.dumps(list(fact.entities), ensure_ascii=False),
                fact.structured.model_dump_json(),
            )
            for fact_index, fact in enumerate(facts)
        ]

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    UPDATE messages
                    SET fact_extraction_status = 'completed',
                        fact_extraction_error = NULL
                    WHERE id = $1
                    """,
                    message.id,
                )
                await connection.execute(
                    """
                    DELETE FROM facts
                    WHERE source_message_id = $1
                      AND fact_index >= $2
                    """,
                    message.id,
                    len(rows),
                )
                stored_fact_ids: list[UUID] = []
                for row in rows:
                    stored = await connection.fetchrow(
                        """
                        INSERT INTO facts (
                            id,
                            fish_id,
                            session_id,
                            source_message_id,
                            fact_index,
                            text,
                            occurred_start,
                            occurred_end,
                            mentioned_at,
                            entities,
                            structured
                        )
                        VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb,
                            $11::jsonb
                        )
                        ON CONFLICT (source_message_id, fact_index) DO UPDATE
                        SET fish_id = EXCLUDED.fish_id,
                            session_id = EXCLUDED.session_id,
                            text = EXCLUDED.text,
                            occurred_start = EXCLUDED.occurred_start,
                            occurred_end = EXCLUDED.occurred_end,
                            mentioned_at = EXCLUDED.mentioned_at,
                            entities = EXCLUDED.entities,
                            structured = EXCLUDED.structured
                        RETURNING id, fact_index
                        """,
                        *row,
                    )
                    if stored is None:
                        raise RuntimeError("database did not return persisted fact")
                    stored_fact_ids.append(cast(UUID, stored["id"]))
                if rows:
                    await lock_entity_index(connection, message.fish_id)
                    for fact, fact_id in zip(
                        facts,
                        stored_fact_ids,
                        strict=True,
                    ):
                        await replace_fact_entities(
                            connection,
                            message.fish_id,
                            fact_id,
                            fact.entities if message.role is Role.USER else (),
                        )
                    embedding_rows = [
                        (
                            fact_id,
                            message.fish_id,
                            embedding_space.provider,
                            embedding_space.model,
                            embedding_space.dimensions,
                            fact.embedding_text,
                            _text_hash(fact.embedding_text),
                            _vector_literal(
                                vector,
                                dimensions=embedding_space.dimensions,
                            ),
                        )
                        for fact, fact_id, vector in zip(
                            facts,
                            stored_fact_ids,
                            vectors,
                            strict=True,
                        )
                    ]
                    await connection.executemany(
                        """
                        INSERT INTO fact_embeddings (
                            fact_id,
                            fish_id,
                            provider,
                            model,
                            dimensions,
                            embedding_text,
                            text_hash,
                            embedding
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                        ON CONFLICT (fact_id) DO UPDATE
                        SET fish_id = EXCLUDED.fish_id,
                            provider = EXCLUDED.provider,
                            model = EXCLUDED.model,
                            dimensions = EXCLUDED.dimensions,
                            embedding_text = EXCLUDED.embedding_text,
                            text_hash = EXCLUDED.text_hash,
                            embedding = EXCLUDED.embedding,
                            created_at = now()
                        """,
                        embedding_rows,
                    )
                await connection.execute(
                    """
                    INSERT INTO session_background_task_progress (
                        session_id,
                        task_name,
                        through_sequence,
                        last_error
                    )
                    VALUES (
                        $1,
                        'fact_extraction',
                        LEAST(
                            $2,
                            COALESCE(
                                (
                                    SELECT min(sequence) - 1
                                    FROM messages
                                    WHERE session_id = $1
                                      AND fact_extraction_status =
                                          'waiting_consumption'
                                ),
                                $2
                            )
                        ),
                        NULL
                    )
                    ON CONFLICT (session_id, task_name) DO UPDATE
                    SET through_sequence = GREATEST(
                            session_background_task_progress.through_sequence,
                            EXCLUDED.through_sequence
                        ),
                        last_error = NULL,
                        updated_at = now()
                    """,
                    message.session_id,
                    message.sequence,
                )

    async def record_failure(
        self,
        pending: PendingFactMessage,
        error: str,
        *,
        terminal: bool,
    ) -> None:
        message = pending.message
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    UPDATE messages
                    SET fact_extraction_attempt_count =
                            fact_extraction_attempt_count + 1,
                        fact_extraction_error = $2,
                        fact_extraction_status = CASE
                            WHEN $3 THEN 'failed'
                            ELSE 'pending'
                        END
                    WHERE id = $1
                    """,
                    message.id,
                    error,
                    terminal,
                )
                if terminal:
                    await connection.execute(
                        """
                        INSERT INTO session_background_task_progress (
                            session_id,
                            task_name,
                            through_sequence,
                            last_error
                        )
                        VALUES (
                            $1,
                            'fact_extraction',
                            LEAST(
                                $2,
                                COALESCE(
                                    (
                                        SELECT min(sequence) - 1
                                        FROM messages
                                        WHERE session_id = $1
                                          AND fact_extraction_status =
                                              'waiting_consumption'
                                    ),
                                    $2
                                )
                            ),
                            NULL
                        )
                        ON CONFLICT (session_id, task_name) DO UPDATE
                        SET through_sequence = GREATEST(
                                session_background_task_progress.through_sequence,
                                EXCLUDED.through_sequence
                            ),
                            last_error = NULL,
                            updated_at = now()
                        """,
                        message.session_id,
                        message.sequence,
                    )

    async def get_progress(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> BackgroundTaskProgress:
        row = await self._pool.fetchrow(
            """
            WITH bounds AS (
                SELECT
                    max(message.sequence) AS max_sequence,
                    min(message.sequence) FILTER (
                        WHERE message.fact_extraction_status = 'waiting_consumption'
                    ) AS waiting_sequence
                FROM messages AS message
                WHERE message.fish_id = $1
                  AND message.session_id = $2
            ),
            target AS (
                SELECT COALESCE(max(message.sequence), 0) AS target_sequence
                FROM messages AS message
                CROSS JOIN bounds
                WHERE message.fish_id = $1
                  AND message.session_id = $2
                  AND message.sequence < bounds.max_sequence
                  AND (
                      bounds.waiting_sequence IS NULL
                      OR message.sequence < bounds.waiting_sequence
                  )
            )
            SELECT
                'fact_extraction' AS task_name,
                COALESCE(progress.through_sequence, 0) AS through_sequence,
                target.target_sequence,
                (
                    SELECT count(*)
                    FROM messages AS message
                    WHERE message.fish_id = $1
                      AND message.session_id = $2
                      AND message.sequence
                          > COALESCE(progress.through_sequence, 0)
                      AND message.sequence <= target.target_sequence
                      AND message.fact_extraction_status = 'pending'
                ) AS pending,
                progress.last_error,
                progress.updated_at
            FROM target
            LEFT JOIN session_background_task_progress AS progress
                ON progress.session_id = $2
               AND progress.task_name = 'fact_extraction'
            """,
            fish_id,
            session_id,
        )
        if row is None:
            raise RuntimeError("database did not return fact extraction progress")
        return _progress_from_record(row)


def _pending_message_from_record(
    record: Mapping[str, object],
) -> PendingFactMessage:
    return PendingFactMessage(
        message=Message(
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
        ),
        target_sequence=cast(int, record["target_sequence"]),
        fact_extraction_attempt_count=cast(
            int,
            record["fact_extraction_attempt_count"],
        ),
        previous_opposite_message=_context_message_from_record(record),
        external_event=_external_event_context(record.get("external_event")),
    )


def _external_event_context(raw: object) -> str | None:
    if raw is None:
        return None
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        return None
    return json.dumps(
        {
            **{k: value.get(k) for k in ("plugin_id", "type")},
            **(
                {"response_events": value.get("response_events")}
                if value.get("response_events")
                else {"data": value.get("data")}
            ),
        },
        ensure_ascii=False,
    )


def _embedding_text_from_record(record: Mapping[str, object]) -> str:
    raw_structured = record.get("structured")
    structured: Mapping[str, object]
    if isinstance(raw_structured, str):
        decoded = json.loads(raw_structured)
        structured = decoded if isinstance(decoded, Mapping) else {}
    elif isinstance(raw_structured, Mapping):
        structured = cast(Mapping[str, object], raw_structured)
    else:
        structured = {}

    return build_fact_embedding_text(
        text=cast(str, record["text"]),
        what=_string_value(structured.get("what")),
        where=_string_values(structured.get("where")),
        who=_string_values(structured.get("who")),
        entities=_string_values(record.get("entities")),
    )


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        decoded = json.loads(value)
        value = decoded
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _context_message_from_record(record: Mapping[str, object]) -> Message | None:
    context_id = record.get("context_id")
    if context_id is None:
        return None
    return Message(
        id=cast(UUID, context_id),
        fish_id=cast(UUID, record["context_fish_id"]),
        session_id=cast(UUID, record["context_session_id"]),
        sequence=cast(int, record["context_sequence"]),
        role=Role(cast(str, record["context_role"])),
        content=cast(str, record["context_content"]),
        status=MessageStatus(cast(str, record["context_status"])),
        created_at=cast(datetime, record["context_created_at"]),
        context_through_sequence=cast(
            int | None,
            record["context_context_through_sequence"],
        ),
        consumed_offset=cast(int | None, record["context_consumed_offset"]),
        finished_at=cast(datetime | None, record["context_finished_at"]),
    )


def _progress_from_record(
    record: Mapping[str, object],
) -> BackgroundTaskProgress:
    return BackgroundTaskProgress(
        task_name=cast(str, record["task_name"]),
        through_sequence=cast(int, record["through_sequence"]),
        target_sequence=cast(int, record["target_sequence"]),
        pending=cast(int, record["pending"]),
        last_error=cast(str | None, record["last_error"]),
        updated_at=cast(datetime | None, record["updated_at"]),
    )


def _vector_literal(vector: Sequence[float], *, dimensions: int) -> str:
    if len(vector) != dimensions:
        raise ValueError(
            f"embedding dimension {len(vector)} does not match expected {dimensions}"
        )
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding values must be finite")
    return "[" + ",".join(str(value) for value in values) + "]"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
