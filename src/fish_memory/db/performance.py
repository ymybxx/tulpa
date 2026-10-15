import json
import math
from collections.abc import Mapping, Sequence
from typing import cast
from uuid import UUID

import asyncpg

from fish_memory.performance.models import JsonValue, PerformanceStep

INSERT_STEPS_SQL = """
INSERT INTO message_performance_steps (
    message_id,
    step_name,
    step_index,
    started_offset_us,
    duration_us,
    input_payload,
    output_payload,
    error
)
VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8)
"""

LIST_STEPS_SQL = """
SELECT
    step_name,
    step_index,
    started_offset_us,
    duration_us,
    input_payload,
    output_payload,
    error
FROM message_performance_steps
WHERE message_id = $1
ORDER BY started_offset_us, id
"""

LIST_MESSAGE_STEPS_SQL = """
SELECT
    step.step_name,
    step.step_index,
    step.started_offset_us,
    step.duration_us,
    step.input_payload,
    step.output_payload,
    step.error
FROM message_performance_steps AS step
JOIN messages AS message ON message.id = step.message_id
WHERE step.message_id = $1
  AND message.fish_id = $2
ORDER BY step.started_offset_us, step.id
"""


class PerformanceStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert_steps(
        self,
        message_id: UUID,
        steps: Sequence[PerformanceStep],
    ) -> None:
        if not steps:
            return

        rows = [
            (
                message_id,
                step.name,
                step.index,
                step.started_offset_us,
                step.duration_us,
                _dump_payload(step.input_payload),
                _dump_payload(step.output_payload),
                step.error,
            )
            for step in steps
        ]
        async with self._pool.acquire() as connection:
            await connection.executemany(INSERT_STEPS_SQL, rows)

    async def list_steps(self, message_id: UUID) -> list[PerformanceStep]:
        rows = await self._pool.fetch(LIST_STEPS_SQL, message_id)
        return [_step_from_record(row) for row in rows]

    async def list_message_steps(
        self,
        fish_id: UUID,
        message_id: UUID,
    ) -> list[PerformanceStep]:
        rows = await self._pool.fetch(
            LIST_MESSAGE_STEPS_SQL,
            message_id,
            fish_id,
        )
        return [_step_from_record(row) for row in rows]

    async def replace_tts_metrics(
        self,
        fish_id: UUID,
        message_id: UUID,
        metrics: dict[str, JsonValue],
    ) -> bool:
        duration_us = _tts_duration_us(metrics)
        payload = _dump_payload(metrics)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                exists = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM messages
                        WHERE id = $1 AND fish_id = $2
                    )
                    """,
                    message_id,
                    fish_id,
                )
                if not exists:
                    return False
                await connection.execute(
                    """
                    DELETE FROM message_performance_steps
                    WHERE message_id = $1 AND step_name = 'tts.client'
                    """,
                    message_id,
                )
                await connection.execute(
                    """
                    INSERT INTO message_performance_steps (
                        message_id,
                        step_name,
                        step_index,
                        started_offset_us,
                        duration_us,
                        output_payload
                    )
                    VALUES ($1, 'tts.client', 0, 0, $2, $3::jsonb)
                    """,
                    message_id,
                    duration_us,
                    payload,
                )
        return True

    async def replace_pet_voice_metrics(
        self,
        fish_id: UUID,
        message_id: UUID,
        metrics: dict[str, JsonValue],
    ) -> bool:
        duration_us = _pet_voice_duration_us(metrics)
        payload = _dump_payload(metrics)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                exists = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM messages
                        WHERE id = $1 AND fish_id = $2
                    )
                    """,
                    message_id,
                    fish_id,
                )
                if not exists:
                    return False
                await connection.execute(
                    """
                    DELETE FROM message_performance_steps
                    WHERE message_id = $1 AND step_name = 'voice.pet'
                    """,
                    message_id,
                )
                await connection.execute(
                    """
                    INSERT INTO message_performance_steps (
                        message_id,
                        step_name,
                        step_index,
                        started_offset_us,
                        duration_us,
                        output_payload
                    )
                    VALUES ($1, 'voice.pet', 0, 0, $2, $3::jsonb)
                    """,
                    message_id,
                    duration_us,
                    payload,
                )
        return True


def _dump_payload(payload: JsonValue) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _tts_duration_us(metrics: dict[str, JsonValue]) -> int:
    value = metrics.get("totalMs")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value) or value < 0:
        return 0
    return round(value * 1000)


def _pet_voice_duration_us(metrics: dict[str, JsonValue]) -> int:
    native_ms = metrics.get("nativeMs")
    if not isinstance(native_ms, dict):
        return 0
    value = native_ms.get("endpointToPlayback")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value) or value < 0:
        return 0
    return round(value * 1000)


def _load_payload(value: object) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, str):
        return cast(JsonValue, json.loads(value))
    return cast(JsonValue, value)


def _step_from_record(record: Mapping[str, object]) -> PerformanceStep:
    return PerformanceStep(
        name=cast(str, record["step_name"]),
        index=cast(int, record["step_index"]),
        started_offset_us=cast(int, record["started_offset_us"]),
        duration_us=cast(int, record["duration_us"]),
        input_payload=_load_payload(record["input_payload"]),
        output_payload=_load_payload(record["output_payload"]),
        error=cast(str | None, record["error"]),
    )
