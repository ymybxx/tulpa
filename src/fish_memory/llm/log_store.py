from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from fish_memory.clock import Clock, SystemClock

LOG_CLEANUP_INTERVAL_SECONDS = 15 * 60


def token_counts(response: object) -> tuple[int | None, int | None]:
    """Input and output tokens a provider reported, streamed or not."""
    if not isinstance(response, dict):
        return None, None
    usage = response.get("usage")
    body = response.get("body")
    if not isinstance(usage, dict) and isinstance(body, dict):
        usage = body.get("usage")
    if not isinstance(usage, dict):
        return None, None

    def count(*keys: str) -> int | None:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return count("prompt_tokens", "input_tokens"), count(
        "completion_tokens", "output_tokens"
    )


class LlmLogStore:
    def __init__(
        self,
        pool: Any,
        *,
        retention_days: Callable[[], int] = lambda: 7,
        clock: Clock | None = None,
    ):
        self.pool = pool
        self._retention_days = retention_days
        self._clock = clock or SystemClock()
        # A completion may arrive after expiry removed the running record.
        # Check age while holding the same lock as cleanup so it cannot restore
        # an expired request (including its images) with a stale cutoff.
        self._retention_lock = asyncio.Lock()

    def _cutoff(self) -> datetime:
        return self._clock.now() - timedelta(days=self._retention_days())

    async def process_once(self) -> bool:
        """Expire whole calls, independently of traffic, via the app supervisor."""
        async with self._retention_lock:
            await self.pool.execute(
                "DELETE FROM llm_call_logs WHERE started_at < $1", self._cutoff()
            )
        # Cleanup has no work queue to drain: always wait for the next interval.
        return False

    async def save(self, value: dict[str, Any]) -> None:
        async with self._retention_lock:
            if value["started_at"] < self._cutoff():
                return
            await self._save(value)

    async def _save(self, value: dict[str, Any]) -> None:
        await self.pool.execute(
            """INSERT INTO llm_call_logs
            (id,call_type,model,request_url,status,started_at,ended_at,
             duration_ms,http_status,error,source,request,response,
             input_tokens,output_tokens,purpose)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13::jsonb,
                    $14,$15,$16)
            ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status,
            ended_at=EXCLUDED.ended_at, duration_ms=EXCLUDED.duration_ms,
            http_status=EXCLUDED.http_status,error=EXCLUDED.error,
            response=EXCLUDED.response,input_tokens=EXCLUDED.input_tokens,
            output_tokens=EXCLUDED.output_tokens""",
            UUID(value["id"]),
            value["call_type"],
            value["model"],
            value["request_url"],
            value["status"],
            value["started_at"],
            value["ended_at"],
            value["duration_ms"],
            value["http_status"],
            value["error"],
            value["source"],
            json.dumps(value["request"], ensure_ascii=False),
            json.dumps(value["response"], ensure_ascii=False),
            *token_counts(value["response"]),
            value.get("purpose"),
        )

    async def usage(self, day_start: datetime, now: datetime) -> list[dict[str, Any]]:
        """Calls and tokens by call type and purpose: today, the last seven
        days, the last hour, and how many ran in the last ten minutes."""
        rows = await self.pool.fetch(
            """SELECT call_type, COALESCE(purpose, '') AS purpose,
                count(*) FILTER (WHERE started_at >= $1) AS today_calls,
                COALESCE(sum(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0))
                    FILTER (WHERE started_at >= $1), 0) AS today_tokens,
                count(*) AS week_calls,
                COALESCE(sum(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0)
                    AS week_tokens,
                count(*) FILTER (WHERE started_at >= $3) AS hour_calls,
                COALESCE(sum(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0))
                    FILTER (WHERE started_at >= $3), 0) AS hour_tokens,
                count(*) FILTER (WHERE started_at >= $4) AS recent_calls
            FROM llm_call_logs
            WHERE started_at >= $2
            GROUP BY call_type, COALESCE(purpose, '')""",
            day_start,
            day_start - timedelta(days=6),
            now - timedelta(hours=1),
            now - timedelta(minutes=10),
        )
        return [{key: row[key] for key in row.keys()} for row in rows]

    async def list_calls(
        self,
        *,
        call_type: str | None = None,
        status: str | None = None,
        before: int | None = None,
        limit: int = 40,
    ) -> dict[str, Any]:
        rows = await self.pool.fetch(
            """SELECT id,sequence,call_type,model,status,started_at,ended_at,
            duration_ms,http_status,error,source
            FROM llm_call_logs WHERE ($1::text IS NULL OR call_type=$1)
            AND ($2::text IS NULL OR status=$2) AND ($3::bigint IS NULL OR sequence<$3)
            ORDER BY sequence DESC LIMIT $4""",
            call_type,
            status,
            before,
            limit + 1,
        )
        return {
            "items": [dict(r) for r in rows[:limit]],
            "next_cursor": rows[limit - 1]["sequence"] if len(rows) > limit else None,
        }

    async def types(self) -> list[str]:
        rows = await self.pool.fetch(
            "SELECT DISTINCT call_type FROM llm_call_logs ORDER BY call_type"
        )
        return [row["call_type"] for row in rows]

    async def get(self, identifier: UUID) -> dict[str, Any] | None:
        row = await self.pool.fetchrow(
            "SELECT * FROM llm_call_logs WHERE id=$1", identifier
        )
        if row is None:
            return None
        result = dict(row)
        for field in ("request", "response"):
            if isinstance(result[field], str):
                result[field] = json.loads(result[field])
        return result

    async def recover(self) -> None:
        await self.pool.execute(
            "UPDATE llm_call_logs SET status='cancelled',"
            "error='应用在调用完成前退出' WHERE status='running'"
        )
