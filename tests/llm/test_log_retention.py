import asyncio
import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from fish_memory.background.supervisor import BackgroundTaskSupervisor
from fish_memory.clock import MutableClock
from fish_memory.llm.log_store import LlmLogStore


class Pool:
    def __init__(self):
        self.writes = []

    async def execute(self, sql, *args):
        self.writes.append((sql, args))


def record(at):
    return {
        "id": str(uuid4()),
        "call_type": "chat",
        "model": "fixture",
        "request_url": "https://fixture.invalid/v1/chat/completions",
        "status": "running",
        "started_at": at,
        "ended_at": None,
        "duration_ms": None,
        "http_status": None,
        "error": None,
        "source": "host",
        "request": {"image_url": "data:image/jpeg;base64,c3ludGhldGlj"},
        "response": None,
    }


@pytest.mark.parametrize("days", [1, 7, 30])
@pytest.mark.asyncio
async def test_cutoff_uses_elapsed_utc_days_and_keeps_boundary_images(days):
    # Crosses a DST transition; elapsed days must not use local calendar dates.
    now = datetime(2026, 11, 2, 1, 30, tzinfo=timezone(timedelta(hours=-6)))
    clock = MutableClock(now)
    pool = Pool()
    store = LlmLogStore(pool, retention_days=lambda: days, clock=clock)
    boundary = now.astimezone(UTC) - timedelta(days=days)
    kept = record(boundary.astimezone(timezone(timedelta(hours=9))))
    await store.save(kept)
    assert json.loads(pool.writes[0][1][11]) == kept["request"]
    await store.save(record(boundary - timedelta(microseconds=1)))
    assert len(pool.writes) == 1
    assert await store.process_once() is False
    assert pool.writes[-1] == (
        "DELETE FROM llm_call_logs WHERE started_at < $1",
        (boundary,),
    )


@pytest.mark.asyncio
async def test_completion_waiting_for_cleanup_cannot_restore_expired_images():
    deleting, release = asyncio.Event(), asyncio.Event()

    class DelayedDelete(Pool):
        async def execute(self, sql, *args):
            if sql.startswith("DELETE"):
                deleting.set()
                await release.wait()
            await super().execute(sql, *args)

    clock = MutableClock(datetime(2026, 10, 1, tzinfo=UTC))
    pool = DelayedDelete()
    store = LlmLogStore(pool, clock=clock)
    running = record(clock.now())
    await store.save(running)
    clock.set(clock.now() + timedelta(days=8))
    cleanup = asyncio.create_task(store.process_once())
    await deleting.wait()
    completion = asyncio.create_task(
        store.save({**running, "status": "completed", "response": {"text": "late"}})
    )
    await asyncio.sleep(0)
    assert not completion.done()
    release.set()
    await asyncio.gather(cleanup, completion)
    assert len(pool.writes) == 2
    assert pool.writes[-1][0].startswith("DELETE")


@pytest.mark.asyncio
async def test_supervisor_cleans_on_start_and_while_idle_using_hot_setting():
    ticks = asyncio.Queue()

    class ObservedPool(Pool):
        async def execute(self, sql, *args):
            await super().execute(sql, *args)
            ticks.put_nowait(args[0])

    clock = MutableClock(datetime(2026, 10, 1, tzinfo=UTC))
    days = 30
    pool = ObservedPool()
    store = LlmLogStore(pool, retention_days=lambda: days, clock=clock)
    supervisor = BackgroundTaskSupervisor(
        store, name="test-log-expiry", poll_interval_seconds=0.01
    )
    supervisor.start()
    try:
        assert await asyncio.wait_for(ticks.get(), 1) == clock.now() - timedelta(
            days=30
        )
        days = 1
        clock.set(clock.now() + timedelta(minutes=15))
        assert await asyncio.wait_for(ticks.get(), 1) == clock.now() - timedelta(days=1)
    finally:
        await supervisor.close()
    writes = len(pool.writes)
    await asyncio.sleep(0.02)
    assert len(pool.writes) == writes
