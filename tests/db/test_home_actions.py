from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.home_actions import HomeActionStore, _from_row
from fish_memory.home.models import HomeActionStatus


def test_decodes_jsonb_text_from_asyncpg() -> None:
    now = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
    row = {
        "id": UUID("00000000-0000-0000-0000-000000000001"),
        "sequence": 1,
        "event_sequence": 2,
        "fish_id": UUID("00000000-0000-0000-0000-000000000002"),
        "session_id": UUID("00000000-0000-0000-0000-000000000003"),
        "source_message_id": UUID("00000000-0000-0000-0000-000000000004"),
        "action": '{"entityId":"light.entry","operation":"turn_on"}',
        "status": "failed",
        "entity_id": "light.entry",
        "service_domain": "light",
        "service_name": "turn_on",
        "accepted": False,
        "confirmed": False,
        "result": '{"reason":"rejected"}',
        "error": "failed",
        "created_at": now,
        "updated_at": now,
        "finished_at": now,
    }

    run = _from_row(row)

    assert run.action == {"entityId": "light.entry", "operation": "turn_on"}
    assert run.result == {"reason": "rejected"}
    assert run.event_sequence == 2


class FakePool:
    def __init__(self, row: dict[str, object] | None = None) -> None:
        self.row = row
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query: str, *args: object) -> dict[str, object] | None:
        self.calls.append((" ".join(query.split()), args))
        return self.row

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.calls.append((" ".join(query.split()), args))
        return []


@pytest.mark.asyncio
async def test_status_updates_advance_the_event_sequence() -> None:
    now = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
    run_id = UUID("00000000-0000-0000-0000-000000000001")
    row = {
        "id": run_id,
        "sequence": 1,
        "event_sequence": 3,
        "fish_id": UUID("00000000-0000-0000-0000-000000000002"),
        "session_id": UUID("00000000-0000-0000-0000-000000000003"),
        "source_message_id": UUID("00000000-0000-0000-0000-000000000004"),
        "action": "{}",
        "status": "succeeded",
        "entity_id": None,
        "service_domain": None,
        "service_name": None,
        "accepted": True,
        "confirmed": False,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "finished_at": now,
    }
    pool = FakePool(row)
    store = HomeActionStore(cast(asyncpg.Pool, pool))

    updated = await store.update_run(
        run_id,
        status=HomeActionStatus.SUCCEEDED,
        accepted=True,
    )

    query, _ = pool.calls[0]
    assert "event_sequence = nextval('home_action_run_events_sequence_seq')" in query
    assert updated.event_sequence == 3


@pytest.mark.asyncio
async def test_event_polling_uses_the_status_event_cursor() -> None:
    pool = FakePool()
    store = HomeActionStore(cast(asyncpg.Pool, pool))
    fish_id = UUID("00000000-0000-0000-0000-000000000002")
    session_id = UUID("00000000-0000-0000-0000-000000000003")

    await store.list_events(
        fish_id,
        session_id,
        after_event_sequence=9,
        limit=100,
    )

    query, args = pool.calls[0]
    assert "event_sequence > $3" in query
    assert "ORDER BY event_sequence" in query
    assert args == (fish_id, session_id, 9, 100)


@pytest.mark.asyncio
async def test_lists_runs_for_one_source_message() -> None:
    pool = FakePool()
    store = HomeActionStore(cast(asyncpg.Pool, pool))
    fish_id = UUID("00000000-0000-0000-0000-000000000002")
    message_id = UUID("00000000-0000-0000-0000-000000000004")

    await store.list_message_runs(fish_id, message_id)

    query, args = pool.calls[0]
    assert "fish_id = $1 AND source_message_id = $2" in query
    assert "ORDER BY sequence" in query
    assert args == (fish_id, message_id)
