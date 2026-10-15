import importlib
import importlib.util
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import asyncpg
import pytest

from fish_memory import models


def test_session_model_keeps_conversation_window_state() -> None:
    session_type = getattr(models, "Session", None)
    assert session_type is not None

    session_id = uuid4()
    fish_id = uuid4()
    created_at = datetime(2026, 8, 9, 9, 0, tzinfo=UTC)
    last_message_at = datetime(2026, 8, 9, 9, 5, tzinfo=UTC)

    session = session_type(
        id=session_id,
        fish_id=fish_id,
        created_at=created_at,
        last_message_at=last_message_at,
        history_window_start_sequence=11,
        history_window_token_count=1200,
    )

    assert session.id == session_id
    assert session.fish_id == fish_id
    assert session.created_at == created_at
    assert session.last_message_at == last_message_at
    assert session.history_window_start_sequence == 11
    assert session.history_window_token_count == 1200


class FakePool:
    def __init__(self) -> None:
        self.fetchrow_results: list[dict[str, object] | None] = []
        self.fetch_results: list[list[dict[str, object]]] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, object] | None:
        self.calls.append((" ".join(query.split()), args))
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        self.calls.append((" ".join(query.split()), args))
        return self.fetch_results.pop(0)


def _sessions_module() -> Any:
    spec = importlib.util.find_spec("fish_memory.db.sessions")
    assert spec is not None
    return importlib.import_module("fish_memory.db.sessions")


def _session_record(
    *,
    session_id: object,
    fish_id: object,
    created_at: datetime,
    last_message_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "id": session_id,
        "fish_id": fish_id,
        "created_at": created_at,
        "last_message_at": last_message_at,
        "history_window_start_sequence": 0,
        "history_window_token_count": 0,
    }


@pytest.mark.asyncio
async def test_session_store_creates_session_for_active_fish() -> None:
    sessions = _sessions_module()
    pool = FakePool()
    fish_id = uuid4()
    session_id = uuid4()
    created_at = datetime(2026, 8, 9, 10, 0, tzinfo=UTC)
    pool.fetchrow_results.append(
        _session_record(
            session_id=session_id,
            fish_id=fish_id,
            created_at=created_at,
        )
    )
    store = sessions.SessionStore(cast(asyncpg.Pool, pool))

    created = await store.create(fish_id, created_at=created_at)

    query, args = pool.calls[0]
    assert "INSERT INTO fish_sessions" in query
    assert "deleted_at IS NULL" in query
    assert args[1:] == (fish_id, created_at)
    assert created.id == session_id
    assert created.fish_id == fish_id


@pytest.mark.asyncio
async def test_session_store_resolves_latest_by_creation_time() -> None:
    sessions = _sessions_module()
    pool = FakePool()
    fish_id = uuid4()
    session_id = uuid4()
    created_at = datetime(2026, 8, 9, 11, 0, tzinfo=UTC)
    pool.fetchrow_results.append(
        _session_record(
            session_id=session_id,
            fish_id=fish_id,
            created_at=created_at,
            last_message_at=datetime(2026, 8, 9, 11, 1, tzinfo=UTC),
        )
    )
    store = sessions.SessionStore(cast(asyncpg.Pool, pool))

    latest = await store.get_latest(fish_id)

    query, args = pool.calls[0]
    assert "ORDER BY created_at DESC, id DESC" in query
    assert "last_message_at" not in query.split("ORDER BY", maxsplit=1)[1]
    assert args == (fish_id,)
    assert latest is not None
    assert latest.id == session_id


@pytest.mark.asyncio
async def test_session_store_lists_latest_created_first() -> None:
    sessions = _sessions_module()
    pool = FakePool()
    fish_id = uuid4()
    newer_id = uuid4()
    older_id = uuid4()
    pool.fetch_results.append(
        [
            _session_record(
                session_id=newer_id,
                fish_id=fish_id,
                created_at=datetime(2026, 8, 9, 11, 0, tzinfo=UTC),
            ),
            _session_record(
                session_id=older_id,
                fish_id=fish_id,
                created_at=datetime(2026, 8, 9, 10, 0, tzinfo=UTC),
            ),
        ]
    )
    store = sessions.SessionStore(cast(asyncpg.Pool, pool))

    listed = await store.list(fish_id)

    query, _args = pool.calls[0]
    assert "ORDER BY created_at DESC, id DESC" in query
    assert [item.id for item in listed] == [newer_id, older_id]


@pytest.mark.asyncio
async def test_session_store_rejects_session_owned_by_another_fish() -> None:
    sessions = _sessions_module()
    pool = FakePool()
    pool.fetchrow_results.append(None)
    store = sessions.SessionStore(cast(asyncpg.Pool, pool))

    with pytest.raises(sessions.SessionNotFoundError):
        await store.require_owned(uuid4(), uuid4())
