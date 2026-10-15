from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

from fish_memory.db.home_actions import HomeActionRun
from fish_memory.home.models import HomeActionStatus
from fish_memory.home.routes import home_router, stream_home_actions
from fish_memory.models import Session

DEFAULT_FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_SESSION_ID = UUID("00000000-0000-0000-0000-000000000101")
RUN_ID = UUID("00000000-0000-0000-0000-000000000201")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000202")
NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


class FakeSessionStore:
    async def require_owned(self, fish_id: UUID, session_id: UUID) -> Session:
        assert fish_id == DEFAULT_FISH_ID
        assert session_id == DEFAULT_SESSION_ID
        return Session(
            id=session_id,
            fish_id=fish_id,
            created_at=NOW,
            last_message_at=NOW,
        )

    async def get_latest(self, fish_id: UUID) -> Session:
        return await self.require_owned(fish_id, DEFAULT_SESSION_ID)


class FakeHomeActionStore:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID, int, int]] = []
        self.event_calls: list[tuple[UUID, UUID, int, int]] = []
        self.event_batches: list[list[HomeActionRun]] = []

    async def list_runs(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        after_sequence: int,
        limit: int,
    ) -> list[HomeActionRun]:
        self.calls.append((fish_id, session_id, after_sequence, limit))
        return [_run(fish_id, session_id, event_sequence=11)]

    async def list_events(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        after_event_sequence: int,
        limit: int,
    ) -> list[HomeActionRun]:
        self.event_calls.append((fish_id, session_id, after_event_sequence, limit))
        return self.event_batches.pop(0) if self.event_batches else []


def _run(
    fish_id: UUID,
    session_id: UUID,
    *,
    event_sequence: int,
    status: HomeActionStatus = HomeActionStatus.SUCCEEDED,
) -> HomeActionRun:
    return HomeActionRun(
        id=RUN_ID,
        sequence=7,
        event_sequence=event_sequence,
        fish_id=fish_id,
        session_id=session_id,
        source_message_id=MESSAGE_ID,
        action={"entityId": "light.entry", "operation": "turn_on"},
        status=status,
        entity_id="light.entry",
        service_domain="light",
        service_name="turn_on",
        accepted=status is HomeActionStatus.SUCCEEDED,
        confirmed=False,
        result={"state": "on"} if status is HomeActionStatus.SUCCEEDED else None,
        error=None,
        created_at=NOW,
        updated_at=NOW,
        finished_at=NOW if status is HomeActionStatus.SUCCEEDED else None,
    )


def _app(store: FakeHomeActionStore) -> FastAPI:
    app = FastAPI()
    app.state.session_store = FakeSessionStore()
    app.state.home_action_store = store
    app.include_router(home_router)
    return app


@pytest.mark.asyncio
async def test_lists_home_actions_for_selected_session() -> None:
    store = FakeHomeActionStore()
    transport = httpx.ASGITransport(app=_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/v1/fishes/{DEFAULT_FISH_ID}/home-actions",
            params={"sessionId": str(DEFAULT_SESSION_ID), "afterSequence": 3},
        )

    assert response.status_code == 200
    assert store.calls == [(DEFAULT_FISH_ID, DEFAULT_SESSION_ID, 3, 200)]
    assert response.json()["items"][0] == {
        "id": str(RUN_ID),
        "sequence": 7,
        "eventSequence": 11,
        "sourceMessageId": str(MESSAGE_ID),
        "action": {"entityId": "light.entry", "operation": "turn_on"},
        "entityId": "light.entry",
        "status": "succeeded",
        "accepted": True,
        "confirmed": False,
        "result": {"state": "on"},
        "error": None,
        "createdAt": NOW.isoformat(),
        "updatedAt": NOW.isoformat(),
    }


@pytest.mark.asyncio
async def test_stream_advances_by_status_event_sequence() -> None:
    store = FakeHomeActionStore()
    store.event_batches = [
        [
            _run(
                DEFAULT_FISH_ID,
                DEFAULT_SESSION_ID,
                event_sequence=10,
                status=HomeActionStatus.EXECUTING,
            )
        ],
        [
            _run(
                DEFAULT_FISH_ID,
                DEFAULT_SESSION_ID,
                event_sequence=11,
            )
        ],
    ]
    app = _app(store)

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    starlette_request = Request(
        {"type": "http", "app": app, "headers": []},
        receive,
    )
    response = await stream_home_actions(
        DEFAULT_FISH_ID,
        starlette_request,
        session_id=DEFAULT_SESSION_ID,
        after_event_sequence=9,
    )

    first = await anext(response.body_iterator)
    second = await anext(response.body_iterator)
    await response.body_iterator.aclose()

    assert "id: 10" in first
    assert "event: home.action.executing" in first
    assert "id: 11" in second
    assert "event: home.action.succeeded" in second
    assert [call[2] for call in store.event_calls] == [9, 10]
