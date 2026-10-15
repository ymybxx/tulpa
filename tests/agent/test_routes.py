from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from fish_memory.agent.models import (
    AgentActivationPolicy,
    AgentMessageEnvelope,
    AgentMessageSubject,
    AgentMessageSubjectKind,
)
from fish_memory.agent.routes import (
    AgentMessageConsumptionRequest,
    agent_router,
    consume_agent_message,
    stream_agent_messages,
)
from fish_memory.db.agent_tasks import AgentMessageSubjectConflictError
from fish_memory.db.sessions import SessionNotFoundError

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_FISH_ID = UUID("00000000-0000-0000-0000-000000000002")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000042")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000043")
OTHER_SESSION_ID = UUID("00000000-0000-0000-0000-000000000045")
SUBJECT_ID = UUID("00000000-0000-0000-0000-000000000044")


def envelope(
    sequence: int,
    *,
    message_id: UUID | None = None,
    consumed_offset: int = 0,
    content: str = "任务已经完成。",
    session_id: UUID = SESSION_ID,
) -> AgentMessageEnvelope:
    return AgentMessageEnvelope(
        message_id=message_id or UUID(int=sequence),
        sequence=sequence,
        session_id=session_id,
        content=content,
        consumed_offset=consumed_offset,
        subject=AgentMessageSubject(
            AgentMessageSubjectKind.RUN_RESULT,
            SUBJECT_ID,
        ),
        activation_policy=AgentActivationPolicy.PLAYBACK_START,
    )


class FakeDeliveryStore:
    def __init__(self) -> None:
        self.items: list[AgentMessageEnvelope] = []
        self.calls: list[tuple[UUID, UUID, int, bool, int, int]] = []
        self.incomplete_calls: list[tuple[UUID, UUID, int, int, int]] = []
        self.finalize_result: object | None = object()
        self.finalize_error: BaseException | None = None
        self.finalize_calls: list[tuple[UUID, UUID, UUID, int, bool]] = []

    async def list_agent_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        include_incomplete: bool,
        limit: int,
        *,
        incomplete_after_sequence: int = 0,
    ) -> list[AgentMessageEnvelope]:
        self.calls.append(
            (
                fish_id,
                session_id,
                after_sequence,
                include_incomplete,
                limit,
                incomplete_after_sequence,
            )
        )
        incomplete = [
            item
            for item in self.items
            if item.session_id == session_id
            and include_incomplete
            and item.sequence > incomplete_after_sequence
            and item.sequence <= after_sequence
            and item.consumed_offset != -1
            and item.consumed_offset < len(item.content)
        ]
        newer = [
            item
            for item in self.items
            if item.session_id == session_id and item.sequence > after_sequence
        ]
        incomplete_ids = {item.message_id for item in incomplete}
        ordered = sorted(incomplete, key=lambda item: item.sequence) + sorted(
            (item for item in newer if item.message_id not in incomplete_ids),
            key=lambda item: item.sequence,
        )
        return ordered[:limit]

    async def list_incomplete_agent_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        through_sequence: int,
        limit: int,
    ) -> list[AgentMessageEnvelope]:
        self.incomplete_calls.append(
            (fish_id, session_id, after_sequence, through_sequence, limit)
        )
        return sorted(
            (
                item
                for item in self.items
                if item.session_id == session_id
                and after_sequence < item.sequence <= through_sequence
                and item.consumed_offset != -1
                and item.consumed_offset < len(item.content)
            ),
            key=lambda item: item.sequence,
        )[:limit]

    async def finalize_agent_message_consumption(
        self,
        message_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> object | None:
        self.finalize_calls.append(
            (message_id, fish_id, session_id, consumed_offset, interrupted)
        )
        if self.finalize_error is not None:
            raise self.finalize_error
        return self.finalize_result


class FakeSessionStore:
    def __init__(self) -> None:
        self.owned = {(FISH_ID, SESSION_ID)}

    async def require_owned(self, fish_id: UUID, session_id: UUID) -> object:
        if (fish_id, session_id) not in self.owned:
            raise SessionNotFoundError(str(session_id))
        return object()


class FakeRequest:
    def __init__(
        self,
        store: FakeDeliveryStore,
        *,
        headers: dict[str, str] | None = None,
        disconnect_after: int | None = None,
    ) -> None:
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                agent_task_store=store,
                session_store=FakeSessionStore(),
            )
        )
        self.headers = headers or {}
        self.disconnect_after = disconnect_after
        self.disconnect_calls = 0

    async def is_disconnected(self) -> bool:
        self.disconnect_calls += 1
        return (
            self.disconnect_after is not None
            and self.disconnect_calls > self.disconnect_after
        )


@pytest.mark.asyncio
async def test_sse_emits_exact_fixed_uuid_camel_case_frame_and_headers() -> None:
    store = FakeDeliveryStore()
    store.items = [envelope(42, message_id=MESSAGE_ID)]
    request = FakeRequest(store)

    response = await stream_agent_messages(
        FISH_ID,
        cast_request(request),
        SESSION_ID,
        after_sequence=41,
    )
    body = response_body(response)
    frame = await anext(body)
    await body.aclose()

    expected = {
        "messageId": str(MESSAGE_ID),
        "sequence": 42,
        "sessionId": str(SESSION_ID),
        "content": "任务已经完成。",
        "consumedOffset": 0,
        "subject": {"kind": "run_result", "id": str(SUBJECT_ID)},
        "activationPolicy": "playback_start",
    }
    assert frame == (
        "id: 42\n"
        "event: agent.message\n"
        f"data: {json.dumps(expected, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


@pytest.mark.asyncio
async def test_sse_delivers_only_the_requested_session() -> None:
    store = FakeDeliveryStore()
    store.items = [
        envelope(41, session_id=OTHER_SESSION_ID, content="old session"),
        envelope(42, message_id=MESSAGE_ID, content="current session"),
    ]

    response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=0,
    )
    body = response_body(response)
    frame = await anext(body)
    await body.aclose()

    assert frame.startswith("id: 42\n")
    assert "current session" in frame
    assert "old session" not in frame
    assert store.calls[0][0:2] == (FISH_ID, SESSION_ID)


@pytest.mark.asyncio
async def test_query_and_valid_last_event_id_use_non_negative_max_cursor() -> None:
    store = FakeDeliveryStore()
    store.items = [envelope(43, consumed_offset=-1)]
    request = FakeRequest(store, headers={"last-event-id": "42"})
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(request),
        SESSION_ID,
        after_sequence=41,
    )

    body = response_body(response)
    await anext(body)
    await body.aclose()

    assert store.calls[0][1:4] == (SESSION_ID, 42, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header",
    ["not-an-integer", "-1", " 42 ", "9" * 100],
)
async def test_invalid_last_event_id_is_ignored(header: str) -> None:
    store = FakeDeliveryStore()
    store.items = [envelope(6, consumed_offset=-1)]
    request = FakeRequest(store, headers={"last-event-id": header})
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(request),
        SESSION_ID,
        after_sequence=5,
    )

    body = response_body(response)
    await anext(body)
    await body.aclose()

    assert store.calls[0][1:4] == (SESSION_ID, 5, False)


@pytest.mark.asyncio
async def test_cursor_past_message_still_replays_incomplete_item() -> None:
    store = FakeDeliveryStore()
    store.items = [envelope(42, message_id=MESSAGE_ID)]
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=100,
    )

    body = response_body(response)
    frame = await anext(body)
    await body.aclose()

    assert frame.startswith("id: 42\n")
    assert store.incomplete_calls[0][1:] == (SESSION_ID, 0, 100, 100)


@pytest.mark.asyncio
async def test_reconnect_restarts_incomplete_scan_without_connection_history() -> None:
    store = FakeDeliveryStore()
    store.items = [envelope(42, message_id=MESSAGE_ID)]

    async def first_frame() -> str:
        response = await stream_agent_messages(
            FISH_ID,
            cast_request(FakeRequest(store)),
            SESSION_ID,
            after_sequence=100,
        )
        body = response_body(response)
        frame = await anext(body)
        await body.aclose()
        return frame

    assert (await first_frame()).startswith("id: 42\n")
    assert (await first_frame()).startswith("id: 42\n")


@pytest.mark.asyncio
async def test_live_stream_replays_late_low_sequence_incomplete_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_memory.agent.routes._POLL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(
        "fish_memory.agent.routes._INCOMPLETE_SCAN_INTERVAL_SECONDS",
        0.0,
    )
    store = FakeDeliveryStore()
    store.items = [envelope(2, consumed_offset=-1, content="new two")]
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=0,
    )
    body = response_body(response)

    first = await asyncio.wait_for(anext(body), timeout=0.2)
    assert first.startswith("id: 2\n")

    late = envelope(1, content="late incomplete one")
    store.items.insert(0, late)
    second = await asyncio.wait_for(anext(body), timeout=0.2)
    assert second.startswith("id: 1\n")

    store.items.append(envelope(3, consumed_offset=-1, content="new three"))
    third = await asyncio.wait_for(anext(body), timeout=0.2)
    assert third.startswith("id: 3\n")

    store.items[0] = envelope(
        1,
        message_id=late.message_id,
        consumed_offset=-1,
        content=late.content,
    )
    store.items.append(envelope(4, consumed_offset=-1, content="new four"))
    fourth = await asyncio.wait_for(anext(body), timeout=0.2)
    await body.aclose()

    assert fourth.startswith("id: 4\n")


@pytest.mark.asyncio
async def test_incomplete_pages_advance_scan_and_do_not_starve_new_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_memory.agent.routes._DELIVERY_PAGE_SIZE", 2)
    store = FakeDeliveryStore()
    store.items = [
        envelope(1, content="one"),
        envelope(2, content="two"),
        envelope(3, content="three"),
        envelope(201, consumed_offset=-1, content="new"),
    ]
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=200,
    )

    body = response_body(response)
    frames = [await anext(body) for _ in range(4)]
    await body.aclose()

    assert [int(frame.splitlines()[0].removeprefix("id: ")) for frame in frames] == [
        1,
        2,
        3,
        201,
    ]
    assert [call[2] for call in store.incomplete_calls[:2]] == [0, 2]


@pytest.mark.asyncio
async def test_new_domain_never_emits_high_incomplete_before_lower_new_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_memory.agent.routes._DELIVERY_PAGE_SIZE", 1)
    store = FakeDeliveryStore()
    store.items = [
        envelope(1, consumed_offset=-1, content="one"),
        envelope(2, consumed_offset=-1, content="two"),
        envelope(3, content="three incomplete"),
    ]
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=0,
    )

    body = response_body(response)
    frames = [await anext(body) for _ in range(3)]
    await body.aclose()

    assert [int(frame.splitlines()[0].removeprefix("id: ")) for frame in frames] == [
        1,
        2,
        3,
    ]


@pytest.mark.asyncio
async def test_reconnect_cursor_from_first_frame_receives_every_remaining_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_memory.agent.routes._DELIVERY_PAGE_SIZE", 2)
    store = FakeDeliveryStore()
    store.items = [
        *(envelope(sequence, consumed_offset=-1) for sequence in range(1, 6)),
        envelope(100, content="high incomplete"),
    ]

    first_response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=0,
    )
    first_body = response_body(first_response)
    first_frame = await anext(first_body)
    await first_body.aclose()
    first_cursor = int(first_frame.splitlines()[0].removeprefix("id: "))

    reconnect_response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=first_cursor,
    )
    reconnect_body = response_body(reconnect_response)
    frames = [await anext(reconnect_body) for _ in range(5)]
    await reconnect_body.aclose()

    assert first_cursor == 1
    assert [int(frame.splitlines()[0].removeprefix("id: ")) for frame in frames] == [
        2,
        3,
        4,
        5,
        100,
    ]


@pytest.mark.asyncio
async def test_one_past_scan_is_followed_by_linear_new_page_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_memory.agent.routes._DELIVERY_PAGE_SIZE", 2)
    monkeypatch.setattr(
        "fish_memory.agent.routes._INCOMPLETE_SCAN_INTERVAL_SECONDS",
        60.0,
        raising=False,
    )
    sleep_calls: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("fish_memory.agent.routes.asyncio.sleep", record_sleep)
    store = FakeDeliveryStore()
    store.items = [
        envelope(1, content="past one"),
        envelope(2, content="past two"),
        envelope(3, content="past three"),
        *(envelope(sequence, consumed_offset=-1) for sequence in range(7, 11)),
    ]
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=6,
    )
    body = response_body(response)

    frames = [await anext(body) for _ in range(7)]
    await body.aclose()

    assert [int(frame.splitlines()[0].removeprefix("id: ")) for frame in frames] == [
        1,
        2,
        3,
        7,
        8,
        9,
        10,
    ]
    assert len(store.incomplete_calls) == 2
    assert len(store.calls) == 2
    assert all(call[3] is False for call in store.calls)
    assert sleep_calls == []


@pytest.mark.asyncio
async def test_connection_cursors_prevent_replaying_the_same_incomplete_item() -> None:
    store = FakeDeliveryStore()
    store.items = [
        envelope(42, message_id=MESSAGE_ID),
        envelope(101, consumed_offset=-1, content="new"),
    ]
    response = await stream_agent_messages(
        FISH_ID,
        cast_request(FakeRequest(store)),
        SESSION_ID,
        after_sequence=100,
    )

    body = response_body(response)
    first = await anext(body)
    second = await anext(body)
    await body.aclose()

    assert first.startswith("id: 42\n")
    assert second.startswith("id: 101\n")
    assert store.calls[0][5] == 0


@pytest.mark.asyncio
async def test_heartbeat_is_observable_and_disconnect_stops_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fish_memory.agent.routes._SSE_HEARTBEAT_SECONDS", 0.0)
    monkeypatch.setattr("fish_memory.agent.routes._POLL_INTERVAL_SECONDS", 0.0)
    store = FakeDeliveryStore()
    request = FakeRequest(store)
    response = await stream_agent_messages(FISH_ID, cast_request(request), SESSION_ID)

    body = response_body(response)
    frame = await anext(body)
    await body.aclose()

    assert frame == "event: agent.heartbeat\ndata: {}\n\n"
    assert "event: agent.message" not in frame

    disconnected = FakeRequest(store, disconnect_after=0)
    stopped = await stream_agent_messages(
        FISH_ID,
        cast_request(disconnected),
        SESSION_ID,
    )
    stopped_body = response_body(stopped)
    with pytest.raises(StopAsyncIteration):
        await anext(stopped_body)


def cast_request(request: FakeRequest):
    from fastapi import Request

    return cast(Request, request)


def response_body(
    response: StreamingResponse,
) -> AsyncGenerator[str, None]:
    return cast(AsyncGenerator[str, None], response.body_iterator)


def app_with(store: FakeDeliveryStore) -> FastAPI:
    app = FastAPI()
    app.state.agent_task_store = store
    app.state.session_store = FakeSessionStore()
    app.include_router(agent_router)
    return app


@pytest.mark.asyncio
async def test_consumption_endpoint_uses_agent_store_boundary() -> None:
    store = FakeDeliveryStore()
    transport = httpx.ASGITransport(app=app_with(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/agent-messages/{MESSAGE_ID}/consumption"
            f"?sessionId={SESSION_ID}",
            json={"consumedOffset": 7, "interrupted": True},
        )

    assert response.status_code == 204
    assert store.finalize_calls == [(MESSAGE_ID, FISH_ID, SESSION_ID, 7, True)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"consumedOffset": -1, "interrupted": False},
        {"consumedOffset": "0", "interrupted": False},
        {"consumedOffset": 0, "interrupted": False, "extra": True},
        {"consumedOffset": 0, "interrupted": "false"},
    ],
)
async def test_consumption_body_is_strict_and_forbids_extra_fields(
    body: dict[str, object],
) -> None:
    store = FakeDeliveryStore()
    transport = httpx.ASGITransport(app=app_with(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/agent-messages/{MESSAGE_ID}/consumption"
            f"?sessionId={SESSION_ID}",
            json=body,
        )

    assert response.status_code == 422
    assert store.finalize_calls == []


@pytest.mark.asyncio
async def test_consumption_maps_missing_and_non_agent_subjects() -> None:
    store = FakeDeliveryStore()
    missing = await consume_agent_message(
        FISH_ID,
        MESSAGE_ID,
        SESSION_ID,
        AgentMessageConsumptionRequest(consumedOffset=0, interrupted=False),
        cast_request(FakeRequest(store)),
    )
    assert missing.status_code == 204

    store.finalize_result = None
    transport = httpx.ASGITransport(app=app_with(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{OTHER_FISH_ID}/agent-messages/{MESSAGE_ID}/consumption"
            f"?sessionId={SESSION_ID}",
            json={"consumedOffset": 0, "interrupted": False},
        )
    assert response.status_code == 404

    store.finalize_error = AgentMessageSubjectConflictError(
        "message is not an Agent message"
    )
    transport = httpx.ASGITransport(app=app_with(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/agent-messages/{MESSAGE_ID}/consumption"
            f"?sessionId={SESSION_ID}",
            json={"consumedOffset": 0, "interrupted": False},
        )
    assert response.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", ["-1", "9" * 100])
async def test_invalid_query_cursor_is_rejected_before_stream_starts(
    cursor: str,
) -> None:
    store = FakeDeliveryStore()
    transport = httpx.ASGITransport(app=app_with(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/v1/fishes/{FISH_ID}/agent-messages/events"
            f"?sessionId={SESSION_ID}&afterSequence={cursor}"
        )
    assert response.status_code == 422
    assert store.calls == []


@pytest.mark.asyncio
async def test_agent_stream_and_consumption_require_an_owned_session() -> None:
    store = FakeDeliveryStore()
    transport = httpx.ASGITransport(app=app_with(store))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.get(
            f"/v1/fishes/{FISH_ID}/agent-messages/events?afterSequence=0"
        )
        foreign = await client.get(
            f"/v1/fishes/{FISH_ID}/agent-messages/events"
            f"?sessionId={OTHER_SESSION_ID}&afterSequence=0"
        )
        consumption = await client.post(
            f"/v1/fishes/{FISH_ID}/agent-messages/{MESSAGE_ID}/consumption"
            f"?sessionId={OTHER_SESSION_ID}",
            json={"consumedOffset": 0, "interrupted": False},
        )

    assert missing.status_code == 422
    assert foreign.status_code == 404
    assert consumption.status_code == 404
    assert store.calls == []
    assert store.finalize_calls == []
