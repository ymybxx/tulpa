from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.api.routes import router
from fish_memory.models import MessageStatus
from fish_memory.pet.hub import PetEventHub

from .fakes import FakeApiCoordinator, FakeApiStore, FakeSessionStore

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000010")


def build_test_app(coordinator: FakeApiCoordinator) -> FastAPI:
    app = FastAPI()
    app.state.message_store = FakeApiStore()
    app.state.session_store = FakeSessionStore()
    app.state.answer_coordinator = coordinator
    app.include_router(router)
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize("playback_status", ["completed", "interrupted", "failed"])
async def test_web_interrupt_settles_finished_companion_boundary_without_new_ack(
    playback_status,
):
    coordinator = FakeApiCoordinator([])
    coordinator.answer_playback_owner = lambda *args: "companion"
    coordinator.progress_result = 4
    app = build_test_app(coordinator)
    hub = app.state.pet_event_hub = PetEventHub()
    _, events = hub.subscribe(control_only=True)
    hub.publish_control(
        "pet.playback.progress",
        None,
        {
            "fishId": str(FISH_ID),
            "sessionId": str(UUID(int=2)),
            "messageId": str(MESSAGE_ID),
            "playbackId": str(UUID(int=3)),
            "revision": 4,
            "status": playback_status,
            "displayOffset": 4,
            "consumedOffset": 4,
            "visibleContent": "已经播完",
        },
    )
    await events.get()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/answer/interrupt",
            json={
                "messageId": str(MESSAGE_ID),
                "consumedOffset": 0,
                "playbackOwner": "web",
            },
        )
    assert response.status_code == 204
    assert events.empty()  # no command to an already retired playback
    if playback_status == "interrupted":
        assert coordinator.interrupt_calls == [(FISH_ID, MESSAGE_ID, 4)]
    else:
        assert coordinator.progress_calls == [(FISH_ID, MESSAGE_ID, 4)]
        assert coordinator.interrupt_calls == []


@pytest.mark.asyncio
async def test_active_companion_still_requires_device_ack_not_generation_completion():
    coordinator = FakeApiCoordinator([])
    coordinator.answer_playback_owner = lambda *args: "companion"
    coordinator.runtime.status = (
        MessageStatus.COMPLETED
    )  # LLM done, audio still playing
    app = build_test_app(coordinator)
    hub = app.state.pet_event_hub = PetEventHub()
    hub.publish_control(
        "pet.playback.progress",
        None,
        {
            "fishId": str(FISH_ID),
            "messageId": str(MESSAGE_ID),
            "playbackId": str(UUID(int=3)),
            "revision": 4,
            "status": "playing",
            "displayOffset": 4,
            "consumedOffset": 0,
        },
    )
    requested = []

    async def interrupt(fish_id, message_id):
        requested.append((fish_id, message_id))
        raise TimeoutError

    hub.request_answer_interrupt = interrupt
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/answer/interrupt",
            json={
                "messageId": str(MESSAGE_ID),
                "consumedOffset": 999,
                "playbackOwner": "web",
            },
        )
    assert response.status_code == 503
    assert requested == [(FISH_ID, MESSAGE_ID)]
    assert coordinator.progress_calls == []


@pytest.mark.asyncio
async def test_answer_streams_cached_runtime_without_starting_generation() -> None:
    coordinator = FakeApiCoordinator([])
    runtime = coordinator.runtime
    runtime.answer_message_id = MESSAGE_ID
    runtime.answer_sequence = 8
    runtime.content = "完成🐟"
    runtime.status = MessageStatus.COMPLETED
    transport = httpx.ASGITransport(app=build_test_app(coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/v1/fishes/{FISH_ID}/answer",
            headers={"accept": "text/event-stream"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: answer.snapshot" in response.text
    assert "event: answer.completed" in response.text
    assert "完成🐟" in response.text
    assert coordinator.notifications == []


@pytest.mark.asyncio
async def test_answer_accepts_sequence_cursor_for_a_new_voice_turn() -> None:
    coordinator = FakeApiCoordinator([])
    runtime = coordinator.runtime
    runtime.answer_message_id = MESSAGE_ID
    runtime.answer_sequence = 8
    runtime.content = "新回答"
    runtime.status = MessageStatus.COMPLETED
    transport = httpx.ASGITransport(app=build_test_app(coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            f"/v1/fishes/{FISH_ID}/answer?afterSequence=7",
            headers={"accept": "text/event-stream"},
        )

    assert response.status_code == 200
    assert "新回答" in response.text


@pytest.mark.asyncio
async def test_progress_returns_204() -> None:
    coordinator = FakeApiCoordinator([])
    coordinator.progress_result = 4
    transport = httpx.ASGITransport(app=build_test_app(coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/answer/progress",
            json={"messageId": str(MESSAGE_ID), "consumedOffset": 4},
        )

    assert response.status_code == 204
    assert coordinator.progress_calls == [(FISH_ID, MESSAGE_ID, 4)]


@pytest.mark.asyncio
async def test_interrupt_returns_204_and_cancels_answer_at_offset() -> None:
    coordinator = FakeApiCoordinator([])
    transport = httpx.ASGITransport(app=build_test_app(coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/answer/interrupt",
            json={"messageId": str(MESSAGE_ID), "consumedOffset": 4},
        )

    assert response.status_code == 204
    assert coordinator.interrupt_calls == [(FISH_ID, MESSAGE_ID, 4)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [(ValueError("too long"), 400), (LookupError("missing"), 404)],
)
async def test_progress_maps_domain_errors(
    error: Exception,
    expected_status: int,
) -> None:
    coordinator = FakeApiCoordinator([])
    coordinator.progress_error = error
    transport = httpx.ASGITransport(app=build_test_app(coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/answer/progress",
            json={"messageId": str(MESSAGE_ID), "consumedOffset": 4},
        )

    assert response.status_code == expected_status
    assert "error" in response.json()
