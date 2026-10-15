"""Authenticated WebSocket endpoints for desktop-pet control and audio."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from fish_memory.db.sessions import SessionNotFoundError, SessionStore
from fish_memory.pet.hub import PetEventHub
from fish_memory.pet.protocol import parse_published_control
from fish_memory.tts.facial import FacialProfile

router = APIRouter(prefix="/v1/pet", tags=["desktop-pet"])


def _hub(connection: Request | WebSocket) -> PetEventHub:
    return cast(PetEventHub, connection.app.state.pet_event_hub)


@router.get("/status")
async def pet_status(request: Request) -> dict[str, object]:
    return _hub(request).status()


@router.put("/facial-profile")
async def update_facial_profile(
    request: Request, body: FacialProfile | None = None
) -> dict[str, bool]:
    # Authenticated local desktop renderer publishes once per model/settings
    # revision, outside the latency-sensitive speech request.
    request.app.state.speech_facial_profile = body
    experiment = getattr(request.app.state, "experimental_companion", None)
    return {
        "enabled": bool(
            body
            and body.parameters
            and experiment is not None
            and experiment.configuration.api_key is not None
            and experiment.configuration.api_key.get_secret_value().strip()
        )
    }


class PetTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID = Field(alias="requestId")
    fish_id: UUID = Field(alias="fishId")
    content: str = Field(min_length=1, max_length=32_000)
    attachment_ids: list[UUID] = Field(
        default_factory=list, alias="attachmentIds", max_length=8
    )


@router.post("/chat")
async def pet_text_input(body: PetTextRequest, request: Request) -> dict[str, object]:
    hub = _hub(request)
    if not body.content.strip():
        raise HTTPException(422, "message cannot be blank")
    if hub.publisher_count == 0:
        raise HTTPException(503, "原生语音服务尚未就绪，请稍后重试")
    try:
        return await hub.text_input.submit(
            body.request_id,
            body.fish_id,
            body.content,
            lambda payload: hub.publish_control("pet.chat.submit", None, payload),
            attachment_ids=tuple(body.attachment_ids),
        )
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    except TimeoutError as error:
        raise HTTPException(
            504, "发送状态未确认，请先查看聊天记录；草稿已保留"
        ) from error


class PetNewSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fish_id: UUID = Field(alias="fishId")


@router.post("/session/new")
async def pet_new_session(
    body: PetNewSessionRequest, request: Request
) -> dict[str, str]:
    """「新对话」: TA stops carrying on the current conversation; memory stays."""
    store = cast(SessionStore, request.app.state.session_store)
    try:
        session = await store.create(body.fish_id)
    except SessionNotFoundError as error:
        raise HTTPException(404, "fish 不存在") from error
    history = getattr(request.app.state, "session_history", None)
    if history is not None:
        history.set_active(body.fish_id, session.id)
    hub = _hub(request)
    if hub.publisher_count:
        hub.publish_control(
            "pet.session.new",
            None,
            {"fishId": str(body.fish_id), "sessionId": str(session.id)},
        )
    return {"sessionId": str(session.id)}


@router.post("/listen")
async def pet_listen(request: Request) -> dict[str, bool]:
    """Start listening as if the wake word was heard (the chat page's mic button)."""
    hub = _hub(request)
    if hub.publisher_count == 0:
        raise HTTPException(503, "原生语音服务尚未就绪，请稍后重试")
    hub.publish_control("pet.listen", None, {})
    return {"ok": True}


@router.post("/listen/stop")
async def pet_stop_listening(request: Request) -> dict[str, bool]:
    """Stop listening (the chat page's mic button pressed while TA listens)."""
    hub = _hub(request)
    if hub.publisher_count == 0:
        raise HTTPException(503, "原生语音服务尚未就绪，请稍后重试")
    hub.publish_control("pet.listen.stop", None, {})
    return {"ok": True}


@router.post("/owner-voice/add")
async def pet_owner_voice_add(request: Request) -> dict[str, bool]:
    """The owner confirmed that the sentences TA just missed were theirs."""
    hub = _hub(request)
    if hub.publisher_count == 0:
        raise HTTPException(503, "原生语音服务尚未就绪，请稍后重试")
    hub.publish_control("pet.owner_voice.add", None, {})
    return {"ok": True}


@router.websocket("/events/publish")
async def publish_pet_events(websocket: WebSocket) -> None:
    await websocket.accept()
    hub = _hub(websocket)
    hub.publisher_connected()
    command_subscriber, commands = hub.subscribe(control_only=True)

    async def send_commands() -> None:
        # Restore playback leases on reconnect before waiting for changes.
        payload = cast(dict[str, object], hub.snapshot()["payload"])
        for event in cast(list[dict[str, object]], payload["events"]):
            if event["type"] in {
                "pet.plugin.sessions",
                "pet.microphone",
                "pet.speaker",
            }:
                await websocket.send_json(event)
        while True:
            event = await commands.get()
            if (
                event.kind == "control"
                and isinstance(event.value, str)
                and json.loads(event.value).get("type")
                in {
                    "pet.answer.interrupt",
                    "pet.chat.submit",
                    "pet.listen",
                    "pet.listen.stop",
                    "pet.session.new",
                    "pet.owner_voice.add",
                    "pet.plugin.sessions",
                    "pet.microphone",
                    "pet.speaker",
                }
            ):
                await websocket.send_text(event.value)

    command_task = asyncio.create_task(send_commands())
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            raw_text = message.get("text")
            if isinstance(raw_text, str):
                try:
                    parsed = parse_published_control(json.loads(raw_text))
                except (json.JSONDecodeError, ValueError) as error:
                    await websocket.send_json({"type": "error", "message": str(error)})
                    continue
                hub.publish_control(
                    parsed.event_type,
                    parsed.generation,
                    parsed.payload,
                )
                continue
            raw_bytes = message.get("bytes")
            if isinstance(raw_bytes, bytes):
                try:
                    hub.publish_pcm(raw_bytes)
                except ValueError as error:
                    await websocket.send_json({"type": "error", "message": str(error)})
    except WebSocketDisconnect:
        return
    finally:
        hub.publisher_disconnected()
        hub.unsubscribe(command_subscriber)
        command_task.cancel()
        await asyncio.gather(command_task, return_exceptions=True)


@router.websocket("/events")
async def subscribe_pet_events(websocket: WebSocket) -> None:
    await websocket.accept()
    hub = _hub(websocket)
    subscriber_id, queue = hub.subscribe(
        control_only=websocket.query_params.get("controlOnly") == "1"
    )
    await websocket.send_json(hub.snapshot())
    try:
        while True:
            try:
                outbound = await asyncio.wait_for(queue.get(), timeout=20)
            except TimeoutError:
                await websocket.send_json({"type": "pet.ping"})
                continue
            if outbound.kind == "pcm":
                assert isinstance(outbound.value, bytes)
                await websocket.send_bytes(outbound.value)
            else:
                assert isinstance(outbound.value, str)
                await websocket.send_text(outbound.value)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        hub.unsubscribe(subscriber_id)
        with suppress(RuntimeError):
            await websocket.close()
