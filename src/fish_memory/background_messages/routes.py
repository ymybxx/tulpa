"""Session-scoped delivery for all durable background Assistant messages."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Annotated, cast
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from fish_memory.background_messages.models import BackgroundAssistantMessage
from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.db.sessions import SessionNotFoundError, SessionStore
from fish_memory.tts.speech_tags import speech_prefix

background_message_router = APIRouter(
    prefix="/v1/fishes/{fish_id}/companion-messages",
    tags=["public"],
)

_SSE_HEARTBEAT_SECONDS = 15.0
_POLL_INTERVAL_SECONDS = 0.4
_INCOMPLETE_SCAN_INTERVAL_SECONDS = 2.0
_DELIVERY_PAGE_SIZE = 100
_MAX_NEW_PAGES_PER_CYCLE = 16
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_NON_NEGATIVE_INTEGER_RE = re.compile(r"^[0-9]+$")


class BackgroundMessageConsumptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    consumed_offset: int = Field(
        validation_alias="consumedOffset",
        serialization_alias="consumedOffset",
        ge=-1,
        strict=True,
    )
    interrupted: bool = Field(strict=True)


@background_message_router.get("/events", response_model=None)
async def stream_background_messages(
    fish_id: UUID,
    request: Request,
    session_id: Annotated[UUID, Query(alias="sessionId")],
    after_sequence: Annotated[
        int,
        Query(alias="afterSequence", ge=0, le=_MAX_SEQUENCE),
    ] = 0,
    native_consumer: Annotated[bool, Query(alias="nativeConsumer")] = False,
) -> StreamingResponse:
    await _require_owned_session(request, fish_id, session_id)
    store = cast(BackgroundMessageStore, request.app.state.background_message_store)
    client_cursor = max(after_sequence, _last_event_sequence(request))
    proactive = getattr(request.app.state, "proactive_coordinator", None)
    extensions = getattr(request.app.state, "extension_service", None)

    def format_message(item: BackgroundAssistantMessage) -> str | None:
        stream = proactive.streams.get(item.message_id) if proactive else None
        if item.source == "plugin" and extensions is not None:
            stream = extensions.streams.get(item.message_id)
        if (
            item.source in {"proactive_observation", "plugin"}
            and item.expires_at is None
        ):
            # Live expressions never fall back to replaying a saved prefix after
            # their owner has been revoked or lost. Startup clears orphan rows.
            if (
                not native_consumer
                or stream is None
                or stream.intent.state not in {"pending", "playing"}
            ):
                return None
        return _format_background_message(
            item,
            streaming=bool(native_consumer and stream is not None),
            speech_content=(
                (
                    stream.speech_content
                    if len(stream.text) == len(item.content)
                    else speech_prefix(stream.speech_content, len(item.content))
                )
                if stream is not None and getattr(stream, "speech_tags", False)
                else None
            ),
        )

    async def events() -> AsyncIterator[str]:
        new_cursor = client_cursor
        sent_out_of_order: dict[UUID, int] = {}
        last_sent = time.monotonic()
        next_incomplete_scan_at = 0.0
        while not await request.is_disconnected():
            wake = proactive.streams.wake if proactive is not None else None
            emitted = False
            progressed = False
            now = time.monotonic()
            if now >= next_incomplete_scan_at:
                current_incomplete_ids: set[UUID] = set()
                incomplete_scan_sequence = 0
                through_sequence = new_cursor
                while not await request.is_disconnected():
                    items = await store.list_incomplete_messages_for_delivery(
                        fish_id,
                        session_id,
                        after_sequence=incomplete_scan_sequence,
                        through_sequence=through_sequence,
                        limit=_DELIVERY_PAGE_SIZE,
                    )
                    if not items:
                        break
                    progressed = True
                    for item in items:
                        if await request.is_disconnected():
                            return
                        incomplete_scan_sequence = item.sequence
                        current_incomplete_ids.add(item.message_id)
                        if item.message_id in sent_out_of_order:
                            continue
                        sent_out_of_order[item.message_id] = item.sequence
                        frame = format_message(item)
                        if frame is None:
                            continue
                        emitted = True
                        last_sent = time.monotonic()
                        yield frame
                    if len(items) < _DELIVERY_PAGE_SIZE:
                        break

                sent_out_of_order = {
                    message_id: sequence
                    for message_id, sequence in sent_out_of_order.items()
                    if message_id in current_incomplete_ids or sequence > new_cursor
                }
                next_incomplete_scan_at = now + _INCOMPLETE_SCAN_INTERVAL_SECONDS
            if await request.is_disconnected():
                return

            new_pages = 0
            while (
                new_pages < _MAX_NEW_PAGES_PER_CYCLE
                and not await request.is_disconnected()
            ):
                items = await store.list_messages_for_delivery(
                    fish_id,
                    session_id,
                    after_sequence=new_cursor,
                    include_incomplete=False,
                    limit=_DELIVERY_PAGE_SIZE,
                    incomplete_after_sequence=0,
                )
                if not items:
                    break
                progressed = True
                new_pages += 1
                for item in items:
                    if await request.is_disconnected():
                        return
                    new_cursor = item.sequence
                    if item.message_id in sent_out_of_order:
                        continue
                    sent_out_of_order[item.message_id] = item.sequence
                    frame = format_message(item)
                    if frame is None:
                        continue
                    emitted = True
                    last_sent = time.monotonic()
                    yield frame
                if len(items) < _DELIVERY_PAGE_SIZE:
                    break

            if not emitted and time.monotonic() - last_sent >= _SSE_HEARTBEAT_SECONDS:
                last_sent = time.monotonic()
                yield "event: companion.heartbeat\ndata: {}\n\n"
            if not emitted and not progressed:
                if wake is None:
                    await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                else:
                    try:
                        async with asyncio.timeout(_POLL_INTERVAL_SECONDS):
                            await wake.wait()
                    except TimeoutError:
                        pass

    async def bound_events() -> AsyncIterator[str]:
        experimental = getattr(proactive, "experimental", None)
        experimental_token = (
            uuid4() if experimental is not None and not native_consumer else None
        )
        if experimental_token is not None and experimental is not None:
            experimental.subscribe(experimental_token, fish_id, session_id)
        token = (
            proactive.bind(fish_id, session_id)
            if proactive and native_consumer
            else None
        )
        if extensions is not None and native_consumer:
            extensions.native_consumer(fish_id, session_id, True)
        try:
            async for event in events():
                yield event
        finally:
            if experimental_token is not None and experimental is not None:
                experimental.unsubscribe(experimental_token)
            if extensions is not None and native_consumer:
                extensions.native_consumer(fish_id, session_id, False)
            if token is not None and proactive is not None:
                proactive.unbind(token)

    return StreamingResponse(
        bound_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@background_message_router.post(
    "/{message_id}/consumption",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def consume_background_message(
    fish_id: UUID,
    message_id: UUID,
    session_id: Annotated[UUID, Query(alias="sessionId")],
    body: BackgroundMessageConsumptionRequest,
    request: Request,
) -> Response:
    await _require_owned_session(request, fish_id, session_id)
    if body.consumed_offset == -1:
        # Match ordinary text chat: -1 means fully delivered without audio.
        # A text observer must never settle an active native playback owner.
        settings = request.app.state.settings
        if (
            not settings.desktop_mode
            or settings.companion_playback_enabled
            or body.interrupted
        ):
            raise HTTPException(409, "Text delivery requires muted desktop playback")
    store = cast(BackgroundMessageStore, request.app.state.background_message_store)
    proactive = getattr(request.app.state, "proactive_coordinator", None)
    extensions = getattr(request.app.state, "extension_service", None)
    stream = proactive.streams.get(message_id) if proactive else None
    owner = proactive
    if stream is None and extensions is not None:
        stream = extensions.streams.get(message_id)
        owner = extensions
    if stream is not None and owner is not None:
        if (stream.intent.fish_id, stream.intent.session_id) != (fish_id, session_id):
            raise HTTPException(404, "Companion message not found")
        await owner.streams.settle(message_id, interrupt=body.interrupted)
    message = await store.finalize_consumption(
        message_id,
        fish_id,
        session_id,
        body.consumed_offset,
        body.interrupted,
    )
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Companion message not found",
        )
    if proactive is not None:
        proactive.consumed(message_id, body.consumed_offset, body.interrupted)
    if extensions is not None:
        await extensions.consumed(message_id, body.consumed_offset, body.interrupted)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _require_owned_session(
    request: Request,
    fish_id: UUID,
    session_id: UUID,
) -> None:
    store = cast(SessionStore, request.app.state.session_store)
    try:
        await store.require_owned(fish_id, session_id)
    except SessionNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="session 不存在或不属于当前 fish",
        ) from error


def _last_event_sequence(request: Request) -> int:
    value = request.headers.get("last-event-id")
    if value is None or _NON_NEGATIVE_INTEGER_RE.fullmatch(value) is None:
        return 0
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed if parsed <= _MAX_SEQUENCE else 0


def _format_background_message(
    item: BackgroundAssistantMessage,
    *,
    streaming: bool = False,
    speech_content: str | None = None,
) -> str:
    context = item.reply_context
    data = json.dumps(
        {
            **({"streaming": True} if streaming else {}),
            **(
                {"speechContent": speech_content, "speechTags": True}
                if speech_content is not None
                else {}
            ),
            "source": item.source.value,
            "messageId": str(item.message_id),
            "sequence": item.sequence,
            "sessionId": str(item.session_id),
            "content": item.content,
            "consumedOffset": item.consumed_offset,
            "emotion": item.emotion,
            "interruptPolicy": item.interrupt_policy.value,
            "expiresAt": (
                None if item.expires_at is None else item.expires_at.isoformat()
            ),
            "supersessionKey": item.supersession_key,
            "replyContext": (
                None
                if context is None
                else {
                    "kind": context.kind.value,
                    "id": str(context.id),
                    "activationPolicy": context.activation_policy.value,
                }
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"id: {item.sequence}\nevent: companion.message\ndata: {data}\n\n"


__all__ = [
    "BackgroundMessageConsumptionRequest",
    "background_message_router",
    "consume_background_message",
    "stream_background_messages",
]
