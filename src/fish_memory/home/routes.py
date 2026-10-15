from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from fish_memory.db.home_actions import HomeActionRun, HomeActionStore
from fish_memory.db.sessions import SessionStore

home_router = APIRouter(prefix="/v1/fishes/{fish_id}/home-actions", tags=["public"])
_SSE_HEARTBEAT_SECONDS = 15.0


@home_router.get("")
async def list_home_actions(
    fish_id: UUID,
    request: Request,
    session_id: Annotated[UUID | None, Query(alias="sessionId")] = None,
    after_sequence: Annotated[int, Query(alias="afterSequence", ge=0)] = 0,
) -> dict[str, object]:
    resolved = await _session_id(request, fish_id, session_id)
    if resolved is None:
        return {"items": []}
    store = cast(HomeActionStore, request.app.state.home_action_store)
    items = await store.list_runs(
        fish_id, resolved, after_sequence=after_sequence, limit=200
    )
    return {"items": [_payload(item) for item in items]}


@home_router.get("/events", response_model=None)
async def stream_home_actions(
    fish_id: UUID,
    request: Request,
    session_id: Annotated[UUID | None, Query(alias="sessionId")] = None,
    after_event_sequence: Annotated[int, Query(alias="afterEventSequence", ge=0)] = 0,
) -> StreamingResponse:
    resolved = await _session_id(request, fish_id, session_id)
    store = cast(HomeActionStore, request.app.state.home_action_store)

    async def events() -> AsyncIterator[str]:
        cursor = after_event_sequence
        last_sent = time.monotonic()
        if last_id := request.headers.get("last-event-id"):
            try:
                cursor = max(cursor, int(last_id))
            except ValueError:
                pass
        while resolved is not None and not await request.is_disconnected():
            items = await store.list_events(
                fish_id, resolved, after_event_sequence=cursor, limit=100
            )
            if not items:
                if time.monotonic() - last_sent >= _SSE_HEARTBEAT_SECONDS:
                    last_sent = time.monotonic()
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.4)
                continue
            for item in items:
                cursor = item.event_sequence
                last_sent = time.monotonic()
                data = json.dumps(
                    _payload(item), ensure_ascii=False, separators=(",", ":")
                )
                yield (
                    f"id: {item.event_sequence}\n"
                    f"event: home.action.{item.status.value}\n"
                    f"data: {data}\n\n"
                )

    return StreamingResponse(events(), media_type="text/event-stream")


async def _session_id(
    request: Request, fish_id: UUID, session_id: UUID | None
) -> UUID | None:
    sessions = cast(SessionStore, request.app.state.session_store)
    if session_id is not None:
        return (await sessions.require_owned(fish_id, session_id)).id
    latest = await sessions.get_latest(fish_id)
    return latest.id if latest is not None else None


def _payload(item: HomeActionRun) -> dict[str, object]:
    return {
        "id": str(item.id),
        "sequence": item.sequence,
        "eventSequence": item.event_sequence,
        "sourceMessageId": str(item.source_message_id),
        "action": dict(item.action),
        "entityId": item.entity_id,
        "status": item.status.value,
        "accepted": item.accepted,
        "confirmed": item.confirmed,
        "result": item.result,
        "error": item.error,
        "createdAt": item.created_at.isoformat(),
        "updatedAt": item.updated_at.isoformat(),
    }
