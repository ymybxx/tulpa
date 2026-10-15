"""Native expression transport and bounded local diagnostics."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from fish_memory.background_messages.routes import _require_owned_session

proactive_router = APIRouter(
    prefix="/internal/proactive", tags=["internal"], include_in_schema=False
)


@proactive_router.get("/status")
async def status(request: Request) -> dict[str, object]:
    return request.app.state.proactive_coordinator.status()


@proactive_router.get("/{fish_id}/expressions/{message_id}/events")
async def expression_events(
    fish_id: UUID,
    message_id: UUID,
    request: Request,
    session_id: Annotated[UUID, Query(alias="sessionId")],
    offset: Annotated[int, Query(alias="afterOffset", ge=0)] = 0,
) -> StreamingResponse:
    await _require_owned_session(request, fish_id, session_id)
    stream = request.app.state.proactive_coordinator.streams.get(message_id)
    if stream is None or (stream.intent.fish_id, stream.intent.session_id) != (
        fish_id,
        session_id,
    ):
        raise HTTPException(404, "Expression not found")
    if offset > len(stream.text):
        raise HTTPException(409, "Expression cursor exceeds available text")

    async def events() -> AsyncIterator[str]:
        async for event in stream.events(offset):
            if await request.is_disconnected():
                return
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
