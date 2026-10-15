"""Native interaction events enter History directly, without a summary model."""

import json
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fish_memory.observations.computer_history import ComputerHistoryEvent
from fish_memory.observations.models import (
    ImportantObservationEvent,
    redact_sensitive_observation_text,
)
from fish_memory.observations.store import _lock_active_fish, _recording_enabled

history_router = APIRouter(
    prefix="/internal/fishes/{fish_id}/history", include_in_schema=False
)


class HistoryEventsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[ComputerHistoryEvent] = Field(min_length=1, max_length=32)


def render_computer_event(event: ComputerHistoryEvent) -> str:
    parts = [event.kind]
    if event.app.name:
        parts.append(f"应用：{event.app.name}")
    if event.window.title:
        parts.append(f"窗口：{event.window.title}")
    for payload in (event.keyboard, event.selection, event.mouse):
        if payload is None:
            continue
        values = payload.model_dump(by_alias=True, exclude_none=True)
        for key in ("text", "keyEquivalent", "target"):
            value = values.get(key)
            if value:
                parts.append(
                    f"{key}："
                    + (
                        value
                        if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False)
                    )
                )
    return redact_sensitive_observation_text("；".join(parts))


@history_router.post("/events")
async def record_history_events(
    fish_id: UUID, body: HistoryEventsRequest, request: Request
) -> JSONResponse:
    history = request.app.state.session_history
    if not await request.app.state.fish_store.exists(fish_id):
        return JSONResponse(status_code=404, content={"error": "fish_not_found"})
    frame_cache = getattr(request.app.state, "environment_frame_cache", None)
    if frame_cache is not None and not frame_cache.policy(fish_id).operation_enabled:
        return JSONResponse(content={"status": "disabled"})
    for event in body.events:
        session = await history.session_for_time(fish_id, event.timestamp.timestamp())
        text = render_computer_event(event)
        identifier = uuid5(NAMESPACE_URL, f"fish-history:{fish_id}:{event.id}")
        payload = {
            "session_id": str(session) if session else None,
            "source_event_id": str(event.id),
            "context_text": text,
            "event": event.model_dump(mode="json", by_alias=True, exclude_none=True),
        }
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 32_000:
            payload["event"].pop("ax", None)
        if len(json.dumps(payload, ensure_ascii=False).encode()) > 32_000:
            payload["event"] = {
                key: payload["event"][key] for key in ("id", "timestamp", "kind")
            }
        validated = ImportantObservationEvent(
            source="computer_history",
            observation_type="computer.interaction",
            semantic_key=str(event.id),
            summary=text[:2_000],
            occurred_at=event.timestamp,
            payload=payload,
            app_bundle_id=event.app.bundle_identifier,
            app_name=event.app.name,
        )
        payload = dict(validated.payload)
        async with history.store.pool.acquire() as connection, connection.transaction():
            await _lock_active_fish(connection, fish_id)
            if not await _recording_enabled(
                connection, fish_id, event.app.bundle_identifier
            ):
                continue
            await connection.execute(
                """INSERT INTO observations
                   (id,fish_id,kind,source,observation_type,semantic_key,summary,payload,
                    started_at,ended_at,app_bundle_id,app_name)
                   VALUES($1,$2,'event','computer_history','computer.interaction',
                          $3,$4,$5::jsonb,$6,$6,$7,$8) ON CONFLICT(id) DO NOTHING""",
                identifier,
                fish_id,
                str(event.id),
                text[:2_000],
                json.dumps(payload, ensure_ascii=False),
                event.timestamp,
                event.app.bundle_identifier,
                event.app.name,
            )
    return JSONResponse(content={"accepted": len(body.events)})
