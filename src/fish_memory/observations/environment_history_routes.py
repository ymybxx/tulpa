"""Authenticated local handoff; HTTP never waits for ASR or a model call."""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from fish_memory.db.sessions import SessionNotFoundError
from fish_memory.observations.environment_history_models import (
    MAX_WINDOW_UPLOAD_BYTES,
    EnvironmentHistoryWindow,
    snapshot_id,
)
from fish_memory.observations.environment_history_service import (
    EnvironmentHistoryService,
    HistoryDisabled,
    HistoryQueueFull,
)
from fish_memory.observations.environment_history_store import (
    EnvironmentHistoryConflict,
)

environment_history_router = APIRouter(
    prefix="/internal/fishes/{fish_id}/environment-perception/history",
    tags=["internal", "environment-perception"],
    include_in_schema=False,
)


@environment_history_router.post("/windows")
async def upload_history_window(fish_id: UUID, request: Request) -> JSONResponse:
    if not await request.app.state.fish_store.exists(fish_id):
        return _error(404, "fish_not_found")
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > MAX_WINDOW_UPLOAD_BYTES:
            return _error(413, "history_window_too_large")
        raw.extend(chunk)
    try:
        window = EnvironmentHistoryWindow.model_validate_json(raw)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return _error(422, "invalid_history_window")
    now = request.app.state.clock.now().timestamp()
    if window.ended_at > now + 5 or window.ended_at < now - 300:
        return _error(422, "history_window_outside_capture_window")
    policy = request.app.state.environment_frame_cache.policy(fish_id)
    if not policy.history_enabled:
        return _error(409, "environment_perception_disabled")
    if window.frames and not policy.screen_enabled:
        return _error(409, "screen_perception_disabled")
    if window.audio_captured_samples and not policy.audio_enabled:
        return _error(409, "audio_perception_disabled")
    history = getattr(request.app.state, "session_history", None)
    if window.session_id is None and history is not None:
        session = await history.session_for_time(fish_id, window.started_at)
        window = window.model_copy(update={"session_id": session})
    elif window.session_id is not None:
        try:
            await request.app.state.session_store.require_owned(
                fish_id, window.session_id
            )
        except SessionNotFoundError:
            return _error(404, "session_not_found")
    service = _service(request)
    service.set_enabled(
        fish_id,
        policy.history_enabled,
    )
    try:
        state = await service.enqueue(fish_id, window)
    except HistoryDisabled:
        return _error(409, "environment_perception_disabled")
    except HistoryQueueFull:
        return _error(429, "history_queue_full")
    except EnvironmentHistoryConflict:
        return _error(409, "history_window_conflict")
    return JSONResponse(
        status_code=202,
        content={
            "snapshotId": str(
                snapshot_id(
                    fish_id,
                    window.capture_session_id,
                    window.window_index,
                    window.period_seconds,
                )
            ),
            "status": state,
        },
    )


@environment_history_router.get("/status")
async def history_status(fish_id: UUID, request: Request) -> JSONResponse:
    if not await request.app.state.fish_store.exists(fish_id):
        return _error(404, "fish_not_found")
    minute, tail = await request.app.state.environment_history_store.context_snapshots(
        fish_id
    )
    return JSONResponse(
        content={
            **_service(request).status(fish_id),
            "latestMinuteEndedAt": minute.ended_at.isoformat() if minute else None,
            "latestWindowEndedAt": tail[-1].ended_at.isoformat() if tail else None,
            "unmergedWindows": len(tail),
        }
    )


@environment_history_router.get("/windows/{window_id}")
async def history_window_status(
    fish_id: UUID, window_id: UUID, request: Request
) -> JSONResponse:
    if not await request.app.state.fish_store.exists(fish_id):
        return _error(404, "fish_not_found")
    stored = await request.app.state.environment_history_store.get(fish_id, window_id)
    return JSONResponse(
        content={
            "snapshotId": str(window_id),
            "status": "completed"
            if stored
            else _service(request).window_status(fish_id, window_id),
        }
    )


def _service(request: Request) -> EnvironmentHistoryService:
    return request.app.state.environment_history_service


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code}})
