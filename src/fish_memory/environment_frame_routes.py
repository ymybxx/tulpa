"""Internal authenticated routes for the ephemeral environment frame slot."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from fish_memory.db.fishes import FishStore
from fish_memory.environment_frames import (
    EnvironmentFrameCache,
    EnvironmentFrameReferenceRequest,
    EnvironmentFrameUploadRequest,
    EnvironmentPerceptionState,
    EnvironmentPerceptionStateRequest,
)

environment_frame_router = APIRouter(
    prefix="/internal/fishes/{fish_id}/environment-perception",
    tags=["internal", "environment-perception"],
    include_in_schema=False,
)


class EnvironmentPerceptionStateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enabled: bool
    operation_enabled: bool = Field(serialization_alias="operationEnabled")
    screen_enabled: bool = Field(serialization_alias="screenEnabled")
    audio_enabled: bool = Field(serialization_alias="audioEnabled")
    audio_supported: bool = Field(serialization_alias="audioSupported")
    frame_available: bool = Field(serialization_alias="frameAvailable")
    observed_at: float | None = Field(
        default=None,
        serialization_alias="observedAt",
    )
    expires_at: float | None = Field(
        default=None,
        serialization_alias="expiresAt",
    )


@environment_frame_router.put(
    "/state",
    response_model=EnvironmentPerceptionStateResponse,
    response_model_by_alias=True,
)
async def set_environment_perception_state(
    fish_id: UUID,
    body: EnvironmentPerceptionStateRequest,
    request: Request,
) -> EnvironmentPerceptionStateResponse | JSONResponse:
    if not await _fish_store(request).exists(fish_id):
        return _fish_not_found()
    cache = _cache(request)
    policy = body.policy()
    cache.set_policy(fish_id, policy)
    proactive = getattr(request.app.state, "proactive_coordinator", None)
    if proactive is not None:
        proactive.policy_changed(fish_id)
    history = getattr(request.app.state, "environment_history_service", None)
    if history is not None:
        history.set_enabled(fish_id, policy.history_enabled)
    if policy.frames_enabled:
        probe = getattr(
            request.app.state,
            "request_environment_image_capability_probe",
            None,
        )
        if callable(probe):
            probe()
    return _state_response(cache.state(fish_id))


@environment_frame_router.get(
    "/state",
    response_model=EnvironmentPerceptionStateResponse,
    response_model_by_alias=True,
)
async def get_environment_perception_state(
    fish_id: UUID,
    request: Request,
) -> EnvironmentPerceptionStateResponse | JSONResponse:
    if not await _fish_store(request).exists(fish_id):
        return _fish_not_found()
    return _state_response(_cache(request).state(fish_id))


@environment_frame_router.put(
    "/frame",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
async def put_environment_frame(
    fish_id: UUID,
    body: EnvironmentFrameUploadRequest,
    request: Request,
) -> Response | JSONResponse:
    if not await _fish_store(request).exists(fish_id):
        return _fish_not_found()
    try:
        jpeg = body.decode_jpeg()
    except ValueError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"code": "invalid_environment_frame"}},
        )
    cache = _cache(request)
    if not cache.policy(fish_id).frames_enabled:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": {"code": "environment_perception_disabled"}},
        )
    # Stale/superseded frames are safe no-ops; the newest valid slot wins.
    cache.put(fish_id, body, jpeg)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@environment_frame_router.get("/frame", response_class=Response)
async def inspect_environment_frame(fish_id: UUID, request: Request) -> Response:
    """Inspect the authorized in-memory input and its capture time; do not capture."""
    if not await _fish_store(request).exists(fish_id):
        raise HTTPException(404, "Fish not found")
    cache = _cache(request)
    frame = cache.snapshot(fish_id)
    if frame is None or not cache.state(fish_id).frame_available:
        raise HTTPException(404, "No current environment frame")
    return Response(
        frame.jpeg,
        media_type=frame.mime_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Observed-At": str(frame.observed_at),
            "X-Display-Id": str(frame.display_id),
        },
    )


@environment_frame_router.patch(
    "/frame",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
async def refresh_environment_frame(
    fish_id: UUID, body: EnvironmentFrameReferenceRequest, request: Request
) -> Response | JSONResponse:
    if not await _fish_store(request).exists(fish_id):
        return _fish_not_found()
    if not _cache(request).refresh(fish_id, body):
        # The sender retains the JPEG and can retry one full upload after an
        # API restart, cache clear, binding change, or a missed earlier upload.
        return JSONResponse(
            status_code=409, content={"error": {"code": "frame_reference_unavailable"}}
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@environment_frame_router.delete(
    "/frame",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
async def clear_environment_frame(
    fish_id: UUID,
    request: Request,
) -> Response | JSONResponse:
    if not await _fish_store(request).exists(fish_id):
        return _fish_not_found()
    _cache(request).clear(fish_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _cache(request: Request) -> EnvironmentFrameCache:
    return request.app.state.environment_frame_cache


def _fish_store(request: Request) -> FishStore:
    return request.app.state.fish_store


def _state_response(
    value: EnvironmentPerceptionState,
) -> EnvironmentPerceptionStateResponse:
    return EnvironmentPerceptionStateResponse(
        enabled=value.enabled,
        operation_enabled=value.operation_enabled,
        screen_enabled=value.screen_enabled,
        audio_enabled=value.audio_enabled,
        audio_supported=value.audio_supported,
        frame_available=value.frame_available,
        observed_at=value.observed_at,
        expires_at=value.expires_at,
    )


def _fish_not_found() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": {"code": "fish_not_found"}},
    )
