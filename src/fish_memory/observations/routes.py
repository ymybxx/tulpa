from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from fish_memory.db.fishes import FishStore
from fish_memory.observations.computer_history import (
    ComputerHistoryCheckpointUpdateRequest,
    ComputerHistorySegmentConflictError,
    ComputerHistorySegmentRequest,
    ComputerHistorySummarizationError,
    ComputerHistorySummaryCheckpoint,
    ComputerHistorySummaryService,
    ComputerHistoryValidationError,
)
from fish_memory.observations.models import (
    AppRecordingPolicy,
    Observation,
    ObservationKind,
)
from fish_memory.observations.service import ObservationService
from fish_memory.observations.store import (
    ObservationFishNotFoundError,
    StaleStateObservationError,
)

logger = logging.getLogger(__name__)

observation_router = APIRouter(tags=["public", "observations"])
observation_internal_router = APIRouter(
    tags=["internal", "observations"],
    include_in_schema=False,
)


class StateObservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(min_length=1, max_length=64)
    observation_type: str = Field(
        validation_alias="observationType",
        min_length=1,
        max_length=64,
    )
    scope_key: str = Field(
        validation_alias="scopeKey",
        min_length=1,
        max_length=200,
    )
    semantic_key: str = Field(
        validation_alias="semanticKey",
        min_length=1,
        max_length=512,
    )
    summary: str = Field(min_length=1, max_length=2_000)
    payload: dict[str, object] = Field(default_factory=dict)
    app_bundle_id: str | None = Field(
        default=None,
        validation_alias="appBundleId",
        max_length=255,
    )
    app_name: str | None = Field(
        default=None,
        validation_alias="appName",
        max_length=200,
    )
    observed_at: AwareDatetime = Field(validation_alias="observedAt")


class ImportantObservationEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(min_length=1, max_length=64)
    observation_type: str = Field(
        validation_alias="observationType",
        min_length=1,
        max_length=64,
    )
    semantic_key: str = Field(
        validation_alias="semanticKey",
        min_length=1,
        max_length=512,
    )
    summary: str = Field(min_length=1, max_length=2_000)
    payload: dict[str, object] = Field(default_factory=dict)
    app_bundle_id: str | None = Field(
        default=None,
        validation_alias="appBundleId",
        max_length=255,
    )
    app_name: str | None = Field(
        default=None,
        validation_alias="appName",
        max_length=200,
    )
    occurred_at: AwareDatetime = Field(validation_alias="occurredAt")


class ObservationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: UUID
    sequence: int
    fish_id: UUID = Field(serialization_alias="fishId")
    kind: ObservationKind
    source: str
    observation_type: str = Field(serialization_alias="observationType")
    scope_key: str | None = Field(serialization_alias="scopeKey")
    semantic_key: str = Field(serialization_alias="semanticKey")
    summary: str
    payload: dict[str, object]
    app_bundle_id: str | None = Field(serialization_alias="appBundleId")
    app_name: str | None = Field(serialization_alias="appName")
    started_at: datetime = Field(serialization_alias="startedAt")
    ended_at: datetime = Field(serialization_alias="endedAt")
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")


class AppRecordingPolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enabled: bool = Field(strict=True)
    app_name: str | None = Field(
        default=None,
        validation_alias="appName",
        max_length=200,
    )


class AppRecordingPolicyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fish_id: UUID = Field(serialization_alias="fishId")
    app_bundle_id: str = Field(serialization_alias="appBundleId")
    app_name: str | None = Field(serialization_alias="appName")
    recording_enabled: bool = Field(serialization_alias="recordingEnabled")
    created_at: datetime = Field(serialization_alias="createdAt")
    updated_at: datetime = Field(serialization_alias="updatedAt")


class ObservationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ObservationResponse]


class AppRecordingPolicyListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AppRecordingPolicyResponse]


class AppRecordingPreflightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool


class ComputerHistoryCheckpointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = Field(serialization_alias="schemaVersion")
    segment_id: UUID = Field(serialization_alias="segmentId")
    started_at: datetime = Field(serialization_alias="startedAt")
    ended_at: datetime = Field(serialization_alias="endedAt")
    revision: int
    through_event_count: int = Field(serialization_alias="throughEventCount")
    through_byte_offset: int = Field(serialization_alias="throughByteOffset")
    through_suppressed_event_count: int = Field(
        serialization_alias="throughSuppressedEventCount"
    )
    delta_chain_digest: str = Field(serialization_alias="deltaChainDigest")
    last_successful_summary_at: datetime | None = Field(
        serialization_alias="lastSuccessfulSummaryAt"
    )
    status: str
    summary: str | None
    applications: list[str]
    finalized_at: datetime | None = Field(serialization_alias="finalizedAt")
    observation_id: UUID | None = Field(serialization_alias="observationId")
    final_event_digest: str | None = Field(serialization_alias="finalEventDigest")


@observation_internal_router.post(
    "/internal/fishes/{fish_id}/computer-history/segments/summarize",
    response_model=ObservationResponse,
    response_model_by_alias=True,
)
async def summarize_computer_history_segment(
    fish_id: UUID,
    body: ComputerHistorySegmentRequest,
    request: Request,
) -> ObservationResponse | JSONResponse:
    try:
        observation = await _computer_history_service(request).summarize_segment(
            fish_id,
            body,
        )
    except ObservationFishNotFoundError:
        return _fish_not_found()
    except ComputerHistorySegmentConflictError as error:
        return _error_response(
            status.HTTP_409_CONFLICT,
            "computer_history_segment_conflict",
            str(error),
        )
    except ComputerHistoryValidationError as error:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_computer_history_segment",
            str(error),
        )
    except ComputerHistorySummarizationError as error:
        return _error_response(
            status.HTTP_502_BAD_GATEWAY,
            "computer_history_summary_failed",
            str(error),
        )
    return _observation_response(observation)


@observation_internal_router.get(
    "/internal/fishes/{fish_id}/computer-history/segments/{segment_id}",
    response_model=ObservationResponse,
    response_model_by_alias=True,
)
async def get_computer_history_segment(
    fish_id: UUID,
    segment_id: UUID,
    request: Request,
) -> ObservationResponse | JSONResponse:
    observation = await _computer_history_service(request).get_segment(
        fish_id,
        segment_id,
    )
    if observation is None:
        return _observation_not_found()
    return _observation_response(observation)


@observation_internal_router.get(
    "/internal/fishes/{fish_id}/computer-history/segments/{segment_id}/checkpoint",
    response_model=ComputerHistoryCheckpointResponse,
    response_model_by_alias=True,
)
async def get_computer_history_checkpoint(
    fish_id: UUID,
    segment_id: UUID,
    request: Request,
) -> ComputerHistoryCheckpointResponse | JSONResponse:
    checkpoint = await _computer_history_service(request).get_checkpoint(
        fish_id,
        segment_id,
    )
    if checkpoint is None:
        return _error_response(
            status.HTTP_404_NOT_FOUND,
            "computer_history_checkpoint_not_found",
            "Computer History checkpoint does not exist",
        )
    return _checkpoint_response(checkpoint)


@observation_internal_router.post(
    "/internal/fishes/{fish_id}/computer-history/segments/{segment_id}/checkpoint",
    response_model=ComputerHistoryCheckpointResponse,
    response_model_by_alias=True,
)
async def update_computer_history_checkpoint(
    fish_id: UUID,
    segment_id: UUID,
    body: ComputerHistoryCheckpointUpdateRequest,
    request: Request,
) -> ComputerHistoryCheckpointResponse | JSONResponse:
    try:
        checkpoint = await _computer_history_service(request).update_checkpoint(
            fish_id,
            segment_id,
            body,
        )
    except ObservationFishNotFoundError:
        return _fish_not_found()
    except ComputerHistorySegmentConflictError as error:
        return _error_response(
            status.HTTP_409_CONFLICT,
            "computer_history_checkpoint_conflict",
            str(error),
        )
    except ComputerHistoryValidationError as error:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_computer_history_checkpoint",
            str(error),
        )
    except ComputerHistorySummarizationError as error:
        return _error_response(
            status.HTTP_502_BAD_GATEWAY,
            "computer_history_checkpoint_summary_failed",
            str(error),
        )
    return _checkpoint_response(checkpoint)


@observation_internal_router.delete(
    "/internal/fishes/{fish_id}/computer-history/segments/{segment_id}/checkpoint",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
async def acknowledge_computer_history_checkpoint(
    fish_id: UUID,
    segment_id: UUID,
    request: Request,
    observation_id: Annotated[UUID, Query(alias="observationId")],
    final_event_digest: Annotated[
        str,
        Query(
            alias="finalEventDigest",
            min_length=64,
            max_length=64,
            pattern=r"^[0-9a-f]{64}$",
        ),
    ],
) -> Response | JSONResponse:
    try:
        await _computer_history_service(request).acknowledge_checkpoint(
            fish_id,
            segment_id,
            observation_id=observation_id,
            final_event_digest=final_event_digest,
        )
    except ComputerHistorySegmentConflictError as error:
        return _error_response(
            status.HTTP_409_CONFLICT,
            "computer_history_checkpoint_ack_conflict",
            str(error),
        )
    except ComputerHistoryValidationError as error:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "invalid_computer_history_checkpoint_ack",
            str(error),
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@observation_internal_router.post(
    "/internal/fishes/{fish_id}/observations/states",
    response_model=ObservationResponse,
    response_model_by_alias=True,
)
async def record_state_observation(
    fish_id: UUID,
    body: StateObservationRequest,
    request: Request,
) -> ObservationResponse | Response | JSONResponse:
    service = _service(request)
    try:
        observation = await service.observe_state(
            fish_id,
            source=body.source,
            observation_type=body.observation_type,
            scope_key=body.scope_key,
            semantic_key=body.semantic_key,
            summary=body.summary,
            payload=body.payload,
            app_bundle_id=body.app_bundle_id,
            app_name=body.app_name,
            observed_at=body.observed_at,
        )
    except StaleStateObservationError as error:
        return _error_response(
            status.HTTP_409_CONFLICT,
            "stale_observation",
            str(error),
        )
    except ObservationFishNotFoundError:
        return _fish_not_found()
    except ValueError as error:
        return _invalid_observation(error)
    if observation is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    _notify_proactive_observation(request, observation)
    return _observation_response(observation)


@observation_internal_router.post(
    "/internal/fishes/{fish_id}/observations/events",
    response_model=ObservationResponse,
    response_model_by_alias=True,
)
async def record_important_observation_event(
    fish_id: UUID,
    body: ImportantObservationEventRequest,
    request: Request,
) -> ObservationResponse | Response | JSONResponse:
    service = _service(request)
    try:
        observation = await service.record_important_event(
            fish_id,
            source=body.source,
            observation_type=body.observation_type,
            semantic_key=body.semantic_key,
            summary=body.summary,
            payload=body.payload,
            app_bundle_id=body.app_bundle_id,
            app_name=body.app_name,
            occurred_at=body.occurred_at,
        )
    except ObservationFishNotFoundError:
        return _fish_not_found()
    except ValueError as error:
        return _invalid_observation(error)
    if observation is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    _notify_proactive_observation(request, observation)
    return _observation_response(observation)


@observation_internal_router.get(
    "/internal/fishes/{fish_id}/observation-app-policies/preflight",
    response_model=AppRecordingPreflightResponse,
)
async def preflight_observation_app_recording(
    fish_id: UUID,
    request: Request,
    app_bundle_id: Annotated[str | None, Query(alias="appBundleId")] = None,
) -> AppRecordingPreflightResponse:
    try:
        allowed = await _service(request).preflight_app_recording(
            fish_id,
            app_bundle_id,
        )
    except (ObservationFishNotFoundError, ValueError):
        allowed = False
    return AppRecordingPreflightResponse(allowed=allowed)


@observation_router.get(
    "/v1/fishes/{fish_id}/observations",
    response_model=ObservationListResponse,
    response_model_by_alias=True,
)
async def list_observations(
    fish_id: UUID,
    request: Request,
    start_at: Annotated[datetime | None, Query(alias="startAt")] = None,
    end_at: Annotated[datetime | None, Query(alias="endAt")] = None,
    app_bundle_id: Annotated[str | None, Query(alias="appBundleId")] = None,
    observation_type: Annotated[
        str | None,
        Query(alias="observationType"),
    ] = None,
    kind: ObservationKind | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> ObservationListResponse | JSONResponse:
    if not await _fish_exists(request, fish_id):
        return _fish_not_found()
    try:
        observations = await _service(request).query(
            fish_id,
            start_at=start_at,
            end_at=end_at,
            app_bundle_id=app_bundle_id,
            observation_type=observation_type,
            kind=kind,
            limit=limit,
        )
    except ValueError as error:
        return _invalid_observation(error)
    return ObservationListResponse(
        items=[_observation_response(item) for item in observations]
    )


@observation_router.get(
    "/v1/fishes/{fish_id}/observations/{observation_id}",
    response_model=ObservationResponse,
    response_model_by_alias=True,
)
async def get_observation(
    fish_id: UUID,
    observation_id: UUID,
    request: Request,
) -> ObservationResponse | JSONResponse:
    observation = await _service(request).get(fish_id, observation_id)
    if observation is None:
        return _observation_not_found()
    return _observation_response(observation)


@observation_router.delete(
    "/v1/fishes/{fish_id}/observations/{observation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
async def delete_observation(
    fish_id: UUID,
    observation_id: UUID,
    request: Request,
) -> Response | JSONResponse:
    deleted = await _service(request).delete(fish_id, observation_id)
    if not deleted:
        return _observation_not_found()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@observation_router.get(
    "/v1/fishes/{fish_id}/observation-app-policies",
    response_model=AppRecordingPolicyListResponse,
    response_model_by_alias=True,
)
async def list_observation_app_policies(
    fish_id: UUID,
    request: Request,
) -> AppRecordingPolicyListResponse | JSONResponse:
    if not await _fish_exists(request, fish_id):
        return _fish_not_found()
    policies = await _service(request).list_app_recording_policies(fish_id)
    return AppRecordingPolicyListResponse(
        items=[_policy_response(policy) for policy in policies]
    )


@observation_router.put(
    "/v1/fishes/{fish_id}/observation-app-policies/{app_bundle_id}",
    response_model=AppRecordingPolicyResponse,
    response_model_by_alias=True,
)
async def put_observation_app_policy(
    fish_id: UUID,
    app_bundle_id: str,
    body: AppRecordingPolicyRequest,
    request: Request,
) -> AppRecordingPolicyResponse | JSONResponse:
    try:
        policy = await _service(request).set_app_recording(
            fish_id,
            app_bundle_id,
            enabled=body.enabled,
            app_name=body.app_name,
        )
    except ObservationFishNotFoundError:
        return _fish_not_found()
    except ValueError as error:
        return _invalid_observation(error)
    return _policy_response(policy)


def _service(request: Request) -> ObservationService:
    service = getattr(request.app.state, "observation_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "observations_unavailable",
                "message": "Observation service is not initialized",
            },
        )
    return cast(ObservationService, service)


def _computer_history_service(request: Request) -> ComputerHistorySummaryService:
    service = getattr(request.app.state, "computer_history_summary_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "computer_history_summary_unavailable",
                "message": "Computer History summary service is not initialized",
            },
        )
    return cast(ComputerHistorySummaryService, service)


def _notify_proactive_observation(
    request: Request,
    observation: Observation,
) -> None:
    coordinator = getattr(request.app.state, "proactive_observation_coordinator", None)
    notify = getattr(coordinator, "notify", None)
    if not callable(notify):
        return
    try:
        notify(observation)
    except Exception:
        logger.exception(
            "Unable to notify proactive Observation coordinator for %s",
            observation.id,
        )


async def _fish_exists(request: Request, fish_id: UUID) -> bool:
    store = cast(FishStore, request.app.state.fish_store)
    return await store.exists(fish_id)


def _observation_response(observation: Observation) -> ObservationResponse:
    return ObservationResponse(
        id=observation.id,
        sequence=observation.sequence,
        fish_id=observation.fish_id,
        kind=observation.kind,
        source=observation.source,
        observation_type=observation.observation_type,
        scope_key=observation.scope_key,
        semantic_key=observation.semantic_key,
        summary=observation.summary,
        payload=dict(observation.payload),
        app_bundle_id=observation.app_bundle_id,
        app_name=observation.app_name,
        started_at=observation.started_at,
        ended_at=observation.ended_at,
        created_at=observation.created_at,
        updated_at=observation.updated_at,
    )


def _checkpoint_response(
    checkpoint: ComputerHistorySummaryCheckpoint,
) -> ComputerHistoryCheckpointResponse:
    return ComputerHistoryCheckpointResponse(
        schema_version=1,
        segment_id=checkpoint.segment_id,
        started_at=checkpoint.started_at,
        ended_at=checkpoint.ended_at,
        revision=checkpoint.revision,
        through_event_count=checkpoint.through_event_count,
        through_byte_offset=checkpoint.through_byte_offset,
        through_suppressed_event_count=(checkpoint.through_suppressed_event_count),
        delta_chain_digest=checkpoint.delta_chain_digest,
        last_successful_summary_at=checkpoint.last_successful_summary_at,
        status=checkpoint.status,
        summary=checkpoint.stored_summary,
        applications=list(checkpoint.applications),
        finalized_at=checkpoint.finalized_at,
        observation_id=checkpoint.observation_id,
        final_event_digest=checkpoint.final_event_digest,
    )


def _policy_response(policy: AppRecordingPolicy) -> AppRecordingPolicyResponse:
    return AppRecordingPolicyResponse(
        fish_id=policy.fish_id,
        app_bundle_id=policy.app_bundle_id,
        app_name=policy.app_name,
        recording_enabled=policy.recording_enabled,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
    )


def _fish_not_found() -> JSONResponse:
    return _error_response(
        status.HTTP_404_NOT_FOUND,
        "fish_not_found",
        "Fish does not exist",
    )


def _observation_not_found() -> JSONResponse:
    return _error_response(
        status.HTTP_404_NOT_FOUND,
        "observation_not_found",
        "Observation does not exist",
    )


def _invalid_observation(error: ValueError) -> JSONResponse:
    return _error_response(
        status.HTTP_400_BAD_REQUEST,
        "invalid_observation",
        str(error),
    )


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )
