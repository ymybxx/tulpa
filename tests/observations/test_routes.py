import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.service import ApiAuthService
from fish_memory.observations.computer_history import (
    COMPUTER_HISTORY_FORMAT,
    COMPUTER_HISTORY_OBSERVATION_TYPE,
    COMPUTER_HISTORY_SCOPE_KEY,
    COMPUTER_HISTORY_SOURCE,
    ComputerHistoryCheckpointUpdateRequest,
    ComputerHistorySegmentConflictError,
    ComputerHistorySegmentRequest,
    ComputerHistorySummaryCheckpoint,
    ComputerHistorySummaryService,
)
from fish_memory.observations.models import (
    AppRecordingPolicy,
    Observation,
    ObservationKind,
)
from fish_memory.observations.routes import (
    observation_internal_router,
    observation_router,
)
from fish_memory.observations.service import ObservationService
from fish_memory.observations.store import ObservationFishNotFoundError

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
OBSERVATION_ID = UUID("00000000-0000-0000-0000-000000000002")
NOW = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
SEGMENT_ID = UUID("00000000-0000-0000-0000-000000000003")
SEGMENT_ENDED = NOW + timedelta(minutes=10)


class FakeObservationService:
    def __init__(self) -> None:
        self.observation: Observation | None = _observation()
        self.state_calls: list[dict[str, object]] = []
        self.event_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []
        self.deleted: list[tuple[UUID, UUID]] = []
        self.policy_calls: list[dict[str, object]] = []
        self.preflight_calls: list[tuple[UUID, str | None]] = []
        self.preflight_allowed = True
        self.preflight_error: Exception | None = None

    async def observe_state(
        self,
        fish_id: UUID,
        **kwargs: object,
    ) -> Observation | None:
        self.state_calls.append({"fish_id": fish_id, **kwargs})
        return self.observation

    async def record_important_event(
        self,
        fish_id: UUID,
        **kwargs: object,
    ) -> Observation | None:
        self.event_calls.append({"fish_id": fish_id, **kwargs})
        return self.observation

    async def query(
        self,
        fish_id: UUID,
        **kwargs: object,
    ) -> tuple[Observation, ...]:
        self.query_calls.append({"fish_id": fish_id, **kwargs})
        return () if self.observation is None else (self.observation,)

    async def get(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> Observation | None:
        return self.observation

    async def delete(
        self,
        fish_id: UUID,
        observation_id: UUID,
    ) -> bool:
        self.deleted.append((fish_id, observation_id))
        return self.observation is not None

    async def set_app_recording(
        self,
        fish_id: UUID,
        app_bundle_id: str,
        **kwargs: object,
    ) -> AppRecordingPolicy:
        self.policy_calls.append(
            {
                "fish_id": fish_id,
                "app_bundle_id": app_bundle_id,
                **kwargs,
            }
        )
        return _policy()

    async def list_app_recording_policies(
        self,
        fish_id: UUID,
    ) -> tuple[AppRecordingPolicy, ...]:
        return (_policy(),)

    async def preflight_app_recording(
        self,
        fish_id: UUID,
        app_bundle_id: str | None,
    ) -> bool:
        self.preflight_calls.append((fish_id, app_bundle_id))
        if self.preflight_error is not None:
            raise self.preflight_error
        return self.preflight_allowed


class FakeFishStore:
    def __init__(self, *, exists: bool = True) -> None:
        self._exists = exists

    async def exists(self, fish_id: UUID) -> bool:
        return self._exists


class FakeProactiveCoordinator:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def notify(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


class FakeComputerHistoryService:
    def __init__(self) -> None:
        self.observation: Observation | None = _computer_history_observation()
        self.summary_calls: list[tuple[UUID, ComputerHistorySegmentRequest]] = []
        self.get_calls: list[tuple[UUID, UUID]] = []
        self.summary_error: Exception | None = None
        self.checkpoint: ComputerHistorySummaryCheckpoint | None = _checkpoint()
        self.checkpoint_calls: list[
            tuple[UUID, UUID, ComputerHistoryCheckpointUpdateRequest]
        ] = []
        self.checkpoint_get_calls: list[tuple[UUID, UUID]] = []
        self.checkpoint_error: Exception | None = None
        self.checkpoint_ack_calls: list[tuple[UUID, UUID, UUID, str]] = []
        self.checkpoint_ack_error: Exception | None = None

    async def summarize_segment(
        self,
        fish_id: UUID,
        body: ComputerHistorySegmentRequest,
    ) -> Observation:
        self.summary_calls.append((fish_id, body))
        if self.summary_error is not None:
            raise self.summary_error
        assert self.observation is not None
        return self.observation

    async def get_segment(
        self,
        fish_id: UUID,
        segment_id: UUID,
    ) -> Observation | None:
        self.get_calls.append((fish_id, segment_id))
        return self.observation if segment_id == SEGMENT_ID else None

    async def get_checkpoint(
        self,
        fish_id: UUID,
        segment_id: UUID,
    ) -> ComputerHistorySummaryCheckpoint | None:
        self.checkpoint_get_calls.append((fish_id, segment_id))
        return self.checkpoint if segment_id == SEGMENT_ID else None

    async def update_checkpoint(
        self,
        fish_id: UUID,
        segment_id: UUID,
        body: ComputerHistoryCheckpointUpdateRequest,
    ) -> ComputerHistorySummaryCheckpoint:
        self.checkpoint_calls.append((fish_id, segment_id, body))
        if self.checkpoint_error is not None:
            raise self.checkpoint_error
        assert self.checkpoint is not None
        return self.checkpoint

    async def acknowledge_checkpoint(
        self,
        fish_id: UUID,
        segment_id: UUID,
        *,
        observation_id: UUID,
        final_event_digest: str,
    ) -> bool:
        self.checkpoint_ack_calls.append(
            (fish_id, segment_id, observation_id, final_event_digest)
        )
        if self.checkpoint_ack_error is not None:
            raise self.checkpoint_ack_error
        return True


def _observation() -> Observation:
    return Observation(
        id=OBSERVATION_ID,
        sequence=7,
        fish_id=FISH_ID,
        kind=ObservationKind.STATE,
        source="workspace",
        observation_type="activity",
        scope_key="active-window",
        semantic_key="editing:fish-memory",
        summary="Editing Fish Memory",
        payload={"document": "observations.py"},
        app_bundle_id="com.microsoft.vscode",
        app_name="Visual Studio Code",
        started_at=NOW,
        ended_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _computer_history_body() -> dict[str, object]:
    event = {
        "schemaVersion": 1,
        "ordinal": 1,
        "id": "00000000-0000-0000-0000-000000000004",
        "timestamp": (NOW + timedelta(seconds=1))
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "kind": "window.changed",
        "app": {},
        "window": {},
    }
    events_jsonl = json.dumps(event, separators=(",", ":")) + "\n"
    return {
        "schemaVersion": 1,
        "segmentId": str(SEGMENT_ID),
        "startedAt": NOW.isoformat().replace("+00:00", "Z"),
        "endedAt": SEGMENT_ENDED.isoformat().replace("+00:00", "Z"),
        "eventCount": 1,
        "suppressedEventCount": 0,
        "eventDigest": hashlib.sha256(events_jsonl.encode()).hexdigest(),
        "eventsJsonl": events_jsonl,
    }


def _checkpoint_body() -> dict[str, object]:
    event = {
        "schemaVersion": 1,
        "ordinal": 1,
        "id": "00000000-0000-0000-0000-000000000004",
        "timestamp": (NOW + timedelta(seconds=1))
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "kind": "keyboard.text_input",
        "app": {"name": "ChatGPT"},
        "window": {"title": "Conversation"},
        "keyboard": {"text": "hello"},
    }
    delta = json.dumps(event, separators=(",", ":")) + "\n"
    return {
        "schemaVersion": 1,
        "requestId": "00000000-0000-0000-0000-000000000005",
        "startedAt": NOW.isoformat().replace("+00:00", "Z"),
        "endedAt": SEGMENT_ENDED.isoformat().replace("+00:00", "Z"),
        "baseRevision": 0,
        "fromEventCount": 0,
        "throughEventCount": 1,
        "fromByteOffset": 0,
        "throughByteOffset": len(delta.encode()),
        "throughSuppressedEventCount": 0,
        "deltaDigest": hashlib.sha256(delta.encode()).hexdigest(),
        "deltaEventsJsonl": delta,
        "finalize": False,
    }


def _checkpoint() -> ComputerHistorySummaryCheckpoint:
    summary = {
        "title": "Recent activity",
        "description": "The user typed in ChatGPT.",
        "memorySummary": "The user typed hello in a conversation.",
        "relevantPriorContext": [],
        "importantContext": [],
        "recordingSections": [{"heading": "Chat", "details": ["Typed hello."]}],
        "suggestion": None,
    }
    return ComputerHistorySummaryCheckpoint.model_validate(
        {
            "schemaVersion": 1,
            "fishId": str(FISH_ID),
            "segmentId": str(SEGMENT_ID),
            "startedAt": NOW.isoformat(),
            "endedAt": SEGMENT_ENDED.isoformat(),
            "revision": 1,
            "throughEventCount": 1,
            "throughByteOffset": cast(int, _checkpoint_body()["throughByteOffset"]),
            "throughSuppressedEventCount": 0,
            "deltaChainDigest": "a" * 64,
            "fullSummary": summary,
            "applications": ["ChatGPT"],
            "inheritedObservationIds": [],
            "deltaCapsules": [
                {
                    "requestId": _checkpoint_body()["requestId"],
                    "fromEventCount": 0,
                    "throughEventCount": 1,
                    "deltaDigest": _checkpoint_body()["deltaDigest"],
                    "summary": summary,
                    "applications": ["ChatGPT"],
                    "createdAt": (NOW + timedelta(minutes=1)).isoformat(),
                }
            ],
            "lastRequestId": _checkpoint_body()["requestId"],
            "lastDeltaDigest": _checkpoint_body()["deltaDigest"],
            "lastSuccessfulSummaryAt": (NOW + timedelta(minutes=1)).isoformat(),
            "status": "active",
            "finalizedAt": None,
            "observationId": None,
            "finalEventDigest": None,
            "createdAt": NOW.isoformat(),
            "updatedAt": (NOW + timedelta(minutes=1)).isoformat(),
        }
    )


def _computer_history_observation() -> Observation:
    body = _computer_history_body()
    return Observation(
        id=OBSERVATION_ID,
        sequence=8,
        fish_id=FISH_ID,
        kind=ObservationKind.STATE,
        source=COMPUTER_HISTORY_SOURCE,
        observation_type=COMPUTER_HISTORY_OBSERVATION_TYPE,
        scope_key=COMPUTER_HISTORY_SCOPE_KEY,
        semantic_key=f"segment:{SEGMENT_ID}",
        summary="Reviewed a window.",
        payload={
            "format": COMPUTER_HISTORY_FORMAT,
            "documentMarkdown": "# Reviewed a window\n",
            "title": "Reviewed a window",
            "description": "Reviewed a window.",
            "applications": [],
            "segmentId": str(SEGMENT_ID),
            "segmentStartedAt": body["startedAt"],
            "segmentEndedAt": body["endedAt"],
            "eventCount": 1,
            "suppressedEventCount": 0,
            "eventDigest": body["eventDigest"],
            "citations": [f"segment:{SEGMENT_ID}"],
        },
        app_bundle_id=None,
        app_name=None,
        started_at=NOW,
        ended_at=SEGMENT_ENDED,
        created_at=SEGMENT_ENDED,
        updated_at=SEGMENT_ENDED,
    )


def _policy() -> AppRecordingPolicy:
    return AppRecordingPolicy(
        fish_id=FISH_ID,
        app_bundle_id="com.example.private",
        app_name="Private App",
        recording_enabled=False,
        created_at=NOW,
        updated_at=NOW,
    )


def _app(
    service: FakeObservationService,
    *,
    fish_exists: bool = True,
    coordinator: FakeProactiveCoordinator | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.observation_service = cast(ObservationService, service)
    app.state.fish_store = FakeFishStore(exists=fish_exists)
    if coordinator is not None:
        app.state.proactive_observation_coordinator = coordinator
    app.include_router(observation_internal_router)
    app.include_router(observation_router)
    return app


def _computer_history_app(
    tmp_path: Path,
    service: FakeComputerHistoryService,
    *,
    coordinator: FakeProactiveCoordinator | None = None,
) -> FastAPI:
    credentials = ApiKeyCredentialStore(
        api_key="fm_computer_history_secret",
        key_file=tmp_path / "api-key",
    )
    auth = ApiAuthService(credentials)
    app = FastAPI()
    app.state.computer_history_summary_service = cast(
        ComputerHistorySummaryService,
        service,
    )
    if coordinator is not None:
        app.state.proactive_observation_coordinator = coordinator
    app.add_middleware(ApiAuthMiddleware, auth_service=auth)
    app.include_router(observation_internal_router)
    return app


@pytest.mark.asyncio
async def test_internal_state_endpoint_uses_structured_observation_service() -> None:
    service = FakeObservationService()
    coordinator = FakeProactiveCoordinator()
    transport = httpx.ASGITransport(app=_app(service, coordinator=coordinator))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/observations/states",
            json={
                "source": "workspace",
                "observationType": "activity",
                "scopeKey": "active-window",
                "semanticKey": "editing:fish-memory",
                "summary": "Editing Fish Memory",
                "payload": {"document": "observations.py"},
                "appBundleId": "com.microsoft.vscode",
                "appName": "Visual Studio Code",
                "observedAt": NOW.isoformat(),
            },
        )

    assert response.status_code == 200
    assert response.json()["observationType"] == "activity"
    assert response.json()["fishId"] == str(FISH_ID)
    assert service.state_calls[0]["semantic_key"] == "editing:fish-memory"
    assert coordinator.observations == [_observation()]


@pytest.mark.asyncio
async def test_recording_policy_suppression_returns_no_content() -> None:
    service = FakeObservationService()
    service.observation = None
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/observations/events",
            json={
                "source": "accessibility",
                "observationType": "error",
                "semanticKey": "build-error",
                "summary": "Build failed",
                "appBundleId": "com.example.private",
                "occurredAt": NOW.isoformat(),
            },
        )

    assert response.status_code == 204
    assert response.content == b""


@pytest.mark.asyncio
async def test_list_get_and_delete_are_fish_scoped() -> None:
    service = FakeObservationService()
    transport = httpx.ASGITransport(app=_app(service))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        listed = await client.get(
            f"/v1/fishes/{FISH_ID}/observations",
            params={
                "startAt": NOW.isoformat(),
                "appBundleId": "com.microsoft.vscode",
                "observationType": "activity",
                "kind": "state",
                "limit": 20,
            },
        )
        fetched = await client.get(
            f"/v1/fishes/{FISH_ID}/observations/{OBSERVATION_ID}"
        )
        deleted = await client.delete(
            f"/v1/fishes/{FISH_ID}/observations/{OBSERVATION_ID}"
        )

    assert listed.status_code == 200
    assert len(listed.json()["items"]) == 1
    assert fetched.status_code == 200
    assert deleted.status_code == 204
    assert service.query_calls[0]["app_bundle_id"] == "com.microsoft.vscode"
    assert service.deleted == [(FISH_ID, OBSERVATION_ID)]


@pytest.mark.asyncio
async def test_app_recording_policy_can_be_listed_and_changed() -> None:
    service = FakeObservationService()
    transport = httpx.ASGITransport(app=_app(service))
    url = f"/v1/fishes/{FISH_ID}/observation-app-policies"

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        listed = await client.get(url)
        updated = await client.put(
            f"{url}/com.example.private",
            json={"enabled": False, "appName": "Private App"},
        )

    assert listed.status_code == 200
    assert listed.json()["items"][0]["recordingEnabled"] is False
    assert updated.status_code == 200
    assert service.policy_calls[0]["enabled"] is False


@pytest.mark.asyncio
async def test_internal_app_recording_preflight_returns_only_allow_decision() -> None:
    service = FakeObservationService()
    transport = httpx.ASGITransport(app=_app(service))
    url = f"/internal/fishes/{FISH_ID}/observation-app-policies/preflight"

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            url,
            params={"appBundleId": "com.example.editor"},
        )

    assert response.status_code == 200
    assert response.json() == {"allowed": True}
    assert service.preflight_calls == [(FISH_ID, "com.example.editor")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ValueError("invalid bundle id"), ObservationFishNotFoundError(str(FISH_ID))],
)
async def test_internal_app_recording_preflight_fails_closed(
    error: Exception,
) -> None:
    service = FakeObservationService()
    service.preflight_error = error
    transport = httpx.ASGITransport(app=_app(service))
    url = f"/internal/fishes/{FISH_ID}/observation-app-policies/preflight"

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(url)

    assert response.status_code == 200
    assert response.json() == {"allowed": False}
    assert service.preflight_calls == [(FISH_ID, None)]


@pytest.mark.asyncio
async def test_list_returns_project_error_shape_for_missing_fish() -> None:
    service = FakeObservationService()
    transport = httpx.ASGITransport(app=_app(service, fish_exists=False))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(f"/v1/fishes/{FISH_ID}/observations")

    assert response.status_code == 404
    assert response.json() == {
        "error": {"code": "fish_not_found", "message": "Fish does not exist"}
    }


@pytest.mark.asyncio
async def test_computer_history_post_get_are_authenticated_and_non_proactive(
    tmp_path: Path,
) -> None:
    service = FakeComputerHistoryService()
    coordinator = FakeProactiveCoordinator()
    transport = httpx.ASGITransport(
        app=_computer_history_app(
            tmp_path,
            service,
            coordinator=coordinator,
        )
    )
    headers = {"Authorization": "Bearer fm_computer_history_secret"}
    body = _computer_history_body()

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.post(
            f"/internal/fishes/{FISH_ID}/computer-history/segments/summarize",
            json=body,
        )
        unauthorized_get = await client.get(
            f"/internal/fishes/{FISH_ID}/computer-history/segments/{SEGMENT_ID}"
        )
        posted = await client.post(
            f"/internal/fishes/{FISH_ID}/computer-history/segments/summarize",
            headers=headers,
            json=body,
        )
        fetched = await client.get(
            f"/internal/fishes/{FISH_ID}/computer-history/segments/{SEGMENT_ID}",
            headers=headers,
        )
        missing_id = UUID(int=999)
        missing = await client.get(
            f"/internal/fishes/{FISH_ID}/computer-history/segments/{missing_id}",
            headers=headers,
        )

    assert unauthorized.status_code == 401
    assert unauthorized_get.status_code == 401
    assert posted.status_code == 200
    assert fetched.status_code == 200
    assert missing.status_code == 404
    value = posted.json()
    assert value["scopeKey"] == COMPUTER_HISTORY_SCOPE_KEY
    assert value["semanticKey"] == f"segment:{SEGMENT_ID}"
    assert value["payload"]["segmentId"] == str(SEGMENT_ID)
    assert value["payload"]["eventDigest"] == body["eventDigest"]
    assert service.get_calls == [(FISH_ID, SEGMENT_ID), (FISH_ID, missing_id)]
    assert coordinator.observations == []


@pytest.mark.asyncio
async def test_computer_history_checkpoint_post_get_are_authenticated_and_compact(
    tmp_path: Path,
) -> None:
    service = FakeComputerHistoryService()
    transport = httpx.ASGITransport(app=_computer_history_app(tmp_path, service))
    path = (
        f"/internal/fishes/{FISH_ID}/computer-history/segments/{SEGMENT_ID}/checkpoint"
    )
    headers = {"Authorization": "Bearer fm_computer_history_secret"}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.get(path)
        posted = await client.post(path, headers=headers, json=_checkpoint_body())
        fetched = await client.get(path, headers=headers)
        missing = await client.get(
            (
                f"/internal/fishes/{FISH_ID}/computer-history/segments/"
                f"{UUID(int=999)}/checkpoint"
            ),
            headers=headers,
        )

    assert unauthorized.status_code == 401
    assert posted.status_code == 200
    assert fetched.status_code == 200
    assert missing.status_code == 404
    body = posted.json()
    assert body == fetched.json()
    assert body["schemaVersion"] == 1
    assert body["segmentId"] == str(SEGMENT_ID)
    assert body["revision"] == 1
    assert body["throughEventCount"] == 1
    assert body["lastSuccessfulSummaryAt"] is not None
    assert "typed hello" in body["summary"].lower()
    assert body["status"] == "active"
    assert body["observationId"] is None
    assert "fullSummary" not in body
    assert "deltaCapsules" not in body
    assert service.checkpoint_calls[0][:2] == (FISH_ID, SEGMENT_ID)


@pytest.mark.asyncio
async def test_computer_history_checkpoint_conflict_maps_to_409(tmp_path: Path) -> None:
    service = FakeComputerHistoryService()
    service.checkpoint_error = ComputerHistorySegmentConflictError(
        "checkpoint baseRevision is stale"
    )
    transport = httpx.ASGITransport(app=_computer_history_app(tmp_path, service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            (
                f"/internal/fishes/{FISH_ID}/computer-history/segments/"
                f"{SEGMENT_ID}/checkpoint"
            ),
            headers={"Authorization": "Bearer fm_computer_history_secret"},
            json=_checkpoint_body(),
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == ("computer_history_checkpoint_conflict")


@pytest.mark.asyncio
async def test_computer_history_checkpoint_ack_is_authenticated_and_idempotent(
    tmp_path: Path,
) -> None:
    service = FakeComputerHistoryService()
    transport = httpx.ASGITransport(app=_computer_history_app(tmp_path, service))
    digest = "a" * 64
    path = (
        f"/internal/fishes/{FISH_ID}/computer-history/segments/"
        f"{SEGMENT_ID}/checkpoint?observationId={OBSERVATION_ID}"
        f"&finalEventDigest={digest}"
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.delete(path)
        first = await client.delete(
            path,
            headers={"Authorization": "Bearer fm_computer_history_secret"},
        )
        retry = await client.delete(
            path,
            headers={"Authorization": "Bearer fm_computer_history_secret"},
        )

    assert unauthorized.status_code == 401
    assert first.status_code == 204 and first.content == b""
    assert retry.status_code == 204 and retry.content == b""
    assert service.checkpoint_ack_calls == [
        (FISH_ID, SEGMENT_ID, OBSERVATION_ID, digest),
        (FISH_ID, SEGMENT_ID, OBSERVATION_ID, digest),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement",
    [
        {"unexpected": True},
        {"eventCount": "1"},
        {"suppressedEventCount": False},
        {"startedAt": "2026-09-01T16:00:00+08:00"},
    ],
)
async def test_computer_history_request_envelope_is_strict(
    tmp_path: Path,
    replacement: dict[str, object],
) -> None:
    service = FakeComputerHistoryService()
    transport = httpx.ASGITransport(app=_computer_history_app(tmp_path, service))
    body = {**_computer_history_body(), **replacement}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/computer-history/segments/summarize",
            headers={"Authorization": "Bearer fm_computer_history_secret"},
            json=body,
        )

    assert response.status_code == 422
    assert service.summary_calls == []


@pytest.mark.asyncio
async def test_computer_history_segment_conflict_maps_to_409(tmp_path: Path) -> None:
    service = FakeComputerHistoryService()
    service.summary_error = ComputerHistorySegmentConflictError(
        "segmentId is already associated with different data"
    )
    transport = httpx.ASGITransport(app=_computer_history_app(tmp_path, service))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/computer-history/segments/summarize",
            headers={"Authorization": "Bearer fm_computer_history_secret"},
            json=_computer_history_body(),
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "computer_history_segment_conflict"
