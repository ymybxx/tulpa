from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.service import ApiAuthService
from fish_memory.environment_frame_routes import environment_frame_router
from fish_memory.environment_frames import (
    EnvironmentFrameCache,
    EnvironmentFrameReferenceRequest,
    EnvironmentFrameUploadRequest,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000002")
JPEG = b"\xff\xd8\xff\xff\xd9"


@pytest.mark.asyncio
async def test_frame_reference_is_authenticated_fresh_and_scoped_to_cached_content(
    tmp_path,
):
    cache = EnvironmentFrameCache(now=lambda: 100)
    app = _app(tmp_path, cache)
    headers = {"Authorization": "Bearer fm_environment_secret"}
    base = f"/internal/fishes/{FISH_ID}/environment-perception"
    original = _body(observed_at=100)
    original["contentHash"] = hashlib.sha256(JPEG).hexdigest()
    reference = {
        **original,
        "frameId": str(UUID(int=44)),
        "revision": 44,
        "observedAt": 101,
        "expiresAt": 104,
    }
    reference.pop("jpegBase64")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.patch(f"{base}/frame", json=reference)).status_code == 401
        await client.put(f"{base}/state", headers=headers, json={"enabled": True})
        assert (
            await client.patch(f"{base}/frame", headers=headers, json=reference)
        ).status_code == 409
        assert (
            await client.put(f"{base}/frame", headers=headers, json=original)
        ).status_code == 204
        jpeg = cache.snapshot(FISH_ID).jpeg
        assert (
            await client.patch(f"{base}/frame", headers=headers, json=reference)
        ).status_code == 204
        latest = cache.snapshot(FISH_ID)
        assert latest.jpeg is jpeg
        assert latest.frame_id == UUID(int=44) and latest.observed_at == 101
        assert not cache.refresh(
            UUID(int=55), EnvironmentFrameReferenceRequest.model_validate(reference)
        )
        for patch in ({"contentHash": "a" * 64}, {"displayID": 2}, {"pixelWidth": 800}):
            assert (
                await client.patch(
                    f"{base}/frame", headers=headers, json={**reference, **patch}
                )
            ).status_code == 409
        await client.delete(f"{base}/frame", headers=headers)
        assert (
            await client.patch(f"{base}/frame", headers=headers, json=reference)
        ).status_code == 409


def test_frame_hash_is_verified_against_the_actual_jpeg():
    bad = {**_body(observed_at=100), "contentHash": "a" * 64}
    with pytest.raises(ValueError):
        EnvironmentFrameUploadRequest.model_validate(bad).decode_jpeg()


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeFishStore:
    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    async def exists(self, fish_id: UUID) -> bool:
        del fish_id
        return self._exists


def test_cache_retains_the_last_uploaded_frame_until_explicitly_cleared() -> None:
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    request = _request(observed_at=100)

    assert cache.put(FISH_ID, request, JPEG) is False
    cache.set_enabled(FISH_ID, True)
    assert cache.put(FISH_ID, request, JPEG) is True
    snapshot = cache.snapshot(FISH_ID)
    assert snapshot is not None
    assert snapshot.jpeg == JPEG
    assert snapshot.sha256 == (
        "7effd0cb9d1b578c75a7724ccbbe3d39357e15a95dca344d3dde7366a65b49c3"
    )

    clock.now = 103
    assert cache.snapshot(FISH_ID) is not None
    assert cache.state(FISH_ID).frame_available is True
    assert (
        cache.lease_frame(
            FISH_ID,
            SESSION_ID,
            "static-screen-turn",
            now=103,
            capability=True,
        )
        is not None
    )
    cache.clear(FISH_ID)
    assert cache.snapshot(FISH_ID) is None


def test_turn_lease_reuses_exact_frame_across_replacement_then_clears() -> None:
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_enabled(FISH_ID, True)
    first_request = _request(observed_at=100, frame_id=3)
    assert cache.put(FISH_ID, first_request, JPEG)
    first = cache.lease_frame(
        FISH_ID,
        SESSION_ID,
        "turn-1",
        now=100,
        capability=True,
    )
    assert first is not None

    clock.now = 101
    second_jpeg = b"\xff\xd8\xff\x01\xff\xd9"
    assert cache.put(
        FISH_ID,
        _request(observed_at=101, frame_id=4, byte_length=len(second_jpeg)),
        second_jpeg,
    )
    repeated = cache.lease_frame(
        FISH_ID,
        SESSION_ID,
        "turn-1",
        now=101,
        capability=True,
    )
    next_turn = cache.lease_frame(
        FISH_ID,
        SESSION_ID,
        "turn-2",
        now=101,
        capability=True,
    )

    assert repeated == first
    assert next_turn is not None and next_turn.jpeg == second_jpeg
    assert (
        cache.lease_frame(
            FISH_ID,
            SESSION_ID,
            "unsupported",
            now=101,
            capability=False,
        )
        is None
    )
    cache.set_enabled(FISH_ID, False)
    assert cache.is_enabled(FISH_ID) is False
    assert (
        cache.lease_frame(
            FISH_ID,
            SESSION_ID,
            "turn-1",
            now=102,
            capability=True,
        )
        is None
    )


def test_upload_model_rejects_wrong_ttl_dimensions_and_non_jpeg() -> None:
    with pytest.raises(ValidationError):
        EnvironmentFrameUploadRequest.model_validate(
            _body(observed_at=100) | {"expiresAt": 104}
        )
    with pytest.raises(ValidationError):
        EnvironmentFrameUploadRequest.model_validate(
            _body(observed_at=100) | {"pixelWidth": 961}
        )
    invalid = EnvironmentFrameUploadRequest.model_validate(
        _body(observed_at=100)
        | {
            "jpegBase64": base64.b64encode(b"abcde").decode(),
        }
    )
    with pytest.raises(ValueError, match="invalid JPEG payload"):
        invalid.decode_jpeg()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_history", [False, True])
async def test_internal_routes_require_auth_and_keep_only_memory_state(
    tmp_path: Path,
    with_history: bool,
) -> None:
    cache = EnvironmentFrameCache(now=lambda: 100)
    app = _app(tmp_path, cache)
    history_calls: list[tuple[UUID, bool]] = []

    class History:
        def set_enabled(self, fish_id: UUID, enabled: bool) -> None:
            history_calls.append((fish_id, enabled))

    if with_history:
        app.state.environment_history_service = History()
    probe_calls: list[None] = []
    app.state.request_environment_image_capability_probe = lambda: probe_calls.append(
        None
    )
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": "Bearer fm_environment_secret"}
    base = f"/internal/fishes/{FISH_ID}/environment-perception"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get(f"{base}/state")).status_code == 401
        enabled = await client.put(
            f"{base}/state",
            headers=headers,
            json={"enabled": True},
        )
        uploaded = await client.put(
            f"{base}/frame",
            headers=headers,
            json=_body(observed_at=100),
        )
        state = await client.get(f"{base}/state", headers=headers)
        cleared = await client.delete(f"{base}/frame", headers=headers)
        disabled = await client.put(
            f"{base}/state", headers=headers, json={"enabled": False}
        )

    assert enabled.json() == {
        "enabled": True,
        "operationEnabled": True,
        "screenEnabled": True,
        "audioEnabled": False,
        "audioSupported": True,
        "frameAvailable": False,
        "observedAt": None,
        "expiresAt": None,
    }
    assert probe_calls == [None]
    assert uploaded.status_code == 204 and uploaded.content == b""
    assert state.json()["frameAvailable"] is True
    assert cleared.status_code == 204 and cleared.content == b""
    assert cache.snapshot(FISH_ID) is None
    assert disabled.json()["enabled"] is False
    assert history_calls == (
        [(FISH_ID, True), (FISH_ID, False)] if with_history else []
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bits", range(8))
async def test_internal_policy_controls_capture_and_media_worker_independently(
    tmp_path, bits
):
    cache = EnvironmentFrameCache(now=lambda: 100)
    app = _app(tmp_path, cache)
    worker_calls = []

    class History:
        def set_enabled(self, fish_id, enabled):
            worker_calls.append((fish_id, enabled))

    app.state.environment_history_service = History()
    policy = {
        "operationEnabled": bool(bits & 1),
        "screenEnabled": bool(bits & 2),
        "audioEnabled": bool(bits & 4),
    }
    base = f"/internal/fishes/{FISH_ID}/environment-perception"
    headers = {"Authorization": "Bearer fm_environment_secret"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put(f"{base}/state", headers=headers, json=policy)
        assert response.status_code == 200
        assert {key: response.json()[key] for key in policy} == policy
        assert response.json()["enabled"] == bool(bits)
        frame = await client.put(
            f"{base}/frame", headers=headers, json=_body(observed_at=100)
        )
        assert frame.status_code == (204 if bits & 2 else 409)
    assert worker_calls == [(FISH_ID, bool(bits & 6))]


@pytest.mark.asyncio
@pytest.mark.parametrize("screen", [False, True])
async def test_unsupported_audio_state_keeps_only_requested_screen_input(
    tmp_path, screen
):
    cache = EnvironmentFrameCache(now=lambda: 100)
    app = _app(tmp_path, cache)
    base = f"/internal/fishes/{FISH_ID}/environment-perception"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put(
            f"{base}/state",
            headers={"Authorization": "Bearer fm_environment_secret"},
            json={
                "operationEnabled": False,
                "screenEnabled": screen,
                "audioEnabled": True,
                "audioSupported": False,
            },
        )
    assert response.status_code == 200
    assert response.json()["enabled"] is screen
    assert response.json()["screenEnabled"] is screen
    assert response.json()["audioEnabled"] is False
    assert response.json()["audioSupported"] is False
    assert cache.policy(FISH_ID).history_enabled is screen


def _request(
    *,
    observed_at: float,
    frame_id: int = 3,
    byte_length: int = len(JPEG),
) -> EnvironmentFrameUploadRequest:
    return EnvironmentFrameUploadRequest.model_validate(
        _body(
            observed_at=observed_at,
            frame_id=frame_id,
            byte_length=byte_length,
        )
    )


def _body(
    *,
    observed_at: float,
    frame_id: int = 3,
    byte_length: int = len(JPEG),
) -> dict[str, object]:
    jpeg = JPEG if byte_length == len(JPEG) else b"\xff\xd8\xff\x01\xff\xd9"
    return {
        "frameId": f"00000000-0000-0000-0000-{frame_id:012d}",
        "revision": frame_id,
        "observedAt": observed_at,
        "expiresAt": observed_at + 3,
        "displayID": 1,
        "pixelWidth": 960,
        "pixelHeight": 540,
        "byteLength": byte_length,
        "mimeType": "image/jpeg",
        "jpegBase64": base64.b64encode(jpeg).decode("ascii"),
    }


@pytest.mark.asyncio
async def test_debug_frame_observes_existing_authorized_bytes_only(tmp_path: Path):
    cache = EnvironmentFrameCache(now=lambda: 100.0)
    app = _app(tmp_path, cache)
    base = f"/internal/fishes/{FISH_ID}/environment-perception"
    headers = {"Authorization": "Bearer fm_environment_secret"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get(f"{base}/frame")).status_code == 401
        assert (await client.get(f"{base}/frame", headers=headers)).status_code == 404
        await client.put(f"{base}/state", headers=headers, json={"enabled": True})
        await client.put(f"{base}/frame", headers=headers, json=_body(observed_at=100))
        response = await client.get(f"{base}/frame", headers=headers)
        assert response.status_code == 200
        assert response.content == cache.snapshot(FISH_ID).jpeg
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-observed-at"] == "100.0"
        await client.put(f"{base}/state", headers=headers, json={"enabled": False})
        assert (await client.get(f"{base}/frame", headers=headers)).status_code == 404


def _app(tmp_path: Path, cache: EnvironmentFrameCache) -> FastAPI:
    credentials = ApiKeyCredentialStore(
        api_key="fm_environment_secret",
        key_file=tmp_path / "api-key",
    )
    app = FastAPI()
    app.state.environment_frame_cache = cache
    app.state.fish_store = FakeFishStore()
    app.add_middleware(
        ApiAuthMiddleware,
        auth_service=ApiAuthService(credentials),
    )
    app.include_router(environment_frame_router)
    return app


def test_policy_revision_tracks_changes_even_when_restoring_original_flags():
    cache = EnvironmentFrameCache()
    initial = cache.policy_revision
    cache.set_enabled(FISH_ID, False)
    assert cache.policy_revision == initial
    cache.set_enabled(FISH_ID, True)
    cache.set_enabled(FISH_ID, True)
    assert cache.policy_revision == initial + 1
    cache.set_enabled(FISH_ID, False)
    assert cache.policy_revision == initial + 2
