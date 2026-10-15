from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.service import ApiAuthService
from fish_memory.environment_frames import (
    MAX_ENVIRONMENT_FRAME_BYTES,
    EnvironmentPerceptionPolicy,
)
from fish_memory.observations.environment_history_audio import SystemAudioTranscript
from fish_memory.observations.environment_history_client import (
    EnvironmentHistoryModelClient,
)
from fish_memory.observations.environment_history_models import (
    MAX_WINDOW_UPLOAD_BYTES,
    EnvironmentHistoryWindow,
    SemanticSnapshot,
    snapshot_id,
)
from fish_memory.observations.environment_history_routes import (
    environment_history_router,
)
from fish_memory.observations.environment_history_service import (
    EnvironmentHistoryService,
    HistoryDisabled,
    HistoryQueueFull,
)
from fish_memory.observations.environment_history_store import (
    EnvironmentHistoryConflict,
)

FISH = UUID(int=1)
CAPTURE = UUID(int=2)
START = datetime(2026, 9, 3, tzinfo=UTC).timestamp()
KEY = "fm_history_synthetic_test_key"
PATH = f"/internal/fishes/{FISH}/environment-perception/history"
_JPEG_BYTES = BytesIO()
with Image.new("RGB", (2, 1), "red") as _image:
    _image.save(_JPEG_BYTES, format="JPEG")
JPEG = _JPEG_BYTES.getvalue()


def _payload() -> dict[str, object]:
    return {
        "captureSessionId": str(CAPTURE),
        "windowIndex": 0,
        "startedAt": START,
        "endedAt": START + 10,
        "frames": [
            {
                "frameId": str(UUID(int=index + 10)),
                "revision": index + 1,
                "observedAt": START + offset,
                "expiresAt": START + offset + 3,
                "displayID": 1,
                "pixelWidth": 2,
                "pixelHeight": 1,
                "byteLength": len(JPEG),
                "mimeType": "image/jpeg",
                "jpegBase64": base64.b64encode(JPEG).decode(),
            }
            for index, offset in enumerate((1, 9))
        ],
        "audioStatus": "complete",
        "audioPcmBase64": base64.b64encode(b"\x01\x00" * 160_000).decode(),
        "audioSampleRate": 16000,
        "audioCapturedSamples": 160_000,
        "audioGapReason": None,
    }


class FakeFishStore:
    exists_value = True
    calls = 0

    async def exists(self, fish_id: UUID) -> bool:
        assert fish_id == FISH
        self.calls += 1
        return self.exists_value


class FakeHistoryService:
    def __init__(self) -> None:
        self.enabled = False
        self.enqueued: list[EnvironmentHistoryWindow] = []
        self.error: Exception | None = None

    def set_enabled(self, fish_id: UUID, enabled: bool) -> None:
        assert fish_id == FISH
        self.enabled = enabled

    async def enqueue(self, fish_id: UUID, window: EnvironmentHistoryWindow) -> str:
        assert fish_id == FISH
        if not self.enabled:
            raise HistoryDisabled()
        if self.error is not None:
            raise self.error
        self.enqueued.append(window)
        return "pending"

    def status(self, fish_id: UUID) -> dict[str, object]:
        assert fish_id == FISH
        return {"enabled": self.enabled, "pendingWindows": len(self.enqueued)}

    def window_status(self, fish_id: UUID, key: UUID) -> str:
        assert fish_id == FISH
        del key
        return "pending"


class FakeHistoryStore:
    async def context_snapshots(self, fish_id: UUID):
        assert fish_id == FISH
        return None, ()

    async def get(self, fish_id: UUID, key: UUID):
        assert fish_id == FISH
        del key
        # GET status must never serialize a stored document or raw material.
        return SimpleNamespace(document="private semantic content")


def _app(tmp_path: Path, *, enabled: bool = True):
    app = FastAPI()
    service = FakeHistoryService()
    app.state.fish_store = FakeFishStore()
    app.state.environment_history_service = service
    app.state.environment_history_store = FakeHistoryStore()
    app.state.environment_frame_cache = SimpleNamespace(
        policy=lambda _: EnvironmentPerceptionPolicy(
            screen_enabled=enabled, audio_enabled=enabled
        )
    )
    app.state.clock = SimpleNamespace(
        now=lambda: datetime.fromtimestamp(START + 10, UTC)
    )
    app.add_middleware(
        ApiAuthMiddleware,
        auth_service=ApiAuthService(
            ApiKeyCredentialStore(api_key=KEY, key_file=tmp_path / "api-key")
        ),
    )
    app.include_router(environment_history_router)
    return app, service


def _client(app: FastAPI, *, authenticated: bool = True) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://local.test",
        headers={"Authorization": f"Bearer {KEY}"} if authenticated else {},
    )


@pytest.mark.asyncio
async def test_accepts_two_timestamped_frames_and_ten_seconds_of_pcm(
    tmp_path,
):
    app, service = _app(tmp_path)
    payload = _payload()
    async with _client(app) as client:
        response = await client.post(f"{PATH}/windows", json=payload)
    assert response.status_code == 202
    assert response.json() == {
        "snapshotId": str(snapshot_id(FISH, CAPTURE, 0, 10)),
        "status": "pending",
    }
    assert len(service.enqueued) == 1
    accepted = service.enqueued[0]
    assert [frame.observed_at for frame in accepted.frames] == [START + 1, START + 9]
    # Expired live-frame TTL does not reject a historical capture inside this window.
    assert accepted.frames[0].expires_at < START + 10
    assert len(accepted.decode_pcm()) == 320_000
    assert "pcm" not in response.text.lower()
    assert "jpeg" not in response.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("screen,audio", [(True, False), (False, True)])
async def test_independent_inputs_reject_disabled_sources_and_accept_enabled_source(
    tmp_path, screen, audio
):
    app, service = _app(tmp_path)
    app.state.environment_frame_cache.policy = lambda _: EnvironmentPerceptionPolicy(
        screen_enabled=screen, audio_enabled=audio
    )
    async with _client(app) as client:
        rejected = await client.post(f"{PATH}/windows", json=_payload())
        assert rejected.status_code == 409
        assert rejected.json()["error"]["code"] == (
            "audio_perception_disabled" if screen else "screen_perception_disabled"
        )
        permitted = _payload()
        if screen:
            permitted.update(
                {
                    "audioStatus": "unavailable",
                    "audioCapturedSamples": 0,
                    "audioPcmBase64": None,
                    "audioGapReason": "system_audio_disabled",
                }
            )
        else:
            permitted["frames"] = []
        accepted = await client.post(f"{PATH}/windows", json=permitted)
    assert accepted.status_code == 202
    assert len(service.enqueued) == 1
    assert bool(service.enqueued[0].frames) == screen
    assert bool(service.enqueued[0].decode_pcm()) == audio


@pytest.mark.asyncio
async def test_accepts_ten_frames_but_rejects_eleven(tmp_path):
    app, service = _app(tmp_path)
    payload = _payload()
    original = cast(list[dict[str, object]], payload["frames"])[0]
    frames = [
        dict(
            original,
            frameId=str(UUID(int=100 + i)),
            observedAt=START + i,
            expiresAt=START + i + 3,
            revision=i + 1,
        )
        for i in range(11)
    ]
    async with _client(app) as client:
        response = await client.post(
            f"{PATH}/windows", json={**payload, "frames": frames[:10]}
        )
        assert response.status_code == 202
        response = await client.post(
            f"{PATH}/windows", json={**payload, "frames": frames}
        )
        assert response.status_code == 422
    assert len(service.enqueued) == 1
    assert len(service.enqueued[0].frames) == 10
    assert len(service.enqueued[0].decode_pcm()) == 320_000
    assert "pcm" not in response.text.lower()
    assert "jpeg" not in response.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [2, 5, 6, 9, 10])
async def test_accepts_aligned_early_cuts_and_rejects_mismatched_audio(
    tmp_path, seconds
):
    app, service = _app(tmp_path)
    payload = _payload()
    payload.update(
        endedAt=START + seconds,
        frames=cast(list[dict[str, object]], payload["frames"])[:1],
        audioCapturedSamples=seconds * 16000,
        audioPcmBase64=base64.b64encode(b"\x01\x00" * seconds * 16000).decode(),
    )
    async with _client(app) as client:
        assert (await client.post(f"{PATH}/windows", json=payload)).status_code == 202
        payload["endedAt"] = START + seconds + 1
        assert (await client.post(f"{PATH}/windows", json=payload)).status_code == 422
    assert len(service.enqueued[0].decode_pcm()) == seconds * 32000


@pytest.mark.asyncio
async def test_ten_maximum_size_frames_and_audio_fit_the_bounded_upload(tmp_path):
    app, service = _app(tmp_path)
    payload = _payload()
    jpeg = JPEG[:-2] + bytes(MAX_ENVIRONMENT_FRAME_BYTES - len(JPEG)) + JPEG[-2:]
    original = cast(list[dict[str, object]], payload["frames"])[0]
    payload["frames"] = [
        dict(
            original,
            frameId=str(UUID(int=100 + i)),
            observedAt=START + i,
            expiresAt=START + i + 3,
            byteLength=len(jpeg),
            jpegBase64=base64.b64encode(jpeg).decode(),
        )
        for i in range(10)
    ]
    body = json.dumps(payload).encode()
    assert 3_000_000 < len(body) < MAX_WINDOW_UPLOAD_BYTES
    async with _client(app) as client:
        assert (await client.post(f"{PATH}/windows", content=body)).status_code == 202
    assert len(service.enqueued[0].frames) == 10


@pytest.mark.asyncio
async def test_unauthorized_upload_does_not_reach_store_or_processing(tmp_path):
    app, service = _app(tmp_path)
    async with _client(app, authenticated=False) as client:
        response = await client.post(f"{PATH}/windows", json=_payload())
    assert response.status_code == 401
    assert app.state.fish_store.calls == 0
    assert service.enqueued == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "status", "code"),
    [
        ("missing", 404, "fish_not_found"),
        ("disabled", 409, "environment_perception_disabled"),
        ("queue", 429, "history_queue_full"),
        ("conflict", 409, "history_window_conflict"),
    ],
)
async def test_processing_admission_errors_have_stable_http_codes(
    tmp_path, kind, status, code
):
    app, service = _app(tmp_path, enabled=kind != "disabled")
    if kind == "missing":
        app.state.fish_store.exists_value = False
    if kind == "queue":
        service.error = HistoryQueueFull("private details")
    if kind == "conflict":
        service.error = EnvironmentHistoryConflict("private details")
    async with _client(app) as client:
        response = await client.post(f"{PATH}/windows", json=_payload())
    assert response.status_code == status
    assert response.json() == {"error": {"code": code}}
    assert service.enqueued == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "encoding",
        "unknown_field",
        "frame_order",
        "sample_count",
        "old_window",
        "index_overflow",
    ],
)
async def test_invalid_media_is_rejected_without_reflecting_input(tmp_path, invalid):
    app, service = _app(tmp_path)
    payload = _payload()
    if invalid == "encoding":
        payload["audioPcmBase64"] = "private invalid audio"
    elif invalid == "unknown_field":
        payload["rawTranscript"] = "private invalid content"
    elif invalid == "frame_order":
        payload["frames"] = list(
            reversed(cast(list[dict[str, object]], payload["frames"]))
        )
    elif invalid == "sample_count":
        payload["audioCapturedSamples"] = 100
    elif invalid == "index_overflow":
        payload["windowIndex"] = 2**63
    else:
        payload["frames"] = []
        payload["startedAt"] = START - 400
        payload["endedAt"] = START - 390
    async with _client(app) as client:
        response = await client.post(f"{PATH}/windows", json=payload)
    assert response.status_code == 422
    assert "private" not in response.text
    assert service.enqueued == []


@pytest.mark.asyncio
async def test_body_limit_stops_consuming_an_oversized_stream(tmp_path):
    app, service = _app(tmp_path)
    read: list[int] = []

    async def body():
        read.append(1)
        yield b"x" * (MAX_WINDOW_UPLOAD_BYTES // 2 + 1)
        read.append(2)
        yield b"x" * (MAX_WINDOW_UPLOAD_BYTES // 2 + 1)
        raise AssertionError("route must stop reading after its byte limit")

    async with _client(app) as client:
        response = await client.post(f"{PATH}/windows", content=body())
    assert response.status_code == 413
    assert read == [1, 2]
    assert not service.enqueued


@pytest.mark.asyncio
async def test_local_http_windows_use_asr_and_model_without_periodic_merge(
    tmp_path,
):
    app, _ = _app(tmp_path)
    allow_model = asyncio.Event()
    minute_saved = asyncio.Event()
    calls: list[dict[str, Any]] = []
    audio_inputs: list[bytes] = []

    class Provider:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            calls.append(kwargs)
            await allow_model.wait()
            content = kwargs["messages"][1]["content"]
            text = (
                '{"audiovisualSummary":"影片剧情继续发展。",'
                '"activitySummary":"用户切换应用。"}'
            )
            if isinstance(content, str):
                data = json.loads(content)
                if any("audio" in segment for segment in data.get("segments", [])):
                    text = json.dumps(
                        {
                            "paragraphs": [
                                {
                                    "text": "有人说了本地生成的转写。",
                                    "sources": [
                                        segment["sourceId"]
                                        for segment in data["segments"]
                                    ],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=text),
                    )
                ]
            )

        async def close(self):
            pass

    class Transcriber:
        async def transcribe(self, pcm):
            audio_inputs.append(pcm)
            return SystemAudioTranscript("只在本地生成的转写", "Speech")

        async def close(self):
            pass

    class Repository:
        def __init__(self):
            self.values: dict[UUID, SemanticSnapshot] = {}

        async def get(self, fish_id, key):
            assert fish_id == FISH
            return self.values.get(key)

        async def pending_minute_windows(
            self, fish_id, limit=4, *, since=None, through=None
        ):
            del fish_id, limit, since, through
            return ()

        async def delete_compacted_windows(self, fish_id, *, ended_before):
            del fish_id, ended_before
            return 0

        async def recent_hour_snapshots(self, fish_id, *, since, through):
            del since, through
            assert fish_id == FISH
            return tuple(self.values.values())

        async def save(self, value, *, evidence=None):
            self.values[value.id] = value
            if value.period_seconds == 60:
                minute_saved.set()
            assert "pcm" not in json.dumps(evidence).lower()
            return value

        async def minute_windows(self, fish_id, capture, first):
            assert fish_id == FISH and capture == CAPTURE
            return tuple(
                sorted(
                    (
                        v
                        for v in self.values.values()
                        if v.period_seconds == 10
                        and first <= v.window_index < first + 6
                    ),
                    key=lambda v: v.window_index,
                )
            )

        async def child_snapshots(
            self,
            fish_id,
            capture,
            first,
            *,
            child_period_seconds,
            parent_period_seconds,
        ):
            if parent_period_seconds != 60:
                return ()
            assert child_period_seconds == 10
            values = await self.minute_windows(fish_id, capture, first)
            return values if len(values) == 6 else ()

    repository = Repository()
    provider = Provider()

    def now():
        return datetime.fromtimestamp(START + 60, UTC)

    model = EnvironmentHistoryModelClient(
        api_key="unused",
        base_url="https://unused.invalid",
        model="qwen3.7-flash",
        client=cast(Any, provider),
    )
    service = EnvironmentHistoryService(
        cast(Any, repository), model, Transcriber(), now=now, retry_delays=()
    )
    app.state.environment_history_service = service
    app.state.environment_history_store = repository
    app.state.clock = SimpleNamespace(now=now)
    try:
        async with _client(app) as client:
            for index in range(6):
                payload = _payload()
                payload["windowIndex"] = index
                payload["startedAt"] = START + index * 10
                payload["endedAt"] = START + (index + 1) * 10
                for frame in cast(list[dict[str, Any]], payload["frames"]):
                    frame["observedAt"] += index * 10
                    frame["expiresAt"] += index * 10
                response = await client.post(f"{PATH}/windows", json=payload)
                assert response.status_code == 202
        # No HTTP request awaited model completion or minute compaction.
        assert not minute_saved.is_set()
        allow_model.set()
        from tests.observations.test_environment_history import settle

        await settle(service)
        assert not minute_saved.is_set()
        assert len(audio_inputs) == 6
        assert all(pcm == b"\x01\x00" * 160_000 for pcm in audio_inputs)
        windows = [c for c in calls if isinstance(c["messages"][1]["content"], list)]
        merges = [c for c in calls if isinstance(c["messages"][1]["content"], str)]
        assert len(windows) == 6 and not merges
        for call in windows:
            parts = call["messages"][1]["content"]
            assert sum(p["type"] == "image_url" for p in parts) == 1
            window_input = next(
                part["text"]
                for part in parts
                if part["type"] == "text"
                and part["text"].startswith("这是当前时段看到的画面：时间")
            )
            assert window_input.endswith("\n你听到的声音：只在本地生成的转写")
            assert "panels" not in window_input
    finally:
        allow_model.set()
        await service.close()


@pytest.mark.asyncio
async def test_oversized_upload_and_status_responses_never_return_raw_media(tmp_path):
    app, service = _app(tmp_path)
    async with _client(app) as client:
        response = await client.post(
            f"{PATH}/windows", content=b"x" * (MAX_WINDOW_UPLOAD_BYTES + 1)
        )
        status = await client.get(f"{PATH}/status")
        window = await client.get(f"{PATH}/windows/{snapshot_id(FISH, CAPTURE, 0, 10)}")
    assert response.status_code == 413
    assert response.json() == {"error": {"code": "history_window_too_large"}}
    assert status.status_code == 200 and status.json()["unmergedWindows"] == 0
    assert window.json()["status"] == "completed"
    assert "private" not in window.text
    assert service.enqueued == []
