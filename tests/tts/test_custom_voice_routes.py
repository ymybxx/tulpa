import wave
from datetime import UTC, datetime
from io import BytesIO

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr

from fish_memory.tts.custom_voices import CustomVoice, CustomVoiceProviderError
from fish_memory.tts.routes import tts_internal_router


def make_wav(*, sample_rate: int = 16_000, duration_seconds: int = 5) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * sample_rate * duration_seconds)
    return output.getvalue()


class FakeVoiceClient:
    def __init__(
        self,
        *,
        fail: bool = False,
        create_error: CustomVoiceProviderError | None = None,
    ) -> None:
        self.created_audio: bytes | None = None
        self.deleted: list[str] = []
        self.fail = fail
        self.create_error = create_error

    async def list_voices(self) -> list[CustomVoice]:
        if self.fail:
            raise httpx.ConnectError("provider offline")
        return [
            CustomVoice(
                id="qwen-audio-3.0-tts-flash-fishabc-1",
                target_model="qwen-audio-3.0-tts-flash",
                status="OK",
                created_at=datetime(2026, 8, 10, tzinfo=UTC),
            )
        ]

    async def create_voice(self, audio: bytes, *, target_model: str) -> CustomVoice:
        if self.create_error is not None:
            raise self.create_error
        self.created_audio = audio
        return CustomVoice(
            id=f"{target_model}-fishabc-2",
            target_model=target_model,
            status="DEPLOYING",
        )

    async def delete_voice(self, voice_id: str) -> None:
        self.deleted.append(voice_id)


class FakeFishStore:
    def __init__(self, references: int = 0) -> None:
        self.references = references
        self.cleared: list[str] = []

    async def count_voice_references(self, voice_id: str) -> int:
        return self.references

    async def clear_voice_references(self, voice_id: str) -> None:
        self.cleared.append(voice_id)


def build_app(references: int = 0) -> tuple[FastAPI, FakeVoiceClient, FakeFishStore]:
    app = FastAPI()
    voice_client = FakeVoiceClient()
    fish_store = FakeFishStore(references)
    app.state.custom_voice_client = voice_client
    app.state.fish_store = fish_store
    app.state.settings = type(
        "Settings",
        (),
        {
            "tts_api_key": SecretStr("secret"),
            "tts_provider": "dashscope",
        },
    )()
    app.include_router(tts_internal_router)
    return app, voice_client, fish_store


@pytest.mark.asyncio
async def test_lists_and_creates_custom_voices_from_raw_wav() -> None:
    app, voice_client, _ = build_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/internal/tts/custom-voices")
        created = await client.post(
            "/internal/tts/custom-voices?targetModel=qwen-audio-3.0-tts-flash",
            content=make_wav(),
            headers={"Content-Type": "audio/wav"},
        )

    assert listed.status_code == 200
    assert listed.json()["items"][0]["status"] == "OK"
    assert created.status_code == 201
    assert voice_client.created_audio == make_wav()


@pytest.mark.asyncio
async def test_create_custom_voice_rejects_unsupported_wav_format() -> None:
    app, voice_client, _ = build_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/internal/tts/custom-voices?targetModel=qwen-audio-3.0-tts-flash",
            content=make_wav(sample_rate=8_000),
            headers={"Content-Type": "audio/wav"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_voice_audio"
    assert "16 kHz" in response.json()["error"]["message"]
    assert voice_client.created_audio is None


@pytest.mark.asyncio
async def test_delete_custom_voice_requires_confirmation_when_used() -> None:
    app, voice_client, fish_store = build_app(references=2)
    transport = httpx.ASGITransport(app=app)
    voice_id = "qwen-audio-3.0-tts-flash-fishabc-1"

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        blocked = await client.delete(f"/internal/tts/custom-voices/{voice_id}")
        deleted = await client.delete(
            f"/internal/tts/custom-voices/{voice_id}?force=true"
        )

    assert blocked.status_code == 409
    assert blocked.json()["error"]["details"]["fishCount"] == 2
    assert deleted.status_code == 204
    assert voice_client.deleted == [voice_id]
    assert fish_store.cleared == [voice_id]


@pytest.mark.asyncio
async def test_custom_voice_provider_failure_has_stable_api_error() -> None:
    app, _, _ = build_app()
    app.state.custom_voice_client = FakeVoiceClient(fail=True)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/internal/tts/custom-voices")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "custom_voice_provider_error"


@pytest.mark.asyncio
async def test_custom_voice_rejection_exposes_provider_reason() -> None:
    app, _, _ = build_app()
    app.state.custom_voice_client = FakeVoiceClient(
        create_error=CustomVoiceProviderError(
            status_code=400,
            provider_code="InvalidParameter",
            provider_message="The audio sample rate must be at least 16000 Hz.",
            request_id="request-123",
        )
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/internal/tts/custom-voices?targetModel=qwen-audio-3.0-tts-flash",
            content=make_wav(),
            headers={"Content-Type": "audio/wav"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "custom_voice_rejected",
            "message": (
                "百炼拒绝创建音色：The audio sample rate must be at least 16000 Hz."
            ),
            "details": {
                "providerCode": "InvalidParameter",
                "requestId": "request-123",
            },
        }
    }
