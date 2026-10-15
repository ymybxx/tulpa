import json

import httpx
import pytest

from fish_memory.tts.custom_voices import (
    CustomVoiceProviderError,
    DashScopeVoiceClient,
)


@pytest.mark.asyncio
async def test_create_voice_uploads_wav_and_enrolls_it() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "policy": "policy",
                        "signature": "signature",
                        "upload_dir": "dashscope-instant/account/path",
                        "upload_host": "https://upload.example.test",
                        "oss_access_key_id": "key-id",
                        "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true",
                    }
                },
            )
        if request.url.host == "upload.example.test":
            return httpx.Response(200)
        return httpx.Response(
            200,
            json={
                "output": {
                    "voice_id": "qwen-audio-3.0-tts-flash-fishabc-123",
                }
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    voices = DashScopeVoiceClient(
        api_key="secret",
        http_client=client,
        prefix_factory=lambda: "fishabc",
    )

    voice = await voices.create_voice(
        b"RIFFvalid-wave-data",
        target_model="qwen-audio-3.0-tts-flash",
    )

    assert voice.id == "qwen-audio-3.0-tts-flash-fishabc-123"
    assert voice.target_model == "qwen-audio-3.0-tts-flash"
    assert voice.status == "OK"
    assert [request.method for request in requests] == ["GET", "POST", "POST"]
    enrollment = json.loads(requests[-1].content)
    assert enrollment["input"] == {
        "action": "create_voice",
        "target_model": "qwen-audio-3.0-tts-flash",
        "prefix": "fishabc",
        "url": "oss://dashscope-instant/account/path/voice.wav",
        "language_hints": ["zh"],
        "enable_preprocess": True,
        "max_prompt_audio_length": 20.0,
    }
    assert requests[-1].headers["X-DashScope-OssResourceResolve"] == "enable"
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_rejection_preserves_dashscope_error_details() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "The audio sample rate must be at least 16000 Hz.",
                "request_id": "request-123",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    voices = DashScopeVoiceClient(api_key="secret", http_client=client)

    with pytest.raises(CustomVoiceProviderError) as caught:
        await voices.list_voices()

    assert caught.value.status_code == 400
    assert caught.value.provider_code == "InvalidParameter"
    assert caught.value.provider_message == (
        "The audio sample rate must be at least 16000 Hz."
    )
    assert caught.value.request_id == "request-123"
    await client.aclose()


@pytest.mark.asyncio
async def test_list_and_delete_custom_voices() -> None:
    actions: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        action = payload["input"]["action"]
        actions.append(action)
        if action == "list_voice":
            return httpx.Response(
                200,
                json={
                    "output": {
                        "voice_list": [
                            {
                                "voice_id": "qwen-audio-3.0-tts-flash-mine-1",
                                "status": "OK",
                                "gmt_create": "2026-08-10 12:00:00",
                            }
                        ]
                    }
                },
            )
        return httpx.Response(200, json={"output": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    voices = DashScopeVoiceClient(api_key="secret", http_client=client)

    result = await voices.list_voices()
    await voices.delete_voice(result[0].id)

    assert result[0].target_model == "qwen-audio-3.0-tts-flash"
    assert result[0].status == "OK"
    assert actions == ["list_voice", "delete_voice"]
    await client.aclose()
