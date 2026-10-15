"""Use the active character's voice through the running desktop API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from uuid import UUID

import httpx
import websockets
from websockets.exceptions import WebSocketException

_RECEIVE_TIMEOUT_SECONDS = 60.0
_STREAM_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class RuntimeSpeech:
    api_url: str
    api_key: str = field(repr=False)
    provider: str
    model: str
    voice: str
    sample_rate: int


async def load_runtime_speech(api_url: str, api_key: str) -> RuntimeSpeech:
    """Read the current binding each run; cloud credentials stay in the API."""
    try:
        url = httpx.URL(api_url)
    except httpx.InvalidURL:
        raise ValueError("桌面运行服务地址无效") from None
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise ValueError("桌面运行服务地址无效")
    api_url = str(url).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(
            headers=headers,
            timeout=10,
            trust_env=False,
        ) as client:
            response = await client.get(f"{api_url}/v1/fishes/assistant")
            response.raise_for_status()
            binding = response.json()
            fish_id = binding.get("fishId") if isinstance(binding, dict) else None
            if not isinstance(fish_id, str):
                raise ValueError
            fish_id = str(UUID(fish_id))
            response = await client.get(f"{api_url}/v1/fishes/{fish_id}/voice")
            response.raise_for_status()
            voice = response.json()
    except httpx.HTTPError:
        raise ValueError("无法读取当前角色的声音，请确认桌面服务已就绪") from None
    except ValueError:
        raise ValueError("当前角色或声音配置无效，请先选择角色并配置声音") from None

    if not isinstance(voice, dict) or not isinstance(voice.get("config"), dict):
        raise ValueError("当前角色的声音配置无效")
    provider = voice.get("provider")
    config = voice["config"]
    model, selected_voice = config.get("model"), config.get("voice")
    if (
        provider not in ("dashscope", "local")
        or not isinstance(model, str)
        or not model.strip()
        or len(model) > 120
        or not isinstance(selected_voice, str)
        or not selected_voice.strip()
        or len(selected_voice) > 120
        or config.get("kind")
        not in (("local",) if provider == "local" else ("preset", "custom"))
    ):
        raise ValueError("当前角色的声音配置无效")
    return RuntimeSpeech(
        api_url,
        api_key,
        provider,
        model,
        selected_voice,
        16_000 if provider == "local" else 24_000,
    )


async def runtime_audio(
    speech: RuntimeSpeech, text: str
) -> AsyncGenerator[bytes, None]:
    """Yield PCM using the same start/append/finish protocol as voice preview."""
    url = f"{speech.api_url.replace('http', 'ws', 1)}/v1/tts/stream"
    deadline = asyncio.get_running_loop().time() + _STREAM_TIMEOUT_SECONDS
    try:
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {speech.api_key}"},
            open_timeout=10,
            close_timeout=1,
            max_size=8 * 1024 * 1024,
            proxy=None,
        ) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "start",
                        "model": speech.model,
                        "voice": speech.voice,
                        "audio_format": "pcm",
                        "sample_rate": speech.sample_rate,
                    }
                )
            )
            ready = False
            received_audio = False
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                message = await asyncio.wait_for(
                    socket.recv(),
                    timeout=min(_RECEIVE_TIMEOUT_SECONDS, remaining),
                )
                if isinstance(message, bytes):
                    if not ready:
                        raise ValueError("语音服务在就绪前返回音频")
                    if message:
                        received_audio = True
                        yield message
                    continue
                try:
                    event = json.loads(message)
                except ValueError:
                    raise ValueError("语音服务返回了无效消息") from None
                if not isinstance(event, dict):
                    raise ValueError("语音服务返回了无效消息")
                if event.get("type") == "ready":
                    # The caller already fixed its PCM playback rate. Both
                    # runtime providers honor sample_rate, including local resampling.
                    if event.get("sampleRate") != speech.sample_rate:
                        raise ValueError("语音服务返回的采样率不匹配，请重试")
                    if not ready:
                        ready = True
                        await socket.send(json.dumps({"type": "append", "text": text}))
                        await socket.send(json.dumps({"type": "finish"}))
                elif event.get("type") == "error":
                    # Never echo remote bodies: provider errors can include secrets.
                    raise ValueError("语音生成失败，请检查当前角色的声音及语音服务配置")
                elif event.get("type") == "finished":
                    if event.get("stopped"):
                        raise ValueError("语音生成已中断")
                    if not received_audio:
                        raise ValueError("语音服务没有返回音频")
                    return
    except TimeoutError:
        raise ValueError("语音生成超时，请稍后重试") from None
    except (WebSocketException, OSError):
        raise ValueError("语音服务连接中断，请稍后重试") from None
