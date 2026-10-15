"""Direct provider adapters using read-only desktop configuration."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from fish_memory.experimental.config import JEV_ENDPOINT, ExperimentalConfiguration
from fish_memory.experimental.decision import decision_connection
from fish_memory.experimental.jev import JevClient
from fish_memory.tts.models import TtsStartRequest
from fish_memory.tts.provider import (
    DEFAULT_DASHSCOPE_WEBSOCKET_URL,
    DashScopeTtsSession,
    build_cancel_task,
    build_continue_task,
    build_finish_task,
    build_run_task,
    parse_server_event,
)
from fish_memory.tts.voice_config import resolve_voice_config

from .core import Channel, decode_jev, questions, reasoning_prompt, validate_face
from .runtime_speech import RuntimeSpeech, runtime_audio


@dataclass(frozen=True)
class Configuration:
    reasoning_url: str
    reasoning_model: str
    reasoning_key: str = field(repr=False)
    jev_model: str
    jev_key: str = field(repr=False)
    speech_key: str = field(repr=False)
    tts_model: str
    tts_voice: str
    sample_rate: int = 22_050
    jev_endpoint: str = JEV_ENDPOINT
    runtime_speech: RuntimeSpeech | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            **(
                {"reasoningModel": self.reasoning_model, "reasoningThinking": False}
                if self.reasoning_model
                else {}
            ),
            "jevModel": self.jev_model,
            "ttsModel": self.tts_model,
            "ttsVoice": self.tts_voice,
            "sampleRate": self.sample_rate,
            **(
                {"ttsProvider": self.runtime_speech.provider}
                if self.runtime_speech
                else {}
            ),
        }


def load_configuration(
    directory: Path | None = None,
    *,
    require_reasoning: bool = True,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    runtime_speech: RuntimeSpeech | None = None,
) -> Configuration:
    directory = directory or (
        Path.home() / "Library/Application Support/fish-memory-desktop"
    )
    configuration = json.loads((directory / "config.json").read_text())
    secrets = json.loads((directory / "secrets.json").read_text())["values"]
    keys = secrets.get("modelApiKeys", {})
    assigned = configuration.get("modelAssignments", {})
    reasoning = (
        next(
            (
                model
                for model in configuration.get("models", {}).get("text", [])
                if model["id"] == assigned.get("reasoning")
            ),
            None,
        )
        if require_reasoning
        else None
    )
    speech = next(
        (
            model
            for model in configuration.get("models", {}).get("voice", [])
            if model["id"] == assigned.get("speech")
        ),
        None,
    )
    if runtime_speech is None and (speech is None or speech["provider"] != "dashscope"):
        raise ValueError("表情测试需要已配置的DashScope语音服务")
    if require_reasoning and reasoning is None:
        raise ValueError("对比Demo需要已配置的reasoning模型")
    reasoning_key = keys.get(reasoning["id"], "") if reasoning else ""
    decision = decision_connection(configuration, secrets)
    jev_key = decision.api_key if decision else ""
    speech_key = keys.get(speech["id"]) if speech and runtime_speech is None else ""
    required = (jev_key,) if runtime_speech else (jev_key, speech_key)
    if not all(isinstance(key, str) and key.strip() for key in required):
        if runtime_speech:
            raise ValueError("表情测试需要先接上决策模型（百炼或 Jev）")
        raise ValueError("表情测试需要先接上决策模型（百炼或 Jev）和 DashScope 语音")
    if require_reasoning and not (
        isinstance(reasoning_key, str) and reasoning_key.strip()
    ):
        raise ValueError("对比Demo所需的reasoning密钥尚未配置")
    voice = resolve_voice_config(None, provider="dashscope").config
    return Configuration(
        reasoning_url=reasoning["baseUrl"].rstrip("/") if reasoning else "",
        reasoning_model=reasoning["model"] if reasoning else "",
        reasoning_key=reasoning_key,
        jev_model=decision.model if decision else "",
        jev_key=jev_key,
        jev_endpoint=decision.endpoint if decision else JEV_ENDPOINT,
        speech_key=speech_key,
        tts_model=runtime_speech.model if runtime_speech else tts_model or voice.model,
        tts_voice=runtime_speech.voice if runtime_speech else tts_voice or voice.voice,
        sample_rate=runtime_speech.sample_rate if runtime_speech else 22_050,
        runtime_speech=runtime_speech,
    )


class ProviderFailure(Exception):
    """Safe error details with optional provider body, never authorization headers."""

    def __init__(self, message: str, raw: Any = None):
        super().__init__(message)
        self.raw = raw


class Providers:
    def __init__(self, configuration: Configuration, *, vary: bool = False):
        self.configuration = configuration
        # Speech varies unsure steps as TA does; batch comparisons stay exact.
        self.variation = random.Random() if vary else None
        self.http = httpx.AsyncClient(timeout=60)
        self.jev_http = httpx.AsyncClient(timeout=30)
        self.jev = JevClient(client=self.jev_http)
        self._completed = {"jev": 0, "reasoning": 0}

    async def close(self) -> None:
        await self.http.aclose()
        await self.jev.close()

    async def face(
        self, mode: str, state: dict[str, Any], channels: dict[str, Channel]
    ) -> dict[str, Any]:
        state = {
            key: state[key]
            for key in ("previousSegments", "currentSegment")
            if key in state
        }
        started = time.perf_counter()
        # Concurrent first requests both remain cold. "Warm" means this shared
        # HTTP client has completed a request, not proof of TCP connection reuse.
        connection_state = "cold" if self._completed[mode] == 0 else "warm"
        raw = None
        try:
            if mode == "jev":
                raw = await self.jev.evaluate(
                    ExperimentalConfiguration(
                        enabled=True,
                        api_key=SecretStr(self.configuration.jev_key),
                        model=self.configuration.jev_model,
                        endpoint=self.configuration.jev_endpoint,
                    ),
                    state,
                    questions(channels),
                    kind="demo.facial.jev",
                )
                targets = decode_jev(raw, channels, self.variation)
            elif mode == "reasoning":
                body = {
                    "model": self.configuration.reasoning_model,
                    "messages": [
                        {"role": "system", "content": reasoning_prompt(channels)},
                        {
                            "role": "user",
                            "content": json.dumps(state, ensure_ascii=False),
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "max_tokens": max(800, len(channels) * 60 + 128),
                    "stream": False,
                }
                if "dashscope.aliyuncs.com" in self.configuration.reasoning_url:
                    body["enable_thinking"] = False
                response = await self.http.post(
                    self.configuration.reasoning_url + "/chat/completions",
                    headers={
                        "Authorization": "Bearer " + self.configuration.reasoning_key
                    },
                    json=body,
                )
                response.raise_for_status()
                raw = response.json()
                targets = validate_face(
                    json.loads(raw["choices"][0]["message"]["content"]), channels
                )
            else:
                raise ValueError("未知表情模式")
        except httpx.HTTPStatusError as error:
            raise ProviderFailure(f"{mode} HTTP {error.response.status_code}") from None
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise ProviderFailure(
                f"{mode}返回格式无效：{type(error).__name__}", raw
            ) from None
        self._completed[mode] += 1
        return {
            "mode": mode,
            "targets": targets,
            "raw": raw,
            "elapsedMs": round((time.perf_counter() - started) * 1000, 2),
            "connectionState": connection_state,
            "warmthBasis": "prior_completed_request_on_shared_client",
            "cacheHit": False,
        }

    async def audio(self, text: str) -> AsyncIterator[bytes]:
        if self.configuration.runtime_speech is not None:
            try:
                async for pcm in runtime_audio(self.configuration.runtime_speech, text):
                    yield pcm
            except ValueError as error:
                raise ProviderFailure(str(error)) from None
            return
        session = DashScopeTtsSession(
            websocket_url=DEFAULT_DASHSCOPE_WEBSOCKET_URL,
            api_key=self.configuration.speech_key,
        )
        # The configured rate is a plain int; pydantic checks it is one DashScope takes.
        request = TtsStartRequest.model_validate(
            {
                "model": self.configuration.tts_model,
                "voice": self.configuration.tts_voice,
                "sample_rate": self.configuration.sample_rate,
            }
        )
        try:
            async with asyncio.timeout(60):
                await session.connect()
                await session.send(build_run_task(request, task_id=session.task_id))
                while True:
                    data = await session.receive()
                    if isinstance(data, bytes):
                        raise ProviderFailure("TTS在task-started前返回音频")
                    event = parse_server_event(json.loads(data))
                    if event.kind == "failed":
                        raise ProviderFailure("TTS启动失败", event.payload)
                    if event.kind == "started":
                        break
                await session.send(build_continue_task(session.task_id, text))
                await session.send(build_finish_task(session.task_id))
                while True:
                    data = await session.receive()
                    if isinstance(data, bytes):
                        yield data
                        continue
                    event = parse_server_event(json.loads(data))
                    if event.kind == "failed":
                        raise ProviderFailure("TTS合成失败", event.payload)
                    if event.kind == "finished":
                        return
        except asyncio.CancelledError:
            if session.is_connected:
                with suppress(Exception):
                    async with asyncio.timeout(1):
                        await session.send(build_cancel_task(session.task_id))
            raise
        finally:
            await session.close()
