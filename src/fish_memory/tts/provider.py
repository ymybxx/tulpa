import asyncio
import json
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import websockets
from websockets.exceptions import ConnectionClosed

from fish_memory.failures import ProviderError
from fish_memory.tts.models import TtsStartRequest

DEFAULT_DASHSCOPE_WEBSOCKET_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


@dataclass(frozen=True)
class TtsServerEvent:
    kind: str
    text: str | None = None
    sentence_index: int | None = None
    error: str | None = None
    payload: dict[str, Any] | None = None
    code: str | None = None

    def failure(self) -> ProviderError:
        return ProviderError(self.error or "TTS task failed", self.code)


def build_run_task(request: TtsStartRequest, *, task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": request.model,
            "parameters": {
                "text_type": "PlainText",
                "voice": request.voice,
                "format": request.audio_format,
                "sample_rate": request.sample_rate,
                "volume": 50,
                "rate": 1.0,
                "pitch": 1.0,
                "enable_ssml": False,
            },
            "input": {},
        },
    }


def build_continue_task(task_id: str, text: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "continue-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {"text": text}},
    }


def build_finish_task(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


def build_cancel_task(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {"directive": "cancel"}},
    }


def _output(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return {}
    output = payload.get("output")
    return output if isinstance(output, dict) else {}


def parse_server_event(event: dict[str, Any]) -> TtsServerEvent:
    header = event.get("header")
    header = header if isinstance(header, dict) else {}
    event_name = header.get("event") or header.get("action")
    output = _output(event)

    if event_name == "task-started":
        return TtsServerEvent(kind="started", payload=event)
    if event_name == "task-finished":
        return TtsServerEvent(kind="finished", payload=event)
    if event_name == "task-failed":
        message = output.get("message") or header.get("error_message")
        code = header.get("error_code")
        return TtsServerEvent(
            kind="failed",
            error=str(message or "TTS task failed"),
            payload=event,
            code=code if isinstance(code, str) else None,
        )
    if event_name != "result-generated":
        return TtsServerEvent(kind="other", payload=event)

    output_type = output.get("type")
    sentence_index = output.get("sentence_id")
    if not isinstance(sentence_index, int):
        sentence_index = output.get("sentence_index")
    sentence = output.get("sentence")
    if not isinstance(sentence_index, int) and isinstance(sentence, dict):
        sentence_index = sentence.get("index")
    if not isinstance(sentence_index, int):
        sentence_index = None
    original_text = output.get("original_text")
    text = (
        original_text
        if isinstance(original_text, str)
        else sentence
        if isinstance(sentence, str)
        else None
    )
    if output_type == "sentence-synthesis":
        return TtsServerEvent(
            kind="audio",
            text=text,
            sentence_index=sentence_index,
            payload=event,
        )
    if output_type == "sentence-begin":
        return TtsServerEvent(
            kind="sentence_begin",
            text=text,
            sentence_index=sentence_index,
            payload=event,
        )
    if output_type == "sentence-end":
        return TtsServerEvent(
            kind="sentence_end",
            text=text,
            sentence_index=sentence_index,
            payload=event,
        )
    return TtsServerEvent(kind="other", payload=event)


class DashScopeTtsSession:
    def __init__(self, *, websocket_url: str, api_key: str) -> None:
        self.websocket_url = websocket_url
        self.api_key = api_key
        self.task_id = str(uuid4())
        self._socket: Any = None
        self._send_lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        socket = self._socket
        if socket is None:
            return False
        if getattr(socket, "closed", False) is True:
            return False
        open_state = getattr(socket, "open", None)
        if open_state is False:
            return False
        state = getattr(socket, "state", None)
        if state is not None:
            state_name = getattr(state, "name", str(state)).lower()
            if state_name != "open":
                return False
        return True

    async def connect(self) -> None:
        self._socket = await websockets.connect(
            self.websocket_url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=8 * 1024 * 1024,
        )

    def begin_task(self) -> str:
        """Start a new provider task while keeping the WebSocket connection."""
        self.task_id = str(uuid4())
        return self.task_id

    async def send(self, message: dict[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError("TTS WebSocket is not connected")
        async with self._send_lock:
            try:
                await self._socket.send(json.dumps(message, ensure_ascii=False))
            except ConnectionClosed:
                self._socket = None
                raise

    async def receive(self) -> str | bytes:
        if self._socket is None:
            raise RuntimeError("TTS WebSocket is not connected")
        try:
            return await self._socket.recv()
        except ConnectionClosed:
            self._socket = None
            raise

    async def close(self) -> None:
        if self._socket is not None:
            try:
                with suppress(Exception):
                    await self._socket.close()
            finally:
                self._socket = None

    async def __aenter__(self) -> "DashScopeTtsSession":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
