"""Non-blocking companion-side publisher for desktop-pet events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, cast
from uuid import UUID

import websockets

from fish_memory.pet.protocol import pack_pcm_frame
from fish_memory.pet.types import PetEmotion

logger = logging.getLogger(__name__)


class PetEventSink(Protocol):
    def facial(self, payload: dict[str, object], generation: UUID) -> None: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def state(self, value: str, generation: UUID | None) -> None: ...

    def expression(
        self,
        emotion: PetEmotion,
        generation: UUID | None,
    ) -> None: ...

    def audio_start(self, generation: UUID, sample_rate: int) -> None: ...

    def audio_pause(self, generation: UUID | None) -> None: ...

    def audio_resume(self, generation: UUID | None) -> None: ...

    def audio_stop(self, generation: UUID | None) -> None: ...

    def cancel(self, generation: UUID | None) -> None: ...

    def playback_progress(self, payload: dict[str, object]) -> None: ...

    def companion_activity(self, payload: dict[str, object]) -> None: ...

    def proactive_started(self, payload: dict[str, object]) -> None: ...

    def hearing(self, text: str) -> None: ...

    def pcm(self, value: bytes) -> None: ...


class NoopPetEventSink:
    def facial(self, payload: dict[str, object], generation: UUID) -> None:
        del payload, generation

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def state(self, value: str, generation: UUID | None) -> None:
        del value, generation

    def expression(
        self,
        emotion: PetEmotion,
        generation: UUID | None,
    ) -> None:
        del emotion, generation

    def audio_start(self, generation: UUID, sample_rate: int) -> None:
        del generation, sample_rate

    def audio_pause(self, generation: UUID | None) -> None:
        del generation

    def audio_resume(self, generation: UUID | None) -> None:
        del generation

    def audio_stop(self, generation: UUID | None) -> None:
        del generation

    def cancel(self, generation: UUID | None) -> None:
        del generation

    def playback_progress(self, payload: dict[str, object]) -> None:
        del payload

    def companion_activity(self, payload: dict[str, object]) -> None:
        del payload

    def proactive_started(self, payload: dict[str, object]) -> None:
        del payload

    def hearing(self, text: str) -> None:
        del text

    def pcm(self, value: bytes) -> None:
        del value


@dataclass(frozen=True, slots=True)
class _Outbound:
    value: str | bytes
    control: bool


WebsocketConnector = Callable[[str], Awaitable[Any]]


class PetEventPublisher:
    """Reconnects in the background and never blocks the companion audio path."""

    def __init__(
        self,
        api_url: str,
        *,
        api_key: str | None,
        queue_size: int = 256,
        websocket_connector: WebsocketConnector | None = None,
    ) -> None:
        if queue_size < 2:
            raise ValueError("pet event queue must contain at least two events")
        self._url = _api_websocket_url(api_url, "/v1/pet/events/publish")
        self._api_key = api_key
        self._queue_size = queue_size
        self._connector = websocket_connector or self._connect
        self._queue: deque[_Outbound] = deque()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._socket: Any = None
        self._closed = False
        self._control_serial = 0
        self._latest_control: dict[str, tuple[int, str]] = {}
        self._has_connected = False
        self._interrupt_handler: Callable[[UUID, UUID], Awaitable[None]] | None = None
        self._listen_handler: Callable[[], Awaitable[bool]] | None = None
        self._stop_listening_handler: Callable[[], Awaitable[bool]] | None = None
        self._owner_voice_add_handler: Callable[[], Awaitable[None]] | None = None
        self._new_session_handler: Callable[[UUID, UUID], Awaitable[bool]] | None = None
        self._text_handler: (
            Callable[[UUID, str, tuple[UUID, ...]], Awaitable[dict[str, object]]] | None
        ) = None
        self._text_task: asyncio.Task[None] | None = None
        self._plugin_handler: (
            Callable[[frozenset[tuple[UUID, UUID]]], Awaitable[None]] | None
        ) = None
        self._audio_generation: UUID | None = None
        self._audio_suspended = False
        self._sample_position = 0
        self._pcm_chunk_bytes = 1_280
        self._pcm_buffer = bytearray()
        self._pcm_buffer_position = 0

    async def start(self) -> None:
        if self._task is None:
            self._closed = False
            self._task = asyncio.create_task(
                self._run(),
                name="desktop-pet-event-publisher",
            )

    async def close(self) -> None:
        self._closed = True
        if self._text_task is not None:
            self._text_task.cancel()
            await asyncio.gather(self._text_task, return_exceptions=True)
            self._text_task = None
        self._wake.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        socket = self._socket
        self._socket = None
        if socket is not None:
            with suppress(Exception):
                await socket.close()

    def state(self, value: str, generation: UUID | None) -> None:
        self._control("pet.state", generation, {"state": value})

    def facial(self, payload: dict[str, object], generation: UUID) -> None:
        self._control("pet.facial", generation, payload)

    def expression(
        self,
        emotion: PetEmotion,
        generation: UUID | None,
    ) -> None:
        self._control(
            "pet.expression",
            generation,
            {"emotion": emotion.value},
        )

    def audio_start(self, generation: UUID, sample_rate: int) -> None:
        self._audio_generation = generation
        self._audio_suspended = False
        self._sample_position = 0
        self._pcm_chunk_bytes = max(2, round(sample_rate * 0.04) * 2)
        self._pcm_buffer.clear()
        self._pcm_buffer_position = 0
        self._control(
            "pet.audio.start",
            generation,
            {"sampleRate": sample_rate, "format": "pcm_s16le", "channels": 1},
        )

    def audio_pause(self, generation: UUID | None) -> None:
        if generation is None or generation == self._audio_generation:
            self._audio_suspended = True
            self._discard_pcm_buffer()
        self._control("pet.audio.pause", generation, {})

    def audio_resume(self, generation: UUID | None) -> None:
        if generation is None or generation == self._audio_generation:
            self._audio_suspended = False
        self._control("pet.audio.resume", generation, {})

    def audio_stop(self, generation: UUID | None) -> None:
        if generation is None or generation == self._audio_generation:
            self._flush_pcm_buffer()
        self._control("pet.audio.stop", generation, {})
        if generation is None or generation == self._audio_generation:
            self._audio_generation = None
            self._audio_suspended = False
            self._sample_position = 0
            self._pcm_buffer.clear()

    def cancel(self, generation: UUID | None) -> None:
        if generation is None or generation == self._audio_generation:
            self._discard_pcm_buffer()
        self._control("pet.generation.cancel", generation, {})
        if generation is None or generation == self._audio_generation:
            self._audio_generation = None
            self._audio_suspended = False
            self._sample_position = 0

    def pcm(self, value: bytes) -> None:
        generation = self._audio_generation
        if generation is None or not value or len(value) % 2:
            return
        if self._audio_suspended:
            self._sample_position += len(value) // 2
            return
        if not self._pcm_buffer:
            self._pcm_buffer_position = self._sample_position
        self._pcm_buffer.extend(value)
        self._sample_position += len(value) // 2
        while len(self._pcm_buffer) >= self._pcm_chunk_bytes:
            chunk = bytes(self._pcm_buffer[: self._pcm_chunk_bytes])
            del self._pcm_buffer[: self._pcm_chunk_bytes]
            self._enqueue(
                _Outbound(
                    pack_pcm_frame(generation, self._pcm_buffer_position, chunk),
                    control=False,
                )
            )
            self._pcm_buffer_position += len(chunk) // 2

    def _flush_pcm_buffer(self) -> None:
        generation = self._audio_generation
        if generation is None or not self._pcm_buffer:
            return
        chunk = bytes(self._pcm_buffer)
        self._pcm_buffer.clear()
        self._enqueue(
            _Outbound(
                pack_pcm_frame(generation, self._pcm_buffer_position, chunk),
                control=False,
            )
        )

    def _discard_pcm_buffer(self) -> None:
        self._pcm_buffer.clear()
        self._pcm_buffer_position = self._sample_position

    def playback_progress(self, payload: dict[str, object]) -> None:
        self._control(
            "pet.playback.progress", UUID(str(payload["playbackId"])), payload
        )

    def companion_activity(self, payload: dict[str, object]) -> None:
        self._control("pet.companion.activity", None, payload)

    def proactive_started(self, payload: dict[str, object]) -> None:
        self._control("pet.proactive.started", None, payload)

    def hearing(self, text: str) -> None:
        self._control("pet.voice.hearing", None, {"text": text[-2000:]})

    def owner_voice_unrecognized(self, payload: dict[str, object]) -> None:
        self._control("pet.owner_voice.unrecognized", None, payload)

    def _control(
        self,
        event_type: str,
        generation: UUID | None,
        payload: dict[str, object],
    ) -> None:
        value = json.dumps(
            {
                "type": event_type,
                "generation": str(generation) if generation is not None else None,
                "payload": payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._control_serial += 1
        key = event_type
        if event_type == "pet.playback.progress":
            key += ":" + str(payload["messageId"])
        self._latest_control[key] = (self._control_serial, value)
        progress_keys = [
            key
            for key in self._latest_control
            if key.startswith("pet.playback.progress:")
        ]
        for key in progress_keys[:-32]:
            del self._latest_control[key]
        self._enqueue(_Outbound(value, control=True))

    def _enqueue(self, outbound: _Outbound) -> None:
        if self._closed:
            return
        if len(self._queue) >= self._queue_size:
            for index, queued in enumerate(self._queue):
                if not queued.control:
                    del self._queue[index]
                    break
            else:
                if not outbound.control:
                    return
                self._queue.popleft()
        self._queue.append(outbound)
        self._wake.set()

    def set_answer_interrupt_handler(
        self, handler: Callable[[UUID, UUID], Awaitable[None]]
    ) -> None:
        self._interrupt_handler = handler

    def set_text_input_handler(
        self,
        handler: Callable[[UUID, str, tuple[UUID, ...]], Awaitable[dict[str, object]]],
    ) -> None:
        self._text_handler = handler

    async def _submit_text(self, payload: dict[str, object]) -> None:
        result: dict[str, object] = {"requestId": payload["requestId"]}
        try:
            assert self._text_handler is not None
            attachment_ids = payload.get("attachmentIds") or []
            submitted = await self._text_handler(
                UUID(str(payload["fishId"])),
                str(payload["content"]),
                tuple(UUID(str(item)) for item in cast(list[object], attachment_ids)),
            )
            result.update(ok=True, **submitted)
        except ValueError as error:
            result.update(ok=False, error=str(error), retryable=True)
        except Exception:
            logger.exception(
                "Quick-chat acceptance failed: request_id=%s", payload["requestId"]
            )
            result.update(
                ok=False, error="发送状态未确认，请先查看聊天记录；草稿已保留"
            )
        self._control("pet.chat.result", None, result)

    def set_plugin_playback_handler(
        self, handler: Callable[[frozenset[tuple[UUID, UUID]]], Awaitable[None]]
    ) -> None:
        self._plugin_handler = handler

    def set_microphone_handler(
        self, handler: Callable[[bool], Awaitable[None]]
    ) -> None:
        self._microphone_handler = handler

    def set_new_session_handler(
        self, handler: Callable[[UUID, UUID], Awaitable[bool]]
    ) -> None:
        self._new_session_handler = handler

    def set_listen_handler(self, handler: Callable[[], Awaitable[bool]]) -> None:
        self._listen_handler = handler

    def set_stop_listening_handler(
        self, handler: Callable[[], Awaitable[bool]]
    ) -> None:
        self._stop_listening_handler = handler

    def set_owner_voice_add_handler(
        self, handler: Callable[[], Awaitable[None]]
    ) -> None:
        self._owner_voice_add_handler = handler

    def set_playback_handler(self, handler: Callable[[bool], Awaitable[None]]) -> None:
        self._playback_handler = handler

    async def _read_commands(self, socket: Any) -> None:
        while True:
            raw = await socket.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if (
                event.get("type") == "pet.chat.submit"
                and self._text_handler is not None
            ):
                payload = event.get("payload", {})
                if self._text_task is not None and not self._text_task.done():
                    self._control(
                        "pet.chat.result",
                        None,
                        {
                            "requestId": payload["requestId"],
                            "ok": False,
                            "error": "上一条文字正在发送，请稍后重试",
                            "retryable": True,
                        },
                    )
                else:
                    self._text_task = asyncio.create_task(self._submit_text(payload))
                continue
            if event.get("type") == "pet.listen":
                if self._listen_handler is not None:
                    await self._listen_handler()
                continue
            if event.get("type") == "pet.session.new":
                if self._new_session_handler is not None:
                    payload = event.get("payload", {})
                    await self._new_session_handler(
                        UUID(payload["fishId"]), UUID(payload["sessionId"])
                    )
                continue
            if event.get("type") == "pet.listen.stop":
                if self._stop_listening_handler is not None:
                    await self._stop_listening_handler()
                continue
            if event.get("type") == "pet.owner_voice.add":
                if self._owner_voice_add_handler is not None:
                    await self._owner_voice_add_handler()
                continue
            if event.get("type") in {"pet.microphone", "pet.speaker"}:
                enabled = event.get("payload", {}).get("enabled")
                handler = getattr(
                    self,
                    "_microphone_handler"
                    if event["type"] == "pet.microphone"
                    else "_playback_handler",
                    None,
                )
                if isinstance(enabled, bool) and handler is not None:
                    await handler(enabled)
                    self._control(event["type"] + ".state", None, {"enabled": enabled})
                continue
            if (
                event.get("type") == "pet.answer.interrupt"
                and self._interrupt_handler is not None
            ):
                payload = event["payload"]
                await self._interrupt_handler(
                    UUID(payload["fishId"]), UUID(payload["messageId"])
                )
            elif event.get("type") == "pet.plugin.sessions" and self._plugin_handler:
                await self._plugin_handler(
                    frozenset(
                        (UUID(item["fishId"]), UUID(item["sessionId"]))
                        for item in event["payload"]["sessions"]
                    )
                )

    async def _run(self) -> None:
        retry_seconds = 0.25
        while not self._closed:
            command_task = None
            try:
                socket = await self._connector(self._url)
                self._socket = socket
                if callable(getattr(socket, "recv", None)):
                    command_task = asyncio.create_task(self._read_commands(socket))
                    command_task.add_done_callback(lambda _: self._wake.set())
                retry_seconds = 0.25
                if self._has_connected:
                    for _, cached in sorted(self._latest_control.values()):
                        await socket.send(cached)
                self._has_connected = True
                while not self._closed:
                    if command_task is not None and command_task.done():
                        command_task.result()
                        raise ConnectionError("Companion control connection ended")
                    if not self._queue:
                        self._wake.clear()
                        await self._wake.wait()
                        continue
                    outbound = self._queue[0]
                    await socket.send(outbound.value)
                    self._queue.popleft()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug(
                    "Desktop-pet event connection unavailable; retrying",
                    exc_info=True,
                )
                try:
                    await asyncio.sleep(retry_seconds)
                except asyncio.CancelledError:
                    raise
                retry_seconds = min(5.0, retry_seconds * 2)
            finally:
                if command_task is not None:
                    command_task.cancel()
                    await asyncio.gather(command_task, return_exceptions=True)
                socket = self._socket
                self._socket = None
                if socket is not None:
                    with suppress(Exception):
                        await socket.close()

    async def _connect(self, url: str) -> Any:
        headers = (
            {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        )
        return await websockets.connect(
            url,
            additional_headers=headers,
            max_size=8 * 1024 * 1024,
        )


def _api_websocket_url(api_url: str, path: str) -> str:
    if api_url.startswith("https://"):
        base = "wss://" + api_url.removeprefix("https://")
    elif api_url.startswith("http://"):
        base = "ws://" + api_url.removeprefix("http://")
    else:
        base = api_url
    return f"{base.rstrip('/')}/{path.lstrip('/')}"
