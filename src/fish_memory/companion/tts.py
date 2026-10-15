"""Native client for the application's reusable TTS WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

import websockets
from websockets.exceptions import ConnectionClosed

from fish_memory import failures
from fish_memory.companion.client import api_websocket_url
from fish_memory.companion.diagnostics import AudioDiagnosticsObserver
from fish_memory.companion.playback import (
    BoundPcmSink,
    PlaybackArbiter,
    PlaybackOwner,
)
from fish_memory.tts.speech_tags import strip_speech_tags, supports_speech_tags

DEFAULT_TTS_MODEL = "qwen-audio-3.0-tts-flash"
DEFAULT_TTS_VOICE = "longanhuan_v3.6"
_CONNECTION_CONTROL_TIMEOUT_SECONDS = 0.1
_CONNECTION_CLOSE_TIMEOUT_SECONDS = 0.1
_RECEIVE_TASK_CLOSE_TIMEOUT_SECONDS = 0.1

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _TextSegment:
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True, slots=True)
class TtsConfiguration:
    model: str
    voice: str
    sample_rate: int
    speech_tags: bool = False


@dataclass(frozen=True, slots=True)
class TtsProviderState:
    ready: bool
    active: bool
    error_free: bool
    provider: str | None
    configuration: TtsConfiguration | None


@dataclass(frozen=True, slots=True)
class DraftTtsPending:
    """The existing synthesis is waiting for matching public text or flush."""


@dataclass(slots=True)
class _PlaybackBinding:
    arbiter: PlaybackArbiter | None
    owner: PlaybackOwner | None
    sink: PcmSink | BoundPcmSink

    def is_current(self) -> bool:
        return self.arbiter is None or (
            self.owner is not None and self.arbiter.is_current(self.owner)
        )


class PcmSink(Protocol):
    @property
    def played_offset(self) -> int: ...

    @property
    def display_offset(self) -> int: ...

    async def mark(self, end_offset: int) -> None: ...

    async def start(self) -> None: ...

    async def append(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None: ...

    async def stop(self) -> None: ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def duck(self) -> None: ...

    async def unduck(self) -> None: ...

    async def wait_until_drained(self) -> None: ...


WebsocketConnector = Callable[[str], Awaitable[Any]]
BeforeStart = Callable[[], Awaitable[None]]


class ApiTtsOutput:
    consumption_is_confirmed = True

    def __init__(
        self,
        api_url: str,
        *,
        sink: PcmSink,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_TTS_VOICE,
        sample_rate: int = 22_050,
        api_key: str | None = None,
        websocket_connector: WebsocketConnector | None = None,
        diagnostics: AudioDiagnosticsObserver | None = None,
        before_start: BeforeStart | None = None,
        facial: bool = False,
    ) -> None:
        self._url = api_websocket_url(api_url, "/v1/tts/stream")
        self._sink = sink
        self._model = model
        self._voice = voice
        self._sample_rate = sample_rate
        self._api_key = api_key
        self._connector = websocket_connector or self._connect
        self._diagnostics = diagnostics
        self._before_start = before_start
        self._facial = facial
        self._speech_tags = False
        self._tagged_input = False
        self._speech_tag_capability: bool | None = None
        self._facial_cues: list[tuple[int, int, dict[str, object]]] = []
        self._facial_stream_offsets: dict[str, int] = {}
        self._pending_facial_base_offset = 0
        self._active_facial_stream_id = ""
        self._pending_audio_segment_index: int | None = None
        self._segment_audio_ranges: dict[tuple[str, int], tuple[int, int]] = {}
        self._finished_audio_segments: set[tuple[str, int]] = set()
        self._playback_arbiter: PlaybackArbiter | None = None
        self._playback_owner: PlaybackOwner | None = None
        self._active_sink: PcmSink | BoundPcmSink = sink
        self._connection_playback: _PlaybackBinding | None = None
        self._socket: Any = None
        self._receive_task: asyncio.Task[None] | None = None
        # Closing an interrupted provider connection happens in the background.
        # Its receive coroutine may therefore outlive the connection it belongs
        # to (some WebSocket implementations do not observe cancellation until
        # ``recv`` returns).  Fence every receive loop with an ownership
        # generation so a detached connection can never mutate the next turn.
        self._connection_generation = 0
        self._connection_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._ready = asyncio.Event()
        self._finished = asyncio.Event()
        self._pending_audio_text: str | None = None
        self._pending_audio_sentence_index: int | None = None
        # Provider audio events may omit their source text. Keep the append
        # order as a fallback so received PCM still maps to answer characters.
        self._pending_text_segments: deque[_TextSegment] = deque()
        self._submitted_text = ""
        self._flushed_offset = 0
        self._submitted_text_offset = 0
        self._audio_search_offset = 0
        self._received_audio_end = 0
        self._sentence_audio_received = False
        self._active_audio_sentence_index: int | None = None
        self._active_audio_text: str | None = None
        self._active_audio_range: tuple[int, int] | None = None
        self._error: Exception | None = None
        self._audio_packet_count = 0
        self._audio_byte_count = 0
        self._accept_audio = False
        self._provider_task_active = False
        self._first_audio_recorded = False
        self._first_pcm_queued_recorded = False
        self._actual_provider: str | None = None
        self._actual_model: str | None = None
        self._actual_voice: str | None = None
        self._actual_sample_rate: int | None = None

    @property
    def played_offset(self) -> int:
        return self._active_sink.played_offset

    def pop_facial_cue(self) -> dict[str, object] | None:
        """Only expose cues whose sentence PCM the device has started consuming."""
        if not self._playback_is_current():
            return None
        cue = None
        remaining = []
        display, played = self.display_offset, self.played_offset
        for start, end, candidate in self._facial_cues:
            segment_key = candidate.get("_segment")
            if isinstance(segment_key, tuple) and hasattr(
                self._active_sink, "played_audio_bytes"
            ):
                byte_range = self._segment_audio_ranges.get(segment_key)
                rendered = self._active_sink.played_audio_bytes
                if not isinstance(rendered, int) or isinstance(rendered, bool):
                    remaining.append((start, end, candidate))
                    continue
                if byte_range is not None:
                    audio_start, audio_end = byte_range
                    if (
                        segment_key in self._finished_audio_segments
                        and rendered >= audio_end
                    ):
                        continue
                    if rendered > audio_start:
                        cue = {
                            key: value
                            for key, value in candidate.items()
                            if key != "_segment"
                        }
                        continue
                remaining.append((start, end, candidate))
                continue
            if end <= played or end < display:
                continue  # Already finished, or the device has entered a later line.
            if start < display <= end:
                cue = candidate
            else:
                remaining.append((start, end, candidate))
        self._facial_cues = remaining
        return cue

    @property
    def display_offset(self) -> int:
        return self._active_sink.display_offset

    @property
    def configuration(self) -> TtsConfiguration:
        return TtsConfiguration(
            self._model, self._voice, self._sample_rate, self._speech_tags
        )

    @property
    def provider_state(self) -> TtsProviderState:
        actual_configuration = (
            TtsConfiguration(
                self._actual_model,
                self._actual_voice,
                self._actual_sample_rate,
                self._speech_tags,
            )
            if self._actual_model is not None
            and self._actual_voice is not None
            and self._actual_sample_rate is not None
            else None
        )
        return TtsProviderState(
            ready=self._ready.is_set(),
            active=self._provider_task_active,
            error_free=self._error is None,
            provider=self._actual_provider,
            configuration=actual_configuration,
        )

    def configure(self, *, model: str, voice: str) -> None:
        if self._provider_task_active:
            raise RuntimeError("cannot change TTS voice during an active answer")
        if model == self._model and voice == self._voice:
            return
        self._model = model
        self._voice = voice
        # A provider connection is keyed by model and voice. It will be
        # replaced lazily when the next answer starts.
        socket, receive_task = self._detach_connection()
        self._schedule_connection_cleanup(socket, receive_task)

    def set_speech_tags(self, enabled: bool) -> None:
        if self._provider_task_active and self._speech_tags != enabled:
            raise RuntimeError("cannot change speech protocol during an active answer")
        self._speech_tags = enabled
        self._tagged_input = enabled

    def set_speech_tag_capability(self, supported: bool) -> None:
        self._speech_tag_capability = supported

    def activate_diagnostics(
        self, diagnostics: AudioDiagnosticsObserver | None
    ) -> None:
        """Attach public timing only after a private provider is adopted."""
        if diagnostics is None or self._diagnostics is diagnostics:
            return
        self._diagnostics = diagnostics
        if self._ready.is_set() and self._playback_is_current():
            self._record_event(
                "tts_ready",
                metadata={
                    "connectionReused": True,
                    "provider": self._actual_provider,
                    "model": self._actual_model,
                    "voice": self._actual_voice,
                },
            )

    def bind_playback(
        self,
        arbiter: PlaybackArbiter,
        owner: PlaybackOwner,
    ) -> None:
        """Fence provider callbacks to one immutable live owner token."""
        connection_playback = self._connection_playback
        if (
            connection_playback is not None
            and connection_playback.owner is not None
            and connection_playback.owner != owner
        ):
            socket, receive_task = self._detach_connection()
            self._schedule_connection_cleanup(socket, receive_task)
        self._playback_arbiter = arbiter
        self._playback_owner = owner
        bind_playback_sink = getattr(self._sink, "bind_playback_sink", None)
        self._active_sink = (
            bind_playback_sink(arbiter, owner)
            if callable(bind_playback_sink)
            else arbiter.bind_pcm(owner, self._sink)
        )
        if connection_playback is not None and connection_playback.owner is None:
            # A private draft provider becomes public only after its buffered
            # PCM has attached under this exact owner. It is safe to adopt that
            # one still-active provider connection; an already-owned connection
            # is detached above instead of being rebound.
            connection_playback.arbiter = arbiter
            connection_playback.owner = owner
            connection_playback.sink = self._active_sink

    async def start(self, *, text_offset: int = 0) -> None:
        if (
            isinstance(text_offset, bool)
            or not isinstance(text_offset, int)
            or text_offset < 0
        ):
            raise ValueError("TTS text_offset must be a nonnegative integer")
        if not self._playback_is_current():
            return
        if self._before_start is not None:
            await self._before_start()
        self._speech_tags = (
            self._tagged_input
            and self._speech_tag_capability is not False
            and supports_speech_tags("dashscope", self._model)
        )
        sink = self._active_sink
        await sink.start()
        if not self._playback_is_current():
            return
        reset = getattr(sink, "reset", None)
        if callable(reset):
            try:
                reset(text_offset)
            except TypeError:
                if text_offset:
                    raise
                reset()
        self._pending_audio_text = None
        self._pending_audio_sentence_index = None
        self._pending_text_segments.clear()
        self._submitted_text = ""
        self._flushed_offset = 0
        self._submitted_text_offset = text_offset
        self._facial_cues.clear()
        self._facial_stream_offsets.clear()
        self._segment_audio_ranges.clear()
        self._finished_audio_segments.clear()
        self._pending_audio_segment_index = None
        self._audio_search_offset = text_offset
        self._received_audio_end = text_offset
        self._sentence_audio_received = False
        self._active_audio_sentence_index = None
        self._active_audio_text = None
        self._active_audio_range = None
        connection_reused = self._socket_is_usable()
        await self._ensure_connected()
        self._audio_packet_count = 0
        self._audio_byte_count = 0
        self._first_audio_recorded = False
        self._first_pcm_queued_recorded = False
        try:
            await self._begin_provider_task()
        except ConnectionClosed:
            # The API may close the reusable socket after a cancelled provider
            # task. Reconnect once before declaring this answer turn failed.
            await self._restart_provider_task()
            connection_reused = False
        ready_metadata: dict[str, object] = {
            "connectionReused": connection_reused,
        }
        if self._actual_provider is not None:
            ready_metadata["provider"] = self._actual_provider
        if self._actual_model is not None:
            ready_metadata["model"] = self._actual_model
        if self._actual_voice is not None:
            ready_metadata["voice"] = self._actual_voice
        self._record_event("tts_ready", metadata=ready_metadata)

    async def append(self, text: str) -> None:
        if not text or not self._playback_is_current():
            return
        if not self._socket_is_usable():
            await self._restart_provider_task()
        elif not self._provider_task_active:
            # The provider can finish a duplex task on its own (for example
            # after an idle timeout). The application socket remains open and
            # is waiting for the next start message in that state.
            await self._begin_provider_task()
        display_text = strip_speech_tags(text) if self._tagged_input else text
        if self._tagged_input and not self._speech_tags:
            text = display_text
            if not text.strip():
                return
        segment = self._append_text_segment(display_text)
        payload = {"type": "append", "text": text}
        if self._speech_tags:
            payload["display_text"] = display_text
        try:
            await self._send(payload)
        except ConnectionClosed:
            await self._restart_provider_task(text_base_offset=segment.start_offset)
            self._pending_text_segments.append(segment)
            await self._send(payload)
        self._record_event(
            "tts_text_submitted",
            metadata={
                "startOffset": segment.start_offset,
                "endOffset": segment.end_offset,
                "textLength": len(display_text),
            },
        )

    async def flush(self) -> None:
        """Commit submitted text without waiting for audio or ending playback."""
        end_offset = len(self._submitted_text)
        if (
            not self._playback_is_current()
            or not self._provider_task_active
            or end_offset <= self._flushed_offset
        ):
            return
        await self._send({"type": "flush"})
        self._flushed_offset = end_offset

    async def finish(self) -> None:
        if not self._playback_is_current():
            return
        if not self._provider_task_active:
            self._raise_error()
            await self._active_sink.wait_until_drained()
            return
        await self._send({"type": "finish"})
        await asyncio.wait_for(self._finished.wait(), timeout=120)
        self._raise_error()
        await self._active_sink.wait_until_drained()
        logger.info(
            "原生 TTS 播放完成: audio_packets=%s, audio_bytes=%s, played_offset=%s",
            self._audio_packet_count,
            self._audio_byte_count,
            self._active_sink.played_offset,
        )

    async def stop(self) -> None:
        await self.interrupt()

    async def pause(self) -> None:
        if self._playback_is_current():
            await self._active_sink.pause()

    async def resume(self) -> None:
        if self._playback_is_current():
            await self._active_sink.resume()

    async def duck(self) -> None:
        if self._playback_is_current():
            await self._active_sink.duck()

    async def unduck(self) -> None:
        if self._playback_is_current():
            await self._active_sink.unduck()

    async def interrupt(self) -> None:
        """Stop local playback without waiting for the provider task.

        A provider cancellation acknowledgement can take up to a second. A
        speech interruption must hand the audio device to the next answer
        immediately, so detach the old API socket first and close it in the
        background. Normal answer turns still reuse the socket after finish.
        """
        self._accept_audio = False
        self._facial_cues.clear()
        self._facial_stream_offsets.clear()
        self._provider_task_active = False
        self._pending_audio_text = None
        self._pending_audio_sentence_index = None
        self._pending_text_segments.clear()
        self._active_audio_sentence_index = None
        self._active_audio_text = None
        self._active_audio_range = None
        sink = self._active_sink
        reset = getattr(sink, "reset", None)
        if callable(reset):
            reset()
        socket, receive_task = self._detach_connection()
        self._schedule_connection_cleanup(socket, receive_task)

    async def close(self) -> None:
        self._accept_audio = False
        self._facial_cues.clear()
        self._facial_stream_offsets.clear()
        await self._reset_connection()
        await self._wait_for_connection_cleanup()
        close_sink = getattr(self._sink, "close", None)
        if callable(close_sink):
            await close_sink()
        else:
            await self._sink.stop()

    async def close_transport(self) -> None:
        """Retire only this provider transport, leaving a shared PCM sink alive."""
        self._accept_audio = False
        self._facial_cues.clear()
        self._facial_stream_offsets.clear()
        await self._reset_connection()
        await self._wait_for_connection_cleanup()

    async def _ensure_connected(self) -> None:
        if self._socket_is_usable():
            return
        await self._reset_connection()
        socket = await self._connector(self._url)
        self._connection_generation += 1
        generation = self._connection_generation
        self._socket = socket
        playback = _PlaybackBinding(
            self._playback_arbiter,
            self._playback_owner,
            self._active_sink,
        )
        self._connection_playback = playback
        self._receive_task = asyncio.create_task(
            self._receive_loop(socket, generation, playback)
        )

    async def _restart_provider_task(
        self, *, text_base_offset: int | None = None
    ) -> None:
        await self._reset_connection()
        await self._ensure_connected()
        await self._begin_provider_task(text_base_offset=text_base_offset)

    async def _begin_provider_task(
        self, *, text_base_offset: int | None = None
    ) -> None:
        self._provider_task_active = False
        self._ready.clear()
        self._finished.clear()
        self._error = None
        self._pending_audio_text = None
        self._pending_audio_sentence_index = None
        self._pending_text_segments.clear()
        self._active_audio_sentence_index = None
        self._active_audio_text = None
        self._active_audio_range = None
        self._actual_provider = None
        self._actual_model = None
        self._actual_voice = None
        self._actual_sample_rate = None
        self._accept_audio = True
        self._provider_task_active = True
        # Freeze the base before sending start: by the time a delayed ready
        # arrives, an append/retry may already have extended submitted text.
        self._pending_facial_base_offset = (
            self._submitted_text_offset + len(self._submitted_text)
            if text_base_offset is None
            else text_base_offset
        )
        try:
            await self._send(
                {
                    "type": "start",
                    "model": self._model,
                    "voice": self._voice,
                    "audio_format": "pcm",
                    "sample_rate": self._sample_rate,
                    **({"facial": True} if self._facial else {}),
                    **({"speech_tags": True} if self._speech_tags else {}),
                }
            )
            await asyncio.wait_for(self._ready.wait(), timeout=15)
            self._raise_error()
        except Exception:
            self._provider_task_active = False
            raise

    async def _reset_connection(self) -> None:
        socket, receive_task = self._detach_connection()
        await self._close_connection(socket, receive_task)

    def _detach_connection(self) -> tuple[Any, asyncio.Task[None] | None]:
        self._provider_task_active = False
        # Invalidate ownership before background cleanup is scheduled.  This
        # is deliberately synchronous so interrupt() stays low latency while
        # any cancellation-resistant recv() is still safely quarantined.
        self._connection_generation += 1
        receive_task = self._receive_task
        self._receive_task = None
        socket = self._socket
        self._socket = None
        self._connection_playback = None
        return socket, receive_task

    def _schedule_connection_cleanup(
        self,
        socket: Any,
        receive_task: asyncio.Task[None] | None,
    ) -> None:
        if socket is None and receive_task is None:
            return
        task = asyncio.create_task(self._close_connection(socket, receive_task))
        self._connection_cleanup_tasks.add(task)
        task.add_done_callback(self._connection_cleanup_finished)

    async def _close_connection(
        self,
        socket: Any,
        receive_task: asyncio.Task[None] | None,
    ) -> None:
        if socket is not None:
            # Best effort only. Closing the application socket also causes the
            # server-side provider task to be cancelled; never hold up the
            # next answer on this control message.
            await self._run_bounded_cleanup(
                socket.send(json.dumps({"type": "stop"})),
                max_wait_seconds=_CONNECTION_CONTROL_TIMEOUT_SECONDS,
            )
            # Close the transport before waiting on recv(). A native/network
            # recv may delay cancellation but should still wake when its socket
            # closes. The close handshake itself is also strictly bounded.
            await self._run_bounded_cleanup(
                socket.close(),
                max_wait_seconds=_CONNECTION_CLOSE_TIMEOUT_SECONDS,
            )
        if receive_task is not None and receive_task is not asyncio.current_task():
            receive_task.cancel()
            done, _ = await asyncio.wait(
                {receive_task},
                timeout=_RECEIVE_TASK_CLOSE_TIMEOUT_SECONDS,
            )
            if receive_task in done:
                self._consume_task_result(receive_task)
            else:
                # A cancellation-resistant recv must never hold close() or the
                # next answer hostage. Consume its eventual result so a late
                # transport exception cannot become an unhandled task error.
                receive_task.add_done_callback(self._consume_task_result)

    async def _run_bounded_cleanup(
        self,
        operation: Awaitable[Any],
        *,
        max_wait_seconds: float,
    ) -> None:
        task = asyncio.ensure_future(operation)
        done, _ = await asyncio.wait({task}, timeout=max_wait_seconds)
        if task not in done:
            task.cancel()
            task.add_done_callback(self._consume_task_result)
            return
        self._consume_task_result(task)

    def _connection_cleanup_finished(self, task: asyncio.Task[None]) -> None:
        self._connection_cleanup_tasks.discard(task)
        self._consume_task_result(task)

    @staticmethod
    def _consume_task_result(task: asyncio.Future[Any]) -> None:
        with suppress(asyncio.CancelledError, Exception):
            task.result()

    async def _wait_for_connection_cleanup(self) -> None:
        tasks = tuple(self._connection_cleanup_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _socket_is_usable(self) -> bool:
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
        task = self._receive_task
        return task is not None and not task.done()

    async def _send(self, message: Mapping[str, object]) -> None:
        if self._socket is None:
            raise RuntimeError("TTS WebSocket is not connected")
        await self._socket.send(json.dumps(message, ensure_ascii=False))

    def _owns_connection(self, socket: Any, generation: int) -> bool:
        return self._socket is socket and self._connection_generation == generation

    def _playback_is_current(self) -> bool:
        arbiter = self._playback_arbiter
        owner = self._playback_owner
        return arbiter is None or (owner is not None and arbiter.is_current(owner))

    async def _receive_loop(
        self,
        socket: Any,
        generation: int,
        playback: _PlaybackBinding,
    ) -> None:
        try:
            while True:
                raw = await socket.recv()
                # A detached recv() can complete after interrupt/configure has
                # already installed a new connection and reset the shared turn
                # events.  Do not let any frame from that former owner touch
                # PCM, readiness, completion, metadata, or error state.
                if not self._owns_connection(socket, generation):
                    return
                if not playback.is_current():
                    return
                if isinstance(raw, bytes):
                    if not self._accept_audio:
                        self._pending_audio_text = None
                        self._pending_audio_sentence_index = None
                        continue
                    text = self._pending_audio_text
                    sentence_index = self._pending_audio_sentence_index
                    self._pending_audio_text = None
                    self._pending_audio_sentence_index = None
                    start_offset, end_offset = self._resolve_audio_range(
                        text,
                        sentence_index,
                    )
                    if raw:
                        self._received_audio_end = max(
                            self._received_audio_end, end_offset
                        )
                        self._sentence_audio_received = True
                    if not self._first_audio_recorded:
                        self._first_audio_recorded = True
                        self._record_event("tts_first_audio_received")
                    self._audio_packet_count += 1
                    segment_index = self._pending_audio_segment_index
                    self._pending_audio_segment_index = None
                    if segment_index is not None and self._active_facial_stream_id:
                        key = (self._active_facial_stream_id, segment_index)
                        old = self._segment_audio_ranges.get(key)
                        self._segment_audio_ranges[key] = (
                            old[0] if old is not None else self._audio_byte_count,
                            self._audio_byte_count + len(raw),
                        )
                    self._audio_byte_count += len(raw)
                    sink = playback.sink
                    await sink.append(
                        raw,
                        start_offset=start_offset,
                        end_offset=end_offset,
                    )
                    if (
                        not self._owns_connection(socket, generation)
                        or not playback.is_current()
                        or sink is not playback.sink
                    ):
                        return
                    if not self._first_pcm_queued_recorded:
                        self._first_pcm_queued_recorded = True
                        self._record_event("tts_first_pcm_queued")
                    continue
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                message_type = message.get("type")
                if message_type == "ready":
                    self._finished.clear()
                    if self._ready.is_set():
                        # Implicit provider recovery resets its sentence IDs,
                        # but must retain pending/cumulative answer text.
                        self._active_audio_sentence_index = None
                        self._active_audio_text = None
                        self._active_audio_range = None
                        self._sentence_audio_received = False
                    stream_id = message.get("facialStreamId")
                    if self._facial and isinstance(stream_id, str) and stream_id:
                        self._active_facial_stream_id = stream_id
                        if stream_id not in self._facial_stream_offsets:
                            # An API may recover an append already in flight
                            # when its prior provider finished. Such a ready
                            # has no matching explicit start to freeze a base.
                            recovered_base = (
                                self._pending_text_segments[0].start_offset
                                if self._pending_text_segments
                                else self._audio_search_offset
                            )
                            self._facial_stream_offsets[stream_id] = (
                                recovered_base
                                if self._ready.is_set()
                                else self._pending_facial_base_offset
                            )
                    if self._accept_audio:
                        self._provider_task_active = True
                    provider = message.get("provider")
                    model = message.get("model")
                    voice = message.get("voice")
                    sample_rate = message.get("sampleRate")
                    self._actual_provider = (
                        provider if isinstance(provider, str) else None
                    )
                    self._actual_model = model if isinstance(model, str) else None
                    self._actual_voice = voice if isinstance(voice, str) else None
                    self._actual_sample_rate = (
                        sample_rate
                        if isinstance(sample_rate, int)
                        and not isinstance(sample_rate, bool)
                        else None
                    )
                    self._ready.set()
                elif message_type == "sentence_begin":
                    self._accept_explicit_audio_range(message)
                    text = message.get("text")
                    sentence_index = message.get("sentenceIndex")
                    if (
                        self._accept_audio
                        and isinstance(text, str)
                        and text
                        and isinstance(sentence_index, int)
                        and not isinstance(sentence_index, bool)
                    ):
                        # DashScope may merge several append directives into
                        # one provider sentence. Resolve that announced text
                        # before its audio packets arrive so an empty audio
                        # annotation can reuse the complete merged range.
                        self._resolve_audio_range(text, sentence_index)
                elif message_type == "facial":
                    if self._facial and self._accept_audio:
                        text = message.get("text")
                        profile_id = message.get("profileId")
                        stream_id = message.get("streamId")
                        relative_start = message.get("startOffset")
                        relative_end = message.get("endOffset")
                        segment_index = message.get("segmentIndex")
                        segment_cue = self._speech_tags and isinstance(
                            segment_index, int
                        )
                        if (
                            isinstance(text, str)
                            and (text or segment_cue)
                            and isinstance(profile_id, str)
                            and isinstance(stream_id, str)
                            and stream_id in self._facial_stream_offsets
                            and isinstance(relative_start, int)
                            and not isinstance(relative_start, bool)
                            and isinstance(relative_end, int)
                            and not isinstance(relative_end, bool)
                            and 0 <= relative_start <= relative_end
                            and (relative_start < relative_end or segment_cue)
                        ):
                            base = self._facial_stream_offsets[stream_id]
                            start, end = base + relative_start, base + relative_end
                            local_start = start - self._submitted_text_offset
                            local_end = end - self._submitted_text_offset
                            if (
                                0
                                <= local_start
                                <= local_end
                                <= len(self._submitted_text)
                                and self._submitted_text[local_start:local_end] == text
                                and (
                                    segment_cue
                                    or (
                                        end > self.played_offset
                                        and end >= self.display_offset
                                    )
                                )
                            ):
                                self._facial_cues.append(
                                    (
                                        start,
                                        end,
                                        {
                                            "profileId": profile_id,
                                            "targets": message.get("targets"),
                                            **(
                                                {"_segment": (stream_id, segment_index)}
                                                if segment_cue
                                                else {}
                                            ),
                                        },
                                    )
                                )
                                self._facial_cues.sort(
                                    key=lambda entry: (entry[0], entry[1])
                                )
                                self._record_event(
                                    "tts_facial_ready",
                                    metadata={
                                        "waitMs": message.get("waitMs"),
                                        "errorType": message.get("error"),
                                        "endOffset": end,
                                    },
                                )
                elif message_type == "audio":
                    self._accept_explicit_audio_range(message)
                    segment_index = message.get("segmentIndex")
                    self._pending_audio_segment_index = (
                        segment_index
                        if self._speech_tags and isinstance(segment_index, int)
                        else None
                    )
                    if self._accept_audio and not self._first_audio_recorded:
                        # These are durations measured at the API, not native
                        # event timestamps. In particular firstPacketMs is
                        # captured before the JEV gate can delay forwarding.
                        timing = {
                            field: value
                            for field, value in (
                                ("providerFirstPacketMs", message.get("firstPacketMs")),
                                ("facialWaitMs", message.get("facialWaitMs")),
                            )
                            if isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                            and value >= 0
                        }
                        if timing:
                            self._record_event(
                                "tts_first_audio_timing", metadata=timing
                            )
                    text = message.get("text")
                    self._pending_audio_text = (
                        text if self._accept_audio and isinstance(text, str) else None
                    )
                    sentence_index = message.get("sentenceIndex")
                    self._pending_audio_sentence_index = (
                        sentence_index
                        if self._accept_audio and isinstance(sentence_index, int)
                        else None
                    )
                elif message_type == "sentence_end":
                    sentence_index = message.get("sentenceIndex")
                    if (
                        self._accept_audio
                        and self._sentence_audio_received
                        and self._active_audio_range is not None
                        and sentence_index == self._active_audio_sentence_index
                    ):
                        await playback.sink.mark(self._active_audio_range[1])
                        self._sentence_audio_received = False
                elif message_type == "segment_finished":
                    segment_index = message.get("segmentIndex")
                    if isinstance(segment_index, int):
                        self._finished_audio_segments.add(
                            (self._active_facial_stream_id, segment_index)
                        )
                    # Internal provider tasks restart their sentence indexes.
                    # Retain cumulative text/PCM offsets but retire the old
                    # annotation so repeated continuation text maps forward.
                    if self._accept_audio:
                        await playback.sink.mark(self._received_audio_end)
                    self._pending_audio_text = None
                    self._pending_audio_sentence_index = None
                    self._active_audio_sentence_index = None
                    self._active_audio_text = None
                    self._active_audio_range = None
                    self._sentence_audio_received = False
                elif message_type == "finished":
                    if self._accept_audio and not message.get("stopped"):
                        await playback.sink.mark(self._received_audio_end)
                    self._provider_task_active = False
                    self._finished.set()
                elif message_type == "error":
                    self._provider_task_active = False
                    self._error = failures.ReportedFailure(
                        str(message.get("message", "TTS failed")),
                        message.get("failure"),
                    )
                    self._ready.set()
                    self._finished.set()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._owns_connection(socket, generation):
                self._error = error
                self._ready.set()
                self._finished.set()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"TTS stream failed: {self._error}") from self._error

    def _record_event(
        self,
        name: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        diagnostics = self._diagnostics
        if diagnostics is None:
            return
        try:
            diagnostics.record_event(name, source="tts", metadata=metadata)
        except Exception:
            logger.debug("Unable to record TTS diagnostic event", exc_info=True)

    def _append_text_segment(self, text: str) -> _TextSegment:
        start_offset = self._submitted_text_offset + len(self._submitted_text)
        self._submitted_text += text
        segment = _TextSegment(
            start_offset=start_offset,
            end_offset=self._submitted_text_offset + len(self._submitted_text),
            text=text,
        )
        self._pending_text_segments.append(segment)
        return segment

    def _accept_explicit_audio_range(self, message: dict[str, Any]) -> None:
        if not self._speech_tags or not self._accept_audio:
            return
        start, end = message.get("startOffset"), message.get("endOffset")
        index, text = message.get("sentenceIndex"), message.get("text")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(index, int)
            or isinstance(index, bool)
            or not isinstance(text, str)
            or not 0 <= start <= end
        ):
            return
        base = self._pending_facial_base_offset
        local_start = base + start - self._submitted_text_offset
        local_end = base + end - self._submitted_text_offset
        if not (
            0 <= local_start <= local_end <= len(self._submitted_text)
            and self._submitted_text[local_start:local_end] == text
        ):
            return
        self._active_audio_sentence_index = index
        self._active_audio_text = text
        self._active_audio_range = (base + start, base + end)
        self._audio_search_offset = max(self._audio_search_offset, base + end)
        self._consume_pending_text_through(base + end)

    def _resolve_audio_range(
        self,
        text: str | None,
        sentence_index: int | None = None,
    ) -> tuple[int, int]:
        search_start = max(
            self._submitted_text_offset,
            min(
                self._submitted_text_offset + len(self._submitted_text),
                self._audio_search_offset,
            ),
        )
        if (
            sentence_index is not None
            and sentence_index == self._active_audio_sentence_index
            and self._active_audio_range is not None
            and (not text or text == self._active_audio_text)
        ):
            return self._active_audio_range

        segment = self._find_pending_text_segment(text, search_start)
        # Missing/unmatched annotations cannot authorize unseen text. The
        # provider's sentence_begin resolves known ranges before empty packets.

        if segment is None:
            audio_range = (search_start, search_start)
        else:
            start_offset = max(search_start, segment.start_offset)
            end_offset = max(start_offset, segment.end_offset)
            audio_range = (start_offset, end_offset)
            self._audio_search_offset = max(
                self._audio_search_offset,
                end_offset,
            )

        if sentence_index is not None:
            self._active_audio_sentence_index = sentence_index
            self._active_audio_text = text or None
            self._active_audio_range = audio_range
        else:
            self._active_audio_sentence_index = None
            self._active_audio_text = None
            self._active_audio_range = None
        return audio_range

    def _find_pending_text_segment(
        self,
        text: str | None,
        search_start: int,
    ) -> _TextSegment | None:
        if not text:
            return None
        self._discard_pending_text_segments(search_start)
        if self._pending_text_segments:
            first = self._pending_text_segments[0]
            start_offset = max(search_start, first.start_offset)
            available = "".join(
                (
                    segment.text[start_offset - segment.start_offset :]
                    if segment is first
                    else segment.text
                )
                for segment in self._pending_text_segments
            )
            if available.startswith(text):
                end_offset = start_offset + len(text)
                self._consume_pending_text_through(end_offset)
                return _TextSegment(start_offset, end_offset, text)
        return None

    def _consume_pending_text_through(self, end_offset: int) -> None:
        while self._pending_text_segments:
            segment = self._pending_text_segments[0]
            if segment.end_offset <= end_offset:
                self._pending_text_segments.popleft()
                continue
            if segment.start_offset < end_offset:
                consumed = end_offset - segment.start_offset
                self._pending_text_segments[0] = _TextSegment(
                    start_offset=end_offset,
                    end_offset=segment.end_offset,
                    text=segment.text[consumed:],
                )
            return

    def _discard_pending_text_segments(self, through_offset: int) -> None:
        while (
            self._pending_text_segments
            and self._pending_text_segments[0].end_offset <= through_offset
        ):
            self._pending_text_segments.popleft()

    async def _connect(self, url: str) -> Any:
        headers = (
            {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        )
        return await websockets.connect(
            url,
            additional_headers=headers,
            max_size=8 * 1024 * 1024,
        )
