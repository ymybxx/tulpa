"""Strict full-duplex PCM bridge for the local LiveKit Swift audio helper."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import struct
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic_ns
from typing import Protocol, TypeVar, cast

from fish_memory.companion.audio import PlayedOffsetTracker
from fish_memory.companion.playback import PlaybackOwner

logger = logging.getLogger(__name__)

FRAME_MAGIC = b"FMA1"
PLAYBACK_MAGIC = b"FMP1"
CONTROL_MAGIC = b"FMC1"
RENDER_MAGIC = b"FMR1"

FRAME_PAYLOAD_BYTES = 336
RENDER_PAYLOAD_BYTES = 348
PCM_FRAME_BYTES = 320
CONTROL_PAYLOAD_BYTES = 9

CONTROL_RESET = 1
CONTROL_PAUSE = 2
CONTROL_RESUME = 3
CONTROL_DUCK = 4
CONTROL_UNDUCK = 5
CONTROL_STOP = 6
CONTROL_FLUSH = 7
CONTROL_ENSURE_PROCESSED = 8
CONTROL_MUTE_MICROPHONE = 9
CONTROL_UNMUTE_MICROPHONE = 10
CONTROL_MUTE_SPEAKER = 11
CONTROL_UNMUTE_SPEAKER = 12

_HEADER = struct.Struct("<4sI")
_CAPTURE_PREFIX = struct.Struct("<QQ")
_PLAYBACK_GENERATION = struct.Struct("<Q")
_CONTROL_PAYLOAD = struct.Struct("<BQ")
_RENDER_PREFIX = struct.Struct("<QQQI")
_SILENCE_FRAME = bytes(PCM_FRAME_BYTES)
_STDERR_TAIL_BYTES = 4_096
_MAX_PLAYBACK_PACKET_BYTES = 32_768
_QueueItem = TypeVar("_QueueItem")


class _HelperStdin(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def close(self) -> None: ...


class _HelperProcess(Protocol):
    stdin: _HelperStdin | None
    stdout: asyncio.StreamReader | None
    stderr: asyncio.StreamReader | None

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


ProcessFactory = Callable[..., Awaitable[_HelperProcess]]


class LiveKitAudioProtocolError(RuntimeError):
    """The helper emitted bytes outside the full-duplex binary contract."""


@dataclass(frozen=True, slots=True)
class _PlaybackSegment:
    owner: PlaybackOwner | None
    start_byte: int
    end_byte: int


class LiveKitProcessedPcmInput:
    """Bridge VPIO-processed capture and TTS playback over one helper process.

    Capture and speaker PCM share the native Voice Processing I/O graph. The
    helper acknowledges bytes from native playback-render callbacks, so text
    offsets and drain completion never advance merely because Python queued a
    packet. This is render progress, not a physically-audible completion signal.
    Malformed, stale, or impossible acknowledgements fail the audio run instead
    of silently corrupting interruption state.
    """

    sample_rate = 16_000
    block_size = 160
    capture_is_preprocessed = True
    stream_delay_ms: None = None

    def __init__(
        self,
        helper_path: str,
        *,
        queue_size: int = 64,
        writer_queue_size: int = 128,
        max_inflight_playback_bytes: int = 64_000,
        startup_timeout_seconds: float = 5.0,
        frame_watchdog_timeout_seconds: float = 5.0,
        drain_watchdog_timeout_seconds: float = 5.0,
        shutdown_timeout_seconds: float = 0.5,
        process_factory: ProcessFactory | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        if not helper_path.strip():
            raise ValueError("LiveKit audio helper path is required")
        if queue_size < 1:
            raise ValueError("LiveKit audio queue size must be positive")
        if writer_queue_size < 1:
            raise ValueError("LiveKit audio writer queue size must be positive")
        if (
            max_inflight_playback_bytes < PCM_FRAME_BYTES
            or max_inflight_playback_bytes % 2
        ):
            raise ValueError(
                "LiveKit playback window must contain at least one PCM16 frame"
            )
        if (
            startup_timeout_seconds <= 0
            or frame_watchdog_timeout_seconds <= 0
            or drain_watchdog_timeout_seconds <= 0
        ):
            raise ValueError("LiveKit audio timeouts must be positive")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("LiveKit audio shutdown timeout must be positive")

        self.helper_path = helper_path
        self._microphone_enabled = True
        self._microphone_packet: bytes | None = None
        self._playback_enabled = True
        self._speaker_packet: bytes | None = None
        self._queue_size = queue_size
        self._writer_queue_size = writer_queue_size
        self._max_inflight_playback_bytes = max_inflight_playback_bytes
        self._playback_packet_bytes = min(
            _MAX_PLAYBACK_PACKET_BYTES,
            max_inflight_playback_bytes,
        )
        self._startup_timeout_seconds = startup_timeout_seconds
        self._frame_watchdog_timeout_seconds = frame_watchdog_timeout_seconds
        self._drain_watchdog_timeout_seconds = drain_watchdog_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._process_factory = process_factory or cast(
            ProcessFactory,
            asyncio.create_subprocess_exec,
        )
        self._validate_helper_path = process_factory is None
        self._clock_ns = clock_ns

        self._frames: asyncio.Queue[tuple[bytes, bytes, int]] = asyncio.Queue(
            maxsize=queue_size
        )
        self._outbound: asyncio.Queue[bytes] = asyncio.Queue(maxsize=writer_queue_size)
        self._urgent_outbound: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
        self._ensure_processed_packet: bytes | None = None
        self._writer_wakeup = asyncio.Event()
        self._process: _HelperProcess | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._first_frame = asyncio.Event()
        self._playback_progress = asyncio.Event()
        self._terminal_error: BaseException | None = None
        self._stderr_tail = bytearray()
        self._last_sequence: int | None = None
        self._dropped_frames = 0
        self._last_dequeued_frame_age_ms = 0.0
        self._last_dequeued_frame_queue_depth = 0
        self._closing = False

        self._generation = 0
        self._queued_valid_bytes = 0
        self._sent_valid_bytes = 0
        self._acknowledged_valid_bytes = 0
        self._segments: deque[_PlaybackSegment] = deque()
        self._tracker = PlayedOffsetTracker()
        self._paused = False
        self._ducked = False
        self._stopped = False
        self._playback_control_revision = 0
        self._playback_started_reported = False
        self._playback_started_callback: (
            Callable[[PlaybackOwner | None], None] | None
        ) = None
        self._rendered_audio_callback: Callable[[bytes], None] | None = None

    @property
    def played_offset(self) -> int:
        return self._tracker.played_offset

    @property
    def played_audio_bytes(self) -> int:
        return self._tracker.consumed_bytes

    @property
    def display_offset(self) -> int:
        return self._tracker.display_offset

    async def mark(self, end_offset: int) -> None:
        if not self._stopped:
            self._tracker.mark(end_offset)

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def frame_queue_depth(self) -> int:
        return self._frames.qsize()

    @property
    def last_dequeued_frame_age_ms(self) -> float:
        return self._last_dequeued_frame_age_ms

    @property
    def last_dequeued_frame_queue_depth(self) -> int:
        return self._last_dequeued_frame_queue_depth

    @property
    def playback_generation(self) -> int:
        return self._generation

    @property
    def queued_playback_bytes(self) -> int:
        return self._queued_valid_bytes

    @property
    def rendered_playback_bytes(self) -> int:
        return self._acknowledged_valid_bytes

    async def start(self) -> None:
        # Capture and TTS share this object, so either can be the first caller.
        # Serialize the entire startup, including the first native input frame.
        async with self._lifecycle_lock:
            await self._start_helper()

    async def _start_helper(self) -> None:
        reader = self._reader_task
        writer = self._writer_task
        if (
            self._process is not None
            and reader is not None
            and not reader.done()
            and writer is not None
            and not writer.done()
        ):
            if self._terminal_error is not None:
                raise self._failure()
            return
        if self._process is not None:
            failure = self._failure()
            await self._close_helper()
            raise failure
        startup_started_ns = self._clock_ns()
        if self._validate_helper_path:
            path = Path(self.helper_path)
            if not await asyncio.to_thread(_is_executable_file, path):
                raise RuntimeError(
                    f"LiveKit audio helper is missing or not executable: {path}"
                )

        self._reset_transport_state()
        launch_microphone = self._microphone_enabled
        launch_playback = self._playback_enabled
        self._begin_generation(text_offset=0)
        try:
            process = await self._process_factory(
                self.helper_path,
                env={
                    **os.environ,
                    "FISH_MEMORY_MICROPHONE_ENABLED": str(
                        self._microphone_enabled
                    ).lower(),
                    "FISH_MEMORY_PLAYBACK_ENABLED": str(launch_playback).lower(),
                },
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as error:
            raise RuntimeError("Unable to start LiveKit audio helper") from error
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._process = process
            await self._close_helper()
            raise RuntimeError(
                "LiveKit audio helper requires stdin, stdout, and stderr pipes"
            )

        self._process = process
        if launch_microphone != self._microphone_enabled:
            self.set_microphone_enabled(self._microphone_enabled)
        if launch_playback != self._playback_enabled:
            self.set_playback_enabled(self._playback_enabled)
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(process.stderr),
            name="livekit-audio-helper-stderr",
        )
        self._writer_task = asyncio.create_task(
            self._write_packets(process.stdin),
            name="livekit-audio-helper-writer",
        )
        self._reader_task = asyncio.create_task(
            self._read_packets(process.stdout),
            name="livekit-audio-helper-packets",
        )
        self._enqueue_control_nowait(CONTROL_RESET, self._generation)
        try:
            await asyncio.wait_for(
                self._first_frame.wait(),
                timeout=self._startup_timeout_seconds,
            )
            # Let an immediate EOF/protocol/write error win over one stale frame.
            await asyncio.sleep(0)
            if self._terminal_error is not None:
                raise self._failure()
        except TimeoutError:
            await self._close_helper()
            raise RuntimeError(
                "LiveKit audio helper did not deliver its first frame within "
                f"{self._startup_timeout_seconds:g} seconds"
                f"{self._stderr_suffix()}"
            ) from None
        except BaseException:
            failure = self._failure() if self._terminal_error is not None else None
            await self._close_helper()
            if failure is not None:
                raise failure from self._terminal_error
            raise
        startup_ms = max(0, self._clock_ns() - startup_started_ns) / 1_000_000
        logger.info(
            "LiveKit 本地音频已就绪: backend=Apple VPIO, duplex=shared, "
            "sample_rate=16000, frame_ms=10, startup_ms=%.1f",
            startup_ms,
        )

    def set_microphone_enabled(self, enabled: bool) -> None:
        self._microphone_enabled = enabled
        self._clear_queue(self._frames)
        if self._process is not None:
            self._microphone_packet = _control_packet(
                CONTROL_UNMUTE_MICROPHONE if enabled else CONTROL_MUTE_MICROPHONE, 0
            )
            self._writer_wakeup.set()

    def set_playback_enabled(self, enabled: bool) -> None:
        self._playback_enabled = enabled
        if self._process is not None:
            self._speaker_packet = _control_packet(
                CONTROL_UNMUTE_SPEAKER if enabled else CONTROL_MUTE_SPEAKER, 0
            )
            self._writer_wakeup.set()

    async def run(
        self,
        on_frame: Callable[[bytes, bytes], object],
        stop_event: asyncio.Event,
        *,
        start_if_needed: bool = True,
    ) -> None:
        try:
            if start_if_needed:
                await self.start()
            loop = asyncio.get_running_loop()
            frame_deadline = loop.time() + self._frame_watchdog_timeout_seconds
            while not stop_event.is_set():
                self._raise_if_failed()
                remaining = frame_deadline - loop.time()
                if remaining <= 0:
                    raise RuntimeError(
                        "LiveKit audio helper stopped delivering frames for "
                        f"{self._frame_watchdog_timeout_seconds:g} seconds"
                        f"{self._stderr_suffix()}"
                    )
                try:
                    pcm, rendered_pcm, captured_at_ns = await asyncio.wait_for(
                        self._frames.get(),
                        timeout=min(0.1, remaining),
                    )
                except TimeoutError:
                    continue
                self._last_dequeued_frame_age_ms = round(
                    max(0, self._clock_ns() - captured_at_ns) / 1_000_000,
                    3,
                )
                self._last_dequeued_frame_queue_depth = self._frames.qsize()
                result = on_frame(pcm, rendered_pcm)
                if isinstance(result, Awaitable):
                    await result
                frame_deadline = loop.time() + self._frame_watchdog_timeout_seconds
        finally:
            await self.close()

    def reset(self, text_offset: int = 0) -> None:
        if isinstance(text_offset, bool) or not isinstance(text_offset, int):
            raise ValueError("text_offset must be a nonnegative integer")
        if text_offset < 0:
            raise ValueError("text_offset must be a nonnegative integer")
        self._discard_outbound_packets()
        self._begin_generation(text_offset=text_offset)
        if self._process is not None:
            self._enqueue_control_nowait(CONTROL_RESET, self._generation)

    def notify_voice_activity(self) -> None:
        """Ask the active helper to restore VPIO without delaying capture."""
        self._raise_if_failed()
        if self._process is None or self._ensure_processed_packet is not None:
            return
        packet = _control_packet(CONTROL_ENSURE_PROCESSED, self._generation)
        try:
            self._urgent_outbound.put_nowait(packet)
        except asyncio.QueueFull:
            # This is an advisory backstop for the native acoustic gate. Never
            # fail an otherwise healthy capture run because critical playback
            # controls already occupy the bounded urgent queue.
            logger.debug(
                "Dropped voice-processing restore advisory: urgent queue is full"
            )
            return
        self._ensure_processed_packet = packet
        self._writer_wakeup.set()

    async def append(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        await self._append_pcm(
            pcm,
            start_offset=start_offset,
            end_offset=end_offset,
            owner=None,
        )

    async def append_owned(
        self,
        owner: PlaybackOwner,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        await self._append_pcm(
            pcm,
            start_offset=start_offset,
            end_offset=end_offset,
            owner=owner,
        )

    async def _append_pcm(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
        owner: PlaybackOwner | None,
    ) -> None:
        if not pcm:
            return
        if len(pcm) % 2:
            raise ValueError("PCM must contain complete 16-bit samples")
        self._raise_if_failed()
        if self._process is None:
            raise RuntimeError("LiveKit audio helper is not started")
        if self._stopped:
            raise RuntimeError("LiveKit playback requires reset after stop")

        generation = self._generation
        start_byte = self._queued_valid_bytes
        end_byte = start_byte + len(pcm)
        self._tracker.add_segment(
            byte_length=len(pcm),
            start_offset=start_offset,
            end_offset=end_offset,
        )
        self._segments.append(_PlaybackSegment(owner, start_byte, end_byte))
        self._queued_valid_bytes = end_byte
        try:
            for offset in range(0, len(pcm), self._playback_packet_bytes):
                if generation != self._generation or self._stopped:
                    return
                await self._queue_packet(
                    _packet(
                        PLAYBACK_MAGIC,
                        _PLAYBACK_GENERATION.pack(generation)
                        + pcm[offset : offset + self._playback_packet_bytes],
                    )
                )
        except BaseException:
            # The native side may have received only a prefix of this generation.
            # Invalidate it so a later append can never be mistaken for the tail.
            if generation == self._generation:
                self.reset()
            raise

    def set_playback_started_callback(
        self,
        callback: Callable[[PlaybackOwner | None], None] | None,
    ) -> None:
        self._playback_started_callback = callback

    def set_rendered_audio_callback(
        self,
        callback: Callable[[bytes], None] | None,
    ) -> None:
        self._rendered_audio_callback = callback

    async def pause(self) -> None:
        self._paused = True
        self._playback_control_revision += 1
        await self._send_control(CONTROL_PAUSE)
        self._signal_playback_progress()

    async def resume(self) -> None:
        self._paused = False
        self._playback_control_revision += 1
        await self._send_control(CONTROL_RESUME)
        self._signal_playback_progress()

    async def duck(self) -> None:
        self._ducked = True
        await self._send_control(CONTROL_DUCK)

    async def unduck(self) -> None:
        self._ducked = False
        await self._send_control(CONTROL_UNDUCK)

    async def stop(self) -> None:
        if self._process is None:
            return
        self._discard_outbound_packets()
        self._stopped = True
        self._paused = True
        self._ducked = False
        self._segments.clear()
        self._tracker = PlayedOffsetTracker()
        self._playback_started_reported = False
        await self._send_control(CONTROL_STOP)
        self._signal_playback_progress()

    async def wait_until_drained(self) -> None:
        if self._stopped:
            return
        generation = self._generation
        target = self._queued_valid_bytes
        if target <= self._acknowledged_valid_bytes:
            return
        await self._send_control(CONTROL_FLUSH, generation=generation)
        await self._wait_for_rendered_bytes(generation, target)

    async def wait_until_owner_drained(self, owner: PlaybackOwner) -> None:
        if self._stopped:
            return
        generation = self._generation
        targets = [
            segment.end_byte for segment in self._segments if segment.owner == owner
        ]
        if not targets:
            return
        target = max(targets)
        if target <= self._acknowledged_valid_bytes:
            return
        await self._send_control(CONTROL_FLUSH, generation=generation)
        await self._wait_for_rendered_bytes(generation, target)

    async def close(self) -> None:
        async with self._lifecycle_lock:
            await self._close_helper()

    async def _close_helper(self) -> None:
        process = self._process
        reader_task = self._reader_task
        writer_task = self._writer_task
        stderr_task = self._stderr_task
        stdin = process.stdin if process is not None else None
        self._process = None
        self._reader_task = None
        self._writer_task = None
        self._stderr_task = None
        self._closing = True
        self._stopped = True
        self._segments.clear()
        self._signal_playback_progress()
        try:
            for task in (reader_task, writer_task):
                if task is not None and not task.done():
                    task.cancel()
            if stdin is not None:
                with suppress(BrokenPipeError, ConnectionResetError):
                    stdin.close()
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(
                        asyncio.shield(process.wait()),
                        timeout=self._shutdown_timeout_seconds,
                    )
                except TimeoutError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()
            tasks = [task for task in (reader_task, writer_task) if task is not None]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if stderr_task is not None and not stderr_task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(stderr_task),
                        timeout=self._shutdown_timeout_seconds,
                    )
                except TimeoutError:
                    stderr_task.cancel()
            if stderr_task is not None:
                await asyncio.gather(stderr_task, return_exceptions=True)
        finally:
            self._closing = False
            self._clear_queue(self._frames)
            self._discard_outbound_packets()
            self._last_dequeued_frame_queue_depth = 0

    async def _wait_for_rendered_bytes(
        self,
        generation: int,
        target: int,
    ) -> None:
        loop = asyncio.get_running_loop()
        acknowledged = self._acknowledged_valid_bytes
        control_revision = self._playback_control_revision
        deadline = loop.time() + self._drain_watchdog_timeout_seconds
        while True:
            self._playback_progress.clear()
            self._raise_if_failed()
            if generation != self._generation or self._stopped:
                return
            current = self._acknowledged_valid_bytes
            if current >= target:
                return
            if (
                current != acknowledged
                or self._paused
                or control_revision != self._playback_control_revision
            ):
                acknowledged = current
                control_revision = self._playback_control_revision
                deadline = loop.time() + self._drain_watchdog_timeout_seconds
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError(
                    "LiveKit audio helper stopped acknowledging playback for "
                    f"{self._drain_watchdog_timeout_seconds:g} seconds"
                    f"{self._stderr_suffix()}"
                )
            try:
                await asyncio.wait_for(
                    self._playback_progress.wait(),
                    timeout=min(0.1 if self._paused else remaining, remaining),
                )
            except TimeoutError:
                continue

    async def _send_control(
        self,
        opcode: int,
        *,
        generation: int | None = None,
    ) -> None:
        self._raise_if_failed()
        if self._process is None:
            raise RuntimeError("LiveKit audio helper is not started")
        value = self._generation if generation is None else generation
        await self._queue_packet(
            _control_packet(opcode, value),
            urgent=opcode != CONTROL_FLUSH,
        )

    def _enqueue_control_nowait(self, opcode: int, generation: int) -> None:
        try:
            self._urgent_outbound.put_nowait(_control_packet(opcode, generation))
            self._writer_wakeup.set()
        except asyncio.QueueFull as error:  # pragma: no cover - reset purges first
            failure = RuntimeError("LiveKit audio writer queue is full")
            self._set_terminal_error(failure)
            raise failure from error

    async def _queue_packet(self, packet: bytes, *, urgent: bool = False) -> None:
        self._raise_if_failed()
        queue = self._urgent_outbound if urgent else self._outbound
        try:
            queue.put_nowait(packet)
        except asyncio.QueueFull:
            try:
                await asyncio.wait_for(
                    queue.put(packet),
                    timeout=self._drain_watchdog_timeout_seconds,
                )
            except TimeoutError:
                failure = RuntimeError(
                    "LiveKit audio helper stopped accepting playback packets"
                )
                self._set_terminal_error(failure)
                raise failure from None
        self._writer_wakeup.set()
        self._raise_if_failed()

    async def _write_packets(self, stdin: _HelperStdin) -> None:
        pending: bytes | None = None
        try:
            while True:
                self._writer_wakeup.clear()
                microphone_packet = self._microphone_packet
                if microphone_packet is not None:
                    self._microphone_packet = None
                    stdin.write(microphone_packet)
                    await stdin.drain()
                speaker_packet = self._speaker_packet
                if speaker_packet is not None:
                    self._speaker_packet = None
                    stdin.write(speaker_packet)
                    await stdin.drain()
                try:
                    packet = self._urgent_outbound.get_nowait()
                except asyncio.QueueEmpty:
                    packet = None
                if packet is not None:
                    try:
                        generation = _PLAYBACK_GENERATION.unpack_from(packet, 9)[0]
                        if generation == self._generation:
                            stdin.write(packet)
                            await stdin.drain()
                    finally:
                        if self._ensure_processed_packet is packet:
                            self._ensure_processed_packet = None
                    continue

                if pending is None:
                    try:
                        pending = self._outbound.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                if pending is not None:
                    is_pcm = pending[:4] == PLAYBACK_MAGIC
                    generation = _PLAYBACK_GENERATION.unpack_from(
                        pending,
                        8 if is_pcm else 9,
                    )[0]
                    if generation != self._generation or self._stopped:
                        pending = None
                        continue
                    pcm_bytes = len(pending) - 16 if is_pcm else 0
                    in_flight = max(
                        0,
                        self._sent_valid_bytes - self._acknowledged_valid_bytes,
                    )
                    available = max(0, self._max_inflight_playback_bytes - in_flight)
                    if not is_pcm or available:
                        amount = min(pcm_bytes, available)
                        if is_pcm and amount < pcm_bytes:
                            # A previous append can end between 10 ms frames.
                            # Fill the remaining window rather than waiting for
                            # an acknowledgement of a still-incomplete frame.
                            packet = _packet(
                                PLAYBACK_MAGIC,
                                _PLAYBACK_GENERATION.pack(generation)
                                + pending[16 : 16 + amount],
                            )
                            pending = _packet(
                                PLAYBACK_MAGIC,
                                _PLAYBACK_GENERATION.pack(generation)
                                + pending[16 + amount :],
                            )
                        else:
                            packet = pending
                            pending = None
                        self._sent_valid_bytes += amount
                        stdin.write(packet)
                        await stdin.drain()
                        continue
                # Only PCM is flow-controlled. Stop/reset/pause/resume bypass
                # this wait, while flush remains ordered behind all prior PCM.
                await self._writer_wakeup.wait()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._closing:
                self._set_terminal_error(error)

    async def _read_packets(self, stdout: asyncio.StreamReader) -> None:
        try:
            while True:
                header = await stdout.readexactly(_HEADER.size)
                magic, payload_length = _HEADER.unpack(header)
                if magic == FRAME_MAGIC:
                    expected_length = FRAME_PAYLOAD_BYTES
                elif magic == RENDER_MAGIC:
                    expected_length = RENDER_PAYLOAD_BYTES
                else:
                    raise LiveKitAudioProtocolError(
                        f"invalid LiveKit audio packet magic: {magic!r}"
                    )
                if payload_length != expected_length:
                    raise LiveKitAudioProtocolError(
                        "invalid LiveKit audio payload length: "
                        f"{payload_length}, expected {expected_length} for "
                        f"{magic.decode('ascii', errors='replace')}"
                    )
                payload = await stdout.readexactly(payload_length)
                if magic == FRAME_MAGIC:
                    self._accept_capture(payload)
                else:
                    self._accept_render(payload)
        except asyncio.CancelledError:
            raise
        except asyncio.IncompleteReadError as error:
            if not self._closing:
                self._set_terminal_error(
                    RuntimeError(
                        "LiveKit audio helper stdout closed unexpectedly "
                        f"after {len(error.partial)} partial bytes"
                    )
                )
        except Exception as error:
            if not self._closing:
                self._set_terminal_error(error)

    def _accept_capture(self, payload: bytes) -> None:
        sequence, captured_at_ns = _CAPTURE_PREFIX.unpack_from(payload)
        pcm = payload[_CAPTURE_PREFIX.size :]
        if len(pcm) != PCM_FRAME_BYTES:
            raise LiveKitAudioProtocolError(
                f"invalid LiveKit capture PCM length: {len(pcm)}"
            )
        self._accept_sequence(sequence)
        # VPIO already used the synchronized native speaker reference. Feeding
        # an asynchronous FMR packet back into Python would be a stale reverse
        # reference and could keep the standby fast path disabled after TTS ends.
        frame = (bytes(pcm), _SILENCE_FRAME, captured_at_ns)
        try:
            self._frames.put_nowait(frame)
        except asyncio.QueueFull:
            # Voice interaction values recency over completeness. Drop the
            # oldest queued frame instead of retaining a latency backlog.
            self._frames.get_nowait()
            self._dropped_frames += 1
            self._frames.put_nowait(frame)
        self._first_frame.set()

    def _accept_render(self, payload: bytes) -> None:
        (
            generation,
            cumulative_valid_bytes,
            _rendered_at_ns,
            valid_byte_count,
        ) = _RENDER_PREFIX.unpack_from(payload)
        rendered_pcm = bytes(payload[_RENDER_PREFIX.size :])
        if len(rendered_pcm) != PCM_FRAME_BYTES:
            raise LiveKitAudioProtocolError(
                f"invalid LiveKit rendered PCM length: {len(rendered_pcm)}"
            )
        if valid_byte_count > PCM_FRAME_BYTES or valid_byte_count % 2:
            raise LiveKitAudioProtocolError(
                f"invalid LiveKit rendered valid byte count: {valid_byte_count}"
            )
        if generation < self._generation or self._stopped:
            return
        if generation > self._generation:
            raise LiveKitAudioProtocolError(
                "LiveKit rendered generation is ahead of Python: "
                f"current={self._generation}, rendered={generation}"
            )
        previous = self._acknowledged_valid_bytes
        if cumulative_valid_bytes < previous:
            raise LiveKitAudioProtocolError(
                "LiveKit rendered byte count moved backwards: "
                f"previous={previous}, current={cumulative_valid_bytes}"
            )
        if cumulative_valid_bytes > self._queued_valid_bytes:
            raise LiveKitAudioProtocolError(
                "LiveKit acknowledged more bytes than Python queued: "
                f"queued={self._queued_valid_bytes}, "
                f"rendered={cumulative_valid_bytes}"
            )
        if cumulative_valid_bytes % 2:
            raise LiveKitAudioProtocolError(
                "LiveKit cumulative rendered byte count must be PCM16 aligned"
            )
        delta = cumulative_valid_bytes - previous
        if valid_byte_count > delta:
            raise LiveKitAudioProtocolError(
                "LiveKit rendered frame exceeds its cumulative acknowledgement: "
                f"delta={delta}, valid={valid_byte_count}"
            )
        first_owner = self._owner_at_byte(previous) if delta else None
        if delta:
            self._tracker.consume(delta)
            self._acknowledged_valid_bytes = cumulative_valid_bytes
            while (
                self._segments and self._segments[0].end_byte <= cumulative_valid_bytes
            ):
                self._segments.popleft()
            if not self._playback_started_reported and not self._stopped:
                self._playback_started_reported = True
                self._invoke_playback_started(first_owner)

        if valid_byte_count and not self._stopped:
            self._invoke_rendered_audio(rendered_pcm)
        self._signal_playback_progress()

    def _owner_at_byte(self, byte_offset: int) -> PlaybackOwner | None:
        for segment in self._segments:
            if segment.start_byte <= byte_offset < segment.end_byte:
                return segment.owner
        return None

    def _accept_sequence(self, sequence: int) -> None:
        previous = self._last_sequence
        if previous is None:
            if sequence != 0:
                raise LiveKitAudioProtocolError(
                    f"first LiveKit audio sequence must be 0, got {sequence}"
                )
        elif sequence <= previous:
            raise LiveKitAudioProtocolError(
                "LiveKit audio sequence must increase strictly: "
                f"previous={previous}, current={sequence}"
            )
        else:
            self._dropped_frames += sequence - previous - 1
        self._last_sequence = sequence

    async def _drain_stderr(self, stderr: asyncio.StreamReader) -> None:
        pending = bytearray()
        while True:
            chunk = await stderr.read(1_024)
            if not chunk:
                return
            self._stderr_tail.extend(chunk)
            overflow = len(self._stderr_tail) - _STDERR_TAIL_BYTES
            if overflow > 0:
                del self._stderr_tail[:overflow]
            pending.extend(chunk)
            *lines, rest = pending.split(b"\n")
            pending = bytearray(rest[-_STDERR_TAIL_BYTES:])
            for line in lines:
                _log_vpio_transition(line)

    def _begin_generation(self, *, text_offset: int) -> None:
        self._generation += 1
        self._queued_valid_bytes = 0
        self._sent_valid_bytes = 0
        self._acknowledged_valid_bytes = 0
        self._segments.clear()
        self._tracker = PlayedOffsetTracker(initial_offset=text_offset)
        self._paused = False
        self._ducked = False
        self._stopped = False
        self._playback_started_reported = False
        self._signal_playback_progress()

    def _reset_transport_state(self) -> None:
        self._clear_queue(self._frames)
        self._discard_outbound_packets()
        self._microphone_packet = None
        self._speaker_packet = None
        self._first_frame = asyncio.Event()
        self._playback_progress = asyncio.Event()
        self._writer_wakeup = asyncio.Event()
        self._terminal_error = None
        self._stderr_tail.clear()
        self._last_sequence = None
        self._dropped_frames = 0
        self._last_dequeued_frame_age_ms = 0.0
        self._last_dequeued_frame_queue_depth = 0
        self._closing = False

    def _discard_outbound_packets(self) -> None:
        self._clear_queue(self._outbound)
        self._clear_queue(self._urgent_outbound)
        self._ensure_processed_packet = None
        self._writer_wakeup.set()

    @staticmethod
    def _clear_queue(queue: asyncio.Queue[_QueueItem]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _set_terminal_error(self, error: BaseException) -> None:
        if self._terminal_error is None:
            self._terminal_error = error
        self._first_frame.set()
        self._signal_playback_progress()

    def _signal_playback_progress(self) -> None:
        self._playback_progress.set()
        self._writer_wakeup.set()

    def _raise_if_failed(self) -> None:
        if self._terminal_error is not None:
            raise self._failure()

    def _invoke_playback_started(self, owner: PlaybackOwner | None) -> None:
        callback = self._playback_started_callback
        if callback is None:
            return
        try:
            callback(owner)
        except Exception:
            logger.exception("LiveKit playback-start callback failed")

    def _invoke_rendered_audio(self, pcm: bytes) -> None:
        callback = self._rendered_audio_callback
        if callback is None:
            return
        try:
            callback(pcm)
        except Exception:
            logger.exception("LiveKit rendered-audio callback failed")

    def _failure(self) -> RuntimeError:
        error = self._terminal_error
        detail = str(error) if error is not None else "unknown helper failure"
        return RuntimeError(
            f"LiveKit audio helper failed: {detail}{self._stderr_suffix()}"
        )

    def _stderr_suffix(self) -> str:
        if not self._stderr_tail:
            return ""
        detail = self._stderr_tail.decode("utf-8", errors="replace").strip()
        return f"; stderr tail: {detail}" if detail else ""


_VPIO_TRANSITION_PREFIX = b"fish-memory-audio state adaptive-vpio transition "
# Only these known fields reach the log; other native output (device names,
# arbitrary errors) stays in the bounded tail used for failure messages.
_VPIO_TRANSITION_FIELDS = (
    "reason",
    "actualBypassed",
    "playbackGeneration",
    "succeeded",
)
_VPIO_TRANSITION_VALUE = re.compile(r"[A-Za-z0-9_-]{1,32}")


def _log_vpio_transition(line: bytes | bytearray) -> None:
    """Record Apple VPIO power-saving switches, so echo reports can be checked."""
    if not line.startswith(_VPIO_TRANSITION_PREFIX):
        return
    text = line[len(_VPIO_TRANSITION_PREFIX) :].decode(errors="replace")
    fields = dict(item.split("=", 1) for item in text.split() if "=" in item)
    kept = [
        f"{key}={fields[key]}"
        for key in _VPIO_TRANSITION_FIELDS
        if _VPIO_TRANSITION_VALUE.fullmatch(fields.get(key, ""))
    ]
    if kept:
        logger.info("VPIO 省电切换: %s", " ".join(kept))


def _packet(magic: bytes, payload: bytes) -> bytes:
    return _HEADER.pack(magic, len(payload)) + payload


def _control_packet(opcode: int, generation: int) -> bytes:
    if opcode not in {
        CONTROL_RESET,
        CONTROL_PAUSE,
        CONTROL_RESUME,
        CONTROL_DUCK,
        CONTROL_UNDUCK,
        CONTROL_STOP,
        CONTROL_FLUSH,
        CONTROL_ENSURE_PROCESSED,
        CONTROL_MUTE_MICROPHONE,
        CONTROL_UNMUTE_MICROPHONE,
        CONTROL_MUTE_SPEAKER,
        CONTROL_UNMUTE_SPEAKER,
    }:
        raise ValueError(f"unsupported LiveKit audio control opcode: {opcode}")
    return _packet(CONTROL_MAGIC, _CONTROL_PAYLOAD.pack(opcode, generation))


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
