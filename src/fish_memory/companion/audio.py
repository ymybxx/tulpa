"""Small native PCM input/output adapters and playback progress tracking."""

import asyncio
import math
import sys
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from time import monotonic
from typing import Any, Protocol, cast

from fish_memory.companion.playback import PlaybackOwner

DEFAULT_DUCK_GAIN = 0.2
DEFAULT_FRAME_WATCHDOG_TIMEOUT_SECONDS = 5.0
# A sentence up to this long is shown whole as it starts, like a subtitle. A
# longer one is revealed as it is heard: DashScope can merge a whole paragraph
# into one provider sentence, and live a 104-character sentence appeared in
# full while its voice still had half a minute to go.
REVEAL_ABOVE_CHARS = 30


def apply_pcm16_gain(buffer: Any, gain: float) -> None:
    """Scale signed 16-bit PCM in place without allocating another frame."""
    if not 0 <= gain <= 1:
        raise ValueError("PCM gain must be between 0 and 1")
    if gain == 1:
        return
    samples = memoryview(buffer).cast("h")
    for index, sample in enumerate(samples):
        samples[index] = round(sample * gain)


@dataclass(slots=True)
class _AudioSegment:
    byte_length: int
    start_offset: int
    end_offset: int
    consumed: int = 0


@dataclass(slots=True)
class _PendingDeviceRender:
    deadline: float
    byte_count: int
    owners: tuple[PlaybackOwner | None, ...]
    pcm: bytes


def _windows_stream_options(
    sd: Any,
    input_device: str | int | None,
    output_device: str | int | None,
    *,
    playback_only: bool = False,
) -> dict[str, object]:
    """Use shared WASAPI conversion for the 16 kHz speech stream on Windows.

    A normal Windows mixer commonly runs at 48 kHz. PortAudio's default host
    API can also be MME, so select WASAPI defaults explicitly. Explicit device
    choices remain respected and only WASAPI devices receive WASAPI options.
    """
    apis = sd.query_hostapis()
    wasapi = next(
        (i for i, api in enumerate(apis) if "WASAPI" in api["name"].upper()),
        None,
    )
    if wasapi is None:
        raise RuntimeError("Windows audio requires the WASAPI host API")
    api = apis[wasapi]
    devices = (
        api["default_input_device"] if input_device is None else input_device,
        api["default_output_device"] if output_device is None else output_device,
    )
    if playback_only:
        output = devices[1]
        if isinstance(output, int) and output < 0:
            raise RuntimeError("Windows default speaker is unavailable")
        return {
            "device": output,
            "extra_settings": sd.WasapiSettings(auto_convert=True)
            if sd.query_devices(output, "output")["hostapi"] == wasapi
            else None,
        }
    if any(isinstance(device, int) and device < 0 for device in devices):
        raise RuntimeError("Windows default microphone or speaker is unavailable")
    settings = tuple(
        sd.WasapiSettings(auto_convert=True)
        if sd.query_devices(device, kind)["hostapi"] == wasapi
        else None
        for device, kind in zip(devices, ("input", "output"), strict=True)
    )
    return {"device": devices, "extra_settings": settings}


class _OutputStream(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


def _stop_and_close_stream(stream: _OutputStream) -> None:
    """Attempt both native cleanup operations and retain every failure."""
    errors: list[BaseException] = []
    for operation in (stream.stop, stream.close):
        try:
            operation()
        except BaseException as error:
            errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("audio stream stop and close both failed", errors)


class PlayedOffsetTracker:
    """Track started sentences separately from confirmed sentence-end marks.

    PCM ranges authorize display when rendered; zero-byte marks authorize
    consumption only after every preceding PCM byte has been rendered. A long
    range is displayed in step with its rendered share of PCM: exact once all
    of its audio has arrived, estimated from the ranges already finished before.
    """

    def __init__(self, *, initial_offset: int = 0) -> None:
        if initial_offset < 0:
            raise ValueError("initial_offset must be nonnegative")
        self._segments: deque[_AudioSegment] = deque()
        self._played_offset = initial_offset
        self._display_offset = initial_offset
        self._queued_end_offset = initial_offset
        self._consumed_bytes = 0
        # PCM queued and rendered per text range, the range still receiving
        # audio, and bytes per character over the ranges that stopped.
        self._range_bytes: dict[tuple[int, int], int] = {}
        self._range_rendered: dict[tuple[int, int], int] = {}
        self._open_range: tuple[int, int] | None = None
        self._closed_bytes = self._closed_chars = 0

    @property
    def consumed_bytes(self) -> int:
        return self._consumed_bytes

    @property
    def played_offset(self) -> int:
        return self._played_offset

    @property
    def display_offset(self) -> int:
        return self._display_offset

    def mark(self, end_offset: int) -> None:
        if not 0 <= end_offset <= self._queued_end_offset:
            raise ValueError("playback mark exceeds queued text")
        self._close_range()
        self._segments.append(_AudioSegment(0, end_offset, end_offset))
        self.consume(0)

    def add_segment(
        self,
        *,
        byte_length: int,
        start_offset: int | None = None,
        end_offset: int | None = None,
        text_length: int | None = None,
    ) -> None:
        # text_length remains accepted for simple PCM callers. TTS callers
        # should provide the absolute answer range so multiple audio packets
        # cannot lose their position in the full response.
        if start_offset is None and end_offset is None:
            if text_length is None:
                raise ValueError("audio text range is missing")
            start_offset = self._queued_end_offset
            end_offset = start_offset + text_length
        elif start_offset is None or end_offset is None:
            raise ValueError("audio text range must have both offsets")
        if byte_length <= 0 or start_offset < 0 or end_offset < start_offset:
            raise ValueError("audio segment lengths are invalid")
        key = (start_offset, end_offset)
        if key != self._open_range:
            self._close_range()
            self._open_range = key
        self._range_bytes[key] = self._range_bytes.get(key, 0) + byte_length
        self._segments.append(_AudioSegment(byte_length, start_offset, end_offset))
        self._queued_end_offset = max(self._queued_end_offset, end_offset)

    def consume(self, byte_length: int) -> None:
        remaining = max(0, byte_length)
        while self._segments:
            segment = self._segments[0]
            if segment.byte_length == 0:
                self._played_offset = max(self._played_offset, segment.end_offset)
                self._segments.popleft()
                continue
            if not remaining:
                break
            available = segment.byte_length - segment.consumed
            consumed = min(remaining, available)
            segment.consumed += consumed
            self._consumed_bytes += consumed
            remaining -= consumed
            if consumed:
                key = (segment.start_offset, segment.end_offset)
                self._range_rendered[key] = self._range_rendered.get(key, 0) + consumed
                self._display_offset = max(self._display_offset, self._shown(key))
            if segment.consumed == segment.byte_length:
                self._segments.popleft()

    def _close_range(self) -> None:
        key, self._open_range = self._open_range, None
        if key is not None and key[1] > key[0]:
            self._closed_bytes += self._range_bytes[key]
            self._closed_chars += key[1] - key[0]

    def _shown(self, key: tuple[int, int]) -> int:
        """How much of a range to show once some of its PCM is rendered."""
        start, end = key
        chars = end - start
        if chars <= REVEAL_ABOVE_CHARS:
            return end
        total = float(self._range_bytes[key])
        if key == self._open_range:
            # More audio may still come; without a rate to judge by, show it all.
            if not self._closed_chars:
                return end
            total = max(total, chars * self._closed_bytes / self._closed_chars)
        heard = min(1.0, self._range_rendered[key] / total)
        return start + max(1, math.ceil(chars * heard - 1e-9))


class PcmPlayback:
    """Plays mono signed-16-bit PCM through the optional sounddevice package."""

    def __init__(
        self,
        *,
        sample_rate: int = 22_050,
        duck_gain: float = DEFAULT_DUCK_GAIN,
        block_size: int | None = None,
        latency: str | float | None = None,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not 0 <= duck_gain <= 1:
            raise ValueError("duck_gain must be between 0 and 1")
        if block_size is not None and block_size < 1:
            raise ValueError("block_size must be positive")
        self.sample_rate = sample_rate
        self.duck_gain = duck_gain
        self.block_size = block_size
        self.latency = latency
        self._stream_factory = stream_factory
        self._stream: _OutputStream | None = None
        self._queue: deque[bytearray] = deque()
        self._queue_owners: deque[PlaybackOwner | None] = deque()
        self._lock = Lock()
        self._tracker = PlayedOffsetTracker()
        self._paused = False
        self._ducked = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._playback_started_callback: (
            Callable[[PlaybackOwner | None], None] | None
        ) = None
        self._rendered_audio_callback: Callable[[bytes], None] | None = None
        self._playback_started_reported = False

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
        with self._lock:
            self._tracker.mark(end_offset)

    def reset(self, text_offset: int = 0) -> None:
        with self._lock:
            self._queue.clear()
            self._queue_owners.clear()
            self._tracker = PlayedOffsetTracker(initial_offset=text_offset)
            self._paused = False
            self._ducked = False
            self._playback_started_reported = False

    async def start(self) -> None:
        if self._stream is not None:
            return
        if self._stream_factory is None:
            try:
                sd = import_module("sounddevice")
            except ImportError as error:  # pragma: no cover - host dependency
                raise RuntimeError(
                    "native audio requires sounddevice and PortAudio"
                ) from error
            stream_factory = sd.RawOutputStream
        else:
            stream_factory = self._stream_factory
        self._loop = asyncio.get_running_loop()
        stream_options: dict[str, object] = {
            "samplerate": self.sample_rate,
            "channels": 1,
            "dtype": "int16",
            "callback": self._callback,
        }
        if self.block_size is not None:
            stream_options["blocksize"] = self.block_size
        if self.latency is not None:
            stream_options["latency"] = self.latency
        stream: _OutputStream | None = None
        try:
            stream = stream_factory(**stream_options)
            stream.start()
        except Exception:
            self._loop = None
            if stream is not None:
                with suppress(Exception):
                    _stop_and_close_stream(stream)
            raise
        self._stream = stream

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
        with self._lock:
            self._tracker.add_segment(
                byte_length=len(pcm),
                start_offset=start_offset,
                end_offset=end_offset,
            )
            self._queue.append(bytearray(pcm))
            self._queue_owners.append(owner)

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

    async def stop(self) -> None:
        stream = self._stream
        self._stream = None
        with self._lock:
            self._queue.clear()
            self._queue_owners.clear()
            self._paused = False
            self._ducked = False
            self._playback_started_reported = False
        self._loop = None
        if stream is not None:
            _stop_and_close_stream(stream)

    async def pause(self) -> None:
        with self._lock:
            self._paused = True

    async def resume(self) -> None:
        with self._lock:
            self._paused = False

    async def duck(self) -> None:
        with self._lock:
            self._ducked = True

    async def unduck(self) -> None:
        with self._lock:
            self._ducked = False

    async def wait_until_drained(self) -> None:
        while True:
            with self._lock:
                drained = not self._queue
            if drained:
                return
            await asyncio.sleep(0.01)

    async def wait_until_owner_drained(self, owner: PlaybackOwner) -> None:
        while True:
            with self._lock:
                drained = owner not in self._queue_owners
            if drained:
                return
            await asyncio.sleep(0.01)

    def _callback(self, outdata: bytearray, frames: int, *_: object) -> None:
        del frames
        target_length = len(outdata)
        written = 0
        playback_owner: PlaybackOwner | None = None
        playback_owner_selected = False
        with self._lock:
            if self._paused:
                outdata[:target_length] = b"\0" * target_length
                return
            while written < target_length and self._queue:
                current = self._queue[0]
                current_owner = self._queue_owners[0]
                amount = min(target_length - written, len(current))
                outdata[written : written + amount] = current[:amount]
                del current[:amount]
                written += amount
                if amount and not playback_owner_selected:
                    playback_owner = current_owner
                    playback_owner_selected = True
                self._tracker.consume(amount)
                if not current:
                    self._queue.popleft()
                    self._queue_owners.popleft()
            if written < target_length:
                outdata[written:target_length] = b"\0" * (target_length - written)
            if self._ducked:
                apply_pcm16_gain(outdata, self.duck_gain)
            report_playback_started = (
                written > 0 and not self._playback_started_reported
            )
            if report_playback_started:
                self._playback_started_reported = True
        loop = self._loop
        if loop is None or written <= 0:
            return
        rendered_pcm = bytes(outdata)
        if report_playback_started and self._playback_started_callback is not None:
            loop.call_soon_threadsafe(
                self._playback_started_callback,
                playback_owner,
            )
        if self._rendered_audio_callback is not None:
            loop.call_soon_threadsafe(
                self._rendered_audio_callback,
                rendered_pcm,
            )


class SoundDeviceFullDuplex:
    """One synchronized input/output stream for AEC and low-latency playback."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        block_size: int = 160,
        duck_gain: float = DEFAULT_DUCK_GAIN,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        stream_factory: Callable[..., Any] | None = None,
        frame_watchdog_timeout_seconds: float = (
            DEFAULT_FRAME_WATCHDOG_TIMEOUT_SECONDS
        ),
    ) -> None:
        if not 0 <= duck_gain <= 1:
            raise ValueError("duck_gain must be between 0 and 1")
        if frame_watchdog_timeout_seconds <= 0:
            raise ValueError("frame_watchdog_timeout_seconds must be positive")
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.duck_gain = duck_gain
        self.input_device = _normalize_sound_device(input_device)
        self.output_device = _normalize_sound_device(output_device)
        self._stream_factory = stream_factory
        self._frame_watchdog_timeout_seconds = frame_watchdog_timeout_seconds
        self._stream: _OutputStream | None = None
        self._playback_only = False
        self._queue: deque[bytearray] = deque()
        self._queue_owners: deque[PlaybackOwner | None] = deque()
        self._lock = Lock()
        self._tracker = PlayedOffsetTracker()
        self._paused = False
        self._ducked = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._frames: asyncio.Queue[tuple[bytes, bytes, float]] = asyncio.Queue(
            maxsize=64
        )
        self._pending_render: deque[_PendingDeviceRender] = deque()
        self._stream_error: str | None = None
        self._last_device_time: float | None = None
        self._stream_epoch = 0
        self._playback_epoch = 0
        self._frame_telemetry_lock = Lock()
        self._frame_queue_depth = 0
        self._last_dequeued_frame_age_ms = 0.0
        self._last_dequeued_frame_queue_depth = 0
        self._dropped_frames = 0
        self._playback_started_callback: (
            Callable[[PlaybackOwner | None], None] | None
        ) = None
        self._rendered_audio_callback: Callable[[bytes], None] | None = None
        self._playback_started_reported = False

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
        with self._lock:
            self._tracker.mark(end_offset)

    @property
    def dropped_frames(self) -> int:
        with self._frame_telemetry_lock:
            return self._dropped_frames

    @property
    def frame_queue_depth(self) -> int:
        with self._frame_telemetry_lock:
            return self._frame_queue_depth

    @property
    def last_dequeued_frame_age_ms(self) -> float:
        with self._frame_telemetry_lock:
            return self._last_dequeued_frame_age_ms

    @property
    def last_dequeued_frame_queue_depth(self) -> int:
        with self._frame_telemetry_lock:
            return self._last_dequeued_frame_queue_depth

    @property
    def stream_delay_ms(self) -> int | None:
        """Return PortAudio's capture plus playback latency estimate."""
        stream = self._stream
        if stream is None:
            return None
        latency = getattr(stream, "latency", None)
        if isinstance(latency, (tuple, list)) and len(latency) == 2:
            seconds = float(latency[0]) + float(latency[1])
        elif isinstance(latency, (int, float)) and not isinstance(latency, bool):
            seconds = float(latency)
        else:
            return None
        return max(0, round(seconds * 1000))

    async def start(self) -> None:
        if self._stream is not None:
            return
        extra_options: dict[str, object] = {}
        playback_only = False
        if self._stream_factory is None:
            try:
                sd = import_module("sounddevice")
            except ImportError as error:  # pragma: no cover - host dependency
                raise RuntimeError(
                    "native audio requires sounddevice and PortAudio"
                ) from error
            stream_factory = sd.RawStream
            if sys.platform == "win32":
                playback_only = not getattr(self, "_microphone_enabled", True)
                if playback_only:
                    stream_factory = sd.RawOutputStream
                extra_options = _windows_stream_options(
                    sd,
                    self.input_device,
                    self.output_device,
                    playback_only=playback_only,
                )
        else:
            stream_factory = self._stream_factory
        self._stream_epoch += 1
        stream_epoch = self._stream_epoch
        self._stream_error = None
        self._last_device_time = None

        def callback(*args: Any) -> None:
            if playback_only:
                output, frames, timing, status = args
                # Only an output clock exists. The muted direction carries
                # silence for scheduling, never a captured microphone frame.
                self._callback(
                    bytes(frames * 2),
                    output,
                    frames,
                    timing,
                    status,
                    stream_epoch=stream_epoch,
                )
            else:
                self._callback(*args, stream_epoch=stream_epoch)

        stream_options: dict[str, object] = {
            "samplerate": self.sample_rate,
            "channels": 1,
            "dtype": "int16",
            "blocksize": self.block_size,
            "latency": "low",
            "callback": callback,
        }
        if self.input_device is not None or self.output_device is not None:
            stream_options["device"] = (
                self.input_device,
                self.output_device,
            )
        stream_options.update(extra_options)
        stream: _OutputStream | None = None
        try:
            stream = stream_factory(**stream_options)
            stream.start()
        except Exception:
            self._stream_epoch += 1
            if stream is not None:
                with suppress(Exception):
                    _stop_and_close_stream(stream)
            raise
        self._stream = stream
        self._playback_only = playback_only

    async def run(
        self,
        on_frame: Callable[[bytes, bytes], Any],
        stop_event: asyncio.Event,
        *,
        start_if_needed: bool = True,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            if start_if_needed:
                await self.start()
            loop = asyncio.get_running_loop()
            frame_deadline = loop.time() + self._frame_watchdog_timeout_seconds
            while not stop_event.is_set():
                if self._stream_error:
                    raise RuntimeError(self._stream_error)
                if (
                    self._stream is not None
                    and getattr(self._stream, "active", True) is False
                ):
                    raise RuntimeError("native audio device became inactive")
                remaining = frame_deadline - loop.time()
                if remaining <= 0:
                    try:
                        capture_pcm, render_pcm, enqueued_at = self._frames.get_nowait()
                    except asyncio.QueueEmpty:
                        raise RuntimeError(
                            "native audio stream stopped delivering frames for "
                            f"{self._frame_watchdog_timeout_seconds:g} seconds"
                        ) from None
                else:
                    try:
                        capture_pcm, render_pcm, enqueued_at = await asyncio.wait_for(
                            self._frames.get(), timeout=min(0.1, remaining)
                        )
                    except TimeoutError:
                        continue
                self._record_dequeued_frame(enqueued_at)
                await on_frame(capture_pcm, render_pcm)
                frame_deadline = loop.time() + self._frame_watchdog_timeout_seconds
        finally:
            await self.close()

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
        with self._lock:
            self._tracker.add_segment(
                byte_length=len(pcm),
                start_offset=start_offset,
                end_offset=end_offset,
            )
            self._queue.append(bytearray(pcm))
            self._queue_owners.append(owner)

    def set_playback_started_callback(
        self,
        callback: Callable[[PlaybackOwner | None], None] | None,
    ) -> None:
        self._playback_started_callback = callback

    def set_rendered_audio_callback(
        self,
        callback: Callable[[bytes], None] | None,
    ) -> None:
        """Observe frames actually written to the output device.

        Delivery happens on the asyncio loop and is intentionally best effort,
        so an optional visualizer can never block PortAudio's callback thread.
        """
        self._rendered_audio_callback = callback

    async def stop(self) -> None:
        """Stop the current TTS output while keeping microphone capture alive."""
        self.reset()

    async def pause(self) -> None:
        """Mute speaker output without consuming queued TTS audio."""
        with self._lock:
            self._paused = True

    async def resume(self) -> None:
        with self._lock:
            self._paused = False

    async def duck(self) -> None:
        """Lower candidate speech playback while continuing to drain audio."""
        with self._lock:
            self._ducked = True

    async def unduck(self) -> None:
        with self._lock:
            self._ducked = False

    async def close(self) -> None:
        with self._lock:
            self._stream_epoch += 1
            self._playback_epoch += 1
            self._pending_render.clear()
            self._queue.clear()
            self._queue_owners.clear()
            self._paused = False
            self._ducked = False
        stream = self._stream
        self._stream = None
        self._loop = None
        try:
            if stream is not None:
                _stop_and_close_stream(stream)
        finally:
            while True:
                try:
                    self._frames.get_nowait()
                except asyncio.QueueEmpty:
                    break
            with self._frame_telemetry_lock:
                self._frame_queue_depth = 0

    def reset(self, text_offset: int = 0) -> None:
        with self._lock:
            self._playback_epoch += 1
            self._pending_render.clear()
            self._queue.clear()
            self._queue_owners.clear()
            self._tracker = PlayedOffsetTracker(initial_offset=text_offset)
            self._paused = False
            self._ducked = False
            self._playback_started_reported = False

    async def wait_until_drained(self) -> None:
        while True:
            with self._lock:
                if self._stream_error:
                    raise RuntimeError(self._stream_error)
                drained = not self._queue and not self._pending_render
            if drained:
                return
            await asyncio.sleep(0.01)

    async def wait_until_owner_drained(self, owner: PlaybackOwner) -> None:
        """Wait only for PCM queued by one immutable playback owner."""
        while True:
            with self._lock:
                if self._stream_error:
                    raise RuntimeError(self._stream_error)
                drained = owner not in self._queue_owners and not any(
                    owner in pending.owners for pending in self._pending_render
                )
            if drained:
                return
            await asyncio.sleep(0.01)

    def _callback(
        self,
        indata: object,
        outdata: object,
        frames: int,
        time_info: object = None,
        status: object = None,
        *,
        stream_epoch: int | None = None,
    ) -> None:
        output = cast(Any, outdata)
        target_length = len(output)
        written = 0
        owners: list[PlaybackOwner | None] = []
        confirmed: list[_PendingDeviceRender] = []
        with self._lock:
            if stream_epoch is not None and stream_epoch != self._stream_epoch:
                output[:target_length] = bytes(target_length)
                return
            accepted_stream_epoch = self._stream_epoch
            playback_epoch = self._playback_epoch
            # PortAudio flags indicate dropped/inserted frames, so its queued
            # DAC timeline is no longer evidence of playback. Let the existing
            # supervisor rebuild the device + AEC instead of over-reporting it.
            if any(
                getattr(status, name, False)
                for name in (
                    "output_underflow",
                    "output_overflow",
                    "input_overflow",
                )
            ):
                self._stream_error = "native audio stream discontinuity"
            current_time = getattr(time_info, "currentTime", None)
            dac_time = getattr(time_info, "outputBufferDacTime", None)
            if time_info is not None and (
                not isinstance(current_time, (float, int))
                or not isinstance(dac_time, (float, int))
                or not math.isfinite(current_time)
                or not math.isfinite(dac_time)
            ):
                self._stream_error = "native audio device returned invalid timestamps"
            elif current_time is not None:
                if (
                    self._last_device_time is not None
                    and current_time < self._last_device_time
                ):
                    self._stream_error = "native audio device clock moved backwards"
                self._last_device_time = current_time
            if self._stream_error:
                self._pending_render.clear()
                output[:target_length] = bytes(target_length)
                return
            if current_time is not None:
                while (
                    self._pending_render
                    and self._pending_render[0].deadline <= current_time
                ):
                    pending = self._pending_render.popleft()
                    self._tracker.consume(pending.byte_count)
                    confirmed.append(pending)
            if self._paused:
                output[:target_length] = b"\0" * target_length
            else:
                while written < target_length and self._queue:
                    current = self._queue[0]
                    current_owner = self._queue_owners[0]
                    amount = min(target_length - written, len(current))
                    output[written : written + amount] = current[:amount]
                    del current[:amount]
                    written += amount
                    if amount and current_owner not in owners:
                        owners.append(current_owner)
                    if not current:
                        self._queue.popleft()
                        self._queue_owners.popleft()
                if written < target_length:
                    output[written:target_length] = b"\0" * (target_length - written)
                if self._ducked:
                    apply_pcm16_gain(output, self.duck_gain)
            if not getattr(self, "_playback_enabled", True):
                output[:target_length] = bytes(target_length)
            rendered_pcm = bytes(output)
            if written:
                pending = _PendingDeviceRender(
                    deadline=(dac_time + frames / self.sample_rate)
                    if dac_time is not None
                    else 0,
                    byte_count=written,
                    owners=tuple(owners),
                    pcm=rendered_pcm,
                )
                if time_info is None:
                    # Injected/test streams without PortAudio timing preserve
                    # their synchronous callback contract. Real RawStream
                    # callbacks always supply the device clock.
                    self._tracker.consume(written)
                    confirmed.append(pending)
                else:
                    self._pending_render.append(pending)
                    if len(self._pending_render) > 512:
                        self._stream_error = (
                            "native audio playback clock stopped advancing"
                        )
            report_playback_started = (
                bool(confirmed) and not self._playback_started_reported
            )
            if report_playback_started:
                self._playback_started_reported = True
        loop = self._loop
        if loop is not None:
            if confirmed:
                loop.call_soon_threadsafe(
                    self._report_rendered,
                    confirmed,
                    report_playback_started,
                    playback_epoch,
                )
            loop.call_soon_threadsafe(
                self._enqueue_stream_frame,
                bytes(cast(Any, indata)),
                rendered_pcm,
                accepted_stream_epoch,
            )

    def _enqueue_stream_frame(
        self, capture_pcm: bytes, render_pcm: bytes, epoch: int
    ) -> None:
        if epoch == self._stream_epoch:
            self._enqueue_frame(capture_pcm, render_pcm)

    def _report_rendered(
        self, confirmed: list[_PendingDeviceRender], started: bool, epoch: int
    ) -> None:
        if epoch != self._playback_epoch:
            return
        if started and self._playback_started_callback is not None:
            self._playback_started_callback(confirmed[0].owners[0])
        if self._rendered_audio_callback is not None:
            for pending in confirmed:
                self._rendered_audio_callback(pending.pcm)

    @property
    def playback_only(self) -> bool:
        return self._playback_only

    def set_microphone_enabled(self, enabled: bool) -> None:
        if enabled and self._stream is not None and self._playback_only:
            # The supervisor must open an actual input stream before speech
            # can resume; an output clock cannot stand in for microphone data.
            self._stream_error = "microphone enabled; reopen the audio input stream"
        self._microphone_enabled = enabled
        while not self._frames.empty():
            self._frames.get_nowait()
        with self._frame_telemetry_lock:
            self._frame_queue_depth = 0

    def set_playback_enabled(self, enabled: bool) -> None:
        self._playback_enabled = enabled

    def _enqueue_frame(self, capture_pcm: bytes, render_pcm: bytes) -> None:
        if not getattr(self, "_microphone_enabled", True):
            capture_pcm = bytes(len(capture_pcm))
        try:
            self._frames.put_nowait((capture_pcm, render_pcm, monotonic()))
        except asyncio.QueueFull:
            with self._frame_telemetry_lock:
                self._dropped_frames += 1
        else:
            with self._frame_telemetry_lock:
                self._frame_queue_depth += 1

    def _record_dequeued_frame(self, enqueued_at: float) -> None:
        age_ms = round(max(0.0, (monotonic() - enqueued_at) * 1000), 3)
        with self._frame_telemetry_lock:
            self._frame_queue_depth = max(0, self._frame_queue_depth - 1)
            self._last_dequeued_frame_age_ms = age_ms
            self._last_dequeued_frame_queue_depth = self._frame_queue_depth


class SoundDevicePcmInput:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        block_size: int = 320,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._stream_factory = stream_factory

    async def run(
        self,
        on_pcm: Callable[[bytes], Any],
        stop_event: asyncio.Event,
    ) -> None:
        if self._stream_factory is None:
            try:
                sd = import_module("sounddevice")
            except ImportError as error:  # pragma: no cover - host dependency
                raise RuntimeError(
                    "native audio requires sounddevice and PortAudio"
                ) from error
            stream_factory = sd.RawInputStream
        else:
            stream_factory = self._stream_factory

        loop = asyncio.get_running_loop()
        blocks: asyncio.Queue[bytes] = asyncio.Queue()

        def callback(indata: object, *_: object) -> None:
            loop.call_soon_threadsafe(blocks.put_nowait, bytes(cast(Any, indata)))

        stream = stream_factory(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.block_size,
            callback=callback,
        )
        try:
            stream.start()
            while not stop_event.is_set():
                try:
                    pcm = await asyncio.wait_for(blocks.get(), timeout=0.1)
                except TimeoutError:
                    continue
                await on_pcm(pcm)
        finally:
            _stop_and_close_stream(stream)


def _normalize_sound_device(value: str | int | None) -> str | int | None:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value
