"""Synchronous frame adapter around the official local LiveKit Silero stream."""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import sys
import threading
from collections.abc import Callable, Coroutine
from importlib import import_module
from typing import Any, TypeVar

from fish_memory.companion.vad import SileroVadConfig

_FRAME_BYTES = 320
_FRAME_SAMPLES = 160
_WINDOW_SAMPLES = 512
_Result = TypeVar("_Result")
VadFactory = Callable[..., Any]
FrameFactory = Callable[[bytes], Any]


class LiveKitVadError(RuntimeError):
    """The local official VAD stream could not safely produce a frame decision."""


def _load_factories() -> tuple[VadFactory, FrameFactory]:
    # LiveKit registers plugins at import time and requires that registration on
    # the main thread. Model loading/inference can run on our dedicated thread.
    if (
        threading.current_thread() is not threading.main_thread()
        and "livekit.plugins.silero" not in sys.modules
    ):
        raise LiveKitVadError("Prepare the LiveKit Silero plugin on the main thread")
    try:
        silero = import_module("livekit.plugins.silero")
        rtc = import_module("livekit.rtc")
    except Exception as error:
        raise LiveKitVadError(
            "Local VAD requires the livekit-plugins-silero package"
        ) from error

    def frame(pcm: bytes) -> Any:
        return rtc.AudioFrame(
            data=pcm,
            sample_rate=16_000,
            num_channels=1,
            samples_per_channel=_FRAME_SAMPLES,
        )

    return silero.VAD.load, frame


def _validate_config(config: SileroVadConfig) -> None:
    if config.sample_rate != 16_000 or config.window_size != _WINDOW_SAMPLES:
        raise ValueError("LiveKit VAD requires 16 kHz audio and a 512-sample window")
    if not math.isfinite(config.threshold) or not 0 <= config.threshold <= 1:
        raise ValueError("LiveKit VAD threshold must be in [0, 1]")
    for value in (config.min_speech_duration, config.min_silence_duration):
        if not math.isfinite(value) or value < 0:
            raise ValueError("LiveKit VAD durations must be finite and nonnegative")
    if not math.isfinite(config.max_speech_duration) or config.max_speech_duration <= 0:
        raise ValueError("LiveKit VAD buffer duration must be finite and positive")


class LiveKitVadService:
    """Prepare a bundled local plugin; each opened stream owns its own worker."""

    backend_name = "livekit_silero"

    def __init__(
        self,
        config: SileroVadConfig,
        *,
        vad_factory: VadFactory | None = None,
        frame_factory: FrameFactory | None = None,
        call_timeout_seconds: float = 5.0,
        shutdown_timeout_seconds: float = 1.0,
    ) -> None:
        _validate_config(config)
        if (vad_factory is None) != (frame_factory is None):
            raise ValueError("Both LiveKit VAD factories must be supplied together")
        self.config = config
        self._vad_factory = vad_factory
        self._frame_factory = frame_factory
        self._call_timeout_seconds = call_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._ready = False
        self._error: str | None = None

    def _prepare(self) -> None:
        if self._ready:
            return
        try:
            if self._vad_factory is None or self._frame_factory is None:
                self._vad_factory, self._frame_factory = _load_factories()
        except Exception as error:
            self._error = str(error)
            raise
        self._ready = True
        self._error = None

    async def start(self) -> None:
        self._prepare()

    async def wait_until_ready(self, *, poll_seconds: float = 0.5) -> None:
        del poll_seconds  # The official model is bundled; no downloader is started.
        await self.start()

    def open_stream(self) -> LiveKitVad:
        self._prepare()
        assert self._vad_factory is not None and self._frame_factory is not None
        try:
            stream = LiveKitVad(
                self.config,
                vad_factory=self._vad_factory,
                frame_factory=self._frame_factory,
                call_timeout_seconds=self._call_timeout_seconds,
                shutdown_timeout_seconds=self._shutdown_timeout_seconds,
            )
        except Exception as error:
            self._error = str(error)
            raise
        self._error = None
        return stream

    def status(self) -> dict[str, object]:
        available = self._ready and self._error is None
        return {
            "available": available,
            "downloading": False,
            "modelDir": "livekit.plugins.silero.resources",
            "downloadedBytes": 0,
            "totalBytes": None,
            "downloadPercent": 100.0 if available else None,
            "error": self._error,
            "backend": self.backend_name,
        }


class LiveKitVad:
    """Keep the existing synchronous 10 ms contract, using official decisions.

    Every input frame is pushed unchanged into ``silero.VAD.load().stream()``.
    Only a frame that completes a 512-sample inference window waits for that
    window's events. START/END, rather than the pre-transition ``speaking``
    field on INFERENCE_DONE, own the returned speech state.

    Long speech retains the old soft-exit intent by temporarily updating the
    official thresholds. It never manufactures END or truncates ASR input, and
    does not claim sample-identical behavior to sherpa's buffer-size heuristic.
    """

    def __init__(
        self,
        config: SileroVadConfig,
        *,
        vad_factory: VadFactory | None = None,
        frame_factory: FrameFactory | None = None,
        call_timeout_seconds: float = 5.0,
        shutdown_timeout_seconds: float = 1.0,
    ) -> None:
        _validate_config(config)
        if (vad_factory is None) != (frame_factory is None):
            raise ValueError("Both LiveKit VAD factories must be supplied together")
        for value in (call_timeout_seconds, shutdown_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("LiveKit VAD timeouts must be finite and positive")
        if vad_factory is None or frame_factory is None:
            vad_factory, frame_factory = _load_factories()
        self._vad_factory = vad_factory
        self._frame_factory = frame_factory
        self._config = config
        self._call_timeout_seconds = call_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._call_lock = threading.RLock()
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._closed = False
        self._failure: LiveKitVadError | None = None
        self._vad: Any = None
        self._stream: Any = None
        self._consumer: asyncio.Task[None] | None = None
        self._generation = 0
        self._input_samples = 0
        self._inferred_samples = 0
        self._speaking = False
        self._candidate_voice_detected: bool | None = None
        self._long_speech_guard = False
        self._inference_waiter: asyncio.Future[None] | None = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"fish-livekit-vad-{id(self):x}",
            daemon=True,
        )
        self._thread.start()
        try:
            if not self._loop_ready.wait(timeout=call_timeout_seconds):
                raise LiveKitVadError("LiveKit VAD event loop did not start")
            self._call(self._initialize())
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def is_speech(self, pcm: bytes) -> bool:
        if len(pcm) != _FRAME_BYTES:
            raise ValueError("LiveKit VAD requires exactly 320 PCM16 bytes (10 ms)")
        with self._call_lock:
            self._require_open()
            return self._call(self._feed(bytes(pcm)))

    @property
    def candidate_voice_detected(self) -> bool | None:
        """Latest official inference activity, without the duration hold."""
        with self._call_lock:
            return self._candidate_voice_detected

    def reset(self) -> None:
        with self._call_lock:
            self._require_open()
            self._call(self._replace_stream())

    def close(self) -> None:
        with self._call_lock:
            if self._closed:
                return
            self._closed = True
            error: BaseException | None = None
            if self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._close_stream(), self._loop
                )
                try:
                    future.result(timeout=self._shutdown_timeout_seconds)
                except BaseException as caught:
                    future.cancel()
                    error = caught
                    self._failure = LiveKitVadError("LiveKit VAD closure failed")
                finally:
                    self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=self._shutdown_timeout_seconds)
            if self._thread.is_alive():
                raise LiveKitVadError("LiveKit VAD worker missed its shutdown deadline")
            self._vad = None
            if error is not None:
                raise LiveKitVadError("LiveKit VAD did not close cleanly") from error

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{self._thread.name}-inference",
        )
        self._loop.set_default_executor(executor)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.wait(pending, timeout=self._shutdown_timeout_seconds / 2)
                )
            # Normal closure happens between synchronous calls, with no native
            # inference outstanding. A failed native call cannot be force-killed
            # by Python; never wait without a bound on that exceptional path.
            executor.shutdown(wait=self._failure is None, cancel_futures=True)
            self._loop.close()

    def _require_open(self) -> None:
        if self._closed:
            raise LiveKitVadError("LiveKit VAD stream is closed")
        if self._failure is not None:
            raise self._failure
        if not self._thread.is_alive() or not self._loop.is_running():
            raise LiveKitVadError("LiveKit VAD worker is not running")

    def _call(self, coroutine: Coroutine[Any, Any, _Result]) -> _Result:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=self._call_timeout_seconds)
        except concurrent.futures.TimeoutError as error:
            future.cancel()
            failure = LiveKitVadError("LiveKit VAD operation exceeded its deadline")
            self._failure = failure
            raise failure from error
        except Exception as error:
            if isinstance(error, LiveKitVadError):
                self._failure = error
                raise
            failure = LiveKitVadError(
                f"LiveKit VAD operation failed: {type(error).__name__}"
            )
            self._failure = failure
            raise failure from error

    async def _initialize(self) -> None:
        config = self._config
        self._vad = self._vad_factory(
            min_speech_duration=config.min_speech_duration,
            min_silence_duration=config.min_silence_duration,
            prefix_padding_duration=config.speech_pre_roll_frames * 0.01,
            max_buffered_speech=max(config.max_speech_duration, 60.0),
            activation_threshold=config.threshold,
            deactivation_threshold=max(config.threshold - 0.15, 0.01),
            sample_rate=16_000,
            force_cpu=True,
        )
        await self._replace_stream()

    async def _replace_stream(self) -> None:
        await self._close_stream()
        self._restore_long_speech_options()
        self._input_samples = 0
        self._inferred_samples = 0
        self._speaking = False
        self._candidate_voice_detected = None
        self._stream = self._vad.stream()
        self._consumer = asyncio.create_task(
            self._consume_events(self._stream, self._generation),
            name="fish-livekit-vad-events",
        )

    async def _close_stream(self) -> None:
        self._generation += 1
        consumer = self._consumer
        stream = self._stream
        self._consumer = None
        self._stream = None
        if consumer is not None:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        if stream is not None:
            await stream.aclose()
        waiter = self._inference_waiter
        self._inference_waiter = None
        if waiter is not None and not waiter.done():
            waiter.cancel()

    async def _feed(self, pcm: bytes) -> bool:
        if self._failure is not None:
            raise self._failure
        stream = self._stream
        if stream is None:
            raise LiveKitVadError("LiveKit VAD has no active official stream")
        self._input_samples += _FRAME_SAMPLES
        target = self._input_samples // _WINDOW_SAMPLES * _WINDOW_SAMPLES
        waiter: asyncio.Future[None] | None = None
        if target > self._inferred_samples:
            waiter = self._loop.create_future()
            self._inference_waiter = waiter
        stream.push_frame(self._frame_factory(pcm))
        if waiter is not None:
            try:
                await waiter
                # LiveKit 1.7.1 emits INFERENCE_DONE followed synchronously by
                # any same-window START/END. Let its tee/metrics peer and this
                # consumer drain that transition before returning the state.
                await asyncio.sleep(0)
            finally:
                if self._inference_waiter is waiter:
                    self._inference_waiter = None
        if self._failure is not None:
            raise self._failure
        return self._speaking

    async def _consume_events(self, stream: Any, generation: int) -> None:
        try:
            async for event in stream:
                if generation != self._generation or self._closed:
                    return
                kind = getattr(event.type, "value", event.type)
                index = event.samples_index
                if isinstance(index, bool) or not isinstance(index, int):
                    raise LiveKitVadError("Invalid LiveKit VAD event sample index")
                if kind == "inference_done":
                    if index != self._inferred_samples + _WINDOW_SAMPLES:
                        raise LiveKitVadError("LiveKit VAD windows are out of order")
                    self._inferred_samples = index
                    self._candidate_voice_detected = self._window_activity(event)
                    self._observe_long_speech(event)
                    # Defer readiness behind the producer's queued event batch;
                    # the feed coroutine then gives the tee consumer its turn.
                    self._loop.call_soon(self._complete_window, generation)
                elif kind in {"start_of_speech", "end_of_speech"}:
                    if index != self._inferred_samples:
                        raise LiveKitVadError("Unmatched LiveKit VAD transition window")
                    self._speaking = kind == "start_of_speech"
                    if self._speaking:
                        self._observe_long_speech(event)
                    else:
                        self._restore_long_speech_options()
                else:
                    raise LiveKitVadError("LiveKit VAD emitted an unsupported event")
            if generation == self._generation and not self._closed:
                raise LiveKitVadError("LiveKit VAD event stream ended unexpectedly")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if generation != self._generation or self._closed:
                return
            self._failure = (
                error
                if isinstance(error, LiveKitVadError)
                else LiveKitVadError(
                    f"LiveKit VAD event stream failed: {type(error).__name__}"
                )
            )
            waiter = self._inference_waiter
            if waiter is not None and not waiter.done():
                waiter.set_exception(self._failure)

    def _complete_window(self, generation: int) -> None:
        if generation != self._generation or self._closed:
            return
        waiter = self._inference_waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    def _window_activity(self, event: Any) -> bool | None:
        # LiveKit emits its filtered probability before updating START/END.
        # Match that same window's hysteresis branch, not event.speaking alone:
        # speaking stays true throughout min_silence_duration.
        probability = getattr(event, "probability", None)
        speaking = getattr(event, "speaking", None)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            or not isinstance(speaking, bool)
        ):
            return None
        activation = self._config.threshold
        deactivation = max(activation - 0.15, 0.01)
        if self._long_speech_guard:
            activation = max(activation, 0.9)
            deactivation = max(self._config.threshold - 0.15, 0.75)
        return probability >= activation or (speaking and probability > deactivation)

    def _observe_long_speech(self, event: Any) -> None:
        if not self._speaking or self._long_speech_guard:
            return
        duration = getattr(event, "speech_duration", 0.0)
        if duration <= self._config.max_speech_duration:
            return
        self._vad.update_options(
            activation_threshold=max(self._config.threshold, 0.9),
            deactivation_threshold=max(self._config.threshold - 0.15, 0.75),
            min_silence_duration=0.1,
        )
        self._long_speech_guard = True

    def _restore_long_speech_options(self) -> None:
        if not self._long_speech_guard:
            return
        self._vad.update_options(
            activation_threshold=self._config.threshold,
            deactivation_threshold=max(self._config.threshold - 0.15, 0.01),
            min_silence_duration=self._config.min_silence_duration,
        )
        self._long_speech_guard = False
