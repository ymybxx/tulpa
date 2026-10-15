from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fish_memory.companion.livekit_vad as livekit_vad
from fish_memory.companion.livekit_vad import (
    LiveKitVad,
    LiveKitVadError,
    LiveKitVadService,
)
from fish_memory.companion.vad import SileroVadConfig


class FakeStream:
    def __init__(
        self,
        states: list[bool],
        *,
        defer_transition: bool = False,
        fail: str | None = None,
        probabilities: list[object] | None = None,
    ) -> None:
        self.states = states
        self.defer_transition = defer_transition
        self.fail = fail
        self.probabilities = probabilities
        self.frames: list[bytes] = []
        self.closed = False
        self.samples = 0
        self.inferred = 0
        self.speaking = False
        self.speech_duration = 0.0
        self.events: asyncio.Queue[Any] = asyncio.Queue()
        self.tasks: set[asyncio.Task[None]] = set()

    def push_frame(self, frame: bytes) -> None:
        self.frames.append(frame)
        self.samples += len(frame) // 2
        if self.samples // 512 <= self.inferred // 512:
            return
        if self.fail == "timeout":
            return
        if self.fail == "eof":
            self.events.put_nowait(None)
            return
        self.inferred += 512
        if self.fail == "index":
            self.events.put_nowait(
                SimpleNamespace(type="inference_done", samples_index=9, speaking=False)
            )
            return
        state = self.states[min(self.inferred // 512 - 1, len(self.states) - 1)]
        probability = (
            float(state)
            if self.probabilities is None
            else self.probabilities[
                min(self.inferred // 512 - 1, len(self.probabilities) - 1)
            ]
        )
        if self.speaking:
            self.speech_duration += 0.032
        self.events.put_nowait(
            SimpleNamespace(
                type="inference_done",
                samples_index=self.inferred,
                speaking=self.speaking,
                speech_duration=self.speech_duration,
                probability=probability,
            )
        )
        if state == self.speaking:
            return
        self.speaking = state
        if state:
            self.speech_duration = 0.032
        event = SimpleNamespace(
            type="start_of_speech" if state else "end_of_speech",
            samples_index=self.inferred,
            speech_duration=self.speech_duration,
        )
        if self.defer_transition:

            async def emit() -> None:
                await asyncio.sleep(0)
                self.events.put_nowait(event)

            task = asyncio.create_task(emit())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        else:
            self.events.put_nowait(event)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        event = await self.events.get()
        if event is None:
            raise StopAsyncIteration
        return event

    async def aclose(self) -> None:
        self.closed = True
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.events.put_nowait(None)
        if self.fail == "close":
            raise ValueError("close failed")


class FakeVadFactory:
    def __init__(
        self,
        states: list[bool] | None = None,
        *,
        defer_transition: bool = False,
        fail: str | None = None,
        probabilities: list[object] | None = None,
    ) -> None:
        self.states = states or [False]
        self.defer_transition = defer_transition
        self.fail = fail
        self.probabilities = probabilities
        self.options: list[dict[str, object]] = []
        self.updates: list[dict[str, object]] = []
        self.streams: list[FakeStream] = []
        self.load_thread: threading.Thread | None = None

    def __call__(self, **options: object) -> FakeVadFactory:
        self.options.append(options)
        self.load_thread = threading.current_thread()
        if self.fail == "load":
            raise ValueError("model load failed")
        return self

    def stream(self) -> FakeStream:
        value = FakeStream(
            self.states,
            defer_transition=self.defer_transition,
            fail=self.fail,
            probabilities=self.probabilities,
        )
        self.streams.append(value)
        return value

    def update_options(self, **options: object) -> None:
        self.updates.append(options)


def config() -> SileroVadConfig:
    return SileroVadConfig(model_dir=Path("/not-used-v4-model"))


def make_vad(factory: FakeVadFactory, **kwargs: object) -> LiveKitVad:
    return LiveKitVad(
        config(),
        vad_factory=factory,
        frame_factory=bytes,
        **kwargs,
    )


@pytest.mark.parametrize("defer_transition", [False, True])
def test_vad_returns_post_transition_state_for_the_same_window(
    defer_transition: bool,
) -> None:
    factory = FakeVadFactory([True, False], defer_transition=defer_transition)
    vad = make_vad(factory)
    try:
        frames = [bytes([i]) * 320 for i in range(7)]
        actual = [vad.is_speech(frame) for frame in frames]
        assert actual == [False, False, False, True, True, True, False]
        assert factory.streams[0].frames == frames
        assert factory.load_thread is not threading.current_thread()
    finally:
        vad.close()
    assert not vad._thread.is_alive()
    assert not any(t.name.startswith(vad._thread.name) for t in threading.enumerate())


def test_incomplete_windows_are_pushed_without_inventing_inference() -> None:
    factory = FakeVadFactory([True])
    vad = make_vad(factory)
    try:
        assert [vad.is_speech(bytes(320)) for _ in range(3)] == [False] * 3
        assert len(factory.streams[0].frames) == 3
        assert factory.streams[0].inferred == 0
        assert vad.candidate_voice_detected is None
        assert vad.is_speech(bytes(320)) is True
        assert factory.streams[0].inferred == 512
        assert vad.candidate_voice_detected is True
    finally:
        vad.close()


def test_reset_replaces_stream_and_clears_partial_window_and_state() -> None:
    factory = FakeVadFactory([True])
    vad = make_vad(factory)
    try:
        for _ in range(4):
            vad.is_speech(bytes(320))
        old = factory.streams[0]
        assert vad._speaking
        assert vad.candidate_voice_detected is True
        vad.reset()
        assert old.closed
        assert len(factory.streams) == 2
        assert len(factory.options) == 1
        assert vad.candidate_voice_detected is None
        assert [vad.is_speech(bytes(320)) for _ in range(3)] == [False] * 3
        assert vad.candidate_voice_detected is None
        assert vad.is_speech(bytes(320)) is True
        # Queued events from the retired stream have no consumer or authority.
        vad._loop.call_soon_threadsafe(
            old.events.put_nowait,
            SimpleNamespace(type="end_of_speech", samples_index=512),
        )
        assert vad.is_speech(bytes(320)) is True
    finally:
        vad.close()


def test_same_window_activity_can_be_silent_while_formal_speech_is_still_held() -> None:
    factory = FakeVadFactory([True, True, False], probabilities=[0.9, 0.05, 0.05])
    vad = make_vad(factory)
    try:
        actual = []
        for _ in range(10):
            formal = vad.is_speech(bytes(320))
            actual.append((formal, vad.candidate_voice_detected))
        assert actual == (
            [(False, None)] * 3
            + [(True, True)] * 3
            + [(True, False)] * 3
            + [(False, False)]
        )
        assert len(factory.options) == len(factory.streams) == 1
        assert len(factory.streams[0].frames) == 10
        assert factory.streams[0].inferred == 3 * 512
        assert factory.updates == []
    finally:
        vad.close()


def test_candidate_activity_uses_official_pre_transition_hysteresis() -> None:
    factory = FakeVadFactory(
        [False, True, True, True, False, False],
        probabilities=[0.25, 0.25, 0.11, 0.1, 0.09, 0.11],
    )
    vad = make_vad(factory)
    windows = []
    try:
        for _ in range(20):
            before = factory.streams[0].inferred
            formal = vad.is_speech(bytes(320))
            if factory.streams[0].inferred != before:
                windows.append((formal, vad.candidate_voice_detected))
        assert windows == [
            (False, True),
            (True, True),
            (True, True),
            (True, False),
            (False, False),
            (False, False),
        ]
    finally:
        vad.close()


@pytest.mark.parametrize("probability", [None, float("nan"), True, -0.1, 1.1])
def test_unavailable_probability_does_not_invent_candidate_silence(probability) -> None:
    factory = FakeVadFactory([True], probabilities=[probability])
    vad = make_vad(factory)
    try:
        for _ in range(4):
            formal = vad.is_speech(bytes(320))
        assert formal is True
        assert vad.candidate_voice_detected is None
    finally:
        vad.close()


@pytest.mark.parametrize(
    ("fail", "message"),
    [("timeout", "deadline"), ("eof", "ended unexpectedly"), ("index", "out of order")],
)
def test_stream_failure_raises_instead_of_becoming_permanent_silence(
    fail: str,
    message: str,
) -> None:
    factory = FakeVadFactory(fail=fail)
    vad = make_vad(factory, call_timeout_seconds=0.03)
    try:
        for _ in range(3):
            assert not vad.is_speech(bytes(320))
        with pytest.raises(LiveKitVadError, match=message):
            vad.is_speech(bytes(320))
        with pytest.raises(LiveKitVadError, match=message):
            vad.is_speech(bytes(320))
    finally:
        vad.close()
    assert not vad._thread.is_alive()


def test_load_failure_reclaims_worker_thread() -> None:
    factory = FakeVadFactory(fail="load")
    before = {thread.name for thread in threading.enumerate()}
    with pytest.raises(LiveKitVadError, match="ValueError"):
        make_vad(factory)
    assert {thread.name for thread in threading.enumerate()} == before


def test_close_is_idempotent_and_rejects_future_calls() -> None:
    factory = FakeVadFactory()
    vad = make_vad(factory)
    vad.close()
    vad.close()
    assert factory.streams[0].closed
    assert not vad._thread.is_alive()
    with pytest.raises(LiveKitVadError, match="closed"):
        vad.is_speech(bytes(320))
    with pytest.raises(LiveKitVadError, match="closed"):
        vad.reset()


def test_close_failure_is_bounded_and_does_not_leave_loop_running() -> None:
    vad = make_vad(FakeVadFactory(fail="close"))
    with pytest.raises(LiveKitVadError, match="did not close cleanly"):
        vad.close()
    assert not vad._thread.is_alive()
    vad.close()


def test_official_load_options_keep_configured_voice_boundaries() -> None:
    factory = FakeVadFactory()
    vad = make_vad(factory)
    vad.close()
    assert factory.options == [
        {
            "min_speech_duration": 0.08,
            "min_silence_duration": 0.35,
            "prefix_padding_duration": 0.22,
            "max_buffered_speech": 60.0,
            "activation_threshold": 0.25,
            "deactivation_threshold": 0.1,
            "sample_rate": 16_000,
            "force_cpu": True,
        }
    ]
    assert "onnx_file_path" not in factory.options[0]


@pytest.mark.asyncio
async def test_service_prepares_on_main_thread_and_opens_independent_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeVadFactory()
    prepared: list[threading.Thread] = []

    def prepare():
        prepared.append(threading.current_thread())
        return factory, bytes

    monkeypatch.setattr(livekit_vad, "_load_factories", prepare)
    service = LiveKitVadService(config())
    assert service.backend_name == "livekit_silero"
    assert not service.status()["available"]
    await service.start()
    await service.wait_until_ready()
    assert prepared == [threading.main_thread()]
    assert service.status()["available"]
    assert factory.options == []
    first, second = service.open_stream(), service.open_stream()
    try:
        assert first._thread is not second._thread
        assert len(factory.options) == 2
    finally:
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_service_reports_missing_dependency_without_creating_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable():
        raise LiveKitVadError("plugin is not installed")

    monkeypatch.setattr(livekit_vad, "_load_factories", unavailable)
    service = LiveKitVadService(config())
    with pytest.raises(LiveKitVadError, match="not installed"):
        await service.start()
    assert not service.status()["available"]
    assert service.status()["error"] == "plugin is not installed"


@pytest.mark.parametrize("pcm", [b"", bytes(318), bytes(322)])
def test_vad_rejects_invalid_frame_sizes(pcm: bytes) -> None:
    vad = make_vad(FakeVadFactory())
    try:
        with pytest.raises(ValueError, match="320"):
            vad.is_speech(pcm)
    finally:
        vad.close()


@pytest.mark.parametrize(
    "invalid",
    [
        replace(config(), sample_rate=8000),
        replace(config(), window_size=256),
        replace(config(), threshold=-0.1),
        replace(config(), min_speech_duration=-1),
    ],
)
def test_vad_rejects_unsupported_config(invalid: SileroVadConfig) -> None:
    with pytest.raises(ValueError):
        LiveKitVad(invalid, vad_factory=FakeVadFactory(), frame_factory=bytes)


@pytest.mark.parametrize("threshold", [0.25, 0.95])
def test_long_speech_guard_updates_official_options_without_forcing_end(
    threshold: float,
) -> None:
    factory = FakeVadFactory([True, True, True, False])
    vad = LiveKitVad(
        replace(config(), threshold=threshold, max_speech_duration=0.09),
        vad_factory=factory,
        frame_factory=bytes,
    )
    try:
        states = [vad.is_speech(bytes(320)) for _ in range(12)]
        assert states[-1] is True
        assert factory.updates == [
            {
                "activation_threshold": max(threshold, 0.9),
                "deactivation_threshold": max(threshold - 0.15, 0.75),
                "min_silence_duration": 0.1,
            }
        ]
        assert vad.is_speech(bytes(320)) is False
        assert factory.updates[-1] == {
            "activation_threshold": threshold,
            "deactivation_threshold": max(threshold - 0.15, 0.01),
            "min_silence_duration": 0.35,
        }
        assert len(factory.updates) == 2
        assert factory.options[0]["max_buffered_speech"] == 60.0
    finally:
        vad.close()


def test_reset_restores_long_speech_options_before_opening_fresh_stream() -> None:
    factory = FakeVadFactory([True])
    vad = LiveKitVad(
        replace(config(), max_speech_duration=0.09),
        vad_factory=factory,
        frame_factory=bytes,
    )
    try:
        for _ in range(12):
            vad.is_speech(bytes(320))
        assert vad._long_speech_guard
        vad.reset()
        assert not vad._long_speech_guard
        assert factory.updates[-1] == {
            "activation_threshold": 0.25,
            "deactivation_threshold": 0.1,
            "min_silence_duration": 0.35,
        }
        assert len(factory.updates) == 2
        assert not vad.is_speech(bytes(320))
    finally:
        vad.close()
