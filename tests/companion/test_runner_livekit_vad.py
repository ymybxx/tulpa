import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

import fish_memory.companion.runner as runner_module
from fish_memory.companion.runner import CompanionRunner
from fish_memory.companion.vad import SileroVadConfig, SileroVadService


def test_livekit_audio_selects_official_vad_without_changing_user_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    configs = []

    class Service:
        def __init__(self, config: SileroVadConfig) -> None:
            configs.append(config)

    monkeypatch.setattr(runner_module, "LiveKitVadService", Service)
    settings = SimpleNamespace(
        companion_audio_backend="livekit",
        companion_vad_model_dir=str(tmp_path / "legacy-vad"),
        companion_vad_threshold=0.25,
        companion_vad_min_speech_ms=80,
        companion_vad_min_silence_ms=350,
    )
    service = runner_module._default_vad_service(settings)
    assert isinstance(service, Service)
    config = configs[0]
    assert config.threshold == 0.25
    assert config.min_speech_duration == 0.08
    assert config.min_silence_duration == 0.35
    assert config.speech_pre_roll_frames == 22
    assert config.speech_confirmation_frames == 12
    assert not config.model_dir.exists()


def test_old_audio_mode_keeps_old_vad_without_initializing_livekit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Legacy audio must not initialize LiveKit VAD")

    monkeypatch.setattr(runner_module, "LiveKitVadService", unexpected)
    service = runner_module._default_vad_service(
        SimpleNamespace(companion_audio_backend="sounddevice")
    )
    assert isinstance(service, SileroVadService)


@pytest.mark.asyncio
async def test_partial_vad_startup_closes_first_stream_and_kws() -> None:
    calls = []
    connection = SimpleNamespace(close=lambda: calls.append("kws_close"))
    first = SimpleNamespace(close=lambda: calls.append("vad_close"))
    remaining = [first]

    def open_vad():
        if remaining:
            return remaining.pop()
        raise RuntimeError("second VAD failed")

    runner = _bare_runner()
    runner.settings = SimpleNamespace(wake_word="小鱼")
    runner._kws_service = SimpleNamespace(
        open_stream=AsyncMock(return_value=connection)
    )
    runner._kws_connection = None
    runner._vad_service = SimpleNamespace(open_stream=open_vad)
    runner._open_asr_stream = AsyncMock(return_value=object())

    with pytest.raises(RuntimeError, match="second VAD failed"):
        await runner._open_audio_processor()

    assert calls == ["vad_close", "kws_close"]
    assert runner._active_vad_streams == []


@pytest.mark.asyncio
async def test_vad_workers_close_even_when_processor_close_fails() -> None:
    calls = []

    class Input:
        capture_is_preprocessed = True

        async def start(self) -> None:
            pass

        async def run(
            self, callback, stop_event, *, start_if_needed: bool = True
        ) -> None:
            assert start_if_needed is False
            await callback(bytes(320), bytes(320))
            stop_event.set()

    class Processor:
        async def process_frame(self, *args) -> None:
            pass

        async def close(self) -> None:
            calls.append("processor_close")
            raise RuntimeError("processor close failed")

    runner = _bare_runner()
    runner.settings = SimpleNamespace(wake_word="小鱼")
    runner._audio_input = Input()
    runner._active_vad_streams = [
        SimpleNamespace(close=lambda: calls.append("primary_close")),
        SimpleNamespace(close=lambda: calls.append("interruption_close")),
    ]

    with pytest.raises(RuntimeError, match="processor close failed"):
        await runner._start_audio_backend()
        await runner._run_audio(UUID(int=1), Processor(), asyncio.Event())
    assert calls == ["processor_close", "primary_close", "interruption_close"]
    assert runner._active_vad_streams == []


@pytest.mark.asyncio
async def test_vad_worker_cleanup_continues_and_is_idempotent() -> None:
    calls = []

    def fail_close() -> None:
        calls.append("failed_close")
        raise RuntimeError("native close timeout")

    runner = _bare_runner()
    runner._active_vad_streams = [
        SimpleNamespace(close=fail_close),
        SimpleNamespace(close=lambda: calls.append("second_close")),
        object(),  # Legacy detector owns no async worker.
    ]
    await runner._close_vad_streams()
    await runner._close_vad_streams()
    assert calls == ["failed_close", "second_close"]


@pytest.mark.asyncio
async def test_official_vad_registration_is_main_thread_but_loading_is_not() -> None:
    calls = []
    stream = object()

    class Service:
        backend_name = "livekit_silero"

        async def start(self) -> None:
            calls.append(("prepare", threading.current_thread()))

        def open_stream(self):
            calls.append(("open", threading.current_thread()))
            return stream

    runner = _bare_runner()
    runner._vad_service = Service()
    assert await runner._open_vad_stream() is stream
    assert calls[0] == ("prepare", threading.main_thread())
    assert calls[1][0] == "open"
    assert calls[1][1] is not threading.main_thread()


@pytest.mark.asyncio
async def test_cancelled_vad_initialization_retires_unclaimed_worker() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = []
    stream = SimpleNamespace(close=lambda: calls.append("closed"))

    class Service:
        backend_name = "livekit_silero"

        async def start(self) -> None:
            pass

        def open_stream(self):
            started.set()
            assert release.wait(timeout=1)
            return stream

    runner = _bare_runner()
    runner._vad_service = Service()
    task = asyncio.create_task(runner._open_vad_stream())
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert calls == ["closed"]


def _bare_runner():
    runner = object.__new__(CompanionRunner)
    runner._microphone_enabled = True
    runner._playback_enabled = True
    runner._microphone_lock = asyncio.Lock()
    runner._audio_processor = None
    return runner
