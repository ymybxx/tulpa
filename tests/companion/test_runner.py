import asyncio
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

import fish_memory.companion.runner as runner_module
from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.runner import CompanionRunner
from fish_memory.companion.tts import TtsConfiguration
from fish_memory.companion.voice_drafts import (
    CandidateEvent,
    SpeculativeSilenceGate,
    VoiceDraftTurnController,
)


class _BlockedRunnerDecoder:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def recognize(self, pcm: bytes) -> str:
        del pcm
        self.started.set()
        self.release.wait(timeout=1)
        self.finished.set()
        return "过期草稿"


class _RunnerDraftClient:
    def __init__(self) -> None:
        self.upserts: list[str] = []

    async def upsert_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
    ) -> bool:
        del key, revision
        self.upserts.append(content)
        return True

    async def cancel_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        reason: str,
    ) -> bool:
        del key, revision, reason
        return True


def _open_blocked_runner_draft() -> tuple[
    VoiceDraftTurnController,
    SpeculativeSilenceGate,
    _BlockedRunnerDecoder,
    _RunnerDraftClient,
]:
    client = _RunnerDraftClient()
    decoder = _BlockedRunnerDecoder()
    controller = VoiceDraftTurnController(client=client, decoder=decoder)
    owner = controller.open_turn(
        fish_id=UUID("00000000-0000-0000-0000-000000000001"),
        session_id=UUID("00000000-0000-0000-0000-000000000002"),
        binding_generation=1,
        speech_generation=1,
    )
    controller.candidate(owner, pcm=b"a" * 320)
    gate = SpeculativeSilenceGate(silence_frames=1)
    gate.open_turn()
    return controller, gate, decoder, client


@pytest.mark.asyncio
async def test_runner_applies_device_latency_before_processing_audio(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[object] = []

    class FakeAudioInput:
        stream_delay_ms = 47

        async def start(self) -> None:
            calls.append("start")

        async def run(
            self,
            on_frame,
            stop_event: asyncio.Event,
            *,
            start_if_needed: bool = True,
        ) -> None:
            assert start_if_needed is False
            del stop_event
            calls.append("run")
            await on_frame(b"\0" * 320, b"\0" * 320)

    class FakeProcessor:
        def set_stream_delay(self, delay_ms: int) -> None:
            calls.append(("delay", delay_ms))

        async def process_frame(self, capture_pcm: bytes, render_pcm: bytes) -> None:
            del capture_pcm, render_pcm

        async def close(self) -> None:
            calls.append("close")

    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        companion_aec_delay_mode="auto",
        wake_word="小鱼",
    )
    runner._audio_input = FakeAudioInput()
    runner._session = SimpleNamespace(fish_id=None)

    caplog.set_level("INFO", logger=runner_module.__name__)
    await runner._start_audio_backend()
    await runner._run_audio(
        UUID("00000000-0000-0000-0000-000000000001"),
        FakeProcessor(),
        asyncio.Event(),
    )

    assert calls == ["start", ("delay", 47), "run", "close"]
    assert "已绑定助手 Fish" in caplog.text


@pytest.mark.asyncio
async def test_runner_starts_livekit_playback_before_frames() -> None:
    calls: list[str] = []

    class FakeAudioInput:
        capture_is_preprocessed = True

        async def start(self) -> None:
            calls.append("capture_start")

        async def run(
            self,
            on_frame,
            stop_event: asyncio.Event,
            *,
            start_if_needed: bool = True,
        ) -> None:
            assert start_if_needed is False
            calls.append("capture_run")
            await on_frame(bytes(320), bytes(320))
            stop_event.set()

    class FakeAudioOutput:
        async def start(self) -> None:
            calls.append("playback_start")

    class FakeProcessor:
        async def process_frame(self, capture_pcm: bytes, render_pcm: bytes) -> None:
            assert capture_pcm == bytes(320)
            assert render_pcm == bytes(320)
            calls.append("process")

        async def close(self) -> None:
            calls.append("processor_close")

    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        companion_aec_delay_mode="auto",
        wake_word="小鱼",
    )
    runner._audio_input = FakeAudioInput()
    runner._audio_output = FakeAudioOutput()
    runner._session = SimpleNamespace(fish_id=None)

    await runner._start_audio_backend()
    await runner._run_audio(
        UUID("00000000-0000-0000-0000-000000000001"),
        FakeProcessor(),
        asyncio.Event(),
    )

    assert calls == [
        "capture_start",
        "playback_start",
        "capture_run",
        "process",
        "processor_close",
    ]


@pytest.mark.asyncio
async def test_runner_does_not_report_ready_when_audio_start_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[str] = []

    class FakeAudioInput:
        async def start(self) -> None:
            raise RuntimeError("PortAudio start failed")

    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        companion_aec_delay_mode="fixed",
        wake_word="小鱼",
    )
    runner._audio_input = FakeAudioInput()
    caplog.set_level("INFO", logger=runner_module.__name__)

    with pytest.raises(RuntimeError, match="PortAudio start failed"):
        await runner._start_audio_backend()

    assert calls == []
    assert "已绑定助手 Fish" not in caplog.text


@pytest.mark.asyncio
async def test_runner_applies_microphone_mute_before_playback_only_start(caplog):
    calls = []

    class OutputOnlyAudio:
        playback_only = True

        def set_microphone_enabled(self, enabled):
            calls.append(("microphone", enabled))

        async def start(self):
            assert calls == [("microphone", False)]
            calls.append("start")

        async def run(self, on_frame, stop_event, *, start_if_needed=True):
            assert not start_if_needed
            await on_frame(bytes(320), b"r" * 320)
            stop_event.set()

    class Processor:
        async def process_frame(self, *_):
            pytest.fail("output-only clocks must not reach KWS/ASR/VAD")

        async def close(self):
            calls.append("close")

    runner = _bare_runner()
    runner._microphone_enabled = False
    runner._audio_input = OutputOnlyAudio()
    runner._session = SimpleNamespace(fish_id=None)
    runner.settings = SimpleNamespace(companion_aec_delay_mode="fixed")
    caplog.set_level("INFO", logger=runner_module.__name__)
    await runner._start_audio_backend()
    await runner._run_audio(
        UUID("00000000-0000-0000-0000-000000000001"), Processor(), asyncio.Event()
    )
    assert calls == [("microphone", False), "start", "close"]
    assert "麦克风已关闭，仅播放就绪" in caplog.text


@pytest.mark.asyncio
async def test_supervisor_start_failure_never_allocates_audio_models() -> None:
    calls: list[str] = []
    fish_id = UUID("00000000-0000-0000-0000-000000000001")

    class FakeAudioInput:
        async def start(self) -> None:
            calls.append("capture_start")
            raise RuntimeError("capture unavailable")

        async def close(self) -> None:
            calls.append("capture_close")

    class FakeAudioOutput:
        async def start(self) -> None:
            calls.append("playback_start")

        async def stop(self) -> None:
            calls.append("playback_stop")

    class FakeSession:
        def __init__(self) -> None:
            self.fish_id = fish_id

        async def reset_for_audio_recovery(self, bound_fish_id: UUID) -> bool:
            assert bound_fish_id == fish_id
            calls.append("session_reset")
            return False

    async def unexpected_open() -> tuple[object, object]:
        calls.append("models_open")
        raise AssertionError("models allocated before audio validation")

    runner = _bare_runner()
    runner._stop_event = asyncio.Event()
    runner._audio_input = FakeAudioInput()
    runner._audio_output = FakeAudioOutput()
    runner._session = FakeSession()
    runner._kws_connection = None
    runner._open_audio_processor = unexpected_open

    await runner._supervise_audio(fish_id, asyncio.Event())

    assert calls == [
        "capture_start",
        "session_reset",
        "capture_close",
        "playback_stop",
    ]


@pytest.mark.asyncio
async def test_supervisor_playback_start_failure_never_allocates_audio_models() -> None:
    calls: list[str] = []
    fish_id = UUID("00000000-0000-0000-0000-000000000001")

    class FakeAudioInput:
        async def start(self) -> None:
            calls.append("capture_start")

        async def close(self) -> None:
            calls.append("capture_close")

    class FakeAudioOutput:
        async def start(self) -> None:
            calls.append("playback_start")
            raise RuntimeError("playback unavailable")

        async def stop(self) -> None:
            calls.append("playback_stop")

    class FakeSession:
        def __init__(self) -> None:
            self.fish_id = fish_id

        async def reset_for_audio_recovery(self, bound_fish_id: UUID) -> bool:
            assert bound_fish_id == fish_id
            calls.append("session_reset")
            return False

    async def unexpected_open() -> tuple[object, object]:
        calls.append("models_open")
        raise AssertionError("models allocated before playback validation")

    runner = _bare_runner()
    runner._stop_event = asyncio.Event()
    runner._audio_input = FakeAudioInput()
    runner._audio_output = FakeAudioOutput()
    runner._session = FakeSession()
    runner._kws_connection = None
    runner._open_audio_processor = unexpected_open

    await runner._supervise_audio(fish_id, asyncio.Event())

    assert calls == [
        "capture_start",
        "playback_start",
        "session_reset",
        "capture_close",
        "playback_stop",
    ]


def test_default_livekit_backend_shares_processed_input_and_playback() -> None:
    settings = SimpleNamespace(
        companion_audio_backend="livekit",
        livekit_audio_helper="/tmp/fish-livekit-audio-helper",
        companion_audio_block_size=160,
        companion_input_device=None,
        companion_output_device=None,
    )

    capture, playback = runner_module._default_audio_backend(settings)

    assert capture.capture_is_preprocessed is True
    assert capture.helper_path == "/tmp/fish-livekit-audio-helper"
    assert playback.sample_rate == 16_000
    assert playback.block_size == 160
    assert capture is playback


@pytest.mark.parametrize(
    ("input_device", "output_device"),
    [("3", None), (None, "5"), ("3", "5")],
)
def test_default_livekit_backend_rejects_non_default_devices(
    input_device: str | None,
    output_device: str | None,
) -> None:
    settings = SimpleNamespace(
        companion_audio_backend="livekit",
        livekit_audio_helper="/tmp/fish-livekit-audio-helper",
        companion_audio_block_size=160,
        companion_input_device=input_device,
        companion_output_device=output_device,
    )

    with pytest.raises(ValueError, match="仅支持系统默认输入和输出设备"):
        runner_module._default_audio_backend(settings)


def test_default_livekit_backend_requires_helper_path() -> None:
    settings = SimpleNamespace(
        companion_audio_backend="livekit",
        livekit_audio_helper=None,
        companion_audio_block_size=160,
        companion_input_device=None,
        companion_output_device=None,
    )

    with pytest.raises(ValueError, match="FISH_MEMORY_LIVEKIT_AUDIO_HELPER"):
        runner_module._default_audio_backend(settings)


def test_default_livekit_backend_rejects_blank_helper_path() -> None:
    settings = SimpleNamespace(
        companion_audio_backend="livekit",
        livekit_audio_helper="   ",
        companion_audio_block_size=160,
        companion_input_device=None,
        companion_output_device=None,
    )

    with pytest.raises(ValueError, match="FISH_MEMORY_LIVEKIT_AUDIO_HELPER"):
        runner_module._default_audio_backend(settings)


def test_default_sounddevice_backend_remains_full_duplex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    full_duplex = object()

    def create_sounddevice(**kwargs: object) -> object:
        calls.append(kwargs)
        return full_duplex

    monkeypatch.setattr(
        runner_module,
        "SoundDeviceFullDuplex",
        create_sounddevice,
    )
    settings = SimpleNamespace(
        companion_audio_backend="sounddevice",
        companion_audio_block_size=160,
        companion_input_device="Built-in Microphone",
        companion_output_device="Built-in Output",
    )

    capture, playback = runner_module._default_audio_backend(settings)

    assert capture is full_duplex
    assert playback is full_duplex
    assert calls == [
        {
            "block_size": 160,
            "input_device": "Built-in Microphone",
            "output_device": "Built-in Output",
        }
    ]


@pytest.mark.asyncio
async def test_runner_rebuilds_audio_streams_but_reuses_asr_after_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fish_id = UUID("00000000-0000-0000-0000-000000000001")
    calls: list[object] = []

    class FakeConnection:
        def __init__(self, index: int) -> None:
            self.index = index
            self.closed = False

        def close(self) -> None:
            if not self.closed:
                calls.append(("kws_close", self.index))
                self.closed = True

    connections: list[FakeConnection] = []

    class FakeKwsService:
        async def open_stream(self, wake_word: tuple[str, ...]) -> FakeConnection:
            assert wake_word == ("小鱼",)
            connection = FakeConnection(len(connections) + 1)
            connections.append(connection)
            calls.append(("kws_open", connection.index))
            return connection

    class FakeAsrStream:
        def reset(self) -> None:
            calls.append(("asr_reset", self))

    class FakeAsrService:
        def __init__(self) -> None:
            self.stream = FakeAsrStream()

        def open_stream(self) -> object:
            calls.append(("asr_open", self.stream))
            return self.stream

    class FakeVadService:
        config = SimpleNamespace(
            threshold=0.25,
            min_silence_duration=0.35,
            min_speech_duration=0.08,
        )

        def open_stream(self) -> object:
            stream = object()
            calls.append(("vad_open", stream))
            return stream

    class FakeAudioInput:
        stream_delay_ms = None
        dropped_frames = 0

        def __init__(self) -> None:
            self.runs = 0

        async def start(self) -> None:
            calls.append("audio_start")

        async def run(
            self,
            on_frame,
            stop_event: asyncio.Event,
            *,
            start_if_needed: bool = True,
        ) -> None:
            assert start_if_needed is False
            del on_frame
            self.runs += 1
            calls.append(("audio_run", self.runs))
            if self.runs < 3:
                raise ValueError("vector")
            stop_event.set()

        async def close(self) -> None:
            calls.append("audio_close")

    class FakeAudioOutput:
        async def start(self) -> None:
            calls.append("output_start")

        async def stop(self) -> None:
            calls.append("output_stop")

    class FakeSharedAsrServer:
        def __init__(self) -> None:
            self.bindings: list[object | None] = []

        def bind(self, recognizer: object | None) -> None:
            self.bindings.append(recognizer)

    class FakeProcessor:
        def __init__(self, index: int) -> None:
            self.index = index

        def set_stream_delay(self, delay_ms: int) -> None:
            calls.append(("processor_delay", self.index, delay_ms))

        async def process_frame(self, capture_pcm: bytes, render_pcm: bytes) -> None:
            del capture_pcm, render_pcm

        async def close(self) -> None:
            calls.append(("processor_close", self.index))

    def fake_aec(**kwargs: object) -> object:
        calls.append(("aec_open", kwargs))
        return object()

    def fake_frame_processor(aec: object, vad: object) -> object:
        calls.append(("frame_processor_open", aec, vad))
        return object()

    processors: list[FakeProcessor] = []

    class FakeSession:
        def __init__(self) -> None:
            self.fish_id = fish_id
            self.reset_calls: list[UUID] = []

        async def reset_for_audio_recovery(self, bound_fish_id: UUID) -> bool:
            self.reset_calls.append(bound_fish_id)
            return self.fish_id == bound_fish_id

        async def resume_after_audio_recovery(self, bound_fish_id: UUID) -> bool:
            assert bound_fish_id == self.fish_id
            calls.append("session_resume")
            return True

    def create_processor(
        asr: object,
        frame_processor: object,
        interruption_vad: object,
        *,
        kws_connection: object,
    ) -> FakeProcessor:
        calls.append(
            (
                "processor_open",
                asr,
                frame_processor,
                interruption_vad,
                kws_connection,
            )
        )
        processor = FakeProcessor(len(processors) + 1)
        processors.append(processor)
        return processor

    monkeypatch.setattr(runner_module, "_AUDIO_RESTART_DELAYS_SECONDS", (0.0,))
    monkeypatch.setattr(runner_module, "WebRtcAec", fake_aec)
    monkeypatch.setattr(runner_module, "AudioFrameProcessor", fake_frame_processor)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        wake_word="小鱼",
        companion_aec_delay_mode="fixed",
        companion_aec_delay_ms=50,
    )
    runner._stop_event = asyncio.Event()
    runner._kws_connection = None
    runner._kws_service = FakeKwsService()
    asr_service = FakeAsrService()
    shared_asr_server = FakeSharedAsrServer()
    runner._asr_service = asr_service
    runner._shared_asr_server = shared_asr_server
    runner._vad_service = FakeVadService()
    runner._audio_input = FakeAudioInput()
    runner._audio_output = FakeAudioOutput()
    runner._session = FakeSession()
    runner._create_audio_processor = create_processor
    stop_event = asyncio.Event()

    await runner._supervise_audio(fish_id, stop_event)

    assert runner._session.fish_id == fish_id
    assert runner._session.reset_calls == [fish_id, fish_id]
    assert len(connections) == 3
    assert all(connection.closed for connection in connections)
    assert len(processors) == 3
    assert calls.count("audio_start") == 3
    assert calls.count("audio_close") == 3
    assert calls.count("output_start") == 3
    assert calls.count("output_stop") == 3
    assert calls.count("session_resume") == 2
    for index, call in enumerate(calls):
        if call == "session_resume":
            assert calls[index + 1][0] == "audio_run"
    assert _count_named_calls(calls, "asr_open") == 1
    assert _count_named_calls(calls, "asr_reset") == 3
    assert shared_asr_server.bindings == [asr_service.stream]
    assert _count_named_calls(calls, "vad_open") == 6
    assert _count_named_calls(calls, "aec_open") == 3
    assert ("processor_close", 1) in calls
    assert ("processor_close", 2) in calls
    assert ("processor_close", 3) in calls
    assert runner._kws_connection is None
    capture_starts = [
        index for index, call in enumerate(calls) if call == "audio_start"
    ]
    playback_starts = [
        index for index, call in enumerate(calls) if call == "output_start"
    ]
    asr_opens = [
        index
        for index, call in enumerate(calls)
        if isinstance(call, tuple) and call[0] == "asr_open"
    ]
    assert len(capture_starts) == len(playback_starts) == 3
    assert len(asr_opens) == 1
    asr_resets = [
        index
        for index, call in enumerate(calls)
        if isinstance(call, tuple) and call[0] == "asr_reset"
    ]
    assert len(asr_resets) == 3
    assert capture_starts[0] < playback_starts[0] < asr_opens[0] < asr_resets[0]
    assert all(
        capture < playback < reset
        for capture, playback, reset in zip(
            capture_starts, playback_starts, asr_resets, strict=True
        )
    )


@pytest.mark.asyncio
async def test_runner_uses_identity_aec_for_livekit_preprocessed_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    identity = object()
    frame_processor = object()

    class FakeProcessor:
        def set_stream_delay(self, value: int) -> None:
            calls.append(("delay", value))

        async def close(self) -> None:
            return None

    processor = FakeProcessor()

    class FakeConnection:
        def close(self) -> None:
            return None

    class FakeKwsService:
        async def open_stream(self, wake_word: tuple[str, ...]) -> FakeConnection:
            assert wake_word == ("小鱼",)
            return FakeConnection()

    class FakeVadService:
        config = SimpleNamespace(
            threshold=0.25,
            min_silence_duration=0.35,
            min_speech_duration=0.08,
        )

        def open_stream(self) -> object:
            return object()

    def identity_aec(*, stream_delay_ms: int) -> object:
        calls.append(("identity", stream_delay_ms))
        return identity

    def unexpected_webrtc(**_kwargs: object) -> object:
        raise AssertionError("preprocessed capture constructed a second WebRTC AEC")

    def open_frame_processor(aec: object, _vad: object) -> object:
        assert aec is identity
        return frame_processor

    async def open_asr_stream() -> object:
        return object()

    def create_processor(
        _asr: object,
        current_frame_processor: object,
        _interruption_vad: object,
        *,
        kws_connection: object,
    ) -> FakeProcessor:
        assert current_frame_processor is frame_processor
        assert isinstance(kws_connection, FakeConnection)
        return processor

    monkeypatch.setattr(runner_module, "IdentityAec", identity_aec)
    monkeypatch.setattr(runner_module, "WebRtcAec", unexpected_webrtc)
    monkeypatch.setattr(runner_module, "AudioFrameProcessor", open_frame_processor)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        wake_word="小鱼",
        companion_aec_delay_mode="auto",
        companion_aec_delay_ms=50,
    )
    runner._audio_input = SimpleNamespace(capture_is_preprocessed=True)
    runner._kws_service = FakeKwsService()
    runner._vad_service = FakeVadService()
    runner._kws_connection = None
    runner._open_asr_stream = open_asr_stream
    runner._create_audio_processor = create_processor

    opened, connection = await runner._open_audio_processor()

    assert opened is processor
    assert isinstance(connection, FakeConnection)
    assert calls == [("identity", 50), ("delay", 50)]


@pytest.mark.asyncio
async def test_runner_stops_during_audio_recovery_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fish_id = UUID("00000000-0000-0000-0000-000000000001")
    attempted = asyncio.Event()
    attempts = 0

    async def failing_open() -> tuple[object, object]:
        nonlocal attempts
        attempts += 1
        attempted.set()
        raise RuntimeError("audio device unavailable")

    class FakeAudioInput:
        async def start(self) -> None:
            return

        async def close(self) -> None:
            pass

    class FakeSession:
        def __init__(self, bound_fish_id: UUID) -> None:
            self.fish_id = bound_fish_id

        async def reset_for_audio_recovery(self, bound_fish_id: UUID) -> bool:
            return self.fish_id == bound_fish_id

    monkeypatch.setattr(runner_module, "_AUDIO_RESTART_DELAYS_SECONDS", (60.0,))
    runner = _bare_runner()
    runner._stop_event = asyncio.Event()
    runner._kws_connection = None
    runner._audio_input = FakeAudioInput()
    runner._session = FakeSession(fish_id)
    runner._open_audio_processor = failing_open
    stop_event = asyncio.Event()

    task = asyncio.create_task(runner._supervise_audio(fish_id, stop_event))
    await attempted.wait()
    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert attempts == 1


@pytest.mark.asyncio
async def test_runner_refreshes_bound_fish_voice_configuration() -> None:
    fish_id = UUID("00000000-0000-0000-0000-000000000001")
    configured: list[tuple[str, str]] = []
    draft_configured: list[tuple[str, str]] = []

    class FakeApi:
        async def get_voice_config(self, requested_fish_id: UUID):
            assert requested_fish_id == fish_id
            return SimpleNamespace(model="model-new", voice="voice-new")

    class FakeTtsOutput:
        def configure(self, *, model: str, voice: str) -> None:
            configured.append((model, voice))

    class FakeDraftTts:
        def configure(self, *, model: str, voice: str) -> None:
            draft_configured.append((model, voice))
            raise AssertionError(
                f"ordinary refresh configured draft output: {model}/{voice}"
            )

    runner = _bare_runner()
    runner._session = SimpleNamespace(fish_id=fish_id)
    runner._api = FakeApi()
    runner._tts_output = FakeTtsOutput()
    runner._draft_tts = FakeDraftTts()

    await runner._refresh_voice_config()

    assert configured == [("model-new", "voice-new")]
    assert draft_configured == []


@pytest.mark.asyncio
async def test_draft_voice_load_does_not_touch_active_public_output() -> None:
    fish_id = UUID("00000000-0000-0000-0000-000000000001")

    class FakeApi:
        async def get_voice_config(self, requested_fish_id: UUID):
            assert requested_fish_id == fish_id
            return SimpleNamespace(model="fresh-model", voice="fresh-voice")

    class ActivePublicOutput:
        def configure(self, *, model: str, voice: str) -> None:
            del model, voice
            raise RuntimeError("public provider is active")

    runner = _bare_runner()
    runner._session = SimpleNamespace(fish_id=fish_id)
    runner._api = FakeApi()
    runner._tts_output = ActivePublicOutput()

    configuration = await runner._load_voice_configuration()

    assert configuration == TtsConfiguration(
        "fresh-model",
        "fresh-voice",
        16_000,
    )


@pytest.mark.asyncio
async def test_runner_rebinds_after_releasing_old_audio_resources() -> None:
    old_fish_id = UUID("00000000-0000-0000-0000-000000000001")
    new_fish_id = UUID("00000000-0000-0000-0000-000000000002")
    calls: list[object] = []
    old_stop = asyncio.Event()

    async def old_audio() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            calls.append("old_audio_closed")

    class FakeConnection:
        def close(self) -> None:
            calls.append("old_kws_closed")

    class FakeSession:
        def __init__(self) -> None:
            self.fish_id = old_fish_id

        def quiesce_binding(self, fish_id: UUID | None) -> bool:
            calls.append(("quiesce", fish_id))
            return fish_id == self.fish_id

        async def set_binding(self, fish_id: UUID | None) -> None:
            calls.append(("binding", fish_id))
            self.fish_id = fish_id

    runner = _bare_runner()
    runner._stop_event = asyncio.Event()
    runner._audio_stop = old_stop
    runner._audio_task = asyncio.create_task(old_audio())
    runner._kws_connection = FakeConnection()
    runner._session = FakeSession()
    runner._tts_output = object()

    async def refresh_voice_config() -> None:
        calls.append(("voice", runner._session.fish_id))

    async def supervise_audio(
        fish_id: UUID,
        stop_event: asyncio.Event,
    ) -> None:
        calls.append(("new_audio", fish_id))
        await stop_event.wait()

    runner._refresh_voice_config = refresh_voice_config  # type: ignore[method-assign]
    runner._supervise_audio = supervise_audio  # type: ignore[method-assign]
    await asyncio.sleep(0)

    await runner._on_binding_change(new_fish_id)
    await asyncio.sleep(0)

    assert old_stop.is_set()
    assert calls == [
        ("quiesce", old_fish_id),
        "old_audio_closed",
        "old_kws_closed",
        ("binding", new_fish_id),
        ("voice", new_fish_id),
        ("new_audio", new_fish_id),
    ]
    assert runner._audio_task is not None
    assert runner._audio_task.get_name() == f"companion-audio-{new_fish_id}"

    await runner._stop_audio()


@pytest.mark.asyncio
async def test_runner_unbinding_releases_audio_without_starting_replacement() -> None:
    fish_id = UUID("00000000-0000-0000-0000-000000000001")
    calls: list[object] = []
    audio_stop = asyncio.Event()

    async def old_audio() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            calls.append("old_audio_closed")

    class FakeSession:
        def __init__(self) -> None:
            self.fish_id: UUID | None = fish_id

        def quiesce_binding(self, bound_fish_id: UUID | None) -> bool:
            calls.append(("quiesce", bound_fish_id))
            return bound_fish_id == self.fish_id

        async def set_binding(self, bound_fish_id: UUID | None) -> None:
            calls.append(("binding", bound_fish_id))
            self.fish_id = bound_fish_id

    runner = _bare_runner()
    runner._audio_stop = audio_stop
    runner._audio_task = asyncio.create_task(old_audio())
    runner._kws_connection = None
    runner._session = FakeSession()
    runner._tts_output = object()
    await asyncio.sleep(0)

    await runner._on_binding_change(None)

    assert audio_stop.is_set()
    assert calls == [
        ("quiesce", fish_id),
        "old_audio_closed",
        ("binding", None),
    ]
    assert runner._audio_stop is None
    assert runner._audio_task is None


@pytest.mark.asyncio
async def test_binding_is_quiesced_before_blocked_audio_stop_can_yield() -> None:
    old_fish_id = UUID(int=701)
    new_fish_id = UUID(int=702)
    calls: list[object] = []
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    class FakeSession:
        fish_id: UUID | None = old_fish_id

        def quiesce_binding(self, fish_id: UUID | None) -> bool:
            calls.append(("quiesce", fish_id))
            return True

        async def set_binding(self, fish_id: UUID | None) -> None:
            calls.append(("binding", fish_id))
            self.fish_id = fish_id

    runner = _bare_runner()
    runner._session = FakeSession()
    runner._tts_output = None
    runner._stop_event = asyncio.Event()

    async def blocked_stop_audio() -> None:
        calls.append("stop_started")
        stop_started.set()
        await release_stop.wait()
        calls.append("stop_finished")

    async def supervise_audio(
        fish_id: UUID,
        stop_event: asyncio.Event,
    ) -> None:
        del fish_id
        await stop_event.wait()

    runner._stop_audio = blocked_stop_audio  # type: ignore[method-assign]
    runner._supervise_audio = supervise_audio  # type: ignore[method-assign]
    switching = asyncio.create_task(runner._on_binding_change(new_fish_id))
    await asyncio.wait_for(stop_started.wait(), timeout=1)

    assert calls == [("quiesce", old_fish_id), "stop_started"]
    release_stop.set()
    await switching
    assert calls[-2:] == ["stop_finished", ("binding", new_fish_id)]
    runner._audio_stop.set()
    await runner._audio_task


@pytest.mark.asyncio
async def test_stop_audio_waits_for_live_processor_before_closing_drafts() -> None:
    calls: list[str] = []
    ready = asyncio.Event()
    client = _RunnerDraftClient()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=SimpleNamespace(recognize=lambda _pcm: "候选"),
    )

    async def live_audio() -> None:
        ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            calls.append("processor_close_started")
            controller.open_turn(
                fish_id=UUID("00000000-0000-0000-0000-000000000001"),
                session_id=UUID("00000000-0000-0000-0000-000000000002"),
                binding_generation=1,
                speech_generation=1,
            )
            calls.append("processor_close_finished")
            raise

    runner = _bare_runner()
    runner._audio_stop = asyncio.Event()
    runner._audio_task = asyncio.create_task(live_audio())
    runner._kws_connection = None
    runner._voice_draft_gate = SpeculativeSilenceGate(silence_frames=1)
    runner._voice_draft_vad = object()
    runner._voice_draft_decoder = object()
    runner._voice_draft_controller = controller
    await ready.wait()

    await runner._stop_audio()

    assert calls == ["processor_close_started", "processor_close_finished"]
    assert runner._voice_draft_controller is None
    with pytest.raises(RuntimeError, match="closed"):
        controller.open_turn(
            fish_id=UUID(int=1),
            session_id=UUID(int=2),
            binding_generation=2,
            speech_generation=2,
        )


@pytest.mark.asyncio
async def test_runner_loads_and_warms_smart_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeAnalyzer:
        def __init__(self, **kwargs: object) -> None:
            calls.append(("init", kwargs))

        def warm_up(self) -> None:
            calls.append(("warm_up",))

    monkeypatch.setattr(runner_module, "SmartTurnAnalyzer", FakeAnalyzer)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        companion_smart_turn_enabled=True,
        companion_smart_turn_model_path="/tmp/smart-turn.onnx",
        companion_smart_turn_threshold=0.65,
        companion_smart_turn_num_threads=2,
        companion_smart_turn_reopen_ms=800,
        companion_smart_turn_timeout_ms=500,
    )
    runner._smart_turn_analyzer = None

    await runner._load_smart_turn()

    assert isinstance(runner._smart_turn_analyzer, FakeAnalyzer)
    assert calls == [
        (
            "init",
            {
                "model_path": "/tmp/smart-turn.onnx",
                "threshold": 0.65,
                "num_threads": 2,
            },
        ),
        ("warm_up",),
    ]


@pytest.mark.asyncio
async def test_runner_keeps_smart_turn_optional_when_loading_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingAnalyzer:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(runner_module, "SmartTurnAnalyzer", FailingAnalyzer)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        companion_smart_turn_enabled=True,
        companion_smart_turn_model_path=None,
        companion_smart_turn_threshold=0.5,
        companion_smart_turn_num_threads=1,
    )
    runner._smart_turn_analyzer = None

    await runner._load_smart_turn()

    assert runner._smart_turn_analyzer is None


def test_runner_passes_voice_endpoint_settings_to_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcessor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

    monkeypatch.setattr(runner_module, "CompanionAudioProcessor", FakeProcessor)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        companion_utterance_commit_delay_ms=180,
        companion_smart_turn_reopen_ms=800,
        companion_smart_turn_timeout_ms=500,
        companion_early_interrupt_snapshot_ms=300,
        companion_early_interrupt_min_chars=2,
        companion_single_char_min_speech_ms=200,
    )
    runner._session = object()
    runner._kws_connection = object()
    runner._smart_turn_analyzer = object()
    runner._vad_service = SimpleNamespace(
        config=SimpleNamespace(
            speech_pre_roll_frames=3,
            speech_confirmation_frames=2,
            min_silence_duration=0.35,
            max_speech_duration=20.0,
        )
    )
    runner._observer = object()

    def notify_voice_activity() -> None:
        return

    runner._audio_input = SimpleNamespace(
        capture_is_preprocessed=True,
        dropped_frames=4,
        last_dequeued_frame_age_ms=12.5,
        frame_queue_depth=3,
        notify_voice_activity=notify_voice_activity,
    )
    runner._voice_draft_vad = object()
    runner._voice_draft_gate = object()
    runner._voice_draft_controller = object()
    asr = object()
    frame_processor = object()
    interruption_vad = object()

    processor = runner._create_audio_processor(
        asr,
        frame_processor,
        interruption_vad,
    )

    assert isinstance(processor, FakeProcessor)
    assert captured["args"] == (runner._session, runner._kws_connection, asr)
    assert captured["kwargs"] == {
        "frame_processor": frame_processor,
        "interruption_vad": interruption_vad,
        "defer_interruption_output_until_near_end": True,
        "owner_voice_gate": None,
        "speech_start_frames": 1,
        "speech_pre_roll_frames": 3,
        "speech_confirmation_frames": 2,
        "speech_end_frames": 1,
        "utterance_commit_delay_seconds": 0.18,
        "smart_turn": runner._smart_turn_analyzer,
        "smart_turn_reopen_seconds": 0.8,
        "smart_turn_timeout_seconds": 0.5,
        "early_interrupt_snapshot_seconds": 0.3,
        "early_interrupt_min_chars": 2,
        "single_char_min_speech_ms": 200,
        "vad_trailing_silence_ms": 350,
        "voice_draft_vad": runner._voice_draft_vad,
        "voice_draft_gate": runner._voice_draft_gate,
        "voice_draft_controller": runner._voice_draft_controller,
        "voice_draft_capture_vad": None,
        "voice_draft_max_pcm_bytes": 640_000,
        "diagnostics": runner._observer,
        "dropped_frames": captured["kwargs"]["dropped_frames"],
        "frame_queue_age_ms": cast(dict[str, object], captured["kwargs"])[
            "frame_queue_age_ms"
        ],
        "frame_queue_depth": cast(dict[str, object], captured["kwargs"])[
            "frame_queue_depth"
        ],
        "notify_voice_activity": notify_voice_activity,
    }
    assert captured["kwargs"]["dropped_frames"]() == 4
    assert (
        cast(
            Callable[[], float],
            cast(dict[str, object], captured["kwargs"])["frame_queue_age_ms"],
        )()
        == 12.5
    )
    assert (
        cast(
            Callable[[], int],
            cast(dict[str, object], captured["kwargs"])["frame_queue_depth"],
        )()
        == 3
    )


@pytest.mark.asyncio
async def test_runner_flushes_pet_voice_reports_before_closing_api() -> None:
    calls: list[str] = []

    class AsyncCloser:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            calls.append(self.name)

    class SharedAsrCloser:
        def __init__(self) -> None:
            self.bindings: list[object | None] = []

        def bind(self, recognizer: object | None) -> None:
            self.bindings.append(recognizer)

        async def close(self) -> None:
            calls.append("shared_asr")

    runner = _bare_runner()
    runner._closed = False
    runner._stop_event = asyncio.Event()
    runner._watcher = None
    runner._watch_task = None
    runner._audio_stop = None
    runner._audio_task = None
    runner._kws_connection = None
    shared_asr = SharedAsrCloser()
    runner._shared_asr_server = shared_asr
    runner._authoritative_asr = object()
    runner._playback_arbiter = PlaybackArbiter()
    runner._session = AsyncCloser("session")
    runner._performance_reporter = AsyncCloser("reporter")
    runner._diagnostics = AsyncCloser("diagnostics")
    runner._kws_service = AsyncCloser("kws")
    runner._api = AsyncCloser("api")
    runner._binding_client = AsyncCloser("binding")

    await runner.close()

    assert runner._shared_asr_server is None
    assert runner._authoritative_asr is None
    assert shared_asr.bindings == [None]
    assert calls.index("shared_asr") < calls.index("session")
    assert calls.index("session") < calls.index("reporter") < calls.index("api")


def _count_named_calls(calls: list[object], name: str) -> int:
    return sum(
        1
        for call in calls
        if isinstance(call, tuple) and bool(call) and call[0] == name
    )


@pytest.mark.asyncio
async def test_runner_disabled_mode_does_not_construct_draft_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller_thread = threading.get_ident()
    opened_on: list[int] = []

    class FakeAsrService:
        def open_stream(self) -> object:
            opened_on.append(threading.get_ident())
            return SimpleNamespace(reset=lambda: None)

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("disabled mode constructed a draft primitive")

    monkeypatch.setattr(runner_module, "SharedSenseVoiceDraftDecoder", unexpected)
    monkeypatch.setattr(runner_module, "WebRtcVad", unexpected)
    monkeypatch.setattr(runner_module, "SpeculativeSilenceGate", unexpected)
    monkeypatch.setattr(runner_module, "VoiceDraftTurnController", unexpected)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(companion_voice_draft_enabled=False)
    runner._asr_service = FakeAsrService()
    runner._voice_draft_client = object()
    runner._voice_draft_gate = None
    runner._voice_draft_vad = None
    runner._voice_draft_decoder = None
    runner._voice_draft_controller = None

    stream = await runner._open_asr_stream()

    assert stream is not None
    assert len(opened_on) == 1
    assert caller_thread not in opened_on
    assert runner._voice_draft_gate is None
    assert runner._voice_draft_vad is None
    assert runner._voice_draft_decoder is None
    assert runner._voice_draft_controller is None


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", [False, RuntimeError("old server"), "timeout"])
async def test_runner_local_enabled_server_missing_keeps_authoritative_path(
    capability: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authoritative = SimpleNamespace(reset=lambda: None)
    opened = 0
    authoritative_started = threading.Event()
    observed_during_probe: list[bool] = []

    class FakeAsrService:
        def open_stream(self) -> object:
            nonlocal opened
            opened += 1
            authoritative_started.set()
            return authoritative

    class FakeApi:
        async def voice_draft_capability(self) -> bool:
            if capability == "timeout":
                observed_during_probe.append(
                    await asyncio.to_thread(authoritative_started.wait, 0.02)
                )
                await asyncio.Event().wait()
            if isinstance(capability, BaseException):
                raise capability
            return capability is True

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("server-missing run constructed a draft primitive")

    monkeypatch.setattr(runner_module, "_VOICE_DRAFT_CAPABILITY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(runner_module, "SharedSenseVoiceDraftDecoder", unexpected)
    monkeypatch.setattr(runner_module, "WebRtcVad", unexpected)
    monkeypatch.setattr(runner_module, "SpeculativeSilenceGate", unexpected)
    monkeypatch.setattr(runner_module, "VoiceDraftTurnController", unexpected)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(companion_voice_draft_enabled=True)
    runner._api = FakeApi()
    runner._asr_service = FakeAsrService()
    runner._voice_draft_client = object()
    runner._voice_draft_gate = None
    runner._voice_draft_vad = None
    runner._voice_draft_decoder = None
    runner._voice_draft_controller = None

    stream = await asyncio.wait_for(runner._open_asr_stream(), timeout=0.2)

    assert stream is authoritative
    assert opened == 1
    assert runner._voice_draft_controller is None
    if capability == "timeout":
        assert observed_during_probe == [True]


@pytest.mark.asyncio
async def test_runner_replacement_invalidates_blocked_old_draft() -> None:
    controller, gate, decoder, client = _open_blocked_runner_draft()
    await asyncio.to_thread(decoder.started.wait, 1)
    authoritative = SimpleNamespace(reset=lambda: None)

    class FakeAsrService:
        def open_stream(self) -> object:
            return authoritative

    runner = _bare_runner()
    runner.settings = SimpleNamespace(companion_voice_draft_enabled=False)
    runner._asr_service = FakeAsrService()
    runner._api = SimpleNamespace(
        voice_draft_capability=lambda: asyncio.sleep(0, result=True)
    )
    runner._voice_draft_gate = gate
    runner._voice_draft_decoder = decoder
    runner._voice_draft_controller = controller

    assert await runner._open_asr_stream() is authoritative
    decoder.release.set()
    await asyncio.to_thread(decoder.finished.wait, 1)
    await asyncio.sleep(0)

    assert gate.accept(False) is CandidateEvent.NONE
    assert client.upserts == []
    assert runner._voice_draft_gate is None
    assert runner._voice_draft_decoder is None
    assert runner._voice_draft_controller is None


@pytest.mark.asyncio
async def test_runner_audio_recovery_invalidates_blocked_old_draft() -> None:
    fish_id = UUID("00000000-0000-0000-0000-000000000001")
    controller, gate, decoder, client = _open_blocked_runner_draft()
    await asyncio.to_thread(decoder.started.wait, 1)

    class FakeSession:
        def __init__(self) -> None:
            self.fish_id = fish_id

        async def reset_for_audio_recovery(self, bound_fish_id: UUID) -> bool:
            assert bound_fish_id == fish_id
            return False

    class FakeAudioInput:
        async def start(self) -> None:
            return

        async def close(self) -> None:
            return

    async def fail_open() -> tuple[object, object]:
        raise RuntimeError("audio recovery")

    runner = _bare_runner()
    runner._stop_event = asyncio.Event()
    runner._kws_connection = None
    runner._audio_input = FakeAudioInput()
    runner._session = FakeSession()
    runner._open_audio_processor = fail_open
    runner._voice_draft_gate = gate
    runner._voice_draft_decoder = decoder
    runner._voice_draft_controller = controller

    await runner._supervise_audio(fish_id, asyncio.Event())
    decoder.release.set()
    await asyncio.to_thread(decoder.finished.wait, 1)
    await asyncio.sleep(0)

    assert gate.accept(False) is CandidateEvent.NONE
    assert client.upserts == []


@pytest.mark.asyncio
async def test_runner_shutdown_invalidates_blocked_client_upsert() -> None:
    class BlockingClient(_RunnerDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            del key, revision
            self.started.set()
            await self.release.wait()
            self.upserts.append(content)
            return True

    class AsyncCloser:
        async def close(self) -> None:
            return

    client = BlockingClient()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=SimpleNamespace(recognize=lambda _pcm: "候选"),
    )
    owner = controller.open_turn(
        fish_id=UUID("00000000-0000-0000-0000-000000000001"),
        session_id=UUID("00000000-0000-0000-0000-000000000002"),
        binding_generation=1,
        speech_generation=1,
    )
    controller.candidate(owner, pcm=b"a" * 320)
    await client.started.wait()
    gate = SpeculativeSilenceGate(silence_frames=1)
    gate.open_turn()
    runner = _bare_runner()
    runner._closed = False
    runner._stop_event = asyncio.Event()
    runner._watcher = None
    runner._watch_task = None
    runner._audio_stop = None
    runner._audio_task = None
    runner._kws_connection = None
    runner._playback_arbiter = PlaybackArbiter()
    runner._voice_draft_gate = gate
    runner._voice_draft_decoder = object()
    runner._voice_draft_controller = controller
    runner._session = AsyncCloser()
    runner._pet_events = AsyncCloser()
    runner._performance_reporter = AsyncCloser()
    runner._diagnostics = AsyncCloser()
    runner._kws_service = AsyncCloser()
    runner._api = AsyncCloser()
    runner._binding_client = AsyncCloser()

    await runner.close()
    client.release.set()
    await asyncio.sleep(0)

    assert gate.accept(False) is CandidateEvent.NONE
    assert client.upserts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("candidate_ms", "expected_frames"), [(100, 10), (0, 1)])
async def test_runner_enabled_sensevoice_constructs_one_shared_recognizer(
    monkeypatch: pytest.MonkeyPatch,
    candidate_ms: int,
    expected_frames: int,
) -> None:
    caller_thread = threading.get_ident()
    construction_threads: list[int] = []
    authoritative = SimpleNamespace(reset=lambda: None)
    decoder = object()
    draft_vad = object()
    gate = object()
    controller = object()
    client = object()
    draft_tts = object()

    class FakeAsrService:
        def open_stream(self) -> object:
            construction_threads.append(threading.get_ident())
            return authoritative

    def open_decoder(source: object) -> object:
        assert source is authoritative
        return decoder

    def open_gate(*, silence_frames: int, prepare_silence_frames: int) -> object:
        assert silence_frames == expected_frames
        assert prepare_silence_frames == 1
        return gate

    def open_vad(*, mode: int) -> object:
        assert mode == 2
        return draft_vad

    def open_controller(
        *,
        client: object,
        decoder: object,
        draft_tts: object,
        before_context: object = None,
    ) -> object:
        assert client is runner._voice_draft_client
        assert decoder is runner._voice_draft_decoder
        assert draft_tts is runner._draft_tts
        return controller

    monkeypatch.setattr(runner_module, "SharedSenseVoiceDraftDecoder", open_decoder)
    monkeypatch.setattr(runner_module, "WebRtcVad", open_vad)
    monkeypatch.setattr(runner_module, "SpeculativeSilenceGate", open_gate)
    monkeypatch.setattr(runner_module, "VoiceDraftTurnController", open_controller)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(
        companion_voice_draft_enabled=True,
        companion_voice_draft_candidate_silence_ms=candidate_ms,
    )
    runner._asr_service = FakeAsrService()
    runner._api = SimpleNamespace(
        voice_draft_capability=lambda: asyncio.sleep(0, result=True)
    )
    runner._voice_draft_client = client
    runner._draft_tts = draft_tts
    runner._voice_draft_gate = None
    runner._voice_draft_vad = None
    runner._voice_draft_decoder = None
    runner._voice_draft_controller = None

    stream = await asyncio.wait_for(runner._open_asr_stream(), timeout=1)

    assert stream is authoritative
    assert len(construction_threads) == 1
    assert caller_thread not in construction_threads
    assert runner._voice_draft_gate is gate
    assert runner._voice_draft_vad is draft_vad
    assert runner._voice_draft_decoder is decoder
    assert runner._voice_draft_controller is controller


@pytest.mark.asyncio
async def test_runner_keeps_authoritative_asr_when_draft_decoder_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authoritative = SimpleNamespace(reset=lambda: None)

    class FakeAsrService:
        def open_stream(self) -> object:
            return authoritative

    def fail_decoder(_config: object) -> object:
        raise RuntimeError("draft model allocation failed")

    monkeypatch.setattr(runner_module, "SharedSenseVoiceDraftDecoder", fail_decoder)
    runner = _bare_runner()
    runner.settings = SimpleNamespace(companion_voice_draft_enabled=True)
    runner._asr_service = FakeAsrService()
    runner._api = SimpleNamespace(
        voice_draft_capability=lambda: asyncio.sleep(0, result=True)
    )
    runner._voice_draft_client = object()
    runner._voice_draft_gate = None
    runner._voice_draft_vad = None
    runner._voice_draft_decoder = None
    runner._voice_draft_controller = None

    stream = await runner._open_asr_stream()

    assert stream is authoritative
    assert runner._voice_draft_gate is None
    assert runner._voice_draft_vad is None
    assert runner._voice_draft_decoder is None
    assert runner._voice_draft_controller is None


def _bare_runner():
    runner = object.__new__(CompanionRunner)
    runner._microphone_enabled = True
    runner._playback_enabled = True
    runner._microphone_lock = asyncio.Lock()
    runner._audio_processor = None
    return runner
