import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

from fish_memory.companion.audio_processing import AudioFrameProcessor
from fish_memory.companion.vad import SileroVad, SileroVadConfig


class FakeSileroConfig:
    def __init__(
        self,
        model,
        threshold,
        min_silence_duration,
        min_speech_duration,
        window_size,
        max_speech_duration,
    ):
        self.model = model
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.window_size = window_size
        self.max_speech_duration = max_speech_duration


class FakeVadModelConfig:
    def __init__(self, silero_vad, sample_rate, num_threads, provider, debug):
        self.silero_vad = silero_vad
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.provider = provider
        self.debug = debug

    def validate(self):
        return True


class FakeVoiceActivityDetector:
    instances: list["FakeVoiceActivityDetector"] = []

    def __init__(self, config, buffer_size_in_seconds):
        self.config = config
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.samples: list[float] = []
        self.active = False
        self.__class__.instances.append(self)

    def accept_waveform(self, samples):
        self.samples.extend(samples)
        # The fake keeps the speech state across short low-energy windows,
        # which models the behavior the real VAD must provide.
        if any(abs(sample) > 0.02 for sample in samples):
            self.active = True

    def is_speech_detected(self):
        return self.active

    def reset(self):
        self.active = False
        self.samples.clear()

    def empty(self):
        return True

    def pop(self):
        return None


class BlockingVoiceActivityDetector(FakeVoiceActivityDetector):
    def __init__(self, config, buffer_size_in_seconds):
        super().__init__(config, buffer_size_in_seconds)
        self.accept_started = Event()
        self.release_accept = Event()
        self.reset_entered = Event()
        self.calls: list[str] = []

    def accept_waveform(self, samples):
        self.calls.append("accept:start")
        self.accept_started.set()
        if not self.release_accept.wait(timeout=1):
            raise TimeoutError("test did not release in-flight VAD detection")
        self.calls.append("accept:end")
        super().accept_waveform(samples)

    def is_speech_detected(self):
        self.calls.append("detect")
        return super().is_speech_detected()

    def reset(self):
        self.calls.append("reset")
        self.reset_entered.set()
        super().reset()


class PassthroughAec:
    def process_reverse_stream(self, pcm: bytes) -> bytes:
        return pcm

    def process_stream(self, pcm: bytes) -> bytes:
        return pcm


def test_silero_vad_uses_temporal_detector_and_preserves_short_gap(
    monkeypatch, tmp_path
):
    FakeVoiceActivityDetector.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            SileroVadModelConfig=FakeSileroConfig,
            VadModelConfig=FakeVadModelConfig,
            VoiceActivityDetector=FakeVoiceActivityDetector,
        ),
    )

    model_dir = tmp_path / "vad"
    model_dir.mkdir()
    (model_dir / "silero_vad.onnx").write_bytes(b"model")
    vad = SileroVad(
        SileroVadConfig(
            model_dir=model_dir,
            threshold=0.25,
            min_silence_duration=0.35,
            min_speech_duration=0.08,
            window_size=512,
            max_speech_duration=20.0,
        )
    )

    assert vad.is_speech(b"\x00\x10" * 160) is True
    assert vad.is_speech(b"\x00\x00" * 160) is True
    assert FakeVoiceActivityDetector.instances[0].config.silero_vad.threshold == 0.25
    assert FakeVoiceActivityDetector.instances[0].config.silero_vad.window_size == 512

    vad.reset()
    assert vad.is_speech(b"\x00\x00" * 160) is False


def test_silero_vad_pre_roll_includes_vad_window(tmp_path):
    config = SileroVadConfig(
        model_dir=tmp_path,
        min_speech_duration=0.08,
    )

    assert config.speech_pre_roll_frames == 22
    assert config.speech_confirmation_frames == 12


def test_silero_vad_serializes_reset_while_audio_frame_is_processing(
    monkeypatch, tmp_path
):
    BlockingVoiceActivityDetector.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(
            SileroVadModelConfig=FakeSileroConfig,
            VadModelConfig=FakeVadModelConfig,
            VoiceActivityDetector=BlockingVoiceActivityDetector,
        ),
    )

    model_dir = tmp_path / "vad"
    model_dir.mkdir()
    (model_dir / "silero_vad.onnx").write_bytes(b"model")
    vad = SileroVad(SileroVadConfig(model_dir=model_dir))
    processor = AudioFrameProcessor(PassthroughAec(), vad)
    detector = BlockingVoiceActivityDetector.instances[0]
    reset_call_started = Event()

    def reset_vad() -> None:
        reset_call_started.set()
        processor.reset_vad()

    with ThreadPoolExecutor(max_workers=2) as executor:
        detection = executor.submit(
            processor.process,
            capture_pcm=b"\x00\x10" * 160,
            render_pcm=b"\x00\x00" * 160,
        )
        assert detector.accept_started.wait(timeout=1)
        reset = executor.submit(reset_vad)
        assert reset_call_started.wait(timeout=1)

        reset_overlapped_detection = detector.reset_entered.wait(timeout=0.05)
        detector.release_accept.set()
        assert detection.result(timeout=1).voice_detected is True
        reset.result(timeout=1)

    assert reset_overlapped_detection is False
    assert detector.calls == ["accept:start", "accept:end", "detect", "reset"]
