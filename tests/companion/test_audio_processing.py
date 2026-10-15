import sys
from collections.abc import Iterable
from types import ModuleType

import pytest

from fish_memory.companion.audio_processing import (
    AudioFrameProcessor,
    IdentityAec,
    SpeechEvent,
    SpeechTurnDetector,
    WebRtcAec,
    silence_ms_to_frames,
    speech_pre_roll_ms_to_frames,
)

FRAME = b"c" * 320
RENDER = b"r" * 320


class FakeAudioProcessor:
    instances: list["FakeAudioProcessor"] = []

    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.delays: list[int] = []
        self.formats: list[tuple[str, tuple[object, ...]]] = []
        self.__class__.instances.append(self)

    def set_stream_format(self, *args: object) -> None:
        self.formats.append(("capture", args))

    def set_reverse_stream_format(self, *args: object) -> None:
        self.formats.append(("reverse", args))

    def set_stream_delay(self, delay_ms: int) -> None:
        self.delays.append(delay_ms)

    def process_reverse_stream(self, pcm: bytes) -> bytes:
        return pcm

    def process_stream(self, pcm: bytes) -> bytes:
        return pcm


def install_fake_aec_module(monkeypatch) -> None:
    FakeAudioProcessor.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "aec_audio_processing",
        ModuleType("aec_audio_processing"),
    )
    sys.modules["aec_audio_processing"].AudioProcessor = FakeAudioProcessor


def test_webrtc_aec_fixed_mode_reports_and_sets_stream_delay(monkeypatch) -> None:
    install_fake_aec_module(monkeypatch)

    aec = WebRtcAec(stream_delay_ms=50, delay_mode="fixed")

    assert aec.delay_mode == "fixed"
    assert FakeAudioProcessor.instances[-1].delays == [50]


def test_webrtc_aec_auto_mode_sets_initial_delay_and_allows_updates(
    monkeypatch,
) -> None:
    install_fake_aec_module(monkeypatch)

    aec = WebRtcAec(stream_delay_ms=50, delay_mode="auto")
    aec.set_stream_delay(70)

    assert aec.delay_mode == "auto"
    assert aec.stream_delay_ms == 70
    assert FakeAudioProcessor.instances[-1].delays == [50, 70]


def test_identity_aec_preserves_preprocessed_pcm_and_delay_contract() -> None:
    aec = IdentityAec(stream_delay_ms=50)

    assert aec.process_reverse_stream(RENDER) is None
    assert aec.process_stream(FRAME) == FRAME
    aec.set_stream_delay(70)
    assert aec.stream_delay_ms == 70


def test_identity_aec_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        IdentityAec(stream_delay_ms=-1)


class FakeAec:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes]] = []
        self.delays: list[int] = []

    def set_stream_delay(self, delay_ms: int) -> None:
        self.delays.append(delay_ms)

    def process_reverse_stream(self, pcm: bytes) -> None:
        self.calls.append(("reverse", pcm))

    def process_stream(self, pcm: bytes) -> bytes:
        self.calls.append(("capture", pcm))
        return b"clean" * 64


class FakeVad:
    def __init__(self, values: Iterable[bool]) -> None:
        self._values = iter(values)
        self.frames: list[bytes] = []
        self.reset_calls = 0

    def is_speech(self, pcm: bytes) -> bool:
        self.frames.append(pcm)
        return next(self._values)

    def reset(self) -> None:
        self.reset_calls += 1


def test_audio_frame_processor_feeds_render_before_capture_and_vad() -> None:
    aec = FakeAec()
    vad = FakeVad([True])
    processor = AudioFrameProcessor(aec, vad)

    result = processor.process(capture_pcm=FRAME, render_pcm=RENDER)

    assert aec.calls == [("reverse", RENDER), ("capture", FRAME)]
    assert vad.frames == [b"clean" * 64]
    assert result.cleaned_pcm == b"clean" * 64
    assert result.voice_detected is True
    assert result.candidate_voice_detected is None


def test_audio_frame_processor_carries_candidate_activity_from_the_same_vad_call():
    class ActivityVad(FakeVad):
        candidate_voice_detected: bool | None = None

        def is_speech(self, pcm: bytes) -> bool:
            formal = super().is_speech(pcm)
            self.candidate_voice_detected = False
            return formal

    vad = ActivityVad([True])
    processor = AudioFrameProcessor(FakeAec(), vad)

    result = processor.process(capture_pcm=FRAME, render_pcm=RENDER)

    assert result.voice_detected is True
    assert result.candidate_voice_detected is False
    assert vad.frames == [b"clean" * 64]


def test_audio_frame_processor_rejects_non_10ms_pcm() -> None:
    processor = AudioFrameProcessor(FakeAec(), FakeVad([False]))

    with pytest.raises(ValueError, match="320 bytes"):
        processor.process(capture_pcm=b"short", render_pcm=RENDER)


def test_audio_frame_processor_resets_only_vad_state() -> None:
    aec = FakeAec()
    vad = FakeVad([False])
    processor = AudioFrameProcessor(aec, vad)

    processor.reset_vad()

    assert vad.reset_calls == 1
    assert aec.calls == []


def test_audio_frame_processor_updates_aec_stream_delay() -> None:
    aec = FakeAec()
    processor = AudioFrameProcessor(aec, FakeVad([False]))

    processor.set_stream_delay(73)

    assert aec.delays == [73]


def test_speech_turn_detector_requires_start_confirmation() -> None:
    detector = SpeechTurnDetector(start_frames=2, end_silence_frames=3)

    assert detector.accept(False) is SpeechEvent.NONE
    assert detector.accept(True) is SpeechEvent.NONE
    assert detector.accept(True) is SpeechEvent.START
    assert detector.accept(True) is SpeechEvent.NONE


def test_speech_turn_detector_uses_hangover_and_emits_one_end() -> None:
    detector = SpeechTurnDetector(start_frames=1, end_silence_frames=2)

    assert detector.accept(True) is SpeechEvent.START
    assert detector.accept(False) is SpeechEvent.NONE
    assert detector.accept(False) is SpeechEvent.END
    assert detector.accept(False) is SpeechEvent.NONE


def test_speech_turn_detector_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        SpeechTurnDetector(start_frames=0)


@pytest.mark.parametrize(
    ("silence_ms", "expected_frames"),
    [(10, 1), (200, 20), (350, 35), (351, 36)],
)
def test_silence_ms_maps_to_ceil_of_10ms_audio_frames(
    silence_ms: int, expected_frames: int
) -> None:
    assert silence_ms_to_frames(silence_ms) == expected_frames


@pytest.mark.parametrize(
    ("speech_ms", "expected_frames"),
    [(80, 18), (90, 19), (91, 20)],
)
def test_speech_pre_roll_includes_one_hundred_ms_lead_in(
    speech_ms: int, expected_frames: int
) -> None:
    assert speech_pre_roll_ms_to_frames(speech_ms) == expected_frames


def test_speech_pre_roll_includes_vad_window_when_requested() -> None:
    # Silero needs one complete 512-sample window (32 ms at 16 kHz) before
    # its speech decision can include the true onset.
    assert speech_pre_roll_ms_to_frames(80, window_ms=32) == 22
