"""Synchronous echo-cancellation and voice-activity processing for companion audio."""

import math
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Literal, Protocol

SAMPLE_RATE = 16_000
FRAME_DURATION_MS = 10
FRAME_BYTES = SAMPLE_RATE * FRAME_DURATION_MS // 1000 * 2


def silence_ms_to_frames(silence_ms: int) -> int:
    """Convert a silence duration into complete 10 ms frame windows."""
    if silence_ms < 1:
        raise ValueError("silence duration must be positive")
    return max(1, math.ceil(silence_ms / FRAME_DURATION_MS))


def speech_pre_roll_ms_to_frames(
    speech_ms: int,
    extra_ms: int = 100,
    window_ms: int = 0,
) -> int:
    """Keep speech-decision time, the lead-in, and an optional model window."""
    if speech_ms < 0 or extra_ms < 0 or window_ms < 0:
        raise ValueError("speech, extra, and window durations cannot be negative")
    return max(
        1,
        math.ceil((speech_ms + extra_ms + window_ms) / FRAME_DURATION_MS),
    )


class AecBackend(Protocol):
    def process_reverse_stream(self, pcm: bytes) -> object: ...

    def process_stream(self, pcm: bytes) -> bytes: ...


class VadBackend(Protocol):
    def is_speech(self, pcm: bytes) -> bool: ...


@dataclass(frozen=True, slots=True)
class ProcessedAudioFrame:
    cleaned_pcm: bytes
    voice_detected: bool
    # Optional same-inference activity before the VAD's START/END duration hold.
    candidate_voice_detected: bool | None = None


class AudioFrameProcessor:
    """Keeps the render reference and microphone frame on one 10 ms timeline."""

    def __init__(self, aec: AecBackend, vad: VadBackend) -> None:
        self._aec = aec
        self._vad = vad

    def process(
        self,
        *,
        capture_pcm: bytes,
        render_pcm: bytes,
    ) -> ProcessedAudioFrame:
        _validate_frame(capture_pcm)
        _validate_frame(render_pcm)
        # WebRTC AEC must receive the actual speaker frame before the matching
        # microphone frame is processed.
        self._aec.process_reverse_stream(render_pcm)
        cleaned_pcm = bytes(self._aec.process_stream(capture_pcm))
        _validate_frame(cleaned_pcm, name="cleaned PCM")
        voice_detected = self._vad.is_speech(cleaned_pcm)
        candidate_voice = getattr(self._vad, "candidate_voice_detected", None)
        return ProcessedAudioFrame(
            cleaned_pcm=cleaned_pcm,
            voice_detected=voice_detected,
            candidate_voice_detected=(
                candidate_voice if isinstance(candidate_voice, bool) else None
            ),
        )

    def reset_vad(self) -> None:
        """Clear streaming VAD state without disturbing the AEC filter."""
        reset = getattr(self._vad, "reset", None)
        if callable(reset):
            reset()

    def set_stream_delay(self, delay_ms: int) -> None:
        """Update the capture/render delay used by the echo canceller."""
        setter = getattr(self._aec, "set_stream_delay", None)
        if not callable(setter):
            raise RuntimeError("AEC backend does not support stream-delay updates")
        setter(delay_ms)


class IdentityAec:
    """Keep an already voice-processed capture stream on the existing timeline.

    A native audio backend can perform echo cancellation before it hands PCM to
    Python.  This adapter preserves ``AudioFrameProcessor`` and every downstream
    VAD/ASR/turn boundary while preventing a second WebRTC AEC pass.
    """

    def __init__(self, *, stream_delay_ms: int = 0) -> None:
        if stream_delay_ms < 0:
            raise ValueError("AEC stream delay cannot be negative")
        self._stream_delay_ms = stream_delay_ms

    @property
    def stream_delay_ms(self) -> int:
        return self._stream_delay_ms

    def set_stream_delay(self, delay_ms: int) -> None:
        if delay_ms < 0:
            raise ValueError("AEC stream delay cannot be negative")
        # CompanionAudioProcessor still uses this value for its existing
        # post-attenuation echo guard.  No signal processing happens here.
        self._stream_delay_ms = delay_ms

    def process_reverse_stream(self, pcm: bytes) -> None:
        del pcm

    def process_stream(self, pcm: bytes) -> bytes:
        return bytes(pcm)


class WebRtcAec:
    """Thin adapter around the native WebRTC Audio Processing Module."""

    def __init__(
        self,
        *,
        stream_delay_ms: int = 50,
        delay_mode: Literal["fixed", "auto"] = "fixed",
    ) -> None:
        if delay_mode not in {"fixed", "auto"}:
            raise ValueError("AEC delay mode must be 'fixed' or 'auto'")
        if stream_delay_ms < 0:
            raise ValueError("AEC stream delay cannot be negative")
        try:
            AudioProcessor = import_module("aec_audio_processing").AudioProcessor
        except Exception as error:  # pragma: no cover - optional native module
            raise RuntimeError(
                "WebRTC AEC requires the aec-audio-processing package"
            ) from error

        self._processor = AudioProcessor(
            enable_aec=True,
            enable_ns=True,
            enable_agc=False,
            enable_vad=False,
        )
        self._processor.set_stream_format(SAMPLE_RATE, 1)
        self._processor.set_reverse_stream_format(SAMPLE_RATE, 1)
        self._delay_mode: Literal["fixed", "auto"] = delay_mode
        self._stream_delay_ms = stream_delay_ms
        # Legacy WebRTC AEC requires this call even when the value will be
        # replaced from the active audio device in automatic mode.
        self._processor.set_stream_delay(stream_delay_ms)

    @property
    def delay_mode(self) -> Literal["fixed", "auto"]:
        return self._delay_mode

    @property
    def stream_delay_ms(self) -> int:
        return self._stream_delay_ms

    def set_stream_delay(self, delay_ms: int) -> None:
        if delay_ms < 0:
            raise ValueError("AEC stream delay cannot be negative")
        self._processor.set_stream_delay(delay_ms)
        self._stream_delay_ms = delay_ms

    def process_reverse_stream(self, pcm: bytes) -> object:
        return self._processor.process_reverse_stream(pcm)

    def process_stream(self, pcm: bytes) -> bytes:
        return bytes(self._processor.process_stream(pcm))


class WebRtcVad:
    """Adapter for the standalone WebRTC VAD implementation."""

    def __init__(self, *, mode: int = 2) -> None:
        try:
            webrtcvad = import_module("webrtcvad")
        except Exception as error:  # pragma: no cover - optional native module
            raise RuntimeError("WebRTC VAD requires the webrtcvad package") from error
        if mode not in range(4):
            raise ValueError("VAD mode must be between 0 and 3")
        self._vad = webrtcvad.Vad(mode)

    def is_speech(self, pcm: bytes) -> bool:
        _validate_frame(pcm)
        return bool(self._vad.is_speech(pcm, SAMPLE_RATE))


class SpeechEvent(StrEnum):
    NONE = "none"
    START = "start"
    END = "end"


class SpeechTurnDetector:
    """Turns frame-level VAD results into stable utterance boundaries."""

    def __init__(
        self,
        *,
        start_frames: int = 2,
        end_silence_frames: int = 20,
    ) -> None:
        if start_frames < 1 or end_silence_frames < 1:
            raise ValueError("speech boundary frame counts must be positive")
        self._start_frames = start_frames
        self._end_silence_frames = end_silence_frames
        self._voice_frames = 0
        self._silence_frames = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._voice_frames = 0
        self._silence_frames = 0
        self._active = False

    def accept(self, voice_detected: bool) -> SpeechEvent:
        if not self._active:
            self._silence_frames = 0
            if not voice_detected:
                self._voice_frames = 0
                return SpeechEvent.NONE
            self._voice_frames += 1
            if self._voice_frames < self._start_frames:
                return SpeechEvent.NONE
            self._active = True
            self._voice_frames = 0
            return SpeechEvent.START

        if voice_detected:
            self._silence_frames = 0
            return SpeechEvent.NONE
        self._silence_frames += 1
        if self._silence_frames < self._end_silence_frames:
            return SpeechEvent.NONE
        self._active = False
        self._silence_frames = 0
        return SpeechEvent.END


def _validate_frame(pcm: bytes, *, name: str = "PCM") -> None:
    if len(pcm) != FRAME_BYTES:
        raise ValueError(f"{name} must contain exactly {FRAME_BYTES} bytes (10 ms)")
