"""Semantic end-of-turn detection for local Companion audio."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from fish_memory.model_catalog import (
    SMART_TURN_FILES,
    SMART_TURN_REPO,
    SMART_TURN_REVISION,
)
from fish_memory.model_sources import (
    ModelDownloadError,
    ModelStatusReporter,
    hf_endpoints,
    install_hub_snapshot,
)

logger = logging.getLogger(__name__)

MODEL_REPO_ID = SMART_TURN_REPO
MODEL_FILENAME = "smart-turn-v3.2-cpu.onnx"
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 8
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_MEL_BINS = 80


class _OnnxInput(Protocol):
    name: str


class _OnnxSession(Protocol):
    def get_inputs(self) -> list[_OnnxInput]: ...

    def run(
        self,
        output_names: object,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]: ...


class _FeatureExtractor(Protocol):
    def __call__(self, audio: np.ndarray, **kwargs: object) -> Any: ...


@dataclass(frozen=True, slots=True)
class SmartTurnResult:
    complete: bool
    probability: float
    inference_ms: float


@dataclass(frozen=True, slots=True)
class _FeatureBatch:
    input_features: np.ndarray


class WhisperLogMelFeatureExtractor:
    """Small NumPy implementation of the Whisper 80-bin input transform."""

    def __init__(self) -> None:
        self._window = np.hanning(WHISPER_N_FFT + 1)[:-1].astype(np.float32)
        self._mel_filters = _whisper_mel_filter_bank().astype(np.float32)

    def __call__(
        self,
        audio: np.ndarray,
        *,
        sampling_rate: int,
        return_tensors: str,
        padding: str,
        max_length: int,
        truncation: bool,
        do_normalize: bool,
    ) -> _FeatureBatch:
        if sampling_rate != SAMPLE_RATE:
            raise ValueError(f"Smart Turn requires {SAMPLE_RATE} Hz audio")
        if return_tensors != "np" or padding != "max_length":
            raise ValueError(
                "Smart Turn feature extraction requires fixed NumPy output"
            )
        if max_length != MAX_AUDIO_SAMPLES or not truncation or not do_normalize:
            raise ValueError(
                "Smart Turn feature extraction requires its fixed 8s window"
            )
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.shape != (MAX_AUDIO_SAMPLES,):
            raise ValueError("Smart Turn feature input must be one padded 8s window")

        waveform = (waveform - waveform.mean()) / np.sqrt(
            waveform.var() + np.float32(1e-7)
        )
        padded = np.pad(
            waveform,
            (WHISPER_N_FFT // 2, WHISPER_N_FFT // 2),
            mode="reflect",
        )
        frames = np.lib.stride_tricks.sliding_window_view(padded, WHISPER_N_FFT)[
            ::WHISPER_HOP_LENGTH
        ]
        spectrum = np.fft.rfft(frames * self._window, axis=1)
        magnitudes = (np.abs(spectrum) ** 2).astype(np.float32)[:-1].T
        mel_spectrum = self._mel_filters.T @ magnitudes
        log_spectrum = np.log10(np.maximum(mel_spectrum, np.float32(1e-10)))
        log_spectrum = np.maximum(
            log_spectrum,
            log_spectrum.max() - np.float32(8.0),
        )
        features = ((log_spectrum + np.float32(4.0)) / np.float32(4.0)).astype(
            np.float32, copy=False
        )
        return _FeatureBatch(input_features=features[np.newaxis, ...])


def _whisper_mel_filter_bank() -> np.ndarray:
    """Build the Slaney mel filters used by WhisperFeatureExtractor."""
    mel_min = _hertz_to_slaney_mel(np.asarray([0.0]))[0]
    mel_max = _hertz_to_slaney_mel(np.asarray([SAMPLE_RATE / 2]))[0]
    mel_frequencies = np.linspace(mel_min, mel_max, WHISPER_MEL_BINS + 2)
    filter_frequencies = _slaney_mel_to_hertz(mel_frequencies)
    fft_frequencies = np.linspace(
        0.0,
        SAMPLE_RATE // 2,
        1 + WHISPER_N_FFT // 2,
    )
    filter_widths = np.diff(filter_frequencies)
    slopes = filter_frequencies[np.newaxis, :] - fft_frequencies[:, np.newaxis]
    down_slopes = -slopes[:, :-2] / filter_widths[:-1]
    up_slopes = slopes[:, 2:] / filter_widths[1:]
    filters = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    filters *= (2.0 / (filter_frequencies[2:] - filter_frequencies[:-2]))[np.newaxis, :]
    return filters


def _hertz_to_slaney_mel(frequencies: np.ndarray) -> np.ndarray:
    mels = 3.0 * frequencies / 200.0
    logarithmic = frequencies >= 1000.0
    mels[logarithmic] = 15.0 + np.log(frequencies[logarithmic] / 1000.0) * (
        27.0 / np.log(6.4)
    )
    return mels


def _slaney_mel_to_hertz(mels: np.ndarray) -> np.ndarray:
    frequencies = 200.0 * mels / 3.0
    logarithmic = mels >= 15.0
    frequencies[logarithmic] = 1000.0 * np.exp(
        (np.log(6.4) / 27.0) * (mels[logarithmic] - 15.0)
    )
    return frequencies


def prepare_pcm16(pcm: bytes) -> np.ndarray:
    """Convert PCM16LE to the model's left-padded eight-second window."""
    if len(pcm) % 2:
        raise ValueError("PCM must contain complete 16-bit samples")
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768
    if audio.size > MAX_AUDIO_SAMPLES:
        return audio[-MAX_AUDIO_SAMPLES:]
    if audio.size < MAX_AUDIO_SAMPLES:
        audio = np.pad(audio, (MAX_AUDIO_SAMPLES - audio.size, 0))
    return audio.astype(np.float32, copy=False)


class SmartTurnAnalyzer:
    """Run Smart Turn v3.2 after Silero reports a speech boundary."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        threshold: float = 0.5,
        num_threads: int = 1,
        session: _OnnxSession | None = None,
        feature_extractor: _FeatureExtractor | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Smart Turn threshold must be between 0 and 1")
        if num_threads < 1:
            raise ValueError("Smart Turn num_threads must be positive")
        if (session is None) != (feature_extractor is None):
            raise ValueError("session and feature_extractor must be provided together")

        self.threshold = threshold
        if session is None:
            session, feature_extractor, resolved_path = _load_runtime(
                model_path,
                num_threads=num_threads,
            )
            self.model_path: Path | None = resolved_path
        else:
            self.model_path = Path(model_path).expanduser() if model_path else None
        self._session = session
        self._feature_extractor = cast(_FeatureExtractor, feature_extractor)
        self._input_name = session.get_inputs()[0].name
        self._inference_lock = threading.Lock()

    def warm_up(self) -> None:
        self.predict_pcm16(b"\0\0" * SAMPLE_RATE)

    def predict_pcm16(self, pcm: bytes) -> SmartTurnResult:
        started = time.perf_counter()
        audio = prepare_pcm16(pcm)
        features = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=MAX_AUDIO_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        input_features = np.asarray(features.input_features, dtype=np.float32)
        with self._inference_lock:
            outputs = self._session.run(None, {self._input_name: input_features})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not np.isfinite(probability):
            raise RuntimeError("Smart Turn returned a non-finite probability")
        return SmartTurnResult(
            complete=probability > self.threshold,
            probability=probability,
            inference_ms=(time.perf_counter() - started) * 1000,
        )


def _load_runtime(
    model_path: str | Path | None,
    *,
    num_threads: int,
) -> tuple[_OnnxSession, _FeatureExtractor, Path]:
    import onnxruntime as ort

    resolved_path = _resolve_model_path(model_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Smart Turn model not found: {resolved_path}")

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = num_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(resolved_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    extractor = WhisperLogMelFeatureExtractor()
    return (
        cast(_OnnxSession, session),
        cast(_FeatureExtractor, extractor),
        resolved_path,
    )


def _resolve_model_path(model_path: str | Path | None) -> Path:
    if model_path is not None:
        return Path(model_path).expanduser()

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        # Avoid an HTTP metadata request on every desktop launch when the pinned
        # model is already in the local Hub cache.
        resolved = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            revision=SMART_TURN_REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        reporter = ModelStatusReporter("smart-turn", "对话节奏")
        reporter.preparing()
        try:
            snapshot = install_hub_snapshot(
                MODEL_REPO_ID,
                SMART_TURN_REVISION,
                SMART_TURN_FILES,
                progress=reporter.downloading,
            )
            reporter.ready()
            return snapshot / MODEL_FILENAME
        except (ModelDownloadError, OSError) as error:
            logger.warning("从魔搭准备轮次判断模型失败，改用 Hugging Face: %s", error)
        failure: Exception | None = None
        for endpoint in hf_endpoints():
            try:
                resolved = hf_hub_download(
                    repo_id=MODEL_REPO_ID,
                    filename=MODEL_FILENAME,
                    revision=SMART_TURN_REVISION,
                    endpoint=endpoint,
                )
                break
            except Exception as error:
                failure = error
        else:
            reporter.failed(str(failure))
            assert failure is not None
            raise failure
        reporter.ready()
    return Path(resolved)
