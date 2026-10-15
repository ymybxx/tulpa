import time
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from fish_memory.kws.models import Detection, KwsModelConfig


class KwsUnavailableError(RuntimeError):
    """The local KWS engine cannot be initialized."""


class AudioFormatError(ValueError):
    """Audio bytes are not complete little-endian signed 16-bit samples."""


class KwsStream(Protocol):
    def accept_waveform(
        self, sample_rate: int, samples: NDArray[np.float32]
    ) -> None: ...

    def ready(self) -> bool: ...

    def decode(self) -> None: ...

    def result(self) -> str: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class KwsBackend(Protocol):
    def create_stream(self, keywords_path: Path) -> KwsStream: ...


class _SherpaStream(Protocol):
    def accept_waveform(
        self, sample_rate: int, samples: NDArray[np.float32]
    ) -> None: ...


class _SherpaSpotter(Protocol):
    def is_ready(self, s: _SherpaStream) -> bool: ...

    def decode_stream(self, s: _SherpaStream) -> None: ...

    def get_result(self, s: _SherpaStream) -> str: ...

    def reset_stream(self, s: _SherpaStream) -> None: ...


class WakeWordDetector:
    def __init__(
        self,
        backend: KwsBackend,
        *,
        keywords_path: Path,
        sample_rate: int = 16_000,
    ) -> None:
        self._backend = backend
        self._keywords_path = keywords_path
        self._sample_rate = sample_rate
        self._sample_count = 0
        self._closed = False
        self._stream = backend.create_stream(keywords_path)

    def accept_pcm(self, pcm_bytes: bytes) -> list[Detection]:
        if self._closed:
            raise KwsUnavailableError("KWS detector is closed")
        if len(pcm_bytes) % 2:
            raise AudioFormatError("PCM must contain complete 16-bit samples")
        if not pcm_bytes:
            return []

        # sherpa-onnx's native OnlineStream accepts a 1-D float32 ndarray (and
        # uses that form in its packaged KeywordSpotter examples). Convert the
        # complete buffer in native NumPy code instead of allocating one Python
        # bytes slice, int, and float per sample on the 100 Hz standby path.
        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
        samples *= np.float32(1 / 32_768)
        started_at = time.perf_counter()
        self._stream.accept_waveform(self._sample_rate, samples)
        self._sample_count += len(samples)
        detections: list[Detection] = []
        while self._stream.ready():
            self._stream.decode()
            keyword = self._stream.result().strip()
            if not keyword:
                continue
            detections.append(
                Detection(
                    keyword=keyword,
                    audio_offset_ms=round(
                        self._sample_count * 1000 / self._sample_rate
                    ),
                    processing_ms=round(
                        (time.perf_counter() - started_at) * 1000,
                        3,
                    ),
                )
            )
            self._stream.reset()
        return detections

    def reset(self) -> None:
        if self._closed:
            return
        self._stream.close()
        self._sample_count = 0
        self._stream = self._backend.create_stream(self._keywords_path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.close()


class _SherpaKwsStream:
    def __init__(self, spotter: _SherpaSpotter, stream: _SherpaStream) -> None:
        self._spotter = spotter
        self._stream = stream

    def _require_stream(self) -> _SherpaStream:
        if self._stream is None:
            raise KwsUnavailableError("KWS stream is closed")
        return self._stream

    def accept_waveform(self, sample_rate: int, samples: NDArray[np.float32]) -> None:
        self._require_stream().accept_waveform(sample_rate, samples)

    def ready(self) -> bool:
        return bool(self._spotter.is_ready(self._require_stream()))

    def decode(self) -> None:
        self._spotter.decode_stream(self._require_stream())

    def result(self) -> str:
        return str(self._spotter.get_result(self._require_stream()))

    def reset(self) -> None:
        self._spotter.reset_stream(self._require_stream())

    def close(self) -> None:
        self._stream = None


class SherpaKwsBackend:
    def __init__(self, config: KwsModelConfig, keywords_path: Path) -> None:
        try:
            sherpa_onnx = import_module("sherpa_onnx")
        except Exception as error:
            raise KwsUnavailableError(f"cannot import sherpa-onnx: {error}") from error

        paths = config.model_files()
        missing = [
            path
            for path in paths.values()
            if path.name != "en.phone" and not path.is_file()
        ]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise KwsUnavailableError(f"missing KWS model files: {names}")
        if not keywords_path.is_file():
            raise KwsUnavailableError(f"missing keyword file: {keywords_path}")

        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(paths["tokens"]),
                encoder=str(paths["encoder"]),
                decoder=str(paths["decoder"]),
                joiner=str(paths["joiner"]),
                keywords_file=str(keywords_path),
                num_threads=config.num_threads,
                sample_rate=config.sample_rate,
                feature_dim=config.feature_dim,
                keywords_score=config.score,
                keywords_threshold=config.threshold,
                provider="cpu",
            )
        except Exception as error:
            raise KwsUnavailableError(
                f"cannot initialize sherpa-onnx: {error}"
            ) from error

    def create_stream(self, keywords_path: Path) -> KwsStream:
        del keywords_path
        spotter = cast(_SherpaSpotter, self._spotter)
        stream = cast(_SherpaStream, self._spotter.create_stream())
        return _SherpaKwsStream(spotter, stream)
