from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from fish_memory.kws.detector import AudioFormatError, WakeWordDetector


class FakeStream:
    def __init__(self, results: Sequence[str] = ()) -> None:
        self.results = list(results)
        self.accepted: list[tuple[int, np.ndarray]] = []
        self.reset_count = 0
        self.closed = False

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.accepted.append((sample_rate, samples))

    def ready(self) -> bool:
        return bool(self.results)

    def decode(self) -> None:
        return None

    def result(self) -> str:
        return self.results.pop(0)

    def reset(self) -> None:
        self.reset_count += 1

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    def __init__(self, *streams: FakeStream) -> None:
        self.streams = list(streams)
        self.created: list[FakeStream] = []

    def create_stream(self, keywords_path: Path) -> FakeStream:
        del keywords_path
        stream = self.streams.pop(0) if self.streams else FakeStream()
        self.created.append(stream)
        return stream


def test_detector_rejects_odd_pcm_bytes() -> None:
    backend = FakeBackend(FakeStream())
    detector = WakeWordDetector(backend, keywords_path=Path("keywords.txt"))

    with pytest.raises(AudioFormatError, match="16-bit"):
        detector.accept_pcm(b"\x00")


def test_detector_converts_pcm_and_reports_detection_offset() -> None:
    stream = FakeStream(["小鱼"])
    backend = FakeBackend(stream)
    detector = WakeWordDetector(backend, keywords_path=Path("keywords.txt"))

    pcm_samples = np.zeros(160, dtype="<i2")
    pcm_samples[:3] = [1000, -32_768, 32_767]
    detections = detector.accept_pcm(pcm_samples.tobytes())

    assert detections[0].keyword == "小鱼"
    assert detections[0].audio_offset_ms == 10
    assert stream.accepted[0][0] == 16_000
    assert stream.accepted[0][1].dtype == np.float32
    np.testing.assert_allclose(
        stream.accepted[0][1][:3],
        np.array([1000 / 32_768, -1.0, 32_767 / 32_768], dtype=np.float32),
    )
    assert np.count_nonzero(stream.accepted[0][1][3:]) == 0
    assert stream.reset_count == 1


def test_reset_creates_a_new_private_stream() -> None:
    first = FakeStream()
    second = FakeStream()
    backend = FakeBackend(first, second)
    detector = WakeWordDetector(backend, keywords_path=Path("keywords.txt"))

    detector.reset()
    detector.accept_pcm(b"\x00\x00" * 160)

    assert first.closed is True
    assert backend.created == [first, second]
    assert second.accepted


def test_close_is_idempotent() -> None:
    stream = FakeStream()
    backend = FakeBackend(stream)
    detector = WakeWordDetector(backend, keywords_path=Path("keywords.txt"))

    detector.close()
    detector.close()

    assert stream.closed is True
