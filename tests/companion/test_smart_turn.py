import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fish_memory.companion.smart_turn import WhisperLogMelFeatureExtractor


def _smart_turn_module():
    return import_module("fish_memory.companion.smart_turn")


def test_prepare_pcm16_left_pads_short_audio() -> None:
    smart_turn = _smart_turn_module()
    pcm = np.array([16_384, -32_768], dtype="<i2").tobytes()

    audio = smart_turn.prepare_pcm16(pcm)

    assert audio.shape == (smart_turn.MAX_AUDIO_SAMPLES,)
    assert np.count_nonzero(audio[:-2]) == 0
    assert audio[-2:].tolist() == pytest.approx([0.5, -1.0])


def test_prepare_pcm16_keeps_the_last_eight_seconds() -> None:
    smart_turn = _smart_turn_module()
    samples = np.arange(
        smart_turn.MAX_AUDIO_SAMPLES + 7,
        dtype=np.int32,
    ).astype("<i2")

    audio = smart_turn.prepare_pcm16(samples.tobytes())

    expected = samples[-smart_turn.MAX_AUDIO_SAMPLES :].astype(np.float32) / 32_768
    np.testing.assert_array_equal(audio, expected)


class _FakeSession:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.inputs: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_features")]

    def run(
        self,
        _outputs: object,
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.inputs.append(inputs)
        return [np.array([[self.probability]], dtype=np.float32)]


class _FakeFeatureExtractor:
    def __init__(self) -> None:
        self.audio: np.ndarray | None = None

    def __call__(self, audio: np.ndarray, **_kwargs: object) -> SimpleNamespace:
        self.audio = audio
        return SimpleNamespace(input_features=np.zeros((1, 80, 800), dtype=np.float32))


def test_smart_turn_uses_a_strict_completion_threshold() -> None:
    smart_turn = _smart_turn_module()
    session = _FakeSession(0.5)
    extractor = _FakeFeatureExtractor()
    analyzer = smart_turn.SmartTurnAnalyzer(
        threshold=0.5,
        session=session,
        feature_extractor=extractor,
    )

    at_threshold = analyzer.predict_pcm16(b"\0\0" * 160)
    session.probability = 0.5001
    above_threshold = analyzer.predict_pcm16(b"\0\0" * 160)

    assert at_threshold.complete is False
    assert above_threshold.complete is True
    assert above_threshold.probability == pytest.approx(0.5001)
    assert extractor.audio is not None
    assert extractor.audio.shape == (smart_turn.MAX_AUDIO_SAMPLES,)
    assert list(session.inputs[-1]) == ["input_features"]


def test_prepare_pcm16_rejects_partial_samples() -> None:
    smart_turn = _smart_turn_module()

    with pytest.raises(ValueError, match="16-bit"):
        smart_turn.prepare_pcm16(b"\0")


def test_lightweight_whisper_features_match_transformers_reference() -> None:
    from transformers import WhisperFeatureExtractor

    smart_turn = _smart_turn_module()
    positions = np.arange(smart_turn.MAX_AUDIO_SAMPLES, dtype=np.float32)
    audio = np.float32(0.17) * np.sin(
        positions * np.float32(2 * np.pi * 440 / smart_turn.SAMPLE_RATE)
    ) + np.random.default_rng(17).normal(
        0.0, 0.01, smart_turn.MAX_AUDIO_SAMPLES
    ).astype(np.float32)
    options = {
        "sampling_rate": smart_turn.SAMPLE_RATE,
        "return_tensors": "np",
        "padding": "max_length",
        "max_length": smart_turn.MAX_AUDIO_SAMPLES,
        "truncation": True,
        "do_normalize": True,
    }

    actual = WhisperLogMelFeatureExtractor()(audio, **options).input_features
    expected = WhisperFeatureExtractor(chunk_length=smart_turn.MAX_AUDIO_SECONDS)(
        audio,
        **options,
    ).input_features

    assert actual.shape == (1, 80, 800)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_smart_turn_runtime_does_not_require_transformers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smart_turn = _smart_turn_module()
    model_path = tmp_path / "smart-turn.onnx"
    model_path.write_bytes(b"model")

    class Options:
        execution_mode: object
        inter_op_num_threads: int
        intra_op_num_threads: int
        graph_optimization_level: object

    fake_ort = SimpleNamespace(
        SessionOptions=Options,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=object()),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=object()),
        InferenceSession=lambda *_args, **_kwargs: _FakeSession(0.5),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **_kwargs: model_path),
    )
    monkeypatch.setitem(sys.modules, "transformers", None)

    _session, extractor, resolved = smart_turn._load_runtime(
        model_path,
        num_threads=1,
    )

    assert isinstance(extractor, WhisperLogMelFeatureExtractor)
    assert resolved == model_path


def test_smart_turn_resolves_cached_model_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smart_turn = _smart_turn_module()
    model_path = tmp_path / "smart-turn.onnx"
    model_path.write_bytes(b"model")
    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> Path:
        calls.append(kwargs)
        if kwargs.get("local_files_only") is not True:
            raise AssertionError("cached model resolution must not use the network")
        return model_path

    class LocalEntryNotFoundError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=download),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.errors",
        SimpleNamespace(LocalEntryNotFoundError=LocalEntryNotFoundError),
    )

    resolved = smart_turn._resolve_model_path(None)

    assert resolved == model_path
    assert calls == [
        {
            "repo_id": smart_turn.MODEL_REPO_ID,
            "filename": smart_turn.MODEL_FILENAME,
            "revision": smart_turn.SMART_TURN_REVISION,
            "local_files_only": True,
        }
    ]


def test_smart_turn_downloads_only_after_cache_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smart_turn = _smart_turn_module()
    model_path = tmp_path / "smart-turn.onnx"
    model_path.write_bytes(b"model")
    calls: list[dict[str, object]] = []

    class LocalEntryNotFoundError(Exception):
        pass

    def download(**kwargs: object) -> Path:
        calls.append(kwargs)
        if kwargs.get("local_files_only") is True:
            raise LocalEntryNotFoundError
        return model_path

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=download),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.errors",
        SimpleNamespace(LocalEntryNotFoundError=LocalEntryNotFoundError),
    )

    def modelscope_down(*_args: object, **_kwargs: object) -> Path:
        raise smart_turn.ModelDownloadError("unreachable")

    monkeypatch.setattr(smart_turn, "install_hub_snapshot", modelscope_down)
    resolved = smart_turn._resolve_model_path(None)

    assert resolved == model_path
    assert [call.get("local_files_only") for call in calls] == [True, None]
    assert {call["revision"] for call in calls} == {smart_turn.SMART_TURN_REVISION}

    calls.clear()
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(
        smart_turn, "install_hub_snapshot", lambda *_args, **_kwargs: snapshot
    )
    assert smart_turn._resolve_model_path(None) == snapshot / smart_turn.MODEL_FILENAME
    assert [call.get("local_files_only") for call in calls] == [True]
