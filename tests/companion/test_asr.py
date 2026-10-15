import asyncio
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fish_memory.companion.asr as asr_module
from fish_memory.companion.asr import (
    AsrInferenceBusyError,
    AsrModelConfig,
    AsrUpdate,
    SenseVoiceDraftDecoder,
    SharedSenseVoiceDraftDecoder,
    pcm16le_to_float32,
)


def test_default_bilingual_asr_config_uses_sensevoice_int8_model_files() -> None:
    config = AsrModelConfig(Path("/tmp/asr"))

    files = config.model_files()

    assert config.model == "sensevoice"
    assert config.model_name == (
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    )
    assert files["tokens"].name == "tokens.txt"
    assert files["model"].name == "model.int8.onnx"


def test_pcm16le_conversion_preserves_signed_sample_scale() -> None:
    samples = pcm16le_to_float32(b"\x00\x00\xff\x7f\x00\x80")

    assert samples.dtype == np.float32
    assert samples.flags.c_contiguous
    np.testing.assert_array_equal(
        samples,
        np.asarray([0.0, 32767 / 32768, -1.0], dtype=np.float32),
    )


def test_sensevoice_buffers_the_complete_utterance_before_decoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    class FakeResult:
        text = "What's your name."

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: tuple[int, list[float]] | None = None
            self.result = FakeResult()

        def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
            self.accepted = (sample_rate, samples)

    class FakeRecognizer:
        kwargs: dict[str, object] = {}

        @classmethod
        def from_sense_voice(cls, **kwargs: object) -> "FakeRecognizer":
            cls.kwargs = kwargs
            return cls()

        def create_stream(self) -> FakeStream:
            self.stream = FakeStream()
            return self.stream

        def decode_stream(self, stream: FakeStream) -> None:
            assert stream.accepted is not None

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )

    asr = asr_module.StreamingAsr(config)
    assert asr.accept_pcm(b"\x00\x00\xff\x7f") is None
    assert asr.accept_pcm(b"\x00\x80") is None
    assert asr.buffered_pcm_bytes() == 6

    assert asr.finish_utterance() == "What's your name."
    assert asr.buffered_pcm_bytes() == 0
    assert asr._recognizer.stream.accepted is not None
    sample_rate, samples = asr._recognizer.stream.accepted
    assert sample_rate == 16_000
    np.testing.assert_array_equal(
        samples,
        np.asarray([0.0, 32767 / 32768, -1.0], dtype=np.float32),
    )
    assert FakeRecognizer.kwargs["language"] == "zh"
    assert FakeRecognizer.kwargs["use_itn"] is True
    assert asr.finish_utterance() == ""


def test_sensevoice_preserves_event_from_the_same_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    class FakeStream:
        def __init__(self) -> None:
            self.result = SimpleNamespace(text="呵呵。", event="<|Cough|>")

        def accept_waveform(self, _sample_rate: int, _samples: list[float]) -> None:
            return

    class FakeRecognizer:
        def __init__(self) -> None:
            self.decode_calls = 0

        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, _stream: FakeStream) -> None:
            self.decode_calls += 1

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    asr = asr_module.StreamingAsr(config)
    asr.accept_pcm(b"\0\0")

    update = asr.finish_utterance_update()

    assert update.text == "呵呵。"
    assert update.is_final is True
    assert update.event == "<|Cough|>"
    assert asr._recognizer.decode_calls == 1


def test_sensevoice_draft_decoder_uses_its_own_recognizer_and_decode_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    draft_started = threading.Event()
    release_draft = threading.Event()

    class FakeStream:
        def __init__(self, recognizer_id: int) -> None:
            self.recognizer_id = recognizer_id
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, _sample_rate: int, _samples: list[float]) -> None:
            return

    class FakeRecognizer:
        instances: list["FakeRecognizer"] = []

        def __init__(self) -> None:
            self.recognizer_id = len(self.instances)
            self.instances.append(self)

        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            return FakeStream(self.recognizer_id)

        def decode_stream(self, stream: FakeStream) -> None:
            if stream.recognizer_id == 1:
                draft_started.set()
                release_draft.wait(timeout=1)
                stream.result.text = "草稿"
            else:
                stream.result.text = "问题"

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    authoritative = asr_module.StreamingAsr(config)
    draft = SenseVoiceDraftDecoder(config)
    pcm = b"\0" * 320
    authoritative.accept_pcm(pcm)

    async def run() -> None:
        draft_task = asyncio.create_task(asyncio.to_thread(draft.recognize, pcm))
        await asyncio.to_thread(draft_started.wait, 1)
        final = await asyncio.wait_for(
            asyncio.to_thread(authoritative.finish_utterance),
            timeout=0.1,
        )
        assert final == "问题"
        release_draft.set()
        assert await draft_task == "草稿"

    asyncio.run(run())

    assert len(FakeRecognizer.instances) == 2
    assert authoritative._recognizer is not draft._recognizer
    assert authoritative._decode_lock is not draft._decode_lock


def test_shared_sensevoice_draft_decoder_reuses_authoritative_recognizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: np.ndarray | None = None
            self.result = SimpleNamespace(text="", event="")

        def accept_waveform(self, _sample_rate: int, samples: np.ndarray) -> None:
            self.accepted = samples

    class FakeRecognizer:
        instances = 0

        def __init__(self) -> None:
            type(self).instances += 1

        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            assert stream.accepted is not None
            stream.result.text = f"samples-{stream.accepted.size}"

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    authoritative = asr_module.StreamingAsr(config)
    draft = SharedSenseVoiceDraftDecoder(authoritative)
    authoritative.accept_pcm(b"\x01\x00\x02\x00")

    assert draft.recognize(b"\x03\x00") == "samples-1"
    assert authoritative.buffered_pcm_bytes() == 4
    assert authoritative.finish_utterance() == "samples-2"
    assert FakeRecognizer.instances == 1


def test_shared_sensevoice_prioritizes_waiting_authoritative_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    draft_started = threading.Event()
    release_draft = threading.Event()
    final_started = threading.Event()
    release_final = threading.Event()

    class FakeStream:
        def __init__(self) -> None:
            self.marker = 0
            self.result = SimpleNamespace(text="", event="")

        def accept_waveform(self, _sample_rate: int, samples: np.ndarray) -> None:
            self.marker = round(float(samples[0]) * 32_768)

    class FakeRecognizer:
        instances = 0

        def __init__(self) -> None:
            type(self).instances += 1

        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            if stream.marker == 1:
                draft_started.set()
                assert release_draft.wait(timeout=1)
                stream.result.text = "draft"
                return
            if stream.marker == 2:
                final_started.set()
                assert release_final.wait(timeout=1)
                stream.result.text = "final"

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    authoritative = asr_module.StreamingAsr(config)
    draft = SharedSenseVoiceDraftDecoder(authoritative)
    authoritative.accept_pcm(b"\x02\x00")

    async def run() -> None:
        draft_task = asyncio.create_task(
            asyncio.to_thread(draft.recognize, b"\x01\x00")
        )
        assert await asyncio.to_thread(draft_started.wait, 1)
        final_task = asyncio.create_task(
            asyncio.to_thread(authoritative.finish_utterance)
        )
        async with asyncio.timeout(1):
            while True:
                with authoritative._decode_condition:
                    if authoritative._authoritative_decode_waiters == 1:
                        break
                # Yield long enough for the worker thread to acquire the model
                # condition, rather than spending 100 turns within one tick.
                await asyncio.sleep(0.001)

        with pytest.raises(AsrInferenceBusyError, match="busy"):
            draft.recognize(b"\x03\x00")
        release_draft.set()
        assert await asyncio.to_thread(final_started.wait, 1)
        assert authoritative.try_recognize_snapshot_update(b"\x03\x00") is None
        release_final.set()

        assert await draft_task == "draft"
        assert await final_task == "final"

    asyncio.run(run())

    assert FakeRecognizer.instances == 1


def test_reset_during_shared_snapshot_decode_only_clears_utterance_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    shared_started = threading.Event()
    release_shared = threading.Event()

    class FakeStream:
        result = SimpleNamespace(text="environment", event="Speech")

        def accept_waveform(self, _sample_rate: int, _samples: np.ndarray) -> None:
            return

    class FakeRecognizer:
        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, _stream: FakeStream) -> None:
            shared_started.set()
            assert release_shared.wait(timeout=1)

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    asr = asr_module.StreamingAsr(config)
    asr.accept_pcm(b"\x01\x00")

    async def run() -> None:
        shared = asyncio.create_task(
            asyncio.to_thread(asr.try_recognize_snapshot_update, b"\x02\x00")
        )
        assert await asyncio.to_thread(shared_started.wait, 1)
        await asyncio.wait_for(asyncio.to_thread(asr.reset), timeout=0.1)
        assert asr.buffered_pcm_bytes() == 0
        asr.accept_pcm(b"\x03\x00")
        release_shared.set()
        result = await shared
        assert result == AsrUpdate("environment", True, "Speech")
        assert asr.snapshot_utterance() == b"\x03\x00"

    try:
        asyncio.run(run())
    finally:
        release_shared.set()


def test_sensevoice_draft_decoder_has_snapshot_only_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensevoice = AsrModelConfig(tmp_path / "sensevoice")
    for path in sensevoice.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    class FakeRecognizer:
        kwargs: dict[str, object] = {}

        @classmethod
        def from_sense_voice(cls, **kwargs: object) -> "FakeRecognizer":
            cls.kwargs = kwargs
            return cls()

        def create_stream(self) -> object:
            return SimpleNamespace(
                accept_waveform=lambda _rate, _samples: None,
                result=SimpleNamespace(text="候选", event="Cough"),
            )

        def decode_stream(self, _stream: object) -> None:
            return

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    decoder = SenseVoiceDraftDecoder(sensevoice)

    assert decoder.recognize(b"\0\0") == "候选"
    assert decoder.recognize_update(b"\0\0") == AsrUpdate(
        text="候选", is_final=True, event="Cough"
    )
    assert FakeRecognizer.kwargs == {
        "tokens": str(sensevoice.model_files()["tokens"]),
        "model": str(sensevoice.model_files()["model"]),
        "num_threads": 2,
        "sample_rate": 16_000,
        "feature_dim": 80,
        "decoding_method": "greedy_search",
        "provider": "cpu",
        "language": "zh",
        "use_itn": True,
    }
    assert not hasattr(decoder, "accept_pcm")
    assert not hasattr(decoder, "snapshot_utterance")
    assert not hasattr(decoder, "consume_snapshot")
    assert not hasattr(decoder, "reset")
    with pytest.raises(TypeError, match="immutable bytes"):
        decoder.recognize(bytearray(b"\0\0"))  # type: ignore[arg-type]


def test_sensevoice_snapshot_recognizes_without_clearing_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: tuple[int, list[float]] | None = None
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
            self.accepted = (sample_rate, samples)

    class FakeRecognizer:
        def __init__(self) -> None:
            self.streams: list[FakeStream] = []

        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            stream = FakeStream()
            self.streams.append(stream)
            return stream

        def decode_stream(self, stream: FakeStream) -> None:
            assert stream.accepted is not None
            stream.result.text = f"samples-{len(stream.accepted[1])}"

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    asr = asr_module.StreamingAsr(config)
    first = b"\x01\x00\x02\x00"
    second = b"\x03\x00"
    asr.accept_pcm(first)

    snapshot = asr.snapshot_utterance()
    snapshot_text = asr.recognize_snapshot(snapshot)
    asr.accept_pcm(second)
    final_text = asr.finish_utterance()

    assert snapshot == first
    assert snapshot_text == "samples-2"
    assert final_text == "samples-3"
    assert len(asr._recognizer.streams) == 2


def test_sensevoice_consumes_only_the_committed_snapshot_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    class FakeStream:
        def __init__(self) -> None:
            self.accepted: tuple[int, list[float]] | None = None
            self.result = SimpleNamespace(text="")

        def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
            self.accepted = (sample_rate, samples)

    class FakeRecognizer:
        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

        def create_stream(self) -> FakeStream:
            return FakeStream()

        def decode_stream(self, stream: FakeStream) -> None:
            assert stream.accepted is not None
            stream.result.text = f"samples-{len(stream.accepted[1])}"

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )
    asr = asr_module.StreamingAsr(config)
    first = b"\x01\x00\x02\x00"
    continued = b"\x03\x00"
    asr.accept_pcm(first)
    snapshot = asr.snapshot_utterance()
    asr.accept_pcm(continued)

    assert asr.consume_snapshot(snapshot) is True
    assert asr.snapshot_utterance() == continued
    assert asr.finish_utterance() == "samples-1"


def test_sensevoice_reset_discards_buffered_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")

    class FakeRecognizer:
        @classmethod
        def from_sense_voice(cls, **_: object) -> "FakeRecognizer":
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "sherpa_onnx",
        SimpleNamespace(OfflineRecognizer=FakeRecognizer),
    )

    asr = asr_module.StreamingAsr(config)
    asr.accept_pcm(b"\x01\x00")
    asr.reset()

    assert asr.finish_utterance() == ""


@pytest.mark.asyncio
async def test_asr_status_identifies_sensevoice(tmp_path: Path) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    config.model_dir.mkdir(parents=True)
    for path in config.model_files().values():
        path.write_text("model\n", encoding="utf-8")

    status = asr_module.AsrService(config).status()

    assert status["model"] == "sensevoice"
    assert status["modelName"] == (
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    )


@pytest.mark.asyncio
async def test_download_logs_lifecycle_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = AsrModelConfig(tmp_path / "asr")

    class FakeDownloader:
        def __init__(self, _config, progress_callback=None, attempts=3) -> None:
            del attempts
            self._progress_callback = progress_callback

        def ensure(self) -> None:
            self._progress_callback(128, 256)
            config.model_dir.mkdir(parents=True, exist_ok=True)
            for path in config.model_files().values():
                path.write_text("model\n", encoding="utf-8")

    monkeypatch.setattr(asr_module, "AsrModelDownloader", FakeDownloader)
    service = asr_module.AsrService(config)

    with caplog.at_level(logging.INFO, logger="fish_memory.companion.asr"):
        await service.start()
        for _ in range(20):
            if service.status()["available"]:
                break
            await asyncio.sleep(0.01)

    messages = [record.getMessage() for record in caplog.records]
    assert any("开始下载 ASR 模型" in message for message in messages)
    assert any("ASR 模型下载进度" in message for message in messages)
    assert any("ASR 模型下载完成" in message for message in messages)


@pytest.mark.parametrize("decoder_kind", ["draft", "stream", "shared"])
def test_sensevoice_token_times_are_preserved_from_the_single_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decoder_kind: str
) -> None:
    config = AsrModelConfig(tmp_path / "asr")
    for path in config.model_files().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model\n", encoding="utf-8")
    result = SimpleNamespace(
        text="今天想吃什么？",
        event="Speech",
        tokens=["今", "天", "想", "吃", "什", "么", "？"],
        timestamps=[0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.72],
    )
    calls: list[object] = []

    class Recognizer:
        @classmethod
        def from_sense_voice(cls, **_: object):
            return cls()

        def create_stream(self):
            return SimpleNamespace(
                result=result, accept_waveform=lambda _rate, _samples: None
            )

        def decode_stream(self, stream):
            calls.append(stream)

    monkeypatch.setitem(
        sys.modules, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=Recognizer)
    )
    if decoder_kind == "draft":
        decoder = SenseVoiceDraftDecoder(config)
    else:
        source = asr_module.StreamingAsr(config)
        decoder = (
            SharedSenseVoiceDraftDecoder(source) if decoder_kind == "shared" else source
        )
    pcm = b"\0\0" * 16_000
    update = (
        decoder.recognize_snapshot_update(pcm)
        if isinstance(decoder, asr_module.StreamingAsr)
        else decoder.recognize_update(pcm)
    )
    assert update == AsrUpdate(
        "今天想吃什么？",
        True,
        "Speech",
        ("今", "天", "想", "吃", "什", "么", "？"),
        (0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.72),
    )
    assert len(calls) == 1
    result.tokens.clear()
    result.timestamps.clear()
    assert len(update.tokens) == len(update.timestamps) == 7


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"tokens": ["好"]},
        {"timestamps": [0.1]},
        {"tokens": "好", "timestamps": [0.1]},
        {"tokens": ["好"], "timestamps": "0.1"},
        {"tokens": ["好"], "timestamps": [float("nan")]},
        {"tokens": ["好"], "timestamps": [float("inf")]},
        {"tokens": ["好"], "timestamps": [-0.1]},
        {"tokens": ["好"], "timestamps": [True]},
        {"tokens": ["好"], "timestamps": ["0.1"]},
        {"tokens": ["好"], "timestamps": [10**400]},
        {"tokens": ["好", "吧"], "timestamps": [0.2, 0.1]},
        {"tokens": ["好", "吧"], "timestamps": [0.1]},
        {"tokens": [1], "timestamps": [0.1]},
        {"tokens": [""], "timestamps": [0.1]},
        {"tokens": ["a" * 257], "timestamps": [0.1]},
        {"tokens": ["a"] * 4097, "timestamps": [0.1] * 4097},
        {"tokens": ["a" * 256] * 64, "timestamps": [0.1] * 64},
    ],
)
def test_sensevoice_bad_optional_alignment_keeps_text_and_event(metadata) -> None:
    update = asr_module._sensevoice_update(
        SimpleNamespace(text="  好，今天想吃什么？  ", event=" Speech ", **metadata)
    )
    assert update == AsrUpdate("好，今天想吃什么？", True, "Speech")


def test_sensevoice_alignment_keeps_raw_subwords_and_unicode() -> None:
    update = asr_module._sensevoice_update(
        SimpleNamespace(
            text="I like 苹果。",
            event=None,
            tokens=["▁I", "▁like", "▁", "苹", "果", "。"],
            timestamps=[0, 0.24, 0.4, 0.52, 0.68, 0.68],
        )
    )
    assert update.tokens == ("▁I", "▁like", "▁", "苹", "果", "。")
    assert update.timestamps == (0.0, 0.24, 0.4, 0.52, 0.68, 0.68)
    assert update.text == "I like 苹果。" and update.event == ""
