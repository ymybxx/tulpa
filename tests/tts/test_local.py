import io
import tarfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import SecretStr

from fish_memory.config import Settings
from fish_memory.tts.local import (
    MATCHA_MODEL_NAME,
    LocalTtsModelConfig,
    LocalTtsModelDownloader,
    LocalTtsService,
    PcmResampler,
    resolve_tts_provider,
)
from fish_memory.tts.models import TtsStartRequest
from fish_memory.tts.routes import _run_local_tts_task


def _settings(**values: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_api_key": SecretStr("llm-only-key"),
        "tts_provider": "auto",
        "tts_api_key": None,
        "tts_local_model_dir": None,
        "tts_local_num_threads": 2,
    }
    defaults.update(values)
    return Settings.model_construct(
        **defaults,
    )


def test_auto_provider_uses_local_without_tts_key() -> None:
    assert resolve_tts_provider(_settings()) == "local"


def test_auto_provider_uses_dashscope_with_its_own_key() -> None:
    settings = _settings(tts_api_key=SecretStr("tts-only-key"))

    assert resolve_tts_provider(settings) == "dashscope"


def test_each_character_voice_picks_its_own_provider() -> None:
    from fish_memory.tts.local import model_provider, voice_provider
    from fish_memory.tts.voice_config import FishVoiceConfig, VoiceKind

    local = FishVoiceConfig(
        model="matcha-icefall-zh-en", voice="default", kind=VoiceKind.LOCAL
    )
    cloud = FishVoiceConfig(
        model="qwen-audio-3.0-tts-flash", voice="longanhuan_v3.6", kind="preset"
    )
    keyed = _settings(tts_provider="local", tts_api_key=SecretStr("tts-only-key"))
    keyless = _settings(tts_provider="local")

    # 本机 for one character, 通义 for another, whatever the default is.
    assert voice_provider(local, _settings(tts_api_key=SecretStr("k"))) == "local"
    assert voice_provider(cloud, keyed) == "dashscope"
    # A cloud voice without the key, or no choice at all, uses the default.
    assert voice_provider(cloud, keyless) == "local"
    assert voice_provider(None, keyed) == "local"
    assert model_provider("matcha-icefall-zh-en", keyed) == "local"
    assert model_provider("qwen-audio-3.0-tts-flash", keyed) == "dashscope"
    assert model_provider("qwen-audio-3.0-tts-flash", keyless) == "local"


def test_explicit_dashscope_never_falls_back_to_llm_key() -> None:
    settings = _settings(tts_provider="dashscope")

    with pytest.raises(ValueError, match="FISH_MEMORY_TTS_API_KEY"):
        resolve_tts_provider(settings)


def test_explicit_local_ignores_configured_cloud_tts_key() -> None:
    settings = _settings(
        tts_provider="local",
        tts_api_key=SecretStr("unused-cloud-key"),
    )

    assert resolve_tts_provider(settings) == "local"


def test_dashscope_api_key_helper_never_reads_llm_key() -> None:
    from fish_memory.tts.routes import _api_key

    with pytest.raises(RuntimeError, match="FISH_MEMORY_TTS_API_KEY"):
        _api_key(_settings())


def test_local_model_config_contains_bilingual_matcha_assets(tmp_path: Path) -> None:
    config = LocalTtsModelConfig(model_dir=tmp_path / MATCHA_MODEL_NAME)

    files = config.model_files()

    assert config.model_name == "matcha-icefall-zh-en"
    assert files["acoustic_model"].name == "model-steps-3.onnx"
    assert files["vocoder"].name == "vocos-16khz-univ.onnx"
    assert files["phone_zh"].name == "phone-zh.fst"
    assert files["date_zh"].name == "date-zh.fst"
    assert files["number_zh"].name == "number-zh.fst"


def test_local_model_downloader_installs_archive_and_vocoder_atomically(
    tmp_path: Path,
) -> None:
    config = LocalTtsModelConfig(model_dir=tmp_path / MATCHA_MODEL_NAME)

    def fake_download(url: str, destination: Path, _progress: object) -> None:
        if url == config.vocoder_url:
            destination.write_bytes(b"vocoder")
            return
        with tarfile.open(destination, "w:bz2") as archive:
            for relative in (
                "model-steps-3.onnx",
                "lexicon.txt",
                "tokens.txt",
                "phone-zh.fst",
                "date-zh.fst",
                "number-zh.fst",
                "espeak-ng-data/phontab",
            ):
                payload = relative.encode()
                info = tarfile.TarInfo(f"{MATCHA_MODEL_NAME}/{relative}")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    LocalTtsModelDownloader(config, download_file=fake_download).ensure()

    assert config.is_ready()
    assert config.model_files()["vocoder"].read_bytes() == b"vocoder"


def test_pcm_resampler_preserves_pcm_at_native_rate() -> None:
    resampler = PcmResampler(source_rate=16_000, target_rate=16_000)

    pcm = resampler.convert([0.0, 0.5, -0.5])

    assert pcm == b"\x00\x00\xff?\x00\xc0"


@pytest.mark.asyncio
async def test_local_service_streams_multiple_appends_without_cancelling_task(
    tmp_path: Path,
) -> None:
    config = LocalTtsModelConfig(model_dir=tmp_path / MATCHA_MODEL_NAME)
    for name, path in config.model_files().items():
        if name == "data_dir":
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")

    class FakeSynthesizer:
        def generate(
            self,
            text: str,
            _config: object,
            *,
            callback: object | None = None,
        ) -> None:
            if callable(callback):
                callback([0.1] * len(text), 1.0)

    service = LocalTtsService(
        config,
        synthesizer_factory=lambda _config: FakeSynthesizer(),
        generation_config_factory=object,
    )
    await service.prepare()
    stop_event = threading.Event()

    first = [
        packet
        async for packet in service.stream(
            "你好", sample_rate=16_000, stop_event=stop_event
        )
    ]
    second = [
        packet
        async for packet in service.stream(
            "hello", sample_rate=16_000, stop_event=stop_event
        )
    ]

    assert first
    assert second
    assert not stop_event.is_set()


@pytest.mark.asyncio
async def test_local_tts_task_streams_chinese_and_english_pcm() -> None:
    class FakeBrowser:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    {"type": "append", "text": "你好，"},
                    {"type": "append", "text": "hello."},
                    {"type": "finish"},
                ]
            )
            self.events: list[dict[str, object]] = []
            self.audio: list[bytes] = []

        async def receive_json(self) -> dict[str, object]:
            return next(self.messages)

        async def send_json(self, event: dict[str, object]) -> None:
            self.events.append(event)

        async def send_bytes(self, pcm: bytes) -> None:
            self.audio.append(pcm)

    class FakeService:
        model_name = MATCHA_MODEL_NAME

        async def prepare(self) -> None:
            return None

        async def stream(
            self,
            text: str,
            *,
            sample_rate: int,
            stop_event: threading.Event,
        ) -> AsyncIterator[bytes]:
            assert sample_rate == 16_000
            assert not stop_event.is_set()
            yield text.encode("utf-8") + b"\x00"

    browser = FakeBrowser()
    request = TtsStartRequest(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        sample_rate=16_000,
    )

    await _run_local_tts_task(browser, FakeService(), request, started_at=0.0)

    ready = next(event for event in browser.events if event["type"] == "ready")
    assert ready["provider"] == "local"
    assert ready["model"] == MATCHA_MODEL_NAME
    assert ready["voice"] == "default"
    assert [event["type"] for event in browser.events] == [
        "ready",
        "sentence_begin",
        "audio",
        "sentence_end",
        "sentence_begin",
        "audio",
        "sentence_end",
        "finished",
    ]
    assert len(browser.audio) == 2
