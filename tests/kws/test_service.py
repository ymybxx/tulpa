import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from fish_memory.config import Settings
from fish_memory.kws.detector import KwsUnavailableError, WakeWordDetector
from fish_memory.kws.models import Detection, KwsModelConfig
from fish_memory.kws.service import KwsConnection, KwsService


class BlockingDetector:
    def __init__(self) -> None:
        self.accept_started = threading.Event()
        self.release_accept = threading.Event()
        self.close_entered = threading.Event()
        self.calls: list[str] = []

    def accept_pcm(self, _pcm: bytes) -> list[Detection]:
        self.calls.append("accept:start")
        self.accept_started.set()
        if not self.release_accept.wait(timeout=1):
            raise TimeoutError("test did not release in-flight KWS detection")
        self.calls.append("accept:end")
        return []

    def reset(self) -> None:
        self.calls.append("reset")

    def close(self) -> None:
        self.calls.append("close")
        self.close_entered.set()


def test_connection_close_waits_for_in_flight_accept_and_is_idempotent() -> None:
    detector = BlockingDetector()
    connection = KwsConnection(cast(WakeWordDetector, detector))
    close_started = threading.Event()

    def close_connection() -> None:
        close_started.set()
        connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        accepting = executor.submit(connection.accept_pcm, b"\x00\x00")
        assert detector.accept_started.wait(timeout=1)
        closing = executor.submit(close_connection)
        assert close_started.wait(timeout=1)

        assert detector.close_entered.wait(timeout=0.05) is False
        assert closing.done() is False
        detector.release_accept.set()
        assert accepting.result(timeout=1) == []
        closing.result(timeout=1)

    connection.close()
    connection.reset()
    with pytest.raises(KwsUnavailableError, match="closed"):
        connection.accept_pcm(b"\x00\x00")

    assert detector.calls == ["accept:start", "accept:end", "close"]


def test_status_reports_missing_model_files(tmp_path: Path) -> None:
    service = KwsService(KwsModelConfig(model_dir=tmp_path))

    status = service.status()

    assert status["available"] is False
    assert status["downloading"] is False
    assert status["error"] is None


@pytest.mark.asyncio
async def test_kws_does_not_download_without_voice_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fish_memory.kws.service.find_spec",
        lambda _name: None,
        raising=False,
    )
    monkeypatch.setenv("FISH_MEMORY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FISH_MEMORY_LLM_API_KEY", "test-key")
    monkeypatch.setenv("FISH_MEMORY_LLM_MODEL", "test-model")
    monkeypatch.setenv("FISH_MEMORY_LLM_BASE_URL", "https://example.invalid/v1")
    settings = Settings(_env_file=None)

    service = KwsService.from_settings(settings)
    await service.start()

    status = service.status()
    assert status["available"] is False
    assert status["downloading"] is False
    assert "fish-memory[voice]" in str(status["error"])


@pytest.mark.asyncio
async def test_start_downloads_model_in_background(tmp_path: Path) -> None:
    config = KwsModelConfig(model_dir=tmp_path / "kws")
    started = threading.Event()

    class FakeDownloader:
        def __init__(
            self,
            _config: KwsModelConfig,
            progress_callback,
        ) -> None:
            self._progress_callback = progress_callback

        def ensure(self) -> None:
            started.set()
            self._progress_callback(128, 256)
            config.model_dir.mkdir(parents=True, exist_ok=True)
            for path in config.model_files().values():
                path.write_text("model\n", encoding="utf-8")

    service = KwsService(
        config,
        downloader_factory=lambda _config, _progress_callback: FakeDownloader(
            _config,
            _progress_callback,
        ),
    )

    await service.start()
    assert await asyncio.to_thread(started.wait, 1)
    for _ in range(20):
        if service.status()["available"]:
            break
        await asyncio.sleep(0.01)

    assert service.status()["available"] is True
    assert service.status()["downloadedBytes"] == 128
    assert service.status()["totalBytes"] == 256
    assert service.status()["downloadPercent"] == 50.0
    await service.close()


@pytest.mark.asyncio
async def test_download_logs_lifecycle_and_progress(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = KwsModelConfig(model_dir=tmp_path / "kws")

    class FakeDownloader:
        def __init__(self, _config: KwsModelConfig, progress_callback) -> None:
            self._progress_callback = progress_callback

        def ensure(self) -> None:
            self._progress_callback(128, 256)
            config.model_dir.mkdir(parents=True, exist_ok=True)
            for path in config.model_files().values():
                path.write_text("model\n", encoding="utf-8")

    service = KwsService(
        config,
        downloader_factory=lambda _config, progress_callback: FakeDownloader(
            _config,
            progress_callback,
        ),
    )

    with caplog.at_level(logging.INFO, logger="fish_memory.kws.service"):
        await service.start()
        for _ in range(20):
            if service.status()["available"]:
                break
            await asyncio.sleep(0.01)

    messages = [record.getMessage() for record in caplog.records]
    assert any("开始下载 KWS 模型" in message for message in messages)
    assert any("KWS 模型下载进度" in message for message in messages)
    assert any("KWS 模型下载完成" in message for message in messages)
    await service.close()
