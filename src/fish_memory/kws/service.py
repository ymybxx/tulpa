import asyncio
import logging
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from importlib.util import find_spec
from pathlib import Path

from fish_memory.config import Settings
from fish_memory.kws.detector import (
    KwsBackend,
    KwsUnavailableError,
    SherpaKwsBackend,
    WakeWordDetector,
)
from fish_memory.kws.download import KwsModelDownloader, ProgressCallback
from fish_memory.kws.keywords import (
    KeywordFileManager,
    KeywordPreparationError,
    wake_word_phrases,
)
from fish_memory.kws.models import Detection, KwsModelConfig

BackendFactory = Callable[[KwsModelConfig, Path], KwsBackend]


def _default_downloader_factory(
    config: KwsModelConfig,
    progress_callback: ProgressCallback,
) -> KwsModelDownloader:
    return KwsModelDownloader(config, progress_callback=progress_callback)


DownloaderFactory = Callable[[KwsModelConfig, ProgressCallback], KwsModelDownloader]

logger = logging.getLogger(__name__)
_PROGRESS_LOG_INTERVAL_SECONDS = 1.0


class KwsConnection:
    def __init__(self, detector: WakeWordDetector) -> None:
        self._detector = detector
        # The native KWS stream owns mutable state. Audio frames are accepted
        # in worker threads, while reset/close may be requested by a different
        # worker during turn or stream teardown. Keep every detector operation
        # in one critical section so close cannot invalidate an in-flight call.
        self._detector_lock = threading.Lock()
        self._closed = False

    def accept_pcm(self, pcm: bytes) -> list[Detection]:
        with self._detector_lock:
            if self._closed:
                raise KwsUnavailableError("KWS connection is closed")
            return self._detector.accept_pcm(pcm)

    def reset(self) -> None:
        with self._detector_lock:
            if self._closed:
                return
            self._detector.reset()

    def close(self) -> None:
        with self._detector_lock:
            if self._closed:
                return
            self._closed = True
            self._detector.close()


class KwsService:
    def __init__(
        self,
        config: KwsModelConfig | None,
        *,
        backend_factory: BackendFactory = SherpaKwsBackend,
        downloader_factory: DownloaderFactory = _default_downloader_factory,
    ) -> None:
        self.config = config
        self.default_wake_word = "小鱼"
        self._backend_factory = backend_factory
        self._downloader_factory = downloader_factory
        self._keyword_manager = (
            KeywordFileManager(config) if config is not None else None
        )
        self._backends: dict[tuple[str, ...], KwsBackend] = {}
        self._prepare_lock = asyncio.Lock()
        self._download_task: asyncio.Task[None] | None = None
        self._downloading = False
        self._downloaded_bytes = 0
        self._download_total_bytes: int | None = None
        self._last_progress_log_at = 0.0
        self._error = self._initial_error(config)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        backend_factory: BackendFactory = SherpaKwsBackend,
        downloader_factory: DownloaderFactory = _default_downloader_factory,
    ) -> "KwsService":
        if find_spec("sherpa_onnx") is None:
            service = cls(
                None,
                backend_factory=backend_factory,
                downloader_factory=downloader_factory,
            )
            service.default_wake_word = getattr(settings, "wake_word", "小鱼")
            service._error = "Native KWS is unavailable; install fish-memory[voice]"
            return service
        service = cls(
            KwsModelConfig.from_settings(settings),
            backend_factory=backend_factory,
            downloader_factory=downloader_factory,
        )
        service.default_wake_word = getattr(settings, "wake_word", "小鱼")
        return service

    @staticmethod
    def _initial_error(config: KwsModelConfig | None) -> str | None:
        if config is None:
            return "未配置 FISH_MEMORY_KWS_MODEL_DIR"
        try:
            config.model_files()
        except ValueError as error:
            return str(error)
        return None

    async def start(self) -> None:
        if self.config is None or self._error is not None:
            return
        if self.config.is_ready():
            return
        if self._download_task is not None and not self._download_task.done():
            return
        self._downloaded_bytes = 0
        self._download_total_bytes = None
        self._last_progress_log_at = 0.0
        self._downloading = True
        logger.info(
            "开始下载 KWS 模型: name=%s, directory=%s",
            self.config.model_name,
            self.config.model_dir,
        )
        self._download_task = asyncio.create_task(self._download_model())

    def _on_download_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int | None,
    ) -> None:
        self._downloaded_bytes = max(0, downloaded_bytes)
        self._download_total_bytes = (
            total_bytes if total_bytes is not None and total_bytes > 0 else None
        )
        now = time.monotonic()
        if (
            downloaded_bytes > 0
            and now - self._last_progress_log_at < _PROGRESS_LOG_INTERVAL_SECONDS
            and self._downloaded_bytes != self._download_total_bytes
        ):
            return
        self._last_progress_log_at = now
        if self._download_total_bytes is None:
            logger.info(
                "KWS 模型下载进度: %s",
                _format_bytes(self._downloaded_bytes),
            )
            return
        percent = (
            min(
                self._downloaded_bytes / self._download_total_bytes,
                1.0,
            )
            * 100
        )
        logger.info(
            "KWS 模型下载进度: %.1f%% (%s/%s)",
            percent,
            _format_bytes(self._downloaded_bytes),
            _format_bytes(self._download_total_bytes),
        )

    async def _download_model(self) -> None:
        if self.config is None:
            return
        try:
            downloader = self._downloader_factory(
                self.config,
                self._on_download_progress,
            )
            await asyncio.to_thread(downloader.ensure)
            logger.info("KWS 模型下载完成: directory=%s", self.config.model_dir)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._error = str(error)
            logger.error("KWS 模型下载失败: %s", error)
        finally:
            self._downloading = False

    def status(self) -> dict[str, object]:
        config = self.config
        download_percent = None
        if self._download_total_bytes is not None:
            download_percent = round(
                min(self._downloaded_bytes / self._download_total_bytes, 1.0) * 100,
                1,
            )
        return {
            "available": (
                config is not None and self._error is None and config.is_ready()
            ),
            "downloading": self._downloading,
            "wakeWord": self.default_wake_word,
            "sampleRate": config.sample_rate if config else 16_000,
            "channels": 1,
            "chunkSize": config.chunk_size if config else 8,
            "modelDir": str(config.model_dir) if config else None,
            "error": self._error,
            "downloadedBytes": self._downloaded_bytes,
            "totalBytes": self._download_total_bytes,
            "downloadPercent": download_percent,
        }

    async def open_stream(
        self, wake_word: str | Sequence[str] | None = None
    ) -> KwsConnection:
        if self.config is None or self._keyword_manager is None:
            raise KwsUnavailableError(self._error or "KWS is not configured")
        if self._error is not None:
            raise KwsUnavailableError(self._error)
        if not self.config.is_ready():
            if self._download_task is None or self._download_task.done():
                await self.start()
            raise KwsUnavailableError("KWS 模型正在下载，请稍后重试")

        try:
            phrases = wake_word_phrases(wake_word or self.default_wake_word)
        except KeywordPreparationError as error:
            raise KwsUnavailableError(str(error)) from error
        async with self._prepare_lock:
            keywords_path = await asyncio.to_thread(
                self._keyword_manager.prepare,
                phrases,
            )
            backend = self._backends.get(phrases)
            if backend is None:
                try:
                    backend = await asyncio.to_thread(
                        self._backend_factory,
                        self.config,
                        keywords_path,
                    )
                except KwsUnavailableError as error:
                    self._error = str(error)
                    raise
                self._backends[phrases] = backend
        return KwsConnection(WakeWordDetector(backend, keywords_path=keywords_path))

    async def close(self) -> None:
        if self._download_task is not None and not self._download_task.done():
            self._download_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._download_task
        self._backends.clear()


def _format_bytes(value: int) -> str:
    size = float(max(value, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"
