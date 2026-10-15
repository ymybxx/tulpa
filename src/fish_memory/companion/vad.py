"""Local Silero VAD model management and streaming adapter."""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from urllib.request import Request, urlopen

from fish_memory.companion.asr import pcm16le_to_float32
from fish_memory.companion.audio_processing import speech_pre_roll_ms_to_frames
from fish_memory.model_catalog import SILERO_VAD_FILE
from fish_memory.model_sources import ModelDownloadError, fetch_file

SILERO_VAD_MODEL_NAME = "silero_vad.onnx"
SILERO_VAD_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"asr-models/{SILERO_VAD_MODEL_NAME}"
)

logger = logging.getLogger(__name__)
_PROGRESS_LOG_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class SileroVadConfig:
    model_dir: Path
    threshold: float = 0.25
    min_silence_duration: float = 0.35
    min_speech_duration: float = 0.08
    window_size: int = 512
    max_speech_duration: float = 20.0
    num_threads: int = 1
    sample_rate: int = 16_000

    @classmethod
    def from_settings(cls, settings: object) -> SileroVadConfig:
        configured_dir = getattr(settings, "companion_vad_model_dir", None)
        model_dir = (
            Path(configured_dir).expanduser()
            if isinstance(configured_dir, str) and configured_dir.strip()
            else Path.cwd() / ".cache" / "fish-memory" / "vad"
        )
        return cls(
            model_dir=model_dir,
            threshold=float(getattr(settings, "companion_vad_threshold", 0.25)),
            min_silence_duration=(
                float(getattr(settings, "companion_vad_min_silence_ms", 350)) / 1000
            ),
            min_speech_duration=(
                float(getattr(settings, "companion_vad_min_speech_ms", 80)) / 1000
            ),
            window_size=int(getattr(settings, "companion_vad_window_size", 512)),
            max_speech_duration=float(
                getattr(settings, "companion_vad_max_speech_seconds", 20.0)
            ),
            num_threads=int(getattr(settings, "companion_vad_num_threads", 1)),
        )

    @property
    def model_path(self) -> Path:
        return self.model_dir / SILERO_VAD_MODEL_NAME

    @property
    def speech_pre_roll_frames(self) -> int:
        return speech_pre_roll_ms_to_frames(
            round(self.min_speech_duration * 1000),
            window_ms=math.ceil(self.window_size * 1000 / self.sample_rate),
        )

    @property
    def speech_confirmation_frames(self) -> int:
        confirmation_ms = round(self.min_speech_duration * 1000) + math.ceil(
            self.window_size * 1000 / self.sample_rate
        )
        return max(1, math.ceil(confirmation_ms / 10))

    def is_ready(self) -> bool:
        return self.model_path.is_file() and self.model_path.stat().st_size > 0


class SileroVadDownloader:
    def __init__(
        self,
        config: SileroVadConfig,
        *,
        progress_callback: Callable[[int, int | None], None] | None = None,
        attempts: int = 3,
    ) -> None:
        self._config = config
        self._progress_callback = progress_callback or (
            lambda _downloaded, _total: None
        )
        self._attempts = attempts

    def ensure(self) -> None:
        if self._config.is_ready():
            return
        destination = self._config.model_dir
        destination.mkdir(parents=True, exist_ok=True)
        try:
            # Bundled with the app, then the original release file.
            fetch_file(
                SILERO_VAD_FILE,
                self._config.model_path,
                progress=self._progress_callback,
            )
            return
        except ModelDownloadError as error:
            logger.warning("准备 Silero VAD 模型失败，改用原始下载: %s", error)
        temporary_root = Path(
            tempfile.mkdtemp(prefix="fish-memory-vad-", dir=destination.parent)
        )
        temporary_model = temporary_root / SILERO_VAD_MODEL_NAME
        try:
            last_error: Exception | None = None
            for attempt in range(self._attempts):
                try:
                    self._download(temporary_model)
                    if temporary_model.stat().st_size == 0:
                        raise RuntimeError("下载的 Silero VAD 模型为空")
                    break
                except Exception as error:
                    last_error = error
                    temporary_model.unlink(missing_ok=True)
                    if attempt + 1 < self._attempts:
                        self._progress_callback(0, None)
                        time.sleep(2**attempt)
            else:
                raise RuntimeError(f"下载 Silero VAD 模型失败: {last_error}")
            temporary_model.replace(self._config.model_path)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _download(self, destination: Path) -> None:
        request = Request(
            SILERO_VAD_MODEL_URL,
            headers={"User-Agent": "fish-memory/0.1"},
        )
        with urlopen(request, timeout=60) as response, destination.open("wb") as output:
            total: int | None = None
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    total = int(content_length)
                except ValueError:
                    pass
            downloaded = 0
            self._progress_callback(downloaded, total)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                self._progress_callback(downloaded, total)


class SileroVad:
    """Frame adapter around sherpa-onnx's streaming Silero VAD."""

    def __init__(self, config: SileroVadConfig) -> None:
        if not config.is_ready():
            raise RuntimeError(f"Silero VAD 模型尚未下载完成: {config.model_path}")
        try:
            sherpa_onnx = import_module("sherpa_onnx")
        except Exception as error:  # pragma: no cover - optional runtime boundary
            raise RuntimeError(f"无法加载 sherpa-onnx: {error}") from error

        silero_config = sherpa_onnx.SileroVadModelConfig(
            model=str(config.model_path),
            threshold=config.threshold,
            min_silence_duration=config.min_silence_duration,
            min_speech_duration=config.min_speech_duration,
            window_size=config.window_size,
            max_speech_duration=config.max_speech_duration,
        )
        model_config = sherpa_onnx.VadModelConfig(
            silero_vad=silero_config,
            sample_rate=config.sample_rate,
            num_threads=config.num_threads,
            provider="cpu",
            debug=False,
        )
        if not model_config.validate():
            raise RuntimeError("Silero VAD 配置无效")
        self._detector = sherpa_onnx.VoiceActivityDetector(
            model_config,
            buffer_size_in_seconds=max(config.max_speech_duration, 60.0),
        )
        # sherpa-onnx's streaming VAD owns mutable native state and is not
        # thread-safe. Frame processing normally runs in a worker thread while
        # an utterance finalizer may reset the stream from another one, so every
        # access to the detector must share this lock.
        self._detector_lock = threading.Lock()

    def is_speech(self, pcm: bytes) -> bool:
        samples = pcm16le_to_float32(pcm)
        with self._detector_lock:
            self._detector.accept_waveform(samples)
            active = bool(self._detector.is_speech_detected())
            if not active:
                while not self._detector.empty():
                    self._detector.pop()
            return active

    def reset(self) -> None:
        with self._detector_lock:
            self._detector.reset()


class SileroVadService:
    def __init__(self, config: SileroVadConfig) -> None:
        self.config = config
        self._download_task: asyncio.Task[None] | None = None
        self._downloading = False
        self._downloaded_bytes = 0
        self._total_bytes: int | None = None
        self._last_progress_log_at = 0.0
        self._error: str | None = None

    async def start(self) -> None:
        if self.config.is_ready() or (
            self._download_task is not None and not self._download_task.done()
        ):
            return
        self._downloading = True
        self._last_progress_log_at = 0.0
        logger.info("开始下载 Silero VAD 模型: directory=%s", self.config.model_dir)
        self._download_task = asyncio.create_task(asyncio.to_thread(self._download))

    async def wait_until_ready(self, *, poll_seconds: float = 0.5) -> None:
        await self.start()
        while not self.config.is_ready():
            if self._error is not None:
                raise RuntimeError(self._error)
            await asyncio.sleep(poll_seconds)

    def open_stream(self) -> SileroVad:
        if self._error is not None:
            raise RuntimeError(self._error)
        return SileroVad(self.config)

    def status(self) -> dict[str, object]:
        percent = None
        if self._total_bytes:
            percent = round(min(self._downloaded_bytes / self._total_bytes, 1) * 100, 1)
        return {
            "available": self.config.is_ready() and self._error is None,
            "downloading": self._downloading,
            "modelDir": str(self.config.model_dir),
            "downloadedBytes": self._downloaded_bytes,
            "totalBytes": self._total_bytes,
            "downloadPercent": percent,
            "error": self._error,
        }

    def _download(self) -> None:
        try:
            SileroVadDownloader(
                self.config,
                progress_callback=self._on_progress,
            ).ensure()
            logger.info("Silero VAD 模型下载完成: path=%s", self.config.model_path)
        except Exception as error:
            self._error = str(error)
            logger.error("Silero VAD 模型下载失败: %s", error)
        finally:
            self._downloading = False

    def _on_progress(self, downloaded: int, total: int | None) -> None:
        self._downloaded_bytes = downloaded
        self._total_bytes = total
        now = time.monotonic()
        if (
            downloaded > 0
            and now - self._last_progress_log_at < _PROGRESS_LOG_INTERVAL_SECONDS
            and downloaded != total
        ):
            return
        self._last_progress_log_at = now
        if total is None or total <= 0:
            logger.info("Silero VAD 模型下载进度: %s", _format_bytes(downloaded))
            return
        percent = min(downloaded / total, 1.0) * 100
        logger.info(
            "Silero VAD 模型下载进度: %.1f%% (%s/%s)",
            percent,
            _format_bytes(downloaded),
            _format_bytes(total),
        )


def _format_bytes(value: int) -> str:
    size = float(max(value, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"
