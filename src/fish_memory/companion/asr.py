"""Local multilingual ASR backed by sherpa-onnx."""

import asyncio
import logging
import math
import shutil
import tarfile
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen

import numpy as np

from fish_memory.model_catalog import SENSEVOICE_FILES
from fish_memory.model_sources import ModelStatusReporter, install_catalog

SENSEVOICE_MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
ASR_MODEL_NAME = SENSEVOICE_MODEL_NAME
ASR_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{ASR_MODEL_NAME}.tar.bz2"
)

ProgressCallback = Callable[[int, int | None], None]

logger = logging.getLogger(__name__)
_PROGRESS_LOG_INTERVAL_SECONDS = 1.0
_SENSEVOICE_MODEL_FILES = (
    ("model", "model.int8.onnx"),
    ("tokens", "tokens.txt"),
)
_MAX_ALIGNMENT_TOKENS = 4096
_MAX_ALIGNMENT_TOKEN_CHARS = 256
_MAX_ALIGNMENT_CHARS = 16_000


def pcm16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if len(pcm_bytes) % 2:
        raise ValueError("PCM must contain complete 16-bit samples")
    samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
    samples /= 32_768
    return samples


@dataclass(frozen=True, slots=True)
class AsrModelConfig:
    model_dir: Path
    num_threads: int = 2
    sample_rate: int = 16_000
    feature_dim: int = 80
    model: Literal["sensevoice"] = "sensevoice"

    def __post_init__(self) -> None:
        if self.model != "sensevoice":
            raise ValueError("ASR 仅支持 SenseVoice")

    @classmethod
    def from_settings(cls, settings: object) -> "AsrModelConfig":
        configured_dir = getattr(settings, "asr_model_dir", None)
        model_dir = (
            Path(configured_dir).expanduser()
            if isinstance(configured_dir, str) and configured_dir.strip()
            else Path.cwd() / ".cache" / "fish-memory" / "asr" / SENSEVOICE_MODEL_NAME
        )
        return cls(
            model_dir=model_dir,
            num_threads=int(getattr(settings, "asr_num_threads", 2)),
        )

    @property
    def model_name(self) -> str:
        return SENSEVOICE_MODEL_NAME

    @property
    def model_url(self) -> str:
        return ASR_MODEL_URL

    def model_files(self) -> dict[str, Path]:
        return {
            role: self.model_dir / filename
            for role, filename in _SENSEVOICE_MODEL_FILES
        }

    def is_ready(self) -> bool:
        return all(
            path.is_file() and path.stat().st_size > 0
            for path in self.model_files().values()
        )


@dataclass(frozen=True, slots=True)
class AsrUpdate:
    text: str
    is_final: bool
    event: str = ""
    tokens: tuple[str, ...] = ()
    timestamps: tuple[float, ...] = ()


def validated_token_alignment(
    tokens: object, timestamps: object
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Keep optional native token times only when the entire pair is usable.

    Times are token points in seconds relative to the decoded PCM, not word
    intervals. Validation must never discard the independent transcript.
    """
    if not isinstance(tokens, (list, tuple)) or not isinstance(
        timestamps, (list, tuple)
    ):
        return (), ()
    if not 0 < len(tokens) <= _MAX_ALIGNMENT_TOKENS or len(tokens) != len(timestamps):
        return (), ()
    normalized_tokens: list[str] = []
    normalized_times: list[float] = []
    total_chars = 0
    previous_time = 0.0
    for token, timestamp in zip(tokens, timestamps, strict=True):
        if not isinstance(token, str) or not 0 < len(token) <= (
            _MAX_ALIGNMENT_TOKEN_CHARS
        ):
            return (), ()
        total_chars += len(token)
        if total_chars > _MAX_ALIGNMENT_CHARS:
            return (), ()
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            return (), ()
        try:
            seconds = float(timestamp)
        except (OverflowError, ValueError):
            return (), ()
        if not math.isfinite(seconds) or seconds < previous_time:
            return (), ()
        normalized_tokens.append(token)
        normalized_times.append(seconds)
        previous_time = seconds
    return tuple(normalized_tokens), tuple(normalized_times)


def _sensevoice_update(result: Any) -> AsrUpdate:
    raw_event = getattr(result, "event", "")
    tokens, timestamps = validated_token_alignment(
        getattr(result, "tokens", ()), getattr(result, "timestamps", ())
    )
    return AsrUpdate(
        text=str(result.text).strip(),
        is_final=True,
        event=str(raw_event).strip() if raw_event is not None else "",
        tokens=tokens,
        timestamps=timestamps,
    )


class AsrUnavailableError(RuntimeError):
    pass


class AsrInferenceBusyError(AsrUnavailableError):
    """A low-priority snapshot decode could not claim the shared recognizer."""


def _create_sense_voice_recognizer(
    config: AsrModelConfig,
    sherpa_onnx: Any,
) -> Any:
    files = config.model_files()
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        tokens=str(files["tokens"]),
        model=str(files["model"]),
        num_threads=config.num_threads,
        sample_rate=config.sample_rate,
        feature_dim=config.feature_dim,
        decoding_method="greedy_search",
        provider="cpu",
        language="zh",
        use_itn=True,
    )


class SenseVoiceDraftDecoder:
    """Decode immutable PCM snapshots on an isolated native recognizer."""

    def __init__(self, config: AsrModelConfig) -> None:
        if not config.is_ready():
            raise AsrUnavailableError("ASR 模型尚未下载完成")
        try:
            sherpa_onnx = import_module("sherpa_onnx")
            self._recognizer = _create_sense_voice_recognizer(config, sherpa_onnx)
        except Exception as error:  # pragma: no cover - optional native boundary
            raise AsrUnavailableError(f"无法初始化草稿 ASR: {error}") from error
        self._sample_rate = config.sample_rate
        self._decode_lock = threading.Lock()

    def recognize(self, pcm: bytes) -> str:
        return self.recognize_update(pcm).text

    def recognize_update(self, pcm: bytes) -> AsrUpdate:
        """Preserve the decoded event when this result becomes a committed turn."""
        if type(pcm) is not bytes:
            raise TypeError("draft PCM must be immutable bytes")
        if not pcm:
            return AsrUpdate(text="", is_final=True)
        with self._decode_lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(
                self._sample_rate,
                pcm16le_to_float32(pcm),
            )
            self._recognizer.decode_stream(stream)
            return _sensevoice_update(stream.result)


class SharedSenseVoiceDraftDecoder:
    """Use an authoritative SenseVoice stream for immutable draft snapshots."""

    def __init__(self, source: "StreamingAsr") -> None:
        self._source = source

    def recognize(self, pcm: bytes) -> str:
        return self.recognize_update(pcm).text

    def recognize_update(self, pcm: bytes) -> AsrUpdate:
        if type(pcm) is not bytes:
            raise TypeError("draft PCM must be immutable bytes")
        result = self._source.try_recognize_snapshot_update(pcm)
        if result is None:
            raise AsrInferenceBusyError("shared SenseVoice recognizer is busy")
        return result


class AsrModelDownloader:
    def __init__(
        self,
        config: AsrModelConfig,
        *,
        progress_callback: ProgressCallback | None = None,
        attempts: int = 3,
    ) -> None:
        self._config = config
        self._progress_callback = progress_callback or (
            lambda _downloaded, _total: None
        )
        self._attempts = attempts

    def ensure(self) -> None:
        reporter = ModelStatusReporter("asr", "听力")
        if self._config.is_ready():
            reporter.ready()
            return
        reporter.preparing()
        report = self._progress_callback

        def progress(done: int, total: int | None) -> None:
            report(done, total)
            reporter.downloading(done, total)

        self._progress_callback = progress
        try:
            self._ensure()
        except Exception as error:
            reporter.failed(str(error))
            raise
        finally:
            self._progress_callback = report
        reporter.ready()

    def _ensure(self) -> None:
        destination = self._config.model_dir
        if destination.exists() and any(destination.iterdir()):
            raise AsrUnavailableError(f"ASR 模型目录不完整，拒绝覆盖: {destination}")
        if self._config.model_name == SENSEVOICE_MODEL_NAME and install_catalog(
            destination, SENSEVOICE_FILES, progress=self._progress_callback
        ):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix="fish-memory-asr-", dir=destination.parent)
        )
        try:
            archive_path = temporary_root / f"{self._config.model_name}.tar.bz2"
            last_error: Exception | None = None
            for attempt in range(self._attempts):
                try:
                    self._download(archive_path)
                    if archive_path.stat().st_size == 0:
                        raise AsrUnavailableError("下载的 ASR 模型包为空")
                    break
                except Exception as error:
                    last_error = error
                    archive_path.unlink(missing_ok=True)
                    if attempt + 1 < self._attempts:
                        self._progress_callback(0, None)
                        time.sleep(2**attempt)
            else:
                raise AsrUnavailableError(f"下载 ASR 模型包失败: {last_error}")

            extracted_root = temporary_root / "extracted"
            extracted_root.mkdir()
            self._extract_safely(archive_path, extracted_root)
            extracted_model = extracted_root / self._config.model_name
            if not all(
                (extracted_model / path.relative_to(destination)).is_file()
                for path in self._config.model_files().values()
            ):
                raise AsrUnavailableError("下载的 ASR 模型缺少必要文件")
            if destination.exists():
                destination.rmdir()
            extracted_model.rename(destination)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _download(self, destination: Path) -> None:
        request = Request(
            self._config.model_url,
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

    @staticmethod
    def _extract_safely(archive_path: Path, destination: Path) -> None:
        root = destination.resolve()
        with tarfile.open(archive_path, "r:bz2") as archive:
            for member in archive.getmembers():
                target = (root / member.name).resolve()
                if target != root and root not in target.parents:
                    raise AsrUnavailableError("ASR 模型包包含非法路径")
            archive.extractall(root)


class StreamingAsr:
    def __init__(self, config: AsrModelConfig) -> None:
        self._config = config
        if not config.is_ready():
            raise AsrUnavailableError("ASR 模型尚未下载完成")
        try:
            sherpa_onnx = import_module("sherpa_onnx")
        except Exception as error:  # pragma: no cover - optional host library
            raise AsrUnavailableError(f"无法加载 sherpa-onnx: {error}") from error

        self._stream_lock = threading.Lock()
        self._decode_lock = threading.Lock()
        self._decode_condition = threading.Condition(self._decode_lock)
        self._decode_active = False
        self._authoritative_decode_waiters = 0
        self._pcm_buffer = bytearray()
        try:
            self._recognizer = _create_sense_voice_recognizer(
                config,
                sherpa_onnx,
            )
        except Exception as error:  # pragma: no cover - model/runtime boundary
            raise AsrUnavailableError(f"无法初始化 ASR: {error}") from error

    def accept_pcm(self, pcm: bytes) -> AsrUpdate | None:
        if not pcm:
            return None
        with self._stream_lock:
            self._pcm_buffer.extend(pcm)
        return None

    def finish_utterance_update(self) -> AsrUpdate:
        """Return the current result and reset only at a VAD-defined boundary."""
        with self._stream_lock:
            if not self._pcm_buffer:
                return AsrUpdate(text="", is_final=True)
            pcm = bytes(self._pcm_buffer)
            self._pcm_buffer.clear()
        return self.recognize_snapshot_update(pcm)

    def finish_utterance(self) -> str:
        """Return text for callers that do not consume SenseVoice metadata."""
        return self.finish_utterance_update().text

    def snapshot_utterance(self) -> bytes:
        """Copy buffered PCM without closing the current utterance."""
        with self._stream_lock:
            return bytes(self._pcm_buffer)

    def buffered_pcm_bytes(self) -> int:
        """Return buffered PCM size without copying the utterance."""
        with self._stream_lock:
            return len(self._pcm_buffer)

    def recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate:
        """Decode copied SenseVoice audio and preserve its event metadata."""
        if not pcm:
            return AsrUpdate(text="", is_final=True)
        self._acquire_authoritative_decode()
        try:
            return self._decode_sensevoice_snapshot(pcm)
        finally:
            self._release_decode()

    def try_recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate | None:
        """Decode a low-priority snapshot only when the shared model is idle.

        This method never mutates the authoritative utterance buffer and never
        waits behind another decode. ``None`` means the recognizer is currently
        reserved by an authoritative decode.
        """
        if not pcm:
            return AsrUpdate(text="", is_final=True)
        if not self._try_acquire_speculative_decode():
            return None
        try:
            return self._decode_sensevoice_snapshot(pcm)
        finally:
            self._release_decode()

    def _decode_sensevoice_snapshot(self, pcm: bytes) -> AsrUpdate:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(
            self._config.sample_rate,
            pcm16le_to_float32(pcm),
        )
        self._recognizer.decode_stream(stream)
        return _sensevoice_update(stream.result)

    def _acquire_authoritative_decode(self) -> None:
        with self._decode_condition:
            self._authoritative_decode_waiters += 1
            try:
                while self._decode_active:
                    self._decode_condition.wait()
                self._decode_active = True
            finally:
                self._authoritative_decode_waiters -= 1

    def _try_acquire_speculative_decode(self) -> bool:
        with self._decode_condition:
            if self._decode_active or self._authoritative_decode_waiters:
                return False
            self._decode_active = True
            return True

    def _release_decode(self) -> None:
        with self._decode_condition:
            self._decode_active = False
            self._decode_condition.notify_all()

    def recognize_snapshot(self, pcm: bytes) -> str:
        """Decode copied SenseVoice audio without modifying the live buffer."""
        return self.recognize_snapshot_update(pcm).text

    def consume_snapshot(self, pcm: bytes) -> bool:
        """Remove a decoded SenseVoice prefix without discarding newer audio."""
        with self._stream_lock:
            prefix_length = len(pcm)
            if self._pcm_buffer[:prefix_length] != pcm:
                return False
            del self._pcm_buffer[:prefix_length]
            return True

    def reset(self) -> None:
        with self._stream_lock:
            self._pcm_buffer.clear()


class AsrService:
    def __init__(self, config: AsrModelConfig) -> None:
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
        logger.info(
            "开始下载 ASR 模型: name=%s, directory=%s",
            self.config.model_name,
            self.config.model_dir,
        )
        self._download_task = asyncio.create_task(asyncio.to_thread(self._download))

    async def wait_until_ready(self, *, poll_seconds: float = 0.5) -> None:
        await self.start()
        while not self.config.is_ready():
            if self._error is not None:
                raise AsrUnavailableError(self._error)
            await asyncio.sleep(poll_seconds)

    def open_stream(self) -> StreamingAsr:
        if self._error is not None:
            raise AsrUnavailableError(self._error)
        return StreamingAsr(self.config)

    def status(self) -> dict[str, object]:
        percent = None
        if self._total_bytes:
            percent = round(min(self._downloaded_bytes / self._total_bytes, 1) * 100, 1)
        return {
            "available": self.config.is_ready() and self._error is None,
            "downloading": self._downloading,
            "model": self.config.model,
            "modelName": self.config.model_name,
            "modelDir": str(self.config.model_dir),
            "downloadedBytes": self._downloaded_bytes,
            "totalBytes": self._total_bytes,
            "downloadPercent": percent,
            "error": self._error,
        }

    def _download(self) -> None:
        try:
            AsrModelDownloader(
                self.config,
                progress_callback=self._on_progress,
            ).ensure()
            logger.info("ASR 模型下载完成: directory=%s", self.config.model_dir)
        except Exception as error:
            self._error = str(error)
            logger.error("ASR 模型下载失败: %s", error)
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
            logger.info("ASR 模型下载进度: %s", _format_bytes(downloaded))
            return
        percent = min(downloaded / total, 1.0) * 100
        logger.info(
            "ASR 模型下载进度: %.1f%% (%s/%s)",
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
