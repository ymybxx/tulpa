"""Local Chinese-English streaming TTS backed by sherpa-onnx Matcha."""

import asyncio
import logging
import math
import re
import shutil
import tarfile
import tempfile
import threading
import time
from array import array
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from fish_memory.config import Settings
from fish_memory.model_catalog import ESPEAK_BUNDLE, ESPEAK_TREE_SHA256, MATCHA_FILES
from fish_memory.model_sources import (
    ModelStatusReporter,
    copy_bundled_tree,
    install_catalog,
)

if TYPE_CHECKING:
    from fish_memory.tts.voice_config import FishVoiceConfig

logger = logging.getLogger(__name__)

MATCHA_MODEL_NAME = "matcha-icefall-zh-en"
MATCHA_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "matcha-icefall-zh-en.tar.bz2"
)
MATCHA_VOCODER_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/"
    "vocos-16khz-univ.onnx"
)
MATCHA_VOCODER_NAME = "vocos-16khz-univ.onnx"
LOCAL_TTS_SAMPLE_RATE = 16_000

TtsProviderName = Literal["dashscope", "local"]
ProgressCallback = Callable[[int, int | None], None]
FileDownloader = Callable[[str, Path, ProgressCallback], None]


# Emoji and pictographs: the local voice would spell their English names
# ("smiling face"), so they are dropped before synthesis. The shown text keeps them.
_UNSPOKEN_SYMBOLS = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoticons, pictographs, transport, symbols
    "\U0001f1e6-\U0001f1ff"  # regional indicators (flags)
    "\u2300-\u23ff"  # technical symbols (⌚ ⏰ …)
    "\u2600-\u27bf"  # miscellaneous symbols and dingbats (☀ ★ ✨ …)
    "\u2b00-\u2bff"  # arrows and stars (⭐ …)
    "\ufe0e\ufe0f\u200d\u20e3"  # variation selectors, joiners, keycaps
    "]+"
)


def speakable_text(text: str) -> str:
    """What the local voice should read aloud."""
    return _UNSPOKEN_SYMBOLS.sub("", text)


class LocalTtsUnavailableError(RuntimeError):
    """The local TTS runtime or model is unavailable."""


def resolve_tts_provider(settings: Settings) -> TtsProviderName:
    provider = getattr(settings, "tts_provider", "auto")
    api_key = getattr(settings, "tts_api_key", None)
    if provider == "auto":
        return "dashscope" if api_key is not None else "local"
    if provider == "dashscope" and api_key is None:
        raise ValueError(
            "FISH_MEMORY_TTS_API_KEY is required when "
            "FISH_MEMORY_TTS_PROVIDER=dashscope"
        )
    if provider in {"dashscope", "local"}:
        return provider
    raise ValueError(f"unsupported TTS provider: {provider}")


def voice_provider(
    voice: "FishVoiceConfig | None", settings: Settings
) -> TtsProviderName:
    """The provider a character's own voice needs; the default when it never chose.

    A cloud voice without a 通义 key falls back to the default rather than failing.
    """
    from fish_memory.tts.voice_config import VoiceKind

    if voice is not None:
        if voice.kind is VoiceKind.LOCAL:
            return "local"
        if getattr(settings, "tts_api_key", None) is not None:
            return "dashscope"
    return default_tts_provider(settings)


def model_provider(model: str, settings: Settings) -> TtsProviderName:
    """The provider that serves this voice model, whatever the default is."""
    from fish_memory.tts.catalog import get_local_tts_catalog

    if any(item.id == model for item in get_local_tts_catalog()):
        return "local"
    if getattr(settings, "tts_api_key", None) is None:
        # Without a 通义 key only the default can speak; it rejects a voice it lacks.
        return default_tts_provider(settings)
    return "dashscope"


def default_tts_provider(settings: Settings) -> TtsProviderName:
    """resolve_tts_provider, but local when 通义 was chosen without its key."""
    try:
        return resolve_tts_provider(settings)
    except ValueError:
        return "local"


@dataclass(frozen=True, slots=True)
class LocalTtsModelConfig:
    model_dir: Path
    num_threads: int = 2

    @classmethod
    def from_settings(cls, settings: Settings) -> "LocalTtsModelConfig":
        configured_dir = getattr(settings, "tts_local_model_dir", None)
        model_dir = (
            Path(configured_dir).expanduser()
            if isinstance(configured_dir, str) and configured_dir.strip()
            else Path.cwd() / ".cache" / "fish-memory" / "tts" / MATCHA_MODEL_NAME
        )
        return cls(
            model_dir=model_dir,
            num_threads=int(getattr(settings, "tts_local_num_threads", 2)),
        )

    @property
    def model_name(self) -> str:
        return MATCHA_MODEL_NAME

    @property
    def model_url(self) -> str:
        return MATCHA_MODEL_URL

    @property
    def vocoder_url(self) -> str:
        return MATCHA_VOCODER_URL

    def model_files(self) -> dict[str, Path]:
        return {
            "acoustic_model": self.model_dir / "model-steps-3.onnx",
            "vocoder": self.model_dir / MATCHA_VOCODER_NAME,
            "lexicon": self.model_dir / "lexicon.txt",
            "tokens": self.model_dir / "tokens.txt",
            "phone_zh": self.model_dir / "phone-zh.fst",
            "date_zh": self.model_dir / "date-zh.fst",
            "number_zh": self.model_dir / "number-zh.fst",
            "data_dir": self.model_dir / "espeak-ng-data",
        }

    def is_ready(self) -> bool:
        files = self.model_files()
        return all(
            path.is_dir()
            if name == "data_dir"
            else path.is_file() and path.stat().st_size > 0
            for name, path in files.items()
        )


def _download_file(
    url: str,
    destination: Path,
    on_progress: ProgressCallback,
) -> None:
    request = Request(url, headers={"User-Agent": "fish-memory/0.1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total: int | None = None
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                total = int(content_length)
            except ValueError:
                pass
        downloaded = 0
        on_progress(downloaded, total)
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            on_progress(downloaded, total)


_ENSURE_LOCK = threading.Lock()


class LocalTtsModelDownloader:
    def __init__(
        self,
        config: LocalTtsModelConfig,
        *,
        download_file: FileDownloader = _download_file,
        progress_callback: ProgressCallback | None = None,
        attempts: int = 3,
    ) -> None:
        self._config = config
        self._download_file = download_file
        self._progress_callback = progress_callback or (
            lambda _downloaded, _total: None
        )
        self._attempts = attempts

    def ensure(self) -> None:
        # Startup prefetch and a first spoken reply may both ask; one downloads.
        with _ENSURE_LOCK:
            self._ensure_once()

    def _ensure_once(self) -> None:
        reporter = ModelStatusReporter("tts", "声音")
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
            raise LocalTtsUnavailableError(
                f"本地 TTS 模型目录不完整，拒绝覆盖: {destination}"
            )
        if install_catalog(
            destination,
            MATCHA_FILES,
            progress=self._progress_callback,
            extra=lambda staging: copy_bundled_tree(
                ESPEAK_BUNDLE, staging / "espeak-ng-data", ESPEAK_TREE_SHA256
            ),
        ):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix="fish-memory-tts-", dir=destination.parent)
        )
        try:
            archive_path = temporary_root / f"{MATCHA_MODEL_NAME}.tar.bz2"
            vocoder_path = temporary_root / MATCHA_VOCODER_NAME
            self._download_with_retries(self._config.model_url, archive_path)
            self._download_with_retries(self._config.vocoder_url, vocoder_path)
            extracted_root = temporary_root / "extracted"
            extracted_root.mkdir()
            self._extract_safely(archive_path, extracted_root)
            extracted_model = extracted_root / MATCHA_MODEL_NAME
            if not extracted_model.is_dir():
                raise LocalTtsUnavailableError("本地 TTS 模型包目录结构无效")
            vocoder_path.rename(extracted_model / MATCHA_VOCODER_NAME)
            if not self._is_ready_directory(extracted_model):
                raise LocalTtsUnavailableError("下载的本地 TTS 模型缺少必要文件")
            if destination.exists():
                destination.rmdir()
            extracted_model.rename(destination)
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _download_with_retries(self, url: str, destination: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                self._download_file(url, destination, self._progress_callback)
                if destination.stat().st_size == 0:
                    raise LocalTtsUnavailableError("下载的本地 TTS 模型文件为空")
                return
            except (OSError, TimeoutError, URLError, LocalTtsUnavailableError) as error:
                last_error = error
                destination.unlink(missing_ok=True)
                if attempt + 1 < self._attempts:
                    self._progress_callback(0, None)
                    time.sleep(2**attempt)
        raise LocalTtsUnavailableError(f"下载本地 TTS 模型失败: {last_error}")

    @staticmethod
    def _extract_safely(archive_path: Path, destination: Path) -> None:
        root = destination.resolve()
        with tarfile.open(archive_path, "r:bz2") as archive:
            for member in archive.getmembers():
                target = (root / member.name).resolve()
                if target != root and root not in target.parents:
                    raise LocalTtsUnavailableError("本地 TTS 模型包包含非法路径")
            archive.extractall(root)

    def _is_ready_directory(self, directory: Path) -> bool:
        for name, path in self._config.model_files().items():
            candidate = directory / path.relative_to(self._config.model_dir)
            if name == "data_dir":
                if not candidate.is_dir():
                    return False
            elif not candidate.is_file() or candidate.stat().st_size == 0:
                return False
        return True


class PcmResampler:
    """Small stateful linear resampler for incremental mono float PCM."""

    def __init__(self, *, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        self._source_offset = 0
        self._next_position = 0.0
        self._previous: float | None = None

    def convert(self, samples: Sequence[float]) -> bytes:
        values = [float(sample) for sample in samples]
        if not values:
            return b""
        if self.source_rate == self.target_rate:
            return _float_samples_to_pcm(values)

        start = self._source_offset
        end = start + len(values)
        base = start
        points = values
        if self._previous is not None:
            base -= 1
            points = [self._previous, *values]

        ratio = self.source_rate / self.target_rate
        output: list[float] = []
        last_position = end - 1
        while self._next_position <= last_position:
            left = math.floor(self._next_position)
            fraction = self._next_position - left
            index = left - base
            if fraction and index + 1 >= len(points):
                break
            left_sample = points[index]
            right_sample = points[min(index + 1, len(points) - 1)]
            output.append(left_sample + (right_sample - left_sample) * fraction)
            self._next_position += ratio

        self._source_offset = end
        self._previous = values[-1]
        return _float_samples_to_pcm(output)


def _float_samples_to_pcm(samples: Sequence[float]) -> bytes:
    pcm = array("h")
    for sample in samples:
        clipped = max(-1.0, min(1.0, sample))
        scale = 32_768 if clipped < 0 else 32_767
        pcm.append(int(clipped * scale))
    if pcm.itemsize != 2:  # pragma: no cover - supported CPython platforms
        raise RuntimeError("16-bit PCM array is unavailable")
    return pcm.tobytes()


class LocalTtsService:
    """Downloads, loads, warms and reuses one local Matcha synthesizer."""

    def __init__(
        self,
        config: LocalTtsModelConfig,
        *,
        downloader_factory: Callable[
            [LocalTtsModelConfig, ProgressCallback], LocalTtsModelDownloader
        ]
        | None = None,
        synthesizer_factory: Callable[[LocalTtsModelConfig], Any] | None = None,
        generation_config_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.model_name = config.model_name
        self._downloader_factory = downloader_factory or (
            lambda actual_config, callback: LocalTtsModelDownloader(
                actual_config,
                progress_callback=callback,
            )
        )
        self._synthesizer_factory = synthesizer_factory or _create_synthesizer
        self._generation_config_factory = (
            generation_config_factory or _generation_config
        )
        self._prepare_lock = asyncio.Lock()
        self._synthesis_lock = asyncio.Lock()
        self._synthesizer: Any = None
        self._last_progress_log = 0.0

    @property
    def is_ready(self) -> bool:
        return self._synthesizer is not None

    async def prepare(self) -> None:
        if self._synthesizer is not None:
            return
        async with self._prepare_lock:
            if self._synthesizer is not None:
                return
            if not self.config.is_ready():
                logger.info(
                    "开始下载中英双语本地 TTS 模型: directory=%s",
                    self.config.model_dir,
                )
                downloader = self._downloader_factory(
                    self.config,
                    self._on_download_progress,
                )
                await asyncio.to_thread(downloader.ensure)
                logger.info(
                    "中英双语本地 TTS 模型下载完成: directory=%s",
                    self.config.model_dir,
                )
            self._synthesizer = await asyncio.to_thread(
                self._synthesizer_factory,
                self.config,
            )
            await asyncio.to_thread(self._warm_up)
            logger.info("中英双语本地 TTS 已就绪: model=%s", self.model_name)

    async def stream(
        self,
        text: str,
        *,
        sample_rate: int,
        stop_event: threading.Event,
    ) -> AsyncIterator[bytes]:
        await self.prepare()
        text = speakable_text(text)
        if not text.strip():
            return
        async with self._synthesis_lock:
            loop = asyncio.get_running_loop()
            audio_queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
            local_cancel = threading.Event()
            resampler = PcmResampler(
                source_rate=LOCAL_TTS_SAMPLE_RATE,
                target_rate=sample_rate,
            )

            def callback(samples: Sequence[float], _progress: float) -> int:
                if stop_event.is_set() or local_cancel.is_set():
                    return 0
                pcm = resampler.convert(samples)
                if pcm:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
                return 0 if stop_event.is_set() or local_cancel.is_set() else 1

            def generate() -> None:
                try:
                    generation_config = self._generation_config_factory()
                    self._synthesizer.generate(
                        text,
                        generation_config,
                        callback=callback,
                    )
                except BaseException as error:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, error)
                finally:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, None)

            generation_task = asyncio.create_task(asyncio.to_thread(generate))
            try:
                while True:
                    item = await audio_queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    if not stop_event.is_set():
                        yield item
            finally:
                local_cancel.set()
                await generation_task

    def _warm_up(self) -> None:
        self._synthesizer.generate(
            "你好。Hello.",
            self._generation_config_factory(),
        )

    def _on_download_progress(self, downloaded: int, total: int | None) -> None:
        now = time.monotonic()
        should_throttle = (
            downloaded
            and total
            and downloaded < total
            and now - self._last_progress_log < 1
        )
        if should_throttle:
            return
        self._last_progress_log = now
        if total:
            logger.info(
                "本地 TTS 模型下载进度: %.1f%% (%.1f MB/%.1f MB)",
                downloaded * 100 / total,
                downloaded / 1_000_000,
                total / 1_000_000,
            )
        else:
            logger.info("本地 TTS 模型下载进度: %.1f MB", downloaded / 1_000_000)


def _generation_config() -> Any:
    try:
        sherpa_onnx = import_module("sherpa_onnx")
    except Exception as error:  # pragma: no cover - optional host library
        raise LocalTtsUnavailableError(
            "本地 TTS 需要 voice 依赖，请运行 uv sync --extra voice"
        ) from error
    config = sherpa_onnx.GenerationConfig()
    config.num_steps = 3
    return config


def _create_synthesizer(config: LocalTtsModelConfig) -> Any:
    try:
        sherpa_onnx = import_module("sherpa_onnx")
    except Exception as error:  # pragma: no cover - optional host library
        raise LocalTtsUnavailableError(
            "本地 TTS 需要 voice 依赖，请运行 uv sync --extra voice"
        ) from error
    files = config.model_files()
    tts_config = sherpa_onnx.OfflineTtsConfig()
    tts_config.model.matcha.acoustic_model = str(files["acoustic_model"])
    tts_config.model.matcha.vocoder = str(files["vocoder"])
    tts_config.model.matcha.lexicon = str(files["lexicon"])
    tts_config.model.matcha.tokens = str(files["tokens"])
    tts_config.model.matcha.data_dir = str(files["data_dir"])
    tts_config.model.num_threads = config.num_threads
    tts_config.model.provider = "cpu"
    tts_config.rule_fsts = ",".join(
        str(files[name]) for name in ("phone_zh", "date_zh", "number_zh")
    )
    tts_config.max_num_sentences = 1
    if not tts_config.validate():
        raise LocalTtsUnavailableError("本地 TTS 模型配置无效")
    return sherpa_onnx.OfflineTts(tts_config)


async def prefetch_local_tts_model(settings: Settings, *, force: bool = False) -> None:
    """Download the local voice in the background so the first reply can speak.

    `force`: a character just chose the local voice, whatever the default is.
    """
    try:
        if not force and default_tts_provider(settings) != "local":
            return
        downloader = LocalTtsModelDownloader(
            LocalTtsModelConfig.from_settings(settings)
        )
        await asyncio.to_thread(downloader.ensure)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning("后台准备本地 TTS 模型失败，将在开口时重试: %s", error)
