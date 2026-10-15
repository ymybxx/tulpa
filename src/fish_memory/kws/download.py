import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from fish_memory.kws.models import KwsModelConfig
from fish_memory.model_catalog import kws_files
from fish_memory.model_sources import ModelStatusReporter, install_catalog

ProgressCallback = Callable[[int, int | None], None]
ArchiveDownloader = Callable[[str, Path, ProgressCallback], None]


class KwsModelDownloadError(RuntimeError):
    """The local KWS model could not be downloaded or installed."""


def _download_archive(
    url: str,
    destination: Path,
    on_progress: ProgressCallback,
) -> None:
    request = Request(url, headers={"User-Agent": "fish-memory/0.1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        content_length = response.headers.get("Content-Length")
        total_bytes: int | None = None
        if content_length is not None:
            try:
                total_bytes = int(content_length)
            except ValueError:
                total_bytes = None
        downloaded_bytes = 0
        on_progress(downloaded_bytes, total_bytes)
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded_bytes += len(chunk)
            on_progress(downloaded_bytes, total_bytes)


class KwsModelDownloader:
    def __init__(
        self,
        config: KwsModelConfig,
        *,
        download_archive: ArchiveDownloader = _download_archive,
        attempts: int = 3,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._config = config
        self._download_archive = download_archive
        self._attempts = attempts
        self._progress_callback = progress_callback or (
            lambda _downloaded, _total: None
        )

    def ensure(self) -> None:
        reporter = ModelStatusReporter("kws", "唤醒词")
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
            raise KwsModelDownloadError(f"KWS 模型目录不完整，拒绝覆盖: {destination}")
        # Bundled with the app; the release archive stays as the fallback.
        if install_catalog(
            destination,
            kws_files(self._config.chunk_size),
            progress=self._progress_callback,
        ):
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix="fish-memory-kws-", dir=destination.parent)
        )
        try:
            archive_path = temporary_root / f"{self._config.model_name}.tar.bz2"
            self._download_with_retries(archive_path)
            extracted_root = temporary_root / "extracted"
            extracted_root.mkdir()
            self._extract_safely(archive_path, extracted_root)
            extracted_model = extracted_root / self._config.model_name
            if not self._is_ready_directory(extracted_model):
                raise KwsModelDownloadError("下载的 KWS 模型缺少必要文件")

            if destination.exists():
                destination.rmdir()
            extracted_model.rename(destination)
        except KwsModelDownloadError:
            raise
        except Exception as error:
            raise KwsModelDownloadError(f"KWS 模型下载失败: {error}") from error
        finally:
            shutil.rmtree(temporary_root, ignore_errors=True)

    def _download_with_retries(self, archive_path: Path) -> None:
        last_error: Exception | None = None
        for attempt in range(self._attempts):
            try:
                self._download_archive(
                    self._config.model_url,
                    archive_path,
                    self._progress_callback,
                )
                if archive_path.stat().st_size == 0:
                    raise KwsModelDownloadError("下载的 KWS 模型包为空")
                return
            except (KwsModelDownloadError, OSError, URLError, TimeoutError) as error:
                last_error = error
                archive_path.unlink(missing_ok=True)
                if attempt + 1 < self._attempts:
                    self._progress_callback(0, None)
                    time.sleep(2**attempt)
        raise KwsModelDownloadError(f"下载 KWS 模型包失败: {last_error}")

    @staticmethod
    def _extract_safely(archive_path: Path, destination: Path) -> None:
        root = destination.resolve()
        with tarfile.open(archive_path, "r:bz2") as archive:
            for member in archive.getmembers():
                target = (root / member.name).resolve()
                if target != root and root not in target.parents:
                    raise KwsModelDownloadError("KWS 模型包包含非法路径")
            archive.extractall(root)

    def _is_ready_directory(self, directory: Path) -> bool:
        for path in self._config.model_files().values():
            candidate = directory / path.relative_to(self._config.model_dir)
            if not candidate.is_file() or candidate.stat().st_size == 0:
                return False
        return True
