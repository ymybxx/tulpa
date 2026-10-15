"""Where local speech and memory models come from, and how far along they are.

Each file is taken from the app bundle when present, then from ModelScope
(served inside mainland China), then from the mainland mirror of Hugging Face,
then from its original host. Every download is checked against a pinned size
and SHA-256, so a mirror can only serve the exact bytes of the original.
Progress is written to small JSON files that the desktop app reads to tell the
user what is still being prepared.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

HF_ENDPOINTS_ENV = "FISH_MEMORY_HF_ENDPOINTS"
BUNDLED_MODELS_ENV = "FISH_MEMORY_BUNDLED_MODELS_DIR"
MODEL_STATUS_ENV = "FISH_MEMORY_MODEL_STATUS_DIR"
DEFAULT_HF_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")
# hf-mirror hands large files to an overseas CDN, which crawls without a proxy;
# ModelScope serves them from inside mainland China.
MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"

ProgressCallback = Callable[[int, int | None], None]


class ModelDownloadError(RuntimeError):
    pass


def hf_endpoints() -> tuple[str, ...]:
    configured = os.getenv(HF_ENDPOINTS_ENV, "")
    values = tuple(
        value.strip().rstrip("/") for value in configured.split(",") if value.strip()
    )
    return values or DEFAULT_HF_ENDPOINTS


def hf_urls(repo: str, revision: str, path: str) -> tuple[str, ...]:
    """The file at a pinned commit, from every configured Hugging Face endpoint."""
    return tuple(
        f"{endpoint}/{repo}/resolve/{revision}/{path}" for endpoint in hf_endpoints()
    )


def modelscope_urls(repo: str, path: str) -> tuple[str, ...]:
    """The file on ModelScope; its branch moves, so only the pinned SHA-256 decides."""
    return (f"{MODELSCOPE_ENDPOINT}/models/{repo}/resolve/master/{path}",)


def bundled_models_dir() -> Path | None:
    value = os.getenv(BUNDLED_MODELS_ENV, "").strip()
    return Path(value).expanduser() if value else None


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One model file: its name in the model directory and where it may come from."""

    name: str
    sha256: str
    size: int | None = None
    urls: tuple[str, ...] = ()
    bundled: str | None = None


def file_matches(path: Path, sha256: str, size: int | None = None) -> bool:
    try:
        if size is not None and path.stat().st_size != size:
            return False
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest() == sha256
    except OSError:
        return False


def _stream(
    url: str, destination: Path, on_bytes: Callable[[int, int | None], None]
) -> None:
    request = Request(url, headers={"User-Agent": "fish-memory/0.1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total: int | None = None
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                total = int(length)
            except ValueError:
                total = None
        downloaded = 0
        on_bytes(0, total)
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            on_bytes(downloaded, total)


Downloader = Callable[[str, Path, Callable[[int, int | None], None]], None]


def fetch_file(
    file: ModelFile,
    destination: Path,
    *,
    progress: ProgressCallback | None = None,
    attempts: int = 2,
    download: Downloader | None = None,
) -> str:
    """Place one verified file at `destination`; returns where it came from."""
    report = progress or (lambda _done, _total: None)
    download = download or _stream
    if file_matches(destination, file.sha256, file.size):
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = bundled_models_dir()
    if bundle is not None and file.bundled is not None:
        source = bundle / file.bundled
        if file_matches(source, file.sha256, file.size):
            temporary = destination.with_name(f".{destination.name}.bundled")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
            report(file.size or source.stat().st_size, file.size)
            return "bundle"
    errors: list[str] = []
    for url in file.urls:
        for attempt in range(attempts):
            handle, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            os.close(handle)
            temporary = Path(name)
            try:
                download(url, temporary, report)
                if not file_matches(temporary, file.sha256, file.size):
                    raise ModelDownloadError("校验失败")
                temporary.replace(destination)
                return url
            except (OSError, TimeoutError, URLError, ModelDownloadError) as error:
                errors.append(f"{_host(url)}: {error}")
                if attempt + 1 < attempts:
                    time.sleep(_retry_delay(attempt))
            finally:
                temporary.unlink(missing_ok=True)
        logger.warning("模型下载源不可用，尝试下一个: %s", _host(url))
    raise ModelDownloadError(
        f"{file.name} 下载失败（{'；'.join(errors[-3:]) or '没有可用的下载源'}）"
    )


def _bundled_copy(file: ModelFile) -> Path | None:
    bundle = bundled_models_dir()
    if bundle is None or file.bundled is None:
        return None
    source = bundle / file.bundled
    return source if file_matches(source, file.sha256, file.size) else None


def fetch_files(
    files: Sequence[ModelFile],
    directory: Path,
    *,
    progress: ProgressCallback | None = None,
    download: Downloader | None = None,
) -> None:
    """Fetch several files; progress counts only what has to be downloaded."""
    report = progress or (lambda _done, _total: None)
    remote = [
        file
        for file in files
        if not file_matches(directory / file.name, file.sha256, file.size)
        and _bundled_copy(file) is None
    ]
    sizes = [file.size for file in remote]
    total = (
        sum(size for size in sizes if size is not None)
        if all(size is not None for size in sizes)
        else None
    )
    done = 0
    for file in files:
        counted = file in remote
        base = done

        def on_bytes(
            downloaded: int,
            _total: int | None,
            base: int = base,
            counted: bool = counted,
        ) -> None:
            if counted:
                report(base + downloaded, total)

        fetch_file(file, directory / file.name, progress=on_bytes, download=download)
        if counted:
            done += file.size or 0
    report(done, total)


def tree_digest(root: Path) -> str:
    """One SHA-256 over every file's relative path and SHA-256, in sorted order."""
    digest = hashlib.sha256()
    # WindowsPath compares names case-insensitively. Preserve the catalog's
    # original POSIX component ordering on every host, including mixed-case
    # espeak voices and directory names that prefix another directory name.
    files = (item for item in root.rglob("*") if item.is_file())
    for path in sorted(files, key=lambda item: item.relative_to(root).parts):
        with path.open("rb") as source:
            file_digest = hashlib.file_digest(source, "sha256").hexdigest()
        digest.update(f"{path.relative_to(root).as_posix()}\0{file_digest}\n".encode())
    return digest.hexdigest()


def copy_bundled_tree(name: str, destination: Path, digest: str) -> bool:
    """Copy a verified directory from the app bundle; False when it is not bundled."""
    bundle = bundled_models_dir()
    source = bundle / name if bundle is not None else None
    if source is None or not source.is_dir() or tree_digest(source) != digest:
        return False
    temporary = destination.with_name(f".{destination.name}.bundled")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.copytree(source, temporary)
    shutil.rmtree(destination, ignore_errors=True)
    temporary.replace(destination)
    return True


def install_catalog(
    destination: Path,
    files: Sequence[ModelFile],
    *,
    progress: ProgressCallback | None = None,
    extra: Callable[[Path], bool] | None = None,
) -> bool:
    """Assemble a model directory from the bundle or mirrors.

    Returns False (after logging) when any file is unavailable, so the caller
    can fall back to the original release archive.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        fetch_files(files, staging, progress=progress)
        if extra is not None and not extra(staging):
            return False
        if destination.exists():
            destination.rmdir()
        staging.rename(destination)
        return True
    except (ModelDownloadError, OSError) as error:
        logger.warning(
            "从内置资源或镜像准备 %s 失败，改用原始下载: %s", destination.name, error
        )
        return False
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_hub_snapshot(
    repo: str,
    revision: str,
    files: Sequence[ModelFile],
    *,
    progress: ProgressCallback | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Place verified files into the Hugging Face cache at a pinned commit.

    Afterwards the usual offline Hub lookups for that commit find them, so the
    rest of the code keeps loading models through huggingface_hub. Files are
    staged first; a failed download never leaves a half-filled snapshot.
    """
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    root = cache_dir / f"models--{repo.replace('/', '--')}"
    snapshot = root / "snapshots" / revision
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".download-", dir=root))
    try:
        for file in files:
            existing = snapshot / file.name
            if file_matches(existing, file.sha256, file.size):
                (staging / file.name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(existing, staging / file.name)
        fetch_files(files, staging, progress=progress)
        for file in files:
            target = snapshot / file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / file.name).replace(target)
        return snapshot
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _retry_delay(attempt: int) -> float:
    return float(2**attempt)


def _host(url: str) -> str:
    return url.split("/")[2] if "//" in url else url


# ---------- status the desktop app shows ----------


class ModelStatusReporter:
    """Writes `<component>.json` for the desktop app; throttled while downloading."""

    def __init__(
        self, component: str, label: str, *, directory: Path | None = None
    ) -> None:
        self.component = component
        self.label = label
        configured = os.getenv(MODEL_STATUS_ENV, "").strip()
        self._directory = (
            directory
            if directory is not None
            else (Path(configured) if configured else None)
        )
        self._last_write = 0.0
        self._last_state = ""

    def downloading(self, done: int, total: int | None) -> None:
        now = time.monotonic()
        if self._last_state == "downloading" and now - self._last_write < 0.5:
            return
        self._write("downloading", done=done, total=total)

    def preparing(self) -> None:
        self._write("preparing")

    def ready(self) -> None:
        self._write("ready")

    def failed(self, error: str) -> None:
        self._write("failed", error=error)

    def _write(self, state: str, **fields: object) -> None:
        self._last_state = state
        self._last_write = time.monotonic()
        if self._directory is None:
            return
        payload = {
            "component": self.component,
            "label": self.label,
            "state": state,
            "at": time.time(),
            **fields,
        }
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            target = self._directory / f"{self.component}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(target)
        except OSError:
            logger.debug("无法写入模型状态", exc_info=True)
