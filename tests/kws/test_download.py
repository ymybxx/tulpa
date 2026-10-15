import tarfile
from pathlib import Path

import pytest

from fish_memory.kws.download import KwsModelDownloader, KwsModelDownloadError
from fish_memory.kws.models import KwsModelConfig


def _write_model_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in KwsModelConfig(model_dir=directory).model_files().values():
        path.write_text("model\n", encoding="utf-8")


def test_downloader_promotes_verified_archive_atomically(tmp_path: Path) -> None:
    model_name = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
    model_dir = tmp_path / "cache" / model_name
    config = KwsModelConfig(model_dir=model_dir)

    def fake_download(_url: str, archive_path: Path, _on_progress) -> None:
        source_dir = tmp_path / "source" / model_name
        _write_model_files(source_dir)
        with tarfile.open(archive_path, "w:bz2") as archive:
            archive.add(source_dir, arcname=model_name)

    downloader = KwsModelDownloader(config, download_archive=fake_download)

    downloader.ensure()

    assert config.is_ready()
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_downloader_does_not_replace_incomplete_destination(tmp_path: Path) -> None:
    model_name = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
    model_dir = tmp_path / model_name
    model_dir.mkdir()
    (model_dir / "partial-file").write_text("partial", encoding="utf-8")
    config = KwsModelConfig(model_dir=model_dir)

    def fake_download(_url: str, archive_path: Path, _on_progress) -> None:
        source_dir = tmp_path / "source" / model_name
        _write_model_files(source_dir)
        with tarfile.open(archive_path, "w:bz2") as archive:
            archive.add(source_dir, arcname=model_name)

    downloader = KwsModelDownloader(config, download_archive=fake_download)

    with pytest.raises(KwsModelDownloadError, match="不完整"):
        downloader.ensure()

    assert (model_dir / "partial-file").exists()
