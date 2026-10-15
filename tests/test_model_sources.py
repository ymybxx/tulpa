import hashlib
import json
from pathlib import Path, PureWindowsPath

import pytest

from fish_memory import model_sources
from fish_memory.model_sources import (
    ModelDownloadError,
    ModelFile,
    ModelStatusReporter,
    copy_bundled_tree,
    fetch_file,
    hf_urls,
    install_catalog,
    install_hub_snapshot,
    tree_digest,
)

BODY = b"model bytes"
DIGEST = hashlib.sha256(BODY).hexdigest()


def serving(responses: dict[str, bytes]):
    calls: list[str] = []

    def download(url: str, destination: Path, on_bytes) -> None:
        calls.append(url)
        if url not in responses:
            raise OSError("unreachable")
        destination.write_bytes(responses[url])
        on_bytes(len(responses[url]), len(responses[url]))

    return download, calls


def test_mirror_comes_first_and_origin_is_the_fallback(monkeypatch):
    monkeypatch.delenv(model_sources.HF_ENDPOINTS_ENV, raising=False)
    assert hf_urls("owner/repo", "abc", "model.onnx") == (
        "https://hf-mirror.com/owner/repo/resolve/abc/model.onnx",
        "https://huggingface.co/owner/repo/resolve/abc/model.onnx",
    )


def test_fetch_skips_unreachable_and_tampered_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(model_sources.time, "sleep", lambda _seconds: None)
    download, calls = serving({"https://tampered/x": b"evil", "https://origin/x": BODY})
    file = ModelFile(
        "model.onnx",
        DIGEST,
        len(BODY),
        ("https://down/x", "https://tampered/x", "https://origin/x"),
    )
    assert (
        fetch_file(file, tmp_path / "model.onnx", download=download)
        == "https://origin/x"
    )
    assert (tmp_path / "model.onnx").read_bytes() == BODY
    assert calls.count("https://down/x") == 2
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".")]
    assert fetch_file(file, tmp_path / "model.onnx", download=download) == "existing"


def test_fetch_prefers_the_verified_bundle_and_reports_every_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(model_sources.time, "sleep", lambda _seconds: None)
    bundle = tmp_path / "bundle"
    (bundle / "kws").mkdir(parents=True)
    (bundle / "kws" / "model.onnx").write_bytes(BODY)
    monkeypatch.setenv(model_sources.BUNDLED_MODELS_ENV, str(bundle))
    download, calls = serving({})
    bundled = ModelFile(
        "model.onnx", DIGEST, len(BODY), ("https://origin/x",), bundled="kws/model.onnx"
    )
    assert (
        fetch_file(bundled, tmp_path / "out" / "model.onnx", download=download)
        == "bundle"
    )
    assert calls == []
    missing = ModelFile("other.onnx", DIGEST, len(BODY), ("https://origin/y",))
    with pytest.raises(ModelDownloadError, match="other.onnx"):
        fetch_file(missing, tmp_path / "out" / "other.onnx", download=download)


def test_catalog_installs_all_or_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_sources.time, "sleep", lambda _seconds: None)
    good = ModelFile("a.onnx", DIGEST, len(BODY), ("https://origin/a",))
    bad = ModelFile("b.onnx", DIGEST, len(BODY), ("https://origin/b",))
    download, _calls = serving({"https://origin/a": BODY})
    monkeypatch.setattr(model_sources, "_stream", download)
    destination = tmp_path / "model"
    assert not install_catalog(destination, (good, bad))
    assert not destination.exists()
    assert not list(tmp_path.iterdir())
    assert install_catalog(destination, (good,))
    assert (destination / "a.onnx").read_bytes() == BODY


def test_hub_snapshot_is_filled_only_when_every_file_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(model_sources.time, "sleep", lambda _seconds: None)
    nested = ModelFile("onnx/a.onnx", DIGEST, len(BODY), ("https://origin/a",))
    config = ModelFile("config.json", DIGEST, len(BODY), ("https://origin/c",))
    download, _calls = serving({"https://origin/a": BODY})
    monkeypatch.setattr(model_sources, "_stream", download)
    repo = tmp_path / "models--owner--name"
    with pytest.raises(ModelDownloadError):
        install_hub_snapshot("owner/name", "abc", (nested, config), cache_dir=tmp_path)
    assert not (repo / "snapshots").exists()
    download, calls = serving({"https://origin/a": BODY, "https://origin/c": BODY})
    monkeypatch.setattr(model_sources, "_stream", download)
    snapshot = install_hub_snapshot(
        "owner/name", "abc", (nested, config), cache_dir=tmp_path
    )
    assert snapshot == repo / "snapshots" / "abc"
    assert (snapshot / "onnx" / "a.onnx").read_bytes() == BODY
    assert (snapshot / "config.json").read_bytes() == BODY
    assert [path.name for path in repo.iterdir()] == ["snapshots"]
    calls.clear()
    install_hub_snapshot("owner/name", "abc", (nested, config), cache_dir=tmp_path)
    assert calls == []


def test_first_run_downloads_try_modelscope_first():
    from fish_memory import model_catalog as catalog

    downloaded = [
        *catalog.SENSEVOICE_FILES,
        *(file for file in catalog.MATCHA_FILES if file.bundled is None),
        *catalog.SMART_TURN_FILES,
        *catalog.E5_SMALL_FILES,
    ]
    for file in downloaded:
        assert file.urls[0].startswith("https://www.modelscope.cn/models/"), file.name
    assert catalog.segmentation_urls()[0].startswith("https://www.modelscope.cn/")
    assert all(file.bundled for file in catalog.SENSEVOICE_FILES)


@pytest.mark.parametrize("windows_path_order", [False, True])
def test_tree_digest_preserves_case_sensitive_component_order(
    tmp_path, monkeypatch, windows_path_order
):
    # Match the existing catalog's POSIX order, including a/z before a-/voice.
    # Emulate WindowsPath ordering so this regression also runs on macOS/Linux.
    if windows_path_order:
        monkeypatch.setattr(
            type(tmp_path),
            "__lt__",
            lambda left, right: (
                PureWindowsPath(str(left)) < PureWindowsPath(str(right))
            ),
        )
    names = [
        "z",
        "voices/!v/anika",
        "voices/!v/adam",
        "voices/!v/Alex",
        "a-/voice",
        "a/z",
    ]
    for name in names:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(name.encode())
    assert tree_digest(tmp_path) == (
        "3d60c71a701fa4e2ab2b5169bebb6be0c1f31da4332a289b0895bb56af493605"
    )


def test_bundled_tree_is_copied_only_when_it_matches(tmp_path, monkeypatch):
    source = tmp_path / "bundle" / "tts" / "espeak-ng-data"
    (source / "voices").mkdir(parents=True)
    (source / "phontab").write_bytes(b"a")
    (source / "voices" / "zh").write_bytes(b"b")
    monkeypatch.setenv(model_sources.BUNDLED_MODELS_ENV, str(tmp_path / "bundle"))
    digest = tree_digest(source)
    assert not copy_bundled_tree("tts/espeak-ng-data", tmp_path / "out", "0" * 64)
    assert copy_bundled_tree("tts/espeak-ng-data", tmp_path / "out", digest)
    assert (tmp_path / "out" / "voices" / "zh").read_bytes() == b"b"


def test_status_is_written_for_the_desktop_app(tmp_path):
    reporter = ModelStatusReporter("tts", "语音合成", directory=tmp_path)
    reporter.downloading(10, 100)
    reporter.downloading(20, 100)  # throttled
    first = json.loads((tmp_path / "tts.json").read_text(encoding="utf-8"))
    assert (
        first["state"] == "downloading"
        and first["done"] == 10
        and first["label"] == "语音合成"
    )
    reporter.failed("网络不通")
    assert (
        json.loads((tmp_path / "tts.json").read_text(encoding="utf-8"))["error"]
        == "网络不通"
    )
