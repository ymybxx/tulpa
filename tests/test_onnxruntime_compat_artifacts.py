from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

HELPER_PATH = (
    Path(__file__).parents[1] / "desktop/pet/scripts/onnxruntime-compat-artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("ort_compat_artifacts", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


@pytest.fixture
def source_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    source = tmp_path / "source"
    headers = source / "include/onnxruntime/core/session"
    headers.mkdir(parents=True)
    (headers / "onnxruntime_cxx_api.h").write_text("header\n")
    (source / "LICENSE").write_text("MIT\n")
    (source / "ThirdPartyNotices.txt").write_text("notices\n")
    release = tmp_path / "build/Release"
    (release / "dist").mkdir(parents=True)
    (release / "libonnxruntime.1.28.0.dylib").write_bytes(b"native library")
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel = release / "dist" / f"onnxruntime-1.28.0-{abi}-{abi}-macosx_12_0_arm64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "onnxruntime-1.28.0.dist-info/METADATA",
            "Name: onnxruntime\nVersion: 1.28.0\n",
        )
    script = tmp_path / "build.sh"
    script.write_text("build recipe\n")
    monkeypatch.setattr(
        HELPER.subprocess, "check_output", lambda *args, **kwargs: HELPER.COMMIT
    )

    def check_clean(command: list[str]) -> None:
        assert command[-4:] == ["diff", "--quiet", "HEAD", "--"]

    monkeypatch.setattr(HELPER.subprocess, "check_call", check_clean)
    monkeypatch.setattr(
        HELPER.AUDITOR, "audit_wheel", lambda path: [{"minimumMacOS": "12.3"}]
    )
    return tmp_path, script


def test_publish_keeps_matching_wheel_sdk_and_licenses(
    source_build: tuple[Path, Path],
) -> None:
    root, script = source_build
    HELPER.publish(root, script)
    HELPER.verify(root, script)
    assert len(list((root / "dist").glob("*.whl"))) == 1
    assert (root / "sdk/include/onnxruntime_cxx_api.h").read_text() == "header\n"
    assert (root / "sdk/LICENSE").read_text() == "MIT\n"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["sourceCommit"] == HELPER.COMMIT
    assert "sdk/lib/libonnxruntime.1.28.0.dylib" in manifest["files"]


@pytest.mark.parametrize(
    "changed",
    ["sdk/lib/libonnxruntime.1.28.0.dylib", "sdk/include/onnxruntime_cxx_api.h"],
)
def test_verify_rejects_changed_sdk(
    source_build: tuple[Path, Path], changed: str
) -> None:
    root, script = source_build
    HELPER.publish(root, script)
    (root / changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        HELPER.verify(root, script)


def test_verify_rejects_changed_recipe(source_build: tuple[Path, Path]) -> None:
    root, script = source_build
    HELPER.publish(root, script)
    script.write_text("different recipe")
    with pytest.raises(ValueError, match="recipe, source, or Python ABI"):
        HELPER.verify(root, script)


def test_publish_rejects_other_source_commit(
    source_build: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, script = source_build
    monkeypatch.setattr(
        HELPER.subprocess, "check_output", lambda *args, **kwargs: "other-commit"
    )
    with pytest.raises(ValueError, match="Source commit differs"):
        HELPER.publish(root, script)


def test_metadata_rejects_downgrade_or_other_abi(tmp_path: Path) -> None:
    for name in (
        "onnxruntime-1.19.2-cp311-cp311-macosx_11_0_arm64.whl",
        "onnxruntime-1.28.0-cp999-cp999-macosx_12_0_arm64.whl",
    ):
        with pytest.raises(ValueError, match="version or Python ABI differs"):
            HELPER.wheel_metadata(tmp_path / name)
