from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import sysconfig
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

HELPER_PATH = Path(__file__).parents[1] / "desktop/pet/scripts/sherpa-compat-wheels.py"
SPEC = importlib.util.spec_from_file_location("sherpa_compat_wheels", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


@pytest.fixture
def source_build(tmp_path: Path) -> tuple[Path, Path, Path]:
    script = tmp_path / "build.sh"
    script.write_text("recipe v1\n")
    ort = tmp_path / "ort"
    ort.mkdir()
    (ort / "LICENSE").write_text("MIT license\n")
    (ort / "ThirdPartyNotices.txt").write_text("ORT notices\n")
    (ort / "lib").mkdir()
    (ort / "lib" / "libonnxruntime.1.28.0.dylib").write_bytes(b"ORT")
    source = tmp_path / "source.whl"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("sherpa_onnx/__init__.py", "# upstream package\n")
        archive.writestr(
            f"sherpa_onnx/lib/_sherpa_onnx{sysconfig.get_config_var('EXT_SUFFIX')}",
            b"extension",
        )
        archive.writestr("sherpa_onnx/lib/libonnxruntime.1.28.0.dylib", b"ORT")
        archive.writestr("sherpa_onnx/include/sherpa-onnx/c-api/c-api.h", "header")
        archive.writestr(
            f"{HELPER.MAIN_INFO}/METADATA",
            "Metadata-Version: 2.4\nName: sherpa_onnx\nVersion: 1.13.4\n"
            "License-File: LICENSE\n\nUpstream description.\n",
        )
        archive.writestr(f"{HELPER.MAIN_INFO}/licenses/LICENSE", "Apache license")
        archive.writestr(f"{HELPER.MAIN_INFO}/RECORD", "old record")
        archive.writestr(f"{HELPER.MAIN_INFO}/WHEEL", "old wheel tag")
    return source, script, ort


def test_split_preserves_version_license_and_rebuilds_records(
    source_build: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, script, ort = source_build
    dist = source.parent / "dist"
    monkeypatch.setattr(HELPER, "audit_wheel", lambda path: [{"minimumMacOS": "12.3"}])
    HELPER.split(source, dist, script, ort)
    wheels = sorted(dist.glob("*.whl"))
    assert len(wheels) == 2
    for wheel in wheels:
        core = wheel.name.startswith("sherpa_onnx_core-")
        info = HELPER.CORE_INFO if core else HELPER.MAIN_INFO
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            metadata = Parser().parsestr(archive.read(f"{info}/METADATA").decode())
            assert metadata["Name"] == ("sherpa-onnx-core" if core else "sherpa_onnx")
            assert metadata["Version"] == "1.13.4"
            assert metadata.get_all("Requires-Dist", []) == (
                [] if core else ["sherpa-onnx-core==1.13.4"]
            )
            assert f"{info}/licenses/LICENSE" in names
            assert ("sherpa_onnx/lib/libonnxruntime.1.28.0.dylib" in names) == core
            assert ("sherpa_onnx/__init__.py" in names) != core
            if core:
                assert f"{info}/licenses/onnxruntime-LICENSE" in names
                assert f"{info}/licenses/onnxruntime-ThirdPartyNotices.txt" in names
            for license_path in metadata.get_all("License-File", []):
                assert f"{info}/licenses/{license_path}" in names
            rows = list(
                csv.reader(io.StringIO(archive.read(f"{info}/RECORD").decode()))
            )
            assert {row[0] for row in rows} == names
            for name, digest, length in rows:
                if name.endswith("/RECORD"):
                    assert (digest, length) == ("", "")
                    continue
                data = archive.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                assert digest == "sha256=" + expected.rstrip(b"=").decode()
                assert int(length) == len(data)
    HELPER.verify(dist, script)
    script.write_text("changed recipe\n")
    with pytest.raises(ValueError, match="recipe or Python ABI"):
        HELPER.verify(dist, script)


def test_verify_rejects_corrupt_wheel(
    source_build: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, script, ort = source_build
    dist = source.parent / "dist"
    monkeypatch.setattr(HELPER, "audit_wheel", lambda path: [{"minimumMacOS": "12.3"}])
    HELPER.split(source, dist, script, ort)
    wheel = next(dist.glob("*.whl"))
    wheel.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="validation failed"):
        HELPER.verify(dist, script)


def test_verify_rejects_wrong_abi(
    source_build: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, script, ort = source_build
    dist = source.parent / "dist"
    monkeypatch.setattr(HELPER, "audit_wheel", lambda path: [{"minimumMacOS": "12.3"}])
    HELPER.split(source, dist, script, ort)
    manifest_path = dist / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pythonABI"] = "cpython-314-darwin"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="recipe or Python ABI"):
        HELPER.verify(dist, script)


def test_verify_rejects_changed_runtime_sdk(
    source_build: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, script, ort = source_build
    dist = source.parent / "dist"
    monkeypatch.setattr(HELPER, "audit_wheel", lambda path: [{"minimumMacOS": "12.3"}])
    HELPER.split(source, dist, script, ort)
    HELPER.verify(dist, script, ort)
    (ort / "lib" / "libonnxruntime.1.28.0.dylib").write_bytes(b"different build")
    with pytest.raises(ValueError, match="SDK changed"):
        HELPER.verify(dist, script, ort)


@pytest.mark.parametrize("minimum", ["11.0", "12.3", "12.3.0"])
def test_native_audit_accepts_compatible_three_part_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, minimum: str
) -> None:
    wheel = tmp_path / "native.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lib.dylib", b"\xcf\xfa\xed\xfe" + b"binary")

    def check_output(command: list[str], *, text: bool) -> str:
        if command[0] == "xcrun":
            assert "arm64" in command
            return f"platform MACOS\nminos {minimum}\n"
        return "lib.dylib:\n\t@rpath/libonnxruntime.1.28.0.dylib (version)\n"

    monkeypatch.setattr(HELPER.subprocess, "check_output", check_output)
    assert HELPER.audit_wheel(wheel) == [{"path": "lib.dylib", "minimumMacOS": minimum}]


def test_native_audit_rejects_new_os_or_other_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheel = tmp_path / "native.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lib.dylib", b"\xcf\xfa\xed\xfe" + b"binary")
    monkeypatch.setattr(
        HELPER.subprocess,
        "check_output",
        lambda *args, **kwargs: "platform MACOS\nminos 14.0\n",
    )
    with pytest.raises(ValueError, match="exceeds 12.3"):
        HELPER.audit_wheel(wheel)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("_sherpa.cpython-314-darwin.so", b"\xcf\xfa\xed\xfe")
    with pytest.raises(ValueError, match="Unexpected Python ABI"):
        HELPER.audit_wheel(wheel)
