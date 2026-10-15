"""Publish and verify a same-source CPU ONNX Runtime desktop build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

VERSION = "1.28.0"
COMMIT = "45de2a8b06d62989b3ab55ba7dc58a27ca83f9fc"
AUDITOR_PATH = Path(__file__).with_name("sherpa-compat-wheels.py")
SPEC = importlib.util.spec_from_file_location("sherpa_compat_wheels", AUDITOR_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe(script: Path) -> str:
    return hashlib.sha256(
        script.read_bytes() + Path(__file__).read_bytes() + AUDITOR_PATH.read_bytes()
    ).hexdigest()


def wheel_metadata(path: Path) -> None:
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if not path.name.startswith(f"onnxruntime-{VERSION}-{abi}-{abi}-"):
        raise ValueError("ONNX Runtime wheel version or Python ABI differs")
    with zipfile.ZipFile(path) as archive:
        metadata = Parser().parsestr(
            archive.read(f"onnxruntime-{VERSION}.dist-info/METADATA").decode()
        )
        if metadata["Name"] != "onnxruntime" or metadata["Version"] != VERSION:
            raise ValueError("Unexpected ONNX Runtime wheel metadata")


def audit_sdk(sdk: Path) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory() as tmp:
        native_zip = Path(tmp) / "sdk.zip"
        with zipfile.ZipFile(native_zip, "w") as archive:
            for library in sorted((sdk / "lib").glob("*.dylib")):
                archive.write(library, library.relative_to(sdk).as_posix())
        return AUDITOR.audit_wheel(native_zip)


def publish(root: Path, script: Path) -> None:
    source = root / "source"
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != COMMIT:
        raise ValueError("Source commit differs from the installed 1.28.0 runtime")
    subprocess.check_call(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"])
    release = root / "build/Release"
    wheels = list((release / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one ONNX Runtime Python wheel")
    wheel_metadata(wheels[0])
    wheel_audit = AUDITOR.audit_wheel(wheels[0])
    stage = Path(tempfile.mkdtemp(prefix="publish.", dir=root))
    (stage / "dist").mkdir()
    sdk = stage / "sdk"
    (sdk / "include").mkdir(parents=True)
    (sdk / "lib").mkdir()
    shutil.copy2(wheels[0], stage / "dist" / wheels[0].name)
    for header in (source / "include/onnxruntime/core/session").iterdir():
        if header.suffix in {".h", ".inc"}:
            shutil.copy2(header, sdk / "include" / header.name)
    for library in release.glob("libonnxruntime*.dylib"):
        shutil.copy2(library, sdk / "lib" / library.name)
    if not (sdk / "lib" / f"libonnxruntime.{VERSION}.dylib").is_file():
        raise ValueError("The matching C++ shared library was not built")
    for name in ("LICENSE", "ThirdPartyNotices.txt"):
        shutil.copy2(source / name, sdk / name)
    sdk_audit = audit_sdk(sdk)
    manifest = {
        "recipe": recipe(script),
        "sourceCommit": COMMIT,
        "pythonABI": sysconfig.get_config_var("SOABI"),
        "wheelNative": wheel_audit,
        "sdkNative": sdk_audit,
        "files": {
            path.relative_to(stage).as_posix(): digest(path)
            for directory in (stage / "dist", sdk)
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        },
    }
    for name in ("dist", "sdk"):
        if (root / name).exists():
            (root / name).rename(stage / f"previous-{name}")
        (stage / name).rename(root / name)
    pending = root / "manifest.pending.json"
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(root / "manifest.json")


def verify(root: Path, script: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["recipe"] != recipe(script)
        or manifest["sourceCommit"] != COMMIT
        or manifest["pythonABI"] != sysconfig.get_config_var("SOABI")
    ):
        raise ValueError("ONNX Runtime build recipe, source, or Python ABI changed")
    for name, expected in manifest["files"].items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[0] not in {"dist", "sdk"}
        ):
            raise ValueError("Invalid artifact path")
        if digest(root / path) != expected:
            raise ValueError(f"ONNX Runtime artifact changed: {name}")
    wheels = list((root / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one cached ONNX Runtime wheel")
    wheel_metadata(wheels[0])
    if AUDITOR.audit_wheel(wheels[0]) != manifest["wheelNative"]:
        raise ValueError("ONNX Runtime wheel native audit changed")
    if audit_sdk(root / "sdk") != manifest["sdkNative"]:
        raise ValueError("ONNX Runtime C++ native audit changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("publish", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "publish":
            publish(args.root, args.script)
        else:
            verify(args.root, args.script)
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"ONNX Runtime compatibility validation: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
