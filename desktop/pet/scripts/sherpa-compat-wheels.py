"""Split and validate the desktop-only, unmodified sherpa compatibility build."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

VERSION = "1.13.4"
ORT_VERSION = "1.28.0"
MAIN_INFO = f"sherpa_onnx-{VERSION}.dist-info"
CORE_INFO = f"sherpa_onnx_core-{VERSION}.dist-info"
PLATFORM = "macosx_12_0_arm64"  # macOS >= 11 wheel tags use the major version.
MACHO_MAGIC = {
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recipe(script: Path) -> str:
    return sha256(script.read_bytes() + Path(__file__).read_bytes())


def sdk_fingerprint(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    return sha256(
        "\n".join(
            f"{path.relative_to(root).as_posix()}:{sha256(path.read_bytes())}"
            for path in files
        ).encode()
    )


def audit_wheel(path: Path) -> list[dict[str, str]]:
    audits = []
    expected_abi = sysconfig.get_config_var("EXT_SUFFIX")
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as tmp:
        for name in archive.namelist():
            data = archive.read(name)
            if data[:4] not in MACHO_MAGIC:
                continue
            if (
                "cpython-" in name
                and name.endswith(".so")
                and not name.endswith(expected_abi)
            ):
                raise ValueError(f"Unexpected Python ABI: {name}")
            binary = Path(tmp) / Path(name).name
            binary.write_bytes(data)
            build = subprocess.check_output(
                ["xcrun", "vtool", "-arch", "arm64", "-show-build", str(binary)],
                text=True,
            )
            minimum = re.search(r"\bminos\s+(\d+\.\d+(?:\.\d+)?)", build)
            if minimum is None:
                raise ValueError(f"Missing arm64 minimum macOS target: {name}")
            target = minimum[1]
            target_parts = tuple(map(int, target.split(".")))
            if (target_parts + (0, 0, 0))[:3] > (12, 3, 0):
                raise ValueError(f"macOS target exceeds 12.3: {name}: {target}")
            links = subprocess.check_output(["otool", "-L", str(binary)], text=True)
            for line in links.splitlines()[1:]:
                dependency = line.strip().split(" (", 1)[0]
                if not dependency.startswith(
                    ("@rpath/", "@loader_path/", "/usr/lib/", "/System/Library/")
                ):
                    raise ValueError(f"Non-portable linkage in {name}: {dependency}")
            audits.append({"path": name, "minimumMacOS": target})
    if not audits:
        raise ValueError(f"No native binaries in {path.name}")
    return audits


def write_wheel(path: Path, files: dict[str, bytes], info: str, tag: str) -> None:
    files[f"{info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: fish-memory-sherpa-compat\n"
        f"Root-Is-Purelib: false\nTag: {tag}\n"
    ).encode()
    records = io.StringIO(newline="")
    writer = csv.writer(records)
    for name, data in sorted(files.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow((name, f"sha256={digest}", len(data)))
    writer.writerow((f"{info}/RECORD", "", ""))
    files[f"{info}/RECORD"] = records.getvalue().encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, data)


def split(wheel: Path, dist: Path, script: Path, ort_root: Path) -> None:
    main: dict[str, bytes] = {}
    core: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name in {
                f"{MAIN_INFO}/RECORD",
                f"{MAIN_INFO}/WHEEL",
            }:
                continue
            destination = (
                core
                if name.endswith(".dylib") or name.startswith("sherpa_onnx/include/")
                else main
            )
            destination[name] = archive.read(name)
    runtime_name = f"sherpa_onnx/lib/libonnxruntime.{ORT_VERSION}.dylib"
    if runtime_name not in core:
        raise ValueError(f"Expected the same-version runtime: {runtime_name}")
    if not any(name.endswith(sysconfig.get_config_var("EXT_SUFFIX")) for name in main):
        raise ValueError(
            "The build did not produce an extension for this Python interpreter"
        )
    metadata = main[f"{MAIN_INFO}/METADATA"].decode()
    headers, body = metadata.split("\n\n", 1)
    main[f"{MAIN_INFO}/METADATA"] = (
        headers + f"\nRequires-Dist: sherpa-onnx-core=={VERSION}\n\n" + body
    ).encode()
    core_headers = headers.replace(
        "Name: sherpa_onnx\n", "Name: sherpa-onnx-core\n"
    ).replace("Name: sherpa-onnx\n", "Name: sherpa-onnx-core\n")
    for name, data in main.items():
        if name.startswith(f"{MAIN_INFO}/licenses/"):
            core[name.replace(MAIN_INFO, CORE_INFO, 1)] = data
    for source, filename in [
        ("LICENSE", "onnxruntime-LICENSE"),
        ("ThirdPartyNotices.txt", "onnxruntime-ThirdPartyNotices.txt"),
    ]:
        core[f"{CORE_INFO}/licenses/{filename}"] = (ort_root / source).read_bytes()
        core_headers += f"\nLicense-File: {filename}"
    core[f"{CORE_INFO}/METADATA"] = (
        core_headers + "\n\nNative libraries for the desktop compatibility build.\n"
    ).encode()
    core[f"{CORE_INFO}/top_level.txt"] = b"sherpa_onnx\n"
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    dist.mkdir(parents=True, exist_ok=True)
    main_path = dist / f"sherpa_onnx-{VERSION}-{python_tag}-{python_tag}-{PLATFORM}.whl"
    core_path = dist / f"sherpa_onnx_core-{VERSION}-py3-none-{PLATFORM}.whl"
    write_wheel(main_path, main, MAIN_INFO, f"{python_tag}-{python_tag}-{PLATFORM}")
    write_wheel(core_path, core, CORE_INFO, f"py3-none-{PLATFORM}")
    manifest = {
        "recipe": recipe(script),
        "pythonABI": sysconfig.get_config_var("SOABI"),
        "sdkSHA256": sdk_fingerprint(ort_root),
        "wheels": [],
    }
    for path in (main_path, core_path):
        manifest["wheels"].append(
            {
                "filename": path.name,
                "sha256": sha256(path.read_bytes()),
                "native": audit_wheel(path),
            }
        )
    (dist / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify(dist: Path, script: Path, ort_root: Path | None = None) -> None:
    manifest = json.loads((dist / "manifest.json").read_text())
    if manifest["recipe"] != recipe(script) or manifest[
        "pythonABI"
    ] != sysconfig.get_config_var("SOABI"):
        raise ValueError("Compatibility build recipe or Python ABI changed")
    if ort_root is not None and manifest["sdkSHA256"] != sdk_fingerprint(ort_root):
        raise ValueError("ONNX Runtime SDK changed")
    entries = manifest["wheels"]
    if len(entries) != 2 or len(list(dist.glob("*.whl"))) != 2:
        raise ValueError("Expected exactly the main and core wheels")
    for item in entries:
        name = item["filename"]
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError("Invalid cached wheel path")
        path = dist / name
        if (
            sha256(path.read_bytes()) != item["sha256"]
            or audit_wheel(path) != item["native"]
        ):
            raise ValueError(f"Cached wheel validation failed: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("split", "verify"))
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--ort-root", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "split":
            split(args.wheel, args.dist, args.script, args.ort_root)
        else:
            verify(args.dist, args.script, args.ort_root)
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"sherpa compatibility validation: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
