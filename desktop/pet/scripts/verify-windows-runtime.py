"""Windows x64 package gate; optional DB smoke only on disposable hosted CI."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def pe_machine(path: Path) -> int:
    with path.open("rb") as source:
        if source.read(2) != b"MZ":
            raise ValueError(f"not a PE binary: {path}")
        source.seek(0x3C)
        offset = struct.unpack("<I", source.read(4))[0]
        source.seek(offset)
        if source.read(4) != b"PE\0\0":
            raise ValueError(f"invalid PE header: {path}")
        return struct.unpack("<H", source.read(2))[0]


def audit_binaries(root: Path) -> int:
    binaries = [
        p for p in root.rglob("*") if p.suffix.lower() in {".exe", ".dll", ".pyd"}
    ]
    if not binaries:
        raise ValueError("no Windows runtime binaries found")
    for binary in binaries:
        if pe_machine(binary) != 0x8664:
            raise ValueError(f"runtime binary is not Windows x64: {binary}")
    if not any(p.name.lower() == "pg0.exe" for p in binaries):
        raise ValueError("bundled pg0.exe is missing")
    return len(binaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("the runtime doctor must execute on Windows")
    root = args.runtime.resolve()
    count = audit_binaries(root)
    environment = {
        k: v for k, v in os.environ.items() if not k.startswith("FISH_MEMORY_")
    }
    environment.update(
        FISH_MEMORY_BUNDLED_MODELS_DIR=str(args.models.resolve()),
        HF_HUB_OFFLINE="1",
        PYTHONUTF8="1",
    )
    database_smoke = (
        os.getenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE") == "1"
        and os.getenv("GITHUB_ACTIONS") == "true"
        and os.getenv("RUNNER_ENVIRONMENT") == "github-hosted"
    )
    # pg0 uses SHGetKnownFolderPath, so overriding USERPROFILE cannot isolate
    # its registry/cache. Only touch a disposable hosted CI account; ordinary
    # developer builds run the import/model doctor without starting PostgreSQL.
    with tempfile.TemporaryDirectory(prefix="fish-memory-中文 smoke-") as profile:
        if database_smoke:
            # This verifier uses managed Python without installing the package.
            # Grant the token's user explicitly: an elevated temporary directory
            # can otherwise be owned by Administrators, which PostgreSQL drops.
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
            from fish_memory.runtime_platform import private_directory_permissions

            private_directory_permissions(Path(profile))
        (Path(profile) / ".fish-memory-package-smoke").touch()
        command = [str(root / "fish-memory-runtime.exe"), "doctor"]
        if database_smoke:
            environment["FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE"] = "1"
            environment["TULPA_PG0_HOME"] = str(Path(profile) / "数据库运行时")
            environment["TULPA_PG0_DEBUG"] = "1"
            environment["TEMP"] = profile
            environment["TMP"] = profile
            command.extend(["--database-smoke", "--directory", profile])
        result = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if database_smoke and result.returncode == 0:
            private_home = Path(environment["TULPA_PG0_HOME"])
            if not (private_home / "instances").is_dir() or not any(
                (private_home / "installation").glob("*/bin/initdb.exe")
            ):
                raise RuntimeError(
                    "packaged pg0 ignored its isolated runtime directory"
                )
    if result.returncode:
        raise RuntimeError(
            f"Windows runtime doctor failed ({result.returncode}):\n"
            f"stdout:\n{result.stdout[-16_384:]}\nstderr:\n{result.stderr[-16_384:]}"
        )
    report = json.loads(result.stdout.splitlines()[-1])
    if (
        report.get("ok") is not True
        or report.get("vad", {}).get("backend") != "sherpa_silero"
    ):
        raise RuntimeError("Windows packaged runtime doctor did not pass")
    if database_smoke and report.get("database") != {
        "backend": "pg0",
        "vectorDistance": 1.0,
        "stopped": True,
    }:
        raise RuntimeError("Windows packaged PostgreSQL/pgvector smoke did not pass")
    print(
        json.dumps(
            {
                "windowsX64Binaries": count,
                "doctor": report,
                "databaseSmoke": "passed"
                if database_smoke
                else "skipped: requires disposable hosted Windows CI",
            }
        )
    )


if __name__ == "__main__":
    main()
