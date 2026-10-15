"""Static package metadata, readable before running plugin code."""

from __future__ import annotations

import hashlib
import platform
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,95}$")
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.+-]*)$")
    entrypoint: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
    description: str = Field(default="", max_length=500)
    protocol: Literal[1] = 1
    autonomous: bool = False

    @classmethod
    def read(cls, root: Path) -> Manifest:
        return cls.model_validate_json((root / "fish-plugin.json").read_text())


def target() -> str:
    machine = platform.machine().lower()
    machine = {"aarch64": "arm64", "amd64": "x86_64"}.get(machine, machine)
    return (
        f"{sys.platform}-{machine}-py{sys.version_info.major}{sys.version_info.minor}"
    )


class Bundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal[1] = 1
    manifest: Manifest
    target: str
    files: dict[str, str]

    @field_validator("files")
    @classmethod
    def safe_files(cls, value: dict[str, str]) -> dict[str, str]:
        if not 1 <= len(value) <= 256:
            raise ValueError("Bundle needs 1 to 256 files")
        for name, digest in value.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe package path")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid artifact digest")
        return value


def extract_bundle(archive: Path, destination: Path, expected_target: str) -> Bundle:
    """No code execution; exact membership, size limits, hashes, and no symlinks."""
    with zipfile.ZipFile(archive) as source:
        infos = source.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or len(names) > 257:
            raise ValueError("Duplicate or excessive archive entries")
        if sum(info.file_size for info in infos) > 512 * 1024 * 1024:
            raise ValueError("Bundle exceeds 512 MiB")
        if source.getinfo("bundle.json").file_size > 64 * 1024:
            raise ValueError("Bundle index is too large")
        bundle = Bundle.model_validate_json(source.read("bundle.json"))
        if bundle.target != expected_target:
            raise ValueError(
                f"This artifact targets {bundle.target}; App needs {expected_target}"
            )
        if set(names) != {"bundle.json", *bundle.files}:
            raise ValueError("Archive membership differs from index")
        content = {}
        for name, digest in bundle.files.items():
            info = source.getinfo(name)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not allowed in plugin archives")
            data = source.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Artifact checksum mismatch: {name}")
            content[name] = data
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in content.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (destination / "bundle.json").write_text(bundle.model_dump_json(indent=2))
        (destination / "fish-plugin.json").write_text(
            bundle.manifest.model_dump_json(indent=2)
        )
        return bundle
