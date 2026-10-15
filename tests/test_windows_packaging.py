from __future__ import annotations

import runpy
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "desktop/pet/scripts/verify-windows-runtime.py"


def write_pe(path: Path, machine: int):
    header = bytearray(128)
    header[:2] = b"MZ"
    struct.pack_into("<I", header, 0x3C, 64)
    header[64:68] = b"PE\0\0"
    struct.pack_into("<H", header, 68, machine)
    path.write_bytes(header)


def test_windows_runtime_gate_rejects_mixed_architecture_and_missing_pg0(tmp_path):
    audit = runpy.run_path(str(SCRIPT))["audit_binaries"]
    write_pe(tmp_path / "fish-memory-runtime.exe", 0x8664)
    with pytest.raises(ValueError, match="pg0.exe is missing"):
        audit(tmp_path)
    write_pe(tmp_path / "pg0.exe", 0x8664)
    assert audit(tmp_path) == 2
    write_pe(tmp_path / "wrong.pyd", 0xAA64)
    with pytest.raises(ValueError, match="not Windows x64"):
        audit(tmp_path)


def test_windows_runtime_gate_rejects_non_pe_dll(tmp_path):
    audit = runpy.run_path(str(SCRIPT))["audit_binaries"]
    (tmp_path / "foreign.dll").write_bytes(b"not Windows")
    with pytest.raises(ValueError, match="not a PE"):
        audit(tmp_path)
