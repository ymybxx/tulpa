import hashlib
import json
import os
import runpy
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "desktop/pet/scripts/verify-macos-compatibility.py"
CHECKER = runpy.run_path(str(SCRIPT))
ARM64 = 0x0100000C
X86_64 = 0x01000007


def make_thin(
    minimum=(12, 3, 0),
    *,
    endian="<",
    is64=True,
    cpu=ARM64,
    old=False,
    platform=1,
    commands=None,
):
    if commands is None:
        packed_version = minimum[0] << 16 | minimum[1] << 8 | minimum[2]
        if old:
            commands = struct.pack(endian + "4I", 0x24, 16, packed_version, 0)
        else:
            commands = struct.pack(
                endian + "6I", 0x32, 24, platform, packed_version, 0, 0
            )
    header = [0xFEEDFACF if is64 else 0xFEEDFACE, cpu, 0, 6, 1, len(commands), 0]
    if is64:
        header.append(0)
    return struct.pack(endian + "I" * len(header), *header) + commands


def make_fat(slices, *, endian=">", is64=False):
    entry_layout = endian + ("IIQQII" if is64 else "IIIII")
    table_size = 8 + len(slices) * struct.calcsize(entry_layout)
    content = bytearray(
        struct.pack(endian + "II", 0xCAFEBABF if is64 else 0xCAFEBABE, len(slices))
    )
    offset = table_size
    for cpu, blob in slices:
        fields = [cpu, 0, offset, len(blob), 0]
        if is64:
            fields.append(0)
        content += struct.pack(entry_layout, *fields)
        offset += len(blob)
    for _, blob in slices:
        content += blob
    return bytes(content)


@pytest.mark.parametrize("endian,is64", [("<", True), (">", True), ("<", False)])
def test_thin_minimum_uses_load_command_not_filename(tmp_path, endian, is64):
    path = tmp_path / "claimed-macos-10.9.dylib"
    path.write_bytes(make_thin((14, 2, 1), endian=endian, is64=is64))
    assert CHECKER["inspect_binary"](path) == [(14, 2, 1)]
    result = CHECKER["audit_directory"](tmp_path, (12, 3, 0))
    assert result["incompatible"] == [
        {
            "path": path.name,
            "minimumVersion": "14.2.1",
        }
    ]


def test_legacy_minimum_command(tmp_path):
    path = tmp_path / "legacy.dylib"
    path.write_bytes(make_thin((11, 0, 0), old=True))
    assert CHECKER["inspect_binary"](path) == [(11, 0, 0)]
    assert not CHECKER["audit_directory"](tmp_path, (12, 3, 0))["incompatible"]


@pytest.mark.parametrize(
    "endian,is64",
    [
        (">", False),
        ("<", False),
        (">", True),
        ("<", True),
    ],
)
def test_fat_checks_each_arm64_slice_and_ignores_x86_requirement(
    tmp_path, endian, is64
):
    path = tmp_path / "universal"
    path.write_bytes(
        make_fat(
            [
                (X86_64, make_thin((26, 0, 0), cpu=X86_64)),
                (ARM64, make_thin((12, 3, 0))),
                (ARM64, make_thin((15, 5, 0))),
            ],
            endian=endian,
            is64=is64,
        )
    )
    assert CHECKER["inspect_binary"](path) == [(12, 3, 0), (15, 5, 0)]
    result = CHECKER["audit_directory"](tmp_path, (12, 3, 0))
    assert result["arm64FilesChecked"] == 1
    assert result["arm64SlicesChecked"] == 2
    assert result["incompatible"][0]["minimumVersion"] == "15.5.0"


@pytest.mark.parametrize(
    "blob",
    [
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        make_thin(commands=struct.pack("<II", 0x32, 0)),
        make_thin(commands=struct.pack("<II", 0x32, 128)),
        make_thin(commands=struct.pack("<II", 0x32, 8)),
        make_thin(commands=struct.pack("<6I", 0x32, 24, 1, 0x000C0300, 0, 1)),
        struct.pack(">II", 0xCAFEBABE, 1)
        + struct.pack(">IIIII", ARM64, 0, 9999, 32, 0),
        make_fat([(ARM64, make_thin(cpu=X86_64))]),
    ],
)
def test_malformed_known_macho_fails_closed(tmp_path, blob):
    (tmp_path / "invalid").write_bytes(blob)
    result = CHECKER["audit_directory"](tmp_path, (12, 3, 0))
    assert result["errors"]
    assert result["errors"][0]["path"] == "invalid"


def test_missing_or_wrong_platform_command_fails_closed(tmp_path):
    (tmp_path / "without-minos").write_bytes(
        make_thin(commands=struct.pack("<II", 0x1, 8))
    )
    (tmp_path / "ios-binary").write_bytes(make_thin(platform=2))
    result = CHECKER["audit_directory"](tmp_path, (12, 3, 0))
    assert len(result["errors"]) == 2
    assert any("no macOS" in item["error"] for item in result["errors"])
    assert any("not macOS" in item["error"] for item in result["errors"])


def test_hardlinks_are_not_counted_twice(tmp_path):
    binary = tmp_path / "binary"
    binary.write_bytes(make_thin((14, 0, 0)))
    os.link(binary, tmp_path / "hardlink")
    result = CHECKER["audit_directory"](tmp_path, (12, 3, 0))
    assert result["arm64FilesChecked"] == 1
    assert len(result["incompatible"]) == 1


def test_symlink_directories_and_files_are_skipped(tmp_path):
    real_directory = tmp_path / "A"
    real_directory.mkdir()
    binary = real_directory / "binary"
    binary.write_bytes(make_thin((14, 0, 0)))
    try:
        (tmp_path / "Current").symlink_to(real_directory, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    (tmp_path / "binary-link").symlink_to(binary)
    (tmp_path / "missing-link").symlink_to(tmp_path / "missing")
    result = CHECKER["audit_directory"](tmp_path, (12, 3, 0))
    assert result["arm64FilesChecked"] == 1
    assert result["symlinksSkipped"] == 3
    assert result["incompatible"] == [
        {
            "path": "A/binary",
            "minimumVersion": "14.0.0",
        }
    ]


def test_non_macho_and_x86_files_are_ignored(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"not a Mach-O")
    (tmp_path / "intel").write_bytes(make_thin((26, 0, 0), cpu=X86_64))
    (tmp_path / "arm").write_bytes(make_thin((12, 3, 0)))
    result = CHECKER["audit_directory"](tmp_path, (12, 3, 0))
    assert result["arm64FilesChecked"] == 1
    assert not result["incompatible"]
    assert not result["errors"]


def test_cli_reports_nonzero_without_mutating_binary(tmp_path, capsys):
    path = tmp_path / "library"
    path.write_bytes(make_thin((14, 0, 0)))
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert CHECKER["main"]([str(tmp_path), "--target", "12.3"]) == 1
    output = capsys.readouterr().out
    assert "FAIL" in output and "macOS 14.0.0 required: library" in output
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    assert CHECKER["main"]([str(tmp_path), "--minimum-version", "14", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["minimumVersion"] == "14.0.0"
    assert report["incompatible"] == []


def test_empty_directory_is_not_a_successful_compatibility_check(tmp_path, capsys):
    assert CHECKER["main"]([str(tmp_path)]) == 1
    assert "no arm64 Mach-O files found" in capsys.readouterr().out


@pytest.mark.parametrize("version", ["12.3.4.5", "macOS12", "12.-1", "12.256"])
def test_cli_rejects_invalid_version(tmp_path, version):
    with pytest.raises(SystemExit) as error:
        CHECKER["main"]([str(tmp_path), "--minimum-version", version])
    assert error.value.code == 2
