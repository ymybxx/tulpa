#!/usr/bin/env python3
"""Check actual arm64 Mach-O deployment commands without modifying binaries.

Usage: python verify-macos-compatibility.py APP_OR_RUNTIME --target 12.3
This is a packaging gate, not a substitute for running the app on the oldest OS.
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import BinaryIO

ARM64 = 0x0100000C
THIN_MAGIC = {
    b"\xce\xfa\xed\xfe": ("<", False),
    b"\xcf\xfa\xed\xfe": ("<", True),
    b"\xfe\xed\xfa\xce": (">", False),
    b"\xfe\xed\xfa\xcf": (">", True),
}
FAT_MAGIC = {
    b"\xca\xfe\xba\xbe": (">", False),
    b"\xca\xfe\xba\xbf": (">", True),
    b"\xbe\xba\xfe\xca": ("<", False),
    b"\xbf\xba\xfe\xca": ("<", True),
}
Version = tuple[int, int, int]


def parse_version(text: str) -> Version:
    parts = text.split(".")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("expected a macOS version, e.g. 12.3")
    values = [int(part) for part in parts]
    values += [0] * (3 - len(values))
    if values[0] > 65535 or values[1] > 255 or values[2] > 255:
        raise argparse.ArgumentTypeError("macOS version is outside Mach-O limits")
    return values[0], values[1], values[2]


def version_text(version: Version) -> str:
    return ".".join(map(str, version))


def _decode_version(value: int) -> Version:
    return value >> 16, (value >> 8) & 255, value & 255


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    data = stream.read(length)
    if len(data) != length:
        raise ValueError("truncated Mach-O header or load commands")
    return data


def _inspect_slice(
    stream: BinaryIO, offset: int, slice_size: int, *, require_arm64: bool = False
) -> Version | None:
    stream.seek(offset)
    magic = _read_exact(stream, 4)
    if magic not in THIN_MAGIC:
        raise ValueError("fat slice does not contain a Mach-O header")
    endian, is64 = THIN_MAGIC[magic]
    header_size = 32 if is64 else 28
    header = struct.unpack(
        endian + ("7I" if is64 else "6I"),
        _read_exact(stream, header_size - 4),
    )
    cputype, _, _, ncmds, sizeofcmds = header[:5]
    if cputype != ARM64:
        if require_arm64:
            raise ValueError("fat arm64 entry has a different slice architecture")
        return None
    if header_size > slice_size or sizeofcmds > slice_size - header_size:
        raise ValueError("load commands extend beyond Mach-O slice")
    if sizeofcmds > 16 * 1024 * 1024 or ncmds > sizeofcmds // 8:
        raise ValueError("invalid Mach-O load-command count or size")
    commands = _read_exact(stream, sizeofcmds)
    index = 0
    minimums = []
    for _ in range(ncmds):
        if index + 8 > len(commands):
            raise ValueError("load command extends beyond command table")
        cmd, size = struct.unpack_from(endian + "II", commands, index)
        if size < 8 or index + size > len(commands):
            raise ValueError("invalid Mach-O load-command size")
        if cmd == 0x32:  # LC_BUILD_VERSION
            if size < 24:
                raise ValueError("invalid LC_BUILD_VERSION size")
            platform, minimum, _, ntools = struct.unpack_from(
                endian + "4I", commands, index + 8
            )
            if 24 + ntools * 8 > size:
                raise ValueError("truncated LC_BUILD_VERSION tools")
            if platform != 1:
                raise ValueError(f"arm64 slice targets platform {platform}, not macOS")
            minimums.append(_decode_version(minimum))
        elif cmd == 0x24:  # LC_VERSION_MIN_MACOSX
            if size < 16:
                raise ValueError("invalid LC_VERSION_MIN_MACOSX size")
            minimum, _ = struct.unpack_from(endian + "II", commands, index + 8)
            minimums.append(_decode_version(minimum))
        index += size
    if index != sizeofcmds:
        raise ValueError("Mach-O load-command table has unaccounted bytes")
    if not minimums:
        raise ValueError("arm64 slice has no macOS minimum-version command")
    return max(minimums)


def inspect_binary(path: Path) -> list[Version]:
    """Return every arm64 slice's minimum, or [] for non-arm64/non-Mach-O files."""
    with path.open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        magic = stream.read(4)
        if magic in THIN_MAGIC:
            minimum = _inspect_slice(stream, 0, file_size)
            return [minimum] if minimum is not None else []
        if magic not in FAT_MAGIC:
            return []
        endian, is64 = FAT_MAGIC[magic]
        (count,) = struct.unpack(endian + "I", _read_exact(stream, 4))
        if count > 128:
            raise ValueError("invalid fat architecture count")
        layout = endian + ("IIQQII" if is64 else "IIIII")
        entry_size = struct.calcsize(layout)
        table_end = 8 + count * entry_size
        slices = []
        for _ in range(count):
            fields = struct.unpack(layout, _read_exact(stream, entry_size))
            cputype, offset, size = fields[0], fields[2], fields[3]
            if offset < table_end or size < 28 or offset + size > file_size:
                raise ValueError("fat architecture extends beyond file or into header")
            if cputype == ARM64:
                slices.append((offset, size))
        minimums = []
        for offset, size in slices:
            minimum = _inspect_slice(stream, offset, size, require_arm64=True)
            if minimum is not None:
                minimums.append(minimum)
        return minimums


def audit_directory(root: Path, minimum_version: Version) -> dict:
    """Deduplicate real files, skip symlink aliases, and fail closed on bad Mach-O."""
    seen = set()
    checked = 0
    slices_checked = 0
    symlinks_skipped = 0
    incompatible = []
    errors = []

    def walk_error(error: OSError) -> None:
        errors.append({"path": str(error.filename), "error": str(error)})

    for directory, directories, files in os.walk(
        root, followlinks=False, onerror=walk_error
    ):
        for name in directories[:]:
            if Path(directory, name).is_symlink():
                directories.remove(name)
                symlinks_skipped += 1
        directories.sort()
        for name in sorted(files):
            path = Path(directory, name)
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlinks_skipped += 1
                continue
            try:
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                minimums = inspect_binary(path)
                if not minimums:
                    continue
                checked += 1
                slices_checked += len(minimums)
                required = max(minimums)
                if required > minimum_version:
                    incompatible.append(
                        {
                            "path": relative,
                            "minimumVersion": version_text(required),
                        }
                    )
            except (OSError, ValueError, struct.error) as error:
                errors.append({"path": relative, "error": str(error)})
    if not checked and not errors:
        errors.append({"path": ".", "error": "no arm64 Mach-O files found"})
    return {
        "minimumVersion": version_text(minimum_version),
        "arm64FilesChecked": checked,
        "arm64SlicesChecked": slices_checked,
        "symlinksSkipped": symlinks_skipped,
        "incompatible": incompatible,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="unpacked app or runtime directory")
    parser.add_argument(
        "--target",
        "--minimum-version",
        dest="minimum_version",
        type=parse_version,
        default=(12, 3, 0),
        help="oldest supported macOS version (default: 12.3)",
    )
    parser.add_argument("--json", action="store_true", help="print structured results")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("root must be an existing directory")
    result = audit_directory(root, args.minimum_version)
    failed = bool(result["incompatible"] or result["errors"])
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        status = "FAIL" if failed else "PASS"
        print(
            f"{status}: macOS {result['minimumVersion']}; "
            f"{result['arm64FilesChecked']} unique arm64 Mach-O files; "
            f"{len(result['incompatible'])} incompatible; "
            f"{len(result['errors'])} errors"
        )
        for item in result["incompatible"]:
            print(f"  macOS {item['minimumVersion']} required: {item['path']}")
        for item in result["errors"]:
            print(f"  ERROR: {item['path']}: {item['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
