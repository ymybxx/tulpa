"""Prepare verified PostgreSQL binaries and a minimal pg0 source patch on Windows.

No database is started here, and no existing pg0 installation is modified.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PG_ROOT = "postgresql-18.1.0-x86_64-pc-windows-msvc"
ASM1 = "urn:schemas-microsoft-com:asm.v1"
ASM3 = "urn:schemas-microsoft-com:asm.v3"
WS2019 = "http://schemas.microsoft.com/SMI/2019/WindowsSettings"
for prefix, namespace in (("", ASM1), ("asmv3", ASM3), ("ws2019", WS2019)):
    ET.register_namespace(prefix, namespace)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(spec: dict[str, str], directory: Path) -> Path:
    path = directory / spec["filename"]
    if path.is_file() and sha256(path) == spec["sha256"]:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        spec["url"], headers={"User-Agent": "tulpa-windows-build"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            part.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        if sha256(part) != spec["sha256"]:
            raise RuntimeError(f"Upstream SHA256 mismatch: {spec['filename']}")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


def patch_source(source: Path) -> None:
    main = source / "src/main.rs"
    text = main.read_text(encoding="utf-8")
    anchor = "fn get_base_dir() -> Result<PathBuf, CliError> {\n"
    if text.count(anchor) != 1:
        raise RuntimeError("Upstream get_base_dir changed")
    text = text.replace(
        anchor,
        anchor
        + r"""    // Opt-in private cache; other callers retain the upstream default.
    if let Some(home) = std::env::var_os("TULPA_PG0_HOME") {
        let path = PathBuf::from(home);
        if !path.is_absolute() {
            return Err(CliError::Other("TULPA_PG0_HOME must be absolute".to_string()));
        }
        return Ok(path);
    }
""",
    )
    logging_anchor = """fn init_logging(verbose: bool) {
    let filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("warn")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}
"""
    if text.count(logging_anchor) != 1:
        raise RuntimeError("Upstream init_logging changed")
    text = text.replace(
        logging_anchor,
        """fn init_logging(verbose: bool) {
    // Explicit diagnostic opt-in used only by the disposable Windows CI doctor.
    let filter = if cfg!(windows)
        && std::env::var("TULPA_PG0_DEBUG").ok().as_deref() == Some("1")
    {
        EnvFilter::new("warn,postgresql_commands=debug,postgresql_embedded=debug")
    } else if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("warn")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();
}
""",
    )
    main_anchor = "fn main() {\n    let cli = Cli::parse();\n"
    if text.count(main_anchor) != 1:
        raise RuntimeError("Upstream main entry changed")
    text = text.replace(
        main_anchor,
        """#[cfg(windows)]
fn restore_dll_search_path() -> std::io::Result<()> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }
    // PyInstaller's DLL directory belongs to the Python parent. Restore the
    // standard loader search in this private process before launching Postgres.
    if unsafe { SetDllDirectoryW(std::ptr::null()) } == 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn main() {
    #[cfg(windows)]
    if let Err(error) = restore_dll_search_path() {
        eprintln!("Error: could not restore pg0 DLL search path: {}", error);
        std::process::exit(1);
    }
    let cli = Cli::parse();
""",
    )
    main.write_text(text, encoding="utf-8", newline="\n")
    build = source / "build.rs"
    text = build.read_text(encoding="utf-8")
    anchor = "    let bundle_path = out_dir.join(&filename);\n"
    if text.count(anchor) != 2:
        raise RuntimeError("Upstream bundle download layout changed")
    for variable in ("TULPA_POSTGRES_ARCHIVE", "TULPA_PGVECTOR_ARCHIVE"):
        replacement = f'''    println!("cargo:rerun-if-env-changed={variable}");
    let bundle_path = out_dir.join(&filename);
    if let Some(archive) = env::var_os("{variable}") {{
        assert_eq!(target, "x86_64-pc-windows-msvc", "Windows archive on wrong target");
        let archive = PathBuf::from(archive);
        println!("cargo:rerun-if-changed={{}}", archive.display());
        fs::copy(&archive, &bundle_path).expect("copy verified Tulpa archive");
    }}
'''
        # Replace each original occurrence once; the replacement itself includes
        # the anchor, so split at the boundary between the two functions below.
        function = (
            "bundle_postgresql"
            if variable == "TULPA_POSTGRES_ARCHIVE"
            else "bundle_pgvector"
        )
        start = text.index(f"fn {function}(")
        offset = text.index(anchor, start)
        text = text[:offset] + replacement + text[offset + len(anchor) :]
    build.write_text(text, encoding="utf-8", newline="\n")


def extract_source(archive: Path, output: Path) -> None:
    # The upstream tar also contains SDK node_modules symlinks. Rust needs only
    # these files; skip unrelated trees so Windows needs no symlink privileges.
    with tarfile.open(archive) as bundle:
        for relative in (
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "versions.env",
            "src/main.rs",
            "README.md",
        ):
            member = bundle.getmember("pg0-0.14.2/" + relative)
            if not member.isfile():
                raise RuntimeError(f"Expected source file: {relative}")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    patch_source(output)


def read_manifest(executable: Path) -> bytes | None:
    # Data-only mapping reads resources without executing the binary/DllMain.
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    kernel.LoadLibraryExW.restype = wintypes.HMODULE
    kernel.FreeLibrary.argtypes = [wintypes.HMODULE]
    kernel.FreeLibrary.restype = wintypes.BOOL
    kernel.FindResourceW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p]
    kernel.FindResourceW.restype = ctypes.c_void_p
    kernel.SizeofResource.argtypes = [wintypes.HMODULE, ctypes.c_void_p]
    kernel.SizeofResource.restype = wintypes.DWORD
    kernel.LoadResource.argtypes = [wintypes.HMODULE, ctypes.c_void_p]
    kernel.LoadResource.restype = ctypes.c_void_p
    kernel.LockResource.argtypes = [ctypes.c_void_p]
    kernel.LockResource.restype = ctypes.c_void_p
    module = kernel.LoadLibraryExW(str(executable), None, 0x22)
    if not module:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        resource = kernel.FindResourceW(module, ctypes.c_void_p(1), ctypes.c_void_p(24))
        if not resource:
            error = ctypes.get_last_error()
            if error in (
                1812,
                1813,
                1814,
            ):  # No resource data/type/name in this binary.
                return None
            raise ctypes.WinError(error)
        size = kernel.SizeofResource(module, resource)
        loaded = kernel.LoadResource(module, resource)
        pointer = kernel.LockResource(loaded) if loaded else None
        if not size or not pointer:
            raise ctypes.WinError(ctypes.get_last_error())
        return ctypes.string_at(pointer, size)
    finally:
        kernel.FreeLibrary(module)


def utf8_manifest(original: bytes | None, name: str) -> bytes:
    if original:
        root = ET.fromstring(original.rstrip(b"\0"))
    else:
        root = ET.Element(f"{{{ASM1}}}assembly", {"manifestVersion": "1.0"})
    if root.tag != f"{{{ASM1}}}assembly":
        raise RuntimeError(f"Unexpected original manifest: {name}")
    if root.find(f"{{{ASM1}}}assemblyIdentity") is None:
        root.insert(
            0,
            ET.Element(
                f"{{{ASM1}}}assemblyIdentity",
                {
                    "name": f"Tulpa.PostgreSQL.{name}",
                    "version": "18.1.0.0",
                    "type": "win32",
                },
            ),
        )
    application = root.find(f"{{{ASM3}}}application")
    if application is None:
        application = ET.SubElement(root, f"{{{ASM3}}}application")
    settings = application.find(f"{{{ASM3}}}windowsSettings")
    if settings is None:
        settings = ET.SubElement(application, f"{{{ASM3}}}windowsSettings")
    code_page = settings.find(f"{{{WS2019}}}activeCodePage")
    if code_page is None:
        code_page = ET.SubElement(settings, f"{{{WS2019}}}activeCodePage")
    code_page.text = "UTF-8"
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def verify_manifest(executable: Path) -> None:
    manifest = read_manifest(executable)
    if manifest is None or not any(
        node.text == "UTF-8"
        for node in ET.fromstring(manifest.rstrip(b"\0")).iter(
            f"{{{WS2019}}}activeCodePage"
        )
    ):
        raise RuntimeError(f"UTF-8 process manifest missing: {executable.name}")


def prepare_postgres(postgres: Path, vector: Path, output: Path, mt: Path) -> None:
    with zipfile.ZipFile(postgres) as bundle:
        for entry in bundle.infolist():
            parts = PurePosixPath(entry.filename).parts
            if (
                not parts
                or parts[0] != PG_ROOT
                or ".." in parts
                or any(":" in part for part in parts)
            ):
                raise RuntimeError(
                    f"Unexpected PostgreSQL archive entry: {entry.filename}"
                )
            target = output.joinpath(*parts[1:])
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    with tarfile.open(vector) as bundle:
        for entry in bundle.getmembers():
            parts = PurePosixPath(entry.name).parts[1:]
            if not entry.isfile() or ".." in parts:
                continue
            if parts != ("lib", "vector.dll") and parts[:2] != ("share", "extension"):
                continue
            target = output.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(entry) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    manifests = output.parent / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    executables = sorted(output.rglob("*.exe"))
    required = {
        output / "bin" / name
        for name in ("initdb.exe", "postgres.exe", "pg_ctl.exe", "psql.exe")
    }
    if not required.issubset(executables):
        raise RuntimeError("PostgreSQL archive is incomplete")
    for index, executable in enumerate(executables):
        original = read_manifest(executable)
        manifest = manifests / f"{index}-{executable.name}.manifest"
        manifest.write_bytes(utf8_manifest(original, executable.stem))
        subprocess.run(
            [
                str(mt),
                "-nologo",
                "-manifest",
                str(manifest),
                f"-outputresource:{executable};#1",
            ],
            check=True,
        )
        verify_manifest(executable)
    print(
        f"Verified UTF-8 manifests on {len(executables)} PostgreSQL executables",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--mt", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Preparing PE manifests requires Windows")
    cache = args.cache_root.resolve()
    pins = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    archives = {
        name: download(spec, cache / "downloads") for name, spec in pins.items()
    }
    source = cache / "source"
    extract_source(archives["source"], source)
    prepared = cache / "prepared"
    output = prepared / "postgresql"
    archive = prepared / "postgresql-utf8.zip"
    receipt = prepared / "receipt.json"
    fingerprint = sha256(Path(__file__)) + sha256(HERE / "sources.json")
    cached = json.loads(receipt.read_text()) if receipt.is_file() else {}
    valid = (
        cached.get("fingerprint") == fingerprint
        and archive.is_file()
        and sha256(archive) == cached.get("archiveSha256")
    )
    if valid:
        valid = all(
            (output / p).is_file() and sha256(output / p) == h
            for p, h in cached["executables"].items()
        )
    if not valid:
        shutil.rmtree(prepared, ignore_errors=True)
        prepare_postgres(
            archives["postgresql"], archives["pgvector"], output, args.mt.resolve()
        )
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as bundle:
            for file in sorted(p for p in output.rglob("*") if p.is_file()):
                entry = zipfile.ZipInfo(
                    f"{PG_ROOT}/{file.relative_to(output).as_posix()}",
                    date_time=(2026, 1, 1, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(entry, file.read_bytes())
        receipt.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "archiveSha256": sha256(archive),
                    "executables": {
                        p.relative_to(output).as_posix(): sha256(p)
                        for p in output.rglob("*.exe")
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    for executable in output.rglob("*.exe"):
        verify_manifest(executable)
    result = {
        "postgresqlDirectory": str(output),
        "postgresqlArchive": str(archive),
        "sourceDirectory": str(source),
        "pgvectorArchive": str(archives["pgvector"]),
    }
    (cache / "prepared.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
