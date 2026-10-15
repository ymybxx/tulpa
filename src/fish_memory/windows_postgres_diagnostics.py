"""Failure-only loader probes for the isolated hosted Windows package doctor."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from fish_memory.runtime_platform import (
    private_directory_permissions,
    windows_acl_snapshot,
)


class _DllDirectory:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("DLL directories are only available on Windows")
        from ctypes import wintypes

        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel.GetDllDirectoryW.argtypes = [wintypes.DWORD, wintypes.LPWSTR]
        self._kernel.GetDllDirectoryW.restype = wintypes.DWORD
        self._kernel.SetDllDirectoryW.argtypes = [wintypes.LPCWSTR]
        self._kernel.SetDllDirectoryW.restype = wintypes.BOOL

    def get(self) -> str:
        if sys.platform != "win32":
            raise OSError("DLL directories are only available on Windows")
        buffer = ctypes.create_unicode_buffer(32768)
        ctypes.set_last_error(0)
        length = self._kernel.GetDllDirectoryW(len(buffer), buffer)
        if length >= len(buffer):
            raise OSError("DLL directory exceeds probe buffer")
        if length == 0 and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        return buffer.value

    def set(self, value: str | None) -> None:
        if sys.platform != "win32":
            raise OSError("DLL directories are only available on Windows")
        if not self._kernel.SetDllDirectoryW(value):
            raise ctypes.WinError(ctypes.get_last_error())


def _version(executable: Path) -> dict[str, object]:
    result: dict[str, object] = {"program": executable.name}
    if not executable.is_file():
        return {**result, "error": "missing_executable"}
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        result.update(
            returnCode=completed.returncode,
            returnCodeHex=f"0x{completed.returncode & 0xFFFFFFFF:08X}",
            stdoutTail=completed.stdout[-1024:],
            stderrTail=completed.stderr[-1024:],
        )
    except Exception as error:
        result["error"] = type(error).__name__
    return result


def diagnose_postgres_loader(private_home: Path) -> dict[str, object]:
    """Compare loader behavior without retrying database startup or changing files.

    Only the independent package doctor calls this after database.start failed.
    Never use this process-wide DLL toggle in the concurrent production service.
    """
    if not (
        sys.platform == "win32"
        and os.getenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE") == "1"
        and os.getenv("GITHUB_ACTIONS") == "true"
        and os.getenv("RUNNER_ENVIRONMENT") == "github-hosted"
    ):
        return {"skipped": "requires_hosted_windows_package_doctor"}
    binaries = [
        directory.parent / name
        for directory in sorted(
            (private_home / "installation").glob("*/bin/initdb.exe")
        )[:3]
        for name in ("initdb.exe", "postgres.exe", "pg_ctl.exe")
    ]
    if not binaries:
        return {"error": "no_extracted_postgres_binaries"}
    report: dict[str, object] = {}
    try:
        dll = _DllDirectory()
        original = dll.get()
        report["inheritedDllDirectory"] = original
        report["inherited"] = [_version(exe) for exe in binaries]
        if not original:
            # GetDllDirectory cannot distinguish NULL from the empty-string
            # mode. Do not alter either when exact restoration is impossible.
            report["comparisonSkipped"] = "no_extra_dll_directory"
        else:
            try:
                dll.set(None)
                report["cleared"] = [_version(exe) for exe in binaries]
            finally:
                dll.set(original)
                report["restored"] = dll.get() == original
    except Exception as error:
        report["diagnosticError"] = type(error).__name__
    report["initialization"] = run_initdb_matrix(binaries[0].parent.parent)
    return report


_NO_WINDOW = 0x08000000


def _file_tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2048))
        return stream.read().decode("utf-8", errors="replace")


def _probe_initdb(
    executable: Path,
    root: Path,
    *,
    exe_argv0: bool,
    explicit_executable: bool = True,
    locale_c: bool = False,
) -> dict:
    data = root / "数据库 数据"
    temporary = root / "临时 文件"
    temporary.mkdir(parents=True)
    data.mkdir()
    password_file = temporary / "密码 文件.pgpass"
    # Match pg0's synthetic doctor password and lack of a trailing newline.
    password_file.write_bytes(b"postgres")
    stdout_file, stderr_file = root / "stdout.txt", root / "stderr.txt"
    argv0 = executable if exe_argv0 else executable.with_suffix("")
    arguments = [
        str(argv0),
        "--auth",
        "password",
        "--pgdata",
        str(data),
        "--encoding",
        "UTF8",
        "--pwfile",
        str(password_file),
        "--username",
        "postgres",
    ]
    if locale_c:
        arguments.extend(["--locale", "C"])
    application = {"executable": str(executable)} if explicit_executable else {}
    environment = os.environ.copy()
    environment.update(TEMP=str(temporary), TMP=str(temporary))
    result = {
        "argv0HasExe": exe_argv0,
        "explicitExecutable": explicit_executable,
        "locale": "C" if locale_c else "default",
        "cleanupSafe": True,
        "returnCode": -1,
    }
    process = None
    try:
        # File capture does not wait for EOF from inherited grandchild pipes.
        with stdout_file.open("wb") as stdout, stderr_file.open("wb") as stderr:
            process = subprocess.Popen(
                arguments,
                **application,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=_NO_WINDOW,
            )
            try:
                code = process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                result["timeoutSeconds"] = 45
                # Only the process we just created and its descendants are
                # terminated. Never select PostgreSQL processes by image name.
                killer = subprocess.run(
                    [
                        str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"),
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ],
                    capture_output=True,
                    check=False,
                    timeout=10,
                    creationflags=_NO_WINDOW,
                )
                result["cleanupSafe"] = killer.returncode == 0
                code = process.wait(timeout=5)
            result["returnCode"] = code
            result["returnCodeHex"] = f"0x{code & 0xFFFFFFFF:08X}"
    except Exception as error:
        result["diagnosticError"] = type(error).__name__
        if process is not None and process.poll() is None:
            result["cleanupSafe"] = False
    for label, path in (("stdoutTail", stdout_file), ("stderrTail", stderr_file)):
        if path.is_file():
            result[label] = _file_tail(path)
    result["pgVersionExists"] = (data / "PG_VERSION").is_file()
    result["postgresqlConfExists"] = (data / "postgresql.conf").is_file()
    result["dataInitialized"] = (
        result["pgVersionExists"] and result["postgresqlConfExists"]
    )
    return result


def run_initdb_matrix(postgres_root: Path, *, copy_installation: bool = False) -> dict:
    """Initialize disposable clusters; report results without starting services.

    Both the ordinary-Python early gate and frozen doctor's failure handler use
    this matrix. The caller must still fail its gate or re-raise its original
    startup exception, regardless of the diagnostic results.
    """
    report = {"cases": []}
    root = None
    dll = None
    original = None
    cleanup_safe = True
    try:
        if not (
            sys.platform == "win32"
            and os.getenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE") == "1"
            and os.getenv("GITHUB_ACTIONS") == "true"
            and os.getenv("RUNNER_ENVIRONMENT") == "github-hosted"
        ):
            report["skipped"] = "requires_hosted_windows_package_doctor"
            return report
        executable = postgres_root / "bin/initdb.exe"
        if not executable.is_file():
            report["diagnosticError"] = "missing_initdb_executable"
            return report
        dll = _DllDirectory()
        original = dll.get()
        root = Path(tempfile.mkdtemp(prefix="初始化 诊断-"))
        report["temporaryRootAcl"] = {"before": windows_acl_snapshot(root)}
        private_directory_permissions(root)
        report["temporaryRootAcl"]["after"] = windows_acl_snapshot(root)
        if copy_installation:
            installation = root / "安装 目录"
            shutil.copytree(postgres_root, installation)
            executable = installation / "bin/initdb.exe"
        report["inheritedDllDirectory"] = original
        report["dllDirectoryChangeApplied"] = bool(original)
        report["restrictedExecAlreadySet"] = os.getenv("PG_RESTRICT_EXEC") == "1"
        for label, cleared, exe_argv0, explicit_executable, locale_c in (
            ("继承 DLL 无扩展名", False, False, True, False),
            ("清理 DLL 无扩展名", True, False, True, False),
            ("清理 DLL 有扩展名", True, True, True, False),
            ("省略 executable 默认 locale", True, True, False, False),
            ("指定 executable locale C", True, True, True, True),
            ("省略 executable locale C", True, True, False, True),
        ):
            # Empty GetDllDirectory cannot distinguish NULL from empty-string
            # mode. Preserve it; argv0 comparisons still work in the early gate.
            if original:
                dll.set(None if cleared else original)
            case = _probe_initdb(
                executable,
                root / label,
                exe_argv0=exe_argv0,
                explicit_executable=explicit_executable,
                locale_c=locale_c,
            )
            case.update(label=label, dllDirectoryCleared=cleared)
            report["cases"].append(case)
            cleanup_safe = cleanup_safe and case["cleanupSafe"]
            if not cleanup_safe or "timeoutSeconds" in case:
                break
    except Exception as error:
        report["diagnosticError"] = type(error).__name__
        cleanup_safe = False
    finally:
        if dll is not None and original is not None:
            try:
                if original:
                    dll.set(original)
                report["restored"] = dll.get() == original
            except Exception as error:
                report["restoreError"] = type(error).__name__
        if root is not None:
            if cleanup_safe:
                try:
                    shutil.rmtree(root)
                except Exception as error:
                    report["cleanupError"] = type(error).__name__
            else:
                # Never remove a cluster whose child-tree cleanup is unverified.
                report["retainedDiagnosticRoot"] = str(root)
    return report
