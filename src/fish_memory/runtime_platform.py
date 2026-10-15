"""Small OS boundaries shared by the packaged desktop runtime."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

SHUTDOWN_MESSAGE = "__FISH_MEMORY_SHUTDOWN__"
_T = TypeVar("_T")


async def with_parent_shutdown(work: Awaitable[_T], *, signals: bool = False) -> _T:
    task = asyncio.current_task()

    def cancel() -> None:
        if task is not None and not task.cancelling():
            task.cancel()

    with shutdown_handlers(cancel, signals=signals):
        return await work


@contextmanager
def shutdown_handlers(
    callback: Callable[[], None], *, signals: bool = True, parent: bool = True
) -> Iterator[None]:
    """Route termination to the loop, including Electron's Windows stdin pipe.

    Windows cannot deliver POSIX SIGTERM through ChildProcess.kill. The parent
    opts into this private pipe explicitly; ordinary CLI stdin is never read.
    EOF also closes the runtime if Electron dies before sending the message.
    """
    loop = asyncio.get_running_loop()
    previous = {}
    active = threading.Event()
    active.set()

    def notify() -> None:
        if active.is_set() and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(callback)
            except RuntimeError:
                pass  # The loop may finish between the check and notification.

    if signals:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.getsignal(number)
            if sys.platform == "win32":
                signal.signal(number, lambda *_args: notify())
            else:
                loop.add_signal_handler(number, callback)

    if (
        parent
        and sys.platform == "win32"
        and os.getenv("FISH_MEMORY_PARENT_CONTROL") == "stdin"
        and sys.stdin is not None
    ):
        stream = sys.stdin

        def read_parent() -> None:
            try:
                while active.is_set():
                    line = stream.readline()
                    if not line or line.rstrip("\r\n") == SHUTDOWN_MESSAGE:
                        notify()
                        return
            except (OSError, ValueError):
                notify()

        threading.Thread(
            target=read_parent, name="desktop-parent-control", daemon=True
        ).start()
    try:
        yield
    finally:
        active.clear()
        for number, handler in previous.items():
            if sys.platform != "win32":
                loop.remove_signal_handler(number)
            signal.signal(number, handler)


def private_file_permissions(path: Path, descriptor: int) -> None:
    """Restrict a newly-created, still-empty file before writing secrets."""
    if os.name != "nt":
        os.fchmod(descriptor, 0o600)
        return
    _windows_private_file(path)


def private_file_is_private(path: Path) -> bool:
    if os.name == "nt":
        return _windows_private_file(path, check=True)
    return not bool(path.stat().st_mode & 0o077)


def private_directory_permissions(path: Path) -> None:
    """Restrict a newly-created empty directory to the current user's SID.

    Inheritable user ACEs also permit child processes using a restricted token;
    owner-rights and Administrators-only ACEs are not equivalent to that user.
    """
    if not path.is_dir() or any(path.iterdir()):
        raise ValueError("private directory permissions require an empty directory")
    if sys.platform != "win32":
        path.chmod(0o700)
        return
    _windows_private_file(path, directory=True)


def _windows_private_file(
    path: Path, *, check: bool = False, directory: bool = False
) -> bool:
    # chmod(0600) only changes Windows' read-only flag. Use a protected DACL
    # granting this token's user access, without inherited Users/Everyone ACEs.
    if sys.platform != "win32":
        raise OSError("Windows ACLs are only available on Windows")
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer = ctypes.c_void_p
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [pointer]
    kernel.LocalFree.restype = pointer
    advapi.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        pointer,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.ConvertSidToStringSidW.argtypes = [pointer, ctypes.POINTER(pointer)]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(pointer),
        pointer,
    ]
    advapi.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, pointer]
    token = wintypes.HANDLE()
    sid_text = pointer()
    security = pointer()
    if not advapi.OpenProcessToken(
        kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        required = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(required))
        buffer = ctypes.create_string_buffer(required.value)
        if not advapi.GetTokenInformation(
            token, 1, buffer, required, ctypes.byref(required)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # TOKEN_USER begins with SID_AND_ATTRIBUTES; its first field is PSID.
        sid = ctypes.cast(buffer, ctypes.POINTER(pointer))[0]
        if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        ace_flags = "OICI" if directory else ""
        sddl = f"D:P(A;{ace_flags};FA;;;{ctypes.wstring_at(sid_text)})"
        if check:
            advapi.GetFileSecurityW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                pointer,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            required = wintypes.DWORD()
            advapi.GetFileSecurityW(str(path), 4, None, 0, ctypes.byref(required))
            descriptor = ctypes.create_string_buffer(required.value)
            if not advapi.GetFileSecurityW(
                str(path), 4, descriptor, required, ctypes.byref(required)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return _windows_dacl_is_private(
                advapi, descriptor, sid, directory=directory
            )
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl, 1, ctypes.byref(security), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not advapi.SetFileSecurityW(str(path), 0x80000004, security):
            raise ctypes.WinError(ctypes.get_last_error())
        if not _windows_private_file(path, check=True, directory=directory):
            raise PermissionError("Windows private file ACL verification failed")
        return True
    finally:
        if security:
            kernel.LocalFree(security)
        if sid_text:
            kernel.LocalFree(sid_text)
        kernel.CloseHandle(token)


def _windows_dacl_is_private(advapi, descriptor, user_sid, *, directory=False) -> bool:
    """Validate actual ACEs, allowing Windows to normalize descriptor flags."""
    if sys.platform != "win32":
        raise OSError("Windows ACLs are only available on Windows")
    import ctypes
    from ctypes import wintypes

    class AclSizeInformation(ctypes.Structure):
        _fields_ = [
            ("ace_count", wintypes.DWORD),
            ("bytes_in_use", wintypes.DWORD),
            ("bytes_free", wintypes.DWORD),
        ]

    class AccessAllowedAce(ctypes.Structure):
        _fields_ = [
            ("ace_type", wintypes.BYTE),
            ("flags", wintypes.BYTE),
            ("size", wintypes.WORD),
            ("mask", wintypes.DWORD),
            ("sid_start", wintypes.DWORD),
        ]

    pointer = ctypes.c_void_p
    advapi.GetSecurityDescriptorControl.argtypes = [
        pointer,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetSecurityDescriptorDacl.argtypes = [
        pointer,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(pointer),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi.GetAclInformation.argtypes = [pointer, pointer, wintypes.DWORD, ctypes.c_int]
    advapi.GetAce.argtypes = [pointer, wintypes.DWORD, ctypes.POINTER(pointer)]
    advapi.IsValidSid.argtypes = [pointer]
    advapi.GetLengthSid.argtypes = [pointer]
    advapi.GetLengthSid.restype = wintypes.DWORD
    advapi.EqualSid.argtypes = [pointer, pointer]

    control, revision = wintypes.WORD(), wintypes.DWORD()
    if not advapi.GetSecurityDescriptorControl(
        descriptor, ctypes.byref(control), ctypes.byref(revision)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not control.value & 0x1000:  # SE_DACL_PROTECTED
        return False
    present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), pointer()
    if not advapi.GetSecurityDescriptorDacl(
        descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not present.value or not dacl.value:
        return False  # An absent or NULL DACL grants everyone access.
    info = AclSizeInformation()
    if not advapi.GetAclInformation(dacl, ctypes.byref(info), ctypes.sizeof(info), 2):
        raise ctypes.WinError(ctypes.get_last_error())
    if info.ace_count != 1:
        return False
    ace = pointer()
    if not advapi.GetAce(dacl, 0, ctypes.byref(ace)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not ace.value:
        return False
    entry = AccessAllowedAce.from_address(ace.value)
    sid_offset = AccessAllowedAce.sid_start.offset
    if (
        entry.ace_type != 0  # ACCESS_ALLOWED_ACE_TYPE
        or entry.flags != (0x03 if directory else 0)  # OI | CI for directories only.
        or entry.mask != 0x1F01FF  # FILE_ALL_ACCESS
        or entry.size < sid_offset + 8
    ):
        return False
    sid = ace.value + sid_offset
    return bool(
        advapi.IsValidSid(sid)
        and advapi.GetLengthSid(sid) <= entry.size - sid_offset
        and advapi.EqualSid(sid, user_sid)
    )


def windows_acl_snapshot(path: Path) -> dict[str, str]:
    """Read owner SID and DACL for disposable Windows CI diagnostics."""
    if sys.platform != "win32":
        raise OSError("Windows ACLs are only available on Windows")
    import ctypes
    from ctypes import wintypes

    pointer = ctypes.c_void_p
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [pointer]
    kernel.LocalFree.restype = pointer
    advapi.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        pointer,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetSecurityDescriptorOwner.argtypes = [
        pointer,
        ctypes.POINTER(pointer),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi.ConvertSidToStringSidW.argtypes = [pointer, ctypes.POINTER(pointer)]
    advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        pointer,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(pointer),
        pointer,
    ]
    information = 0x05  # OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
    required = wintypes.DWORD()
    advapi.GetFileSecurityW(str(path), information, None, 0, ctypes.byref(required))
    if not required.value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(required.value)
    if not advapi.GetFileSecurityW(
        str(path), information, descriptor, required, ctypes.byref(required)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    owner, owner_text, sddl = pointer(), pointer(), pointer()
    defaulted = wintypes.BOOL()
    try:
        if not advapi.GetSecurityDescriptorOwner(
            descriptor, ctypes.byref(owner), ctypes.byref(defaulted)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not advapi.ConvertSidToStringSidW(owner, ctypes.byref(owner_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor, 1, information, ctypes.byref(sddl), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"owner": ctypes.wstring_at(owner_text), "sddl": ctypes.wstring_at(sddl)}
    finally:
        if sddl:
            kernel.LocalFree(sddl)
        if owner_text:
            kernel.LocalFree(owner_text)
