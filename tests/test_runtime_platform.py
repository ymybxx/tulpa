from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from fish_memory import runtime_platform as runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [runtime.SHUTDOWN_MESSAGE + "\n", ""])
async def test_windows_parent_message_and_eof_trigger_graceful_cleanup(
    monkeypatch, message
):
    monkeypatch.setenv("FISH_MEMORY_PARENT_CONTROL", "stdin")
    monkeypatch.setattr(
        runtime, "sys", SimpleNamespace(platform="win32", stdin=io.StringIO(message))
    )
    stopped = asyncio.Event()
    with runtime.shutdown_handlers(stopped.set, signals=False):
        await asyncio.wait_for(stopped.wait(), 2)


@pytest.mark.asyncio
async def test_parent_reader_is_opt_in_and_does_not_consume_business_stdin(monkeypatch):
    stream = io.StringIO(runtime.SHUTDOWN_MESSAGE + "\n")
    monkeypatch.delenv("FISH_MEMORY_PARENT_CONTROL", raising=False)
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="win32", stdin=stream))
    with runtime.shutdown_handlers(
        lambda: pytest.fail("unexpected shutdown"), signals=False
    ):
        await asyncio.sleep(0)
    assert stream.tell() == 0


@pytest.mark.asyncio
async def test_parent_shutdown_cancels_work_but_allows_async_finally(monkeypatch):
    monkeypatch.setenv("FISH_MEMORY_PARENT_CONTROL", "stdin")
    monkeypatch.setattr(
        runtime, "sys", SimpleNamespace(platform="win32", stdin=io.StringIO(""))
    )
    cleanup = []

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleanup.append("closed")

    task = asyncio.create_task(runtime.with_parent_shutdown(work()))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert cleanup == ["closed"]


@pytest.mark.asyncio
async def test_windows_facial_reader_preserves_utf8_and_bounds_each_line(
    monkeypatch, tmp_path, capsys
):
    from fish_memory.facial_test import service

    payload = (
        json.dumps(
            {"id": "中文协议", "operation": "unknown"}, ensure_ascii=False
        ).encode()
        + b"\n"
    )
    stream = io.BytesIO(payload)
    monkeypatch.setattr(
        service,
        "sys",
        SimpleNamespace(platform="win32", stdin=SimpleNamespace(buffer=stream)),
    )
    await service.serve_runner(service.ConfiguredRunner(tmp_path), {"mode": "jev"})
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[1]["id"] == "中文协议"
    assert events[1]["message"] == "未知operation"
    stream = io.BytesIO(b"x" * (1024 * 1024 + 1))
    monkeypatch.setattr(service.sys.stdin, "buffer", stream)
    with pytest.raises(ValueError, match="size limit"):
        await service._read_windows_stdin_line()


def test_private_file_uses_real_platform_permissions(tmp_path):
    path = tmp_path / "private 中文.json"
    with path.open("w", encoding="utf-8") as output:
        runtime.private_file_permissions(path, output.fileno())
        output.write("test secret")
    assert runtime.private_file_is_private(path)
    assert path.read_text(encoding="utf-8") == "test secret"
    if os.name != "nt":
        path.chmod(0o644)
        assert not runtime.private_file_is_private(path)


@pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows DACL APIs")
@pytest.mark.parametrize("filename", ["private.json", "private 中文.json"])
def test_windows_private_file_dacl_only_grants_current_user(tmp_path, filename):
    path = tmp_path / filename
    try:
        with path.open("w") as output:
            runtime.private_file_permissions(path, output.fileno())
    except OSError as error:
        report = _windows_acl_report(path)
        pytest.fail(f"Private ACL write or verification failed: {error}; ACL: {report}")
    report = _windows_acl_report(path)
    assert report["protected"] is True, report
    assert report["users"] == [report["current"]], report
    assert report["rights"] == ["FullControl"], report
    assert runtime.private_file_is_private(path), report


@pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows DACL APIs")
@pytest.mark.parametrize("change", ["allow_everyone", "enable_inheritance"])
def test_windows_private_file_rejects_broadened_acl(tmp_path, change):
    path = tmp_path / "temporary acl test.json"
    with path.open("w") as output:
        runtime.private_file_permissions(path, output.fileno())
    if change == "allow_everyone":
        mutation = """
$world = [Security.Principal.SecurityIdentifier]::new('S-1-1-0')
$rule = [Security.AccessControl.FileSystemAccessRule]::new(
    $world, [Security.AccessControl.FileSystemRights]::Read,
    [Security.AccessControl.AccessControlType]::Allow)
$acl.AddAccessRule($rule)
"""
    else:
        mutation = "$acl.SetAccessRuleProtection($false, $true)"
    report = _windows_acl_report(path, mutation)
    assert not runtime.private_file_is_private(path), report


def _windows_acl_report(path, mutation=""):
    executable = shutil.which("pwsh")
    environment = {**os.environ, "FISH_TEST_PRIVATE_FILE": str(path)}
    if executable is None:
        executable = shutil.which("powershell.exe")
        # PowerShell 7's inherited module paths can break Windows PowerShell's
        # built-in Security module. Let the older edition build its own paths.
        environment = {
            key: value
            for key, value in environment.items()
            if key.casefold() != "psmodulepath"
        }
    assert executable is not None, "Windows ACL tests require PowerShell"
    script = """
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$acl = Get-Acl -LiteralPath $env:FISH_TEST_PRIVATE_FILE
"""
    if mutation:
        script += (
            mutation
            + """
# Only the DACL is under test. Copying Get-Acl's full descriptor can require
# SeSecurityPrivilege for the unrelated audit/SACL section on a standard token.
$access = [Security.AccessControl.AccessControlSections]::Access
$accessOnly = [Security.AccessControl.FileSecurity]::new()
$accessOnly.SetSecurityDescriptorSddlForm(
    $acl.GetSecurityDescriptorSddlForm($access), $access)
$file = [IO.FileInfo]::new($env:FISH_TEST_PRIVATE_FILE)
if ('System.IO.FileSystemAclExtensions' -as [type]) {
    [IO.FileSystemAclExtensions]::SetAccessControl($file, $accessOnly)
} else {
    $file.SetAccessControl($accessOnly)
}
$acl = Get-Acl -LiteralPath $env:FISH_TEST_PRIVATE_FILE
"""
        )
    script += """
@{ protected = $acl.AreAccessRulesProtected;
   sddl = $acl.Sddl;
   current = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value;
   rights = @($acl.Access | ForEach-Object { $_.FileSystemRights.ToString() });
   users = @($acl.Access | ForEach-Object {
       $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value
   }) } | ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [executable, "-NoProfile", "-NonInteractive", "-Command", script],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, (
        f"PowerShell ACL inspection failed ({result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows process pipes")
@pytest.mark.parametrize("message", [runtime.SHUTDOWN_MESSAGE + "\n", ""])
def test_real_windows_process_pipe_shutdown_flushes_utf8_and_finally(message):
    code = """
import asyncio
from fish_memory.desktop_runtime import _configure_protocol_stdio
from fish_memory.runtime_platform import with_parent_shutdown
_configure_protocol_stdio()
async def work():
    print('准备好了', flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await asyncio.sleep(0)
        print('已清理', flush=True)
try:
    asyncio.run(with_parent_shutdown(work()))
except asyncio.CancelledError:
    pass
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        input=message,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "FISH_MEMORY_PARENT_CONTROL": "stdin"},
        timeout=10,
        check=True,
    )
    assert result.stdout.splitlines() == ["准备好了", "已清理"]


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows process pipes")
def test_real_windows_facial_service_reads_utf8_jsonl_and_exits_on_eof(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fish_memory.desktop_runtime",
            "facial-test",
            "--config-directory",
            str(tmp_path),
        ],
        input=json.dumps({"id": "中文协议", "operation": "unknown"}, ensure_ascii=False)
        + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=True,
        env={k: v for k, v in os.environ.items() if k != "FISH_MEMORY_PARENT_CONTROL"},
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[0] == {"type": "ready", "mode": "jev"}
    assert events[1]["id"] == "中文协议"
    assert events[1]["type"] == "error"
    assert events[1]["message"] == "未知operation"


def test_windows_plugin_environment_keeps_systemroot_and_unicode(monkeypatch, tmp_path):
    from fish_memory.extensions.manager import PluginManager

    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-inherit")
    environment = PluginManager._environment(tmp_path)
    assert environment["SYSTEMROOT"] == r"C:\Windows"
    assert environment["PYTHONUTF8"] == "1"
    assert "UNRELATED_SECRET" not in environment


def test_private_directory_rejects_existing_content(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    content = directory / "keep.txt"
    content.write_text("keep")
    with pytest.raises(ValueError, match="empty directory"):
        runtime.private_directory_permissions(directory)
    assert content.read_text() == "keep"


def test_private_directory_windows_delegates_to_checked_inheritable_user_acl(
    tmp_path, monkeypatch
):
    directory = tmp_path / "fresh"
    directory.mkdir()
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="win32"))
    calls = []
    monkeypatch.setattr(
        runtime,
        "_windows_private_file",
        lambda path, **options: calls.append((path, options)),
    )
    runtime.private_directory_permissions(directory)
    assert calls == [(directory, {"directory": True})]


@pytest.mark.skipif(sys.platform != "win32", reason="requires real Windows DACL APIs")
def test_windows_private_directory_only_grants_inheritable_current_user_access(
    tmp_path,
):
    directory = tmp_path / "私有 临时目录"
    directory.mkdir(mode=0o700)
    before = runtime.windows_acl_snapshot(directory)
    runtime.private_directory_permissions(directory)
    after = runtime.windows_acl_snapshot(directory)
    report = _windows_acl_report(directory)
    assert report["protected"] is True, report
    assert report["users"] == [report["current"]], report
    assert report["rights"] == ["FullControl"], report
    assert "OICI" in after["sddl"], after
    assert before["owner"] == after["owner"]
    assert runtime._windows_private_file(directory, check=True, directory=True)
    child = directory / "子目录"
    child.mkdir()
    file = child / "继承.txt"
    file.write_text("private")
    child_report = _windows_acl_report(file)
    assert child_report["users"] == [report["current"]], child_report
    assert child_report["rights"] == ["FullControl"], child_report
