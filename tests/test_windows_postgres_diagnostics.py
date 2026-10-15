from pathlib import Path
from subprocess import CompletedProcess

import pytest

from fish_memory import windows_postgres_diagnostics as diagnostics


@pytest.fixture
def hosted_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    monkeypatch.setattr(
        diagnostics, "private_directory_permissions", lambda _path: None
    )
    monkeypatch.setattr(
        diagnostics,
        "windows_acl_snapshot",
        lambda _path: {"owner": "test-owner", "sddl": "test-sddl"},
    )
    binaries = tmp_path / "installation" / "18.1.0" / "bin"
    binaries.mkdir(parents=True)
    for name in ("initdb.exe", "postgres.exe", "pg_ctl.exe"):
        (binaries / name).touch()
    return tmp_path


class FakeDllDirectory:
    def __init__(self):
        self.value = "C:/frozen/_internal"
        self.calls = []

    def get(self):
        return self.value

    def set(self, value):
        self.calls.append(value)
        self.value = value or ""


def test_loader_probe_records_status_and_restores_inherited_directory(
    hosted_windows, monkeypatch
):
    dll = FakeDllDirectory()
    monkeypatch.setattr(diagnostics, "_DllDirectory", lambda: dll)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[1:] == ["--version"]
        assert kwargs["timeout"] == 10
        code = 0xC0000139 if dll.value else 0
        return CompletedProcess(command, code, "version" if not code else "", "")

    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    monkeypatch.setattr(diagnostics, "run_initdb_matrix", lambda _root: {"cases": []})
    result = diagnostics.diagnose_postgres_loader(hosted_windows)
    assert [row["returnCodeHex"] for row in result["inherited"]] == ["0xC0000139"] * 3
    assert [row["returnCode"] for row in result["cleared"]] == [0] * 3
    assert dll.calls == [None, "C:/frozen/_internal"]
    assert result["restored"] is True
    assert len(calls) == 6


def test_loader_probe_restores_directory_even_when_comparison_raises(
    hosted_windows, monkeypatch
):
    dll = FakeDllDirectory()
    monkeypatch.setattr(diagnostics, "_DllDirectory", lambda: dll)

    def version(_executable):
        if not dll.value:
            raise OSError("diagnostic failure")
        return {"returnCode": 1}

    monkeypatch.setattr(diagnostics, "_version", version)
    monkeypatch.setattr(diagnostics, "run_initdb_matrix", lambda _root: {"cases": []})
    result = diagnostics.diagnose_postgres_loader(hosted_windows)
    assert result["diagnosticError"] == "OSError"
    assert dll.value == "C:/frozen/_internal"
    assert dll.calls == [None, "C:/frozen/_internal"]


def test_loader_probe_leaves_empty_dll_directory_mode_unchanged(
    hosted_windows, monkeypatch
):
    dll = FakeDllDirectory()
    dll.value = ""
    monkeypatch.setattr(diagnostics, "_DllDirectory", lambda: dll)
    monkeypatch.setattr(diagnostics, "_version", lambda _exe: {"returnCode": 1})
    monkeypatch.setattr(diagnostics, "run_initdb_matrix", lambda _root: {"cases": []})
    result = diagnostics.diagnose_postgres_loader(hosted_windows)
    assert result["comparisonSkipped"] == "no_extra_dll_directory"
    assert dll.calls == []


def test_loader_probe_does_not_change_normal_development_environment(monkeypatch):
    monkeypatch.delenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE", raising=False)
    monkeypatch.setattr(
        diagnostics, "_DllDirectory", lambda: pytest.fail("unexpected WinAPI")
    )
    assert "skipped" in diagnostics.diagnose_postgres_loader(Path("unused"))


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_home", [False, True])
async def test_failed_diagnostic_preserves_original_start_error(
    hosted_windows, monkeypatch, capsys, missing_home
):
    from fish_memory import desktop_runtime

    (hosted_windows / ".fish-memory-package-smoke").touch()
    if missing_home:
        monkeypatch.delenv("TULPA_PG0_HOME", raising=False)
    else:
        monkeypatch.setenv("TULPA_PG0_HOME", str(hosted_windows))
    original = RuntimeError("original database startup failure")
    calls = []

    class Database:
        def __init__(self, **kwargs):
            pass

        def start(self):
            calls.append("start")
            raise original

        def stop(self):
            calls.append("stop")

    def failed_diagnostic(_private_home):
        raise OSError("unavailable diagnostic path")

    monkeypatch.setattr("pg0.Pg0", Database)
    monkeypatch.setattr(diagnostics, "diagnose_postgres_loader", failed_diagnostic)
    with pytest.raises(RuntimeError) as caught:
        await desktop_runtime._probe_embedded_database(hosted_windows)
    assert caught.value is original
    assert calls == ["start", "stop"]
    expected = "KeyError" if missing_home else "OSError"
    assert f'"diagnosticError": "{expected}"' in capsys.readouterr().err


@pytest.mark.parametrize("original", ["C:/frozen/_internal", ""])
def test_initdb_matrix_covers_argv0_and_restores_exact_dll_mode(
    hosted_windows, monkeypatch, original
):
    dll = FakeDllDirectory()
    dll.value = original
    monkeypatch.setattr(diagnostics, "_DllDirectory", lambda: dll)
    calls = []

    def probe(executable, root, *, exe_argv0, explicit_executable, locale_c):
        assert executable.is_file()
        assert "安装 目录" in str(executable)
        calls.append((exe_argv0, dll.value, root, explicit_executable, locale_c))
        return {"returnCode": 0, "dataInitialized": True, "cleanupSafe": True}

    monkeypatch.setattr(diagnostics, "_probe_initdb", probe)
    result = diagnostics.run_initdb_matrix(
        hosted_windows / "installation/18.1.0", copy_installation=True
    )
    assert [call[0] for call in calls] == [False, False, True, True, True, True]
    assert [call[1] for call in calls] == [original, "", "", "", "", ""]
    assert [call[3:] for call in calls] == [
        (True, False),
        (True, False),
        (True, False),
        (False, False),
        (True, True),
        (False, True),
    ]
    assert len(result["cases"]) == 6
    assert all(case["dataInitialized"] for case in result["cases"])
    assert result["temporaryRootAcl"] == {
        "before": {"owner": "test-owner", "sddl": "test-sddl"},
        "after": {"owner": "test-owner", "sddl": "test-sddl"},
    }
    assert result["restored"] is True
    assert dll.value == original
    assert dll.calls == ([original, *([None] * 5), original] if original else [])
    assert all(not call[2].parent.exists() for call in calls)


def test_initdb_matrix_restores_dll_directory_after_internal_failure(
    hosted_windows, monkeypatch
):
    dll = FakeDllDirectory()
    monkeypatch.setattr(diagnostics, "_DllDirectory", lambda: dll)

    def probe(_executable, _root, **_options):
        if not dll.value:
            raise OSError("diagnostic failure")
        return {"returnCode": 1, "dataInitialized": False, "cleanupSafe": True}

    monkeypatch.setattr(diagnostics, "_probe_initdb", probe)
    result = diagnostics.run_initdb_matrix(hosted_windows / "installation/18.1.0")
    assert result["diagnosticError"] == "OSError"
    assert result["restored"] is True
    assert dll.value == "C:/frozen/_internal"
    assert len(result["cases"]) == 1
    # The test owns this newly created directory; production conservatively
    # retains it whenever child-tree cleanup cannot be verified.
    diagnostics.shutil.rmtree(result["retainedDiagnosticRoot"])


@pytest.mark.parametrize("code", [0, -1073741515])
def test_initdb_probe_matches_rust_argv0_and_records_real_completion(
    tmp_path, monkeypatch, code
):
    executable = tmp_path / "安装 目录/bin/initdb.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()

    class Process:
        def __init__(self, command, **kwargs):
            assert command[0] == str(executable.with_suffix(""))
            assert kwargs["executable"] == str(executable)
            assert kwargs["creationflags"] == 0x08000000
            assert kwargs["stdin"] == diagnostics.subprocess.DEVNULL
            assert "--locale" not in command
            data = Path(command[command.index("--pgdata") + 1])
            assert data.is_dir()  # pg0 creates its empty data directory first.
            password_file = Path(command[command.index("--pwfile") + 1])
            assert password_file.read_bytes() == b"postgres"
            if code == 0:
                (data / "PG_VERSION").write_text("18")
                (data / "postgresql.conf").touch()
            kwargs["stdout"].write(b"x" * 3000)

        def wait(self, *, timeout):
            assert timeout == 45
            return code

    monkeypatch.setattr(diagnostics.subprocess, "Popen", Process)
    result = diagnostics._probe_initdb(executable, tmp_path / "初始化", exe_argv0=False)
    assert result["returnCode"] == code
    assert result["returnCodeHex"] == ("0x00000000" if code == 0 else "0xC0000135")
    assert result["dataInitialized"] is (code == 0)
    assert len(result["stdoutTail"]) == 2048


def test_initdb_timeout_only_terminates_its_owned_child_tree(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("SystemRoot", "C:/Windows")

    class Process:
        pid = 12345

        def __init__(self, _command, **_kwargs):
            self.waits = 0

        def wait(self, *, timeout):
            self.waits += 1
            if self.waits == 1:
                raise diagnostics.subprocess.TimeoutExpired("initdb", timeout)
            assert timeout == 5
            return 1

    def kill(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 10
        return CompletedProcess(command, 0)

    monkeypatch.setattr(diagnostics.subprocess, "Popen", Process)
    monkeypatch.setattr(diagnostics.subprocess, "run", kill)
    result = diagnostics._probe_initdb(
        tmp_path / "initdb.exe", tmp_path / "初始化", exe_argv0=True
    )
    assert result["timeoutSeconds"] == 45
    assert result["cleanupSafe"] is True
    assert result["returnCode"] == 1
    assert calls[0][1:] == ["/PID", "12345", "/T", "/F"]


def test_initdb_matrix_refuses_ordinary_development(monkeypatch):
    monkeypatch.delenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE", raising=False)
    monkeypatch.setattr(
        diagnostics, "_DllDirectory", lambda: pytest.fail("unexpected WinAPI")
    )
    result = diagnostics.run_initdb_matrix(Path("unused"))
    assert result["cases"] == []
    assert "skipped" in result


@pytest.mark.parametrize(
    ("explicit_executable", "locale_c"), [(False, False), (True, True), (False, True)]
)
def test_initdb_probe_isolates_application_name_and_locale(
    tmp_path, monkeypatch, explicit_executable, locale_c
):
    executable = tmp_path / "安装 目录/bin/initdb.exe"
    commands = []

    class Process:
        def __init__(self, command, **kwargs):
            commands.append(command)
            assert command[0] == str(executable)
            assert ("executable" in kwargs) is explicit_executable
            if explicit_executable:
                assert kwargs["executable"] == str(executable)
            assert ("--locale" in command) is locale_c
            if locale_c:
                assert command[command.index("--locale") + 1] == "C"
            assert kwargs["stdin"] == diagnostics.subprocess.DEVNULL
            assert kwargs["creationflags"] == 0x08000000
            data = Path(command[command.index("--pgdata") + 1])
            (data / "PG_VERSION").write_text("18")
            (data / "postgresql.conf").touch()

        def wait(self, *, timeout):
            assert timeout == 45
            return 0

    monkeypatch.setattr(diagnostics.subprocess, "Popen", Process)
    result = diagnostics._probe_initdb(
        executable,
        tmp_path / "初始化",
        exe_argv0=True,
        explicit_executable=explicit_executable,
        locale_c=locale_c,
    )
    assert len(commands) == 1
    assert result["returnCode"] == 0
    assert result["dataInitialized"] is True
    assert result["explicitExecutable"] is explicit_executable
    assert result["locale"] == ("C" if locale_c else "default")


def test_initdb_applies_private_directory_acl_before_copying_or_launching(
    hosted_windows, monkeypatch
):
    dll = FakeDllDirectory()
    monkeypatch.setattr(diagnostics, "_DllDirectory", lambda: dll)
    events = []

    def apply(path):
        assert path.is_dir()
        assert not list(path.iterdir())
        events.append("private-acl")

    def snapshot(_path):
        return {"owner": "S-1-test", "sddl": "after" if events else "before"}

    def probe(executable, _root, **_options):
        assert events == ["private-acl"]
        assert executable.is_file()
        return {"returnCode": 0, "dataInitialized": True, "cleanupSafe": True}

    monkeypatch.setattr(diagnostics, "private_directory_permissions", apply)
    monkeypatch.setattr(diagnostics, "windows_acl_snapshot", snapshot)
    monkeypatch.setattr(diagnostics, "_probe_initdb", probe)
    result = diagnostics.run_initdb_matrix(
        hosted_windows / "installation/18.1.0", copy_installation=True
    )
    assert len(result["cases"]) == 6
    assert result["temporaryRootAcl"]["before"]["sddl"] == "before"
    assert result["temporaryRootAcl"]["after"]["sddl"] == "after"
