from __future__ import annotations

from pathlib import Path

from fish_memory.auth.cli import (
    run_auth_command,
    run_auth_command_from_environment,
)


def test_auth_show_prints_effective_key(
    tmp_path: Path,
    capsys,
) -> None:
    key_file = tmp_path / "api-key"

    exit_code = run_auth_command(
        ["show"],
        api_key=None,
        key_file=key_file,
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip().startswith("fm_")


def test_auth_rotate_prints_replacement_key(
    tmp_path: Path,
    capsys,
) -> None:
    key_file = tmp_path / "api-key"
    run_auth_command(["show"], api_key=None, key_file=key_file)
    original = capsys.readouterr().out.strip()

    exit_code = run_auth_command(
        ["rotate"],
        api_key=None,
        key_file=key_file,
    )

    replacement = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert replacement.startswith("fm_")
    assert replacement != original


def test_environment_command_reads_api_key_from_dotenv(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FISH_MEMORY_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "FISH_MEMORY_API_KEY=fm_dotenv_key\n"
        "FISH_MEMORY_API_KEY_FILE=ignored-key-file\n",
        encoding="utf-8",
    )

    assert run_auth_command_from_environment(["show"]) == 0
    assert capsys.readouterr().out.strip() == "fm_dotenv_key"


def test_auth_rotate_explains_environment_backed_key(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = run_auth_command(
        ["rotate"],
        api_key="fm_managed_key",
        key_file=tmp_path / "unused-api-key",
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "FISH_MEMORY_API_KEY" in captured.err
    assert "restart" in captured.err
