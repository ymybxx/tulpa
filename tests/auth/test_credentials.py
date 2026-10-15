from __future__ import annotations

import os
from pathlib import Path

import pytest

import fish_memory.auth.credentials as credentials_module
from fish_memory.auth.credentials import (
    ApiKeyCredentialStore,
    EnvironmentApiKeyRotationError,
)
from fish_memory.runtime_platform import private_file_permissions
from tests.private_file_helpers import allow_public_read, assert_private_file


def test_generates_private_api_key_once_and_reuses_it(tmp_path: Path) -> None:
    key_file = tmp_path / "auth" / "api-key"

    first = ApiKeyCredentialStore(api_key=None, key_file=key_file)
    second = ApiKeyCredentialStore(api_key=None, key_file=key_file)

    assert first.was_generated is True
    assert second.was_generated is False
    assert first.current_key() == second.current_key()
    assert first.current_key().startswith("fm_")
    assert_private_file(key_file)


def test_environment_api_key_takes_precedence_without_writing_file(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "api-key"

    store = ApiKeyCredentialStore(api_key="fm_configured", key_file=key_file)

    assert store.current_key() == "fm_configured"
    assert store.was_generated is False
    assert not key_file.exists()


def test_rotate_replaces_file_backed_key(tmp_path: Path, monkeypatch) -> None:
    protected = []

    def protect_empty(path, descriptor):
        assert path.read_bytes() == b""
        private_file_permissions(path, descriptor)
        protected.append(path)

    monkeypatch.setattr(credentials_module, "private_file_permissions", protect_empty)
    key_file = tmp_path / "api-key"
    store = ApiKeyCredentialStore(api_key=None, key_file=key_file)
    original = store.current_key()

    replacement = store.rotate()

    assert replacement != original
    assert key_file.read_text(encoding="utf-8").strip() == replacement
    reloaded = ApiKeyCredentialStore(api_key=None, key_file=key_file)
    assert reloaded.current_key() == replacement
    assert_private_file(key_file)
    assert protected[0] == key_file
    assert len(protected) == 2 and protected[1] != key_file


def test_environment_api_key_cannot_be_rotated_by_cli(tmp_path: Path) -> None:
    store = ApiKeyCredentialStore(
        api_key="fm_configured",
        key_file=tmp_path / "api-key",
    )

    with pytest.raises(EnvironmentApiKeyRotationError, match="environment"):
        store.rotate()


def test_rejects_publicly_readable_key_file(tmp_path: Path) -> None:
    key_file = tmp_path / "api-key"
    key_file.write_text("fm_exposed\n", encoding="utf-8")
    allow_public_read(key_file)

    with pytest.raises(PermissionError, match="accessible by other users"):
        ApiKeyCredentialStore(api_key=None, key_file=key_file)


@pytest.mark.parametrize("rotate", [False, True], ids=["create", "rotate"])
def test_permission_failure_leaves_no_partial_key(tmp_path, monkeypatch, rotate):
    path = tmp_path / "api-key"
    store = ApiKeyCredentialStore(api_key=None, key_file=path) if rotate else None
    original = store.current_key() if store is not None else None
    descriptors = []

    def fail_protection(temporary, descriptor):
        assert temporary.read_bytes() == b""
        descriptors.append(descriptor)
        raise PermissionError("cannot protect file")

    monkeypatch.setattr(credentials_module, "private_file_permissions", fail_protection)
    with pytest.raises(PermissionError, match="cannot protect file"):
        if store is None:
            ApiKeyCredentialStore(api_key=None, key_file=path)
        else:
            store.rotate()

    assert list(tmp_path.iterdir()) == ([path] if rotate else [])
    if store is not None:
        assert store.current_key() == original
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
