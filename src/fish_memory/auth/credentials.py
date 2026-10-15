from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Protocol

from fish_memory.runtime_platform import (
    private_file_is_private,
    private_file_permissions,
)


class ApiKeySettings(Protocol):
    @property
    def api_key(self) -> object | None: ...

    @property
    def api_key_file(self) -> str: ...


class EnvironmentApiKeyRotationError(RuntimeError):
    """Raised when a configured environment credential cannot be rotated."""


class ApiKeyCredentialStore:
    def __init__(self, *, api_key: str | None, key_file: Path) -> None:
        configured = api_key.strip() if api_key is not None else ""
        self._configured_key = configured or None
        self.key_file = key_file.expanduser().resolve()
        self.was_generated = False
        if self._configured_key is None:
            self.was_generated = self._ensure_key_file()

    @property
    def is_file_backed(self) -> bool:
        return self._configured_key is None

    def current_key(self) -> str:
        if self._configured_key is not None:
            return self._configured_key
        self._validate_permissions()
        key = self.key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"API key file is blank: {self.key_file}")
        return key

    def rotate(self) -> str:
        if self._configured_key is not None:
            raise EnvironmentApiKeyRotationError(
                "environment-backed API keys must be rotated by changing "
                "FISH_MEMORY_API_KEY and restarting fish-memory"
            )
        key = _generate_api_key()
        self._write_key(key, replace=True)
        return key

    def _ensure_key_file(self) -> bool:
        if self.key_file.exists():
            self._validate_permissions()
            self.current_key()
            return False
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        key = _generate_api_key()
        try:
            descriptor = os.open(
                self.key_file,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            self._validate_permissions()
            self.current_key()
            return False
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                private_file_permissions(self.key_file, stream.fileno())
                stream.write(f"{key}\n")
        except BaseException:
            self.key_file.unlink(missing_ok=True)
            raise
        return True

    def _write_key(self, key: str, *, replace: bool) -> None:
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.key_file.with_name(
            f".{self.key_file.name}.{secrets.token_hex(8)}"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                private_file_permissions(temporary, stream.fileno())
                stream.write(f"{key}\n")
            if replace:
                temporary.replace(self.key_file)
            else:
                temporary.rename(self.key_file)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_permissions(self) -> None:
        if not private_file_is_private(self.key_file):
            raise PermissionError(
                f"API key file is accessible by other users: {self.key_file}; "
                "restrict it to the current user"
            )


def _generate_api_key() -> str:
    return f"fm_{secrets.token_urlsafe(32)}"


def credential_store_from_settings(settings: ApiKeySettings) -> ApiKeyCredentialStore:
    configured = getattr(settings, "api_key", None)
    reveal = getattr(configured, "get_secret_value", None)
    api_key = reveal() if callable(reveal) else configured
    return ApiKeyCredentialStore(
        api_key=api_key if isinstance(api_key, str) else None,
        key_file=Path(getattr(settings, "api_key_file", ".cache/fish-memory/api-key")),
    )
