from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from fish_memory.auth.credentials import (
    ApiKeyCredentialStore,
    EnvironmentApiKeyRotationError,
)

DEFAULT_API_KEY_FILE = ".cache/fish-memory/api-key"


class _AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="FISH_MEMORY_API_KEY",
    )
    api_key_file: str = Field(
        default=DEFAULT_API_KEY_FILE,
        validation_alias="FISH_MEMORY_API_KEY_FILE",
    )


def run_auth_command(
    arguments: list[str],
    *,
    api_key: str | None,
    key_file: Path,
) -> int:
    if arguments not in (["show"], ["rotate"]):
        print("Usage: fish-memory auth show|rotate", file=sys.stderr)
        return 2
    store = ApiKeyCredentialStore(api_key=api_key, key_file=key_file)
    try:
        key = store.current_key() if arguments[0] == "show" else store.rotate()
    except EnvironmentApiKeyRotationError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(key)
    return 0


def run_auth_command_from_environment(arguments: list[str]) -> int:
    settings = _AuthSettings()
    return run_auth_command(
        arguments,
        api_key=(
            settings.api_key.get_secret_value()
            if settings.api_key is not None
            else None
        ),
        key_file=Path(settings.api_key_file),
    )
