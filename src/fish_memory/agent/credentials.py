from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from pydantic import SecretStr

from fish_memory.runtime_platform import (
    private_file_is_private,
    private_file_permissions,
)

_CREDENTIAL_REF_PATTERN = re.compile(r"^[a-z][a-z0-9._:-]{0,127}$")
_MAX_SECRET_CHARS = 8_192


class AgentCredentialLookupError(LookupError):
    def __init__(self, credential_ref: str) -> None:
        self.credential_ref = credential_ref
        super().__init__(credential_ref)


class AgentCredentialResolver:
    def __init__(
        self,
        credentials: Mapping[str, SecretStr],
        *,
        credentials_file: Path | None = None,
    ) -> None:
        self._credentials = dict(credentials)
        self._credentials_file = (
            credentials_file.expanduser().resolve()
            if credentials_file is not None
            else None
        )
        self._stored_credentials: dict[str, SecretStr] = {}
        self._transient_credentials: dict[str, SecretStr] = {}
        self._update_locks: dict[str, asyncio.Lock] = {}
        self._lock = threading.RLock()
        if self._credentials_file is not None:
            self._stored_credentials = self._read_stored_credentials()

    def resolve(self, credential_ref: str) -> str:
        with self._lock:
            credential = self._transient_credentials.get(credential_ref)
            if credential is None:
                credential = self._credentials.get(credential_ref)
            if credential is None:
                credential = self._stored_credentials.get(credential_ref)
            if credential is None:
                raise AgentCredentialLookupError(credential_ref)
            return credential.get_secret_value()

    def resolve_optional(self, credential_ref: str) -> str | None:
        try:
            return self.resolve(credential_ref)
        except AgentCredentialLookupError:
            return None

    def stage(self, credential_ref: str, secret: str) -> None:
        reference = _validated_reference(credential_ref)
        if not reference.startswith("probe:"):
            raise ValueError("Only probe credentials may be staged")
        normalized_secret = _validated_secret(secret)
        with self._lock:
            self._transient_credentials[reference] = SecretStr(normalized_secret)

    def unstage(self, credential_ref: str) -> None:
        reference = _validated_reference(credential_ref)
        with self._lock:
            self._transient_credentials.pop(reference, None)

    @asynccontextmanager
    async def update_lock(self, credential_ref: str) -> AsyncIterator[None]:
        reference = _validated_reference(credential_ref)
        with self._lock:
            update_lock = self._update_locks.setdefault(
                reference,
                asyncio.Lock(),
            )
        async with update_lock:
            yield

    def store(self, credential_ref: str, secret: str) -> None:
        reference = _validated_reference(credential_ref)
        if not reference.startswith("fish:"):
            raise ValueError("Only Fish override credentials may be persisted")
        self._store_persisted(reference, secret)

    def store_global(self, credential_ref: str, secret: str) -> None:
        reference = _validated_reference(credential_ref)
        if not reference.startswith("global:"):
            raise ValueError("Only global Agent credentials may be persisted")
        self._store_persisted(reference, secret)

    def clear_global_credentials(self) -> None:
        if self._credentials_file is None:
            return
        with self._lock:
            updated = {
                reference: credential
                for reference, credential in self._stored_credentials.items()
                if not reference.startswith("global:")
            }
            if len(updated) == len(self._stored_credentials):
                return
            self._write_stored_credentials(updated)
            self._stored_credentials = updated

    def _store_persisted(self, reference: str, secret: str) -> None:
        normalized_secret = _validated_secret(secret)
        if self._credentials_file is None:
            raise RuntimeError("Agent credential persistence is not configured")
        with self._lock:
            updated = dict(self._stored_credentials)
            updated[reference] = SecretStr(normalized_secret)
            self._write_stored_credentials(updated)
            self._stored_credentials = updated

    def remove(self, credential_ref: str) -> None:
        reference = _validated_reference(credential_ref)
        if self._credentials_file is None:
            raise RuntimeError("Agent credential persistence is not configured")
        with self._lock:
            if reference not in self._stored_credentials:
                return
            updated = dict(self._stored_credentials)
            del updated[reference]
            self._write_stored_credentials(updated)
            self._stored_credentials = updated

    def _read_stored_credentials(self) -> dict[str, SecretStr]:
        assert self._credentials_file is not None
        try:
            self._validate_permissions()
            raw = json.loads(self._credentials_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if (
            not isinstance(raw, dict)
            or raw.get("version") != 1
            or not isinstance(raw.get("credentials"), dict)
        ):
            raise ValueError("Agent credentials file is malformed")
        normalized: dict[str, SecretStr] = {}
        stored = cast(dict[object, object], raw["credentials"])
        for raw_ref, raw_secret in stored.items():
            reference = _validated_reference(raw_ref)
            if not reference.startswith(("fish:", "global:")):
                continue
            secret = _validated_secret(raw_secret)
            normalized[reference] = SecretStr(secret)
        return normalized

    def _write_stored_credentials(
        self,
        credentials: Mapping[str, SecretStr],
    ) -> None:
        assert self._credentials_file is not None
        self._credentials_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._credentials_file.with_name(
            f".{self._credentials_file.name}.{secrets.token_hex(8)}"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            payload = {
                "version": 1,
                "credentials": {
                    reference: credential.get_secret_value()
                    for reference, credential in sorted(credentials.items())
                },
            }
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                private_file_permissions(temporary, stream.fileno())
                json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self._credentials_file)
            if os.name == "posix":
                directory = os.open(self._credentials_file.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_permissions(self) -> None:
        assert self._credentials_file is not None
        if not private_file_is_private(self._credentials_file):
            raise PermissionError(
                "Agent credentials file is accessible by other users: "
                f"{self._credentials_file}"
            )

    def __repr__(self) -> str:
        return "AgentCredentialResolver(...)"


def _validated_reference(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Agent credential reference must be text")
    normalized = value.strip()
    if _CREDENTIAL_REF_PATTERN.fullmatch(normalized) is None:
        raise ValueError("Agent credential reference is invalid")
    return normalized


def _validated_secret(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Agent credential secret must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError("Agent credential secret cannot be blank")
    if len(normalized) > _MAX_SECRET_CHARS:
        raise ValueError("Agent credential secret is too long")
    return normalized
