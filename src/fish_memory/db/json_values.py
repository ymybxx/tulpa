"""Canonical, bounded, credential-safe JSON for database persistence."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEGMENT_RE = re.compile(r"[./:-]+")
_NON_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")

_FORBIDDEN_NORMALIZED_KEYS = {
    "secret",
    "password",
    "authorization",
    "api_key",
    "access_token",
    "refresh_token",
    "client_secret",
    "secret_key",
    "bearer",
    "token",
}
_FORBIDDEN_ATOMIC_PARTS = {
    "secret",
    "password",
    "authorization",
    "bearer",
    "token",
}


def canonical_json_object(
    value: Mapping[str, object] | None,
    *,
    label: str,
    maximum_bytes: int,
    reserve_bytes: int = 0,
    none_as_empty: bool = False,
) -> str | None:
    normalized: Mapping[str, object] | None = value
    if normalized is None and none_as_empty:
        normalized = {}
    if normalized is None:
        return None
    if not isinstance(normalized, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    _reject_credential_keys(normalized, label=label)
    try:
        payload = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON values") from exc
    if canonical_json_size(payload) + reserve_bytes > maximum_bytes:
        raise ValueError(f"{label} exceeds {maximum_bytes} UTF-8 bytes")
    return payload


def canonical_json_size(value: Mapping[str, object] | str) -> int:
    if isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return len(payload.encode("utf-8"))


def _reject_credential_keys(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{label} keys must be strings")
            candidates = (raw_key, *_KEY_SEGMENT_RE.split(raw_key))
            for raw_segment in candidates:
                normalized = _normalize_key(raw_segment)
                parts = tuple(part for part in normalized.split("_") if part)
                if (
                    normalized in _FORBIDDEN_NORMALIZED_KEYS
                    or normalized.endswith("_api_key")
                    or any(part in _FORBIDDEN_ATOMIC_PARTS for part in parts)
                ):
                    raise ValueError(f"{label} contains a credential-like key")
            _reject_credential_keys(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_credential_keys(child, label=label)


def _normalize_key(value: str) -> str:
    with_boundaries = _CAMEL_BOUNDARY_RE.sub("_", value)
    return _NON_IDENTIFIER_RE.sub("_", with_boundaries.casefold()).strip("_")
