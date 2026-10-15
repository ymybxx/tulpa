from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID

_SLUG_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
_APP_BUNDLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,254}$")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")
_LABELED_SECRET_RE = re.compile(
    r"\b(api[ _-]?key|access[ _-]?token|auth(?:orization)?|bearer|"
    r"client[ _-]?secret|password|passwd|secret|token)\b\s*"
    r"(?::|=|\bis\b)\s*(?:\"[^\"]{4,}\"|'[^']{4,}'|[^\s,;]{4,})",
    re.IGNORECASE,
)
_CHINESE_LABELED_SECRET_RE = re.compile(
    r"(密码|密钥|令牌|访问令牌|鉴权)\s*(?:：|:|=|是)\s*"
    r"(?:\"[^\"]{4,}\"|'[^']{4,}'|[^\s，,；;]{4,})"
)
_ENV_SECRET_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|TOKEN|SECRET|PASSWORD|"
    r"PASSWD|PWD)[A-Z0-9_]*)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"\bAuthorization\s*[:=]\s*Bearer\s+[A-Za-z0-9._~+/=-]{8,}",
    re.IGNORECASE,
)
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_PROVIDER_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[0-9A-Z]{16})\b"
)
_URL_CREDENTIALS_RE = re.compile(
    r"(https?://)[^/@\s:]+:[^@/\s]+@",
    re.IGNORECASE,
)
_SECRET_QUERY_RE = re.compile(
    r"([?&](?:api[_-]?key|access[_-]?token|password|secret|token)=)[^&#\s]+",
    re.IGNORECASE,
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
    r"(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
    re.DOTALL,
)
_LONG_OPAQUE_VALUE_RE = re.compile(r"\b(?:[A-Fa-f0-9]{40,}|[A-Za-z0-9+/]{48,}={0,2})\b")
_FORBIDDEN_MEDIA_KEYS = {
    "base64",
    "frame",
    "frame_bytes",
    "image",
    "image_data",
    "image_url",
    "jpeg",
    "ocr_blocks",
    "ocr_tokens",
    "png",
    "pixel_data",
    "raw_ocr",
    "raw_pixels",
    "screen_shot",
    "screenshot",
}


class ObservationKind(StrEnum):
    STATE = "state"
    EVENT = "event"


@dataclass(frozen=True, slots=True)
class StateObservation:
    """One sample in a named state stream.

    ``scope_key`` identifies the stream (for example ``active_window``), while
    ``semantic_key`` identifies the current value. Repeated values extend one
    durable interval instead of creating frame-by-frame rows.
    """

    source: str
    observation_type: str
    scope_key: str
    semantic_key: str
    summary: str
    observed_at: datetime
    payload: Mapping[str, object] = field(default_factory=dict)
    app_bundle_id: str | None = None
    app_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _slug("source", self.source))
        object.__setattr__(
            self,
            "observation_type",
            _slug("observation_type", self.observation_type),
        )
        object.__setattr__(
            self,
            "scope_key",
            _bounded_text("scope_key", self.scope_key, 200),
        )
        object.__setattr__(
            self,
            "semantic_key",
            _bounded_text("semantic_key", self.semantic_key, 512),
        )
        object.__setattr__(
            self,
            "summary",
            _bounded_text(
                "summary",
                _redact_sensitive_text(self.summary),
                2_000,
            ),
        )
        object.__setattr__(
            self,
            "app_bundle_id",
            normalize_app_bundle_id(self.app_bundle_id),
        )
        object.__setattr__(
            self,
            "app_name",
            _optional_text("app_name", self.app_name, 200),
        )
        sanitized_payload = _sanitize_payload(self.payload)
        object.__setattr__(self, "payload", sanitized_payload)
        _aware_datetime("observed_at", self.observed_at)
        _validate_payload(sanitized_payload)


@dataclass(frozen=True, slots=True)
class ImportantObservationEvent:
    """A durable, already-filtered event worth retaining for later recall."""

    source: str
    observation_type: str
    semantic_key: str
    summary: str
    occurred_at: datetime
    payload: Mapping[str, object] = field(default_factory=dict)
    app_bundle_id: str | None = None
    app_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _slug("source", self.source))
        object.__setattr__(
            self,
            "observation_type",
            _slug("observation_type", self.observation_type),
        )
        object.__setattr__(
            self,
            "semantic_key",
            _bounded_text("semantic_key", self.semantic_key, 512),
        )
        object.__setattr__(
            self,
            "summary",
            _bounded_text(
                "summary",
                _redact_sensitive_text(self.summary),
                2_000,
            ),
        )
        object.__setattr__(
            self,
            "app_bundle_id",
            normalize_app_bundle_id(self.app_bundle_id),
        )
        object.__setattr__(
            self,
            "app_name",
            _optional_text("app_name", self.app_name, 200),
        )
        sanitized_payload = _sanitize_payload(self.payload)
        object.__setattr__(self, "payload", sanitized_payload)
        _aware_datetime("occurred_at", self.occurred_at)
        _validate_payload(sanitized_payload)


@dataclass(frozen=True, slots=True)
class Observation:
    id: UUID
    sequence: int
    fish_id: UUID
    kind: ObservationKind
    source: str
    observation_type: str
    scope_key: str | None
    semantic_key: str
    summary: str
    payload: Mapping[str, object]
    app_bundle_id: str | None
    app_name: str | None
    started_at: datetime
    ended_at: datetime
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ObservationQuery:
    start_at: datetime | None = None
    end_at: datetime | None = None
    app_bundle_id: str | None = None
    observation_type: str | None = None
    kind: ObservationKind | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.start_at is not None:
            _aware_datetime("start_at", self.start_at)
        if self.end_at is not None:
            _aware_datetime("end_at", self.end_at)
        if (
            self.start_at is not None
            and self.end_at is not None
            and self.end_at < self.start_at
        ):
            raise ValueError("end_at must not be before start_at")
        object.__setattr__(
            self,
            "app_bundle_id",
            normalize_app_bundle_id(self.app_bundle_id),
        )
        if self.observation_type is not None:
            object.__setattr__(
                self,
                "observation_type",
                _slug("observation_type", self.observation_type),
            )
        if not 1 <= self.limit <= 500:
            raise ValueError("limit must be between 1 and 500")


@dataclass(frozen=True, slots=True)
class AppRecordingPolicy:
    fish_id: UUID
    app_bundle_id: str
    app_name: str | None
    recording_enabled: bool
    created_at: datetime
    updated_at: datetime


def normalize_app_bundle_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not _APP_BUNDLE_ID_RE.fullmatch(normalized):
        raise ValueError("app_bundle_id must be a valid bundle identifier")
    return normalized.casefold()


def _slug(name: str, value: str) -> str:
    normalized = value.strip()
    if not _SLUG_RE.fullmatch(normalized):
        raise ValueError(f"{name} must match ^[a-z][a-z0-9._-]{{0,63}}$")
    return normalized


def _bounded_text(name: str, value: str, maximum: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return normalized


def _optional_text(name: str, value: str | None, maximum: int) -> str | None:
    if value is None:
        return None
    return _bounded_text(name, value, maximum)


def _redact_sensitive_text(value: str) -> str:
    redacted = _PRIVATE_KEY_RE.sub("[REDACTED]", value)
    redacted = _AUTHORIZATION_BEARER_RE.sub("Authorization=[REDACTED]", redacted)
    redacted = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", redacted)
    redacted = _ENV_SECRET_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _LABELED_SECRET_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _CHINESE_LABELED_SECRET_RE.sub(r"\1=[REDACTED]", redacted)
    redacted = _JWT_RE.sub("[REDACTED]", redacted)
    redacted = _PROVIDER_TOKEN_RE.sub("[REDACTED]", redacted)
    redacted = _URL_CREDENTIALS_RE.sub(r"\1[REDACTED]@", redacted)
    redacted = _SECRET_QUERY_RE.sub(r"\1[REDACTED]", redacted)
    return _LONG_OPAQUE_VALUE_RE.sub("[REDACTED]", redacted)


def redact_sensitive_observation_text(value: str) -> str:
    """Apply the Observation persistence redactor to one natural-language value."""

    return _redact_sensitive_text(value)


def _aware_datetime(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _validate_payload(value: Mapping[str, object]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("payload must be a JSON object")
    _reject_raw_media(value)


def _sanitize_payload(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("payload must be a JSON object")
    return {key: _sanitize_payload_value(child) for key, child in value.items()}


def _sanitize_payload_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _sanitize_payload_value(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_payload_value(child) for child in value]
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    return value


def _reject_raw_media(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("payload keys must be strings")
            normalized_key = _normalize_payload_key(key)
            if normalized_key in _FORBIDDEN_MEDIA_KEYS:
                raise ValueError("payload must not contain raw screen or camera data")
            _reject_raw_media(child)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_raw_media(child)
        return
    if isinstance(value, str) and value.lstrip().casefold().startswith("data:image/"):
        raise ValueError("payload must not contain raw screen or camera data")


def _normalize_payload_key(value: str) -> str:
    with_boundaries = _CAMEL_BOUNDARY_RE.sub("_", value)
    return _NON_IDENTIFIER_RE.sub("_", with_boundaries.casefold()).strip("_")
