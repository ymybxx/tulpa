"""Ephemeral, fish-scoped screen frames for environment-aware answers."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Three seconds bounds only the native-to-local handoff. Once authenticated and
# accepted, the last frame stays in memory until it is replaced or explicitly
# cleared so a static desktop remains available to the next Ask.
ENVIRONMENT_FRAME_TTL_SECONDS = 3.0
MAX_ENVIRONMENT_FRAME_BYTES = 200 * 1_024
MAX_ENVIRONMENT_FRAME_LONG_EDGE = 960
_MAX_CLOCK_SKEW_SECONDS = 5.0
_TURN_LEASE_TTL_SECONDS = 30.0
_MAX_TURN_LEASES_PER_FISH = 8
ENVIRONMENT_FRAME_HISTORY_SECONDS = 120.0
MAX_ENVIRONMENT_FRAME_HISTORY = 120
VOICE_FRAME_SAMPLE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class EnvironmentPerceptionPolicy:
    operation_enabled: bool = False
    screen_enabled: bool = False
    audio_enabled: bool = False
    audio_supported: bool = True
    # The pet's "inner life": when off, it still sees and hears in the moment
    # but keeps no audiovisual summaries of it.
    summaries_enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self.operation_enabled or self.frames_enabled or self._audio_active

    @property
    def frames_enabled(self) -> bool:
        return self.screen_enabled

    @property
    def history_enabled(self) -> bool:
        return self.summaries_enabled and (self.screen_enabled or self._audio_active)

    @property
    def _audio_active(self) -> bool:
        return self.audio_enabled and self.audio_supported


class EnvironmentPerceptionStateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    enabled: bool | None = Field(default=None, strict=True)
    operation_enabled: bool | None = Field(
        default=None, strict=True, validation_alias="operationEnabled"
    )
    screen_enabled: bool | None = Field(
        default=None, strict=True, validation_alias="screenEnabled"
    )
    audio_enabled: bool | None = Field(
        default=None, strict=True, validation_alias="audioEnabled"
    )
    audio_supported: bool = Field(
        default=True, strict=True, validation_alias="audioSupported"
    )
    audiovisual_enabled: bool | None = Field(
        default=None, strict=True, validation_alias="audiovisualEnabled", exclude=True
    )
    summaries_enabled: bool = Field(
        default=True, strict=True, validation_alias="summariesEnabled"
    )

    @model_validator(mode="after")
    def validate_policy(self) -> EnvironmentPerceptionStateRequest:
        fields = self.model_fields_set - {"audio_supported", "summaries_enabled"}
        independent = {"operation_enabled", "screen_enabled", "audio_enabled"}
        old_independent = {"operation_enabled", "screen_enabled", "audiovisual_enabled"}
        if fields == old_independent and all(
            getattr(self, field) is not None for field in old_independent
        ):
            return self
        if fields == {"enabled"} and self.enabled is not None:
            return self
        if fields == independent and all(
            getattr(self, field) is not None for field in independent
        ):
            return self
        raise ValueError("provide all three perception flags or legacy enabled")

    def policy(self) -> EnvironmentPerceptionPolicy:
        if self.enabled is not None:
            return EnvironmentPerceptionPolicy(
                self.enabled,
                self.enabled,
                False,
                self.audio_supported,
                self.summaries_enabled,
            )
        if self.audiovisual_enabled is not None:
            return EnvironmentPerceptionPolicy(
                bool(self.operation_enabled),
                bool(self.screen_enabled or self.audiovisual_enabled),
                bool(self.audiovisual_enabled and self.audio_supported),
                self.audio_supported,
                self.summaries_enabled,
            )
        return EnvironmentPerceptionPolicy(
            bool(self.operation_enabled),
            bool(self.screen_enabled),
            bool(self.audio_enabled and self.audio_supported),
            self.audio_supported,
            self.summaries_enabled,
        )


class EnvironmentFrameMetadataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    frame_id: UUID = Field(validation_alias="frameId")
    revision: Annotated[int, Field(strict=True, ge=1)]
    observed_at: Annotated[float, Field(validation_alias="observedAt", gt=0)]
    expires_at: Annotated[float, Field(validation_alias="expiresAt", gt=0)]
    display_id: Annotated[int, Field(validation_alias="displayID", strict=True, ge=0)]
    pixel_width: Annotated[
        int,
        Field(validation_alias="pixelWidth", strict=True, ge=1),
    ]
    pixel_height: Annotated[
        int,
        Field(validation_alias="pixelHeight", strict=True, ge=1),
    ]
    byte_length: Annotated[
        int,
        Field(
            validation_alias="byteLength",
            strict=True,
            ge=1,
            le=MAX_ENVIRONMENT_FRAME_BYTES,
        ),
    ]
    content_hash: str | None = Field(
        default=None, validation_alias="contentHash", pattern=r"^[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def validate_metadata(self) -> EnvironmentFrameMetadataRequest:
        if not all(
            math.isfinite(value) for value in (self.observed_at, self.expires_at)
        ):
            raise ValueError("frame timestamps must be finite")
        if (
            abs(self.expires_at - self.observed_at - ENVIRONMENT_FRAME_TTL_SECONDS)
            > 0.001
        ):
            raise ValueError("frame TTL must be exactly three seconds")
        if max(self.pixel_width, self.pixel_height) > MAX_ENVIRONMENT_FRAME_LONG_EDGE:
            raise ValueError("frame dimensions exceed the long-edge limit")
        return self


class EnvironmentFrameReferenceRequest(EnvironmentFrameMetadataRequest):
    content_hash: str = Field(validation_alias="contentHash", pattern=r"^[0-9a-f]{64}$")
    mime_type: str = Field(validation_alias="mimeType", pattern=r"^image/jpeg$")


class EnvironmentFrameUploadRequest(EnvironmentFrameMetadataRequest):
    mime_type: str = Field(validation_alias="mimeType", pattern=r"^image/jpeg$")
    jpeg_base64: str = Field(
        validation_alias="jpegBase64",
        min_length=1,
        max_length=((MAX_ENVIRONMENT_FRAME_BYTES + 2) // 3) * 4,
        repr=False,
    )

    def decode_jpeg(self) -> bytes:
        try:
            payload = base64.b64decode(self.jpeg_base64, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("invalid JPEG payload") from error
        if (
            len(payload) != self.byte_length
            or len(payload) > MAX_ENVIRONMENT_FRAME_BYTES
            or len(payload) < 5
            or payload[:3] != b"\xff\xd8\xff"
            or payload[-2:] != b"\xff\xd9"
            or base64.b64encode(payload).decode("ascii") != self.jpeg_base64
            or (
                self.content_hash is not None
                and hashlib.sha256(payload).hexdigest() != self.content_hash
            )
        ):
            raise ValueError("invalid JPEG payload")
        return payload


@dataclass(frozen=True, slots=True)
class EnvironmentFrameSnapshot:
    fish_id: UUID
    frame_id: UUID
    revision: int
    observed_at: float
    expires_at: float
    display_id: int
    pixel_width: int
    pixel_height: int
    mime_type: str
    jpeg: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class EnvironmentFrameLease:
    frame_id: UUID
    observed_at: float
    expires_at: float
    display_id: int
    pixel_width: int
    pixel_height: int
    mime_type: str
    jpeg: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _StoredLease:
    value: EnvironmentFrameLease
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class _StoredFrameSet:
    frames: tuple[EnvironmentFrameLease, ...]
    started_at: float
    through_at: float
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class EnvironmentPerceptionState:
    enabled: bool
    frame_available: bool
    observed_at: float | None = None
    expires_at: float | None = None
    operation_enabled: bool = False
    screen_enabled: bool = False
    audio_enabled: bool = False
    audio_supported: bool = True


class EnvironmentFrameCache:
    """Thread-safe in-memory slots. Bytes are never persisted or logged."""

    def __init__(self, *, now: Callable[[], float] = time.time) -> None:
        self._now = now
        self._lock = threading.RLock()
        self._enabled: set[UUID] = set()
        self._policies: dict[UUID, EnvironmentPerceptionPolicy] = {}
        self._policy_revision = 0
        self._frames: dict[UUID, EnvironmentFrameSnapshot] = {}
        self._leases: dict[tuple[UUID, UUID, str], _StoredLease] = {}
        self._history: dict[UUID, deque[EnvironmentFrameSnapshot]] = {}
        self._voice_leases: dict[tuple[UUID, UUID, UUID, int], _StoredFrameSet] = {}

    def set_enabled(self, fish_id: UUID, enabled: bool) -> None:
        """Compatibility with the original operation-and-screen switch."""
        self.set_policy(fish_id, EnvironmentPerceptionPolicy(enabled, enabled, False))

    def set_policy(self, fish_id: UUID, policy: EnvironmentPerceptionPolicy) -> None:
        with self._lock:
            if self._policies.get(fish_id, EnvironmentPerceptionPolicy()) != policy:
                self._policy_revision += 1
            if self._policies.get(fish_id) != policy:
                self._clear_leases_locked(fish_id)
                if not policy.frames_enabled:
                    self._history.pop(fish_id, None)
            if policy.enabled:
                # There is one active desktop assistant capture source. A Fish
                # switch cannot retain the previous assistant's screen history.
                for previous in self._policies.keys() - {fish_id}:
                    self._frames.pop(previous, None)
                    self._history.pop(previous, None)
                    self._clear_leases_locked(previous)
                self._policies.clear()
                self._policies[fish_id] = policy
                self._enabled.clear()
            elif not policy.audio_supported:
                self._policies[fish_id] = policy
            else:
                self._policies.pop(fish_id, None)
            if policy.frames_enabled:
                self._enabled.clear()
                self._enabled.add(fish_id)
            else:
                self._enabled.discard(fish_id)
                self._frames.pop(fish_id, None)
                self._history.pop(fish_id, None)
                self._clear_leases_locked(fish_id)

    @property
    def policy_revision(self) -> int:
        with self._lock:
            return self._policy_revision

    def policy(self, fish_id: UUID) -> EnvironmentPerceptionPolicy:
        with self._lock:
            return self._policies.get(fish_id, EnvironmentPerceptionPolicy())

    def is_enabled(self, fish_id: UUID) -> bool:
        return self.policy(fish_id).enabled

    def put(
        self,
        fish_id: UUID,
        request: EnvironmentFrameUploadRequest | EnvironmentFrameReferenceRequest,
        jpeg: bytes,
    ) -> bool:
        now = self._now()
        if (
            not math.isfinite(now)
            or request.expires_at <= now
            or request.observed_at > now + _MAX_CLOCK_SKEW_SECONDS
        ):
            return False
        snapshot = EnvironmentFrameSnapshot(
            fish_id=fish_id,
            frame_id=request.frame_id,
            revision=request.revision,
            observed_at=request.observed_at,
            expires_at=request.expires_at,
            display_id=request.display_id,
            pixel_width=request.pixel_width,
            pixel_height=request.pixel_height,
            mime_type=request.mime_type,
            jpeg=bytes(jpeg),
            sha256=hashlib.sha256(jpeg).hexdigest(),
        )
        with self._lock:
            if fish_id not in self._enabled:
                return False
            previous = self._frames.get(fish_id)
            if previous is not None:
                if request.observed_at < previous.observed_at:
                    return False
                if (
                    request.observed_at == previous.observed_at
                    and request.revision <= previous.revision
                ):
                    return False
            self._prune_leases_locked(fish_id, now)
            self._frames[fish_id] = snapshot
            history = self._history.setdefault(fish_id, deque())
            history.append(snapshot)
            self._prune_history_locked(fish_id, now)
            return True

    def refresh(self, fish_id: UUID, request: EnvironmentFrameReferenceRequest) -> bool:
        """A newly observed identical frame reuses only this Fish's cached bytes."""
        with self._lock:
            previous = self._frames.get(fish_id)
            if (
                previous is None
                or previous.sha256 != request.content_hash
                or previous.display_id != request.display_id
                or previous.pixel_width != request.pixel_width
                or previous.pixel_height != request.pixel_height
                or len(previous.jpeg) != request.byte_length
            ):
                return False
            return self.put(fish_id, request, previous.jpeg)

    def clear(self, fish_id: UUID, *, include_leases: bool = True) -> None:
        with self._lock:
            self._frames.pop(fish_id, None)
            self._history.pop(fish_id, None)
            if include_leases:
                self._clear_leases_locked(fish_id)

    def snapshot(self, fish_id: UUID) -> EnvironmentFrameSnapshot | None:
        with self._lock:
            if fish_id not in self._enabled:
                return None
            return self._frames.get(fish_id)

    def recent_frames(
        self, fish_id: UUID, *, through_at: float
    ) -> tuple[EnvironmentFrameSnapshot, ...]:
        """Freeze at most one actual capture per second from the last ten seconds."""
        if not math.isfinite(through_at):
            return ()
        with self._lock:
            if fish_id not in self._enabled:
                return ()
            available = self._history.get(fish_id, ())
            if not available:
                latest = self._frames.get(fish_id)
                available = (latest,) if latest is not None else ()
            slots: dict[int, EnvironmentFrameSnapshot] = {}
            for frame in available:
                age = through_at - frame.observed_at
                if 0 <= age < 10:
                    slots[math.floor(age)] = frame
            return tuple(sorted(slots.values(), key=lambda frame: frame.observed_at))

    def lease_frame(
        self,
        fish_id: UUID,
        session_id: UUID,
        turn_key: str,
        *,
        now: float,
        capability: bool,
    ) -> EnvironmentFrameLease | None:
        """Freeze one fresh frame for draft/final reuse within one Ask turn."""
        if (
            not capability
            or not turn_key
            or len(turn_key) > 200
            or not math.isfinite(now)
        ):
            return None
        key = (fish_id, session_id, turn_key)
        with self._lock:
            if fish_id not in self._enabled:
                return None
            self._prune_leases_locked(fish_id, now)
            existing = self._leases.get(key)
            if existing is not None:
                return existing.value
            snapshot = self._frames.get(fish_id)
            if snapshot is None:
                return None
            value = EnvironmentFrameLease(
                frame_id=snapshot.frame_id,
                observed_at=snapshot.observed_at,
                expires_at=snapshot.expires_at,
                display_id=snapshot.display_id,
                pixel_width=snapshot.pixel_width,
                pixel_height=snapshot.pixel_height,
                mime_type=snapshot.mime_type,
                jpeg=snapshot.jpeg,
                sha256=snapshot.sha256,
            )
            self._leases[key] = _StoredLease(
                value=value,
                created_at=now,
                expires_at=now + _TURN_LEASE_TTL_SECONDS,
            )
            self._prune_lease_count_locked(fish_id)
            return value

    def lease_voice_frames(
        self,
        fish_id: UUID,
        session_id: UUID,
        turn_id: UUID,
        revision: int,
        *,
        speech_started_at: float,
        frame_snapshot_at: float,
        now: float,
        capability: bool,
    ) -> tuple[EnvironmentFrameLease, ...]:
        """Freeze ordered speech samples plus the current frame for one draft."""
        if (
            not capability
            or revision < 1
            or not all(
                math.isfinite(value)
                for value in (speech_started_at, frame_snapshot_at, now)
            )
            or speech_started_at > frame_snapshot_at
            or frame_snapshot_at > now + _MAX_CLOCK_SKEW_SECONDS
        ):
            return ()
        key = (fish_id, session_id, turn_id, revision)
        with self._lock:
            if fish_id not in self._enabled:
                return ()
            self._prune_leases_locked(fish_id, now)
            existing = self._voice_leases.get(key)
            if existing is not None:
                if (existing.started_at, existing.through_at) != (
                    speech_started_at,
                    frame_snapshot_at,
                ):
                    raise ValueError(
                        "voice frame revision has a different capture range"
                    )
                return existing.frames
            self._prune_history_locked(fish_id, now)
            available = [
                frame
                for frame in self._history.get(fish_id, ())
                if frame.observed_at <= frame_snapshot_at
            ]
            latest = self._frames.get(fish_id)
            if (
                latest is not None
                and latest.observed_at <= frame_snapshot_at
                and not available
            ):
                # Static screens retain their last frame beyond the ring TTL.
                available.append(latest)
            selected = _sample_speech_frames(
                available, speech_started_at, frame_snapshot_at
            )
            frames = tuple(_frame_lease(frame) for frame in selected)
            self._voice_leases[key] = _StoredFrameSet(
                frames=frames,
                started_at=speech_started_at,
                through_at=frame_snapshot_at,
                created_at=now,
                expires_at=now + _TURN_LEASE_TTL_SECONDS,
            )
            self._prune_lease_count_locked(fish_id)
            return frames

    def state(self, fish_id: UUID) -> EnvironmentPerceptionState:
        with self._lock:
            policy = self.policy(fish_id)
            snapshot = self.snapshot(fish_id) if policy.frames_enabled else None
        return EnvironmentPerceptionState(
            enabled=policy.enabled,
            frame_available=snapshot is not None,
            observed_at=snapshot.observed_at if snapshot is not None else None,
            expires_at=snapshot.expires_at if snapshot is not None else None,
            operation_enabled=policy.operation_enabled,
            screen_enabled=policy.screen_enabled,
            audio_enabled=policy.audio_enabled,
            audio_supported=policy.audio_supported,
        )

    def _prune_leases_locked(self, fish_id: UUID, now: float) -> None:
        expired = [
            key
            for key, lease in self._leases.items()
            if key[0] == fish_id and lease.expires_at <= now
        ]
        for key in expired:
            self._leases.pop(key, None)
        for key, lease in tuple(self._voice_leases.items()):
            if key[0] == fish_id and lease.expires_at <= now:
                self._voice_leases.pop(key, None)

    def _prune_history_locked(self, fish_id: UUID, now: float) -> None:
        history = self._history.get(fish_id)
        if history is None:
            return
        seconds = ENVIRONMENT_FRAME_HISTORY_SECONDS
        limit = MAX_ENVIRONMENT_FRAME_HISTORY
        while history and (
            len(history) > limit or history[0].observed_at < now - seconds
        ):
            history.popleft()

    def _prune_lease_count_locked(self, fish_id: UUID) -> None:
        leases = sorted(
            ((key, value) for key, value in self._leases.items() if key[0] == fish_id),
            key=lambda item: item[1].created_at,
        )
        for key, _value in leases[:-_MAX_TURN_LEASES_PER_FISH]:
            self._leases.pop(key, None)
        voice_leases = sorted(
            (
                (key, value)
                for key, value in self._voice_leases.items()
                if key[0] == fish_id
            ),
            key=lambda item: item[1].created_at,
        )
        for key, _value in voice_leases[:-_MAX_TURN_LEASES_PER_FISH]:
            self._voice_leases.pop(key, None)

    def _clear_leases_locked(self, fish_id: UUID) -> None:
        for key in tuple(self._leases):
            if key[0] == fish_id:
                self._leases.pop(key, None)
        for key in tuple(self._voice_leases):
            if key[0] == fish_id:
                self._voice_leases.pop(key, None)


def _sample_speech_frames(
    available: list[EnvironmentFrameSnapshot], started_at: float, through_at: float
) -> list[EnvironmentFrameSnapshot]:
    """Take latest known frames at speech time 0, 1, 2... without duplicates.

    Speech-anchored slots tolerate native timer jitter. A sub-second phrase has
    one initial frame, and no additional Ask-time capture is required.
    """
    if not available:
        return []
    first_slot = max(
        0,
        math.ceil((available[0].observed_at - started_at) / VOICE_FRAME_SAMPLE_SECONDS),
    )
    last_slot = min(
        math.floor((through_at - started_at) / VOICE_FRAME_SAMPLE_SECONDS),
        max(
            0,
            math.ceil(
                (available[-1].observed_at - started_at) / VOICE_FRAME_SAMPLE_SECONDS
            ),
        ),
    )
    selected: list[EnvironmentFrameSnapshot] = []
    cursor = -1
    for slot in range(first_slot, last_slot + 1):
        target = started_at + slot * VOICE_FRAME_SAMPLE_SECONDS
        while (
            cursor + 1 < len(available) and available[cursor + 1].observed_at <= target
        ):
            cursor += 1
        if cursor >= 0 and (
            not selected or selected[-1].frame_id != available[cursor].frame_id
        ):
            selected.append(available[cursor])
    # During startup the first actual frame may arrive after speech began.
    return selected or [available[0]]


def _frame_lease(snapshot: EnvironmentFrameSnapshot) -> EnvironmentFrameLease:
    return EnvironmentFrameLease(
        frame_id=snapshot.frame_id,
        observed_at=snapshot.observed_at,
        expires_at=snapshot.expires_at,
        display_id=snapshot.display_id,
        pixel_width=snapshot.pixel_width,
        pixel_height=snapshot.pixel_height,
        mime_type=snapshot.mime_type,
        jpeg=snapshot.jpeg,
        sha256=snapshot.sha256,
    )
