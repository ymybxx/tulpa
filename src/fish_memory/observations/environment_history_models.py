"""Validated, timestamped inputs and durable semantic environment snapshots."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from types import MappingProxyType
from typing import Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from fish_memory.environment_frames import EnvironmentFrameUploadRequest
from fish_memory.observations.environment_history_hierarchy import (
    level_for_period,
    snapshot_levels,
)
from fish_memory.observations.environment_history_narrative import (
    NarrativeEvent,
    NarrativeParagraph,
    render_events,
    render_paragraphs,
)

HISTORY_SOURCE = "environment_history"
WINDOW_TYPE = "semantic_snapshot_10s"
MINUTE_TYPE = "semantic_snapshot_60s"
FIVE_MINUTE_TYPE = "semantic_snapshot_300s"
HALF_HOUR_TYPE = "semantic_snapshot_1800s"
WINDOW_SECONDS = 10
MAX_WINDOW_FRAMES = 10
# Ten bounded JPEGs plus PCM/base64 and metadata; checked while reading HTTP.
MAX_WINDOW_UPLOAD_BYTES = 3_500_000
WINDOWS_PER_MINUTE = 6
OBSERVATION_TYPE_BY_PERIOD: Mapping[int, str] = MappingProxyType(
    {level.period_seconds: level.observation_type for level in snapshot_levels()}
)
SAMPLE_RATE = 16_000
WINDOW_PCM_BYTES = WINDOW_SECONDS * SAMPLE_RATE * 2
_SILENT_WINDOW_PCM = bytes(WINDOW_PCM_BYTES)
# Six transcripts retain sentence time ranges and scoped voice labels as well
# as the words. The document's UTF-8 byte budget remains authoritative.
SEMANTIC_AUDIO_MAX_CHARACTERS = 12_000
# The persisted Observation envelope adds roughly 0.5 KiB for one minute's
# identity, digest, and six source UUIDs. Keeping the document below 31 KiB
# leaves more than 1 KiB inside the store's fixed 32 KiB payload boundary.
SEMANTIC_DOCUMENT_MAX_BYTES = 31_000
_NAMESPACE = UUID("c85f8509-8673-49c4-8f84-209b19fa69c0")


@lru_cache(maxsize=16)
def _silent_pcm(samples: int) -> bytes:
    return bytes(samples * 2)


class EnvironmentHistoryWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    session_id: UUID | None = Field(default=None, alias="sessionId")
    capture_session_id: UUID = Field(alias="captureSessionId")
    window_index: int = Field(
        alias="windowIndex", strict=True, ge=0, le=9_007_199_254_740_985
    )
    started_at: float = Field(alias="startedAt", gt=0)
    ended_at: float = Field(alias="endedAt", gt=0)
    frames: tuple[EnvironmentFrameUploadRequest, ...] = Field(
        max_length=MAX_WINDOW_FRAMES
    )
    audio_status: Literal["complete", "partial", "unavailable"] = Field(
        alias="audioStatus"
    )
    audio_pcm_base64: str | None = Field(
        default=None,
        alias="audioPcmBase64",
        max_length=((WINDOW_PCM_BYTES + 2) // 3) * 4,
        repr=False,
    )
    audio_sample_rate: Literal[16000] = Field(default=16000, alias="audioSampleRate")
    audio_captured_samples: int = Field(
        alias="audioCapturedSamples", strict=True, ge=0, le=WINDOW_SECONDS * SAMPLE_RATE
    )
    audio_gap_reason: str | None = Field(
        default=None, alias="audioGapReason", max_length=100, pattern=r"^[a-z_]+$"
    )
    audio_silence: bool = Field(default=False, alias="audioSilence", strict=True)

    @model_validator(mode="before")
    @classmethod
    def expand_window_frame_references(cls, value):
        """References are local to this one request, never a stale global image.

        Repeated captures keep their own identities/times; JPEG bytes travel
        once. The regular frame validator verifies each supplied content hash.
        """
        if not isinstance(value, Mapping):
            return value
        frames = value.get("frames")
        if not isinstance(frames, (list, tuple)) or len(frames) > MAX_WINDOW_FRAMES:
            return value
        images: dict[str, str] = {}
        expanded = []
        for frame in frames:
            if isinstance(frame, Mapping):
                digest = frame.get("contentHash", frame.get("content_hash"))
                jpeg = frame.get("jpegBase64", frame.get("jpeg_base64"))
                if isinstance(digest, str):
                    if isinstance(jpeg, str):
                        images[digest] = jpeg
                    elif jpeg is None and digest in images:
                        frame = {**frame, "jpegBase64": images[digest]}
            expanded.append(frame)
        return {**value, "frames": expanded}

    @model_validator(mode="after")
    def validate_window(self) -> EnvironmentHistoryWindow:
        if not all(math.isfinite(t) for t in (self.started_at, self.ended_at)):
            raise ValueError("window timestamps must be finite")
        duration = self.ended_at - self.started_at
        if not 0 < duration <= WINDOW_SECONDS + 0.001:
            raise ValueError("a history window must span at most ten seconds")
        if len(self.frames) > (5 if duration <= 5.001 else MAX_WINDOW_FRAMES):
            raise ValueError("too many frames for the capture window")
        samples = round(duration * SAMPLE_RATE)
        if len({frame.frame_id for frame in self.frames}) != len(self.frames):
            raise ValueError("history frames must have distinct capture identities")
        previous = self.started_at
        for frame in self.frames:
            if not previous <= frame.observed_at <= self.ended_at:
                raise ValueError("frames must be ordered and belong to the window")
            frame.decode_jpeg()
            previous = frame.observed_at
        if self.audio_silence:
            if (
                self.audio_status != "complete"
                or self.audio_captured_samples != samples
                or self.audio_pcm_base64 is not None
                or self.audio_gap_reason is not None
            ):
                raise ValueError(
                    "silent audio requires a fully captured, gap-free window"
                )
        elif self.audio_status == "unavailable":
            if self.audio_pcm_base64 is not None or self.audio_captured_samples:
                raise ValueError("unavailable audio cannot claim captured samples")
        else:
            if len(self.decode_pcm()) != samples * 2:
                raise ValueError("audio length must match the window duration")
            if (
                self.audio_status == "complete"
                and self.audio_captured_samples != samples
            ):
                raise ValueError("complete audio cannot contain a capture gap")
        return self

    @property
    def period_seconds(self) -> int:
        # Preserve old five-second identities and short cuts; actual timestamps
        # determine the audio and History range, independently of the leaf type.
        return 10 if self.ended_at - self.started_at > 5.001 else 5

    def decode_pcm(self) -> bytes:
        if self.audio_silence:
            return _silent_pcm(round((self.ended_at - self.started_at) * SAMPLE_RATE))
        if self.audio_pcm_base64 is None:
            return b""
        try:
            pcm = base64.b64decode(self.audio_pcm_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("invalid system audio encoding") from error
        if base64.b64encode(pcm).decode("ascii") != self.audio_pcm_base64:
            raise ValueError("noncanonical system audio encoding")
        return pcm

    def input_digest(self) -> str:
        payload = self.model_dump(mode="json")
        # Session routing is derived by the host, not captured media. A retry
        # after a session switch/restart must retain the original observation.
        payload.pop("session_id", None)
        # Additive transport fields must not change legacy retry identities.
        if not self.audio_silence:
            payload.pop("audio_silence", None)
        for frame in payload["frames"]:
            if frame.get("content_hash") is None:
                frame.pop("content_hash", None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class SemanticDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Model summaries have their own shorter output limits. Lossless merge
    # fallbacks may contain several complete children; the shared UTF-8 byte
    # budget below is the authoritative storage bound for those documents.
    audiovisual_summary: str = Field(
        default="", alias="audiovisualSummary", max_length=SEMANTIC_DOCUMENT_MAX_BYTES
    )
    activity_summary: str = Field(
        default="", alias="activitySummary", max_length=SEMANTIC_DOCUMENT_MAX_BYTES
    )
    # Old mixed summaries contain no reliable category boundary. Keep them
    # separate instead of silently classifying historical text during reads.
    legacy_summary: str = Field(
        default="", alias="summary", max_length=SEMANTIC_DOCUMENT_MAX_BYTES
    )
    audio: str = Field(default="", max_length=SEMANTIC_AUDIO_MAX_CHARACTERS)

    summary_version: int = Field(default=0, alias="summaryVersion", ge=0)
    unchanged_from: UUID | None = Field(default=None, alias="unchangedFrom")
    narrative_events: tuple[NarrativeEvent, ...] = Field(
        default=(), alias="narrativeEvents", max_length=80
    )
    narrative_paragraphs: tuple[NarrativeParagraph, ...] = Field(
        default=(), alias="narrativeParagraphs", max_length=32
    )
    # Store each original passage once, including material the model did not
    # cite. It remains available to the next merge without becoming a visible
    # "supplementary words" dump after an otherwise successful summary.
    narrative_sources: dict[str, str] = Field(
        default_factory=dict, alias="narrativeSources", max_length=240
    )
    visual_summary: str = Field(
        default="", alias="visualSummary", max_length=SEMANTIC_DOCUMENT_MAX_BYTES
    )

    @model_validator(mode="before")
    @classmethod
    def restore_derived_audiovisual_summary(cls, value):
        """Persist event narration once while retaining the original Python API."""
        if not isinstance(value, Mapping):
            return value
        version = value.get("summaryVersion", value.get("summary_version", 0))
        events = value.get("narrativeEvents", value.get("narrative_events", ()))
        paragraphs = value.get(
            "narrativeParagraphs", value.get("narrative_paragraphs", ())
        )
        if (
            type(version) is int
            and version >= 3
            and "audiovisualSummary" not in value
            and "audiovisual_summary" not in value
            and isinstance(paragraphs, (list, tuple))
            and 0 < len(paragraphs) <= 32
        ):
            value = dict(value)
            value["audiovisualSummary"] = render_paragraphs(
                tuple(NarrativeParagraph.model_validate(p) for p in paragraphs)
            )
        if (
            type(version) is int
            and version >= 2
            and "audiovisualSummary" not in value
            and "audiovisual_summary" not in value
            and isinstance(events, (list, tuple))
            and 0 < len(events) <= 80
        ):
            value = dict(value)
            value["audiovisualSummary"] = render_events(
                tuple(NarrativeEvent.model_validate(event) for event in events)
            )
        return value

    @property
    def history_summary(self) -> str:
        """Visible History content, excluding silent/unchanged checkpoints."""
        if self.unchanged_from is not None:
            return ""
        return "\n".join(
            part
            for part in (
                self.audiovisual_summary,
                self.activity_summary,
                self.visual_summary,
                self.legacy_summary,
            )
            if part
        ).strip()

    @property
    def summary(self) -> str:
        """Compatible display text; never persisted back into category bodies."""
        if self.unchanged_from is not None:
            return "画面保持不变，未听到新的声音。"
        if (
            not self.audiovisual_summary
            and not self.activity_summary
            and not self.visual_summary
        ):
            return self.legacy_summary
        sections = []
        for label, value in (
            ("影音内容", self.audiovisual_summary),
            ("看到的画面", self.visual_summary),
            ("电脑操作", self.activity_summary),
            ("未分类内容", self.legacy_summary),
        ):
            if value:
                sections.append(f"{label}：\n{value}")
        return "\n\n".join(sections)

    @model_serializer
    def serialize_document(self) -> dict[str, object]:
        if (
            not self.audiovisual_summary
            and not self.activity_summary
            and not self.visual_summary
        ):
            document: dict[str, object] = {
                "summary": self.legacy_summary,
                "audio": self.audio,
            }
        else:
            document = {
                "audiovisualSummary": self.audiovisual_summary,
                "activitySummary": self.activity_summary,
                "audio": self.audio,
            }
        if self.legacy_summary:
            document["summary"] = self.legacy_summary
        if self.summary_version:
            document["summaryVersion"] = self.summary_version
        if self.unchanged_from is not None:
            document["unchangedFrom"] = str(self.unchanged_from)
        if self.narrative_events:
            document["narrativeEvents"] = [
                event.model_dump(mode="json", by_alias=True, exclude_defaults=True)
                for event in self.narrative_events
            ]
        if self.narrative_paragraphs:
            document["narrativeParagraphs"] = [
                p.model_dump(mode="json", by_alias=True)
                for p in self.narrative_paragraphs
            ]
        if self.narrative_sources:
            document["narrativeSources"] = self.narrative_sources
        if self.narrative_paragraphs and self.audiovisual_summary == render_paragraphs(
            self.narrative_paragraphs
        ):
            document.pop("audiovisualSummary", None)
        if self.visual_summary:
            document["visualSummary"] = self.visual_summary
        if (
            self.summary_version >= 2
            and self.narrative_events
            and self.audiovisual_summary == render_events(self.narrative_events)
        ):
            document.pop("audiovisualSummary", None)
        return document

    @model_validator(mode="after")
    def validate_storage_budget(self) -> SemanticDocument:
        if self.unchanged_from is not None and any(
            (
                self.audiovisual_summary,
                self.activity_summary,
                self.legacy_summary,
                self.visual_summary,
                self.audio,
                self.narrative_events,
                self.narrative_paragraphs,
                self.narrative_sources,
            )
        ):
            raise ValueError(
                "unchanged coverage cannot repeat observations or dialogue"
            )
        if any(
            source not in self.narrative_sources
            for paragraph in self.narrative_paragraphs
            for source in paragraph.source_ids
        ):
            raise ValueError("narrative paragraph references absent evidence")
        if self.summary_version < 2 and not any(
            value.strip()
            for value in (
                self.audiovisual_summary,
                self.activity_summary,
                self.legacy_summary,
                self.visual_summary,
            )
        ):
            raise ValueError("semantic document needs a nonempty summary")
        if len(self.model_dump_json().encode()) > SEMANTIC_DOCUMENT_MAX_BYTES:
            raise ValueError("semantic document exceeds its storage budget")
        return self


@dataclass(frozen=True, slots=True)
class SemanticSnapshot:
    id: UUID
    fish_id: UUID
    capture_session_id: UUID
    window_index: int
    period_seconds: int
    started_at: datetime
    ended_at: datetime
    document: SemanticDocument
    input_digest: str
    created_at: datetime
    source_ids: tuple[UUID, ...] = ()
    audio_unavailable: bool = False
    session_id: UUID | None = None


def observation_type_for_period(period_seconds: int) -> str:
    """Return the durable Observation type for a supported snapshot period."""
    return level_for_period(period_seconds).observation_type


def snapshot_id(
    fish_id: UUID, capture_session_id: UUID, window_index: int, period_seconds: int
) -> UUID:
    return uuid5(
        _NAMESPACE, f"{fish_id}/{capture_session_id}/{window_index}/{period_seconds}"
    )
