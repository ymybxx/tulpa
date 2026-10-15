from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID

from fish_memory.answer.context import ChatImageAttachment
from fish_memory.clock import format_prompt_timestamp
from fish_memory.context_descriptions import (
    ABILITIES_HEADING,
    ABILITY_AGENT,
    ABILITY_AGENT_LOCAL,
    ABILITY_ATTACHMENTS,
    ABILITY_HOME,
    ABILITY_MICROPHONE,
    ABILITY_RECALL,
    ABILITY_SEARCH,
    PERCEPTION_AUDIO_UNSUPPORTED,
    PERCEPTION_STATUS_LINES,
)
from fish_memory.environment_frames import EnvironmentPerceptionPolicy
from fish_memory.observations.computer_history import ComputerHistoryContextSummary
from fish_memory.observations.media_questions import requests_media_recap

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Abilities:
    """The switches besides perception that decide what TA can do right now."""

    microphone: bool = False
    search: bool = False
    agent: bool = False
    # The Agent runs on this computer, so it can look at and work it.
    agent_local: bool = False
    home: bool = False
    images: bool = True


def render_perception_policy(
    policy: EnvironmentPerceptionPolicy, abilities: Abilities | None = None
) -> str:
    lines = [ABILITIES_HEADING]
    for enabled, (on, off) in zip(
        (policy.operation_enabled, policy.screen_enabled, policy.audio_enabled),
        PERCEPTION_STATUS_LINES,
        strict=True,
    ):
        lines.append(on if enabled else off)
    if not policy.audio_supported:
        lines[-1] = PERCEPTION_AUDIO_UNSUPPORTED
    if abilities is not None:
        lines.append(ABILITY_MICROPHONE[0 if abilities.microphone else 1])
        lines.append(ABILITY_RECALL)
        if abilities.search:
            lines.append(ABILITY_SEARCH)
        if abilities.agent:
            lines.append(
                ABILITY_AGENT_LOCAL if abilities.agent_local else ABILITY_AGENT
            )
        if abilities.home:
            lines.append(ABILITY_HOME)
        lines.append(ABILITY_ATTACHMENTS[0 if abilities.images else 1])
    return "\n".join(lines)


def current_perception_context(loader: object, fish_id: UUID) -> str:
    provider = getattr(loader, "perception_context", None)
    return provider(fish_id) if callable(provider) else ""


ImageInputCapability = Literal["supported", "unsupported", "unknown"]


class ComputerHistoryContextSummaryLoader(Protocol):
    async def get_latest_context_summary(
        self,
        fish_id: UUID,
    ) -> ComputerHistoryContextSummary | None: ...


class EnvironmentFrameLease(Protocol):
    @property
    def observed_at(self) -> float: ...

    @property
    def mime_type(self) -> str: ...

    @property
    def jpeg(self) -> bytes: ...

    @property
    def sha256(self) -> str: ...

    @property
    def pixel_width(self) -> int: ...

    @property
    def pixel_height(self) -> int: ...


class EnvironmentFrameCache(Protocol):
    def is_enabled(self, fish_id: UUID) -> bool: ...

    def policy(self, fish_id: UUID) -> EnvironmentPerceptionPolicy: ...

    def lease_frame(
        self,
        fish_id: UUID,
        session_id: UUID,
        turn_key: str,
        *,
        now: float,
        capability: bool,
    ) -> EnvironmentFrameLease | None: ...

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
    ) -> tuple[EnvironmentFrameLease, ...]: ...


class ImageInputCapabilityProvider(Protocol):
    @property
    def image_input_capability(self) -> ImageInputCapability: ...


@dataclass(frozen=True, slots=True)
class EnvironmentAnswerContext:
    enabled: bool
    summary_context: str
    image: ChatImageAttachment | None
    summary_source: Literal["checkpoint", "observation"] | None = None
    images: tuple[ChatImageAttachment, ...] = ()


class EnvironmentAnswerContextLoader:
    """Load the one bounded environment context used by an Ask turn.

    The frame cache owns freshness and immutable per-turn leases. This adapter only
    serializes a validated lease for the provider and never persists the image.
    """

    def __init__(
        self,
        summary_loader: ComputerHistoryContextSummaryLoader,
        frame_cache: EnvironmentFrameCache,
        image_capability: ImageInputCapabilityProvider,
        *,
        now: Callable[[], datetime],
        abilities: Callable[[], Abilities] | None = None,
    ) -> None:
        self._summary_loader = summary_loader
        self._frame_cache = frame_cache
        self._image_capability = image_capability
        self._now = now
        self._abilities = abilities

    def perception_context(self, fish_id: UUID) -> str:
        # Read live flags while building this request, independently of cached
        # History or whether this model receives image attachments.
        abilities = None
        if self._abilities is not None:
            abilities = replace(
                self._abilities(),
                images=self._image_capability.image_input_capability != "unsupported",
            )
        return render_perception_policy(self._frame_cache.policy(fish_id), abilities)

    async def load_cached(
        self,
        fish_id: UUID,
        *,
        include_image: bool = False,
        include_summary: bool = True,
    ) -> EnvironmentAnswerContext:
        """Consume ready history without leasing frames or requesting capture."""
        policy = self._frame_cache.policy(fish_id)
        if not policy.enabled:
            return EnvironmentAnswerContext(policy.enabled, "", None)
        if (
            include_image
            and policy.frames_enabled
            and self._image_capability.image_input_capability == "unknown"
        ):
            _request_image_capability_probe(self._image_capability)
        summary = (
            await self._summary_loader.get_latest_context_summary(fish_id)
            if include_summary and (policy.operation_enabled or policy.history_enabled)
            else None
        )
        if self._frame_cache.policy(fish_id) != policy:
            return EnvironmentAnswerContext(False, "", None)
        image = None
        snapshot = getattr(self._frame_cache, "snapshot", None)
        if (
            include_image
            and policy.frames_enabled
            and callable(snapshot)
            and self._image_capability.image_input_capability == "supported"
        ):
            frame = snapshot(fish_id)
            if frame is not None and frame.expires_at > self._now().timestamp():
                image = _attachment_from_lease(frame)
        return EnvironmentAnswerContext(
            policy.enabled,
            render_environment_summary_context(summary) if summary else "",
            image,
            summary.source if summary else None,
        )

    async def load(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        turn_id: UUID,
        environment_revision: int | None = None,
        speech_started_at: datetime | None = None,
        frame_snapshot_at: datetime | None = None,
        question: str = "",
        include_summary: bool = True,
    ) -> EnvironmentAnswerContext:
        policy = self._frame_cache.policy(fish_id)
        if not policy.enabled:
            return EnvironmentAnswerContext(
                enabled=False,
                summary_context="",
                image=None,
            )

        # Freeze the image input before any history/database await. Each voice
        # revision retains this ordered set even as the current screen changes.
        images: tuple[ChatImageAttachment, ...] = ()
        media_recap = policy.history_enabled and requests_media_recap(question)
        capability = self._image_capability.image_input_capability
        if policy.frames_enabled and capability == "unknown":
            _request_image_capability_probe(self._image_capability)
        try:
            if (
                policy.screen_enabled
                and environment_revision is not None
                and speech_started_at is not None
                and frame_snapshot_at is not None
            ):
                leases = self._frame_cache.lease_voice_frames(
                    fish_id,
                    session_id,
                    turn_id,
                    environment_revision,
                    speech_started_at=speech_started_at.timestamp(),
                    frame_snapshot_at=frame_snapshot_at.timestamp(),
                    now=self._aware_now().timestamp(),
                    capability=capability == "supported",
                )
                images = tuple(_attachment_from_lease(lease) for lease in leases)
            elif policy.frames_enabled:
                lease = self._frame_cache.lease_frame(
                    fish_id,
                    session_id,
                    environment_turn_key(turn_id),
                    now=self._aware_now().timestamp(),
                    capability=capability == "supported",
                )
                if lease is not None:
                    images = (_attachment_from_lease(lease),)
        except Exception:
            logger.exception("Failed to lease environment frames for fish %s", fish_id)

        summary_context = ""
        summary_source: Literal["checkpoint", "observation"] | None = None
        try:
            contextual = getattr(self._summary_loader, "get_context_for_question", None)
            if not include_summary or not (
                policy.operation_enabled or policy.history_enabled
            ):
                summary = None
            elif media_recap and callable(contextual):
                summary = await contextual(fish_id, question)
            else:
                summary = await self._summary_loader.get_latest_context_summary(fish_id)
        except Exception:
            logger.exception(
                "Failed to load Computer History context for fish %s",
                fish_id,
            )
        else:
            if summary is not None:
                summary_context = render_environment_summary_context(summary)
                summary_source = summary.source
                if media_recap and summary.summary.strip():
                    # Use the recorded programme for a recap, not a foreground
                    # debug page opened afterwards. If no history is available,
                    # retain the already-frozen frame instead of recapturing.
                    images = ()

        # A switch changed while SQL was in flight. Only revoke disabled input;
        # never recapture, reload context, or add newly enabled evidence here.
        current_policy = self._frame_cache.policy(fish_id)
        if not current_policy.enabled:
            return EnvironmentAnswerContext(False, "", None)
        if not current_policy.frames_enabled:
            images = ()
        if not (current_policy.operation_enabled or current_policy.history_enabled):
            summary_context = ""
            summary_source = None
        if media_recap and not current_policy.history_enabled:
            summary_context = ""
            summary_source = None

        return EnvironmentAnswerContext(
            enabled=current_policy.enabled,
            summary_context=summary_context,
            image=images[-1] if images else None,
            summary_source=summary_source,
            images=images if len(images) > 1 else (),
        )

    def _aware_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("environment context clock must return an aware datetime")
        return value


def environment_turn_key(turn_id: UUID) -> str:
    return str(turn_id)


def render_environment_summary_context(
    summary: ComputerHistoryContextSummary,
) -> str:
    # Identity, counters and processing timestamps stay in the typed context
    # for diagnostics. The answering model only needs the observed content.
    return "# 你周围的情况\n\n" + summary.summary.strip()


def _attachment_from_lease(lease: EnvironmentFrameLease) -> ChatImageAttachment:
    if lease.mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("environment frame has an unsupported MIME type")
    if not lease.jpeg:
        raise ValueError("environment frame is empty")
    if lease.pixel_width < 1 or lease.pixel_height < 1:
        raise ValueError("environment frame dimensions must be positive")
    digest = lease.sha256.casefold()
    if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
        raise ValueError("environment frame digest must be lowercase SHA-256")
    observed_at = datetime.fromtimestamp(lease.observed_at, UTC)
    encoded = base64.b64encode(lease.jpeg).decode("ascii")
    return {
        "data_url": f"data:{lease.mime_type};base64,{encoded}",
        "digest": digest,
        "byte_count": len(lease.jpeg),
        "width": lease.pixel_width,
        "height": lease.pixel_height,
        "observed_at": format_prompt_timestamp(observed_at),
    }


def _request_image_capability_probe(provider: object) -> None:
    probe = getattr(provider, "probe_image_input_capability", None)
    if not callable(probe):
        return
    try:
        result = probe()
    except Exception:
        return
    if not hasattr(result, "__await__"):
        return
    task = asyncio.create_task(result)

    def consume(completed: asyncio.Task[object]) -> None:
        if completed.cancelled():
            return
        try:
            completed.exception()
        except Exception:
            pass

    task.add_done_callback(consume)


__all__ = [
    "EnvironmentAnswerContext",
    "EnvironmentAnswerContextLoader",
    "EnvironmentFrameCache",
    "ImageInputCapability",
    "ImageInputCapabilityProvider",
    "environment_turn_key",
    "render_environment_summary_context",
]
