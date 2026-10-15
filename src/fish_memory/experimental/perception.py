"""Frozen screen frames and ready local speech shared by thoughts and channels."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from fish_memory.answer.context import ChatImageAttachment, ChatMessage
from fish_memory.clock import format_prompt_timestamp
from fish_memory.context_descriptions import AUDIOVISUAL_AUDIO_CAPTION
from fish_memory.environment_frames import (
    EnvironmentFrameCache,
    EnvironmentFrameUploadRequest,
)
from fish_memory.observations.environment_history_contact_sheet import (
    build_contact_sheet,
)
from fish_memory.observations.environment_history_models import EnvironmentHistoryWindow

logger = logging.getLogger(__name__)


class HeardProvider(Protocol):
    def heard_between(
        self, fish_id: UUID, since: datetime, through: datetime
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class PerceptionInput:
    images: tuple[ChatImageAttachment, ...] = ()
    heard: str = ""

    def attach(self, message: ChatMessage) -> None:
        if self.heard:
            message["content"] += "\n\n" + AUDIOVISUAL_AUDIO_CAPTION.format(
                transcript=self.heard
            )
        if self.images:
            message["environment_images"] = self.images


class ExperimentalPerception:
    def __init__(
        self, frames: EnvironmentFrameCache, audio: HeardProvider | None = None
    ):
        self.frames, self.audio = frames, audio

    async def capture(
        self, fish_id: UUID, at: datetime, *, model: object
    ) -> PerceptionInput:
        policy = self.frames.policy(fish_id)
        since = at - timedelta(seconds=10)
        heard = (
            self.audio.heard_between(fish_id, since, at)
            if policy.audio_enabled
            and policy.audio_supported
            and self.audio is not None
            else ""
        )
        if (
            not policy.frames_enabled
            or getattr(model, "image_input_capability", "unknown") == "unsupported"
        ):
            return PerceptionInput(heard=heard)
        through_at = at.timestamp()
        frames = self.frames.recent_frames(fish_id, through_at=through_at)
        if not frames:
            return PerceptionInput(heard=heard)
        try:
            # Reuse the same validated ten-second layout as audiovisual summaries.
            # This window is only an image-building input; it is never a History event.
            window = EnvironmentHistoryWindow(
                capture_session_id=fish_id,
                window_index=0,
                started_at=since.timestamp(),
                ended_at=through_at,
                frames=tuple(
                    EnvironmentFrameUploadRequest(
                        frameId=frame.frame_id,
                        revision=frame.revision,
                        observedAt=frame.observed_at,
                        expiresAt=frame.expires_at,
                        displayID=frame.display_id,
                        pixelWidth=frame.pixel_width,
                        pixelHeight=frame.pixel_height,
                        byteLength=len(frame.jpeg),
                        contentHash=frame.sha256,
                        mimeType=frame.mime_type,
                        jpegBase64=base64.b64encode(frame.jpeg).decode("ascii"),
                    )
                    for frame in frames
                ),
                audio_status="unavailable",
                audio_captured_samples=0,
            )
            sheet = await asyncio.to_thread(build_contact_sheet, window, moments=True)
            if self.frames.policy(fish_id) != policy:
                return PerceptionInput()
            if getattr(model, "image_input_capability", "unknown") == "unsupported":
                return PerceptionInput(heard=heard)
            jpeg = base64.b64decode(sheet.jpeg_base64)
            images: tuple[ChatImageAttachment, ...] = (
                {
                    "data_url": f"data:image/jpeg;base64,{sheet.jpeg_base64}",
                    "digest": hashlib.sha256(jpeg).hexdigest(),
                    "byte_count": len(jpeg),
                    "width": sheet.width,
                    "height": sheet.height,
                    "observed_at": format_prompt_timestamp(
                        datetime.fromtimestamp(frames[-1].observed_at, UTC)
                    ),
                    # Sensor details (window length, frame count, offsets) are
                    # for the host; the pet only sees what happened, in order.
                    # Look-alike frames are already one picture, each labelled
                    # with how long ago it was, so a sheet reads as time passing.
                    "description": (
                        "你现在看到的屏幕。"
                        if len(sheet.panels) == 1
                        else "你最近几秒看到的屏幕，是同一块屏幕在不同时刻的样子，"
                        "按时间先后排列，每张上方标着是几秒前，最后一张（now）是现在。"
                    ),
                    "detail": "high",
                },
            )
            return PerceptionInput(images=images, heard=heard)
        except Exception:
            logger.exception("Failed to compose experimental perception input")
            if self.frames.policy(fish_id) != policy:
                return PerceptionInput()
            return PerceptionInput(heard=heard)
