"""Wire helpers for the local desktop-pet WebSocket protocol."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import cast
from uuid import UUID

from fish_memory.pet.types import PET_EMOTIONS

PET_PROTOCOL_VERSION = 1
PCM_HEADER = struct.Struct("<B16sQ")
CONTROL_EVENT_TYPES = frozenset(
    {
        "pet.playback.progress",
        "pet.chat.result",
        "pet.companion.activity",
        "pet.microphone.state",
        "pet.speaker.state",
        "pet.proactive.started",
        "pet.voice.hearing",
        "pet.owner_voice.unrecognized",
        "pet.state",
        "pet.expression",
        "pet.facial",
        "pet.audio.start",
        "pet.audio.pause",
        "pet.audio.resume",
        "pet.audio.stop",
        "pet.generation.cancel",
    }
)


@dataclass(frozen=True, slots=True)
class PublishedControl:
    event_type: str
    generation: UUID | None
    payload: dict[str, object]


def parse_published_control(value: object) -> PublishedControl:
    if not isinstance(value, dict):
        raise ValueError("pet control message must be an object")
    event_type = value.get("type")
    if not isinstance(event_type, str) or event_type not in CONTROL_EVENT_TYPES:
        raise ValueError("unsupported pet control event")
    raw_generation = value.get("generation")
    if raw_generation is None:
        generation = None
    elif isinstance(raw_generation, str):
        try:
            generation = UUID(raw_generation)
        except ValueError as error:
            raise ValueError("pet generation must be a UUID") from error
    else:
        raise ValueError("pet generation must be a UUID or null")
    raw_payload = value.get("payload", {})
    if not isinstance(raw_payload, dict) or any(
        not isinstance(key, str) for key in raw_payload
    ):
        raise ValueError("pet control payload must be an object")
    payload = cast(dict[str, object], raw_payload)
    _validate_payload(event_type, payload)
    return PublishedControl(event_type, generation, payload)


def pack_pcm_frame(generation: UUID, sample_position: int, pcm: bytes) -> bytes:
    if sample_position < 0:
        raise ValueError("PCM sample position must not be negative")
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM must contain complete signed 16-bit samples")
    return (
        PCM_HEADER.pack(
            PET_PROTOCOL_VERSION,
            generation.bytes,
            sample_position,
        )
        + pcm
    )


def unpack_pcm_frame(value: bytes) -> tuple[UUID, int, bytes]:
    if len(value) <= PCM_HEADER.size:
        raise ValueError("pet PCM frame is incomplete")
    version, raw_generation, sample_position = PCM_HEADER.unpack_from(value)
    if version != PET_PROTOCOL_VERSION:
        raise ValueError("unsupported pet PCM protocol version")
    pcm = value[PCM_HEADER.size :]
    if len(pcm) % 2:
        raise ValueError("pet PCM frame is not aligned to int16 samples")
    return UUID(bytes=raw_generation), sample_position, pcm


def _validate_payload(event_type: object, payload: dict[str, object]) -> None:
    if event_type == "pet.facial":
        profile_id = payload.get("profileId")
        if not isinstance(profile_id, str) or not 0 < len(profile_id) <= 100:
            raise ValueError("invalid facial profile")
        targets = payload.get("targets")
        if targets is not None and (
            not isinstance(targets, dict)
            or len(targets) > 49
            or any(
                not isinstance(key, str)
                or len(key) > 100
                or isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                for key, value in targets.items()
            )
        ):
            raise ValueError("invalid facial targets")
    if event_type == "pet.chat.result":
        UUID(str(payload.get("requestId")))
        if not isinstance(payload.get("ok"), bool):
            raise ValueError("invalid quick-chat result")
        if payload["ok"]:
            UUID(str(payload.get("messageId")))
        elif not isinstance(payload.get("error"), str):
            raise ValueError("invalid quick-chat error")
    if event_type == "pet.voice.hearing":
        text = payload.get("text")
        if set(payload) != {"text"} or not isinstance(text, str) or len(text) > 2000:
            raise ValueError("invalid hearing text")
    if event_type == "pet.owner_voice.unrecognized":
        # How much just-missed speech the owner could add; the audio itself
        # never leaves the companion process.
        seconds = payload.get("seconds")
        if (
            not set(payload) <= {"seconds", "enrollmentCount", "error"}
            or isinstance(seconds, bool)
            or not isinstance(seconds, (float, int))
            or not 0 <= seconds <= 60
        ):
            raise ValueError("invalid unrecognized owner voice")
        count = payload.get("enrollmentCount")
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= 20
        ):
            raise ValueError("invalid owner voice enrollment count")
        error = payload.get("error")
        if error is not None and (not isinstance(error, str) or len(error) > 200):
            raise ValueError("invalid owner voice error")
    if event_type in {"pet.microphone.state", "pet.speaker.state"}:
        if set(payload) != {"enabled"} or not isinstance(payload["enabled"], bool):
            raise ValueError("invalid microphone state")
    if event_type in {"pet.companion.activity", "pet.proactive.started"}:
        fields = {"fishId", "sessionId"}
        fields |= (
            {"busy"}
            if event_type == "pet.companion.activity"
            else {"messageId", "startedAt"}
        )
        if event_type == "pet.companion.activity" and "playbackBusy" in payload:
            fields.add("playbackBusy")
        if set(payload) != fields:
            raise ValueError("invalid companion activity fields")
        for name in fields & {"fishId", "sessionId", "messageId"}:
            UUID(str(payload[name]))
        if event_type == "pet.companion.activity":
            if not isinstance(payload["busy"], bool):
                raise ValueError("invalid foreground state")
            if "playbackBusy" in payload and not isinstance(
                payload["playbackBusy"], bool
            ):
                raise ValueError("invalid playback state")
        else:
            value = payload["startedAt"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not 0 < value < 253402300800
            ):
                raise ValueError("invalid playback time")
    if event_type == "pet.playback.progress":
        for name in ("fishId", "sessionId", "messageId", "playbackId"):
            try:
                UUID(str(payload.get(name)))
            except ValueError as error:
                raise ValueError(f"invalid playback {name}") from error
        for name in ("revision", "displayOffset", "consumedOffset"):
            value = payload.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid playback {name}")
        if "messageSequence" in payload:
            value = payload["messageSequence"]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("invalid playback messageSequence")
        if payload.get("status") not in {
            "buffering",
            "playing",
            "paused",
            "completed",
            "interrupted",
            "failed",
        }:
            raise ValueError("invalid playback status")
        content = payload.get("visibleContent")
        if not isinstance(content, str) or len(content) != payload["displayOffset"]:
            raise ValueError("playback visible prefix does not match its offset")
        if cast(int, payload["consumedOffset"]) > cast(int, payload["displayOffset"]):
            raise ValueError("consumed playback exceeds display prefix")
    if event_type == "pet.state":
        state = payload.get("state")
        if not isinstance(state, str) or state not in {
            "unbound",
            "standby",
            "acknowledging",
            "listening",
            "thinking",
            "speaking",
        }:
            raise ValueError("unsupported pet state")
    if event_type == "pet.expression":
        emotion = payload.get("emotion")
        if not isinstance(emotion, str) or emotion not in PET_EMOTIONS:
            raise ValueError("unsupported pet emotion")
        if "gesture" in payload:
            raise ValueError("pet gesture field has been removed")
    if event_type == "pet.audio.start":
        sample_rate = payload.get("sampleRate")
        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or not 8_000 <= sample_rate <= 96_000
        ):
            raise ValueError("pet audio sampleRate is invalid")
