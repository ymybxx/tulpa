"""HTTP client for the native Companion process."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, TypeAlias
from uuid import UUID

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from fish_memory.answer.voice_draft_types import VoiceDraftKey

logger = logging.getLogger(__name__)

_DRAFT_CONTROL_TIMEOUT_SECONDS = 0.5
_DRAFT_CAPABILITY_TIMEOUT_SECONDS = 0.5
_VOICE_COMMIT_TIMEOUT_SECONDS = 10.0
_DRAFT_EVENT_TIMEOUT_SECONDS = 3.0
_MAX_AGENT_MESSAGE_SEQUENCE = 9_223_372_036_854_775_807
_MAX_AGENT_MESSAGE_CHARS = 32_000


@dataclass(frozen=True, slots=True)
class CompanionSessionInfo:
    id: UUID
    created_at: datetime
    last_message_at: datetime | None


@dataclass(frozen=True, slots=True)
class CompanionVoiceConfig:
    model: str
    voice: str
    speech_tags: bool = False


@dataclass(frozen=True, slots=True)
class CompanionAskResult:
    message_id: UUID
    answer_expected: bool
    sequence: int = 0


@dataclass(frozen=True, slots=True)
class CompanionReplyContext:
    kind: Literal["interaction", "run_result"]
    id: UUID

    def __post_init__(self) -> None:
        if self.kind not in {"interaction", "run_result"}:
            raise ValueError("unsupported Companion reply context kind")

    def as_payload(self) -> dict[str, str]:
        return {"kind": self.kind, "id": str(self.id)}


@dataclass(frozen=True, slots=True)
class CompanionMessage:
    """One durable background Assistant message delivered to the Companion.

    ``CompanionAgentMessage`` remains an alias for source compatibility with
    the original Agent-only API. Proactive observations deliberately omit the
    optional reply-context fields.
    """

    message_id: UUID
    sequence: int
    session_id: UUID
    content: str
    consumed_offset: int
    subject_kind: Literal["interaction", "run_result"] | None = None
    subject_id: UUID | None = None
    activation_policy: Literal["playback_start", "playback_complete"] | None = None
    source: Literal["agent", "proactive_observation", "plugin"] = "agent"
    emotion: Literal["neutral", "angry", "happy", "shy", "sad"] = "neutral"
    interrupt_policy: Literal["resume", "drop"] = "resume"
    expires_at: datetime | None = None
    supersession_key: str | None = None
    streaming: bool = False
    speech_tags: bool = False
    speech_content: str = ""

    def __post_init__(self) -> None:
        if self.source == "agent":
            if (
                self.subject_kind is None
                or self.subject_id is None
                or self.activation_policy is None
            ):
                raise ValueError("Agent Companion Message requires reply context")
            if self.interrupt_policy != "resume":
                raise ValueError("Agent Companion Message must resume")
            return
        if self.source not in {"proactive_observation", "plugin"}:
            raise ValueError("unsupported Companion Message source")
        if any(
            value is not None
            for value in (
                self.subject_kind,
                self.subject_id,
                self.activation_policy,
            )
        ):
            raise ValueError("Proactive Companion Message cannot have reply context")
        if self.interrupt_policy != "drop":
            raise ValueError("Proactive Companion Message must drop")


# Compatibility for callers written before the delivery stream was generalized.
CompanionAgentMessage: TypeAlias = CompanionMessage


class _AgentMessageSubjectPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["interaction", "run_result"]
    id: UUID


class _AgentMessagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    message_id: UUID = Field(validation_alias="messageId")
    sequence: int = Field(strict=True, ge=1, le=_MAX_AGENT_MESSAGE_SEQUENCE)
    session_id: UUID = Field(validation_alias="sessionId")
    content: str = Field(min_length=1, max_length=_MAX_AGENT_MESSAGE_CHARS)
    consumed_offset: int = Field(
        validation_alias="consumedOffset",
        strict=True,
        ge=-1,
    )
    subject: _AgentMessageSubjectPayload
    activation_policy: Literal["playback_start", "playback_complete"] = Field(
        validation_alias="activationPolicy"
    )

    @model_validator(mode="after")
    def validate_consumed_offset(self) -> _AgentMessagePayload:
        if self.consumed_offset > len(self.content):
            raise ValueError("Agent consumedOffset exceeds content")
        return self


class _CompanionReplyContextPayload(_AgentMessageSubjectPayload):
    activation_policy: Literal["playback_start", "playback_complete"] = Field(
        validation_alias="activationPolicy"
    )


class _CompanionMessagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: Literal["agent", "proactive_observation", "plugin"]
    streaming: bool = Field(default=False, strict=True)
    speech_tags: bool = Field(default=False, validation_alias="speechTags", strict=True)
    speech_content: str = Field(default="", validation_alias="speechContent")
    message_id: UUID = Field(validation_alias="messageId")
    sequence: int = Field(strict=True, ge=1, le=_MAX_AGENT_MESSAGE_SEQUENCE)
    session_id: UUID = Field(validation_alias="sessionId")
    content: str = Field(min_length=1, max_length=_MAX_AGENT_MESSAGE_CHARS)
    consumed_offset: int = Field(
        validation_alias="consumedOffset",
        strict=True,
        ge=-1,
    )
    emotion: Literal["neutral", "angry", "happy", "shy", "sad"]
    interrupt_policy: Literal["resume", "drop"] = Field(
        validation_alias="interruptPolicy"
    )
    expires_at: datetime | None = Field(
        default=None,
        validation_alias="expiresAt",
    )
    supersession_key: str | None = Field(
        default=None,
        validation_alias="supersessionKey",
        max_length=200,
    )
    reply_context: _CompanionReplyContextPayload | None = Field(
        default=None,
        validation_alias="replyContext",
    )

    @field_validator("expires_at")
    @classmethod
    def require_expiry_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("Companion Message expiresAt must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_delivery_contract(self) -> _CompanionMessagePayload:
        if self.consumed_offset > len(self.content):
            raise ValueError("Companion consumedOffset exceeds content")
        if self.source == "agent":
            if self.streaming:
                raise ValueError("Agent messages cannot use proactive transport")
            if self.reply_context is None:
                raise ValueError("Agent Companion Message requires replyContext")
            if self.interrupt_policy != "resume":
                raise ValueError(
                    "Agent Companion Message must resume after interruption"
                )
            if self.expires_at is not None or self.supersession_key is not None:
                raise ValueError("Agent Companion Message cannot expire or supersede")
        else:
            if self.reply_context is not None:
                raise ValueError("Proactive Companion Message cannot have replyContext")
            if self.interrupt_policy != "drop":
                raise ValueError(
                    "Proactive Companion Message must drop after interruption"
                )
            if not self.supersession_key or (
                not self.streaming and self.expires_at is None
            ):
                raise ValueError(
                    "Proactive Companion Message requires supersessionKey "
                    "and either streaming ownership or expiry"
                )
        return self


class _SessionPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: UUID = Field(validation_alias="sessionId")
    created_at: datetime = Field(validation_alias="createdAt")
    last_message_at: datetime | None = Field(
        default=None,
        validation_alias="lastMessageAt",
    )

    @field_validator("created_at", "last_message_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("session timestamps must be timezone-aware")
        return value


class _SessionListPayload(BaseModel):
    items: list[_SessionPayload]


def parse_sse_event(block: str) -> dict[str, object]:
    event_name = "message"
    data_lines: list[str] = []
    for line in block.splitlines():
        if not line or line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
    if not data_lines:
        raise ValueError("SSE event has no data")
    data = json.loads("\n".join(data_lines))
    if not isinstance(data, dict):
        raise ValueError("SSE event data must be an object")
    return {"event": event_name, "data": data}


class ApiCompanionClient:
    def __init__(
        self,
        api_url: str,
        http_client: httpx.AsyncClient | None = None,
        *,
        api_key: str | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._http_client = http_client or httpx.AsyncClient(timeout=None)
        if api_key:
            self._http_client.headers["Authorization"] = f"Bearer {api_key}"
        self._owns_client = http_client is None
        self._after_sequence: int | None = None
        self._now = now

    @property
    def answer_start_sequence(self) -> int | None:
        return self._after_sequence

    def advance_answer_start_sequence(self, sequence: int) -> None:
        """Advance the reconnect cursor from an observed answer event."""
        if isinstance(sequence, bool) or sequence < 0:
            raise ValueError("answer sequence must be a nonnegative integer")
        if self._after_sequence is None or sequence > self._after_sequence:
            self._after_sequence = sequence

    async def ask(
        self,
        fish_id: UUID,
        content: str,
        *,
        session_id: UUID,
        reply_context: CompanionReplyContext | None = None,
        input_mode: Literal["typed", "spoken"] | None = None,
        attachment_ids: tuple[UUID, ...] = (),
    ) -> CompanionAskResult:
        body: dict[str, object] = {
            "content": content,
            "sessionId": str(session_id),
            "playbackOwner": "companion",
        }
        if input_mode is not None:
            body["inputMode"] = input_mode
        if reply_context is not None:
            body["replyContext"] = reply_context.as_payload()
        if attachment_ids:
            body["attachmentIds"] = [str(item) for item in attachment_ids]
        response = await self._http_client.post(
            f"{self._api_url}/v1/fishes/{fish_id}/ask",
            json=body,
        )
        response.raise_for_status()
        return self._accept_turn_response(response.json(), operation="ask")

    def _accept_turn_response(
        self,
        payload: object,
        *,
        operation: str,
    ) -> CompanionAskResult:
        if not isinstance(payload, dict):
            raise ValueError(f"{operation} response does not contain messageId")
        message_id = payload.get("messageId")
        if not isinstance(message_id, str):
            raise ValueError(f"{operation} response does not contain messageId")
        sequence = payload.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ValueError(f"{operation} response does not contain sequence")
        answer_expected = payload.get("answerExpected")
        if not isinstance(answer_expected, bool):
            raise ValueError(
                f"{operation} response does not contain strict answerExpected"
            )
        self.advance_answer_start_sequence(sequence)
        return CompanionAskResult(UUID(message_id), answer_expected, sequence)

    async def upsert_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        reply_context: CompanionReplyContext | None = None,
        speech_started_at: datetime | None = None,
        frame_snapshot_at: datetime | None = None,
    ) -> bool:
        payload: dict[str, object] = {"revision": revision, "content": content}
        if reply_context is not None:
            payload["replyContext"] = reply_context.as_payload()
        if speech_started_at is not None and frame_snapshot_at is not None:
            payload["speechStartedAt"] = speech_started_at.isoformat()
            payload["frameSnapshotAt"] = frame_snapshot_at.isoformat()
        try:
            response = await self._http_client.put(
                f"{self._voice_turn_url(key)}/draft",
                json=payload,
                timeout=_DRAFT_CONTROL_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return True
        except Exception as error:
            logger.warning(
                "Companion voice draft upsert missed: turn_id=%s revision=%s: %s",
                key.turn_id,
                revision,
                error,
            )
            return False

    async def cancel_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        reason: str,
    ) -> bool:
        try:
            response = await self._http_client.post(
                f"{self._voice_turn_url(key)}/cancel",
                json={"revision": revision, "reason": reason},
                timeout=_DRAFT_CONTROL_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return True
        except Exception as error:
            logger.warning(
                "Companion voice draft cancel missed: turn_id=%s revision=%s: %s",
                key.turn_id,
                revision,
                error,
            )
            return False

    async def commit_voice_turn(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        promote_revision: int,
        content: str,
        reply_context: CompanionReplyContext | None = None,
    ) -> CompanionAskResult:
        payload: dict[str, object] = {
            "revision": revision,
            "content": content,
        }
        if promote_revision > 0:
            payload["promoteRevision"] = promote_revision
        if reply_context is not None:
            payload["replyContext"] = reply_context.as_payload()
        response = await self._http_client.post(
            f"{self._voice_turn_url(key)}/commit",
            json=payload,
            timeout=_VOICE_COMMIT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return self._accept_turn_response(
            response.json(),
            operation="voice commit",
        )

    def iter_voice_draft_events(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
    ) -> AsyncIterator[dict[str, object]]:
        return self._stream_voice_draft_events(key, revision)

    async def _stream_voice_draft_events(
        self,
        key: VoiceDraftKey,
        revision: int,
    ) -> AsyncIterator[dict[str, object]]:
        async with self._http_client.stream(
            "GET",
            f"{self._voice_turn_url(key)}/events",
            headers={"Accept": "text/event-stream"},
            params={"revision": str(revision)},
            timeout=_DRAFT_EVENT_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if lines:
                        try:
                            yield parse_sse_event("\n".join(lines) + "\n\n")
                        except ValueError:
                            pass
                        lines.clear()
                    continue
                lines.append(line)
            if lines:
                yield parse_sse_event("\n".join(lines) + "\n\n")

    def _voice_turn_url(self, key: VoiceDraftKey) -> str:
        return (
            f"{self._api_url}/internal/fishes/{key.fish_id}/sessions/"
            f"{key.session_id}/voice-turns/{key.turn_id}"
        )

    async def voice_draft_capability(self) -> bool:
        try:
            response = await self._http_client.get(
                f"{self._api_url}/internal/capabilities",
                timeout=_DRAFT_CAPABILITY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            return isinstance(payload, dict) and payload.get("voiceDrafts") is True
        except Exception:
            logger.debug("Voice draft capability is unavailable", exc_info=True)
            return False

    async def get_voice_config(self, fish_id: UUID) -> CompanionVoiceConfig:
        response = await self._http_client.get(
            f"{self._api_url}/v1/fishes/{fish_id}/voice"
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
            raise ValueError("voice response does not contain config")
        config = payload["config"]
        model = config.get("model")
        voice = config.get("voice")
        if not isinstance(model, str) or not isinstance(voice, str):
            raise ValueError("voice config must contain model and voice")
        from fish_memory.tts.speech_tags import supports_speech_tags

        return CompanionVoiceConfig(
            model=model,
            voice=voice,
            speech_tags=supports_speech_tags(str(payload.get("provider", "")), model),
        )

    async def get_wake_words(self, fish_id: UUID) -> tuple[str, ...]:
        response = await self._http_client.get(
            f"{self._api_url}/v1/fishes/{fish_id}/wake-words"
        )
        response.raise_for_status()
        words = response.json().get("words")
        if not isinstance(words, list) or not words:
            raise ValueError("wake-words response does not contain words")
        return tuple(str(word) for word in words)

    async def persist_pet_voice_performance(
        self,
        fish_id: UUID,
        message_id: UUID,
        metrics: dict[str, JsonValue],
    ) -> None:
        response = await self._http_client.post(
            f"{self._api_url}/internal/fishes/{fish_id}/messages/{message_id}/"
            "pet-voice-performance",
            json={"metrics": metrics},
        )
        response.raise_for_status()

    async def record_failure(
        self, fish_id: UUID, message_id: UUID, failure: str
    ) -> None:
        response = await self._http_client.post(
            f"{self._api_url}/internal/fishes/{fish_id}/messages/{message_id}/failure",
            json={"failure": failure},
        )
        response.raise_for_status()

    async def resolve_session(
        self,
        fish_id: UUID,
        *,
        inactivity: timedelta = timedelta(hours=1),
    ) -> UUID:
        response = await self._http_client.get(
            f"{self._api_url}/v1/fishes/{fish_id}/sessions"
        )
        response.raise_for_status()
        payload = _SessionListPayload.model_validate(response.json())
        sessions = [_session_info(item) for item in payload.items]
        if not sessions:
            return (await self._create_session(fish_id)).id
        latest = sessions[0]
        if (
            latest.last_message_at is not None
            and self._now() - latest.last_message_at > inactivity
        ):
            return (await self._create_session(fish_id)).id
        return latest.id

    async def _create_session(self, fish_id: UUID) -> CompanionSessionInfo:
        response = await self._http_client.post(
            f"{self._api_url}/v1/fishes/{fish_id}/sessions"
        )
        response.raise_for_status()
        return _session_info(_SessionPayload.model_validate(response.json()))

    async def create_fish(
        self,
        *,
        role: str | None = None,
        setting: str | None = None,
        name: str | None = None,
    ) -> UUID:
        response = await self._http_client.post(
            f"{self._api_url}/v1/fishes",
            json={"role": role, "setting": setting, "name": name},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("fishId"), str):
            raise ValueError("create fish response does not contain fishId")
        return UUID(payload["fishId"])

    async def list_messages(
        self,
        fish_id: UUID,
        *,
        limit: int = 500,
    ) -> dict[str, object]:
        response = await self._http_client.get(
            f"{self._api_url}/v1/fishes/{fish_id}/messages",
            params={"limit": str(limit)},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("messages response must be an object")
        return payload

    async def get_message_performance(
        self,
        fish_id: UUID,
        message_id: UUID,
    ) -> dict[str, object]:
        response = await self._http_client.get(
            f"{self._api_url}/internal/fishes/{fish_id}/messages/"
            f"{message_id}/performance"
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("message performance response must be an object")
        return payload

    async def update_progress(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None:
        response = await self._http_client.post(
            f"{self._api_url}/v1/fishes/{fish_id}/answer/progress",
            json={
                "messageId": str(message_id),
                "playbackOwner": "companion",
                "consumedOffset": consumed_offset,
            },
        )
        response.raise_for_status()

    async def interrupt_answer(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None:
        response = await self._http_client.post(
            f"{self._api_url}/v1/fishes/{fish_id}/answer/interrupt",
            json={
                "messageId": str(message_id),
                "playbackOwner": "companion",
                "consumedOffset": consumed_offset,
            },
        )
        response.raise_for_status()

    async def consume_agent_message(
        self,
        fish_id: UUID,
        message_id: UUID,
        *,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> None:
        if (
            isinstance(consumed_offset, bool)
            or consumed_offset < 0
            or not isinstance(interrupted, bool)
        ):
            raise ValueError("Agent consumption values are invalid")
        response = await self._http_client.post(
            f"{self._api_url}/v1/fishes/{fish_id}/agent-messages/"
            f"{message_id}/consumption",
            params={"sessionId": str(session_id)},
            json={
                "consumedOffset": consumed_offset,
                "interrupted": interrupted,
            },
        )
        response.raise_for_status()

    async def consume_companion_message(
        self,
        fish_id: UUID,
        message_id: UUID,
        *,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
        source: Literal["agent", "proactive_observation", "plugin"],
    ) -> None:
        if (
            isinstance(consumed_offset, bool)
            or consumed_offset < 0
            or not isinstance(interrupted, bool)
        ):
            raise ValueError("Companion Message consumption values are invalid")
        response = await self._http_client.post(
            f"{self._api_url}/v1/fishes/{fish_id}/companion-messages/"
            f"{message_id}/consumption",
            params={"sessionId": str(session_id)},
            json={
                "consumedOffset": consumed_offset,
                "interrupted": interrupted,
            },
        )
        if response.status_code == 404 and source == "agent":
            await self.consume_agent_message(
                fish_id,
                message_id,
                session_id=session_id,
                consumed_offset=consumed_offset,
                interrupted=interrupted,
            )
            return
        if response.status_code == 404 and source in {
            "proactive_observation",
            "plugin",
        }:
            # Supersession or expiry may have atomically removed an unplayed
            # proactive Message before the Companion reports its local drop.
            return
        response.raise_for_status()

    def iter_companion_messages(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        after_sequence: int,
        on_open: Callable[[], None] | None = None,
    ) -> AsyncIterator[CompanionMessage]:
        if (
            isinstance(after_sequence, bool)
            or not 0 <= after_sequence <= _MAX_AGENT_MESSAGE_SEQUENCE
        ):
            raise ValueError("Companion Message cursor is invalid")
        return self._stream_companion_messages(
            fish_id,
            session_id,
            after_sequence,
            on_open,
        )

    async def _stream_companion_messages(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        on_open: Callable[[], None] | None,
    ) -> AsyncIterator[CompanionMessage]:
        async with self._http_client.stream(
            "GET",
            f"{self._api_url}/v1/fishes/{fish_id}/companion-messages/events",
            headers={
                "Accept": "text/event-stream",
                "Last-Event-ID": str(after_sequence),
            },
            params={
                "sessionId": str(session_id),
                "afterSequence": str(after_sequence),
                "nativeConsumer": "true",
            },
        ) as response:
            if response.status_code == 404:
                async for message in self._stream_agent_messages(
                    fish_id,
                    session_id,
                    after_sequence,
                    on_open,
                ):
                    yield message
                return
            response.raise_for_status()
            if on_open is not None:
                on_open()
            lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if lines:
                        message = _parse_companion_sse_lines(lines)
                        if message is not None:
                            yield message
                        lines.clear()
                    continue
                lines.append(line)
            if lines:
                message = _parse_companion_sse_lines(lines)
                if message is not None:
                    yield message

    def iter_plugin_expression(
        self, fish_id: UUID, message_id: UUID, *, session_id: UUID, after_offset: int
    ) -> AsyncIterator[str | dict[str, object]]:
        return self._iter_expression(
            fish_id,
            message_id,
            session_id=session_id,
            after_offset=after_offset,
            source="extensions",
        )

    def iter_proactive_expression(
        self, fish_id: UUID, message_id: UUID, *, session_id: UUID, after_offset: int
    ) -> AsyncIterator[str | dict[str, object]]:
        return self._iter_expression(
            fish_id,
            message_id,
            session_id=session_id,
            after_offset=after_offset,
            source="proactive",
        )

    async def _iter_expression(
        self,
        fish_id: UUID,
        message_id: UUID,
        *,
        session_id: UUID,
        after_offset: int,
        source: str,
    ) -> AsyncIterator[str | dict[str, object]]:
        cursor = after_offset
        async with self._http_client.stream(
            "GET",
            f"{self._api_url}/internal/{source}/{fish_id}/expressions/{message_id}/events",
            params={"sessionId": str(session_id), "afterOffset": str(cursor)},
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                value = json.loads(line[6:])
                if value.get("type") == "heartbeat":
                    continue
                if value.get("offset") != cursor:
                    raise ValueError("Proactive expression cursor mismatch")
                if value.get("type") == "end":
                    if value.get("failed"):
                        raise ValueError("Proactive expression was interrupted")
                    return
                text = value.get("text")
                if value.get("type") != "text" or not isinstance(text, str):
                    raise ValueError("Invalid proactive expression event")
                if cursor + len(text) > 2000:
                    raise ValueError("Proactive expression exceeds text budget")
                cursor += len(text)
                if value.get("speechTags") is True:
                    yield {
                        "text": text,
                        "speechContent": value.get("speechContent", ""),
                    }
                else:
                    yield text
        raise ValueError("Proactive expression ended without completion")

    def iter_agent_messages(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        after_sequence: int,
        on_open: Callable[[], None] | None = None,
    ) -> AsyncIterator[CompanionAgentMessage]:
        if (
            isinstance(after_sequence, bool)
            or not 0 <= after_sequence <= _MAX_AGENT_MESSAGE_SEQUENCE
        ):
            raise ValueError("Agent message cursor is invalid")
        return self._stream_agent_messages(
            fish_id,
            session_id,
            after_sequence,
            on_open,
        )

    async def _stream_agent_messages(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        on_open: Callable[[], None] | None,
    ) -> AsyncIterator[CompanionAgentMessage]:
        async with self._http_client.stream(
            "GET",
            f"{self._api_url}/v1/fishes/{fish_id}/agent-messages/events",
            headers={
                "Accept": "text/event-stream",
                "Last-Event-ID": str(after_sequence),
            },
            params={
                "sessionId": str(session_id),
                "afterSequence": str(after_sequence),
            },
        ) as response:
            response.raise_for_status()
            if on_open is not None:
                on_open()
            lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if lines:
                        message = _parse_agent_sse_lines(lines)
                        if message is not None:
                            yield message
                        lines.clear()
                    continue
                lines.append(line)
            if lines:
                message = _parse_agent_sse_lines(lines)
                if message is not None:
                    yield message

    async def _stream_answer(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int | None,
        on_open: Callable[[], None] | None,
    ) -> AsyncIterator[dict[str, object]]:
        params = {"sessionId": str(session_id)}
        if after_sequence is not None:
            params["afterSequence"] = str(after_sequence)
        async with self._http_client.stream(
            "GET",
            f"{self._api_url}/v1/fishes/{fish_id}/answer",
            headers={"Accept": "text/event-stream"},
            params=params,
        ) as response:
            response.raise_for_status()
            if on_open is not None:
                on_open()
            lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if lines:
                        yield parse_sse_event("\n".join(lines) + "\n\n")
                        lines.clear()
                    continue
                lines.append(line)
            if lines:
                yield parse_sse_event("\n".join(lines) + "\n\n")

    def iter_answer(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        on_open: Callable[[], None] | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        return self._stream_answer(
            fish_id,
            session_id,
            self._after_sequence,
            on_open,
        )

    def iter_home_actions(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        on_open: Callable[[], None] | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        return self._stream_home_actions(fish_id, session_id, on_open)

    async def _stream_home_actions(
        self,
        fish_id: UUID,
        session_id: UUID,
        on_open: Callable[[], None] | None,
    ) -> AsyncIterator[dict[str, object]]:
        async with self._http_client.stream(
            "GET",
            f"{self._api_url}/v1/fishes/{fish_id}/home-actions/events",
            headers={"Accept": "text/event-stream"},
            params={"sessionId": str(session_id)},
        ) as response:
            response.raise_for_status()
            if on_open is not None:
                on_open()
            lines: list[str] = []
            async for line in response.aiter_lines():
                if line == "":
                    if lines:
                        try:
                            yield parse_sse_event("\n".join(lines) + "\n\n")
                        except ValueError:
                            # Heartbeats intentionally contain no data payload.
                            pass
                        lines.clear()
                    continue
                lines.append(line)

    async def close(self) -> None:
        if self._owns_client:
            await self._http_client.aclose()


def api_websocket_url(api_url: str, path: str) -> str:
    if api_url.startswith("https://"):
        base = "wss://" + api_url.removeprefix("https://")
    elif api_url.startswith("http://"):
        base = "ws://" + api_url.removeprefix("http://")
    else:
        base = api_url
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _session_info(value: _SessionPayload) -> CompanionSessionInfo:
    return CompanionSessionInfo(
        id=value.session_id,
        created_at=value.created_at,
        last_message_at=value.last_message_at,
    )


def _parse_agent_sse_lines(lines: list[str]) -> CompanionMessage | None:
    if not any(line.partition(":")[0] == "data" for line in lines):
        return None
    event = parse_sse_event("\n".join(lines) + "\n\n")
    if event.get("event") == "agent.heartbeat":
        return None
    if event.get("event") != "agent.message":
        raise ValueError("Agent message stream contains an unexpected event")
    payload = _AgentMessagePayload.model_validate(event.get("data"))
    return CompanionMessage(
        message_id=payload.message_id,
        sequence=payload.sequence,
        session_id=payload.session_id,
        content=payload.content,
        consumed_offset=payload.consumed_offset,
        subject_kind=payload.subject.kind,
        subject_id=payload.subject.id,
        activation_policy=payload.activation_policy,
    )


def _parse_companion_sse_lines(lines: list[str]) -> CompanionMessage | None:
    if not any(line.partition(":")[0] == "data" for line in lines):
        return None
    event = parse_sse_event("\n".join(lines) + "\n\n")
    if event.get("event") == "companion.heartbeat":
        return None
    if event.get("event") != "companion.message":
        raise ValueError("Companion Message stream contains an unexpected event")
    payload = _CompanionMessagePayload.model_validate(event.get("data"))
    context = payload.reply_context
    return CompanionMessage(
        message_id=payload.message_id,
        sequence=payload.sequence,
        session_id=payload.session_id,
        content=payload.content,
        consumed_offset=payload.consumed_offset,
        subject_kind=context.kind if context is not None else None,
        subject_id=context.id if context is not None else None,
        activation_policy=(context.activation_policy if context is not None else None),
        source=payload.source,
        emotion=payload.emotion,
        interrupt_policy=payload.interrupt_policy,
        expires_at=payload.expires_at,
        supersession_key=payload.supersession_key,
        streaming=payload.streaming,
        speech_tags=payload.speech_tags,
        speech_content=payload.speech_content,
    )
