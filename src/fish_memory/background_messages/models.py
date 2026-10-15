from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from fish_memory.agent.models import AgentActivationPolicy, AgentMessageSubjectKind


class BackgroundMessageSource(StrEnum):
    AGENT = "agent"
    PROACTIVE_OBSERVATION = "proactive_observation"
    PLUGIN = "plugin"


class BackgroundMessageInterruptPolicy(StrEnum):
    RESUME = "resume"
    DROP = "drop"


@dataclass(frozen=True, slots=True)
class BackgroundMessageReplyContext:
    kind: AgentMessageSubjectKind
    id: UUID
    activation_policy: AgentActivationPolicy


@dataclass(frozen=True, slots=True)
class BackgroundAssistantMessage:
    source: BackgroundMessageSource
    message_id: UUID
    sequence: int
    session_id: UUID
    content: str
    consumed_offset: int
    emotion: str
    interrupt_policy: BackgroundMessageInterruptPolicy
    expires_at: datetime | None
    supersession_key: str | None
    reply_context: BackgroundMessageReplyContext | None


__all__ = [
    "BackgroundAssistantMessage",
    "BackgroundMessageInterruptPolicy",
    "BackgroundMessageReplyContext",
    "BackgroundMessageSource",
]
