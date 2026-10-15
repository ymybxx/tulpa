from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(StrEnum):
    INITIALIZED = "initialized"
    GENERATING = "generating"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Session:
    id: UUID
    fish_id: UUID
    created_at: datetime
    last_message_at: datetime | None
    history_window_start_sequence: int = 0
    history_window_token_count: int = 0


@dataclass(frozen=True, slots=True)
class SessionSummary:
    id: UUID
    fish_id: UUID
    created_at: datetime
    last_message_at: datetime | None
    history_window_start_sequence: int = 0
    history_window_token_count: int = 0


@dataclass(frozen=True, slots=True)
class Message:
    id: UUID
    fish_id: UUID
    session_id: UUID
    sequence: int
    role: Role
    content: str
    status: MessageStatus
    created_at: datetime
    context_through_sequence: int | None = None
    consumed_offset: int | None = None
    finished_at: datetime | None = None
    history_token_count: int = 0
    # Which outside service let this reply down and how ("chat:timeout"),
    # when it failed or could not be spoken. See fish_memory.failures.
    failure: str | None = None


def visible_message_content(message: Message) -> str:
    if message.role is Role.USER:
        return message.content
    offset = message.consumed_offset
    if offset in (None, -1):
        return message.content
    return message.content[:offset]


@dataclass(frozen=True, slots=True)
class FishSummary:
    fish_id: UUID
    last_message: str
    last_active_at: datetime | None
    message_count: int
    is_assistant: bool = False
    name: str = ""
    role: str = ""
    evaluation: bool = False
