from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from uuid import UUID

_CONNECTOR_TYPE_SLUG_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")


class ConnectorTypeSlug(str):
    """A stable, provider-neutral connector type identifier."""

    def __new__(cls, value: str) -> ConnectorTypeSlug:
        if not _CONNECTOR_TYPE_SLUG_RE.fullmatch(value):
            raise ValueError("connector type must match ^[a-z][a-z0-9._-]{0,63}$")
        return super().__new__(cls, value)


class AgentRunStatus(StrEnum):
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    ABANDONED = "abandoned"

    @property
    def is_terminal(self) -> bool:
        return self in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
            AgentRunStatus.ABANDONED,
        }


class AgentHistorySource(StrEnum):
    NONE = "none"
    FISH_SESSION = "fish_session"
    HERMES_SESSION = "hermes_session"
    LOCAL_PROJECTION = "local_projection"
    FISH_SESSION_HERMES_SESSION = "fish_session_hermes_session"
    FISH_SESSION_LOCAL_PROJECTION = "fish_session_local_projection"


class AgentInteractionKind(StrEnum):
    APPROVAL = "approval"
    INPUT = "input"


class AgentInteractionState(StrEnum):
    CLOSED = "closed"
    RESOLVED = "resolved"
    WAITING_MESSAGE = "waiting_message"
    WAITING_USER = "waiting_user"
    RESPONDING = "responding"


class AgentMessageSubjectKind(StrEnum):
    RUN_RESULT = "run_result"
    INTERACTION = "interaction"


class AgentActivationPolicy(StrEnum):
    PLAYBACK_START = "playback_start"
    PLAYBACK_COMPLETE = "playback_complete"


class AgentTaskLaneDecision(IntEnum):
    """Compact, provider-facing T-lane decision encoded in ``t``."""

    NOT_TASK = 0
    TASK = 1


@dataclass(frozen=True, slots=True)
class AgentEndpoint:
    id: UUID
    connector_type: ConnectorTypeSlug
    base_url: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgentTarget:
    id: UUID
    endpoint_id: UUID
    external_key: str
    route_prefix: str
    credential_ref: str | None
    connector_config: Mapping[str, object] | None
    display_name: str
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgentResolvedTarget:
    target: AgentTarget
    endpoint: AgentEndpoint


@dataclass(frozen=True, slots=True)
class AgentTask:
    id: UUID
    fish_id: UUID
    session_id: UUID
    target_id: UUID
    external_session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AgentRun:
    id: UUID
    task_id: UUID
    target_id: UUID
    fish_id: UUID
    session_id: UUID
    previous_run_id: UUID | None
    source_message_id: UUID | None
    confirmation_message_id: UUID | None
    input: str
    external_run_id: str | None
    history_source: AgentHistorySource
    status: AgentRunStatus
    output: str | None
    result_message_id: UUID | None
    error_code: str | None
    error_message: str | None
    provider_data: Mapping[str, object] | None
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    finished_at: datetime | None
    origin: str = "user"
    origin_context: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class AgentInteraction:
    id: UUID
    run_id: UUID
    fish_id: UUID
    session_id: UUID
    kind: AgentInteractionKind
    request_message_id: UUID | None
    response_message_id: UUID | None
    next_run_id: UUID | None
    provider_data: Mapping[str, object] | None
    created_at: datetime
    resolved_at: datetime | None
    closed_at: datetime | None

    @property
    def state(self) -> AgentInteractionState:
        if self.closed_at is not None:
            return AgentInteractionState.CLOSED
        if self.resolved_at is not None:
            return AgentInteractionState.RESOLVED
        if self.request_message_id is None:
            return AgentInteractionState.WAITING_MESSAGE
        if self.response_message_id is None:
            return AgentInteractionState.WAITING_USER
        return AgentInteractionState.RESPONDING


@dataclass(frozen=True, slots=True)
class AgentRunWorkItem:
    run: AgentRun
    task: AgentTask
    target: AgentTarget
    endpoint: AgentEndpoint


@dataclass(frozen=True, slots=True)
class AgentInteractionWorkItem:
    interaction: AgentInteraction
    run: AgentRun
    task: AgentTask
    target: AgentTarget
    endpoint: AgentEndpoint


@dataclass(frozen=True, slots=True)
class AgentTaskCandidate:
    """One bounded candidate returned by the persistence layer."""

    task: AgentTask
    run: AgentRun
    interaction: AgentInteraction | None
    target: AgentTarget
    endpoint: AgentEndpoint
    interaction_request_content: str | None = None
    interaction_request_session_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class AgentMessageSubject:
    kind: AgentMessageSubjectKind
    id: UUID


@dataclass(frozen=True, slots=True)
class AgentMessageEnvelope:
    message_id: UUID
    sequence: int
    session_id: UUID
    subject: AgentMessageSubject
    activation_policy: AgentActivationPolicy
    content: str
    consumed_offset: int
