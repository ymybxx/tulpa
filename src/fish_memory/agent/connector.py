from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

import httpx

from fish_memory.agent.credentials import AgentCredentialResolver
from fish_memory.agent.models import (
    AgentResolvedTarget,
    AgentRunWorkItem,
    ConnectorTypeSlug,
)

AGENT_CONNECTOR_CONTRACT_VERSION = 1
_MAX_CONNECTOR_ERROR_CHARS = 4_000
_MAX_RETRY_AFTER_CHARS = 128

AgentConnectorErrorCode = Literal[
    "endpoint_unavailable",
    "authentication_required",
    "permission_denied",
    "rate_limited",
    "invalid_request",
    "unsupported_capability",
    "protocol_error",
    "dispatch_unknown",
    "timeout",
    "cancelled",
    "provider_error",
]


class ConnectorRunState(StrEnum):
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ConnectorRunState.COMPLETED,
            ConnectorRunState.FAILED,
            ConnectorRunState.CANCELLED,
        }


class AgentProviderEventKind(StrEnum):
    STATE = "state"
    APPROVAL = "approval"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class ConnectorDescriptor:
    type_slug: str
    contract_version: int
    display_name: str


@dataclass(frozen=True, slots=True)
class AgentConnectorRuntime:
    credential_resolver: AgentCredentialResolver
    http_client: httpx.AsyncClient | None = None
    request_timeout_seconds: float = 30.0
    sse_read_timeout_seconds: float = 90.0


@dataclass(frozen=True, slots=True)
class AgentCapabilities:
    run_submission: bool
    run_status: bool
    run_events: bool
    session_history: bool
    approval: bool
    steer: bool
    stop: bool


@dataclass(frozen=True, slots=True)
class AgentHistoryMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    input: str
    instructions: str | None = None
    conversation_history: Sequence[AgentHistoryMessage] = ()


@dataclass(frozen=True, slots=True)
class AgentRunAccepted:
    external_run_id: str
    state: ConnectorRunState


@dataclass(frozen=True, slots=True)
class AgentProviderEvent:
    kind: AgentProviderEventKind
    state: ConnectorRunState
    output: str | None = None
    error_message: str | None = None
    approval_choices: tuple[str, ...] = ()
    provider_data: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class AgentRunSnapshot:
    state: ConnectorRunState
    output: str | None = None
    error_message: str | None = None
    provider_data: Mapping[str, object] | None = None


class AgentConnectorError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: AgentConnectorErrorCode,
        retryable: bool = False,
        dispatch_uncertain: bool = False,
        retry_after: str | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.dispatch_uncertain = dispatch_uncertain
        self.retry_after = (
            retry_after[:_MAX_RETRY_AFTER_CHARS] if retry_after is not None else None
        )
        super().__init__(message[:_MAX_CONNECTOR_ERROR_CHARS])


class AgentConnector(Protocol):
    async def probe(self, target: AgentResolvedTarget) -> AgentCapabilities: ...

    def preflight_run(self, work: AgentRunWorkItem) -> AgentRunWorkItem: ...

    async def start_run(
        self,
        work: AgentRunWorkItem,
        request: AgentRunRequest,
    ) -> AgentRunAccepted: ...

    def watch_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
    ) -> AsyncIterator[AgentProviderEvent]: ...

    async def get_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
    ) -> AgentRunSnapshot: ...

    async def get_session_history(
        self,
        work: AgentRunWorkItem,
    ) -> Sequence[AgentHistoryMessage]: ...

    async def respond_approval(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
        choice: str,
        resolve_all: bool,
    ) -> None: ...

    async def steer_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
        input: str,
    ) -> None: ...

    async def stop_run(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
    ) -> None: ...

    async def close(self) -> None: ...


class AgentConnectorFactory(Protocol):
    @property
    def descriptor(self) -> ConnectorDescriptor: ...

    def validate_target(self, target: AgentResolvedTarget) -> AgentResolvedTarget: ...

    def create(self, runtime: AgentConnectorRuntime) -> AgentConnector: ...


class AgentConnectorRegistry:
    def __init__(self) -> None:
        self._factories: dict[ConnectorTypeSlug, AgentConnectorFactory] = {}

    def register(self, factory: AgentConnectorFactory) -> None:
        descriptor = factory.descriptor
        slug = ConnectorTypeSlug(descriptor.type_slug)
        if descriptor.contract_version != AGENT_CONNECTOR_CONTRACT_VERSION:
            raise ValueError(
                "connector contract version "
                f"{descriptor.contract_version} is incompatible with "
                f"{AGENT_CONNECTOR_CONTRACT_VERSION}"
            )
        if slug in self._factories:
            raise ValueError(f"connector type {slug!r} is already registered")
        self._factories[slug] = factory

    def require(self, type_slug: str) -> AgentConnectorFactory:
        factory: AgentConnectorFactory | None = None
        try:
            slug = ConnectorTypeSlug(type_slug)
        except ValueError:
            pass
        else:
            factory = self._factories.get(slug)
        if factory is None:
            raise AgentConnectorError(
                f"connector type {type_slug!r} is not registered",
                code="unsupported_capability",
            )
        return factory


__all__ = [
    "AGENT_CONNECTOR_CONTRACT_VERSION",
    "AgentCapabilities",
    "AgentConnector",
    "AgentConnectorError",
    "AgentConnectorFactory",
    "AgentConnectorRegistry",
    "AgentConnectorRuntime",
    "AgentHistoryMessage",
    "AgentProviderEvent",
    "AgentProviderEventKind",
    "AgentRunAccepted",
    "AgentRunRequest",
    "AgentRunSnapshot",
    "ConnectorDescriptor",
    "ConnectorRunState",
]
