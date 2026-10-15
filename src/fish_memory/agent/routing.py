"""Immutable T-lane candidate mapping and local pre-publish commits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from fish_memory.agent.detection import looks_like_agent_task_retry
from fish_memory.agent.models import (
    AgentInteraction,
    AgentResolvedTarget,
    AgentRun,
    AgentRunStatus,
    AgentTask,
    AgentTaskCandidate,
)
from fish_memory.agent.runtime_state import AgentRuntimeState

if TYPE_CHECKING:
    from fish_memory.agent.replies import PreparedAgentInteractionReply

DEFAULT_ROUTING_POLICY_VERSION = "agent-task-routing-v3"
_MAX_TASK_CANDIDATES = 8
_MAX_TASK_TITLE_CHARS = 200


class AgentTaskDecisionKind(StrEnum):
    NEW_TASK = "new_task"
    CONTINUE_TASK = "continue_task"

    @classmethod
    def from_candidate_index(
        cls,
        candidate_index: int | None,
    ) -> AgentTaskDecisionKind:
        """Decode the compact T-lane task route without a wire ``k`` field."""

        if candidate_index is None:
            return cls.NEW_TASK
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise ValueError("task candidate index must be an integer")
        if candidate_index < 0:
            raise ValueError("task candidate index must be non-negative")
        return cls.CONTINUE_TASK


class AgentTaskRouteMode(StrEnum):
    NEW_RUN = "new_run"
    STEER = "steer"
    RETRY = "retry"
    REPLY_ONLY = "reply_only"


@dataclass(frozen=True, slots=True)
class LocalConnectorCapabilityPolicy:
    """Network-free capability seam used while freezing T-lane context."""

    supported_connector_types: frozenset[str] = frozenset({"hermes"})
    steer_connector_types: frozenset[str] = frozenset({"hermes"})
    routing_policy_version: str = DEFAULT_ROUTING_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.routing_policy_version.strip():
            raise ValueError("routing policy version must not be blank")
        if not self.steer_connector_types.issubset(self.supported_connector_types):
            raise ValueError("steer connector types must be supported")

    def supports_connector(self, connector_type: str) -> bool:
        return connector_type in self.supported_connector_types

    def supports_steer(self, connector_type: str) -> bool:
        return connector_type in self.steer_connector_types


@dataclass(frozen=True, slots=True)
class TaskLaneCandidate:
    index: int
    record: AgentTaskCandidate
    mode: AgentTaskRouteMode

    @property
    def task(self) -> AgentTask:
        return self.record.task

    @property
    def run(self) -> AgentRun:
        return self.record.run

    @property
    def interaction(self) -> AgentInteraction | None:
        return self.record.interaction

    def prompt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "i": self.index,
            "title": _task_title(self.record.run.input),
        }
        if self.record.interaction is not None:
            request = self.record.interaction_request_content
            if request is None or not request.strip():
                raise ValueError("interaction candidate is missing its request content")
            payload["request"] = request
        return payload

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "taskId": str(self.record.task.id),
            "runId": str(self.record.run.id),
            "interactionId": (
                None
                if self.record.interaction is None
                else str(self.record.interaction.id)
            ),
            "targetId": str(self.record.task.target_id),
            "mode": self.mode.value,
        }


@dataclass(frozen=True, slots=True)
class TaskLaneContext:
    candidates: tuple[TaskLaneCandidate, ...]
    routing_policy_version: str = DEFAULT_ROUTING_POLICY_VERSION
    has_recent_agent_reply: bool = False

    @classmethod
    def empty(
        cls,
        routing_policy_version: str = DEFAULT_ROUTING_POLICY_VERSION,
    ) -> TaskLaneContext:
        return cls((), routing_policy_version)

    @classmethod
    def from_records(
        cls,
        records: Sequence[AgentTaskCandidate],
        *,
        current_content: str,
        capability_policy: LocalConnectorCapabilityPolicy,
        has_recent_agent_reply: bool = False,
    ) -> TaskLaneContext:
        if len(records) > _MAX_TASK_CANDIDATES:
            raise ValueError("task candidate context exceeds its bounded limit")
        candidates: list[TaskLaneCandidate] = []
        seen_tasks: set[UUID] = set()
        for index, record in enumerate(records):
            _validate_record(record)
            if record.task.id in seen_tasks:
                raise ValueError("task candidate context contains a duplicate task")
            seen_tasks.add(record.task.id)
            candidates.append(
                TaskLaneCandidate(
                    index=index,
                    record=record,
                    mode=_route_mode(record, current_content, capability_policy),
                )
            )
        return cls(
            tuple(candidates),
            capability_policy.routing_policy_version,
            has_recent_agent_reply,
        )

    def candidate(self, index: int) -> TaskLaneCandidate:
        if isinstance(index, bool) or not isinstance(index, int):
            raise AgentTaskRouteConflictError("candidate index must be an integer")
        if index < 0 or index >= len(self.candidates):
            raise AgentTaskRouteConflictError(
                "candidate index is outside frozen context"
            )
        candidate = self.candidates[index]
        if candidate.index != index:
            raise AgentTaskRouteConflictError("candidate index mapping is inconsistent")
        return candidate

    def prompt_payload(self) -> list[dict[str, object]]:
        return [candidate.prompt_payload() for candidate in self.candidates]

    @property
    def has_interaction_focus(self) -> bool:
        return any(candidate.interaction is not None for candidate in self.candidates)

    @property
    def has_unfinished_task(self) -> bool:
        return any(
            not candidate.run.status.is_terminal for candidate in self.candidates
        )

    @property
    def requires_probe(self) -> bool:
        return (
            self.has_recent_agent_reply
            or self.has_unfinished_task
            or self.has_interaction_focus
        )

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "routingPolicyVersion": self.routing_policy_version,
            "hasRecentAgentReply": self.has_recent_agent_reply,
            "candidates": [
                candidate.fingerprint_payload() for candidate in self.candidates
            ],
        }


class AgentTaskRouteConflictError(RuntimeError):
    """The frozen T decision can no longer be committed safely."""


class AgentInteractionReplyUnrelatedError(AgentTaskRouteConflictError):
    """The selected Interaction does not match the current user message."""


@dataclass(frozen=True, slots=True)
class PreparedAgentTaskRoute:
    """Frozen local T decision, with no run or interaction mutation yet."""

    fish_id: UUID
    target_id: UUID
    kind: AgentTaskDecisionKind
    content: str
    candidate: TaskLaneCandidate | None = None
    interaction_reply: PreparedAgentInteractionReply | None = None


class AgentTaskCommitStore(Protocol):
    async def create_task_with_first_run(
        self,
        fish_id: UUID,
        target_id: UUID,
        title: str,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        input: str,
    ) -> tuple[AgentTask, AgentRun]: ...

    async def create_continuation_run(
        self,
        task_id: UUID,
        expected_previous_run_id: UUID,
        source_message_id: UUID,
        input: str,
        confirmation_message_id: UUID | None,
    ) -> AgentRun: ...

    async def queue_running_steer(
        self,
        run_id: UUID,
        source_message_id: UUID,
    ) -> AgentRun | None: ...

    async def abandon_unknown_and_create_retry(
        self,
        run_id: UUID,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        input: str,
    ) -> AgentRun | None: ...


class AgentInteractionReplyService(Protocol):
    async def prepare_interaction_reply(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        interaction_id: UUID,
        content: str,
    ) -> PreparedAgentInteractionReply: ...

    async def commit_prepared_interaction(
        self,
        prepared: PreparedAgentInteractionReply,
        *,
        source_message_id: UUID,
        confirmation_message_id: UUID,
    ) -> AgentRun | None: ...


class AgentTaskRouteCommitter(Protocol):
    async def prepare_selected_route(
        self,
        *,
        fish_id: UUID,
        resolved_target: AgentResolvedTarget,
        task_context: TaskLaneContext,
        kind: AgentTaskDecisionKind,
        candidate_indexes: Sequence[int],
        content: str,
    ) -> PreparedAgentTaskRoute: ...

    async def commit_prepared_route(
        self,
        prepared: PreparedAgentTaskRoute,
        *,
        source_message_id: UUID,
        confirmation_message_id: UUID,
    ) -> AgentRun: ...

    async def commit_selected_route(
        self,
        *,
        fish_id: UUID,
        resolved_target: AgentResolvedTarget,
        task_context: TaskLaneContext,
        kind: AgentTaskDecisionKind,
        candidate_indexes: Sequence[int],
        source_message_id: UUID,
        confirmation_message_id: UUID,
        content: str,
    ) -> AgentRun: ...


class AgentTaskRouteService:
    def __init__(
        self,
        store: AgentTaskCommitStore,
        *,
        interaction_reply_service: AgentInteractionReplyService | None = None,
        runtime_state: AgentRuntimeState | None = None,
    ) -> None:
        self._store = store
        self._interaction_reply_service = interaction_reply_service
        self._runtime_state = runtime_state

    async def commit_selected_route(
        self,
        *,
        fish_id: UUID,
        resolved_target: AgentResolvedTarget,
        task_context: TaskLaneContext,
        kind: AgentTaskDecisionKind,
        candidate_indexes: Sequence[int],
        source_message_id: UUID,
        confirmation_message_id: UUID,
        content: str,
    ) -> AgentRun:
        prepared = await self.prepare_selected_route(
            fish_id=fish_id,
            resolved_target=resolved_target,
            task_context=task_context,
            kind=kind,
            candidate_indexes=candidate_indexes,
            content=content,
        )
        return await self.commit_prepared_route(
            prepared,
            source_message_id=source_message_id,
            confirmation_message_id=confirmation_message_id,
        )

    async def prepare_selected_route(
        self,
        *,
        fish_id: UUID,
        resolved_target: AgentResolvedTarget,
        task_context: TaskLaneContext,
        kind: AgentTaskDecisionKind,
        candidate_indexes: Sequence[int],
        content: str,
    ) -> PreparedAgentTaskRoute:
        """Freeze routing and any classifier result without queuing Agent work."""

        normalized_content = content.strip()
        if not normalized_content:
            raise AgentTaskRouteConflictError("task input must not be blank")
        indexes = tuple(candidate_indexes)
        if kind is AgentTaskDecisionKind.NEW_TASK:
            if indexes:
                raise AgentTaskRouteConflictError(
                    "new_task must not select a candidate"
                )
            return _prepare_new_task(fish_id, resolved_target, normalized_content)
        if kind is not AgentTaskDecisionKind.CONTINUE_TASK:
            raise AgentTaskRouteConflictError(
                "only new_task and continue_task can be committed"
            )
        if len(indexes) != 1:
            raise AgentTaskRouteConflictError(
                "continue_task must select exactly one candidate"
            )
        candidate = task_context.candidate(indexes[0])
        _validate_record(candidate.record)
        if candidate.task.fish_id != fish_id:
            raise AgentTaskRouteConflictError("task candidate belongs to another Fish")
        if candidate.interaction is not None:
            if (
                candidate.record.interaction_request_session_id
                != candidate.task.session_id
            ):
                raise AgentTaskRouteConflictError(
                    "interaction request does not belong to the Task Session"
                )
            replies = self._interaction_reply_service
            if replies is None:
                raise AgentTaskRouteConflictError(
                    "interaction candidate has no reply service"
                )
            try:
                interaction_reply = await replies.prepare_interaction_reply(
                    fish_id=fish_id,
                    session_id=candidate.task.session_id,
                    interaction_id=candidate.interaction.id,
                    content=normalized_content,
                )
            except AgentInteractionReplyUnrelatedError:
                return _prepare_new_task(fish_id, resolved_target, normalized_content)
            return PreparedAgentTaskRoute(
                fish_id,
                candidate.task.target_id,
                kind,
                normalized_content,
                candidate,
                interaction_reply,
            )
        if candidate.mode is AgentTaskRouteMode.NEW_RUN:
            if not candidate.run.status.is_terminal:
                raise AgentTaskRouteConflictError("new_run candidate is not terminal")
        elif candidate.mode is AgentTaskRouteMode.STEER:
            if candidate.run.status is not AgentRunStatus.RUNNING:
                raise AgentTaskRouteConflictError("steer candidate is not running")
        elif candidate.mode is AgentTaskRouteMode.RETRY:
            if candidate.run.status is not AgentRunStatus.UNKNOWN:
                raise AgentTaskRouteConflictError("retry candidate is not unknown")
        else:
            raise AgentTaskRouteConflictError(
                "reply_only candidate requires exact reply routing"
            )
        return PreparedAgentTaskRoute(
            fish_id,
            candidate.task.target_id,
            kind,
            normalized_content,
            candidate,
        )

    async def commit_prepared_route(
        self,
        prepared: PreparedAgentTaskRoute,
        *,
        source_message_id: UUID,
        confirmation_message_id: UUID,
    ) -> AgentRun:
        """Apply one prepared route using the Store's live ownership/state CAS."""

        if self._runtime_state is not None and not self._runtime_state.enabled:
            raise AgentTaskRouteConflictError("Agent tasks are disabled")
        if prepared.kind is AgentTaskDecisionKind.NEW_TASK:
            _task, run = await self._store.create_task_with_first_run(
                prepared.fish_id,
                prepared.target_id,
                _task_title(prepared.content),
                source_message_id,
                confirmation_message_id,
                prepared.content,
            )
            return run
        candidate = prepared.candidate
        if (
            prepared.kind is not AgentTaskDecisionKind.CONTINUE_TASK
            or candidate is None
            or candidate.task.fish_id != prepared.fish_id
            or candidate.task.target_id != prepared.target_id
        ):
            raise AgentTaskRouteConflictError("prepared task route scope changed")
        if candidate.interaction is not None:
            replies = self._interaction_reply_service
            interaction_reply = prepared.interaction_reply
            if replies is None or interaction_reply is None:
                raise AgentTaskRouteConflictError(
                    "prepared interaction reply is missing"
                )
            if (
                interaction_reply.fish_id != prepared.fish_id
                or interaction_reply.session_id != candidate.task.session_id
                or interaction_reply.snapshot.context.id != candidate.interaction.id
                or interaction_reply.content != prepared.content
            ):
                raise AgentTaskRouteConflictError("prepared interaction scope changed")
            continued = await replies.commit_prepared_interaction(
                interaction_reply,
                source_message_id=source_message_id,
                confirmation_message_id=confirmation_message_id,
            )
            return candidate.run if continued is None else continued
        if candidate.mode is AgentTaskRouteMode.NEW_RUN:
            return await self._store.create_continuation_run(
                candidate.task.id,
                candidate.run.id,
                source_message_id,
                prepared.content,
                confirmation_message_id,
            )
        if candidate.mode is AgentTaskRouteMode.STEER:
            run = await self._store.queue_running_steer(
                candidate.run.id,
                source_message_id,
            )
        elif candidate.mode is AgentTaskRouteMode.RETRY:
            run = await self._store.abandon_unknown_and_create_retry(
                candidate.run.id,
                source_message_id,
                confirmation_message_id,
                prepared.content,
            )
        else:
            raise AgentTaskRouteConflictError(
                "reply_only candidate requires exact reply routing"
            )
        if run is None:
            raise AgentTaskRouteConflictError("task route commit lost its state CAS")
        return run


def _prepare_new_task(
    fish_id: UUID,
    resolved_target: AgentResolvedTarget,
    content: str,
) -> PreparedAgentTaskRoute:
    if resolved_target.target.endpoint_id != resolved_target.endpoint.id:
        raise AgentTaskRouteConflictError("Target does not belong to its Endpoint")
    if not resolved_target.target.enabled:
        raise AgentTaskRouteConflictError("Agent target is unavailable")
    return PreparedAgentTaskRoute(
        fish_id,
        resolved_target.target.id,
        AgentTaskDecisionKind.NEW_TASK,
        content,
    )


def _validate_record(record: AgentTaskCandidate) -> None:
    if record.run.task_id != record.task.id:
        raise ValueError("candidate Run does not belong to its Task")
    if (
        record.run.fish_id != record.task.fish_id
        or record.run.session_id != record.task.session_id
    ):
        raise ValueError("candidate Run does not belong to its Task scope")
    if record.task.target_id != record.target.id:
        raise ValueError("candidate Task does not belong to its Target")
    if record.run.target_id != record.target.id:
        raise ValueError("candidate Run does not belong to its Target")
    if record.target.endpoint_id != record.endpoint.id:
        raise ValueError("candidate Target does not belong to its Endpoint")
    if record.interaction is not None and record.interaction.run_id != record.run.id:
        raise ValueError("candidate Interaction does not belong to its Run")
    if record.interaction is not None and (
        record.interaction.fish_id != record.task.fish_id
        or record.interaction.session_id != record.task.session_id
    ):
        raise ValueError("candidate Interaction does not belong to its Task scope")
    if record.interaction is not None and (
        record.interaction_request_content is None
        or not record.interaction_request_content.strip()
        or record.interaction_request_session_id is None
    ):
        raise ValueError("candidate Interaction request is incomplete")
    if (
        record.interaction is not None
        and record.interaction_request_session_id != record.task.session_id
    ):
        raise ValueError("candidate Interaction request is outside its Task scope")
    if record.interaction is None and (
        record.interaction_request_content is not None
        or record.interaction_request_session_id is not None
    ):
        raise ValueError("candidate without Interaction has request metadata")


def _route_mode(
    record: AgentTaskCandidate,
    content: str,
    policy: LocalConnectorCapabilityPolicy,
) -> AgentTaskRouteMode:
    if record.interaction is not None:
        return AgentTaskRouteMode.REPLY_ONLY
    status = record.run.status
    if status.is_terminal:
        return AgentTaskRouteMode.NEW_RUN
    if status is AgentRunStatus.RUNNING:
        if policy.supports_steer(str(record.endpoint.connector_type)):
            return AgentTaskRouteMode.STEER
        return AgentTaskRouteMode.REPLY_ONLY
    if status is AgentRunStatus.UNKNOWN and looks_like_agent_task_retry(content):
        return AgentTaskRouteMode.RETRY
    return AgentTaskRouteMode.REPLY_ONLY


def _task_title(content: str) -> str:
    compact = " ".join(content.split())
    return compact[:_MAX_TASK_TITLE_CHARS]
