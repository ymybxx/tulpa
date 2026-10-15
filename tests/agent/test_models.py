from datetime import UTC, datetime
from uuid import uuid4

import pytest

from fish_memory.agent.models import (
    AgentActivationPolicy,
    AgentInteraction,
    AgentInteractionKind,
    AgentInteractionState,
    AgentMessageEnvelope,
    AgentMessageSubject,
    AgentMessageSubjectKind,
    AgentRunStatus,
    AgentTaskLaneDecision,
    ConnectorTypeSlug,
)


def _interaction(**changes: object) -> AgentInteraction:
    values: dict[str, object] = {
        "id": uuid4(),
        "run_id": uuid4(),
        "fish_id": uuid4(),
        "session_id": uuid4(),
        "kind": AgentInteractionKind.INPUT,
        "request_message_id": None,
        "response_message_id": None,
        "next_run_id": None,
        "provider_data": None,
        "created_at": datetime.now(UTC),
        "resolved_at": None,
        "closed_at": None,
    }
    values.update(changes)
    return AgentInteraction(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "expected_state"),
    [
        ({"closed_at": datetime.now(UTC)}, AgentInteractionState.CLOSED),
        ({"resolved_at": datetime.now(UTC)}, AgentInteractionState.RESOLVED),
        ({}, AgentInteractionState.WAITING_MESSAGE),
        (
            {"request_message_id": uuid4()},
            AgentInteractionState.WAITING_USER,
        ),
        (
            {"request_message_id": uuid4(), "response_message_id": uuid4()},
            AgentInteractionState.RESPONDING,
        ),
    ],
)
def test_agent_interaction_derives_state_from_message_and_resolution_fields(
    changes: dict[str, object],
    expected_state: AgentInteractionState,
) -> None:
    assert _interaction(**changes).state is expected_state


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (AgentRunStatus.QUEUED, False),
        (AgentRunStatus.DISPATCHING, False),
        (AgentRunStatus.RUNNING, False),
        (AgentRunStatus.WAITING_USER, False),
        (AgentRunStatus.UNKNOWN, False),
        (AgentRunStatus.COMPLETED, True),
        (AgentRunStatus.FAILED, True),
        (AgentRunStatus.CANCELLED, True),
        (AgentRunStatus.ABANDONED, True),
    ],
)
def test_agent_run_status_identifies_only_local_terminal_states(
    status: AgentRunStatus,
    expected: bool,
) -> None:
    assert status.is_terminal is expected


@pytest.mark.parametrize("value", ["hermes", "github.copilot", "v2-agent_1"])
def test_connector_type_slug_accepts_stable_plugin_ready_slugs(value: str) -> None:
    assert ConnectorTypeSlug(value) == value


@pytest.mark.parametrize("value", ["Hermes", "1agent", "agent/remote", "", "a" * 65])
def test_connector_type_slug_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        ConnectorTypeSlug(value)


def test_agent_message_envelope_carries_delivery_content_and_progress() -> None:
    envelope = AgentMessageEnvelope(
        message_id=uuid4(),
        sequence=42,
        session_id=uuid4(),
        subject=AgentMessageSubject(AgentMessageSubjectKind.RUN_RESULT, uuid4()),
        activation_policy=AgentActivationPolicy.PLAYBACK_START,
        content="任务已经完成。",
        consumed_offset=7,
    )

    assert envelope.content == "任务已经完成。"
    assert envelope.consumed_offset == 7


def test_task_lane_wire_decisions_are_stable_compact_integers() -> None:
    assert AgentTaskLaneDecision.NOT_TASK == 0
    assert AgentTaskLaneDecision.TASK == 1
    assert list(AgentTaskLaneDecision) == [
        AgentTaskLaneDecision.NOT_TASK,
        AgentTaskLaneDecision.TASK,
    ]
