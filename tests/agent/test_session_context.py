from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.agent.models import (
    AgentEndpoint,
    AgentHistorySource,
    AgentRun,
    AgentRunStatus,
    AgentRunWorkItem,
    AgentTarget,
    AgentTask,
    ConnectorTypeSlug,
)
from fish_memory.agent.session_context import AgentSessionContextLoader
from fish_memory.answer.history_tokens import estimate_history_message_tokens
from fish_memory.models import Message, MessageStatus, Role

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 8, 31, tzinfo=UTC)
_FISH_ID = UUID(int=1)
_SESSION_ID = UUID(int=2)
_SOURCE_ID = UUID(int=3)


class FakeHistoryStore:
    def __init__(
        self,
        messages: list[Message] | None,
        root_messages: list[Message] | None = None,
    ) -> None:
        self.messages = messages
        self.root_messages = root_messages
        self.calls: list[tuple[UUID, UUID, UUID, int]] = []
        self.root_calls: list[tuple[UUID, UUID, UUID, int]] = []

    async def load_history_before_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        *,
        limit: int,
    ) -> list[Message] | None:
        self.calls.append((fish_id, session_id, message_id, limit))
        return self.messages

    async def load_history_before_task_root_source(
        self,
        fish_id: UUID,
        session_id: UUID,
        task_id: UUID,
        *,
        limit: int,
    ) -> list[Message] | None:
        self.root_calls.append((fish_id, session_id, task_id, limit))
        return self.root_messages


def _work() -> AgentRunWorkItem:
    endpoint = AgentEndpoint(
        id=UUID(int=4),
        connector_type=ConnectorTypeSlug("hermes"),
        base_url="https://agent.example.test",
        created_at=_NOW,
        updated_at=_NOW,
    )
    target = AgentTarget(
        id=UUID(int=5),
        endpoint_id=endpoint.id,
        external_key="default",
        route_prefix="",
        credential_ref="agent:test",
        connector_config=None,
        display_name="Hermes",
        enabled=True,
        created_at=_NOW,
        updated_at=_NOW,
    )
    task = AgentTask(
        id=UUID(int=6),
        fish_id=_FISH_ID,
        session_id=_SESSION_ID,
        target_id=target.id,
        external_session_id="task-session",
        title="Task",
        created_at=_NOW,
        updated_at=_NOW,
    )
    run = AgentRun(
        id=UUID(int=7),
        task_id=task.id,
        target_id=target.id,
        fish_id=_FISH_ID,
        session_id=_SESSION_ID,
        previous_run_id=None,
        source_message_id=_SOURCE_ID,
        confirmation_message_id=None,
        input="current request",
        external_run_id=None,
        history_source=AgentHistorySource.NONE,
        status=AgentRunStatus.QUEUED,
        output=None,
        result_message_id=None,
        error_code=None,
        error_message=None,
        provider_data=None,
        created_at=_NOW,
        started_at=None,
        updated_at=_NOW,
        finished_at=None,
    )
    return AgentRunWorkItem(run=run, task=task, target=target, endpoint=endpoint)


def _message(
    sequence: int,
    role: Role,
    content: str,
    *,
    status: MessageStatus = MessageStatus.COMPLETED,
    consumed_offset: int | None = -1,
) -> Message:
    return Message(
        id=UUID(int=100 + sequence),
        fish_id=_FISH_ID,
        session_id=_SESSION_ID,
        sequence=sequence,
        role=role,
        content=content,
        status=status,
        created_at=_NOW,
        consumed_offset=consumed_offset,
    )


async def test_projects_only_consumed_bounded_source_session_messages() -> None:
    store = FakeHistoryStore(
        [
            _message(1, Role.USER, "earlier request"),
            _message(2, Role.ASSISTANT, "seen-secret-tail", consumed_offset=4),
            _message(
                3,
                Role.ASSISTANT,
                "generating answer must not leave the session",
                status=MessageStatus.GENERATING,
                consumed_offset=None,
            ),
            *[_message(index, Role.USER, f"recent-{index}") for index in range(4, 17)],
        ]
    )

    history = await AgentSessionContextLoader(store).load_for_run(_work())

    assert history is not None
    assert store.calls == [(_FISH_ID, _SESSION_ID, _SOURCE_ID, 12)]
    assert len(history) == 12
    assert history[0].content == "recent-5"
    assert all("generating answer" not in item.content for item in history)
    assert all(len(item.content) <= 4_000 for item in history)
    assert sum(len(item.content) for item in history) <= 48_000
    assert (
        sum(estimate_history_message_tokens(item.content) for item in history) <= 8_000
    )


async def test_rejects_a_source_message_outside_the_run_session() -> None:
    assert (
        await AgentSessionContextLoader(FakeHistoryStore(None)).load_for_run(_work())
        is None
    )


async def test_source_session_token_budget_keeps_the_latest_complete_message() -> None:
    store = FakeHistoryStore(
        [
            _message(sequence, Role.USER, f"{sequence}:" + "界" * 4_000)
            for sequence in range(1, 13)
        ]
    )

    history = await AgentSessionContextLoader(store).load_for_run(_work())

    assert history is not None
    assert len(history) == 1
    assert history[0].content.startswith("12:")
    assert (
        sum(estimate_history_message_tokens(item.content) for item in history) <= 8_000
    )


async def test_continuation_uses_task_root_source_prefix_not_child_source_prefix() -> (
    None
):
    work = _work()
    continuation = AgentRunWorkItem(
        run=work.run.__class__(
            **{
                **{
                    field: getattr(work.run, field)
                    for field in work.run.__dataclass_fields__
                },
                "previous_run_id": UUID(int=8),
            }
        ),
        task=work.task,
        target=work.target,
        endpoint=work.endpoint,
    )
    store = FakeHistoryStore(
        [_message(1, Role.USER, "child-source-prefix-must-not-be-used")],
        [_message(2, Role.USER, "root-prefix")],
    )

    history = await AgentSessionContextLoader(store).load_for_run(continuation)

    assert history is not None
    assert [message.content for message in history] == ["root-prefix"]
    assert store.calls == []
    assert store.root_calls == [(_FISH_ID, _SESSION_ID, work.task.id, 12)]
