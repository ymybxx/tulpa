from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest
from fastapi import Request

from fish_memory.agent.message_delivery import (
    AgentMessageProcessor,
    ComposedAgentMessage,
)
from fish_memory.agent.models import (
    AgentInteractionWorkItem,
    AgentRunStatus,
    AgentRunWorkItem,
)
from fish_memory.agent.routes import stream_agent_messages
from fish_memory.background_messages.models import BackgroundMessageSource
from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import (
    AgentMessageSubjectConflictError,
    TaskStore,
)
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.sessions import SessionStore
from fish_memory.models import MessageStatus


async def stores(
    pool: asyncpg.Pool,
) -> tuple[MessagesStore, TargetStore, TaskStore]:
    await run_migrations(cast(MigrationPool, pool))
    messages = MessagesStore(pool)
    return messages, TargetStore(pool), TaskStore(pool, messages_store=messages)


async def target(targets: TargetStore):
    return await targets.upsert_target(
        "hermes",
        "http://localhost:9000",
        "default",
        "",
        "credential-ref",
        None,
        "Hermes",
    )


async def terminal_run(
    messages: MessagesStore,
    tasks: TaskStore,
    *,
    fish_id: UUID,
    target_id: UUID,
    label: str,
    session_id: UUID | None = None,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
):
    source = await messages.insert_user_message(
        fish_id,
        label,
        session_id=session_id,
    )
    task_row, run = await tasks.create_task_with_first_run(
        fish_id,
        target_id,
        label,
        source.id,
        None,
        label,
    )
    assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
    await tasks.record_dispatch_accepted(run.id, f"external-{run.id}", {})
    await tasks.finish_run(
        run.id,
        status,
        f"{label} output",
        "cancelled" if status is AgentRunStatus.CANCELLED else None,
        None,
        {},
    )
    return source, task_row, run


@dataclass
class OpenGate:
    async def is_active(self) -> bool:
        return False


class StaticComposer:
    async def compose_run(self, work: AgentRunWorkItem) -> ComposedAgentMessage:
        await asyncio.sleep(0)
        return ComposedAgentMessage("neutral", f"result:{work.task.title}", False)

    async def compose_interaction(
        self, work: AgentInteractionWorkItem
    ) -> ComposedAgentMessage:
        await asyncio.sleep(0)
        return ComposedAgentMessage("neutral", f"interaction:{work.task.title}", False)


@pytest.mark.asyncio
async def test_real_selectors_prioritize_interaction_and_failed_terminal(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await stores(postgres_pool)
    configured = await target(targets)
    fish_id = uuid4()
    _source, _task, completed = await terminal_run(
        messages,
        tasks,
        fish_id=fish_id,
        target_id=configured.target.id,
        label="completed",
    )
    _source, _task, failed = await terminal_run(
        messages,
        tasks,
        fish_id=fish_id,
        target_id=configured.target.id,
        label="failed",
        status=AgentRunStatus.FAILED,
    )
    input_source = await messages.insert_user_message(fish_id, "input")
    _task, input_run = await tasks.create_task_with_first_run(
        fish_id,
        configured.target.id,
        "input",
        input_source.id,
        None,
        "input",
    )
    assert await tasks.claim_prepared_run(input_run.id, "none", {}) is not None
    await tasks.record_dispatch_accepted(
        input_run.id,
        f"external-{input_run.id}",
        {},
    )
    _completed, interaction = await tasks.complete_run_requiring_input(
        input_run.id,
        "which destination?",
        {},
    )

    selected_interaction = await tasks.peek_unmaterialized_interaction()
    selected_terminal = await tasks.peek_unmaterialized_terminal_run()

    assert selected_interaction is not None
    assert selected_interaction.interaction.id == interaction.id
    assert selected_terminal is not None
    assert selected_terminal.run.id == failed.id
    assert selected_terminal.run.id != completed.id
    assert selected_terminal.run.id != input_run.id


@pytest.mark.asyncio
async def test_real_multiworker_materialize_cas_has_one_winner(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, first_store = await stores(postgres_pool)
    configured = await target(targets)
    fish_id = uuid4()
    _source, _task, run = await terminal_run(
        messages,
        first_store,
        fish_id=fish_id,
        target_id=configured.target.id,
        label="race",
    )
    second_store = TaskStore(postgres_pool, messages_store=messages)
    semaphore = asyncio.Semaphore(2)
    selected = asyncio.Barrier(2)

    class RacingComposer(StaticComposer):
        async def compose_run(self, work: AgentRunWorkItem) -> ComposedAgentMessage:
            # Both workers must select the same pending run before either CAS.
            await asyncio.wait_for(selected.wait(), timeout=5)
            return await super().compose_run(work)

    processors = [
        AgentMessageProcessor(
            store,
            RacingComposer(),
            background_semaphore=semaphore,
            foreground_gate=OpenGate(),
        )
        for store in (first_store, second_store)
    ]

    assert all(await asyncio.gather(*(item.process_once() for item in processors)))

    row = await postgres_pool.fetchrow(
        """
        SELECT run.result_message_id,
               count(message.id) FILTER (
                   WHERE message.content = 'result:race'
               ) AS matches
        FROM agent_runs AS run
        LEFT JOIN messages AS message ON message.fish_id = $2
        WHERE run.id = $1
        GROUP BY run.result_message_id
        """,
        run.id,
        fish_id,
    )
    assert row is not None
    assert row["result_message_id"] is not None
    assert row["matches"] == 1


@pytest.mark.asyncio
async def test_agent_consumption_boundary_rejects_cross_fish_and_normal_answer(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await stores(postgres_pool)
    configured = await target(targets)
    fish_id = uuid4()
    other_fish_id = uuid4()
    source, _task, run = await terminal_run(
        messages,
        tasks,
        fish_id=fish_id,
        target_id=configured.target.id,
        label="agent",
    )
    result = await tasks.materialize_run_result(
        run.id,
        "abcdefghij",
        emotion="happy",
    )
    assert result is not None

    generic = BackgroundMessageStore(postgres_pool, messages_store=messages)
    envelopes = await generic.list_messages_for_delivery(
        fish_id,
        source.session_id,
        after_sequence=source.sequence,
        include_incomplete=False,
        limit=10,
    )
    assert len(envelopes) == 1
    assert envelopes[0].source is BackgroundMessageSource.AGENT
    assert envelopes[0].emotion == "happy"
    assert envelopes[0].reply_context is not None
    assert envelopes[0].reply_context.kind.value == "run_result"

    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, other_fish_id, source.session_id, 4, True
        )
        is None
    )
    interrupted = await tasks.finalize_agent_message_consumption(
        result.id, fish_id, source.session_id, 4, True
    )
    assert interrupted is not None
    assert interrupted.content == "abcd"
    assert interrupted.status is MessageStatus.INTERRUPTED
    assert interrupted.consumed_offset == 4
    assert (
        await postgres_pool.fetchval(
            "SELECT fact_extraction_status FROM messages WHERE id = $1",
            result.id,
        )
        == "pending"
    )

    ordinary = await messages.insert_generating_answer(
        fish_id,
        source.session_id,
        source.sequence,
    )
    await messages.finish_answer(
        ordinary.id,
        fish_id,
        content="ordinary answer",
        requested_status=MessageStatus.COMPLETED,
    )
    with pytest.raises(AgentMessageSubjectConflictError):
        await tasks.finalize_agent_message_consumption(
            ordinary.id,
            fish_id,
            source.session_id,
            len("ordinary answer"),
            False,
        )


@pytest.mark.asyncio
async def test_delivery_dual_cursor_prioritizes_incomplete_and_keeps_full_content(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await stores(postgres_pool)
    configured = await target(targets)
    fish_id = uuid4()
    materialized = []
    for label in ("older-complete", "middle-incomplete", "newer-complete"):
        _source, _task, run = await terminal_run(
            messages,
            tasks,
            fish_id=fish_id,
            target_id=configured.target.id,
            label=label,
        )
        message = await tasks.materialize_run_result(run.id, label)
        assert message is not None
        materialized.append(message)
    for message in (materialized[0], materialized[1]):
        assert (
            await tasks.finalize_agent_message_consumption(
                message.id,
                fish_id,
                message.session_id,
                len(message.content),
                False,
            )
            is not None
        )

    first_page = await tasks.list_agent_messages_for_delivery(
        fish_id,
        materialized[0].session_id,
        after_sequence=0,
        include_incomplete=True,
        limit=1,
        incomplete_after_sequence=0,
    )
    assert [item.message_id for item in first_page] == [materialized[0].id]
    second_page = await tasks.list_agent_messages_for_delivery(
        fish_id,
        materialized[0].session_id,
        after_sequence=first_page[-1].sequence,
        include_incomplete=True,
        limit=2,
        incomplete_after_sequence=first_page[-1].sequence,
    )
    assert [item.message_id for item in second_page] == [
        materialized[1].id,
        materialized[2].id,
    ]
    assert second_page[-1].content == "newer-complete"
    assert second_page[-1].consumed_offset == 0

    past_cursor = await tasks.list_agent_messages_for_delivery(
        fish_id,
        materialized[0].session_id,
        after_sequence=materialized[2].sequence,
        include_incomplete=True,
        limit=1,
        incomplete_after_sequence=0,
    )
    assert [item.message_id for item in past_cursor] == [materialized[2].id]
    assert (
        await tasks.list_agent_messages_for_delivery(
            fish_id,
            materialized[0].session_id,
            after_sequence=materialized[2].sequence,
            include_incomplete=True,
            limit=1,
            incomplete_after_sequence=past_cursor[-1].sequence,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_delivery_pages_all_past_cursor_incomplete_before_new_domain(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await stores(postgres_pool)
    configured = await target(targets)
    fish_id = uuid4()
    materialized = []
    for label in ("past-one", "past-two", "cursor-complete"):
        _source, _task, run = await terminal_run(
            messages,
            tasks,
            fish_id=fish_id,
            target_id=configured.target.id,
            label=label,
        )
        message = await tasks.materialize_run_result(run.id, label)
        assert message is not None
        materialized.append(message)
    assert (
        await tasks.finalize_agent_message_consumption(
            materialized[2].id,
            fish_id,
            materialized[2].session_id,
            len(materialized[2].content),
            False,
        )
        is not None
    )

    first = await tasks.list_incomplete_agent_messages_for_delivery(
        fish_id,
        materialized[0].session_id,
        after_sequence=0,
        through_sequence=materialized[2].sequence,
        limit=1,
    )
    second = await tasks.list_incomplete_agent_messages_for_delivery(
        fish_id,
        materialized[0].session_id,
        after_sequence=first[-1].sequence,
        through_sequence=materialized[2].sequence,
        limit=1,
    )
    exhausted = await tasks.list_incomplete_agent_messages_for_delivery(
        fish_id,
        materialized[0].session_id,
        after_sequence=second[-1].sequence,
        through_sequence=materialized[2].sequence,
        limit=1,
    )

    assert [item.message_id for item in (*first, *second)] == [
        materialized[0].id,
        materialized[1].id,
    ]
    assert exhausted == ()


@pytest.mark.asyncio
async def test_approval_partial_progress_keeps_full_content_recoverable(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await stores(postgres_pool)
    configured = await target(targets)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "deploy")
    _task, run = await tasks.create_task_with_first_run(
        fish_id,
        configured.target.id,
        "approval",
        source.id,
        None,
        "deploy",
    )
    assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
    await tasks.record_dispatch_accepted(run.id, f"external-{run.id}", {})
    interaction = await tasks.open_approval_interaction(
        run.id,
        {"command": "deploy", "allowed_decisions": ["once", "deny"]},
    )
    request = await tasks.materialize_interaction_request(
        interaction.id,
        "Approve deploy?",
    )
    assert request is not None

    partial = await tasks.finalize_agent_message_consumption(
        request.id,
        fish_id,
        request.session_id,
        4,
        False,
    )
    assert partial is not None
    assert partial.content == "Approve deploy?"
    assert partial.status is MessageStatus.COMPLETED
    assert partial.consumed_offset == 4
    assert (
        await postgres_pool.fetchval(
            "SELECT fact_extraction_status FROM messages WHERE id = $1",
            request.id,
        )
        == "waiting_consumption"
    )

    replay = await tasks.list_agent_messages_for_delivery(
        fish_id,
        request.session_id,
        after_sequence=request.sequence,
        include_incomplete=True,
        limit=1,
        incomplete_after_sequence=0,
    )
    assert [item.message_id for item in replay] == [request.id]
    assert replay[0].content == "Approve deploy?"
    assert replay[0].consumed_offset == 4


@pytest.mark.asyncio
async def test_live_route_does_not_emit_another_session(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await stores(postgres_pool)
    configured = await target(targets)
    fish_id = uuid4()
    old_source, _task, old_run = await terminal_run(
        messages,
        tasks,
        fish_id=fish_id,
        target_id=configured.target.id,
        label="old-session-result",
    )
    new_session = await SessionStore(postgres_pool).create(fish_id)
    new_source, _task, new_run = await terminal_run(
        messages,
        tasks,
        fish_id=fish_id,
        target_id=configured.target.id,
        label="new-session-result",
        session_id=new_session.id,
    )
    old_message = await tasks.materialize_run_result(old_run.id, "old session")
    new_message = await tasks.materialize_run_result(new_run.id, "new session")
    assert old_message is not None and old_message.session_id == old_source.session_id
    assert new_message is not None and new_message.session_id == new_source.session_id

    class LiveRequest:
        def __init__(self) -> None:
            self.app = SimpleNamespace(
                state=SimpleNamespace(
                    agent_task_store=tasks,
                    session_store=SessionStore(postgres_pool),
                )
            )
            self.headers: dict[str, str] = {}

        async def is_disconnected(self) -> bool:
            return False

    response = await stream_agent_messages(
        fish_id,
        cast(Request, LiveRequest()),
        session_id=new_session.id,
        after_sequence=0,
    )
    body = cast(
        AsyncGenerator[str, None],
        response.body_iterator,
    )
    frame = await asyncio.wait_for(anext(body), timeout=1.0)
    await body.aclose()

    assert frame.startswith(f"id: {new_message.sequence}\n")
    assert "new session" in frame
    assert "old session" not in frame
