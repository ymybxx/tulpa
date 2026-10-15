from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import SecretStr

from fish_memory.agent.connector import AgentConnectorRuntime
from fish_memory.agent.credentials import AgentCredentialResolver
from fish_memory.agent.hermes import HermesConnectorFactory
from fish_memory.agent.message_delivery import (
    AgentMessageProcessor,
    ComposedAgentMessage,
)
from fish_memory.agent.models import (
    AgentInteractionWorkItem,
    AgentMessageSubjectKind,
    AgentResolvedTarget,
    AgentRun,
    AgentRunStatus,
    AgentRunWorkItem,
)
from fish_memory.agent.processor import AgentRunProcessor
from fish_memory.agent.replies import (
    AgentReplyContext,
    AgentReplyService,
    InputReplyAction,
    InputReplyClassification,
    RunResultReplyAction,
    RunResultReplyClassification,
)
from fish_memory.agent.routes import stream_agent_messages
from fish_memory.agent.routing import (
    AgentTaskDecisionKind,
    AgentTaskRouteService,
    TaskLaneContext,
)
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import TaskStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.db.sessions import SessionStore
from fish_memory.models import Message, MessageStatus
from tests.agent.fake_hermes import FakeHermes

pytestmark = pytest.mark.asyncio

_FISH_ID = UUID("00000000-0000-0000-0000-000000000801")
_VOICE_REPLY_ID = UUID("00000000-0000-0000-0000-000000000802")


@dataclass(frozen=True)
class OpenForegroundGate:
    async def is_active(self) -> bool:
        return False


class FlowMessageComposer:
    async def compose_run(self, work: AgentRunWorkItem) -> ComposedAgentMessage:
        return ComposedAgentMessage(
            "neutral",
            f"result:{work.run.output}",
            False,
        )

    async def compose_interaction(
        self,
        work: AgentInteractionWorkItem,
    ) -> ComposedAgentMessage:
        request = work.run.output or "approval"
        return ComposedAgentMessage("neutral", f"question:{request}", False)


class FlowReplyClassifier:
    async def classify_input(
        self,
        *,
        request: str,
        content: str,
    ) -> InputReplyClassification:
        assert request.startswith("question:")
        assert content == "Use staging"
        return InputReplyClassification(InputReplyAction.ANSWER)

    async def classify_run_result(
        self,
        *,
        result: str,
        content: str,
    ) -> RunResultReplyClassification:
        del result, content
        return RunResultReplyClassification(RunResultReplyAction.UNRELATED)


class DirectSseRequest:
    def __init__(self, tasks: TaskStore, sessions: SessionStore) -> None:
        self.app = SimpleNamespace(
            state=SimpleNamespace(agent_task_store=tasks, session_store=sessions)
        )
        self.headers: dict[str, str] = {}

    async def is_disconnected(self) -> bool:
        return False


async def _route_new_task(
    messages: MessagesStore,
    route: AgentTaskRouteService,
    target: AgentResolvedTarget,
    content: str,
    *,
    session_id: UUID | None = None,
) -> tuple[Message, AgentRun]:
    source = await messages.insert_user_message(
        _FISH_ID,
        content,
        session_id=session_id,
    )
    confirmation = await messages.insert_generating_answer(
        _FISH_ID,
        source.session_id,
        source.sequence,
    )
    run = await route.commit_selected_route(
        fish_id=_FISH_ID,
        resolved_target=target,
        task_context=TaskLaneContext.empty(),
        kind=AgentTaskDecisionKind.NEW_TASK,
        candidate_indexes=(),
        source_message_id=source.id,
        confirmation_message_id=confirmation.id,
        content=content,
    )
    await messages.finish_answer(
        confirmation.id,
        _FISH_ID,
        content="Task accepted.",
        requested_status=MessageStatus.COMPLETED,
    )
    return source, run


async def _wait_for_run_status(
    pool: asyncpg.Pool,
    run_id: UUID,
    status: AgentRunStatus,
) -> asyncpg.Record:
    for _ in range(200):
        row = await pool.fetchrow(
            """
            SELECT status, output, history_source, result_message_id, task_id
            FROM agent_runs
            WHERE id = $1
            """,
            run_id,
        )
        if row is not None and row["status"] == status.value:
            return row
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach {status.value}")


async def _wait_for_interaction(
    pool: asyncpg.Pool,
    run_id: UUID,
) -> asyncpg.Record:
    for _ in range(200):
        row = await pool.fetchrow(
            """
            SELECT id, request_message_id, response_message_id,
                   resolved_at, next_run_id
            FROM agent_interactions
            WHERE run_id = $1
              AND closed_at IS NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            run_id,
        )
        if row is not None:
            return row
        await asyncio.sleep(0.01)
    raise AssertionError(f"run {run_id} did not open an interaction")


async def _message(pool: asyncpg.Pool, message_id: UUID) -> asyncpg.Record:
    row = await pool.fetchrow(
        "SELECT id, session_id, content FROM messages WHERE id = $1",
        message_id,
    )
    if row is None:
        raise AssertionError(f"message {message_id} was not found")
    return row


def _body(response: StreamingResponse) -> AsyncGenerator[str, None]:
    return cast(AsyncGenerator[str, None], response.body_iterator)


async def test_real_hermes_flow_is_durable_replyable_and_exactly_once(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool)
    targets = TargetStore(postgres_pool)
    tasks = TaskStore(postgres_pool, messages_store=messages)
    target = await targets.upsert_target(
        "hermes",
        "http://localhost:8642",
        "default",
        "",
        "hermes:test",
        {},
        "Hermes test",
    )
    fake = FakeHermes()
    fake.enqueue_completed("Authoritative first result", sse_output="stale")
    fake.enqueue_approval(
        command="deploy staging",
        completed_message="Deployment approved",
    )
    fake.enqueue_input_required("Which environment?")
    fake.enqueue_completed("Staging configured")
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    connector = HermesConnectorFactory().create(
        AgentConnectorRuntime(
            credential_resolver=AgentCredentialResolver(
                {"hermes:test": SecretStr("provider-key")}
            ),
            http_client=client,
            request_timeout_seconds=1,
            sse_read_timeout_seconds=1,
        )
    )
    route = AgentTaskRouteService(tasks)
    message_processor = AgentMessageProcessor(
        tasks,
        FlowMessageComposer(),
        background_semaphore=asyncio.Semaphore(1),
        foreground_gate=OpenForegroundGate(),
    )
    reply_service = AgentReplyService(tasks, FlowReplyClassifier())
    run_processor = AgentRunProcessor(
        tasks,
        {"hermes": connector},
        concurrency=4,
        poll_interval_seconds=0.01,
    )
    try:
        first_source, first_run = await _route_new_task(
            messages,
            route,
            target,
            "complete first task",
        )
        assert await run_processor.process_once() is True
        first = await _wait_for_run_status(
            postgres_pool,
            first_run.id,
            AgentRunStatus.COMPLETED,
        )
        assert first["output"] == "Authoritative first result"
        assert len(fake.run_requests) == 1

        assert await message_processor.process_once() is True
        first_message_id = await postgres_pool.fetchval(
            "SELECT result_message_id FROM agent_runs WHERE id = $1",
            first_run.id,
        )
        assert isinstance(first_message_id, UUID)
        envelopes = await tasks.list_agent_messages_for_delivery(
            _FISH_ID,
            first_source.session_id,
            after_sequence=0,
            include_incomplete=False,
            limit=10,
        )
        first_envelope = next(
            item for item in envelopes if item.message_id == first_message_id
        )
        assert first_envelope.subject.kind is AgentMessageSubjectKind.RUN_RESULT
        assert first_envelope.subject.id == first_run.id
        response = await stream_agent_messages(
            _FISH_ID,
            cast(Request, DirectSseRequest(tasks, SessionStore(postgres_pool))),
            session_id=first_source.session_id,
        )
        stream = _body(response)
        frame = await anext(stream)
        await stream.aclose()
        assert "event: agent.message" in frame
        assert str(first_run.id) in frame
        assert await message_processor.process_once() is False
        assert (
            await postgres_pool.fetchval(
                "SELECT count(*) FROM messages WHERE id = $1",
                first_message_id,
            )
            == 1
        )

        await run_processor.close()
        restarted = AgentRunProcessor(
            tasks,
            {"hermes": connector},
            concurrency=4,
            poll_interval_seconds=0.01,
        )
        assert await restarted.process_once() is False
        await restarted.close()
        assert len(fake.run_requests) == 1
        run_processor = AgentRunProcessor(
            tasks,
            {"hermes": connector},
            concurrency=4,
            poll_interval_seconds=0.01,
        )

        approval_source, approval_run = await _route_new_task(
            messages,
            route,
            target,
            "approval task",
            session_id=first_source.session_id,
        )
        assert await run_processor.process_once() is True
        approval = await _wait_for_interaction(postgres_pool, approval_run.id)
        assert await message_processor.process_once() is True
        approval = await _wait_for_interaction(postgres_pool, approval_run.id)
        approval_message_id = approval["request_message_id"]
        assert isinstance(approval_message_id, UUID)
        approval_message = await _message(postgres_pool, approval_message_id)
        await tasks.finalize_agent_message_consumption(
            approval_message_id,
            _FISH_ID,
            approval_source.session_id,
            len(approval_message["content"]),
            False,
        )
        approval_reply = await reply_service.handle(
            _FISH_ID,
            approval_source.session_id,
            "允许",
            AgentReplyContext(
                AgentMessageSubjectKind.INTERACTION,
                approval["id"],
            ),
            message_id=_VOICE_REPLY_ID,
        )
        assert approval_reply.answer_expected is False
        assert await run_processor.process_once() is True
        await _wait_for_run_status(
            postgres_pool,
            approval_run.id,
            AgentRunStatus.COMPLETED,
        )
        assert fake.approval_requests == [
            {"run_id": fake.run_ids[1], "choice": "once", "resolve_all": False}
        ]
        approval = await _wait_for_interaction(postgres_pool, approval_run.id)
        assert approval["resolved_at"] is not None
        assert (
            await postgres_pool.fetchval(
                "SELECT count(*) FROM agent_runs WHERE task_id = $1",
                approval_run.task_id,
            )
            == 1
        )
        assert await message_processor.process_once() is True

        input_source, input_run = await _route_new_task(
            messages,
            route,
            target,
            "needs input task",
            session_id=first_source.session_id,
        )
        assert await run_processor.process_once() is True
        await _wait_for_run_status(
            postgres_pool,
            input_run.id,
            AgentRunStatus.COMPLETED,
        )
        input_interaction = await _wait_for_interaction(postgres_pool, input_run.id)
        assert await message_processor.process_once() is True
        input_interaction = await _wait_for_interaction(postgres_pool, input_run.id)
        input_message_id = input_interaction["request_message_id"]
        assert isinstance(input_message_id, UUID)
        input_message = await _message(postgres_pool, input_message_id)
        await tasks.finalize_agent_message_consumption(
            input_message_id,
            _FISH_ID,
            input_source.session_id,
            len(input_message["content"]),
            False,
        )
        input_reply = await reply_service.handle(
            _FISH_ID,
            input_source.session_id,
            "Use staging",
            AgentReplyContext(
                AgentMessageSubjectKind.INTERACTION,
                input_interaction["id"],
            ),
            message_id=uuid4(),
        )
        assert input_reply.answer_expected is False
        child = await postgres_pool.fetchrow(
            """
            SELECT id, task_id, previous_run_id, source_message_id, history_source
            FROM agent_runs
            WHERE previous_run_id = $1
            """,
            input_run.id,
        )
        assert child is not None
        assert child["task_id"] == input_run.task_id
        assert child["source_message_id"] == input_reply.message.id

        assert await run_processor.process_once() is True
        child_result = await _wait_for_run_status(
            postgres_pool,
            child["id"],
            AgentRunStatus.COMPLETED,
        )
        assert child_result["history_source"] == "local_projection"
        continuation = fake.run_requests[-1]
        history = continuation["conversation_history"]
        assert isinstance(history, list) and history
        history_content: list[str] = []
        for item in history:
            assert isinstance(item, dict)
            item_content = item.get("content")
            assert isinstance(item_content, str)
            history_content.append(item_content)
        assert "needs input task" in history_content
        assert "question:Which environment?" in history_content
        assert continuation["input"] == "Use staging"
        assert fake.history_requests
        assert len(fake.run_requests) == 4
        assert await message_processor.process_once() is True
    finally:
        await run_processor.close()
        await message_processor.close()
        await connector.close()
        await client.aclose()
