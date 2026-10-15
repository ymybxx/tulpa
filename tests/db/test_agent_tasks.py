from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import asyncpg
import pytest

from fish_memory.agent.models import AgentMessageSubjectKind
from fish_memory.agent.replies import (
    AgentReplyConflictError,
    AgentReplyContext,
)
from fish_memory.db import agent_tasks as tasks_module


class RecordingPool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, query: str, *args: object) -> None:
        self.calls.append((" ".join(query.split()), args))
        return None

    async def fetch(self, query: str, *args: object) -> list[object]:
        self.calls.append((" ".join(query.split()), args))
        return []

    def acquire(self) -> RecordingPool:
        return self

    def transaction(self) -> RecordingPool:
        return self

    async def __aenter__(self) -> RecordingPool:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def store_with(pool: RecordingPool) -> Any:
    return tasks_module.TaskStore(cast(asyncpg.Pool, pool))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [("title", "t" * 201), ("input", "i" * 32_001)],
)
async def test_create_task_validates_materialized_text_before_database_write(
    field: str,
    value: str,
) -> None:
    pool = RecordingPool()
    arguments = {
        "fish_id": uuid4(),
        "target_id": uuid4(),
        "title": "title",
        "source_message_id": uuid4(),
        "confirmation_message_id": None,
        "input": "input",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        await store_with(pool).create_task_with_first_run(**arguments)

    assert pool.calls == []


@pytest.mark.asyncio
async def test_state_update_allows_oversized_incoming_metadata_to_degrade() -> None:
    pool = RecordingPool()

    assert (
        await store_with(pool).update_active_run(
            uuid4(), "running", {"payload": "界" * 22_000}
        )
        is None
    )

    assert len(pool.calls) == 1


@pytest.mark.asyncio
async def test_dispatch_acceptance_rejects_oversized_external_id_before_write() -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="external_run_id"):
        await store_with(pool).record_dispatch_accepted(uuid4(), "x" * 513, {})

    assert pool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["prompt", "response", "options"])
async def test_interaction_rejects_duplicated_visible_text_before_write(
    field: str,
) -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="must not duplicate"):
        await store_with(pool).open_approval_interaction(
            uuid4(), {field: "visible text belongs in Message"}
        )

    assert pool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_key",
    [
        "secret",
        "client_secret",
        "clientSecret",
        "access_token",
        "accessToken",
        "refresh_token",
        "refreshToken",
        "secret_key",
        "secretKey",
        "api_key",
        "apiKey",
        "api-key",
        "x-api-key",
        "authorization",
        "bearer",
    ],
)
async def test_provider_data_rejects_normalized_credential_keys_before_write(
    secret_key: str,
) -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="credential-like"):
        await store_with(pool).update_active_run(
            uuid4(), "running", {"nested": {secret_key: "must not persist"}}
        )

    assert pool.calls == []


@pytest.mark.asyncio
async def test_provider_data_allows_usage_token_counter_keys() -> None:
    pool = RecordingPool()

    assert (
        await store_with(pool).update_active_run(
            uuid4(),
            "running",
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "reasoning_tokens": 30,
            },
        )
        is None
    )
    assert len(pool.calls) == 1


@pytest.mark.asyncio
async def test_candidate_query_is_bounded_scoped_and_uses_latest_lineage_tail() -> None:
    pool = RecordingPool()
    fish_id = uuid4()
    session_id = uuid4()

    candidates = await store_with(pool).list_task_candidates(
        fish_id,
        session_id,
        recent_terminal_limit=2,
        limit=7,
        supported_connector_types=frozenset({"hermes", "fake.valid"}),
    )

    assert candidates == ()
    assert len(pool.calls) == 1
    query, arguments = pool.calls[0]
    assert arguments == (fish_id, session_id, 2, 7, ["fake.valid", "hermes"])
    assert "NOT EXISTS ( SELECT 1 FROM agent_runs AS child" in query
    assert "WITH recent_assistant AS MATERIALIZED" in query
    assert "session_interactions AS" in query
    assert "recent_task_replies AS" in query
    assert "request.session_id = $2" in query
    assert "response_message_id IS NULL" in query
    assert "created_at >= now() - interval '1 hour'" in query
    assert "OR session_rank = 1" in query
    assert "interaction.focus_rank = 1" in query
    assert "interaction.request_content AS interaction_request_content" in query
    assert "source.session_id = $2" in query
    assert "task.fish_id = $1" in query
    assert "AND target.enabled" in query
    assert "resolved_at IS NULL" in query
    assert "closed_at IS NULL" in query
    assert "result.consumed_offset = -1" in query
    assert "result.consumed_offset >= char_length(result.content)" in query
    assert "result.status IN ('completed', 'interrupted')" in query
    assert "result.fact_extraction_status <> 'waiting_consumption'" in query
    assert "recent_reply.task_id IS NOT NULL" in query
    connector_filter = "endpoint.connector_type = ANY($5::text[])"
    assert connector_filter in query
    assert query.index(connector_filter) < query.index("LIMIT $4")
    assert "recent_reply_sequence DESC NULLS LAST" in query
    assert "LIMIT $4" in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connector_types",
    ["hermes", frozenset({"Bad Slug"})],
    ids=["plain-string", "invalid-slug"],
)
async def test_candidate_query_rejects_invalid_connector_filter_before_database(
    connector_types: object,
) -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="connector"):
        await store_with(pool).list_task_candidates(
            uuid4(),
            uuid4(),
            supported_connector_types=connector_types,  # type: ignore[arg-type]
        )

    assert pool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "required_sql"),
    [
        (
            AgentMessageSubjectKind.INTERACTION,
            (
                "LEFT(request.content, $5)",
                "task.fish_id = $2",
                "request.session_id = $3",
            ),
        ),
        (
            AgentMessageSubjectKind.RUN_RESULT,
            (
                "LEFT(result.content, $5)",
                "task.fish_id = $2",
                "result.session_id = $3",
                "child.previous_run_id = run.id",
            ),
        ),
    ],
)
async def test_reply_subject_read_is_exact_scoped_and_bounded(
    kind: AgentMessageSubjectKind,
    required_sql: tuple[str, ...],
) -> None:
    pool = RecordingPool()
    context = AgentReplyContext(kind, uuid4())
    fish_id = uuid4()
    session_id = uuid4()

    with pytest.raises(AgentReplyConflictError, match="replyable"):
        await store_with(pool).load_reply_subject(context, fish_id, session_id)

    assert len(pool.calls) == 1
    query, arguments = pool.calls[0]
    assert arguments == (context.id, fish_id, session_id, None, 32_000)
    for clause in required_sql:
        assert clause in query
