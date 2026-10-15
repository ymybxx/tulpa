import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.agent.models import (
    AgentInteractionKind,
    AgentMessageSubjectKind,
    AgentRunStatus,
    AgentTaskCandidate,
)
from fish_memory.agent.replies import (
    AgentReplyConflictError,
    AgentReplyContext,
    AgentReplyService,
    InputReplyAction,
    InputReplyClassification,
)
from fish_memory.agent.routing import (
    AgentTaskDecisionKind,
    AgentTaskRouteConflictError,
    AgentTaskRouteService,
    LocalConnectorCapabilityPolicy,
    TaskLaneContext,
)
from fish_memory.db.agent_targets import TargetStore
from fish_memory.db.agent_tasks import (
    AgentInteractionConflictError,
    AgentRunConflictError,
    TaskStore,
)
from fish_memory.db.facts import FactStore
from fish_memory.db.messages import MessageIdentityConflictError, MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.models import MessageStatus


def canonical_json_bytes(value: object) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def mapping_field(
    value: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    child = value[key]
    assert isinstance(child, Mapping)
    return cast(Mapping[str, object], child)


async def migrated_stores(
    pool: asyncpg.Pool,
) -> tuple[MessagesStore, TargetStore, TaskStore]:
    await run_migrations(cast(MigrationPool, pool))
    messages = MessagesStore(pool)
    return messages, TargetStore(pool), TaskStore(pool, messages_store=messages)


async def configured_target(
    targets: TargetStore,
    *,
    connector_type: str = "hermes",
    external_key: str = "default",
    display_name: str = "Hermes",
):
    return await targets.upsert_target(
        connector_type,
        "http://localhost:9000",
        external_key,
        "" if external_key == "default" else f"/p/{external_key}",
        "credential-ref" if connector_type == "hermes" else None,
        None,
        display_name,
    )


async def start_run(tasks: TaskStore, run_id) -> None:
    assert await tasks.claim_prepared_run(run_id, "none", {}) is not None
    await tasks.record_dispatch_accepted(run_id, f"external-{run_id}", {})


def frozen_task_context(task, run, target, content: str) -> TaskLaneContext:
    return TaskLaneContext.from_records(
        (
            AgentTaskCandidate(
                task=task,
                run=run,
                interaction=None,
                target=target.target,
                endpoint=target.endpoint,
            ),
        ),
        current_content=content,
        capability_policy=LocalConnectorCapabilityPolicy(),
    )


@pytest.mark.asyncio
async def test_targets_normalize_config_and_resolve_default_override_and_disabled(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, _tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    await messages.insert_user_message(fish_id, "hello")
    default = await configured_target(targets)
    override = await configured_target(
        targets,
        connector_type="fake.valid",
        external_key="coder",
        display_name="Coder",
    )

    assert default.target.connector_config == {}
    assert {
        str(item.endpoint.connector_type) for item in await targets.list_targets()
    } == {
        "hermes",
        "fake.valid",
    }
    assert (await targets.resolve_for_fish(fish_id, default.target.id)) == default
    assert await targets.get_fish_target(fish_id) is None
    assert await targets.set_fish_target(fish_id, override.target.id) is True
    assert await targets.get_fish_target(fish_id) == override
    assert (await targets.resolve_for_fish(fish_id, default.target.id)) == override
    assert await targets.set_fish_target(fish_id, None) is True
    assert await targets.get_fish_target(fish_id) is None
    assert (await targets.resolve_for_fish(fish_id, default.target.id)) == default

    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = false WHERE id = $1", override.target.id
    )
    assert await targets.set_fish_target(fish_id, override.target.id) is False
    await postgres_pool.execute(
        "UPDATE fishes SET agent_target_id = $2 WHERE id = $1",
        fish_id,
        override.target.id,
    )
    assert await targets.resolve_for_fish(fish_id, default.target.id) is None


@pytest.mark.asyncio
async def test_task_target_is_frozen_and_first_run_is_atomic(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "do it")
    confirmation = await messages.insert_generating_answer(
        fish_id, source.session_id, source.sequence
    )
    first_target = await configured_target(targets)
    second_target = await configured_target(
        targets, connector_type="fake.valid", external_key="other"
    )

    task, run = await tasks.create_task_with_first_run(
        fish_id,
        first_target.target.id,
        "A durable task",
        source.id,
        confirmation.id,
        source.content,
    )
    await targets.set_fish_target(fish_id, second_target.target.id)

    assert task.target_id == first_target.target.id
    assert task.external_session_id == f"fm-task-{task.id}"
    assert run.task_id == task.id
    assert run.target_id == first_target.target.id
    assert run.status is AgentRunStatus.QUEUED
    assert run.source_message_id == source.id
    assert run.confirmation_message_id == confirmation.id
    counts = await postgres_pool.fetchrow(
        "SELECT (SELECT count(*) FROM agent_tasks) AS tasks, "
        "(SELECT count(*) FROM agent_runs) AS runs"
    )
    assert counts is not None
    assert dict(counts) == {"tasks": 1, "runs": 1}


@pytest.mark.asyncio
async def test_task_lineage_rejects_messages_from_another_local_session(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "start")
    target = await configured_target(targets)
    task, first = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    assert task.session_id == source.session_id
    assert first.session_id == source.session_id
    assert await tasks.fail_queued_run(first.id, "failed", "failed") is not None

    other_session_id = uuid4()
    await postgres_pool.execute(
        "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2)",
        other_session_id,
        fish_id,
    )
    other_source = await messages.insert_user_message(
        fish_id,
        "continue elsewhere",
        session_id=other_session_id,
    )

    with pytest.raises(AgentRunConflictError, match="Agent task Session"):
        await tasks.create_continuation_run(
            task.id,
            first.id,
            other_source.id,
            other_source.content,
            None,
        )


@pytest.mark.asyncio
async def test_task_candidates_use_latest_tail_scope_consumption_and_stable_priority(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    other_fish_id = uuid4()
    seed = await messages.insert_user_message(fish_id, "session seed")
    session_id = seed.session_id
    other_session_id = uuid4()
    await postgres_pool.execute(
        "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2)",
        other_session_id,
        fish_id,
    )
    target = await configured_target(targets)

    async def source(content: str, *, fish=fish_id, session=session_id):
        return await messages.insert_user_message(fish, content, session_id=session)

    active_source = await source("active first")
    active_task, active_first = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "active latest tail",
        active_source.id,
        None,
        active_source.content,
    )
    await start_run(tasks, active_first.id)
    await tasks.finish_run(active_first.id, "completed", "first done", None, None, {})
    active_next_source = await source("active next")
    active_tail = await tasks.create_continuation_run(
        active_task.id,
        active_first.id,
        active_next_source.id,
        active_next_source.content,
        None,
    )

    input_source = await source("needs input")
    input_task, input_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "open interaction",
        input_source.id,
        None,
        input_source.content,
    )
    await start_run(tasks, input_run.id)
    _completed, open_interaction = await tasks.complete_run_requiring_input(
        input_run.id,
        "need an answer",
        {},
    )
    assert (
        await tasks.materialize_interaction_request(
            open_interaction.id,
            "need an answer",
        )
        is not None
    )

    consumed_source = await source("consumed terminal")
    consumed_task, consumed_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "consumed terminal",
        consumed_source.id,
        None,
        consumed_source.content,
    )
    await start_run(tasks, consumed_run.id)
    await tasks.finish_run(consumed_run.id, "completed", "done", None, None, {})
    consumed_result = await tasks.materialize_run_result(
        consumed_run.id, "fully consumed result"
    )
    assert consumed_result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            consumed_result.id,
            fish_id,
            consumed_result.session_id,
            len(consumed_result.content),
            False,
        )
        is not None
    )

    older_source = await source("older consumed terminal")
    older_task, older_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "older consumed terminal",
        older_source.id,
        None,
        older_source.content,
    )
    await start_run(tasks, older_run.id)
    await tasks.finish_run(older_run.id, "completed", "done", None, None, {})
    older_result = await tasks.materialize_run_result(older_run.id, "older result")
    assert older_result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            older_result.id,
            fish_id,
            older_result.session_id,
            -1,
            False,
        )
        is not None
    )

    unread_source = await source("unread terminal")
    unread_task, unread_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "unread terminal",
        unread_source.id,
        None,
        unread_source.content,
    )
    await start_run(tasks, unread_run.id)
    await tasks.finish_run(unread_run.id, "completed", "done", None, None, {})
    assert await tasks.materialize_run_result(unread_run.id, "not consumed") is not None

    offset_only_source = await source("offset only terminal")
    offset_only_task, offset_only_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "offset only terminal",
        offset_only_source.id,
        None,
        offset_only_source.content,
    )
    await start_run(tasks, offset_only_run.id)
    await tasks.finish_run(offset_only_run.id, "completed", "done", None, None, {})
    offset_only_result = await tasks.materialize_run_result(
        offset_only_run.id, "offset advanced without barrier"
    )
    assert offset_only_result is not None
    await postgres_pool.execute(
        "UPDATE messages SET consumed_offset = char_length(content) WHERE id = $1",
        offset_only_result.id,
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT fact_extraction_status FROM messages WHERE id = $1",
            offset_only_result.id,
        )
        == "waiting_consumption"
    )

    no_result_source = await source("no result terminal")
    no_result_task, no_result_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "no result terminal",
        no_result_source.id,
        None,
        no_result_source.content,
    )
    await start_run(tasks, no_result_run.id)
    await tasks.finish_run(no_result_run.id, "failed", None, "failed", "failed", {})

    other_session_source = await source("other session", session=other_session_id)
    other_session_task, _ = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "other session",
        other_session_source.id,
        None,
        other_session_source.content,
    )
    other_fish_source = await source("other fish", fish=other_fish_id, session=None)
    other_fish_task, _ = await tasks.create_task_with_first_run(
        other_fish_id,
        target.target.id,
        "other fish",
        other_fish_source.id,
        None,
        other_fish_source.content,
    )

    base = datetime(2026, 8, 30, 12, tzinfo=UTC)
    for run_id, updated_at in (
        (active_tail.id, base + timedelta(seconds=1)),
        (input_run.id, base + timedelta(seconds=2)),
        (consumed_run.id, base + timedelta(seconds=5)),
        (older_run.id, base + timedelta(seconds=4)),
        (offset_only_run.id, base + timedelta(seconds=6)),
    ):
        await postgres_pool.execute(
            "UPDATE agent_runs SET updated_at = $2 WHERE id = $1",
            run_id,
            updated_at,
        )

    first = await tasks.list_task_candidates(
        fish_id,
        session_id,
        recent_terminal_limit=5,
        limit=3,
    )
    second = await tasks.list_task_candidates(
        fish_id,
        session_id,
        recent_terminal_limit=5,
        limit=3,
    )

    assert [item.task.id for item in first] == [
        input_task.id,
        active_task.id,
        older_task.id,
    ]
    assert [item.run.id for item in first] == [
        input_run.id,
        active_tail.id,
        older_run.id,
    ]
    assert first[0].interaction is not None
    assert first[0].interaction.id == open_interaction.id
    assert [item.task.id for item in second] == [item.task.id for item in first]
    excluded = {
        consumed_task.id,
        unread_task.id,
        offset_only_task.id,
        no_result_task.id,
        other_session_task.id,
        other_fish_task.id,
    }
    assert not excluded.intersection(item.task.id for item in first)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_tail_state", ["terminal", "active", "open", "unread"])
async def test_frozen_t_route_never_advances_a_newer_tail(
    postgres_pool: asyncpg.Pool,
    new_tail_state: str,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, first = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    failed_first = await tasks.fail_queued_run(first.id, "failed", "first")
    assert failed_first is not None
    context = frozen_task_context(task, failed_first, target, "继续这个任务")

    second_source = await messages.insert_user_message(
        fish_id, "second", session_id=source.session_id
    )
    second_id = uuid4()
    await postgres_pool.execute(
        """
        INSERT INTO agent_runs (
            id, task_id, target_id, fish_id, session_id, previous_run_id,
            source_message_id,
            input, status
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, 'second', 'queued')
        """,
        second_id,
        task.id,
        target.target.id,
        fish_id,
        source.session_id,
        first.id,
        second_source.id,
    )
    if new_tail_state == "terminal":
        assert await tasks.fail_queued_run(second_id, "failed", "second") is not None
    elif new_tail_state == "open":
        await start_run(tasks, second_id)
        await tasks.complete_run_requiring_input(second_id, "need input", {})
    elif new_tail_state == "unread":
        await start_run(tasks, second_id)
        await tasks.finish_run(second_id, "completed", "done", None, None, {})
        assert (
            await tasks.materialize_run_result(second_id, "unread result") is not None
        )

    current_source = await messages.insert_user_message(
        fish_id, "继续这个任务", session_id=source.session_id
    )
    confirmation = await messages.insert_generating_answer(
        fish_id, source.session_id, current_source.sequence
    )

    with pytest.raises((AgentRunConflictError, AgentTaskRouteConflictError)):
        await AgentTaskRouteService(tasks).commit_selected_route(
            fish_id=fish_id,
            resolved_target=target,
            task_context=context,
            kind=AgentTaskDecisionKind.CONTINUE_TASK,
            candidate_indexes=(0,),
            source_message_id=current_source.id,
            confirmation_message_id=confirmation.id,
            content=current_source.content,
        )

    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
        )
        == 2
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE confirmation_message_id = $1",
            confirmation.id,
        )
        == 0
    )


@pytest.mark.asyncio
async def test_disabled_target_is_removed_from_task_candidates(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "task")
    target = await configured_target(targets)
    task, _run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    assert any(
        item.task.id == task.id
        for item in await tasks.list_task_candidates(fish_id, source.session_id)
    )

    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = false WHERE id = $1", target.target.id
    )

    assert all(
        item.task.id != task.id
        for item in await tasks.list_task_candidates(fish_id, source.session_id)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["new_run", "retry", "steer"])
async def test_disabled_target_blocks_frozen_t_route_before_mutation(
    postgres_pool: asyncpg.Pool,
    operation: str,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    if operation == "new_run":
        frozen_run = await tasks.fail_queued_run(run.id, "failed", "failed")
        content = "继续这个任务"
    elif operation == "retry":
        assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
        frozen_run = await tasks.mark_run_unknown(
            run.id, "dispatch_unknown", "unknown", {}
        )
        content = "重试这个任务"
    else:
        await start_run(tasks, run.id)
        frozen_run = await tasks.update_active_run(run.id, "running", {})
        content = "把任务改成另一种实现"
    assert frozen_run is not None
    context = frozen_task_context(task, frozen_run, target, content)
    current_source = await messages.insert_user_message(
        fish_id, content, session_id=source.session_id
    )
    confirmation = await messages.insert_generating_answer(
        fish_id, source.session_id, current_source.sequence
    )
    before = await postgres_pool.fetchrow(
        "SELECT status, provider_data FROM agent_runs WHERE id = $1", run.id
    )
    assert before is not None
    before_run_count = await postgres_pool.fetchval(
        "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
    )
    before_message_count = await postgres_pool.fetchval(
        "SELECT count(*) FROM messages WHERE fish_id = $1", fish_id
    )
    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = false WHERE id = $1", target.target.id
    )

    with pytest.raises((AgentRunConflictError, AgentTaskRouteConflictError)):
        await AgentTaskRouteService(tasks).commit_selected_route(
            fish_id=fish_id,
            resolved_target=target,
            task_context=context,
            kind=AgentTaskDecisionKind.CONTINUE_TASK,
            candidate_indexes=(0,),
            source_message_id=current_source.id,
            confirmation_message_id=confirmation.id,
            content=content,
        )

    after = await postgres_pool.fetchrow(
        "SELECT status, provider_data FROM agent_runs WHERE id = $1", run.id
    )
    assert after is not None
    assert dict(after) == dict(before)
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
        )
        == before_run_count
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE fish_id = $1", fish_id
        )
        == before_message_count
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE confirmation_message_id = $1",
            confirmation.id,
        )
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["input", "result_followup"])
async def test_disabled_target_blocks_reply_store_before_user_or_run_write(
    postgres_pool: asyncpg.Pool,
    operation: str,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    interaction_id: UUID | None = None
    if operation == "input":
        _completed, interaction = await tasks.complete_run_requiring_input(
            run.id, "need input", {}
        )
        interaction_id = interaction.id
        reply_subject = await tasks.materialize_interaction_request(
            interaction.id, "Which branch?"
        )
    else:
        await tasks.finish_run(run.id, "completed", "done", None, None, {})
        reply_subject = await tasks.materialize_run_result(run.id, "done")
    assert reply_subject is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            reply_subject.id,
            fish_id,
            reply_subject.session_id,
            len(reply_subject.content),
            False,
        )
        is not None
    )
    before_user_count = await postgres_pool.fetchval(
        "SELECT count(*) FROM messages WHERE fish_id = $1 AND role = 'user'", fish_id
    )
    before_run_count = await postgres_pool.fetchval(
        "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
    )
    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = false WHERE id = $1", target.target.id
    )

    with pytest.raises(AgentRunConflictError, match="target"):
        if interaction_id is not None:
            await tasks.insert_and_resolve_input(
                interaction_id, fish_id, source.session_id, "release/next"
            )
        else:
            await tasks.insert_and_create_result_followup(
                run.id,
                fish_id,
                source.session_id,
                "continue",
                "continue",
            )

    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE fish_id = $1 AND role = 'user'",
            fish_id,
        )
        == before_user_count
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
        )
        == before_run_count
    )
    if interaction_id is not None:
        interaction = await postgres_pool.fetchrow(
            """
            SELECT response_message_id, next_run_id, resolved_at
            FROM agent_interactions
            WHERE id = $1
            """,
            interaction_id,
        )
        assert interaction is not None
        assert dict(interaction) == {
            "response_message_id": None,
            "next_run_id": None,
            "resolved_at": None,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confirmation_status", [MessageStatus.INTERRUPTED, MessageStatus.FAILED]
)
async def test_confirmation_terminal_state_does_not_cancel_queued_run(
    postgres_pool: asyncpg.Pool,
    confirmation_status: MessageStatus,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "task")
    confirmation = await messages.insert_generating_answer(
        fish_id, source.session_id, source.sequence
    )
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "task",
        source.id,
        confirmation.id,
        source.content,
    )

    assert (
        await messages.finish_answer(
            confirmation.id,
            fish_id,
            content="partial confirmation",
            requested_status=confirmation_status,
        )
        is confirmation_status
    )

    stored = await postgres_pool.fetchrow(
        """
        SELECT status, confirmation_message_id
        FROM agent_runs
        WHERE id = $1
        """,
        run.id,
    )
    assert stored is not None
    assert dict(stored) == {
        "status": "queued",
        "confirmation_message_id": confirmation.id,
    }


@pytest.mark.asyncio
async def test_one_active_run_and_unknown_retry_guard(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, first = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    second_source = await messages.insert_user_message(
        fish_id, "second", session_id=source.session_id
    )

    with pytest.raises(AgentRunConflictError):
        await tasks.create_continuation_run(
            task.id, first.id, second_source.id, "second", None
        )

    assert await tasks.claim_prepared_run(first.id, "none", {}) is not None
    unknown = await tasks.mark_run_unknown(
        first.id, "dispatch_unknown", "uncertain", {}
    )
    assert unknown is not None and unknown.status is AgentRunStatus.UNKNOWN
    with pytest.raises(AgentRunConflictError):
        await tasks.create_continuation_run(
            task.id, first.id, second_source.id, "second", None
        )

    retry = await tasks.abandon_unknown_and_create_retry(
        first.id, second_source.id, None, "retry"
    )
    assert retry is not None
    assert retry.previous_run_id == first.id
    assert retry.status is AgentRunStatus.QUEUED
    assert (
        await postgres_pool.fetchval(
            "SELECT status FROM agent_runs WHERE id = $1", first.id
        )
        == "abandoned"
    )


@pytest.mark.asyncio
async def test_prepared_claim_is_single_winner_and_restart_marks_uncertain_dispatch(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )

    first, second = await asyncio.gather(
        tasks.claim_prepared_run(run.id, "none", {"attempt": 1}),
        tasks.claim_prepared_run(run.id, "none", {"attempt": 2}),
    )
    assert sum(item is not None for item in (first, second)) == 1
    assert await tasks.mark_unidentified_dispatches_unknown() == 1
    stored = await postgres_pool.fetchrow(
        "SELECT status, external_run_id FROM agent_runs WHERE id = $1", run.id
    )
    assert stored is not None
    assert dict(stored) == {"status": "unknown", "external_run_id": None}


@pytest.mark.asyncio
async def test_input_reply_is_atomic_and_concurrent_loser_leaves_no_user_message(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        run.id, "Which branch?", {"request_kind": "input"}
    )
    request = await tasks.materialize_interaction_request(
        interaction.id, "Which branch should I use?"
    )
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )

    async def respond(content: str):
        try:
            return await tasks.insert_and_resolve_input(
                interaction.id, fish_id, source.session_id, content
            )
        except AgentInteractionConflictError:
            return None

    first, second = await asyncio.gather(respond("main"), respond("release"))
    winner = first or second
    assert winner is not None
    response = winner.message
    resolved = winner.interaction
    next_run = winner.run
    assert resolved is not None and next_run is not None
    assert resolved.response_message_id == response.id
    assert resolved.next_run_id == next_run.id
    assert next_run.previous_run_id == run.id
    assert next_run.source_message_id == response.id
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE role = 'user' "
            "AND content = ANY($1::text[])",
            ["main", "release"],
        )
        == 1
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
        )
        == 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "content"),
    [
        (InputReplyAction.ANSWER, "Immediately"),
        (InputReplyAction.UNRELATED, "看下斗罗大陆动漫完结了吗"),
    ],
)
async def test_t_continue_classifies_before_consuming_waiting_interaction(
    postgres_pool: asyncpg.Pool,
    action: InputReplyAction,
    content: str,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    first = await messages.insert_user_message(fish_id, "check memory")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "check memory",
        first.id,
        None,
        first.content,
    )
    await start_run(tasks, run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        run.id,
        "How soon?",
        {},
    )
    request = await tasks.materialize_interaction_request(interaction.id, "How soon?")
    assert request is not None
    await tasks.finalize_agent_message_consumption(
        request.id,
        fish_id,
        request.session_id,
        len(request.content),
        False,
    )
    candidates = await tasks.list_task_candidates(fish_id, first.session_id)
    context = TaskLaneContext.from_records(
        candidates,
        current_content=content,
        capability_policy=LocalConnectorCapabilityPolicy(),
    )
    index = next(
        candidate.index
        for candidate in context.candidates
        if candidate.interaction is not None
        and candidate.interaction.id == interaction.id
    )
    source = await messages.insert_user_message(
        fish_id,
        content,
        session_id=first.session_id,
    )
    confirmation = await messages.insert_generating_answer(
        fish_id,
        first.session_id,
        source.sequence,
    )

    class FixedClassifier:
        async def classify_input(self, **_kwargs: object):
            return InputReplyClassification(action)

        async def classify_run_result(self, **_kwargs: object):
            raise AssertionError("T lane already selected the Interaction")

    child = await AgentTaskRouteService(
        tasks,
        interaction_reply_service=AgentReplyService(tasks, FixedClassifier()),
    ).commit_selected_route(
        fish_id=fish_id,
        resolved_target=target,
        task_context=context,
        kind=AgentTaskDecisionKind.CONTINUE_TASK,
        candidate_indexes=(index,),
        source_message_id=source.id,
        confirmation_message_id=confirmation.id,
        content=source.content,
    )

    stored_interaction = await postgres_pool.fetchrow(
        "SELECT response_message_id, next_run_id, resolved_at "
        "FROM agent_interactions WHERE id = $1",
        interaction.id,
    )
    assert stored_interaction is not None
    if action is InputReplyAction.ANSWER:
        assert stored_interaction["response_message_id"] == source.id
        assert stored_interaction["next_run_id"] == child.id
        assert stored_interaction["resolved_at"] is not None
        assert child.task_id == task.id
        assert child.previous_run_id == run.id
    else:
        assert stored_interaction["response_message_id"] is None
        assert stored_interaction["next_run_id"] is None
        assert stored_interaction["resolved_at"] is None
        assert child.task_id != task.id
        assert child.previous_run_id is None
    assert child.source_message_id == source.id
    assert child.confirmation_message_id == confirmation.id
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = $1",
            source.id,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_agent_message_waits_for_full_consumption_before_history_and_facts(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    facts = FactStore(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "The task is complete")
    assert result is not None

    initial = await postgres_pool.fetchrow(
        "SELECT status, consumed_offset, history_token_count, "
        "fact_extraction_status FROM messages WHERE id = $1",
        result.id,
    )
    assert initial is not None
    assert dict(initial) == {
        "status": "completed",
        "consumed_offset": 0,
        "history_token_count": 0,
        "fact_extraction_status": "waiting_consumption",
    }
    await messages.update_consumed_offset(result.id, fish_id, 4)
    partial = await postgres_pool.fetchrow(
        "SELECT history_token_count, fact_extraction_status "
        "FROM messages WHERE id = $1",
        result.id,
    )
    assert partial is not None
    assert dict(partial) == {
        "history_token_count": 0,
        "fact_extraction_status": "waiting_consumption",
    }
    await messages.insert_user_message(fish_id, "after", session_id=source.session_id)
    pending_before_barrier = await facts.list_pending_messages(limit=10, batch_size=10)
    assert [item.message.id for item in pending_before_barrier] == [source.id]
    assert pending_before_barrier[0].target_sequence < result.sequence
    progress_before_barrier = await facts.get_progress(fish_id, source.session_id)
    assert progress_before_barrier.target_sequence < result.sequence

    assert await messages.update_consumed_offset(
        result.id, fish_id, len(result.content)
    ) == len(result.content)
    finalized = await tasks.finalize_agent_message_consumption(
        result.id, fish_id, result.session_id, -1, False
    )
    repeated = await tasks.finalize_agent_message_consumption(
        result.id, fish_id, result.session_id, len(result.content), False
    )
    assert finalized is not None and repeated == finalized
    stored = await postgres_pool.fetchrow(
        "SELECT history_token_count, fact_extraction_status "
        "FROM messages WHERE id = $1",
        result.id,
    )
    assert stored is not None
    assert stored["history_token_count"] > 0
    assert stored["fact_extraction_status"] == "pending"


@pytest.mark.asyncio
async def test_interrupted_agent_message_is_truncated_and_releases_barrier(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "abcdefghij")
    assert result is not None

    finalized = await tasks.finalize_agent_message_consumption(
        result.id, fish_id, result.session_id, 4, True
    )
    assert finalized is not None
    assert finalized.content == "abcd"
    assert finalized.status.value == "interrupted"
    assert finalized.consumed_offset == 4
    assert (
        await postgres_pool.fetchval(
            "SELECT fact_extraction_status FROM messages WHERE id = $1", result.id
        )
        == "pending"
    )


@pytest.mark.asyncio
async def test_agent_message_delivery_has_exact_subject_and_activation_policy(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["approve", "deny"]}
    )
    request = await tasks.materialize_interaction_request(
        interaction.id, "Approve the operation?"
    )
    assert request is not None

    envelopes = await tasks.list_agent_messages_for_delivery(
        fish_id,
        source.session_id,
        after_sequence=0,
        include_incomplete=True,
        limit=10,
    )
    assert len(envelopes) == 1
    assert envelopes[0].message_id == request.id
    assert envelopes[0].subject.id == interaction.id
    assert envelopes[0].subject.kind.value == "interaction"
    assert envelopes[0].activation_policy.value == "playback_complete"
    assert envelopes[0].content == request.content
    assert envelopes[0].consumed_offset == 0


@pytest.mark.asyncio
async def test_agent_delivery_skips_legacy_blank_message_without_blocking_later(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    first_source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _first_task, first_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "first task",
        first_source.id,
        None,
        "first",
    )
    await start_run(tasks, first_run.id)
    await tasks.finish_run(first_run.id, "completed", "done", None, None, {})
    legacy = await tasks.materialize_run_result(first_run.id, "legacy result")
    assert legacy is not None
    await postgres_pool.execute(
        "UPDATE messages SET content = '' WHERE id = $1",
        legacy.id,
    )

    second_source = await messages.insert_user_message(
        fish_id,
        "second",
        session_id=first_source.session_id,
    )
    _second_task, second_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "second task",
        second_source.id,
        None,
        "second",
    )
    await start_run(tasks, second_run.id)
    await tasks.finish_run(second_run.id, "completed", "done", None, None, {})
    valid = await tasks.materialize_run_result(second_run.id, "valid result")
    assert valid is not None

    envelopes = await tasks.list_agent_messages_for_delivery(
        fish_id,
        first_source.session_id,
        after_sequence=0,
        include_incomplete=True,
        limit=10,
    )

    assert [envelope.message_id for envelope in envelopes] == [valid.id]


@pytest.mark.asyncio
async def test_approval_dispatch_and_running_steer_have_guarded_recovery(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["approve", "deny"]}
    )
    request = await tasks.materialize_interaction_request(interaction.id, "Approve?")
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    mutation = await tasks.insert_and_bind_approval_response(
        interaction.id,
        fish_id,
        source.session_id,
        "yes, please approve",
        {"choice": "approve", "resolve_all": True},
    )
    response = mutation.message
    bound = mutation.interaction
    assert bound is not None
    assert bound.response_message_id == response.id
    assert "yes, please approve" not in json.dumps(bound.provider_data)
    claimed = await tasks.claim_pending_approval_response()
    assert claimed is not None and claimed.interaction.id == interaction.id
    assert await tasks.claim_pending_approval_response() is None
    unknown = await tasks.mark_approval_dispatch_unknown(
        interaction.id, "timeout", "response uncertain"
    )
    assert unknown is not None
    resolved = await tasks.resolve_approval(interaction.id)
    assert resolved is not None and resolved.resolved_at is not None
    assert await tasks.update_active_run(run.id, "running", {}) is not None

    steer_source = await messages.insert_user_message(
        fish_id, "change direction", session_id=source.session_id
    )
    queued = await tasks.queue_running_steer(run.id, steer_source.id)
    assert queued is not None
    steer_work = await tasks.claim_pending_steer()
    assert steer_work is not None and steer_work.run.id == run.id
    assert await tasks.claim_pending_steer() is None
    assert (
        await tasks.mark_steer_dispatch_unknown(run.id, "timeout", "steer uncertain")
        is not None
    )
    same_unknown = await tasks.queue_running_steer(run.id, steer_source.id)
    assert same_unknown is not None
    assert same_unknown.provider_data is not None
    assert mapping_field(same_unknown.provider_data, "steer")["state"] == "unknown"
    assert await tasks.claim_pending_steer() is None
    blocked_source = await messages.insert_user_message(
        fish_id, "another direction", session_id=source.session_id
    )
    assert await tasks.queue_running_steer(run.id, blocked_source.id) is None

    await tasks.finish_run(run.id, "completed", "phase one", None, None, {})
    continuation_source = await messages.insert_user_message(
        fish_id, "continue", session_id=source.session_id
    )
    continuation = await tasks.create_continuation_run(
        _task.id, run.id, continuation_source.id, "continue", None
    )
    await start_run(tasks, continuation.id)
    accepted_source = await messages.insert_user_message(
        fish_id, "accepted steer", session_id=source.session_id
    )
    assert (
        await tasks.queue_running_steer(continuation.id, accepted_source.id) is not None
    )
    assert await tasks.claim_pending_steer() is not None
    accepted = await tasks.record_steer_accepted(continuation.id)
    assert accepted is not None
    assert accepted.provider_data is not None
    assert mapping_field(accepted.provider_data, "steer")["state"] == "accepted"
    same_accepted = await tasks.queue_running_steer(continuation.id, accepted_source.id)
    assert same_accepted is not None
    assert same_accepted.provider_data is not None
    assert mapping_field(same_accepted.provider_data, "steer")["state"] == "accepted"
    assert await tasks.claim_pending_steer() is None
    replacement_source = await messages.insert_user_message(
        fish_id, "new steer", session_id=source.session_id
    )
    assert (
        await tasks.queue_running_steer(continuation.id, replacement_source.id)
        is not None
    )
    assert await tasks.claim_pending_steer() is not None
    assert await tasks.mark_unidentified_steer_dispatches_unknown() == 1

    second_approval = await tasks.open_approval_interaction(
        continuation.id, {"allowed_decisions": ["approve", "deny"]}
    )
    second_request = await tasks.materialize_interaction_request(
        second_approval.id, "Approve again?"
    )
    assert second_request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            second_request.id,
            fish_id,
            second_request.session_id,
            len(second_request.content),
            False,
        )
        is not None
    )
    await tasks.insert_and_bind_approval_response(
        second_approval.id,
        fish_id,
        source.session_id,
        "deny this time",
        "deny",
    )
    assert await tasks.claim_pending_approval_response() is not None
    assert await tasks.mark_unidentified_approval_dispatches_unknown() == 1
    assert await tasks.resolve_approval(second_approval.id) is not None

    third_approval = await tasks.open_approval_interaction(
        continuation.id, {"allowed_decisions": ["approve", "deny"]}
    )
    closed = await tasks.close_open_interaction(continuation.id)
    assert closed is not None and closed.id == third_approval.id


@pytest.mark.asyncio
async def test_terminal_snapshot_preserves_inflight_approval_for_resolution(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "approval task")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "approval task",
        source.id,
        None,
        source.content,
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id,
        {"allowed_decisions": ["once", "deny"]},
    )
    request = await tasks.materialize_interaction_request(
        interaction.id,
        "Approve the operation?",
    )
    assert request is not None
    await tasks.finalize_agent_message_consumption(
        request.id,
        fish_id,
        request.session_id,
        len(request.content),
        False,
    )
    await tasks.insert_and_bind_approval_response(
        interaction.id,
        fish_id,
        source.session_id,
        "allow once",
        {"choice": "once", "resolve_all": False},
    )
    assert await tasks.claim_pending_approval_response() is not None

    await tasks.finish_run(
        run.id,
        AgentRunStatus.COMPLETED,
        "done",
        None,
        None,
        {},
    )
    before_resolve = await postgres_pool.fetchrow(
        """
        SELECT resolved_at, closed_at,
               provider_data #>> '{approval_response,state}' AS response_state
        FROM agent_interactions
        WHERE id = $1
        """,
        interaction.id,
    )
    assert before_resolve is not None
    assert before_resolve["resolved_at"] is None
    assert before_resolve["closed_at"] is None
    assert before_resolve["response_state"] == "dispatching"

    resolved = await tasks.resolve_approval(interaction.id)
    assert resolved is not None
    assert resolved.resolved_at is not None
    assert resolved.closed_at is None


@pytest.mark.asyncio
async def test_input_required_resolves_inflight_approval_before_opening_input(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "approval then input")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "approval then input",
        source.id,
        None,
        source.content,
    )
    await start_run(tasks, run.id)
    approval = await tasks.open_approval_interaction(
        run.id,
        {"allowed_decisions": ["once", "deny"]},
    )
    request = await tasks.materialize_interaction_request(
        approval.id,
        "Approve before asking for input?",
    )
    assert request is not None
    await tasks.finalize_agent_message_consumption(
        request.id,
        fish_id,
        request.session_id,
        len(request.content),
        False,
    )
    await tasks.insert_and_bind_approval_response(
        approval.id,
        fish_id,
        source.session_id,
        "allow once",
        {"choice": "once", "resolve_all": False},
    )
    assert await tasks.claim_pending_approval_response() is not None

    completed, input_interaction = await tasks.complete_run_requiring_input(
        run.id,
        "Which environment?",
        {},
    )
    replayed_run, replayed_interaction = await tasks.complete_run_requiring_input(
        run.id,
        "Which environment?",
        {},
    )
    input_request = await tasks.materialize_interaction_request(
        input_interaction.id,
        "Which environment?",
    )
    assert input_request is not None
    await tasks.finalize_agent_message_consumption(
        input_request.id,
        fish_id,
        input_request.session_id,
        len(input_request.content),
        False,
    )
    await tasks.insert_and_resolve_input(
        input_interaction.id,
        fish_id,
        source.session_id,
        "staging",
    )
    terminal_replay, resolved_input = await tasks.complete_run_requiring_input(
        run.id,
        "Which environment?",
        {},
    )

    approval_state = await postgres_pool.fetchrow(
        """
        SELECT resolved_at, closed_at,
               provider_data #>> '{approval_response,state}' AS response_state
        FROM agent_interactions
        WHERE id = $1
        """,
        approval.id,
    )
    assert completed.status is AgentRunStatus.COMPLETED
    assert input_interaction.kind is AgentInteractionKind.INPUT
    assert replayed_run.id == completed.id
    assert replayed_interaction.id == input_interaction.id
    assert terminal_replay.id == completed.id
    assert resolved_input.id == input_interaction.id
    assert resolved_input.resolved_at is not None
    assert approval_state is not None
    assert approval_state["resolved_at"] is not None
    assert approval_state["closed_at"] is None
    assert approval_state["response_state"] == "accepted"
    open_kinds = await postgres_pool.fetch(
        """
        SELECT kind
        FROM agent_interactions
        WHERE run_id = $1
          AND resolved_at IS NULL
          AND closed_at IS NULL
        """,
        run.id,
    )
    assert [row["kind"] for row in open_kinds] == []


@pytest.mark.asyncio
async def test_stale_run_listing_and_failure_are_bounded_and_cas_guarded(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    seed = await messages.insert_user_message(fish_id, "seed")
    target = await configured_target(targets)

    async def create_run(label: str):
        source = await messages.insert_user_message(
            fish_id,
            label,
            session_id=seed.session_id,
        )
        _task, run = await tasks.create_task_with_first_run(
            fish_id,
            target.target.id,
            label,
            source.id,
            None,
            label,
        )
        return run

    queued = await create_run("queued")
    dispatching = await create_run("dispatching")
    assert await tasks.claim_prepared_run(dispatching.id, "none", {}) is not None
    running = await create_run("running")
    await start_run(tasks, running.id)
    unknown = await create_run("unknown")
    await start_run(tasks, unknown.id)
    unknown_interaction = await tasks.open_approval_interaction(unknown.id, {})
    assert (
        await tasks.mark_run_unknown(
            unknown.id,
            "provider_unreachable",
            "provider unreachable",
            {},
        )
        is not None
    )
    waiting = await create_run("waiting")
    await start_run(tasks, waiting.id)
    waiting_interaction = await tasks.open_approval_interaction(waiting.id, {})
    completed = await create_run("completed")
    await start_run(tasks, completed.id)
    await tasks.finish_run(completed.id, "completed", "done", None, None, {})
    fresh = await create_run("fresh")

    cutoff = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)
    ordered_runs = (queued, dispatching, running, unknown, waiting, completed)
    for index, run in enumerate(ordered_runs):
        await postgres_pool.execute(
            "UPDATE agent_runs SET updated_at = $2 WHERE id = $1",
            run.id,
            cutoff - timedelta(minutes=len(ordered_runs) - index),
        )
    await postgres_pool.execute(
        "UPDATE agent_runs SET updated_at = $2 WHERE id = $1",
        fresh.id,
        cutoff + timedelta(seconds=1),
    )

    stale = await tasks.list_stale_run_work(cutoff)
    assert [item.run.id for item in stale] == [
        dispatching.id,
        running.id,
        unknown.id,
    ]
    assert [item.run.id for item in await tasks.list_stale_run_work(cutoff, 2)] == [
        dispatching.id,
        running.id,
    ]
    with pytest.raises(ValueError, match="limit must be positive"):
        await tasks.list_stale_run_work(cutoff, 0)

    assert await tasks.fail_stale_run(queued.id, "late", "late") is None
    for run in (dispatching, running, unknown):
        failed = await tasks.fail_stale_run(
            run.id,
            "agent_timeout",
            "Agent task timed out",
        )
        assert failed is not None
        assert failed.status is AgentRunStatus.FAILED
        assert failed.error_code == "agent_timeout"
        assert failed.error_message == "Agent task timed out"
        assert failed.finished_at is not None
        assert await tasks.fail_stale_run(run.id, "late", "late") is None

    assert (
        await postgres_pool.fetchval(
            "SELECT closed_at FROM agent_interactions WHERE id = $1",
            unknown_interaction.id,
        )
        is not None
    )
    assert await tasks.fail_stale_run(waiting.id, "late", "late") is None
    assert await tasks.fail_stale_run(completed.id, "late", "late") is None
    waiting_state = await postgres_pool.fetchrow(
        """
        SELECT run.status, interaction.closed_at
        FROM agent_runs AS run
        JOIN agent_interactions AS interaction ON interaction.run_id = run.id
        WHERE run.id = $1 AND interaction.id = $2
        """,
        waiting.id,
        waiting_interaction.id,
    )
    assert waiting_state is not None
    assert dict(waiting_state) == {"status": "waiting_user", "closed_at": None}


@pytest.mark.asyncio
async def test_queue_failure_terminal_replay_and_result_followup_are_cas_guarded(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    peeked = await tasks.peek_queued_run()
    assert peeked is not None and peeked.run.id == run.id
    failed = await tasks.fail_queued_run(run.id, "invalid_request", "bad input")
    assert failed is not None and failed.status is AgentRunStatus.FAILED
    assert await tasks.fail_queued_run(run.id, "invalid_request", "again") is None

    second_source = await messages.insert_user_message(
        fish_id, "retry", session_id=source.session_id
    )
    continuation = await tasks.create_continuation_run(
        task.id, run.id, second_source.id, "retry", None
    )
    await start_run(tasks, continuation.id)
    finished = await tasks.finish_run(
        continuation.id, "completed", "done", None, None, {"cursor": "1"}
    )
    replayed = await tasks.finish_run(
        continuation.id, "completed", "different", None, None, {"cursor": "2"}
    )
    assert replayed == finished
    with pytest.raises(AgentRunConflictError):
        await tasks.finish_run(
            continuation.id, "failed", None, "provider_error", "late", {}
        )
    result = await tasks.materialize_run_result(continuation.id, "Finished")
    assert result is not None
    assert await tasks.materialize_run_result(continuation.id, "Duplicate") is None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )

    mutation = await tasks.insert_and_create_result_followup(
        continuation.id,
        fish_id,
        source.session_id,
        "make one change",
        "changes_requested",
    )
    user = mutation.message
    followup = mutation.run
    assert followup is not None
    assert followup.source_message_id == user.id
    assert followup.previous_run_id == continuation.id
    with pytest.raises(AgentRunConflictError):
        await tasks.insert_and_create_result_followup(
            continuation.id,
            fish_id,
            source.session_id,
            "another change",
            "changes_requested",
        )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE content = 'another change'"
        )
        == 0
    )


@pytest.mark.asyncio
async def test_full_consumption_rolls_back_offset_when_transition_is_cancelled(
    postgres_pool: asyncpg.Pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "atomic playback")
    assert result is not None

    async def cancel_rebuild(*_args: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(messages, "_rebuild_history_window", cancel_rebuild)
    with pytest.raises(asyncio.CancelledError):
        await messages.update_consumed_offset(result.id, fish_id, len(result.content))

    stored = await postgres_pool.fetchrow(
        """
        SELECT consumed_offset, history_token_count, fact_extraction_status
        FROM messages
        WHERE id = $1
        """,
        result.id,
    )
    assert stored is not None
    assert dict(stored) == {
        "consumed_offset": 0,
        "history_token_count": 0,
        "fact_extraction_status": "waiting_consumption",
    }
    envelopes = await tasks.list_agent_messages_for_delivery(
        fish_id,
        result.session_id,
        after_sequence=result.sequence,
        include_incomplete=True,
        limit=10,
    )
    assert [item.message_id for item in envelopes] == [result.id]

    monkeypatch.undo()
    assert await messages.update_consumed_offset(
        result.id, fish_id, len(result.content)
    ) == len(result.content)
    assert await messages.update_consumed_offset(result.id, fish_id, -1) == len(
        result.content
    )
    released = await postgres_pool.fetchrow(
        """
        SELECT consumed_offset, fact_extraction_status
        FROM messages
        WHERE id = $1
        """,
        result.id,
    )
    assert released is not None
    assert dict(released) == {
        "consumed_offset": len(result.content),
        "fact_extraction_status": "pending",
    }


@pytest.mark.asyncio
async def test_delivery_recovers_legacy_full_offset_waiting_message(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "legacy full offset")
    assert result is not None
    await postgres_pool.execute(
        "UPDATE messages SET consumed_offset = char_length(content) WHERE id = $1",
        result.id,
    )

    envelopes = await tasks.list_agent_messages_for_delivery(
        fish_id,
        result.session_id,
        after_sequence=result.sequence,
        include_incomplete=True,
        limit=10,
    )
    assert [item.message_id for item in envelopes] == [result.id]


@pytest.mark.asyncio
async def test_approval_resolution_and_terminal_finish_share_run_first_lock_order(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["approve"]}
    )
    request = await tasks.materialize_interaction_request(interaction.id, "Approve?")
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    await tasks.insert_and_bind_approval_response(
        interaction.id,
        fish_id,
        source.session_id,
        "approve",
        "approve",
    )
    assert await tasks.claim_pending_approval_response() is not None

    resolved, finished = await asyncio.wait_for(
        asyncio.gather(
            tasks.resolve_approval(interaction.id),
            tasks.finish_run(run.id, "completed", "done", None, None, {}),
        ),
        timeout=5,
    )
    assert finished.status is AgentRunStatus.COMPLETED
    state = await postgres_pool.fetchrow(
        """
        SELECT run.status, interaction.resolved_at, interaction.closed_at
        FROM agent_runs AS run
        JOIN agent_interactions AS interaction ON interaction.run_id = run.id
        WHERE run.id = $1
        """,
        run.id,
    )
    assert state is not None
    assert state["status"] == "completed"
    assert (state["resolved_at"] is None) != (state["closed_at"] is None)
    if resolved is not None:
        assert resolved.resolved_at is not None


@pytest.mark.asyncio
async def test_resolve_approval_does_not_guess_authoritative_run_status(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["approve"]}
    )
    request = await tasks.materialize_interaction_request(interaction.id, "Approve?")
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    await tasks.insert_and_bind_approval_response(
        interaction.id,
        fish_id,
        source.session_id,
        "approve",
        "approve",
    )
    assert await tasks.claim_pending_approval_response() is not None
    assert await tasks.resolve_approval(interaction.id) is not None
    assert (
        await postgres_pool.fetchval(
            "SELECT status FROM agent_runs WHERE id = $1", run.id
        )
        == "waiting_user"
    )


@pytest.mark.asyncio
async def test_lineage_uses_unique_chain_tail_and_rejects_branches(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, first = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    assert await tasks.fail_queued_run(first.id, "failed", "first") is not None
    second_source = await messages.insert_user_message(
        fish_id, "second", session_id=source.session_id
    )
    second = await tasks.create_continuation_run(
        task.id, first.id, second_source.id, "second", None
    )
    assert await tasks.fail_queued_run(second.id, "failed", "second") is not None
    manual_tail_id = UUID("00000000-0000-0000-0000-000000000001")
    await postgres_pool.execute(
        """
        INSERT INTO agent_runs (
            id, task_id, target_id, fish_id, session_id, previous_run_id,
            source_message_id,
            input, status, created_at, updated_at, finished_at
        )
        SELECT $1, task_id, target_id, fish_id, session_id, id, $2, 'manual', 'failed',
               created_at, updated_at, updated_at
        FROM agent_runs
        WHERE id = $3
        """,
        manual_tail_id,
        second_source.id,
        second.id,
    )
    await postgres_pool.execute(
        """
        UPDATE agent_runs
        SET created_at = date_trunc('second', now())
        WHERE task_id = $1
        """,
        task.id,
    )
    third_source = await messages.insert_user_message(
        fish_id, "third", session_id=source.session_id
    )
    third = await tasks.create_continuation_run(
        task.id, manual_tail_id, third_source.id, "third", None
    )
    assert third.previous_run_id == manual_tail_id
    assert await tasks.fail_queued_run(third.id, "failed", "third") is not None

    with pytest.raises(asyncpg.UniqueViolationError):
        await postgres_pool.execute(
            """
            INSERT INTO agent_runs (
                id, task_id, target_id, fish_id, session_id, previous_run_id,
                source_message_id,
                input, status
            )
            SELECT $1, task_id, target_id, fish_id, session_id,
                   previous_run_id, $2, 'branch', 'failed'
            FROM agent_runs
            WHERE id = $3
            """,
            uuid4(),
            third_source.id,
            third.id,
        )

    await postgres_pool.execute(
        """
        INSERT INTO agent_runs (
            id, task_id, target_id, fish_id, session_id, source_message_id,
            input, status
        )
        SELECT $1, id, target_id, fish_id, session_id, $2, 'orphan', 'failed'
        FROM agent_tasks
        WHERE id = $3
        """,
        uuid4(),
        third_source.id,
        task.id,
    )
    fourth_source = await messages.insert_user_message(
        fish_id, "fourth", session_id=source.session_id
    )
    with pytest.raises(AgentRunConflictError, match="exactly one lineage tail"):
        await tasks.create_continuation_run(
            task.id, third.id, fourth_source.id, "fourth", None
        )


@pytest.mark.asyncio
async def test_steer_rejects_cross_session_user_message(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    cross_session_id = uuid4()
    await postgres_pool.execute(
        "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2)",
        cross_session_id,
        fish_id,
    )
    cross_session = await messages.insert_user_message(
        fish_id, "other session", session_id=cross_session_id
    )

    assert await tasks.queue_running_steer(run.id, cross_session.id) is None
    stored = await postgres_pool.fetchval(
        "SELECT provider_data -> 'steer' FROM agent_runs WHERE id = $1",
        run.id,
    )
    assert stored is None


@pytest.mark.asyncio
async def test_approval_reserve_rejects_unsafe_boundary_before_user_write(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    base = {
        "allowed_decisions": ["approve"],
        "padding": "",
        "approval_response": {
            "choice": "approve",
            "resolve_all": False,
            "state": "queued",
        },
    }
    recovered = json.loads(json.dumps(base))
    recovered["approval_response"].update(
        {
            "state": "unknown",
            "error_code": "\x01" * 512,
            "error_message": "\x01" * 4_000,
        }
    )
    exact_reserve = canonical_json_bytes(recovered) - canonical_json_bytes(base)
    safe_padding = 65_536 - exact_reserve - canonical_json_bytes(base)
    unsafe_provider = {
        "allowed_decisions": ["approve"],
        "padding": "x" * (safe_padding + 1),
    }
    interaction = await tasks.open_approval_interaction(run.id, unsafe_provider)
    request = await tasks.materialize_interaction_request(interaction.id, "Approve?")
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    before = await postgres_pool.fetchval(
        "SELECT count(*) FROM messages WHERE role = 'user'"
    )

    with pytest.raises(ValueError, match="provider_data exceeds"):
        await tasks.insert_and_bind_approval_response(
            interaction.id,
            fish_id,
            source.session_id,
            "approve",
            "approve",
        )

    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE role = 'user'"
        )
        == before
    )

    safe_provider = {
        "allowed_decisions": ["approve"],
        "padding": "x" * safe_padding,
    }
    await postgres_pool.execute(
        "UPDATE agent_interactions SET provider_data = $2::jsonb WHERE id = $1",
        interaction.id,
        json.dumps(safe_provider, separators=(",", ":")),
    )
    await tasks.insert_and_bind_approval_response(
        interaction.id,
        fish_id,
        source.session_id,
        "approve",
        "approve",
    )
    assert await tasks.claim_pending_approval_response() is not None
    unknown = await tasks.mark_approval_dispatch_unknown(
        interaction.id,
        "\x01" * 512,
        "\x01" * 4_000,
    )
    assert unknown is not None and unknown.provider_data is not None
    assert canonical_json_bytes(unknown.provider_data) == 65_536


@pytest.mark.asyncio
async def test_steer_reserve_rejects_unsafe_boundary_then_reaches_unknown_limit(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    steer_source = await messages.insert_user_message(
        fish_id, "steer", session_id=source.session_id
    )
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    await start_run(tasks, run.id)
    base = {
        "padding": "",
        "steer": {
            "source_message_id": str(steer_source.id),
            "state": "queued",
        },
    }
    recovered = json.loads(json.dumps(base))
    recovered["steer"].update(
        {
            "state": "unknown",
            "error_code": "\x01" * 512,
            "error_message": "\x01" * 4_000,
        }
    )
    exact_reserve = canonical_json_bytes(recovered) - canonical_json_bytes(base)
    safe_padding = 65_536 - exact_reserve - canonical_json_bytes(base)
    assert (
        await tasks.update_active_run(
            run.id, "running", {"padding": "x" * (safe_padding + 1)}
        )
        is not None
    )

    with pytest.raises(ValueError, match="provider_data exceeds"):
        await tasks.queue_running_steer(run.id, steer_source.id)
    assert (
        await postgres_pool.fetchval(
            "SELECT provider_data -> 'steer' FROM agent_runs WHERE id = $1", run.id
        )
        is None
    )

    assert (
        await tasks.update_active_run(
            run.id, "running", {"padding": "x" * safe_padding}
        )
        is not None
    )
    assert await tasks.queue_running_steer(run.id, steer_source.id) is not None
    assert await tasks.claim_pending_steer() is not None
    unknown = await tasks.mark_steer_dispatch_unknown(
        run.id,
        "\x01" * 512,
        "\x01" * 4_000,
    )
    assert unknown is not None and unknown.provider_data is not None
    assert canonical_json_bytes(unknown.provider_data) == 65_536


@pytest.mark.asyncio
async def test_disabled_targets_stay_visible_to_queue_and_remote_reconciliation(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "remote")
    target = await configured_target(targets)
    _remote_task, remote = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "remote", source.id, None, "remote"
    )
    second_source = await messages.insert_user_message(
        fish_id, "uncertain", session_id=source.session_id
    )
    _uncertain_task, uncertain = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "uncertain", second_source.id, None, "uncertain"
    )
    await start_run(tasks, remote.id)
    assert await tasks.claim_prepared_run(uncertain.id, "none", {}) is not None
    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = false WHERE id = $1",
        target.target.id,
    )

    # A disabled queued target remains visible so Processor can fail it explicitly.
    third_source = await messages.insert_user_message(
        fish_id, "queued", session_id=source.session_id
    )
    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = true WHERE id = $1", target.target.id
    )
    _queued_task, queued = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "queued", third_source.id, None, "queued"
    )
    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = false WHERE id = $1", target.target.id
    )
    peeked = await tasks.peek_queued_run()
    assert peeked is not None and peeked.run.id == queued.id
    assert peeked.target.enabled is False

    reconcilable = await tasks.list_reconcilable_run_work(limit=10)
    assert [item.run.id for item in reconcilable] == [remote.id]
    assert reconcilable[0].run.external_run_id is not None
    assert reconcilable[0].target.enabled is False


@pytest.mark.asyncio
async def test_active_update_and_dispatch_acceptance_are_cas_guarded(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "first"
    )
    assert await tasks.claim_prepared_run(run.id, "none", {}) is not None
    accepted = await tasks.record_dispatch_accepted(run.id, "external-one", {})
    replay = await tasks.record_dispatch_accepted(run.id, "external-one", {"x": 1})
    assert replay == accepted
    with pytest.raises(AgentRunConflictError):
        await tasks.record_dispatch_accepted(run.id, "external-two", {})

    waiting = await tasks.update_active_run(run.id, "waiting_user", {"cursor": "1"})
    assert waiting is not None and waiting.status is AgentRunStatus.WAITING_USER
    terminal = await tasks.finish_run(
        run.id, "completed", "done", None, None, {"cursor": "2"}
    )
    assert terminal.status is AgentRunStatus.COMPLETED
    assert await tasks.update_active_run(run.id, "running", {"cursor": "3"}) is None
    assert (
        await postgres_pool.fetchval(
            "SELECT status FROM agent_runs WHERE id = $1", run.id
        )
        == "completed"
    )


@pytest.mark.asyncio
async def test_task_creation_rejects_invalid_source_and_confirmation_identity(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    other_fish = uuid4()
    source = await messages.insert_user_message(fish_id, "source")
    other_source = await messages.insert_user_message(other_fish, "other")
    target = await configured_target(targets)
    assistant = await messages.insert_generating_answer(
        fish_id, source.session_id, source.sequence
    )
    other_session_id = uuid4()
    await postgres_pool.execute(
        "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2)",
        other_session_id,
        fish_id,
    )
    other_session_user = await messages.insert_user_message(
        fish_id, "new session", session_id=other_session_id
    )
    other_session_assistant = await messages.insert_generating_answer(
        fish_id,
        other_session_user.session_id,
        other_session_user.sequence,
    )

    invalid_pairs = [
        (assistant.id, None),
        (other_source.id, None),
        (source.id, source.id),
        (source.id, other_session_assistant.id),
    ]
    for invalid_source, invalid_confirmation in invalid_pairs:
        with pytest.raises((AgentRunConflictError, RuntimeError)):
            await tasks.create_task_with_first_run(
                fish_id,
                target.target.id,
                "invalid",
                invalid_source,
                invalid_confirmation,
                "invalid",
            )
    assert await postgres_pool.fetchval("SELECT count(*) FROM agent_tasks") == 0


@pytest.mark.asyncio
async def test_delivery_covers_result_input_incomplete_and_cross_subject_collision(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "result")
    target = await configured_target(targets)
    result_task, result_run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "result", source.id, None, "result"
    )
    await start_run(tasks, result_run.id)
    await tasks.finish_run(result_run.id, "completed", "done", None, None, {})
    result_message = await tasks.materialize_run_result(result_run.id, "result message")
    assert result_message is not None

    input_source = await messages.insert_user_message(
        fish_id, "input", session_id=source.session_id
    )
    input_task, input_run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "input", input_source.id, None, "input"
    )
    await start_run(tasks, input_run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        input_run.id, "need input", {"kind": "input"}
    )
    input_message = await tasks.materialize_interaction_request(
        interaction.id, "input message"
    )
    assert input_message is not None

    envelopes = await tasks.list_agent_messages_for_delivery(
        fish_id,
        source.session_id,
        after_sequence=0,
        include_incomplete=True,
        limit=10,
    )
    by_id = {item.message_id: item for item in envelopes}
    assert by_id[result_message.id].subject.kind.value == "run_result"
    assert by_id[result_message.id].activation_policy.value == "playback_start"
    assert by_id[input_message.id].subject.kind.value == "interaction"
    assert by_id[input_message.id].activation_policy.value == "playback_start"

    collision_source = await messages.insert_user_message(
        fish_id, "collision", session_id=source.session_id
    )
    with pytest.raises(RuntimeError, match="already has a subject"):
        await tasks.create_task_with_first_run(
            fish_id,
            target.target.id,
            "collision",
            collision_source.id,
            result_message.id,
            "collision",
        )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_tasks WHERE id <> ALL($1::uuid[])",
            [result_task.id, input_task.id],
        )
        == 0
    )


@pytest.mark.asyncio
async def test_fact_and_history_release_then_terminal_failure_advances_watermark(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    facts = FactStore(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "source")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, "source"
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "agent fact")
    assert result is not None
    after = await messages.insert_user_message(
        fish_id, "after", session_id=source.session_id
    )

    before_history = await messages.load_history(
        fish_id,
        source.session_id,
        before_sequence=after.sequence + 1,
    )
    assert result.id not in {message.id for message in before_history}
    assert await messages.update_consumed_offset(
        result.id, fish_id, len(result.content)
    ) == len(result.content)
    after_history = await messages.load_history(
        fish_id,
        source.session_id,
        before_sequence=after.sequence + 1,
    )
    assert result.id in {message.id for message in after_history}
    pending = await facts.list_pending_messages(limit=10, batch_size=10)
    result_pending = next(item for item in pending if item.message.id == result.id)
    await facts.record_failure(
        result_pending, "terminal extraction failure", terminal=True
    )
    progress = await facts.get_progress(fish_id, source.session_id)
    assert progress.through_sequence >= result.sequence
    assert (
        await postgres_pool.fetchval(
            "SELECT fact_extraction_status FROM messages WHERE id = $1", result.id
        )
        == "failed"
    )


@pytest.mark.asyncio
async def test_local_projection_uses_visible_chain_and_excludes_current_source(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "start task")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        run.id, "provider needs a branch", {"request_kind": "input"}
    )
    stored_provider = await postgres_pool.fetchval(
        "SELECT provider_data FROM agent_runs WHERE id = $1", run.id
    )
    assert json.loads(stored_provider) == {"request_kind": "input"}
    request = await tasks.materialize_interaction_request(
        interaction.id, "Which branch should I use?"
    )
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    mutation = await tasks.insert_and_resolve_input(
        interaction.id,
        fish_id,
        source.session_id,
        "Use release/next",
    )
    response = mutation.message
    next_run = mutation.run
    assert next_run is not None

    assert await tasks.load_local_projection(next_run.id) == (
        ("user", "start task"),
        ("assistant", "Which branch should I use?"),
        ("assistant", "provider needs a branch"),
    )
    assert await tasks.load_local_projection(next_run.id, limit=2) == (
        ("assistant", "Which branch should I use?"),
        ("assistant", "provider needs a branch"),
    )
    assert await tasks.load_user_message_content(response.id) == "Use release/next"
    assert await tasks.load_user_message_content(request.id) is None

    assert await tasks.fail_queued_run(next_run.id, "cancelled", "test") is not None
    later_source = await messages.insert_user_message(
        fish_id, "continue later", session_id=source.session_id
    )
    later = await tasks.create_continuation_run(
        _task.id, next_run.id, later_source.id, later_source.content, None
    )
    later_projection = await tasks.load_local_projection(later.id)
    assert ("user", "Use release/next") in later_projection


@pytest.mark.asyncio
async def test_local_projection_ignores_zero_offset_interrupted_request(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        run.id, "provider output", {"request_kind": "input"}
    )
    request = await tasks.materialize_interaction_request(
        interaction.id, "This was not played"
    )
    assert request is not None
    interrupted = await tasks.finalize_agent_message_consumption(
        request.id, fish_id, request.session_id, 0, True
    )
    assert interrupted is not None
    assert interrupted.status is MessageStatus.INTERRUPTED
    assert interrupted.content == ""
    mutation = await tasks.insert_and_resolve_input(
        interaction.id, fish_id, source.session_id, "continue"
    )
    next_run = mutation.run
    assert next_run is not None

    assert await tasks.load_local_projection(next_run.id) == (
        ("user", "first"),
        ("assistant", "provider output"),
    )


@pytest.mark.asyncio
async def test_reconcilable_approval_response_read_does_not_reclaim_or_post(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["deny"]}
    )
    request = await tasks.materialize_interaction_request(interaction.id, "Deny?")
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    await tasks.insert_and_bind_approval_response(
        interaction.id, fish_id, source.session_id, "deny", "deny"
    )
    claimed = await tasks.claim_pending_approval_response()
    assert claimed is not None

    dispatching = await tasks.list_reconcilable_approval_responses()
    assert [item.interaction.id for item in dispatching] == [interaction.id]
    assert await tasks.mark_unidentified_approval_dispatches_unknown() == 1
    unknown = await tasks.list_reconcilable_approval_responses()
    assert [item.interaction.id for item in unknown] == [interaction.id]
    assert await tasks.claim_pending_approval_response() is None

    second_source = await messages.insert_user_message(
        fish_id, "second", session_id=source.session_id
    )
    _second_task, second_run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "second", second_source.id, None, "second"
    )
    await start_run(tasks, second_run.id)
    second_interaction = await tasks.open_approval_interaction(
        second_run.id, {"allowed_decisions": ["deny"]}
    )
    second_request = await tasks.materialize_interaction_request(
        second_interaction.id, "Deny 2?"
    )
    assert second_request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            second_request.id,
            fish_id,
            second_request.session_id,
            len(second_request.content),
            False,
        )
        is not None
    )
    await tasks.insert_and_bind_approval_response(
        second_interaction.id, fish_id, source.session_id, "deny 2", "deny"
    )
    assert await tasks.claim_pending_approval_response() is not None

    run_page_one = await tasks.list_reconcilable_run_work(limit=1)
    run_cursor = (run_page_one[-1].run.created_at, run_page_one[-1].run.id)
    run_page_two = await tasks.list_reconcilable_run_work(limit=1, after=run_cursor)
    assert {item.run.id for item in (*run_page_one, *run_page_two)} == {
        run.id,
        second_run.id,
    }
    approval_page_one = await tasks.list_reconcilable_approval_responses(limit=1)
    approval_cursor = (
        approval_page_one[-1].interaction.created_at,
        approval_page_one[-1].interaction.id,
    )
    approval_page_two = await tasks.list_reconcilable_approval_responses(
        limit=1, after=approval_cursor
    )
    assert {
        item.interaction.id for item in (*approval_page_one, *approval_page_two)
    } == {interaction.id, second_interaction.id}


@pytest.mark.asyncio
async def test_reply_subject_snapshot_is_exact_bounded_and_consumption_gated(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id,
        {
            "allowed_decisions": ["once", "deny"],
            "recoveredWithoutDetails": False,
        },
    )
    request = await tasks.materialize_interaction_request(
        interaction.id, "Approve this operation?"
    )
    assert request is not None
    context = AgentReplyContext(AgentMessageSubjectKind.INTERACTION, interaction.id)

    with pytest.raises(AgentReplyConflictError, match="replyable"):
        await tasks.load_reply_subject(context, fish_id, source.session_id)

    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    snapshot = await tasks.load_reply_subject(context, fish_id, source.session_id)

    assert snapshot.context == context
    assert snapshot.source_message_id == request.id
    assert snapshot.source_content == request.content
    assert snapshot.interaction_kind is AgentInteractionKind.APPROVAL
    assert snapshot.allowed_decisions == ("once", "deny")
    assert snapshot.recovered_without_details is False
    with pytest.raises(AgentReplyConflictError, match="replyable"):
        await tasks.load_reply_subject(context, uuid4(), source.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "allowed"),
    [
        ({"choice": "deny", "resolve_all": False}, True),
        ({"choice": "once", "resolve_all": False}, False),
    ],
)
async def test_disabled_target_preserves_only_approval_deny_safety_path(
    postgres_pool: asyncpg.Pool,
    decision: dict[str, object],
    allowed: bool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["once", "deny"]}
    )
    request = await tasks.materialize_interaction_request(interaction.id, "Approve?")
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    await postgres_pool.execute(
        "UPDATE agent_targets SET enabled = false WHERE id = $1", target.target.id
    )
    before = await postgres_pool.fetchval(
        "SELECT count(*) FROM messages WHERE fish_id = $1 AND role = 'user'", fish_id
    )

    if allowed:
        mutation = await tasks.insert_and_bind_approval_response(
            interaction.id,
            fish_id,
            source.session_id,
            "拒绝",
            decision,
        )
        assert mutation.interaction is not None
        assert mutation.interaction.response_message_id == mutation.message.id
    else:
        with pytest.raises(AgentReplyConflictError, match="target"):
            await tasks.insert_and_bind_approval_response(
                interaction.id,
                fish_id,
                source.session_id,
                "允许",
                decision,
            )
        assert (
            await postgres_pool.fetchval(
                "SELECT count(*) FROM messages WHERE fish_id = $1 AND role = 'user'",
                fish_id,
            )
            == before
        )


@pytest.mark.asyncio
async def test_input_reply_preserves_voice_identity_and_replays_without_second_run(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        run.id, "need input", {}
    )
    request = await tasks.materialize_interaction_request(
        interaction.id, "Which branch?"
    )
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    turn_id = uuid4()
    created_at = datetime(2026, 8, 30, 7, 1, tzinfo=UTC)

    first = await tasks.insert_and_resolve_input(
        interaction.id,
        fish_id,
        source.session_id,
        "release/next",
        message_id=turn_id,
        created_at=created_at,
    )
    replay = await tasks.insert_and_resolve_input(
        interaction.id,
        fish_id,
        source.session_id,
        "release/next",
        message_id=turn_id,
        created_at=created_at + timedelta(seconds=5),
    )

    assert first.inserted is True and first.replayed is False
    assert replay.inserted is False and replay.replayed is True
    assert replay.message == first.message
    assert first.message.id == turn_id
    assert first.message.created_at == created_at
    assert first.interaction is not None and first.run is not None
    assert replay.run == first.run
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
        )
        == 2
    )
    with pytest.raises(MessageIdentityConflictError):
        await tasks.insert_and_resolve_input(
            interaction.id,
            fish_id,
            source.session_id,
            "different",
            message_id=turn_id,
            created_at=created_at,
        )


@pytest.mark.asyncio
async def test_result_ordinary_reply_revalidates_tail_and_reports_message_replay(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, run.id)
    turn_id = uuid4()
    created_at = datetime(2026, 8, 30, 7, 2, tzinfo=UTC)

    first = await tasks.insert_validated_ordinary_reply(
        context,
        fish_id,
        source.session_id,
        "thanks",
        message_id=turn_id,
        created_at=created_at,
    )
    replay = await tasks.insert_validated_ordinary_reply(
        context,
        fish_id,
        source.session_id,
        "thanks",
        message_id=turn_id,
        created_at=created_at + timedelta(seconds=5),
    )

    assert first.inserted is True and first.replayed is False
    assert replay.inserted is False and replay.replayed is True
    assert replay.message == first.message

    next_source = await messages.insert_user_message(
        fish_id, "new work", session_id=source.session_id
    )
    await tasks.create_continuation_run(
        task.id, run.id, next_source.id, next_source.content, None
    )
    with pytest.raises(AgentReplyConflictError, match="replyable"):
        await tasks.insert_validated_ordinary_reply(
            context,
            fish_id,
            source.session_id,
            "late acknowledgement",
        )


@pytest.mark.asyncio
async def test_concurrent_result_followups_leave_only_winner_voice_message(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    ids = (uuid4(), uuid4())

    async def follow(content: str, message_id: UUID):
        try:
            return await tasks.insert_and_create_result_followup(
                run.id,
                fish_id,
                source.session_id,
                content,
                "continue",
                message_id=message_id,
                created_at=datetime(2026, 8, 30, 7, 3, tzinfo=UTC),
            )
        except AgentReplyConflictError:
            return None

    one, two = await asyncio.gather(follow("one", ids[0]), follow("two", ids[1]))

    winner = one or two
    assert winner is not None and winner.run is not None
    assert sum(item is not None for item in (one, two)) == 1
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = ANY($1::uuid[])", list(ids)
        )
        == 1
    )
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM agent_runs WHERE task_id = $1", task.id
        )
        == 2
    )


@pytest.mark.asyncio
async def test_same_voice_message_cannot_mutate_two_result_subjects_concurrently(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    first_source = await messages.insert_user_message(fish_id, "first")
    second_source = await messages.insert_user_message(
        fish_id, "second", session_id=first_source.session_id
    )
    target = await configured_target(targets)
    first_task, first_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "first task",
        first_source.id,
        None,
        first_source.content,
    )
    second_task, second_run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "second task",
        second_source.id,
        None,
        second_source.content,
    )
    for run in (first_run, second_run):
        await start_run(tasks, run.id)
        await tasks.finish_run(run.id, "completed", "done", None, None, {})
        result = await tasks.materialize_run_result(run.id, f"Finished {run.id}")
        assert result is not None
        assert (
            await tasks.finalize_agent_message_consumption(
                result.id, fish_id, result.session_id, len(result.content), False
            )
            is not None
        )

    turn_id = uuid4()
    created_at = datetime(2026, 8, 30, 7, 4, tzinfo=UTC)

    async def follow(run_id: UUID):
        try:
            return await tasks.insert_and_create_result_followup(
                run_id,
                fish_id,
                first_source.session_id,
                "continue",
                "continue",
                message_id=turn_id,
                created_at=created_at,
            )
        except AgentReplyConflictError:
            return None

    first, second = await asyncio.gather(
        follow(first_run.id),
        follow(second_run.id),
    )

    assert sum(item is not None for item in (first, second)) == 1
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = $1", turn_id
        )
        == 1
    )
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*)
            FROM agent_runs
            WHERE task_id = ANY($1::uuid[]) AND previous_run_id IS NOT NULL
            """,
            [first_task.id, second_task.id],
        )
        == 1
    )


@pytest.mark.asyncio
async def test_t_confirmed_child_is_not_durable_handled_reply_proof(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, run.id)
    turn_id = uuid4()
    ordinary = await tasks.insert_validated_ordinary_reply(
        context,
        fish_id,
        source.session_id,
        "continue",
        message_id=turn_id,
    )
    confirmation = await messages.insert_generating_answer(
        fish_id,
        source.session_id,
        ordinary.message.sequence,
    )
    child = await tasks.create_continuation_run(
        task.id,
        run.id,
        ordinary.message.id,
        ordinary.message.content,
        confirmation.id,
    )
    assert child.confirmation_message_id == confirmation.id

    replay = await tasks.load_reply_subject(
        context,
        fish_id,
        source.session_id,
        message_id=turn_id,
        content="continue",
    )
    mutation = await tasks.insert_validated_ordinary_reply(
        context,
        fish_id,
        source.session_id,
        "continue",
        message_id=turn_id,
    )

    assert replay.replay is not None and replay.replay.handled is False
    assert mutation.handled is False
    assert mutation.replayed is True


@pytest.mark.asyncio
async def test_ordinary_turn_replays_false_after_other_message_consumes_subject(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, run.id)
    ordinary_turn_id = uuid4()
    ordinary = await tasks.insert_validated_ordinary_reply(
        context,
        fish_id,
        source.session_id,
        "thanks",
        message_id=ordinary_turn_id,
    )
    consumed = await tasks.insert_and_create_result_followup(
        run.id,
        fish_id,
        source.session_id,
        "continue",
        "continue",
        message_id=uuid4(),
    )
    assert consumed.handled is True

    replay = await tasks.load_reply_subject(
        context,
        fish_id,
        source.session_id,
        message_id=ordinary_turn_id,
        content=ordinary.message.content,
    )
    mutation = await tasks.insert_validated_ordinary_reply(
        context,
        fish_id,
        source.session_id,
        ordinary.message.content,
        message_id=ordinary_turn_id,
    )

    assert replay.replay is not None and replay.replay.handled is False
    assert mutation.handled is False
    assert mutation.message == ordinary.message


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["handled", "ordinary"])
async def test_same_subject_final_store_disposition_overrides_losing_action(
    postgres_pool: asyncpg.Pool,
    winner: str,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, run.id)
    turn_id = uuid4()

    if winner == "handled":
        first = await tasks.insert_and_create_result_followup(
            run.id,
            fish_id,
            source.session_id,
            "continue",
            "continue",
            message_id=turn_id,
        )
        second = await tasks.insert_validated_ordinary_reply(
            context,
            fish_id,
            source.session_id,
            "continue",
            message_id=turn_id,
        )
        expected_handled = True
    else:
        first = await tasks.insert_validated_ordinary_reply(
            context,
            fish_id,
            source.session_id,
            "continue",
            message_id=turn_id,
        )
        second = await tasks.insert_and_create_result_followup(
            run.id,
            fish_id,
            source.session_id,
            "continue",
            "continue",
            message_id=turn_id,
        )
        expected_handled = False

    assert first.message == second.message
    assert first.handled is second.handled is expected_handled
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = $1", turn_id
        )
        == 1
    )
    assert await postgres_pool.fetchval(
        """
            SELECT count(*)
            FROM agent_runs
            WHERE task_id = $1 AND previous_run_id = $2
            """,
        task.id,
        run.id,
    ) == (1 if expected_handled else 0)


@pytest.mark.asyncio
async def test_same_subject_concurrent_actions_return_one_store_disposition(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Finished")
    assert result is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            result.id, fish_id, result.session_id, len(result.content), False
        )
        is not None
    )
    context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, run.id)
    turn_id = uuid4()

    handled, ordinary = await asyncio.gather(
        tasks.insert_and_create_result_followup(
            run.id,
            fish_id,
            source.session_id,
            "continue",
            "continue",
            message_id=turn_id,
        ),
        tasks.insert_validated_ordinary_reply(
            context,
            fish_id,
            source.session_id,
            "continue",
            message_id=turn_id,
        ),
    )

    assert handled.message == ordinary.message
    assert handled.handled is ordinary.handled
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = $1", turn_id
        )
        == 1
    )


@pytest.mark.asyncio
async def test_wrong_context_rejects_durable_direct_interaction_and_result_replies(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    first_source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)

    interactions = []
    for label in ("approval-a", "approval-b"):
        source = await messages.insert_user_message(
            fish_id, label, session_id=first_source.session_id
        )
        _task, run = await tasks.create_task_with_first_run(
            fish_id, target.target.id, label, source.id, None, source.content
        )
        await start_run(tasks, run.id)
        interaction = await tasks.open_approval_interaction(
            run.id, {"allowed_decisions": ["deny"]}
        )
        request = await tasks.materialize_interaction_request(
            interaction.id, f"Deny {label}?"
        )
        assert request is not None
        assert (
            await tasks.finalize_agent_message_consumption(
                request.id, fish_id, request.session_id, len(request.content), False
            )
            is not None
        )
        interactions.append(interaction)

    interaction_turn_id = uuid4()
    await tasks.insert_and_bind_approval_response(
        interactions[0].id,
        fish_id,
        first_source.session_id,
        "deny",
        "deny",
        message_id=interaction_turn_id,
    )
    with pytest.raises(AgentReplyConflictError, match="another Agent subject"):
        await tasks.load_reply_subject(
            AgentReplyContext(
                AgentMessageSubjectKind.INTERACTION,
                interactions[1].id,
            ),
            fish_id,
            first_source.session_id,
            message_id=interaction_turn_id,
            content="deny",
        )
    with pytest.raises(AgentReplyConflictError, match="another Agent subject"):
        await tasks.insert_and_bind_approval_response(
            interactions[1].id,
            fish_id,
            first_source.session_id,
            "deny",
            "deny",
            message_id=interaction_turn_id,
        )

    results = []
    for label in ("result-a", "result-b"):
        source = await messages.insert_user_message(
            fish_id, label, session_id=first_source.session_id
        )
        _task, run = await tasks.create_task_with_first_run(
            fish_id, target.target.id, label, source.id, None, source.content
        )
        await start_run(tasks, run.id)
        await tasks.finish_run(run.id, "completed", "done", None, None, {})
        result = await tasks.materialize_run_result(run.id, f"Finished {label}")
        assert result is not None
        assert (
            await tasks.finalize_agent_message_consumption(
                result.id, fish_id, result.session_id, len(result.content), False
            )
            is not None
        )
        results.append(run)

    result_turn_id = uuid4()
    await tasks.insert_and_create_result_followup(
        results[0].id,
        fish_id,
        first_source.session_id,
        "continue",
        "continue",
        message_id=result_turn_id,
    )
    with pytest.raises(AgentReplyConflictError, match="another Agent subject"):
        await tasks.load_reply_subject(
            AgentReplyContext(
                AgentMessageSubjectKind.RUN_RESULT,
                results[1].id,
            ),
            fish_id,
            first_source.session_id,
            message_id=result_turn_id,
            content="continue",
        )
    with pytest.raises(AgentReplyConflictError, match="another Agent subject"):
        await tasks.insert_and_create_result_followup(
            results[1].id,
            fish_id,
            first_source.session_id,
            "continue",
            "continue",
            message_id=result_turn_id,
        )


@pytest.mark.asyncio
async def test_direct_association_folds_only_exact_resolved_input_child(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        run.id, "need input", {}
    )
    request = await tasks.materialize_interaction_request(
        interaction.id, "Which branch?"
    )
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    turn_id = uuid4()
    mutation = await tasks.insert_and_resolve_input(
        interaction.id,
        fish_id,
        source.session_id,
        "release/next",
        message_id=turn_id,
    )

    replay = await tasks.load_reply_subject(
        AgentReplyContext(AgentMessageSubjectKind.INTERACTION, interaction.id),
        fish_id,
        source.session_id,
        message_id=turn_id,
        content="release/next",
    )

    assert mutation.run is not None
    assert mutation.interaction is not None
    assert mutation.interaction.next_run_id == mutation.run.id
    assert mutation.interaction.resolved_at is not None
    assert mutation.interaction.closed_at is None
    assert replay.replay is not None and replay.replay.handled is True


@pytest.mark.asyncio
async def test_approval_response_plus_direct_child_is_ambiguous_not_folded(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    interaction = await tasks.open_approval_interaction(
        run.id, {"allowed_decisions": ["deny"]}
    )
    request = await tasks.materialize_interaction_request(interaction.id, "Deny?")
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    turn_id = uuid4()
    response = await tasks.insert_and_bind_approval_response(
        interaction.id,
        fish_id,
        source.session_id,
        "deny",
        "deny",
        message_id=turn_id,
    )
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    child = await tasks.create_continuation_run(
        task.id,
        run.id,
        response.message.id,
        response.message.content,
        None,
    )
    assert child.confirmation_message_id is None

    with pytest.raises(AgentReplyConflictError, match="ambiguous direct"):
        await tasks.load_reply_subject(
            AgentReplyContext(AgentMessageSubjectKind.INTERACTION, interaction.id),
            fish_id,
            source.session_id,
            message_id=turn_id,
            content="deny",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_state", ["next_run_mismatch", "unresolved"])
async def test_invalid_input_child_association_is_ambiguous_not_folded(
    postgres_pool: asyncpg.Pool,
    invalid_state: str,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    target = await configured_target(targets)
    _task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    _completed, interaction = await tasks.complete_run_requiring_input(
        run.id, "need input", {}
    )
    request = await tasks.materialize_interaction_request(
        interaction.id, "Which branch?"
    )
    assert request is not None
    assert (
        await tasks.finalize_agent_message_consumption(
            request.id, fish_id, request.session_id, len(request.content), False
        )
        is not None
    )
    turn_id = uuid4()
    await tasks.insert_and_resolve_input(
        interaction.id,
        fish_id,
        source.session_id,
        "release/next",
        message_id=turn_id,
    )
    if invalid_state == "next_run_mismatch":
        unrelated_source = await messages.insert_user_message(
            fish_id, "unrelated", session_id=source.session_id
        )
        _unrelated_task, unrelated_run = await tasks.create_task_with_first_run(
            fish_id,
            target.target.id,
            "unrelated",
            unrelated_source.id,
            None,
            unrelated_source.content,
        )
        await postgres_pool.execute(
            "UPDATE agent_interactions SET next_run_id = $2 WHERE id = $1",
            interaction.id,
            unrelated_run.id,
        )
    else:
        await postgres_pool.execute(
            """
            UPDATE agent_interactions
            SET next_run_id = NULL, resolved_at = NULL
            WHERE id = $1
            """,
            interaction.id,
        )

    with pytest.raises(AgentReplyConflictError, match="ambiguous direct"):
        await tasks.load_reply_subject(
            AgentReplyContext(AgentMessageSubjectKind.INTERACTION, interaction.id),
            fish_id,
            source.session_id,
            message_id=turn_id,
            content="release/next",
        )


@pytest.mark.asyncio
async def test_provider_snapshots_preserve_db_current_steer_through_terminal_states(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "first")
    steer_source = await messages.insert_user_message(
        fish_id, "change", session_id=source.session_id
    )
    target = await configured_target(targets)
    task, run = await tasks.create_task_with_first_run(
        fish_id, target.target.id, "task", source.id, None, source.content
    )
    await start_run(tasks, run.id)
    assert await tasks.queue_running_steer(run.id, steer_source.id) is not None

    running = await tasks.update_active_run(run.id, "running", None)
    assert running is not None and running.provider_data is not None
    assert mapping_field(running.provider_data, "steer")["state"] == "queued"
    stopping = await tasks.update_active_run(
        run.id, "running", {"provider_status": "stopping"}
    )
    assert stopping is not None and stopping.provider_data is not None
    assert mapping_field(stopping.provider_data, "steer")["state"] == "queued"
    assert stopping.provider_data["provider_status"] == "stopping"
    unknown = await tasks.mark_run_unknown(
        run.id,
        "provider_error",
        "missing",
        {"provider_status": "missing"},
    )
    assert unknown is not None and unknown.provider_data is not None
    assert mapping_field(unknown.provider_data, "steer")["state"] == "queued"
    assert unknown.provider_data["provider_status"] == "missing"

    incoming = {
        "padding": "x" * 65_000,
        "pending_steer": "provider-next",
    }
    finished = await tasks.finish_run(run.id, "completed", "done", None, None, incoming)
    assert finished.provider_data is not None
    assert mapping_field(finished.provider_data, "steer")["state"] == "queued"
    assert finished.provider_data["pending_steer"] == "provider-next"
    assert canonical_json_bytes(finished.provider_data) <= 65_536

    followup_source = await messages.insert_user_message(
        fish_id, "next", session_id=source.session_id
    )
    followup = await tasks.create_continuation_run(
        task.id, run.id, followup_source.id, followup_source.content, None
    )
    await start_run(tasks, followup.id)
    assert await tasks.queue_running_steer(followup.id, steer_source.id) is not None
    completed, _interaction = await tasks.complete_run_requiring_input(
        followup.id,
        "which target?",
        incoming,
    )
    assert completed.provider_data is not None
    assert mapping_field(completed.provider_data, "steer")["state"] == "queued"
    assert completed.provider_data["pending_steer"] == "provider-next"
    assert canonical_json_bytes(completed.provider_data) <= 65_536
