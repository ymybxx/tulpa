from datetime import UTC, datetime
from uuid import uuid4

import asyncpg
import pytest

from fish_memory.models import MessageStatus
from tests.integration.test_agents_postgres import (
    configured_target,
    migrated_stores,
    start_run,
)


@pytest.mark.asyncio
async def test_recent_agent_reply_is_bounded_to_five_assistant_messages_and_session(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "find somewhere")
    session_id = source.session_id
    target = await configured_target(targets)
    confirmation = await messages.insert_generating_answer(
        fish_id,
        session_id,
        source.sequence,
    )
    assert (
        await messages.finish_answer(
            confirmation.id,
            fish_id,
            content="I will look into it",
            requested_status=MessageStatus.COMPLETED,
        )
        is MessageStatus.COMPLETED
    )
    _task, run = await tasks.create_task_with_first_run(
        fish_id,
        target.target.id,
        "find somewhere",
        source.id,
        confirmation.id,
        source.content,
    )
    confirmation_followup = await messages.insert_user_message(
        fish_id,
        "anything else?",
        session_id=session_id,
    )
    assert await tasks.has_recent_agent_reply(
        fish_id,
        session_id,
        before_sequence=confirmation_followup.sequence,
        limit=5,
    )
    await start_run(tasks, run.id)
    await tasks.finish_run(run.id, "completed", "done", None, None, {})
    result = await tasks.materialize_run_result(run.id, "Agent result")
    assert result is not None

    current = await messages.insert_user_message(
        fish_id,
        "one more requirement",
        session_id=session_id,
    )
    assert await tasks.has_recent_agent_reply(
        fish_id,
        session_id,
        before_sequence=current.sequence,
        limit=5,
    )

    other_session_id = uuid4()
    await postgres_pool.execute(
        "INSERT INTO fish_sessions (id, fish_id) VALUES ($1, $2)",
        other_session_id,
        fish_id,
    )
    other_session = await messages.insert_user_message(
        fish_id,
        "other session",
        session_id=other_session_id,
    )
    assert not await tasks.has_recent_agent_reply(
        fish_id,
        other_session.session_id,
        before_sequence=None,
        limit=5,
    )

    for index in range(5):
        filler = await messages.insert_user_message(
            fish_id,
            f"filler {index}",
            session_id=session_id,
        )
        answer = await messages.insert_generating_answer(
            fish_id,
            session_id,
            filler.sequence,
        )
        assert (
            await messages.finish_answer(
                answer.id,
                fish_id,
                content=f"ordinary answer {index}",
                requested_status=MessageStatus.COMPLETED,
            )
            is MessageStatus.COMPLETED
        )

    after_five = await messages.insert_user_message(
        fish_id,
        "now unrelated",
        session_id=session_id,
    )
    assert not await tasks.has_recent_agent_reply(
        fish_id,
        session_id,
        before_sequence=after_five.sequence,
        limit=5,
    )


@pytest.mark.asyncio
async def test_supported_connector_filter_runs_before_candidate_limit(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    hermes_source = await messages.insert_user_message(fish_id, "Hermes task")
    session_id = hermes_source.session_id
    hermes_target = await configured_target(targets, connector_type="hermes")
    fake_target = await configured_target(
        targets,
        connector_type="fake.valid",
        external_key="fake",
    )
    hermes_task, hermes_run = await tasks.create_task_with_first_run(
        fish_id,
        hermes_target.target.id,
        "Older supported task",
        hermes_source.id,
        None,
        hermes_source.content,
    )
    await postgres_pool.execute(
        "UPDATE agent_runs SET updated_at = $2 WHERE id = $1",
        hermes_run.id,
        datetime(2026, 8, 31, 1, tzinfo=UTC),
    )

    for index in range(8):
        source = await messages.insert_user_message(
            fish_id,
            f"unsupported {index}",
            session_id=session_id,
        )
        _task, run = await tasks.create_task_with_first_run(
            fish_id,
            fake_target.target.id,
            f"Newer unsupported task {index}",
            source.id,
            None,
            source.content,
        )
        await postgres_pool.execute(
            "UPDATE agent_runs SET updated_at = $2 WHERE id = $1",
            run.id,
            datetime(2026, 8, 31, 2, index, tzinfo=UTC),
        )

    candidates = await tasks.list_task_candidates(
        fish_id,
        session_id,
        limit=8,
        supported_connector_types=frozenset({"hermes"}),
    )

    assert [candidate.task.id for candidate in candidates] == [hermes_task.id]


@pytest.mark.asyncio
async def test_interaction_focus_keeps_recent_or_latest_pending_in_current_session(
    postgres_pool: asyncpg.Pool,
) -> None:
    messages, targets, tasks = await migrated_stores(postgres_pool)
    fish_id = uuid4()
    target = await configured_target(targets)
    initial = await messages.insert_user_message(fish_id, "start")
    session_id = initial.session_id

    async def open_input(title: str, request_text: str):
        source = await messages.insert_user_message(
            fish_id,
            title,
            session_id=session_id,
        )
        task, run = await tasks.create_task_with_first_run(
            fish_id,
            target.target.id,
            title,
            source.id,
            None,
            source.content,
        )
        await start_run(tasks, run.id)
        _completed, interaction = await tasks.complete_run_requiring_input(
            run.id,
            request_text,
            {},
        )
        request = await tasks.materialize_interaction_request(
            interaction.id,
            request_text,
        )
        assert request is not None
        return task, interaction, request

    old_task, old_interaction, _old_request = await open_input(
        "old pending",
        "Old question?",
    )
    await postgres_pool.execute(
        "UPDATE agent_interactions SET created_at = now() - interval '2 hours' "
        "WHERE id = $1",
        old_interaction.id,
    )

    latest_only = await tasks.list_task_candidates(fish_id, session_id)
    old_candidate = next(item for item in latest_only if item.task.id == old_task.id)
    assert old_candidate.interaction is not None
    assert old_candidate.interaction_request_content == "Old question?"
    assert old_candidate.interaction_request_session_id == session_id

    newer_task, newer_interaction, newer_request = await open_input(
        "newer resolved",
        "Newer question?",
    )
    await tasks.finalize_agent_message_consumption(
        newer_request.id,
        fish_id,
        newer_request.session_id,
        len(newer_request.content),
        False,
    )
    await tasks.insert_and_resolve_input(
        newer_interaction.id,
        fish_id,
        session_id,
        "answered",
    )
    await postgres_pool.execute(
        "UPDATE agent_interactions SET created_at = now() - interval '90 minutes' "
        "WHERE id = $1",
        newer_interaction.id,
    )

    after_newer_resolved = await tasks.list_task_candidates(fish_id, session_id)
    assert old_task.id not in {item.task.id for item in after_newer_resolved}
    assert newer_task.id in {item.task.id for item in after_newer_resolved}

    recent_task, recent_interaction, recent_request = await open_input(
        "recent pending",
        "Recent question?",
    )
    await postgres_pool.execute(
        "UPDATE agent_interactions SET created_at = now() - interval '30 minutes' "
        "WHERE id = $1",
        recent_interaction.id,
    )

    with_recent = await tasks.list_task_candidates(fish_id, session_id)
    assert recent_task.id in {item.task.id for item in with_recent}
    assert await tasks.has_recent_agent_reply(
        fish_id,
        session_id,
        before_sequence=recent_request.sequence + 1,
        limit=5,
    )
