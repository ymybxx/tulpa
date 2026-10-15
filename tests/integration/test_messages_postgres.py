import asyncio
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import asyncpg
import pytest

from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.voice_draft_types import voice_answer_message_id
from fish_memory.db.messages import MessageIdentityConflictError, MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.models import MessageStatus
from fish_memory.performance.tracker import MessagePerformanceTracker

NOW = datetime(2026, 8, 9, 13, tzinfo=UTC)


@pytest.mark.asyncio
async def test_explicit_voice_ids_are_idempotent(postgres_pool: asyncpg.Pool) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    fish_id = uuid4()
    user_id = uuid4()

    first = await store.insert_user_message(fish_id, "问题", message_id=user_id)
    second = await store.insert_user_message(
        fish_id,
        "问题",
        session_id=first.session_id,
        message_id=user_id,
    )
    answer_id = voice_answer_message_id(user_id)
    answer = await store.insert_generating_answer(
        fish_id,
        first.session_id,
        first.sequence,
        message_id=answer_id,
    )
    repeated = await store.insert_generating_answer(
        fish_id,
        first.session_id,
        first.sequence,
        message_id=answer_id,
    )

    assert (second.id, second.sequence) == (first.id, first.sequence)
    assert repeated.id == answer.id

    with pytest.raises(MessageIdentityConflictError, match=str(user_id)):
        await store.insert_user_message(
            fish_id,
            "不同的问题",
            session_id=first.session_id,
            message_id=user_id,
        )
    with pytest.raises(MessageIdentityConflictError, match=str(answer_id)):
        await store.insert_generating_answer(
            fish_id,
            first.session_id,
            first.sequence + 1,
            message_id=answer_id,
        )


@pytest.mark.asyncio
async def test_messages_are_isolated_and_progress_only_moves_forward(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    first_fish = uuid4()
    second_fish = uuid4()

    first_user = await store.insert_user_message(first_fish, "first")
    second_user = await store.insert_user_message(second_fish, "other fish")
    latest_user = await store.insert_user_message(first_fish, "latest")

    first_history = await store.load_history(
        first_fish,
        latest_user.session_id,
        before_sequence=latest_user.sequence + 1,
    )

    assert [message.content for message in first_history] == ["first", "latest"]
    assert second_user.sequence not in {message.sequence for message in first_history}
    assert len({first_user.sequence, second_user.sequence, latest_user.sequence}) == 3

    answer = await store.insert_generating_answer(
        first_fish,
        latest_user.session_id,
        latest_user.sequence,
    )
    await store.finish_answer(
        answer.id,
        first_fish,
        content="0123456789",
        requested_status=MessageStatus.COMPLETED,
    )
    offsets = await asyncio.gather(
        store.update_consumed_offset(answer.id, first_fish, 7),
        store.update_consumed_offset(answer.id, first_fish, 3),
        store.update_consumed_offset(answer.id, first_fish, 9),
    )
    with pytest.raises(ValueError, match="content length"):
        await store.update_consumed_offset(answer.id, first_fish, 11)

    assert max(offset for offset in offsets if offset is not None) == 9
    async with postgres_pool.acquire() as connection:
        stored_offset = await connection.fetchval(
            "SELECT consumed_offset FROM messages WHERE id = $1",
            answer.id,
        )
        session_last_message_at = await connection.fetchval(
            "SELECT last_message_at FROM fish_sessions WHERE id = $1",
            latest_user.session_id,
        )
    assert stored_offset == 9
    assert session_last_message_at == answer.created_at


@pytest.mark.asyncio
async def test_history_token_window_shrinks_to_retain_limit_at_message_boundaries(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(
        postgres_pool,
        history_token_limit=40,
        history_token_retain=20,
    )
    fish_id = uuid4()

    messages = [await store.insert_user_message(fish_id, "aaa") for _ in range(4)]
    history = await store.load_history(
        fish_id,
        messages[-1].session_id,
        before_sequence=messages[-1].sequence + 1,
    )

    assert [message.id for message in history] == [messages[-1].id]
    async with postgres_pool.acquire() as connection:
        window = await connection.fetchrow(
            """
            SELECT history_window_start_sequence, history_window_token_count
            FROM fish_sessions
            WHERE id = $1
            """,
            messages[-1].session_id,
        )
    assert window is not None
    assert window["history_window_start_sequence"] == messages[-1].sequence
    assert window["history_window_token_count"] == 13


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reset_uninitialized_window",
    [True, False],
    ids=["uninitialized-window", "initialized-window-rebuild"],
)
async def test_draft_context_matches_real_committed_retention_window(
    postgres_pool: asyncpg.Pool,
    reset_uninitialized_window: bool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(
        postgres_pool,
        history_token_limit=40,
        history_token_retain=20,
    )
    fish_id = uuid4()
    first = await store.insert_user_message(fish_id, "old one")
    await store.insert_user_message(
        fish_id,
        "old two",
        session_id=first.session_id,
    )
    if reset_uninitialized_window:
        await postgres_pool.execute(
            """
            UPDATE fish_sessions
            SET history_window_start_sequence = 0,
                history_window_token_count = 0
            WHERE id = $1
            """,
            first.session_id,
        )

    loader = AnswerContextLoader(
        store,
        now=lambda: NOW,
        system_prompt="base prompt",
    )
    draft = await loader.load_for_draft(
        fish_id,
        first.session_id,
        content="abcdefghijkl",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )
    committed_user = await store.insert_user_message(
        fish_id,
        "abcdefghijkl",
        session_id=first.session_id,
        created_at=NOW,
    )
    committed = await loader.load_for_generation(
        fish_id,
        first.session_id,
        through_sequence=committed_user.sequence,
        tracker=MessagePerformanceTracker(),
    )

    assert (
        draft.messages
        == committed.messages
        == [
            {"role": "system", "content": "base prompt"},
            {
                "role": "user",
                "content": (
                    "<message_time>2026-08-09 13:00:00+00:00</message_time>\n"
                    "abcdefghijkl"
                ),
            },
        ]
    )
    assert draft.recall_conversation == committed.recall_conversation == ()


@pytest.mark.asyncio
async def test_timeline_joins_sessions_and_keeps_the_answer_being_generated(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    fish_id = uuid4()
    other_fish = uuid4()
    first = await store.insert_user_message(fish_id, "上周的话")
    await store.insert_user_message(other_fish, "别的角色")
    later_session = uuid4()
    async with postgres_pool.acquire() as connection:
        await connection.execute(
            "INSERT INTO fish_sessions (id, fish_id, created_at) "
            "VALUES ($1, $2, now())",
            later_session,
            fish_id,
        )
    second = await store.insert_user_message(
        fish_id, "今天的话", session_id=later_session
    )
    answer = await store.insert_generating_answer(
        fish_id, later_session, second.sequence
    )

    page = await store.list_timeline_page(fish_id, limit=10)
    assert [message.content for message in page.items[:2]] == ["上周的话", "今天的话"]
    assert page.items[-1].id == answer.id
    assert page.items[-1].status == MessageStatus.GENERATING
    assert not page.has_before

    older = await store.list_timeline_page(fish_id, limit=1, before=second.sequence)
    assert [message.id for message in older.items] == [first.id]
    newer = await store.list_timeline_page(fish_id, limit=1, after=first.sequence)
    assert [message.id for message in newer.items] == [second.id]
    assert newer.has_after


@pytest.mark.asyncio
async def test_search_matches_literally_and_skips_unfinished_answers(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    fish_id = uuid4()
    percent = await store.insert_user_message(fish_id, "进度到 50% 了")
    plain = await store.insert_user_message(fish_id, "进度到 50 了")
    answer = await store.insert_generating_answer(
        fish_id, plain.session_id, plain.sequence
    )
    await store.finish_answer(
        answer.id,
        fish_id,
        content="Build 报错了",
        requested_status=MessageStatus.COMPLETED,
    )
    thinking = await store.insert_user_message(fish_id, "再看看报错")
    await store.insert_generating_answer(
        fish_id, thinking.session_id, thinking.sequence
    )

    assert [m.id for m in await store.search_messages(fish_id, "50%", limit=10)] == [
        percent.id
    ]
    assert [
        m.content for m in await store.search_messages(fish_id, "build", limit=10)
    ] == ["Build 报错了"]
    assert [
        m.content for m in await store.search_messages(fish_id, "报错", limit=10)
    ] == [
        "再看看报错",
        "Build 报错了",
    ]


@pytest.mark.asyncio
async def test_input_modes_are_kept_per_message_and_first_record_wins(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    fish_id = uuid4()
    typed = await store.insert_user_message(fish_id, "打的字")
    spoken = await store.insert_user_message(fish_id, "说的话")
    await store.record_input_mode(typed.id, "typed")
    await store.record_input_mode(spoken.id, "spoken")
    await store.record_input_mode(spoken.id, "typed")

    assert await store.input_modes(fish_id, [typed.id, spoken.id]) == {
        typed.id: "typed",
        spoken.id: "spoken",
    }
    assert await store.input_modes(uuid4(), [typed.id]) == {}


@pytest.mark.asyncio
async def test_an_interrupted_answer_keeps_only_what_was_said_and_its_rest_apart(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    fish_id = uuid4()
    question = await store.insert_user_message(fish_id, "讲个故事")
    answer = await store.insert_generating_answer(
        fish_id, question.session_id, question.sequence
    )
    await store.finish_answer(
        answer.id,
        fish_id,
        content="小橘猫迷了路。它沿着河走，找到了家。",
        requested_status=MessageStatus.COMPLETED,
    )

    assert await store.interrupt_terminal_answer(answer.id, fish_id, 7) == 7

    page = await store.list_timeline_page(fish_id, limit=10)
    # TA's own history holds only what was said ...
    assert page.items[-1].content == "小橘猫迷了路。"
    # ... and the chat page can still show the rest struck through.
    assert await store.unsaid_texts(fish_id, [answer.id]) == {
        answer.id: "它沿着河走，找到了家。"
    }
    assert await store.unsaid_texts(uuid4(), [answer.id]) == {}


@pytest.mark.asyncio
async def test_a_reply_keeps_which_service_let_it_down(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = MessagesStore(postgres_pool)
    fish_id = uuid4()
    source = await store.insert_user_message(fish_id, "问题")
    failed = await store.insert_generating_answer(
        fish_id, source.session_id, source.sequence
    )
    await store.finish_answer(
        failed.id,
        fish_id,
        content="",
        requested_status=MessageStatus.FAILED,
        failure="jev:timeout",
    )
    second = await store.insert_user_message(
        fish_id, "再问", session_id=source.session_id
    )
    unspoken = await store.insert_generating_answer(
        fish_id, source.session_id, second.sequence
    )
    await store.finish_answer(
        unspoken.id,
        fish_id,
        content="helello.",
        requested_status=MessageStatus.COMPLETED,
    )
    assert await store.record_failure(unspoken.id, fish_id, "voice:rejected")
    # The first reason stays.
    assert not await store.record_failure(unspoken.id, fish_id, "voice:error")
    async with postgres_pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT id, failure FROM messages WHERE role = 'assistant'"
            " AND fish_id = $1",
            fish_id,
        )
    assert {row["id"]: row["failure"] for row in rows} == {
        failed.id: "jev:timeout",
        unspoken.id: "voice:rejected",
    }
    page = await store.list_timeline_page(fish_id, limit=10)
    assert {m.id: m.failure for m in page.items if m.role.value == "assistant"} == {
        failed.id: "jev:timeout",
        unspoken.id: "voice:rejected",
    }
