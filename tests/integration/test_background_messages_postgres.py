from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import asyncpg
import pytest

from fish_memory.background_messages.models import (
    BackgroundMessageInterruptPolicy,
    BackgroundMessageSource,
)
from fish_memory.background_messages.processor import BackgroundMessageExpiryProcessor
from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.models import MessageStatus
from fish_memory.observations.service import ObservationService
from fish_memory.observations.store import ObservationStore


async def stores(
    pool: asyncpg.Pool,
    *,
    now: datetime,
) -> tuple[MessagesStore, BackgroundMessageStore, ObservationService]:
    await run_migrations(cast(MigrationPool, pool))
    messages = MessagesStore(pool, now=lambda: now)
    background = BackgroundMessageStore(
        pool,
        messages_store=messages,
        now=lambda: now,
    )
    observations = ObservationService(ObservationStore(pool))
    return messages, background, observations


async def test_codex_notification_receipt_is_idempotent_without_task(
    postgres_pool: asyncpg.Pool,
) -> None:
    now = datetime.now(UTC)
    messages, background, _ = await stores(postgres_pool, now=now)
    fish_id, notification_id = uuid4(), uuid4()
    source = await messages.insert_user_message(fish_id, "你好", created_at=now)
    for _ in range(2):
        message = await background.materialize_proactive_observation(
            fish_id,
            source.session_id,
            "Codex 那边有结果了。",
            emotion="happy",
            expires_at=now + timedelta(hours=1),
            supersession_key=f"codex-completion:{notification_id}",
            message_id=notification_id,
        )
        assert message.id == notification_id
    assert await background.has_delivery(notification_id)
    assert await postgres_pool.fetchval("SELECT count(*) FROM agent_tasks") == 0
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id=$1", notification_id
        )
        == 1
    )
    delivered = await background.finalize_consumption(
        notification_id, fish_id, source.session_id, -1, False
    )
    assert delivered is not None
    assert delivered.content == message.content
    assert delivered.consumed_offset == -1
    assert not await background.list_incomplete_messages_for_delivery(
        fish_id, source.session_id, 0, message.sequence, 100
    )


async def test_startup_drops_old_codex_queue_preserving_heard_history_and_other_events(
    postgres_pool: asyncpg.Pool,
) -> None:
    now = datetime.now(UTC)
    messages, background, _ = await stores(postgres_pool, now=now)
    fish = uuid4()
    user = await messages.insert_user_message(fish, "你好", created_at=now)
    queued = []
    for key in (
        "codex-completion:unplayed",
        "codex-completion:partial",
        "codex-completion:played",
        "other-timed-event",
    ):
        queued.append(
            await background.materialize_proactive_observation(
                fish,
                user.session_id,
                "已听到。未播放的尾部",
                emotion="neutral",
                expires_at=now + timedelta(hours=1),
                supersession_key=key,
            )
        )
    unplayed, partial, played, unrelated = queued
    await messages.update_consumed_offset(partial.id, fish, 4)
    await background.finalize_consumption(played.id, fish, user.session_id, -1, False)

    assert await background.cancel_orphaned_proactive_streams() == 2
    assert await background.cancel_orphaned_proactive_streams() == 0
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id=$1", unplayed.id
        )
        == 0
    )
    async with postgres_pool.acquire() as connection:
        preserved = await messages.get_message_on_connection(connection, partial.id)
        assert (
            await messages.get_message_on_connection(connection, played.id) is not None
        )
    assert preserved is not None and preserved.content == "已听到。"
    assert preserved.status is MessageStatus.INTERRUPTED
    pending = await background.list_incomplete_messages_for_delivery(
        fish, user.session_id, 0, unrelated.sequence, 100
    )
    assert [m.message_id for m in pending] == [unrelated.id]


@pytest.mark.asyncio
async def test_proactive_message_is_independent_from_agent_and_consumption_gated(
    postgres_pool: asyncpg.Pool,
) -> None:
    now = datetime.now(UTC)
    messages, background, observations = await stores(postgres_pool, now=now)
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "正在写代码", created_at=now)
    observation = await observations.record_important_event(
        fish_id,
        source="accessibility",
        observation_type="activity_change",
        semantic_key="tests-passing",
        summary="测试通过",
        occurred_at=now,
    )
    assert observation is not None

    proactive = await background.materialize_proactive_observation(
        fish_id,
        source.session_id,
        "测试终于都通过了。",
        emotion="happy",
        expires_at=now + timedelta(minutes=1),
        supersession_key="desktop-attention",
        source_observation_id=observation.id,
        created_at=now + timedelta(milliseconds=1),
    )
    replay = await background.materialize_proactive(
        fish_id,
        source.session_id,
        observation.id,
        "重试不应重复投递",
        "sad",
        expires_at=now + timedelta(minutes=2),
        supersession_key="different-retry-key",
    )
    assert replay is not None and replay.id == proactive.id
    assert (
        await postgres_pool.fetchval(
            """
            SELECT count(*)
            FROM background_assistant_deliveries
            WHERE source_observation_id = $1
            """,
            observation.id,
        )
        == 1
    )

    counts = await postgres_pool.fetchrow(
        """
        SELECT
            (SELECT count(*) FROM agent_tasks) AS tasks,
            (SELECT count(*) FROM agent_runs) AS runs,
            (SELECT count(*) FROM agent_interactions) AS interactions
        """
    )
    assert counts is not None
    assert tuple(counts.values()) == (0, 0, 0)
    assert (
        await postgres_pool.fetchval(
            "SELECT fact_extraction_status FROM messages WHERE id = $1",
            proactive.id,
        )
        == "waiting_consumption"
    )
    history = await messages.load_history(
        fish_id,
        source.session_id,
        before_sequence=proactive.sequence + 1,
    )
    assert proactive.id not in {item.id for item in history}

    delivered = await background.list_messages_for_delivery(
        fish_id,
        source.session_id,
        after_sequence=source.sequence,
        include_incomplete=False,
        limit=10,
    )
    assert len(delivered) == 1
    envelope = delivered[0]
    assert envelope.source is BackgroundMessageSource.PROACTIVE_OBSERVATION
    assert envelope.emotion == "happy"
    assert envelope.interrupt_policy is BackgroundMessageInterruptPolicy.DROP
    assert envelope.reply_context is None

    assert (
        await background.finalize_consumption(
            proactive.id,
            uuid4(),
            source.session_id,
            2,
            True,
        )
        is None
    )
    interrupted = await background.finalize_consumption(
        proactive.id,
        fish_id,
        source.session_id,
        2,
        True,
    )
    assert interrupted is not None
    assert interrupted.status is MessageStatus.INTERRUPTED
    assert interrupted.content == "测试"
    assert interrupted.consumed_offset == 2
    assert (
        await postgres_pool.fetchval(
            "SELECT fact_extraction_status FROM messages WHERE id = $1",
            proactive.id,
        )
        == "pending"
    )


@pytest.mark.asyncio
async def test_same_key_messages_wait_for_companion_then_expire_after_grace(
    postgres_pool: asyncpg.Pool,
) -> None:
    current = [datetime.now(UTC)]
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, now=lambda: current[0])
    background = BackgroundMessageStore(
        postgres_pool,
        messages_store=messages,
        now=lambda: current[0],
    )
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "source")
    first = await background.materialize_proactive_observation(
        fish_id,
        source.session_id,
        "旧观察",
        emotion="neutral",
        expires_at=current[0] + timedelta(minutes=1),
        supersession_key="active-window",
    )
    second = await background.materialize_proactive_observation(
        fish_id,
        source.session_id,
        "新观察",
        emotion="neutral",
        expires_at=current[0] + timedelta(minutes=1),
        supersession_key="active-window",
        created_at=current[0] + timedelta(milliseconds=1),
    )

    active = await background.list_messages_for_delivery(
        fish_id,
        source.session_id,
        after_sequence=source.sequence,
        include_incomplete=False,
        limit=10,
    )
    assert [item.message_id for item in active] == [first.id, second.id]

    current[0] += timedelta(minutes=7)
    processor = BackgroundMessageExpiryProcessor(background)
    assert await processor.process_once() is True
    expired_count = await postgres_pool.fetchval(
        """
        SELECT (SELECT count(*) FROM messages WHERE id = $1)
             + (SELECT count(*)
                FROM background_assistant_deliveries
                WHERE message_id = $1)
        """,
        second.id,
    )
    assert expired_count == 0
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = $1",
            first.id,
        )
        == 0
    )
    assert await processor.process_once() is False
    assert (
        await postgres_pool.fetchval(
            "SELECT last_message_at FROM fish_sessions WHERE id = $1",
            source.session_id,
        )
        == source.created_at
    )


@pytest.mark.asyncio
async def test_companion_supersession_keeps_only_spoken_prefix(
    postgres_pool: asyncpg.Pool,
) -> None:
    now = datetime.now(UTC)
    await run_migrations(cast(MigrationPool, postgres_pool))
    messages = MessagesStore(postgres_pool, now=lambda: now)
    background = BackgroundMessageStore(
        postgres_pool,
        messages_store=messages,
        now=lambda: now,
    )
    fish_id = uuid4()
    source = await messages.insert_user_message(fish_id, "source")
    first = await background.materialize_proactive_observation(
        fish_id,
        source.session_id,
        "已经说出的前缀",
        emotion="neutral",
        expires_at=now + timedelta(minutes=1),
        supersession_key="partial",
    )
    partial = await background.finalize_consumption(
        first.id,
        fish_id,
        source.session_id,
        4,
        False,
    )
    assert partial is not None and partial.consumed_offset == 4

    replacement = await background.materialize_proactive_observation(
        fish_id,
        source.session_id,
        "替代消息",
        emotion="neutral",
        expires_at=now + timedelta(minutes=1),
        supersession_key="partial",
        created_at=now + timedelta(milliseconds=1),
    )
    interrupted = await background.finalize_consumption(
        first.id,
        fish_id,
        source.session_id,
        4,
        True,
    )
    assert interrupted is not None

    persisted = await postgres_pool.fetchrow(
        """
        SELECT message.content,
               message.status,
               message.fact_extraction_status,
               delivery.cancelled_at
        FROM messages AS message
        JOIN background_assistant_deliveries AS delivery
          ON delivery.message_id = message.id
        WHERE message.id = $1
        """,
        first.id,
    )
    assert persisted is not None
    assert persisted["content"] == "已经说出"
    assert persisted["status"] == "interrupted"
    assert persisted["fact_extraction_status"] == "pending"
    assert persisted["cancelled_at"] is None

    assert await background.cancel_proactive_for_fish(fish_id) == 1
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM messages WHERE id = $1",
            replacement.id,
        )
        == 0
    )


async def test_message_origins_name_the_app_behind_what_ta_said(
    postgres_pool: asyncpg.Pool,
) -> None:
    now = datetime.now(UTC)
    messages, background, _ = await stores(postgres_pool, now=now)
    fish_id = uuid4()
    asked = await messages.insert_user_message(fish_id, "在吗", created_at=now)
    seen = uuid4()
    await postgres_pool.execute(
        """INSERT INTO observations
           (id,fish_id,kind,source,observation_type,semantic_key,
            summary,payload,app_name,started_at,ended_at)
           VALUES($1,$2,'event','environment_history','window','w','写代码','{}',
                  'Visual Studio Code',$3,$3)""",
        seen,
        fish_id,
        now,
    )
    sensed = await background.materialize_proactive_observation(
        fish_id,
        asked.session_id,
        "还在改那个函数呀？",
        emotion="neutral",
        expires_at=None,
        supersession_key="a",
        source_observation_id=seen,
    )
    idle = await background.materialize_proactive_observation(
        fish_id,
        asked.session_id,
        "我刚想到一件事。",
        emotion="neutral",
        expires_at=None,
        supersession_key="b",
    )

    origins = await messages.message_origins(fish_id, [asked.id, sensed.id, idle.id])

    assert set(origins) == {sensed.id, idle.id}
    assert origins[sensed.id].observation_source == "environment_history"
    assert origins[sensed.id].app_name == "Visual Studio Code"
    assert origins[idle.id].observation_type is None
    assert await messages.message_origins(uuid4(), [sensed.id]) == {}
