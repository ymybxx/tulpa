from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
import pytest

from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.db.autonomous_tasks import AutonomousTaskStore
from tests.integration.test_agents_postgres import (
    configured_target,
    migrated_stores,
    start_run,
)


@pytest.mark.asyncio
async def test_autonomous_run_preserves_origin_and_separates_result_delivery(
    postgres_pool: asyncpg.Pool,
):
    messages, targets, runs = await migrated_stores(postgres_pool)
    fish = uuid4()
    user = await messages.insert_user_message(fish, "一起看看这部电影")
    target = await configured_target(targets)
    autonomous = AutonomousTaskStore(postgres_pool)
    kwargs = dict(
        fish_id=fish,
        session_id=user.session_id,
        target_id=target.target.id,
        purpose="核实片尾名单中的拍摄地",
        origin_context={"focusId": "f1", "history": []},
        cooldown_seconds=600,
        starts_per_hour=3,
    )
    work = await autonomous.create(**kwargs)
    assert work is not None
    assert work.run.origin == "proactive" and work.run.source_message_id is None
    assert await postgres_pool.fetchval("SELECT count(*) FROM messages") == 1
    assert await autonomous.create(**kwargs) is None
    # Ordinary user tasks may start concurrently.
    _, user_run = await runs.create_task_with_first_run(
        fish, target.target.id, "用户任务", user.id, None, "用户要求的工作"
    )
    await start_run(runs, work.run.id)
    await runs.finish_run(work.run.id, "completed", "拍摄地点已核实", None, None, {})
    assert await runs.peek_unmaterialized_terminal_run() is None
    await start_run(runs, user_run.id)
    await runs.finish_run(user_run.id, "completed", "用户任务结果", None, None, {})
    normal = await runs.peek_unmaterialized_terminal_run()
    assert normal is not None and normal.run.id == user_run.id
    assert not await autonomous.available(
        600, 3
    )  # durable cooldown, even after completion


@pytest.mark.asyncio
async def test_unconfirmed_stop_keeps_slot_and_queued_stop_never_dispatches(
    postgres_pool: asyncpg.Pool,
):
    messages, targets, runs = await migrated_stores(postgres_pool)
    fish = uuid4()
    user = await messages.insert_user_message(fish, "看看这段")
    target = await configured_target(targets)
    autonomous = AutonomousTaskStore(postgres_pool)
    work = await autonomous.create(
        fish_id=fish,
        session_id=user.session_id,
        target_id=target.target.id,
        purpose="核实来源",
        origin_context={},
        cooldown_seconds=0,
        starts_per_hour=3,
    )
    assert work is not None
    stopped = await autonomous.request_stop(work.run.id)
    assert stopped.run.status.value == "cancelled"
    assert await runs.claim_prepared_run(work.run.id, "none", {}) is None
    second = await autonomous.create(
        fish_id=fish,
        session_id=user.session_id,
        target_id=target.target.id,
        purpose="另一个探索",
        origin_context={},
        cooldown_seconds=0,
        starts_per_hour=3,
    )
    assert second is not None
    await start_run(runs, second.run.id)
    await autonomous.request_stop(second.run.id)
    await runs.mark_run_unknown(second.run.id, "stop_unconfirmed", "未确认", None)
    assert not await autonomous.available(0, 3)
    assert (await autonomous.get(second.run.id)).run.origin_context["stopRequested"]


@pytest.mark.asyncio
async def test_streamed_prefix_cannot_overwrite_consumed_text(
    postgres_pool: asyncpg.Pool,
):
    messages, _, _ = await migrated_stores(postgres_pool)
    fish = uuid4()
    user = await messages.insert_user_message(fish, "一起看")
    background = BackgroundMessageStore(postgres_pool)
    message = await background.materialize_proactive_observation(
        fish,
        user.session_id,
        "第一句。",
        emotion="happy",
        expires_at=None,
        supersession_key="focus:1",
        message_id=uuid4(),
    )
    assert await background.update_proactive_content(
        message.id, fish, user.session_id, "第一句。第二句。"
    )
    assert not await background.update_proactive_content(
        message.id, fish, user.session_id, "换掉原话"
    )
    played = await background.finalize_consumption(
        message.id, fish, user.session_id, 4, True
    )
    assert played.content == "第一句。"
    assert not await background.update_proactive_content(
        message.id, fish, user.session_id, "第一句。第二句。第三句。"
    )


@pytest.mark.asyncio
async def test_live_delivery_has_no_time_expiry_and_restart_cleans_only_orphans(
    postgres_pool: asyncpg.Pool,
):
    messages, _, _ = await migrated_stores(postgres_pool)
    fish = uuid4()
    user = await messages.insert_user_message(fish, "一起看")
    background = BackgroundMessageStore(postgres_pool)
    pending = await background.materialize_proactive_observation(
        fish,
        user.session_id,
        "还在等首播。",
        emotion="neutral",
        expires_at=None,
        supersession_key="focus:1",
        created_at=datetime.now(UTC) - timedelta(hours=1),
    )
    assert await background.expire_proactive_messages() == 0
    delivered = await background.list_messages_for_delivery(
        fish, user.session_id, 0, False, 100
    )
    assert [item.message_id for item in delivered] == [pending.id]
    completed = await background.materialize_proactive_observation(
        fish,
        user.session_id,
        "已经说完。",
        emotion="neutral",
        expires_at=None,
        supersession_key="focus:2",
    )
    await background.finalize_consumption(
        completed.id, fish, user.session_id, len(completed.content), False
    )
    legacy = await background.materialize_proactive_observation(
        fish,
        user.session_id,
        "旧版定时消息。",
        emotion="neutral",
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
        supersession_key="legacy",
    )
    assert await background.cancel_orphaned_proactive_streams() == 1
    remaining = await postgres_pool.fetch("SELECT id FROM messages")
    assert {row["id"] for row in remaining} == {user.id, completed.id, legacy.id}
