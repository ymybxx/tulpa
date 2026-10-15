import asyncio
from uuid import uuid4

import pytest

from fish_memory.agent.models import AgentRunStatus
from fish_memory.db.plugin_tasks import PluginTaskConflict, PluginTaskStore
from tests.integration.test_agents_postgres import (
    configured_target,
    migrated_stores,
    start_run,
)


async def setup(pool):
    messages, targets, runs = await migrated_stores(pool)
    user = await messages.insert_user_message(uuid4(), "测试插件研究")
    target = await configured_target(targets)
    context = dict(
        pluginId="test",
        instanceId=str(uuid4()),
        generation=str(uuid4()),
        requestId=str(uuid4()),
        requestDigest="one",
        contextKey="A:1",
        reference={"text": "冻结材料"},
    )
    return (
        PluginTaskStore(pool),
        runs,
        dict(
            fish_id=user.fish_id,
            session_id=user.session_id,
            target_id=target.target.id,
            purpose="研究 A",
            origin_context=context,
        ),
        user,
    )


@pytest.mark.asyncio
async def test_plugin_task_idempotency_scope_and_no_chat_delivery(postgres_pool):
    store, runs, kwargs, user = await setup(postgres_pool)
    first, repeated = await asyncio.gather(
        store.create(**kwargs), store.create(**kwargs)
    )
    assert first.run.id == repeated.run.id and first.run.origin == "plugin"
    assert first.run.source_message_id is None
    with pytest.raises(PluginTaskConflict):
        await store.create(
            **{
                **kwargs,
                "origin_context": {
                    **kwargs["origin_context"],
                    "requestDigest": "changed",
                },
            }
        )
    with pytest.raises(PluginTaskConflict):
        await store.create(
            **{
                **kwargs,
                "origin_context": {
                    **kwargs["origin_context"],
                    "requestId": str(uuid4()),
                    "generation": str(uuid4()),
                },
            }
        )
    await runs.create_task_with_first_run(
        user.fish_id, kwargs["target_id"], "普通任务", user.id, None, "用户工作"
    )
    await start_run(runs, first.run.id)
    await runs.finish_run(
        first.run.id, AgentRunStatus.COMPLETED, "研究结果", None, None, {}
    )
    assert await runs.peek_unmaterialized_terminal_run() is None
    candidates = await runs.list_task_candidates(user.fish_id, user.session_id)
    assert all(c.task.id != first.run.task_id for c in candidates)
    assert await postgres_pool.fetchval("SELECT count(*) FROM messages") == 1
    assert len(await store.active()) == 0


@pytest.mark.asyncio
async def test_plugin_stops_and_committed_notifications(postgres_pool):
    store, runs, kwargs, _ = await setup(postgres_pool)
    received = asyncio.Queue()
    async with postgres_pool.acquire() as connection:

        def notified(_c, _p, _ch, payload):
            received.put_nowait(payload)

        await connection.add_listener("fish_plugin_tasks", notified)
        work = await store.create(**kwargs)
        assert await asyncio.wait_for(received.get(), 2) == str(work.run.id)
        stopped = await store.request_stop(work.run.id)
        assert stopped.run.status == AgentRunStatus.CANCELLED
        assert await runs.claim_prepared_run(work.run.id, "none", {}) is None
        kwargs["origin_context"]["requestId"] = str(uuid4())
        second = await store.create(**kwargs)
        await start_run(runs, second.run.id)
        await store.request_stop(second.run.id)
        await runs.mark_run_unknown(
            second.run.id, "stop_unconfirmed", "not confirmed", None
        )
        kwargs["origin_context"]["requestId"] = str(uuid4())
        with pytest.raises(PluginTaskConflict):
            await store.create(**kwargs)
        await connection.remove_listener("fish_plugin_tasks", notified)
