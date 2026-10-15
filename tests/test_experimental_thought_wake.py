import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from fish_memory.history.models import HistoryUnit
from tests.test_experimental_companion import NOW, prepare, runtime, setup


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "enabled,ready,expected", [(True, False, 0), (True, True, 1), (False, False, 1)]
)
async def test_only_enabled_thought_mode_requires_a_thought_for_timer(
    enabled,
    ready,
    expected,
):
    f = setup()
    f.service.configure(
        f.service.configuration.model_copy(update={"thoughts_enabled": enabled})
    )
    owner, contexts, _ = runtime(f)
    check = owner.history.store.has_thought_since_last_speech
    check.return_value = ready
    contexts.load_for_proactive = AsyncMock(return_value=contexts.loaded)
    f.clock.now += timedelta(seconds=5)
    try:
        await owner.tick()
        if owner._decision is not None:
            await owner._decision
        assert len(f.jev.calls) == expected
        assert contexts.load_for_proactive.await_count == expected
        assert check.await_count == int(enabled)
        assert f.service.last_decision_at == (f.clock.now if expected else None)
        assert not owner.busy and not f.models.channels
        if expected:
            # Wait does not consume thought eligibility; only speech resets it.
            f.clock.now += timedelta(seconds=5)
            await owner.tick()
            await owner._decision
            assert len(f.jev.calls) == 2
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["as", "event"])
async def test_events_do_not_require_a_thought(label):
    f = setup()
    owner, _, _ = runtime(f)
    check = owner.history.store.has_thought_since_last_speech
    check.return_value = False
    try:
        owner.observe(
            *owner.binding, HistoryUnit(uuid4(), label, "新的外界内容", NOW, NOW)
        )
        await owner._decision
        assert len(f.jev.calls) == 1
        check.assert_not_awaited()
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_user_chat_does_not_require_a_thought():
    f = setup("s")
    owner, _, _ = runtime(f)
    check = owner.history.store.has_thought_since_last_speech
    check.return_value = False
    answer = prepare(f)
    try:
        assert (await answer.initial_ready()).name == "s"
        assert len(f.jev.calls) == 1
        check.assert_not_awaited()
    finally:
        await answer.cancel()
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_thought_arrival_does_not_trigger_jev_or_replay_a_skipped_tick():
    f = setup()
    owner, _, _ = runtime(f)
    check = owner.history.store.has_thought_since_last_speech
    check.return_value = False
    f.clock.now += timedelta(seconds=5)
    try:
        await owner.tick()
        assert owner._decision is None
        check.return_value = True
        owner.observe(
            *owner.binding,
            HistoryUnit(
                uuid4(),
                "event",
                "宠物的内心想法（尚未表达）：听起来有些不同。",
                f.clock.now,
                f.clock.now,
                triggers_decision=False,
                interrupts_observation=False,
            ),
        )
        await asyncio.sleep(0)
        assert not f.jev.calls and owner._decision is None
        await owner.tick()
        await owner._decision
        assert len(f.jev.calls) == 1
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["event", "session"])
async def test_pending_timer_check_does_not_block_events_or_cross_sessions(change):
    f = setup()
    owner, _, _ = runtime(f)
    started, release = asyncio.Event(), asyncio.Event()

    async def check(*args):
        started.set()
        await release.wait()
        return True

    owner.history.store.has_thought_since_last_speech = check
    f.clock.now += timedelta(seconds=5)
    timer = asyncio.create_task(owner.tick())
    try:
        await started.wait()
        assert not owner.busy
        if change == "event":
            owner.observe(
                *owner.binding,
                HistoryUnit(
                    uuid4(), "event", "搜索结果出来了", f.clock.now, f.clock.now
                ),
            )
            await owner._decision
            assert len(f.jev.calls) == 1
        else:
            owner.bind(owner.binding[0], uuid4())
        release.set()
        await timer
        assert len(f.jev.calls) == (1 if change == "event" else 0)
    finally:
        release.set()
        await timer
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_failed_thought_check_skips_one_tick_and_can_recover():
    f = setup()
    owner, _, _ = runtime(f)
    check = owner.history.store.has_thought_since_last_speech
    check.side_effect = RuntimeError("test database unavailable")
    f.clock.now += timedelta(seconds=5)
    try:
        await owner.tick()
        assert not owner.busy and not f.jev.calls
        assert f.service.last_decision_at is None
        check.side_effect = None
        await owner.tick()
        await owner._decision
        assert len(f.jev.calls) == 1
    finally:
        await owner.close()
        await f.service.close()
