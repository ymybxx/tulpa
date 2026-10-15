"""Plugin receipt -> History -> JEV -> channel, without direct speech bypass."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.extensions.models import CreateInstance
from fish_memory.extensions.store import ExtensionStore
from fish_memory.history.models import render_history
from tests.extensions.test_service import make_service
from tests.integration.test_session_history import setup as setup_history
from tests.test_experimental_companion import runtime, setup


@pytest.mark.asyncio
async def test_live_event_follows_current_session_and_reaches_jev(postgres_pool):
    fish, old_session, history = await setup_history(postgres_pool)
    new_session = uuid4()
    await postgres_pool.execute(
        "INSERT INTO fish_sessions(id,fish_id) VALUES($1,$2)", new_session, fish
    )
    f = setup("wait")
    f.clock.now = datetime.now(UTC)
    owner, contexts, _ = runtime(f)
    extension = make_service(on_response_state_changed=owner.refresh_busy)
    extension.store = ExtensionStore(postgres_pool)
    extension.proactive.experimental = owner
    extension.session_target = AsyncMock(return_value=(fish, old_session))
    item, token = await extension.create(
        CreateInstance(plugin_id="test.live", fish_id=fish, session_id=new_session)
    )
    extension.attach(item)
    owner.bind(fish, old_session)
    history.on_unit = owner.observe

    async def load_context(fish_id, session_id, *args, **kwargs):
        text = render_history(await history.read(fish_id, session_id))
        return LoadedAnswerContext(
            [
                {"role": "system", "content": "直播中。"},
                {"role": "user", "content": f"History:\n{text}"},
            ],
            (),
        )

    contexts.load_for_proactive = load_context

    def entry(name):
        return {"event_id": str(uuid4()), "type": "live.enter", "data": {"name": name}}

    try:
        await extension.request(
            item,
            "register",
            {
                "events": [
                    {
                        "type": "live.enter",
                        "speak": False,
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "x-fish-history-template": (
                                "直播间观众「${name}」进入了直播间。"
                            ),
                        },
                    }
                ]
            },
        )
        owner.foreground_busy = True
        old = entry("旧会话观众")
        await extension.request(item, "push", old)
        await history.flush(fish, old_session)
        assert not f.jev.calls

        # The plugin socket stays connected while the actual conversation moves.
        f.clock.now = datetime.now(UTC)
        owner.foreground_busy = False
        owner.bind(fish, new_session)
        extension.session_target.return_value = fish, new_session
        receipt = await extension.request(item, "push", entry("新会话观众"))
        assert receipt["session_id"] == str(new_session)
        assert receipt["reason"] == "experimental_event"
        await history.flush(fish, new_session)
        assert owner._decision is not None
        await owner._decision
        assert len(f.jev.calls) == 1
        state, _ = f.jev.calls[0]
        assert "直播间观众「新会话观众」进入了直播间。" in state
        assert "旧会话观众" not in state
        assert "本轮触发：新事件" in state
        assert item.connected
        assert (
            extension.authenticate([(b"authorization", f"Bearer {token}".encode())])
            is item
        )

        old_receipt = await extension.request(item, "push", old)
        assert old_receipt["duplicate"]
        assert old_receipt["session_id"] == str(old_session)
        assert "新会话观众" not in render_history(await history.read(fish, old_session))
        assert "旧会话观众" not in render_history(await history.read(fish, new_session))
        assert len(f.jev.calls) == 1 and not extension.llm.calls
    finally:
        await extension.close()
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("speak", [False, True])
@pytest.mark.parametrize("action", ["t", "wait"])
async def test_live_request_uses_jev_without_plugin_speech(
    postgres_pool, speak, action
):
    fish, session, history = await setup_history(postgres_pool)
    f = setup(action)
    f.clock.now = datetime.now(UTC)
    owner, contexts, _ = runtime(f)
    # Drive the scheduler explicitly; database projection may outlast its timer.
    owner._timer.cancel()
    await asyncio.gather(owner._timer, return_exceptions=True)
    extension = make_service(on_response_state_changed=owner.refresh_busy)
    extension.store = ExtensionStore(postgres_pool)
    extension.proactive.experimental = owner
    owner.foreground_generation = extension.has_active_response
    f.service.tasks = SimpleNamespace(
        available=AsyncMock(return_value=True),
        execute=AsyncMock(
            return_value=SimpleNamespace(text="已整理脱口秀的结构与练习方法。")
        ),
    )
    f.models.query = "学习脱口秀的结构与练习方法"
    item, _ = await extension.create(
        CreateInstance(
            plugin_id="test.live", fish_id=fish, session_id=session, autonomous=False
        )
    )
    extension.attach(item)
    owner.bind(fish, session)
    history.on_unit = owner.observe

    async def load_context(*args, **kwargs):
        text = render_history(await history.read(fish, session))
        return LoadedAnswerContext(
            [
                {"role": "system", "content": "直播中。"},
                {"role": "user", "content": f"History:\n{text}"},
            ],
            (),
        )

    contexts.load_for_proactive = load_context

    def event(text):
        return {
            "event_id": str(uuid4()),
            "type": "live.reply",
            "data": {"viewer_name": "伊利丹illidan", "text": text},
        }

    try:
        await extension.request(
            item,
            "register",
            {
                "events": [
                    {
                        "type": "live.reply",
                        "speak": speak,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "viewer_name": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["viewer_name", "text"],
                            "x-fish-history-template": (
                                "直播间观众「${viewer_name}」发来弹幕：${text}"
                            ),
                        },
                    }
                ]
            },
        )
        # A busy-time arrival is still history, even if projection happens later.
        owner.foreground_busy = True
        first = event("你能不能先去学习一下再来讲")
        assert (await extension.request(item, "push", first))[
            "disposition"
        ] == "stored_only"
        f.clock.now = datetime.now(UTC)
        owner.foreground_busy = False
        await history.flush(fish, session)
        assert owner._decision is None and not f.jev.calls

        second = event("学脱口秀")
        receipt = await extension.request(item, "push", second)
        assert receipt["reason"] == "experimental_event"
        await history.flush(fish, session)
        # Busy-time arrivals are queued and merged with subsequent receipts.
        assert owner._decision is None and not f.jev.calls
        await owner.tick()
        assert owner._decision is not None
        await owner._decision
        await asyncio.gather(*f.service._jobs)

        assert len(f.jev.calls) == 1
        state, available = f.jev.calls[0]
        assert "t" in available
        assert "你能不能先去学习一下再来讲" in state and "学脱口秀" in state
        assert "本轮触发：新事件" in state
        assert not extension.llm.calls and not extension._pending_events
        assert not extension.has_active_response()
        extension.delivery.materialize_proactive_observation.assert_not_awaited()
        if action == "t":
            assert f.models.channels == ["t"]
            f.service.tasks.execute.assert_awaited_once()
            intent, _ = f.service.tasks.execute.call_args.args
            assert intent.purpose == f.models.query
            assert "学脱口秀" in intent.evidence[0].text
            assert [event.observation_type for _, event in f.observations.events] == [
                "experimental.started",
                "experimental.result",
            ]
            stream = owner.streams.get(owner._speaking)
            owner.consumed(stream.intent.id, len(stream.text), False)
        else:
            assert not f.models.channels and not f.observations.events
            f.service.tasks.execute.assert_not_awaited()

        # Duplicate receipts, playback completion and history reload do not replay.
        assert (await extension.request(item, "push", second))["duplicate"]
        await history.flush(fish, session)
        assert len(f.jev.calls) == 1
        text = render_history(await history.read(fish, session))
        assert text.count("event:") == 2

        # A fresh timed opportunity can see both recorded events, without a FIFO.
        f.jev.action = "wait"
        f.clock.now += timedelta(seconds=6)
        await owner.tick()
        await owner._decision
        assert len(f.jev.calls) == 2 and "学脱口秀" in f.jev.calls[-1][0]
    finally:
        await extension.close()
        await owner.close()
        await f.service.close()
