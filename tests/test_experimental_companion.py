import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from fish_memory.answer.context import INTERRUPTION_MARKER
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.lane_stream import BlankFieldError
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.answer.parallel_generation import ParallelAnswerGenerationPipeline
from fish_memory.answer.prepared_answer import PreparedInitialAnswer
from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.answer.voice_drafts import VoiceDraftLease
from fish_memory.clock import format_prompt_timestamp
from fish_memory.experimental.channels import (
    SPEECH_REPETITION_PENALTY,
    ChannelResult,
    channel_response_schema,
    generate_channel,
    render_state,
)
from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.jev import JevClient
from fish_memory.experimental.perception import PerceptionInput
from fish_memory.experimental.prepared import ExperimentalPreparedAnswer
from fish_memory.experimental.runtime import TIMED_WAKE, ExperimentalRuntime
from fish_memory.experimental.service import ExperimentalService
from fish_memory.history.models import HistoryUnit
from fish_memory.llm.client import _provider_messages
from fish_memory.llm.types import ChatStreamChunk
from fish_memory.models import MessageStatus, Role
from fish_memory.observations.environment_history_models import SemanticDocument
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.proactive.expression import ExpressionComposer, ExpressionRegistry
from fish_memory.recall.models import RecalledFact
from tests.answer.fakes import FakeStore

NOW = datetime(2026, 9, 19, 12, tzinfo=UTC)


class Jev:
    def __init__(self, action="wait"):
        self.action = action
        self.calls = []
        self.filter_calls = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.admitted = True

    async def decide(self, config, state, available):
        self.calls.append((state, available))
        await self.gate.wait()
        return self.action

    async def admit(self, config, state):
        self.filter_calls.append(state)
        return self.admitted

    async def close(self):
        pass


class Models:
    def __init__(self):
        self.channels = []
        self.requests = []
        self.parameters = []
        self.started = {name: asyncio.Event() for name in ("s", "r", "t", "w")}
        self.summaries = []
        self.gates = {}
        self.on_summary = lambda: None
        self.query = "去年一起听过哪首歌？"

    async def stream_json(self, messages, **kwargs):
        action = re.search(r"本轮 (\w) 通道", messages[0]["content"])[1]
        self.channels.append(action)
        self.requests.append((action, messages))
        self.parameters.append((action, kwargs))
        self.started[action].set()
        if action in self.gates:
            await self.gates[action].wait()
        value = {"m": "happy", "a": "直接说话。" if action == "s" else "让我回想一下。"}
        if action != "s":
            value["q"] = self.query
        value["e"] = 0
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        for start in range(0, len(raw), 7):
            yield ChatStreamChunk(content=raw[start : start + 7])

    async def stream(self, messages, **kwargs):
        self.summaries.append(messages)
        self.on_summary()
        yield ChatStreamChunk(content="回忆到去年露营一起听过夜曲，当天下雨。")


class Recall:
    def __init__(self):
        self.calls = []
        self.gate = asyncio.Event()
        self.gate.set()

    async def recall(self, fish, question, depth, **kwargs):
        self.calls.append(question)
        await self.gate.wait()
        fact = RecalledFact(
            uuid4(),
            uuid4(),
            Role.USER,
            "去年露营听过夜曲。",
            None,
            None,
            NOW - timedelta(days=100),
            (),
            (),
            None,
            None,
        )
        return SimpleNamespace(facts=(fact,), relation_units=(), memory_unit_order=())


class Observations:
    def __init__(self):
        self.events = []

    async def record_event(self, fish, event):
        self.events.append((fish, event))


class Contexts:
    def __init__(self):
        self.loaded = LoadedAnswerContext(
            [
                {"role": "system", "content": "你喜欢音乐。"},
                {"role": "user", "content": "继续陪我聊刚才那首歌。"},
            ],
            (),
        )

    async def load_for_proactive(self, *args, **kwargs):
        return self.loaded


class Delivery:
    def __init__(self):
        self.streams = []

    async def begin(self, stream):
        self.streams.append(stream)

    async def finish(self, stream):
        pass


class Screen:
    def __init__(self):
        self.calls = []
        self.models = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        self.gate.set()
        self.image = {
            "data_url": "data:image/jpeg;base64,frozen-ten-frame-sheet",
            "digest": "a" * 64,
            "byte_count": 10,
            "width": 1920,
            "height": 900,
            "observed_at": NOW.isoformat(),
            "description": "用户电脑屏幕最近10秒的10帧合成画面。",
            "detail": "high",
        }

    async def capture(self, fish, at, *, model):
        self.calls.append((fish, at))
        self.models.append(model)
        self.started.set()
        await self.gate.wait()
        return PerceptionInput(
            images=(self.image,), heard="[00:01—00:02] 声源A：刚才的转调很好听。"
        )


class AvailableTasks:
    async def available(self, context):
        return True


def setup(action="wait"):
    clock = SimpleNamespace(now=NOW)
    models, jev, recall, observations = Models(), Jev(action), Recall(), Observations()
    service = ExperimentalService(
        ExperimentalConfiguration(
            enabled=True, chat_pick_enabled=True, api_key=SecretStr("test-only")
        ),
        judge=jev,
        chat=models,
        reasoning=models,
        recall=recall,
        observations=observations,
        now=lambda: clock.now,
        search_enabled=True,
        recall_token_budget=5000,
    )
    return SimpleNamespace(
        clock=clock,
        models=models,
        jev=jev,
        recall=recall,
        observations=observations,
        service=service,
    )


def prepare(fixture):
    return ExperimentalPreparedAnswer(
        fixture.service,
        uuid4(),
        Contexts().loaded,
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
    )


def runtime(fixture):
    streams, delivery, contexts = ExpressionRegistry(), Delivery(), Contexts()
    owner = ExperimentalRuntime(
        fixture.service,
        contexts=contexts,
        expression=ExpressionComposer(delivery),
        streams=streams,
        foreground_generation=lambda: False,
        cancel_delivery=lambda *args: None,
    )
    owner.bind(uuid4(), uuid4())
    owner.history = SimpleNamespace(
        store=SimpleNamespace(
            has_thought_since_last_speech=AsyncMock(return_value=True)
        )
    )
    return owner, contexts, delivery


async def settle_work(fixture):
    await asyncio.gather(*fixture.service._jobs)


@pytest.mark.asyncio
async def test_user_candidates_and_jev_start_in_parallel_without_running_tools():
    f = setup("r")
    f.jev.gate.clear()
    answer = prepare(f)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert set(f.models.channels) == {"s", "r", "w"}
    assert len(f.jev.calls) == 1
    assert not f.recall.calls and not f.observations.events
    assert not answer.sink.snapshot().content
    f.jev.gate.set()
    selected = await answer.initial_ready()
    assert selected.name == "r" and selected.query
    assert answer.sink.snapshot().content == "让我回想一下。"
    assert not f.recall.calls
    await answer.release_to_public(
        PromotableAnswerSink(),
        source_message_id=uuid4(),
        answer_message_id=uuid4(),
        runtime=None,
        session_id=uuid4(),
    )
    await settle_work(f)
    assert len(f.recall.calls) == 1
    await f.service.close()


@pytest.mark.asyncio
async def test_selected_lane_does_not_wait_for_other_lane():
    f = setup("s")
    f.models.gates["w"] = asyncio.Event()
    answer = prepare(f)
    result = await asyncio.wait_for(answer.initial_ready(), 1)
    assert result.name == "s"
    assert answer.sink.snapshot().content == "直接说话。"
    assert not f.recall.calls
    await f.service.close()


@pytest.mark.asyncio
async def test_invalid_selected_s_reply_finishes_and_allows_the_next_user_message():
    class SingleGeneratingStore(FakeStore):
        active_answer = None

        async def insert_generating_answer(self, *args, **kwargs):
            assert self.active_answer is None, "previous answer still generating"
            answer = await super().insert_generating_answer(*args, **kwargs)
            self.active_answer = answer.id
            return answer

        async def finish_answer(self, *args, **kwargs):
            result = await super().finish_answer(*args, **kwargs)
            self.active_answer = None
            return result

    class EmptyS:
        async def stream_json(self, *args, **kwargs):
            yield ChatStreamChunk(content='{"m":"neutral","a":"","e":0}')

    f = setup("s")
    f.service.search_enabled = False
    f.service.recall = None
    f.service.chat = EmptyS()
    fish, session, source_id, answer_id = (uuid4() for _ in range(4))
    store = SingleGeneratingStore(fish, session)
    coordinator = AnswerCoordinator(
        store, f.models, system_prompt="你陪伴用户。", experiment=f.service
    )
    tracker = MessagePerformanceTracker()
    prepared = ExperimentalPreparedAnswer(
        f.service, fish, Contexts().loaded, PromotableAnswerSink(), tracker
    )
    lease = VoiceDraftLease(
        VoiceDraftKey(fish, session, source_id), 1, "candidate", "input", NOW, prepared
    )
    try:
        await coordinator.handle_user_message(
            fish,
            session,
            1,
            message_id=source_id,
            answer_message_id=answer_id,
            tracker=tracker,
            draft_lease=lease,
        )
        current = coordinator.get_runtime(fish, session)
        await current.llm_task
        assert current.status is MessageStatus.FAILED
        assert prepared._cancelled
        assert store.finish_calls[0]["requested_status"] is MessageStatus.FAILED
        assert store.finish_calls[0]["content"] == ""
        assert store.active_answer is None

        f.service.chat = f.models
        store.latest_user_sequence = 2
        await coordinator.handle_user_message(
            fish, session, 2, message_id=uuid4(), tracker=MessagePerformanceTracker()
        )
        await current.llm_task
        assert current.status is MessageStatus.COMPLETED
        assert current.content == "直接说话。"
        assert store.active_answer is None
    finally:
        await asyncio.gather(
            coordinator.close(), f.service.close(), return_exceptions=True
        )


@pytest.mark.asyncio
async def test_user_candidates_share_one_screen_while_jev_runs_in_parallel():
    f = setup("s")
    screen = f.service.perception = Screen()
    f.service.tasks = AvailableTasks()
    screen.gate.clear()
    f.jev.gate.clear()
    answer = prepare(f)
    try:
        await asyncio.wait_for(screen.started.wait(), 1)
        await asyncio.sleep(0)
        assert len(f.jev.calls) == 1
        assert not f.models.requests
        assert "base64" not in f.jev.calls[0][0]
        screen.gate.set()
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in f.models.started.values())), 1
        )
        assert {name for name, _ in f.models.requests} == {"s", "r", "t", "w"}
        assert len(screen.calls) == 1
        for _, messages in f.models.requests:
            assert messages[-1]["environment_images"] == (screen.image,)
            assert messages[-1]["content"].count("刚才的转调很好听") == 1
            assert "你喜欢音乐。" in messages[0]["content"]
            assert all("你喜欢音乐。" not in m["content"] for m in messages[1:])
            assert _provider_messages(messages)[-1]["content"][-1]["text"] == (
                f"当前时间：{format_prompt_timestamp(NOW)}"
            )
        for action, parameters in f.models.parameters:
            assert parameters == {
                **({"temperature": 1.0, "top_p": 0.9} if action == "s" else {}),
                "response_schema": channel_response_schema(query=action != "s"),
                "repetition_penalty": SPEECH_REPETITION_PENALTY,
            }
        s_messages = next(
            messages for action, messages in f.models.requests if action == "s"
        )
        assert "你喜欢音乐。" in s_messages[0]["content"]
        assert "你喜欢音乐。" not in s_messages[-1]["content"]
        f.jev.gate.set()
        assert (await answer.initial_ready()).name == "s"
    finally:
        await answer.cancel()
        await f.service.close()


@pytest.mark.asyncio
async def test_cancelled_user_revision_cannot_send_its_pending_screen():
    f = setup("s")
    screen = f.service.perception = Screen()
    screen.gate.clear()
    f.jev.gate.clear()
    answer = prepare(f)
    await screen.started.wait()
    await answer.cancel()
    screen.gate.set()
    f.jev.gate.set()
    assert answer._perception_task.cancelled()
    assert not f.models.requests
    await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["s", "r", "t", "w", "wait"])
async def test_autonomous_channel_gets_latest_screen_after_jev(action):
    f = setup(action)
    screen = f.service.perception = Screen()
    f.service.tasks = AvailableTasks()
    f.service.launch = lambda *args: None
    owner, _, _ = runtime(f)
    f.jev.gate.clear()
    try:
        owner.trigger(TIMED_WAKE)
        await asyncio.sleep(0)
        assert not screen.calls and not f.models.requests
        f.clock.now += timedelta(seconds=1)
        f.jev.gate.set()
        await owner._decision
        assert len(f.jev.calls) == 1
        assert "base64" not in f.jev.calls[0][0]
        if action == "wait":
            assert not screen.calls and not f.models.requests
        else:
            assert screen.calls == [(owner.binding[0], f.clock.now)]
            assert f.models.requests[0][0] == action
            # Nobody said the trigger; the speaker is told so, or it answers "好的".
            # Jev weighs whether to speak; the speaker only hears that it wants to.
            assert (
                "本轮触发：这不是用户说的话。你现在想说话。"
                in (f.models.requests[0][1][-1]["content"])
            )
            assert "安静陪伴" not in f.models.requests[0][1][-1]["content"]
            assert "安静陪伴" in f.jev.calls[0][0]
            assert f.models.requests[0][1][-1]["environment_images"] == (screen.image,)
            assert "刚才的转调很好听" in f.models.requests[0][1][-1]["content"]
            assert (
                _provider_messages(f.models.requests[0][1])[-1]["content"][-1]["text"]
                == f.jev.calls[0][0].rsplit("\n\n", 1)[-1]
            )
            assert f.models.parameters[0][1] == {
                **({"temperature": 1.0, "top_p": 0.9} if action == "s" else {}),
                "response_schema": channel_response_schema(query=action != "s"),
                "repetition_penalty": SPEECH_REPETITION_PENALTY,
            }
            assert "你喜欢音乐。" in f.models.requests[0][1][0]["content"]
            assert all(
                "你喜欢音乐。" not in m["content"] for m in f.models.requests[0][1][1:]
            )
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_wait_has_no_output_or_execution():
    f = setup()
    answer = prepare(f)
    await answer.release_to_public(
        PromotableAnswerSink(),
        source_message_id=uuid4(),
        answer_message_id=uuid4(),
        runtime=None,
        session_id=uuid4(),
    )
    assert answer.result.name == "wait"
    assert answer.sink.snapshot().content == ""
    assert not f.observations.events and not f.recall.calls
    await f.service.close()


@pytest.mark.asyncio
async def test_cancelled_revision_cannot_publish_or_execute():
    f = setup("r")
    f.jev.gate.clear()
    answer = prepare(f)
    await asyncio.sleep(0)
    await answer.cancel()
    f.jev.gate.set()
    with pytest.raises(asyncio.CancelledError):
        await answer.release_to_public(
            PromotableAnswerSink(),
            source_message_id=uuid4(),
            answer_message_id=uuid4(),
            runtime=None,
            session_id=uuid4(),
        )
    assert not f.observations.events and not f.recall.calls
    await f.service.close()


@pytest.mark.asyncio
async def test_mode_switch_revokes_private_preparation():
    f = setup("s")
    f.jev.gate.clear()
    answer = prepare(f)
    await asyncio.sleep(0)
    f.service.configure(ExperimentalConfiguration())
    with pytest.raises(asyncio.CancelledError):
        await answer.initial_ready()
    await f.service.close()


@pytest.mark.asyncio
async def test_recall_summary_is_only_payload_and_event_time_is_summary_completion():
    f = setup()
    f.models.on_summary = lambda: setattr(f.clock, "now", NOW + timedelta(seconds=20))
    session, origin = uuid4(), uuid4()
    channel = ChannelResult("r", "去年一起听过哪首歌？")
    f.service.launch(uuid4(), session, origin, channel, Contexts().loaded, "上下文")
    await settle_work(f)
    _, event = f.observations.events[-1]
    assert event.occurred_at == NOW + timedelta(seconds=20)
    assert event.payload["session_id"] == str(session)
    assert "去年一起听过哪首歌" in event.payload["context_text"]
    assert "回忆到去年露营一起听过夜曲" in event.payload["context_text"]
    assert '"u"' not in event.payload["context_text"]
    assert len(f.models.summaries) == 1
    await f.service.close()


@pytest.mark.asyncio
async def test_recall_picked_in_chat_hands_over_the_memories_without_a_summary():
    f = setup()
    channel = ChannelResult("r", "去年一起听过哪首歌？")
    f.service.launch(
        uuid4(),
        uuid4(),
        uuid4(),
        channel,
        Contexts().loaded,
        "上下文",
        summarize_recall=False,
    )
    await settle_work(f)
    _, event = f.observations.events[-1]
    text = event.payload["context_text"]
    assert "围绕“去年一起听过哪首歌？”找回的记录" in text
    assert '"u"' not in text
    assert not f.models.summaries
    await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action,name", [("r", "回忆"), ("t", "研究"), ("w", "搜索")])
async def test_work_started_is_recorded_before_execution_and_result(
    monkeypatch, action, name
):
    f = setup()
    executing, complete = asyncio.Event(), asyncio.Event()
    fish, session, origin = uuid4(), uuid4(), uuid4()
    channel = ChannelResult(action, "刚才那集的演员是谁？")

    async def execute(*args):
        assert len(f.observations.events) == 1
        event = f.observations.events[0][1]
        assert event.observation_type == "experimental.started"
        assert event.payload["status"] == "started"
        assert event.payload["session_id"] == str(session)
        assert event.payload["origin_id"] == str(origin)
        assert f"{name}已开始，正在等待结果" in event.payload["context_text"]
        assert event.occurred_at == NOW
        executing.set()
        await complete.wait()
        return "找到了该集演员的信息。"

    monkeypatch.setattr(f.service, "_execute", execute)
    f.service.launch(fish, session, origin, channel, Contexts().loaded, "上下文")
    f.service.launch(fish, session, origin, channel, Contexts().loaded, "上下文")
    try:
        await asyncio.wait_for(executing.wait(), timeout=1)
        assert not f.jev.calls
        f.clock.now += timedelta(seconds=10)
        complete.set()
        await settle_work(f)
        assert len(f.observations.events) == 2
        result = f.observations.events[-1][1]
        assert result.observation_type == "experimental.result"
        assert result.payload["status"] == "completed"
        assert result.payload["origin_id"] == str(origin)
        assert f"{name}结果出来了" in result.payload["context_text"]
        assert result.occurred_at == f.clock.now
    finally:
        await f.service.close()


@pytest.mark.asyncio
async def test_work_picked_in_chat_runs_while_speaking_up_is_off(monkeypatch):
    f = setup()
    f.service.configure(
        ExperimentalConfiguration(
            enabled=False, chat_pick_enabled=True, api_key=SecretStr("test-only")
        )
    )
    ran = asyncio.Event()

    async def execute(*args):
        ran.set()
        return "明天杭州有小雨。"

    monkeypatch.setattr(f.service, "_execute", execute)
    f.service.launch(
        uuid4(),
        uuid4(),
        uuid4(),
        ChannelResult("w", "杭州明天下雨吗"),
        Contexts().loaded,
        "上下文",
    )
    try:
        await asyncio.wait_for(ran.wait(), timeout=1)
        await settle_work(f)
        assert f.observations.events[-1][1].payload["status"] == "completed"
    finally:
        await f.service.close()


@pytest.mark.asyncio
async def test_failed_work_tells_ta_which_service_let_it_down(monkeypatch):
    from fish_memory import failures

    f = setup()

    async def execute(*args):
        raise failures.noted(ConnectionError("refused"), "reasoning")

    monkeypatch.setattr(f.service, "_execute", execute)
    f.service.launch(
        uuid4(),
        uuid4(),
        uuid4(),
        ChannelResult("w", "演员是谁"),
        Contexts().loaded,
        "x",
    )
    try:
        await settle_work(f)
        text = f.observations.events[-1][1].payload["context_text"]
        assert "本次处理失败（连不上推理大脑），未获得可靠结果。" in text
    finally:
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_work_failure_or_shutdown_records_terminal_fact(monkeypatch, cancelled):
    f = setup()
    executing, finish = asyncio.Event(), asyncio.Event()

    async def execute(*args):
        executing.set()
        await finish.wait()
        raise RuntimeError("test provider failed")

    monkeypatch.setattr(f.service, "_execute", execute)
    f.service.launch(
        uuid4(),
        uuid4(),
        uuid4(),
        ChannelResult("w", "演员是谁"),
        Contexts().loaded,
        "上下文",
    )
    try:
        await asyncio.wait_for(executing.wait(), timeout=1)
        f.clock.now += timedelta(seconds=3)
        if cancelled:
            await f.service.close()
        else:
            finish.set()
            await settle_work(f)
        assert [e.payload["status"] for _, e in f.observations.events] == [
            "started",
            "cancelled" if cancelled else "failed",
        ]
        terminal = f.observations.events[-1][1]
        assert terminal.occurred_at == f.clock.now
        assert ("等待已中止" if cancelled else "处理失败") in terminal.payload[
            "context_text"
        ]
        assert "结果出来了" not in terminal.payload["context_text"]
    finally:
        if not cancelled:
            await f.service.close()


@pytest.mark.asyncio
async def test_work_does_not_start_before_start_event_is_persisted(monkeypatch):
    f = setup()
    writing, persisted, executing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    record = f.observations.record_event

    async def record_event(fish, event):
        if event.observation_type == "experimental.started":
            writing.set()
            await persisted.wait()
        await record(fish, event)

    async def execute(*args):
        executing.set()
        return "结果"

    monkeypatch.setattr(f.observations, "record_event", record_event)
    monkeypatch.setattr(f.service, "_execute", execute)
    f.service.launch(
        uuid4(),
        uuid4(),
        uuid4(),
        ChannelResult("r", "往事"),
        Contexts().loaded,
        "上下文",
    )
    try:
        await asyncio.wait_for(writing.wait(), timeout=1)
        assert not executing.is_set()
        persisted.set()
        await settle_work(f)
        assert executing.is_set()
    finally:
        await f.service.close()


@pytest.mark.asyncio
async def test_started_event_does_not_wake_jev_but_next_tick_sees_it():
    f = setup()
    owner, contexts, _ = runtime(f)
    unit = HistoryUnit(
        uuid4(),
        "event",
        "关于“演员是谁”的搜索已开始，正在等待结果。",
        NOW,
        NOW,
        triggers_decision=False,
    )
    owner.observe(*owner.binding, unit)
    contexts.loaded.messages.append(
        {"role": "user", "content": unit.render(), "context_reference": True}
    )
    try:
        await asyncio.sleep(0)
        assert not f.jev.calls
        f.clock.now += timedelta(seconds=5)
        await owner.tick()
        await owner._decision
        assert len(f.jev.calls) == 1
        assert "event: 关于“演员是谁”的搜索已开始，正在等待结果。" in f.jev.calls[0][0]
        assert "triggers_decision" not in f.jev.calls[0][0]
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_event_while_busy_waits_and_is_handed_over_once_idle():
    f = setup()
    owner, _, _ = runtime(f)
    fish, session = owner.binding
    owner.foreground_busy = True
    unit = HistoryUnit(uuid4(), "event", "搜索结果出来了：新结论。", NOW, NOW)
    owner.observe(fish, session, unit)
    await owner.tick()
    await asyncio.sleep(0)
    assert not f.jev.calls
    owner.foreground_busy = False
    owner.observe(fish, session, unit)
    # No idle interval is needed: the waiting event goes out on the next tick.
    await owner.tick()
    await owner._decision
    assert len(f.jev.calls) == 1
    assert "新结论" in f.jev.calls[0][0]
    await owner.tick()
    assert len(f.jev.calls) == 1
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_found_results_wake_ta_while_speaking_up_is_off():
    f = setup()
    f.service.configure(
        ExperimentalConfiguration(
            enabled=False, chat_pick_enabled=True, api_key=SecretStr("test-only")
        )
    )
    owner, _, _ = runtime(f)
    fish, session = owner.binding
    try:
        owner.observe(
            fish, session, HistoryUnit(uuid4(), "event", "打开了编辑器", NOW, NOW)
        )
        f.clock.now += timedelta(seconds=30)
        await owner.tick()
        await asyncio.sleep(0)
        assert not f.jev.calls

        found = "关于“杭州明天天气”的搜索结果出来了。"
        owner.observe(
            fish,
            session,
            HistoryUnit(uuid4(), "event", found, NOW, NOW, reports_work=True),
        )
        await owner._decision
        assert len(f.jev.calls) == 1 and found in f.jev.calls[0][0]

        owner.foreground_busy = True
        owner.observe(
            fish,
            session,
            HistoryUnit(
                uuid4(), "event", "回忆结果出来了。", NOW, NOW, reports_work=True
            ),
        )
        owner.foreground_busy = False
        await owner.tick()
        await owner._decision
        assert len(f.jev.calls) == 2 and "回忆结果出来了" in f.jev.calls[1][0]
        await owner.tick()
        assert len(f.jev.calls) == 2
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_a_message_for_the_user_waits_for_the_timed_wake_while_speaking_up():
    f = setup()
    owner, _, _ = runtime(f)
    fish, session = owner.binding
    try:
        assert owner.active
        # Sent in from outside: kept, but it does not wake TA on its own.
        owner.observe(
            fish,
            session,
            HistoryUnit(
                uuid4(),
                "event",
                "快递到了",
                NOW,
                NOW,
                triggers_decision=False,
                reports_work=True,
            ),
        )
        await asyncio.sleep(0)
        assert owner._decision is None and not f.jev.calls
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_a_message_for_the_user_is_answered_while_speaking_up_is_off():
    f = setup()
    f.service.configure(
        ExperimentalConfiguration(
            enabled=False, chat_pick_enabled=True, api_key=SecretStr("test-only")
        )
    )
    owner, _, _ = runtime(f)
    fish, session = owner.binding
    try:
        owner.observe(
            fish,
            session,
            HistoryUnit(
                uuid4(),
                "event",
                "只是记下的天气",
                NOW,
                NOW,
                triggers_decision=False,
            ),
        )
        await asyncio.sleep(0)
        assert owner._decision is None
        owner.observe(
            fish,
            session,
            HistoryUnit(
                uuid4(),
                "event",
                "快递到了",
                NOW,
                NOW,
                triggers_decision=False,
                reports_work=True,
            ),
        )
        await owner._decision
        assert len(f.jev.calls) == 1 and "快递到了" in f.jev.calls[0][0]
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_events_while_busy_are_merged_into_one_decision():
    f = setup()
    owner, _, _ = runtime(f)
    fish, session = owner.binding
    owner.foreground_busy = True
    for content in ("打开了编辑器", "切到了浏览器"):
        owner.observe(fish, session, HistoryUnit(uuid4(), "event", content, NOW, NOW))
    owner.foreground_busy = False
    await owner.tick()
    await owner._decision
    assert len(f.jev.calls) == 1
    assert "打开了编辑器" in f.jev.calls[0][0] and "切到了浏览器" in f.jev.calls[0][0]
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_an_empty_line_leaves_ta_quiet_without_an_error(caplog):
    f = setup("s")
    owner, _, _ = runtime(f)

    async def blank(messages, **kwargs):
        yield ChatStreamChunk(content='{"m":"happy","a":"","e":0}')

    f.models.stream_json = blank
    owner.observe(*owner.binding, HistoryUnit(uuid4(), "event", "打开网页", NOW, NOW))
    await owner._decision
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    assert owner._speaking is None
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_own_speech_never_triggers_decision():
    f = setup()
    owner, _, _ = runtime(f)
    owner.observe(*owner.binding, HistoryUnit(uuid4(), "a", "我想起来了。", NOW, NOW))
    await asyncio.sleep(0)
    assert not f.jev.calls
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_event_processing_is_one_shot_and_serial():
    f = setup()
    owner, _, _ = runtime(f)
    f.jev.gate.clear()
    first = HistoryUnit(uuid4(), "event", "打开网页", NOW, NOW)
    second = HistoryUnit(uuid4(), "event", "切换窗口", NOW, NOW)
    owner.observe(*owner.binding, first)
    owner.observe(*owner.binding, second)
    await asyncio.sleep(0)
    assert len(f.jev.calls) == 1
    f.jev.gate.set()
    await owner._decision
    # The second event waited for the first decision instead of being lost.
    await owner.tick()
    await owner._decision
    assert len(f.jev.calls) == 2
    assert "切换窗口" in f.jev.calls[1][0]
    owner.observe(*owner.binding, second)
    await owner.tick()
    assert len(f.jev.calls) == 2
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_event_received_while_busy_but_projected_late_still_waits_its_turn():
    f = setup()
    owner, _, _ = runtime(f)
    owner.foreground_busy = True
    f.clock.now += timedelta(seconds=2)
    received = f.clock.now
    unit = HistoryUnit(uuid4(), "event", "忙碌时出现的变化", received, received)
    f.clock.now += timedelta(seconds=2)
    owner.foreground_busy = False
    owner.observe(*owner.binding, unit, received_at=received)
    await asyncio.sleep(0)
    assert not f.jev.calls
    await owner.tick()
    await owner._decision
    assert len(f.jev.calls) == 1
    assert "忙碌时出现的变化" in f.jev.calls[0][0]
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_autonomous_speech_remains_busy_until_consumed_without_completion_wake():
    f = setup("s")
    owner, _, delivery = runtime(f)
    owner.trigger("用户让我继续讲述")
    await owner._decision
    assert owner.busy and len(delivery.streams) == 1
    stream = delivery.streams[0]
    owner.consumed(stream.intent.id, 2, True)
    assert not owner.busy
    assert stream.intent.consumed_offset == 2
    assert stream.intent.state == "interrupted"
    assert len(f.jev.calls) == 1
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_speech_after_an_event_remembers_the_observation_behind_it():
    f = setup("s")
    owner, _, delivery = runtime(f)
    unit = HistoryUnit(uuid4(), "event", "打开了编辑器", NOW, NOW)
    owner.observe(*owner.binding, unit)
    await owner._decision
    assert delivery.streams[0].intent.source_observation_id == unit.source_id
    owner.consumed(delivery.streams[0].intent.id, 2, False)
    owner.trigger("定时唤醒")
    await owner._decision
    assert delivery.streams[1].intent.source_observation_id is None
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_pending_user_preparation_prevents_autonomous_interleaving():
    f = setup("s")
    owner, _, _ = runtime(f)
    answer = prepare(f)
    await answer.initial_ready()
    owner.trigger("屏幕变化")
    assert len(f.jev.calls) == 1
    await answer.cancel()
    await owner.close()
    await f.service.close()


@pytest.mark.asyncio
async def test_event_only_plugin_response_blocks_companion_until_playback_finishes():
    from tests.extensions.test_service import connected, event, make_service

    f = setup()
    owner, _, _ = runtime(f)
    extension = make_service(on_response_state_changed=owner.refresh_busy)
    owner.foreground_generation = extension.has_active_response
    extension.proactive.responding_for.side_effect = lambda _: owner.busy
    item, _ = await connected(extension, autonomous=False)
    owner.bind(item.fish_id, item.session_id)
    try:
        receipt = await extension.request(item, "push", event("一条直播弹幕"))
        assert receipt["disposition"] == "responding"
        assert owner.active and owner.busy
        assert owner._busy_started == f.clock.now
        f.clock.now += timedelta(seconds=1)
        received = f.clock.now
        unit = HistoryUnit(uuid4(), "event", "忙碌期间的新变化", received, received)
        owner.observe(*owner.binding, unit, received_at=received)
        await owner.tick()
        assert not f.jev.calls
        stream = extension.streams.get(extension._owners[item.fish_id])
        await extension.streams.settle(stream.intent.id, interrupt=False)
        assert owner.busy  # Generated text is not a playback receipt.
        f.clock.now += timedelta(seconds=1)
        await extension.consumed(stream.intent.id, len(stream.text), False)
        assert not owner.busy
        late = HistoryUnit(uuid4(), "event", "延迟投影的旧变化", received, received)
        owner.observe(*owner.binding, late, received_at=received)
        assert not f.jev.calls
        owner.trigger("新的定时触发")
        await owner._decision
        assert len(f.jev.calls) == 1
    finally:
        await extension.close()
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_av_admission_only_sends_current_summaries_while_busy(admitted):
    f = setup()
    owner, contexts, _ = runtime(f)
    owner.foreground_busy = True
    contexts.load_for_proactive = AsyncMock(
        side_effect=AssertionError("Observation filtering must not load context")
    )
    assert not await owner.admit_window(SemanticDocument(summary_version=4))
    assert not f.jev.filter_calls
    f.jev.admitted = admitted
    document = SemanticDocument(
        summary_version=4,
        audiovisualSummary="屏幕中央的宠物气泡文字更新。",
        activitySummary="无明显电脑操作。",
        audio="未总结的声音不属于本次过滤输入。",
    )
    assert await owner.admit_window(document) is admitted
    assert len(f.jev.filter_calls) == 1 and not f.jev.calls
    assert f.jev.filter_calls[0] == {
        "audiovisualSummary": "屏幕中央的宠物气泡文字更新。",
        "activitySummary": "无明显电脑操作。",
    }
    contexts.load_for_proactive.assert_not_awaited()
    await owner.close()
    await f.service.close()


def test_interrupted_state_contains_only_supplied_prefix_and_small_marker():
    state = render_state(
        [{"role": "assistant", "content": "说到这里\n" + INTERRUPTION_MARKER}],
        NOW,
        "新输入",
    )
    assert "a: 说到这里(被打断)" in state
    assert INTERRUPTION_MARKER not in state


@pytest.mark.asyncio
async def test_http_jev_choice_uses_official_endpoint_and_text_only_state():
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(
            200, json={"answers": {"next_action": {"type": "choice", "choice": "s"}}}
        )

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    assert await client.decide(config, "用户刚说你好", ("s", "wait")) == "s"
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert requests[0].headers["Authorization"] == "Bearer test-only"
    assert "test-only" not in requests[0].content.decode()
    assert list(body["questions"]["next_action"]["criteria"]) == ["s", "wait"]
    await client.close()


@pytest.mark.asyncio
async def test_jev_failures_say_they_came_from_jev(monkeypatch):
    from fish_memory import failures

    health = failures.ServiceHealth()
    monkeypatch.setattr(failures, "health", health)
    replies = [httpx.Response(401), httpx.ReadTimeout("slow")]

    async def handle(request):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    try:
        for kind in ("key", "timeout"):
            with pytest.raises(Exception) as raised:
                await client.decide(config, "用户刚说你好", ("s", "wait"))
            assert failures.failure_of(raised.value) == failures.ServiceFailure(
                "jev", kind
            )
        assert health.problems() == (failures.ServiceFailure("jev", "key"),)
        with pytest.raises(failures.MissingKey) as raised:
            await client.decide(
                ExperimentalConfiguration(enabled=True), "你好", ("s", "wait")
            )
        assert failures.failure_of(raised.value) == failures.ServiceFailure(
            "jev", "key"
        )
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("probability,admitted", [(0.49, False), (0.5, True)])
async def test_http_jev_observation_preserves_state_as_an_object(probability, admitted):
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(
            200, json={"answers": {"new_observation": {"noul": probability}}}
        )

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    state = {
        "audiovisualSummary": '字幕显示 "hello"。\n下一行仍含字面量 \\n。',
        "activitySummary": "操作者切换了应用。",
    }
    try:
        assert await client.admit(config, state) is admitted
        body = json.loads(requests[0].content)
        assert body["state"] == state
        assert isinstance(body["state"], dict)
        assert list(body["questions"]) == ["new_observation"]
        assert "test-only" not in requests[0].content.decode()
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, -0.1, 1.1, "0.9", None])
async def test_jev_rejects_invalid_observation_probability(value):
    async def handle(request):
        return httpx.Response(
            200, json={"answers": {"new_observation": {"noul": value}}}
        )

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    with pytest.raises(ValueError):
        await client.admit(
            config, {"audiovisualSummary": "画面", "activitySummary": ""}
        )
    await client.close()


@pytest.mark.asyncio
async def test_ordinary_pipeline_keeps_original_router_when_experiment_is_off():
    f = setup()
    f.service.configure(ExperimentalConfiguration())

    class OrdinaryModel:
        def json_stream_request_payload(self, messages):
            return {"messages": messages}

        async def stream_json(self, messages, **kwargs):
            yield ChatStreamChunk(
                content='{"m":"neutral","a":"普通回答。","r":0,"e":0}'
            )

    pipeline = ParallelAnswerGenerationPipeline(
        OrdinaryModel(),
        recall_service=None,
        recall_context_token_budget=5000,
        enable_search=False,
        experiment=f.service,
    )
    sink = PromotableAnswerSink()
    prepared = pipeline.prepare_initial(
        uuid4(),
        Contexts().loaded.messages,
        (),
        sink,
        MessagePerformanceTracker(),
        enable_home=False,
    )
    assert isinstance(prepared, PreparedInitialAnswer)
    result = await prepared.initial_ready()
    assert result.name == "recall" and result.recall_depth == 0
    assert sink.snapshot().content == "普通回答。"
    assert not f.jev.calls
    await prepared.cancel()
    await f.service.close()


@pytest.mark.asyncio
async def test_home_request_takes_the_route_that_can_control_home_while_chat_picks():
    f = setup("r")
    pipeline = ParallelAnswerGenerationPipeline(
        f.models,
        recall_service=f.recall,
        recall_context_token_budget=5000,
        enable_search=True,
        experiment=f.service,
    )
    prepared = pipeline.prepare_initial(
        uuid4(),
        Contexts().loaded.messages,
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=True,
    )
    assert isinstance(prepared, PreparedInitialAnswer)
    assert not f.jev.calls
    await prepared.cancel()
    await f.service.close()


@pytest.mark.asyncio
async def test_experiment_omits_home_and_work_survives_speech_stop():
    f = setup("r")
    pipeline = ParallelAnswerGenerationPipeline(
        f.models,
        recall_service=f.recall,
        recall_context_token_budget=5000,
        enable_search=True,
        experiment=f.service,
    )
    prepared = pipeline.prepare_initial(
        uuid4(),
        Contexts().loaded.messages,
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=False,
    )
    assert isinstance(prepared, ExperimentalPreparedAnswer)
    f.recall.gate.clear()
    await prepared.release_to_public(
        PromotableAnswerSink(),
        source_message_id=uuid4(),
        answer_message_id=uuid4(),
        runtime=None,
        session_id=uuid4(),
    )
    await asyncio.sleep(0)
    await prepared.cancel()
    assert set(f.models.channels) <= {"s", "r", "w"}
    f.recall.gate.set()
    await settle_work(f)
    assert len(f.observations.events) == 2
    await f.service.close()


@pytest.mark.asyncio
async def test_agent_task_uses_selected_goal_and_publishes_result_event():
    f = setup("t")
    calls = []

    class Agent:
        async def available(self, context):
            return True

        async def execute(self, intent, context):
            calls.append(intent.purpose)
            assert intent.evidence and "继续陪我聊" in intent.evidence[0].text
            return SimpleNamespace(text="研究完成：找到了三份原始资料。")

    f.service.tasks = Agent()
    prepared = prepare(f)
    await prepared.initial_ready()
    assert "t" in f.models.channels and not calls
    await prepared.release_to_public(
        PromotableAnswerSink(),
        source_message_id=uuid4(),
        answer_message_id=uuid4(),
        runtime=None,
        session_id=uuid4(),
    )
    await settle_work(f)
    assert len(calls) == 1
    assert "研究结果出来了" in f.observations.events[-1][1].payload["context_text"]
    assert not f.models.summaries
    await f.service.close()


@pytest.mark.asyncio
async def test_each_judgement_has_its_own_switch_apart_from_speaking_up():
    f = setup("s")
    owner, _, _ = runtime(f)
    f.service.configure(f.service.configuration.model_copy(update={"enabled": False}))
    # 主动陪伴 off: no speaking up, but replies are still picked and 内心 still
    # judges what it saw and heard.
    assert not f.service.enabled and not owner.active
    assert f.service.chat_pick_enabled
    assert (await prepare(f).initial_ready()).name == "s"
    f.jev.admitted = True
    document = SemanticDocument(
        summary_version=4, audiovisualSummary="视频里在讲火山。", activitySummary=""
    )
    assert await owner.admit_window(document)
    assert len(f.jev.filter_calls) == 1
    f.service.configure(
        f.service.configuration.model_copy(update={"chat_pick_enabled": False})
    )
    assert not f.service.chat_pick_enabled
    await owner.close()
    await f.service.close()


def test_channel_schema_holds_the_model_to_the_parsed_object_in_order():
    # Plain JSON mode let qwen answer "[0]"; the parser then dropped the reply.
    spoken = channel_response_schema(query=False)
    assert list(spoken["properties"]) == ["m", "a", "e"]
    assert spoken["required"] == ["m", "a", "e"]
    assert spoken["additionalProperties"] is False
    assert list(channel_response_schema(query=True)["properties"]) == [
        "m",
        "a",
        "q",
        "e",
    ]


class BlankFirst:
    """Leaves what to do empty the first time, as qwen once did live."""

    def __init__(self, second):
        self.replies = ['{"m":"neutral","a":"我去想想。","q":"","e":0}', second]

    async def stream_json(self, messages, **kwargs):
        raw = self.replies.pop(0)
        for start in range(0, len(raw), 7):
            yield ChatStreamChunk(content=raw[start : start + 7])


class Heard:
    def __init__(self):
        self.text = ""

    async def set_emotion(self, emotion):
        pass

    async def append(self, text, *, continuation=False):
        self.text += text

    async def flush(self):
        pass

    async def flush_through(self, end_offset):
        pass

    async def set_end(self, end):
        pass


CONTEXT = [
    {"role": "system", "content": "你陪伴用户。"},
    {"role": "user", "content": "你还记得我们看花那天吗"},
]


@pytest.mark.asyncio
async def test_a_blank_query_is_asked_again_without_a_second_line():
    heard = Heard()
    chat = BlankFirst('{"m":"happy","a":"另一句。","q":"和用户一起看花那天","e":0}')

    result = await generate_channel(chat, "r", CONTEXT, heard)

    assert result.query == "和用户一起看花那天"
    assert heard.text == "我去想想。"


@pytest.mark.asyncio
async def test_a_line_lost_with_its_blank_query_is_said_on_the_second_ask():
    heard = Heard()
    chat = BlankFirst('{"m":"happy","a":"另一句。","q":"和用户一起看花那天","e":0}')

    async def at_once(messages, **kwargs):
        yield ChatStreamChunk(content=chat.replies.pop(0))

    chat.stream_json = at_once
    result = await generate_channel(chat, "r", CONTEXT, heard)

    assert result.query == "和用户一起看花那天"
    assert heard.text == "另一句。"


@pytest.mark.asyncio
async def test_a_query_blank_twice_still_fails():
    chat = BlankFirst('{"m":"neutral","a":"再说一句。","q":" ","e":0}')

    with pytest.raises(BlankFieldError):
        await generate_channel(chat, "t", CONTEXT, PromotableAnswerSink())


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["r", "w", "t"])
async def test_the_line_before_acting_leaves_the_outcome_open(action):
    f = setup(action)
    await generate_channel(f.models, action, CONTEXT, PromotableAnswerSink())

    rules = f.models.requests[0][1][0]["content"]
    assert "不预先说" in rules
    assert "不是对用户说的话" in rules


@pytest.mark.asyncio
@pytest.mark.parametrize("attached", [True, False])
async def test_a_turn_with_files_tells_jev_the_reply_sees_them(attached):
    f = setup()
    last = {"role": "user", "content": "History:\nu: 看看这个 [发来文件：a.txt]"}
    if attached:
        last["user_attachments"] = ({"kind": "text", "name": "a.txt", "text": "内容"},)
    system = {"role": "system", "content": "你陪伴用户。"}
    context = LoadedAnswerContext([system, last], ())
    prepared = ExperimentalPreparedAnswer(
        f.service, uuid4(), context, PromotableAnswerSink(), MessagePerformanceTracker()
    )
    try:
        note = "用户这次发来的图片和文件，内容直接可见。"
        assert (note in prepared.state) is attached
        assert (note in prepared.expression_context[-1]["content"]) is attached
    finally:
        await prepared.cancel()
        await f.service.close()
