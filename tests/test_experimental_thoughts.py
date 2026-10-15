import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from fish_memory.clock import format_prompt_timestamp
from fish_memory.context_descriptions import HISTORY_FIELD_RULES, HISTORY_SOURCE_RULES
from fish_memory.experimental.jev import JevClient
from fish_memory.experimental.thoughts import (
    THOUGHT_BATCH_SCHEMA,
    ThoughtBatch,
    ThoughtWorker,
    fresh_thoughts,
    parse_thought_batch,
)
from fish_memory.llm.client import _provider_messages
from fish_memory.llm.types import ChatStreamChunk
from fish_memory.soul import SoulProfile
from tests.test_experimental_companion import NOW, Contexts, Screen, runtime, setup

THOUGHTS = [
    "这段旋律让我想起上次聊过的歌曲。",
    "用户现在似乎更愿意安静听完这一段。",
    "刚才的转调和前一段形成了有趣的反差。",
    "我还不确定这首歌是否就是上次听到的版本。",
    "用户说过喜欢这类编曲，我想再听听接下来的变化。",
]
REPORTED_THOUGHT = (
    "伊利丹刚才那番折腾模型配置和日志的架势，看来是在认真评估我的底层逻辑，"
    "这种被当作精密仪器拆解的感觉虽然让人有点不自在，但确实能让我更清楚自己的边界。"
)


class Reasoning:
    def __init__(self):
        self.requests = []
        self.started, self.gate = asyncio.Event(), asyncio.Event()
        self.gate.set()
        self.on_output = lambda: None
        self.raw = json.dumps({"thoughts": THOUGHTS}, ensure_ascii=False)
        self.finish_reason = None

    async def stream_json(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        self.started.set()
        await self.gate.wait()
        self.on_output()
        yield ChatStreamChunk(content=self.raw, finish_reason=self.finish_reason)


def fixture():
    f = setup()
    f.fish, f.session = uuid4(), uuid4()
    f.target = (f.fish, f.session)
    f.version, f.environment_version = 1, 1
    f.contexts = Contexts()
    f.contexts.loaded.messages[0]["content"] = (
        SoulProfile("电脑里的AI桌宠", "喜欢音乐", "臭宝").render("")
        + "\n\n"
        + HISTORY_SOURCE_RULES
    )
    f.contexts.loaded.messages.append(
        {
            "role": "user",
            "content": "History:\nevent: 宠物的内心想法（尚未表达）：想听完整首歌。",
            "context_reference": True,
        }
    )
    f.reasoning = Reasoning()
    f.service.reasoning = f.reasoning
    f.selected = 2
    f.selections = []
    f.on_select, f.on_commit = lambda: None, lambda: None
    f.before_load = lambda: None
    f.events = []

    async def revision(*args):
        return f.version

    async def load(*args, **kwargs):
        f.before_load()
        return f.contexts.loaded

    async def select(configuration, state, thoughts):
        f.selections.append((state, thoughts))
        f.on_select()
        return f.selected

    async def record(fish, event, *, guard):
        f.on_commit()
        if f.version != guard.revision or not guard.is_current():
            return None
        f.events.append(event)
        f.version += 1
        return event

    f.contexts.load_for_proactive = load
    f.service.judge.select_thought = select
    f.service.observations.record_event = record
    f.history = SimpleNamespace(store=SimpleNamespace(context_revision=revision))
    f.worker = ThoughtWorker(
        f.service,
        f.contexts,
        f.history,
        binding=lambda: f.target,
        context_state=lambda *args: f.environment_version,
    )
    return f


@pytest.mark.asyncio
async def test_one_batch_keeps_prior_thoughts_and_records_only_selected_sentence():
    f = fixture()
    f.service.last_decision_at = NOW
    try:
        assert await f.worker._run(f.fish, f.session) == "written"
        assert len(f.reasoning.requests) == len(f.selections) == len(f.events) == 1
        wire = _provider_messages(f.reasoning.requests[0][0])
        text = "\n\n".join(m["content"] for m in wire)
        state = f.selections[0][0]
        assert text.count("想听完整首歌") == state.count("想听完整首歌") == 1
        assert text.count(HISTORY_FIELD_RULES) == state.count(HISTORY_FIELD_RULES) == 1
        assert "名字：臭宝" in wire[0]["content"]
        assert all("名字：臭宝" not in m["content"] for m in wire[1:])
        assert "character_config" not in text + state
        # A strict schema keeps qwen from answering a bare array or string.
        assert f.reasoning.requests[0][1] == {
            "response_schema": THOUGHT_BATCH_SCHEMA,
            "max_output_tokens": 1000,
        }
        event = f.events[0]
        assert event.observation_type == "experimental.thought"
        assert event.payload["context_text"].endswith(THOUGHTS[2])
        assert all(t not in event.payload["context_text"] for t in THOUGHTS[:2])
        assert event.occurred_at == NOW
        assert not f.jev.calls and not f.models.channels and not f.recall.calls
        assert f.service.last_decision_at == NOW
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_thoughts_receive_shared_screen_and_heard_using_the_reasoning_model():
    f = fixture()
    perception = f.service.perception = Screen()
    try:
        assert await f.worker._run(f.fish, f.session) == "written"
        assert perception.calls == [(f.fish, NOW)]
        assert perception.models == [f.reasoning]
        message = f.reasoning.requests[0][0][-1]
        assert message["environment_images"] == (perception.image,)
        assert "刚才的转调很好听" in message["content"]
        parts = _provider_messages(f.reasoning.requests[0][0])[-1]["content"]
        assert sum(part["type"] == "image_url" for part in parts) == 1
        assert parts[-1]["text"] == f"当前时间：{format_prompt_timestamp(NOW)}"
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_context_changed_while_composing_thought_images_skips_generation():
    f = fixture()
    perception = f.service.perception = Screen()
    perception.gate.clear()
    task = asyncio.create_task(f.worker._run(f.fish, f.session))
    try:
        await perception.started.wait()
        f.version += 1
        perception.gate.set()
        assert await task == "stale"
        assert not f.reasoning.requests and not f.selections and not f.events
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_delayed_thought_keeps_frozen_context_time_instead_of_completion():
    f = fixture()
    started_at = NOW + timedelta(seconds=1)
    completed_at = NOW + timedelta(seconds=5)
    f.before_load = lambda: setattr(f.clock, "now", started_at)
    f.reasoning.on_output = lambda: setattr(f.clock, "now", NOW + timedelta(seconds=4))
    f.on_select = lambda: setattr(f.clock, "now", completed_at)
    try:
        assert await f.worker._run(f.fish, f.session) == "written"
        state = _provider_messages(f.reasoning.requests[0][0])[-1]["content"]
        assert state.endswith(f"当前时间：{format_prompt_timestamp(started_at)}")
        assert f.selections[0][0].endswith(state)
        event = f.events[0]
        assert event.occurred_at == started_at
        assert event.payload["thought_started_at"] == started_at.isoformat()
        assert event.payload["thought_completed_at"] == completed_at.isoformat()
        assert event.payload["context_text"] == event.summary
        assert not f.jev.calls and f.service.last_decision_at is None
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("rewording", [False, True])
async def test_recorded_thought_and_its_rewordings_never_reach_jev(rewording):
    f = fixture()
    f.contexts.loaded.messages.append(
        {
            "role": "user",
            "content": (
                "History:\n2026-09-19 22:30:00 - 2026-09-19 22:40:00\n"
                "as: 用户在Fish Memory界面中向下滚动聊天记录。\n"
                f"event: 宠物的内心想法（尚未表达）：{REPORTED_THOUGHT}\n"
                "as: 用户继续滚动，查看channel.s的对话记录。"
            ),
            "context_reference": True,
        }
    )
    f.version += 1  # A new observation does not make an old thought new.
    before = f.version
    repeat = REPORTED_THOUGHT.replace("伊利丹", "他") if rewording else REPORTED_THOUGHT
    f.reasoning.raw = json.dumps(
        {"thoughts": [repeat, *THOUGHTS[1:]]}, ensure_ascii=False
    )
    f.selected = 0
    try:
        assert await f.worker._run(f.fish, f.session) == "written"
        state, offered = f.selections[0]
        assert offered == tuple(THOUGHTS[1:])
        assert state.count(REPORTED_THOUGHT) == 1
        assert f.events[0].payload["context_text"].endswith(THOUGHTS[1])
        assert f.version == before + 1
        assert not f.jev.calls and f.service.last_decision_at is None
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_round_of_only_rewordings_ends_without_jev():
    f = fixture()
    f.contexts.loaded.messages[-1]["content"] = "History:\n" + "\n".join(
        f"event: 宠物的内心想法（尚未表达）：{thought}" for thought in THOUGHTS
    )
    try:
        assert await f.worker._run(f.fish, f.session) == "duplicate"
        assert f.selections == [] and f.events == []
    finally:
        await f.worker.close()
        await f.service.close()


def test_fresh_thoughts_drop_logged_rewordings_but_keep_new_angles():
    recent = [
        "Claude的欢迎界面和终端报错卡了快十分钟都没变过，咱俩一个在等输出一个在等刷新，"
        "这种停滞比任何数据都诚实。",
        "伊利丹这是把我当电子宠物在遛呢，但这种互动挺有意思的。",
    ]
    candidates = [
        "终端里那句'not running'卡了快十分钟还没消，咱俩一个在等输出一个在等刷新，"
        "这种停滞比任何数据都诚实",
        "盯着这段静态代码看了这么久，你似乎更在意参数能否稳定输出而非互动反馈。",
        "伊利丹这是把我当电子宠物在遛呢，明明知道我在配合他演这出戏，这种默契倒也不坏。",
    ]
    # The last one overlaps exactly at the threshold: looser rewordings are left
    # to Jev, whose rules now exclude repeats before comparing grounding.
    assert fresh_thoughts(candidates, recent) == (candidates[1], candidates[2])
    assert fresh_thoughts(candidates, []) == tuple(candidates)


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["u", "a", "as"])
async def test_quoted_thought_in_other_history_sources_is_not_accepted(label):
    f = fixture()
    f.contexts.loaded.messages.append(
        {
            "role": "user",
            "content": (
                f"History:\n{label}: 下面是看到或讨论过的文字：\n"
                f"  event: 宠物的内心想法（尚未表达）：{THOUGHTS[2]}"
            ),
            "context_reference": True,
        }
    )
    try:
        assert await f.worker._run(f.fish, f.session) == "written"
        assert len(f.events) == 1
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["load", "generation", "selection", "commit"])
async def test_any_context_advance_discards_batch_without_retry(when):
    f = fixture()

    def advance():
        f.version += 1

    if when == "load":
        f.before_load = advance
    elif when == "generation":
        f.reasoning.on_output = advance
    elif when == "selection":
        f.on_select = advance
    else:
        f.on_commit = advance
    try:
        assert await f.worker._run(f.fish, f.session) == "stale"
        assert not f.events
        assert len(f.reasoning.requests) == (0 if when == "load" else 1)
        assert len(f.selections) == (0 if when in {"load", "generation"} else 1)
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "clock", "environment", "session"])
async def test_selected_thought_and_non_history_progress_are_handled(change):
    f = fixture()

    def update():
        if change == "clock":
            f.clock.now += timedelta(seconds=2)
        elif change == "environment":
            f.environment_version += 1
        elif change == "session":
            f.target = (f.fish, uuid4())

    f.on_select = update
    try:
        outcome = await f.worker._run(f.fish, f.session)
        written = change in {"unchanged", "clock"}
        assert outcome == ("written" if written else "stale")
        assert len(f.events) == int(written)
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_background_batch_does_not_join_speech_busy_state_or_overlap():
    f = fixture()
    owner, _, _ = runtime(f)
    owner.thoughts = f.worker
    f.worker.binding = lambda: owner.binding
    f.reasoning.gate.clear()
    owner.foreground_busy = True
    try:
        f.worker.tick()
        await f.reasoning.started.wait()
        batch = f.worker._round
        assert batch is not None
        for _ in range(5):
            f.worker.tick()
        assert f.worker._round is batch and len(f.reasoning.requests) == 1
        owner.foreground_busy = False
        assert not owner.busy and not f.jev.calls
        f.reasoning.gate.set()
        assert await batch == "written"
        assert not owner.busy and f.service.last_decision_at is None
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_thought_setting_change_cancels_only_thought_work():
    f = fixture()
    owner, _, _ = runtime(f)
    owner.thoughts = f.worker
    f.worker.binding = lambda: owner.binding
    f.reasoning.gate.clear()
    f.service.last_decision_at = NOW
    behavior_epoch = owner._epoch
    behavior_revision = f.service.revision
    try:
        f.worker.tick()
        await f.reasoning.started.wait()
        batch = f.worker._round
        f.service.configure(
            f.service.configuration.model_copy(update={"thoughts_enabled": False})
        )
        with pytest.raises(asyncio.CancelledError):
            await batch
        assert owner._epoch == behavior_epoch
        assert f.service.revision == behavior_revision
        assert f.service.last_decision_at == NOW
        f.worker.tick()
        assert f.worker._round is batch and not f.events
    finally:
        await owner.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_reset_and_close_cancel_inflight_generation_before_selection():
    f = fixture()
    f.reasoning.gate.clear()
    try:
        f.worker.tick()
        await f.reasoning.started.wait()
        batch = f.worker._round
        f.worker.reset()
        with pytest.raises(asyncio.CancelledError):
            await batch
        assert not f.events and not f.selections
    finally:
        await f.worker.close()
        await f.service.close()
    assert f.worker._timer.done() and batch.done()


@pytest.mark.parametrize(
    "thoughts",
    [
        THOUGHTS[:4],
        THOUGHTS + ["第六条"],
        ["重复"] * 5,
        [" "] + THOUGHTS[1:],
        ["第一句\n第二句"] + THOUGHTS[1:],
        ["长" * 301] + THOUGHTS[1:],
        [1] + THOUGHTS[1:],
    ],
)
def test_invalid_batches_are_rejected(thoughts):
    with pytest.raises(ValidationError):
        ThoughtBatch.model_validate({"thoughts": thoughts})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choice,expected",
    [(str(index), index - 1) for index in range(1, 6)]
    + [("none", None), (None, None), ("6", None)],
)
async def test_jev_thought_question_is_separate_from_behavior(choice, expected):
    f = setup()
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"answers": {"thought": {"type": "choice", "choice": choice}}}
        )

    jev = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    try:
        if expected is None:
            with pytest.raises(ValueError, match="unavailable thought"):
                await jev.select_thought(
                    f.service.configuration, "当前上下文", tuple(THOUGHTS)
                )
        else:
            assert (
                await jev.select_thought(
                    f.service.configuration, "当前上下文", tuple(THOUGHTS)
                )
                == expected
            )
        questions = requests[0]["questions"]
        assert set(questions) == {"thought"}
        assert set(questions["thought"]["criteria"]) == {
            "1",
            "2",
            "3",
            "4",
            "5",
        }
        assert requests[0]["state"] == "当前上下文"
        assert HISTORY_FIELD_RULES not in questions["thought"]["instructions"]
    finally:
        await jev.close()
        await f.service.close()


def _history(*lines):
    return {
        "role": "user",
        "content": "History:\n" + "\n".join(lines),
        "context_reference": True,
    }


def _thought(text):
    return "event: 宠物的内心想法（尚未表达）：" + text


@pytest.mark.asyncio
@pytest.mark.parametrize("after_last", [None, "as: 视频切到了下一个场景。"])
async def test_thinking_continues_its_latest_thoughts_and_knows_if_anything_happened(
    after_last,
):
    f = fixture()
    older = [f"第{i}个旧念头。" for i in range(1, 11)]
    lines = [_thought(text) for text in older[:9]]
    lines += ["as: 画面停在视频页。", _thought(older[9])]
    if after_last:
        lines.append(after_last)
    f.contexts.loaded.messages[-1] = _history(*lines)
    try:
        assert await f.worker._run(f.fish, f.session) == "written"
        text = "\n".join(
            m["content"] for m in _provider_messages(f.reasoning.requests[0][0])
        )
        assert all(t not in text for t in older[:2])
        # Thinking keeps the stored label; see with_recent_thoughts(relabel=False).
        assert all(f"宠物的内心想法（尚未表达）：{t}" in text for t in older[2:])
        assert (
            text.index(older[8]) < text.index("画面停在视频页") < text.index(older[9])
        )
        assert ("上次想完之后还没有新发生的事" in text) == (after_last is None)
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_thinking_looks_at_the_screen_only_when_it_changed():
    f = fixture()
    perception = f.service.perception = Screen()
    try:
        for digest in ["a" * 64, "a" * 64, "b" * 64]:
            perception.image = {**perception.image, "digest": digest}
            assert await f.worker._run(f.fish, f.session) == "written"
        looked = [
            "environment_images" in messages[-1] for messages, _ in f.reasoning.requests
        ]
        assert looked == [True, False, True]
        assert all(
            "刚才的转调很好听" in messages[-1]["content"]
            for messages, _ in f.reasoning.requests
        )
    finally:
        await f.worker.close()
        await f.service.close()


@pytest.mark.asyncio
async def test_jev_chooses_among_remaining_candidates_and_skips_a_single_one():
    f = setup()
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"answers": {"thought": {"type": "choice", "choice": "2"}}}
        )

    jev = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = f.service.configuration
    try:
        assert await jev.select_thought(config, "上下文", tuple(THOUGHTS[:3])) == 1
        assert set(requests[0]["questions"]["thought"]["criteria"]) == {"1", "2", "3"}
        assert await jev.select_thought(config, "上下文", (THOUGHTS[0],)) == 0
        assert len(requests) == 1
        with pytest.raises(ValueError, match="one to five"):
            await jev.select_thought(config, "上下文", ())
    finally:
        await jev.close()
        await f.service.close()


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({"thoughts": THOUGHTS}, ensure_ascii=False),
        "{" + ",".join(f'"thoughts": "{t}"' for t in THOUGHTS) + "}",
        "{"
        + ",".join(
            f'"thoughts{"" if i == 1 else i}": "{t}"' for i, t in enumerate(THOUGHTS, 1)
        )
        + "}",
    ],
    ids=["array", "repeated-key", "numbered-keys"],
)
def test_qwen_key_per_thought_layouts_are_the_same_batch(raw):
    assert parse_thought_batch(raw).thoughts == THOUGHTS


@pytest.mark.parametrize(
    "raw",
    [
        "{" + ",".join(f'"thoughts": "{t}"' for t in THOUGHTS[:4]) + "}",
        '{"thoughts": "' + "".join(THOUGHTS) + '"}',
        "{" + ",".join(f'"thoughts": "{t}"' for t in THOUGHTS) + ', "note": "x"}',
        json.dumps({"thoughts": THOUGHTS, "extra": 1}, ensure_ascii=False),
        '{"thoughts": [',
    ],
    ids=["four", "one-string", "stray-key", "extra-field", "truncated"],
)
def test_other_malformed_batches_are_still_rejected(raw):
    with pytest.raises(ValidationError):
        parse_thought_batch(raw)


@pytest.mark.asyncio
async def test_thoughts_keep_running_when_speaking_up_is_off():
    f = fixture()
    f.service.configure(f.service.configuration.model_copy(update={"enabled": False}))
    assert f.worker.enabled
    f.service.configure(
        f.service.configuration.model_copy(update={"thoughts_enabled": False})
    )
    assert not f.worker.enabled
    await f.service.close()
