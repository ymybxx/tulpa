import copy
import json
from datetime import timedelta
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from fish_memory.answer.context import build_chat_messages
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.answer.speech_style import SPOKEN_TEXT_TURN_REMINDER
from fish_memory.clock import format_prompt_timestamp
from fish_memory.context_descriptions import HISTORY_FIELD_RULES
from fish_memory.experimental.channels import (
    ChannelResult,
    generate_channel,
    render_state,
)
from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.context import expression_context
from fish_memory.experimental.jev import JevClient
from fish_memory.experimental.perception import PerceptionInput
from fish_memory.llm.ark_search import _input_messages
from fish_memory.llm.client import _provider_messages
from fish_memory.llm.types import ChatStreamChunk
from fish_memory.soul import SoulProfile
from tests.test_experimental_companion import NOW, Models, Screen, setup


def context_at(now):
    timestamp = f"当前时间：{format_prompt_timestamp(now)}"
    messages = build_chat_messages(
        SoulProfile("电脑里的AI桌宠", "自然交流", "臭宝").render("") + "\n" + timestamp,
        [],
        static_profile="用户喜欢音乐。",
        history_text="as: 一首歌刚刚播放完。\na: 这段旋律很熟悉。",
        current_input="继续聊这首歌。",
    )
    messages[0]["system_request_time"] = timestamp
    return messages


def without_final_time(messages, timestamp):
    wire = copy.deepcopy(messages)
    content = wire[-1]["content"]
    if isinstance(content, str):
        assert content.endswith("\n\n" + timestamp)
        wire[-1]["content"] = content.removesuffix("\n\n" + timestamp)
    else:
        assert content[-1]["text"] == timestamp
        assert content.pop()["type"] in {"text", "input_text"}
    assert json.dumps(messages, ensure_ascii=False).count(timestamp) == 1
    return wire


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["s", "r", "t", "w"])
@pytest.mark.parametrize("with_image", [False, True])
async def test_channel_clock_changes_only_the_last_provider_text(action, with_image):
    model = Models()
    images = (Screen().image,) if with_image else ()
    wires = []
    for now in (NOW, NOW + timedelta(seconds=5)):
        messages = context_at(now)
        original = copy.deepcopy(messages)
        await generate_channel(
            model,
            action,
            expression_context(messages, now, "定时唤醒"),
            PromotableAnswerSink(),
            perception=PerceptionInput(
                images=images, heard="[00:01—00:02] 声源A：正在播放的原话。"
            ),
        )
        request = model.requests[-1][1]
        timestamp = f"当前时间：{format_prompt_timestamp(now)}"
        assert messages == original
        pair = []
        for transport in (_provider_messages, _input_messages):
            wire = transport(request)
            system_text = json.dumps(wire[0], ensure_ascii=False)
            serialized = json.dumps(wire, ensure_ascii=False)
            assert "名字：臭宝" in system_text
            assert serialized.count("名字：臭宝") == 1
            assert "character_config" not in serialized
            assert "字段说明：" in system_text
            assert serialized.count("字段说明：") == 1
            assert "当前时间：" not in json.dumps(wire[0], ensure_ascii=False)
            assert "latestCapturedAt" not in json.dumps(wire[0])
            assert '"request_time"' not in json.dumps(wire)
            assert '"response_reminder"' not in json.dumps(wire)
            assert serialized.count(SPOKEN_TEXT_TURN_REMINDER) == 2
            last = wire[-1]["content"]
            if isinstance(last, str):
                assert last.endswith(SPOKEN_TEXT_TURN_REMINDER + "\n\n" + timestamp)
            else:
                assert last[-2]["text"] == SPOKEN_TEXT_TURN_REMINDER
                assert last[-2]["type"] in {"text", "input_text"}
                assert last[-1]["text"] == timestamp
                if with_image:
                    assert any(
                        item["type"] in {"image_url", "input_image"}
                        for item in last[:-2]
                    )
            pair.append(without_final_time(wire, timestamp))
        wires.append(pair)
    # All role boundaries, persona, History, trigger and images remain identical.
    assert wires[0] == wires[1]


@pytest.mark.asyncio
async def test_decision_keeps_one_clock_after_the_complete_state(system_timezone):
    system_timezone("Asia/Shanghai")
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "answers": {
                    "next_action": {"type": "choice", "choice": "wait"},
                    "should_research": {"type": "noul", "noul": 0.1},
                }
            },
        )

    client = JevClient(client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    config = ExperimentalConfiguration(enabled=True, api_key=SecretStr("test-only"))
    try:
        for now in (NOW, NOW + timedelta(seconds=5)):
            state = render_state(context_at(now), now, "定时唤醒")
            assert state.endswith(f"当前时间：{format_prompt_timestamp(now)}")
            assert state.count("当前时间：") == 1
            assert "character_config" not in state
            assert state.count("名字：臭宝") == 1
            assert state.count(HISTORY_FIELD_RULES) == 1
            assert "u: # 记忆摘要" not in state
            assert "# 记忆摘要" in state and "u: 继续聊这首歌。" in state
            assert await client.decide(config, state) == "wait"
        first, second = requests
        assert first["questions"] == second["questions"]
        assert (
            first["state"].rsplit("\n\n", 1)[0] == second["state"].rsplit("\n\n", 1)[0]
        )
    finally:
        await client.close()


def test_relocating_clock_does_not_rewrite_historical_time_or_user_quotes():
    messages = context_at(NOW)
    old_time = "当前时间：2025-01-01T00:00:00+08:00"
    messages[-1]["content"] += "\nu: 请保留这段旧日志：\n" + old_time
    state = render_state(messages, NOW, "用户输入")
    expression = _provider_messages(expression_context(messages, NOW, "用户输入"))
    assert old_time in state
    assert old_time in json.dumps(expression, ensure_ascii=False)
    assert messages[-1]["content"].endswith(old_time)


@pytest.mark.asyncio
async def test_selected_web_search_keeps_clock_after_its_question_and_images():
    fixture = setup("w")
    requests = []

    class Search:
        async def stream_json(self, messages, **kwargs):
            assert kwargs == {"force_search": True}
            requests.append(messages)
            yield ChatStreamChunk(content='{"searchContext":"找到资料。"}')

    fixture.service.reasoning = Search()
    normalized = []
    try:
        for now in (NOW, NOW + timedelta(seconds=5)):
            fixture.clock.now = now
            source = context_at(now)
            source[-1]["environment_images"] = (Screen().image,)
            original = copy.deepcopy(source)
            result = await fixture.service._execute(
                uuid4(),
                uuid4(),
                ChannelResult("w", "这首歌是谁唱的？"),
                LoadedAnswerContext(source, ()),
                render_state(source, now, "联网搜索"),
            )
            assert result == "找到资料。"
            assert source == original
            timestamp = f"当前时间：{format_prompt_timestamp(now)}"
            normalized.append(
                [
                    without_final_time(transport(requests[-1]), timestamp)
                    for transport in (_provider_messages, _input_messages)
                ]
            )
        assert normalized[0] == normalized[1]
    finally:
        await fixture.service.close()


def test_changing_screen_capture_time_leaves_system_and_history_unchanged():
    source = expression_context(context_at(NOW), NOW, "定时唤醒")
    source[-1]["environment_images"] = (Screen().image,)
    later = copy.deepcopy(source)
    later[-1]["environment_images"][0]["observed_at"] = (
        NOW + timedelta(seconds=1)
    ).isoformat()
    later[-1]["environment_images"][0]["description"] = "一秒后的屏幕画面。"
    for transport in (_provider_messages, _input_messages):
        before, after = transport(source), transport(later)
        assert before[:-1] == after[:-1]
        assert before[-1]["content"][-1] == after[-1]["content"][-1]
        assert before[-1]["content"] != after[-1]["content"]
