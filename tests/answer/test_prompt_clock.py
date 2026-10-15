import copy
import json
from functools import partial

import pytest

from fish_memory.agent.prompt import build_task_lane_messages
from fish_memory.agent.routing import TaskLaneContext
from fish_memory.answer.context import (
    ChatMessage,
    add_initial_answer_contract,
    build_chat_messages,
    build_web_search_context_messages,
    with_request_time_last,
)
from fish_memory.answer.generation import AnswerGenerationPipeline
from fish_memory.answer.lane_context import (
    build_home_lane_messages,
    build_lane_base_messages,
    build_recall_lane_messages,
    build_web_lane_messages,
)
from fish_memory.home.answer import build_home_query_continuation_messages
from fish_memory.home.models import HomeActionStatus, HomeIntentType, HomeRunOutcome
from fish_memory.llm.ark_search import _input_messages
from fish_memory.llm.client import (
    _diagnostic_messages,
    _provider_messages,
    _without_environment_images,
)


def context_at(timestamp, with_image):
    result = build_chat_messages(
        "宠物生日：2026-09-13。",
        [],
        static_profile="用户喜欢音乐。",
        history_text=(
            "2026-09-22 22:50:00+08:00 - 2026-09-22 23:00:00+08:00\na: 刚才说完的话。"
        ),
        current_input="当前时间：这是用户输入的原文，不应改写。",
    )
    result[-1]["request_time"] = timestamp
    if with_image:
        result[-1]["environment_image"] = {
            "data_url": "data:image/jpeg;base64,frame",
            "digest": "a" * 64,
            "byte_count": 5,
            "width": 10,
            "height": 10,
            "observed_at": "2026-09-22T23:00:00+08:00",
        }
    return result


def continuation(messages):
    # No model or Recall execution is needed to build the real continuation.
    pipeline = AnswerGenerationPipeline(
        llm=None,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
    )
    base = build_lane_base_messages(
        messages, include_memory=True, include_environment_image=True
    )
    return pipeline._build_continuation_messages(
        base, "我想起来了。", None, search_context="本次查到的事实。"
    )


@pytest.mark.parametrize(
    "builder",
    [
        add_initial_answer_contract,
        build_recall_lane_messages,
        build_web_lane_messages,
        build_home_lane_messages,
        partial(build_task_lane_messages, task_context=TaskLaneContext.empty()),
        partial(build_lane_base_messages, include_memory=True),
        partial(build_lane_base_messages, include_memory=False),
        build_web_search_context_messages,
        continuation,
        partial(
            build_home_query_continuation_messages,
            shallow_answer="正在检查。",
            outcome=HomeRunOutcome(
                intent=HomeIntentType.QUERY,
                status=HomeActionStatus.FAILED,
                error="离线",
            ),
        ),
    ],
    ids=[
        "ordinary",
        "recall",
        "web",
        "home",
        "task",
        "recall-base",
        "web-base",
        "search",
        "continuation",
        "home-continuation",
    ],
)
@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize(
    "transport", [_provider_messages, _input_messages, _diagnostic_messages]
)
def test_clock_changes_only_final_provider_text(builder, with_image, transport):
    normalized = []
    for second in ("20", "25"):
        timestamp = f"当前时间：2026-09-22T23:01:{second}+08:00"
        source = context_at(timestamp, with_image)
        original = copy.deepcopy(source)
        messages = builder(source)
        wire = transport(messages)
        assert source == original
        assert json.dumps(wire, ensure_ascii=False).count(timestamp) == 1
        assert timestamp not in json.dumps(wire[:-1], ensure_ascii=False)
        assert "宠物生日：2026-09-13。" in str(source)
        assert "当前时间：这是用户输入的原文，不应改写。" in str(wire)
        last = wire[-1]["content"]
        if isinstance(last, str):
            assert last.endswith("\n\n" + timestamp)
            wire[-1]["content"] = last.removesuffix("\n\n" + timestamp)
        else:
            assert last[-1]["text"] == timestamp
            last.pop()
        normalized.append(wire)
    assert normalized[0] == normalized[1]


def test_appended_results_and_image_retry_keep_one_clock_at_end():
    timestamp = "当前时间：2026-09-22T23:01:20+08:00"
    source = context_at(timestamp, True)
    source.append({"role": "user", "content": "新到的工具结果"})
    original = copy.deepcopy(source)
    for messages in (source, _without_environment_images(source)):
        moved = with_request_time_last(messages)
        assert moved == with_request_time_last(moved)
        assert moved[-1]["request_time"] == timestamp
        for transport in (_provider_messages, _input_messages, _diagnostic_messages):
            wire = transport(messages)
            assert timestamp not in json.dumps(wire[:-1], ensure_ascii=False)
            assert timestamp in str(wire[-1])
    assert source == original


def test_explicit_latest_clock_wins_without_modifying_dialogue_dates():
    source: list[ChatMessage] = [
        {"role": "system", "content": "当前时间：原始设定", "request_time": "旧时钟"},
        {"role": "user", "content": "当前时间：用户引用", "request_time": "新时钟"},
    ]
    result = with_request_time_last(source)
    assert [m["content"] for m in result] == [m["content"] for m in source]
    assert "request_time" not in result[0]
    assert result[-1]["request_time"] == "新时钟"
