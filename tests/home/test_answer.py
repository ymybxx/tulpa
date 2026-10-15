import importlib
from types import MappingProxyType

from fish_memory.home.models import (
    HomeActionStatus,
    HomeEntity,
    HomeIntentType,
    HomeQueryCategory,
    HomeQueryState,
    HomeRunOutcome,
    ResolvedQueryScope,
)


def answer_module():
    return importlib.import_module("fish_memory.home.answer")


def query_outcome() -> HomeRunOutcome:
    entity = HomeEntity(
        entity_id="light.living_room",
        domain="light",
        state="on",
        attributes=MappingProxyType({"brightness": 128, "secret": "omit"}),
        aliases=("客厅主灯",),
        area_name="客厅",
    )
    return HomeRunOutcome(
        intent=HomeIntentType.QUERY,
        status=HomeActionStatus.SUCCEEDED,
        query=ResolvedQueryScope(
            entities=(HomeQueryState(entity, HomeQueryCategory.ACTIVE),)
        ),
    )


def test_home_query_continuation_exposes_display_data_without_internal_ids() -> None:
    messages = answer_module().build_home_query_continuation_messages(
        [
            {"role": "system", "content": "role"},
            {"role": "user", "content": "客厅灯开着吗？"},
        ],
        "我看看。",
        query_outcome(),
    )

    prompt = messages[-1]["content"]
    assert '"append"' in prompt
    assert '"end"' not in prompt
    assert "light.living_room" not in prompt
    assert "客厅主灯" in prompt
    assert "客厅" in prompt
    assert '"brightness":128' in prompt
    assert "secret" not in prompt
    assert "按房间组织" in prompt


def test_home_query_failure_still_requires_a_completed_append() -> None:
    outcome = HomeRunOutcome(
        intent=HomeIntentType.QUERY,
        status=HomeActionStatus.FAILED,
        error="Home Assistant 未连接",
    )

    messages = answer_module().build_home_query_continuation_messages(
        [
            {"role": "system", "content": "role"},
            {"role": "user", "content": "客厅灯开着吗？"},
        ],
        "我看看。",
        outcome,
    )

    assert "Home Assistant 未连接" in messages[-1]["content"]
    assert "append 必须完成当前回答" in messages[-1]["content"]


def test_home_query_continuation_language_follows_current_user_content() -> None:
    messages = answer_module().build_home_query_continuation_messages(
        [
            {"role": "system", "content": "中文协议"},
            {"role": "user", "content": "Is the living-room light on?"},
        ],
        "Let me check.",
        query_outcome(),
    )

    prompt = messages[-1]["content"]
    assert "# 原始当前用户消息" in prompt
    assert '{"content":"Is the living-room light on?"}' in prompt
    assert "当前用户消息 content" in prompt
    assert "用户明确指定的语言" in prompt
    assert "当前消息的主要语言" in prompt
    assert "协议、示例、工具结果的语言" in prompt
