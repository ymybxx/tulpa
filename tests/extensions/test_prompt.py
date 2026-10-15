import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from fish_plugin_sdk import Activity

from fish_memory.answer.context import (
    ANSWER_LANGUAGE_RULES,
    INTERRUPTION_MARKER,
    add_perception_context,
    build_chat_messages,
)
from fish_memory.answer.lane_context import build_recall_lane_messages
from fish_memory.context_descriptions import HISTORY_SOURCE_RULES
from fish_memory.extensions.models import EventDefinition
from fish_memory.extensions.prompt import event_messages
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.soul import SoulProfile


def native(index, role=Role.ASSISTANT, content=None, offset=None):
    return Message(
        id=UUID(int=index),
        fish_id=UUID(int=999),
        session_id=UUID(int=998),
        sequence=index,
        role=role,
        content=content or f"内容{index}",
        status=MessageStatus.COMPLETED,
        created_at=datetime(2026, 9, 12, tzinfo=UTC),
        consumed_offset=offset,
    )


def expression(history, **kwargs):
    return event_messages(
        rules="base",
        profile=SoulProfile("猫", "好奇", "鱼"),
        activity=None,
        definition=kwargs.pop(
            "definition", EventDefinition(type="game.move", schema={"type": "object"})
        ),
        data=kwargs.pop("data", {}),
        context=kwargs.pop("context", {}),
        history=history,
        **kwargs,
    )


def historical_payload(message):
    return json.loads(message["content"].split("\n")[-1])["history_event"]


@pytest.mark.parametrize("fields", [None, [], ["perception"]])
def test_event_context_matches_chat_and_cannot_be_trimmed_by_sdk(fields):
    history = [
        native(1, Role.USER, "之前的问题"),
        native(2, content="已经播出。不应看见的尾巴", offset=5),
    ]
    definition = EventDefinition(
        type="test.turn",
        schema={"type": "object", "description": "推进话题"},
        context_fields=fields,
    )
    context = {
        "memory": {"static": "长期偏好", "dynamic": "当前安排"},
        "perception": {"text": "当前画面"},
    }
    image = {
        "data_url": "data:image/jpeg;base64,AA",
        "digest": "a",
        "byte_count": 1,
        "width": 1,
        "height": 1,
        "observed_at": "now",
    }
    result = expression(
        history,
        definition=definition,
        data={"text": "外部原话"},
        context=context,
        image=image,
    )
    baseline = build_chat_messages(
        "ordinary",
        history,
        static_profile="长期偏好",
        dynamic_snapshot="当前安排",
        current_input="新的用户问题",
    )
    add_perception_context(baseline, summary_context="当前画面", image=image)
    assert result[1:-1] == baseline[1:-1]
    assert result[-1]["environment_image"] is image
    assert "不应看见" not in str(result) and INTERRUPTION_MARKER in str(result)
    assert ANSWER_LANGUAGE_RULES in result[0]["content"]
    assert "推进话题" in result[0]["content"] and "外部原话" not in result[0]["content"]
    assert json.loads(result[-1]["content"]) == {
        "event": {"type": "test.turn", "data": {"text": "外部原话"}}
    }


def test_shared_fifty_message_budget_keeps_snapshots_and_perception_separate():
    history = [native(i) for i in range(1, 201)]
    context = {
        "memory": {"static": "长期偏好", "dynamic": "当前安排"},
        "perception": {"text": "新画面"},
    }
    messages = expression(history, context=context, data={"hosting_brief": "仍然暂停"})
    assert len([m for m in messages if m["role"] == "assistant"]) == 50
    assert "内容151" in messages[2]["content"] and "内容200" in messages[-3]["content"]
    assert "长期偏好" in messages[1]["content"]
    assert "当前安排" not in str(messages)
    assert "仍然暂停" in messages[-1]["content"]
    assert "新画面" in messages[-2]["content"]
    lane = build_recall_lane_messages(messages)
    assert len([m for m in lane if m["role"] == "assistant"]) == 50
    assert len(history) == 200


def test_empty_or_short_history_does_not_invent_or_repeat_current_input():
    for history in ([], [native(1, Role.USER, "问题"), native(2, content="已回答")]):
        messages = expression(history, data={"text": "新事件"})
        assert len(messages) == len(history) + 2
        assert sum("新事件" in m["content"] for m in messages) == 1


def test_screen_history_provenance_survives_expression_and_recall_without_images():
    history = "08:00 - 08:10\nas: 用户打开了设置并查看日志。"
    messages = expression([], context={"history_text": history})
    routed = build_recall_lane_messages(messages)
    for built in (messages, routed):
        assert built[0]["content"].count(HISTORY_SOURCE_RULES) == 1
        assert built[1]["content"] == "History:\n" + history
        assert not any("environment_image" in message for message in built)
        assert "直播" not in messages[0]["content"]


@pytest.mark.parametrize("memory", [{}, {"static": "长期偏好", "dynamic": "当前安排"}])
def test_event_types_and_native_chat_share_one_chronological_history(memory):
    base = datetime(2026, 9, 12, tzinfo=UTC)
    history = [
        replace(native(1, Role.USER, "本机安排"), created_at=base),
        replace(native(2, content="欢迎观众"), created_at=base + timedelta(seconds=2)),
        replace(
            native(3, Role.USER, "本机补充"), created_at=base + timedelta(seconds=3)
        ),
        replace(
            native(4, content="谢谢礼物。尚未播放", offset=5),
            created_at=base + timedelta(seconds=5),
        ),
    ]
    events = [
        {
            "type": kind,
            "data": {"text": text},
            "status": "completed",
            "received_at": received,
        }
        for kind, text, received in (
            ("live.gift", "送来礼物", "2026-09-12T00:00:04+00:00"),
            ("live.reply", "新的弹幕", "2026-09-12T00:00:06Z"),
            ("live.enter", "观众进房", "2026-09-12T08:00:01+08:00"),
        )
    ]
    messages = expression(
        list(reversed(history)),
        definition=EventDefinition(
            type="live.idle", schema={"type": "object"}, include_in_context=False
        ),
        data={"topic": "本轮话题", "speech_content": "本轮的新内容"},
        context={"events": events, "memory": memory},
    )
    start = 2 if memory else 1
    if memory:
        assert "长期偏好" in messages[1]["content"]
    timeline = messages[start:-1]
    for message, expected in zip(
        timeline,
        (
            "本机安排",
            "观众进房",
            "欢迎观众",
            "本机补充",
            "送来礼物",
            "谢谢礼物。",
            "新的弹幕",
        ),
        strict=True,
    ):
        assert expected in message["content"]
    assert [historical_payload(timeline[i]) for i in (1, 4, 6)] == [
        {**events[2], "received_at": "2026-09-12T00:00:01+00:00"},
        {**events[0], "received_at": "2026-09-12T00:00:04+00:00"},
        {**events[1], "received_at": "2026-09-12T00:00:06+00:00"},
    ]
    assert timeline[2]["role"] == timeline[5]["role"] == "assistant"
    assert INTERRUPTION_MARKER in timeline[5]["content"]
    assert "尚未播放" not in str(messages) and "related_events" not in str(messages)
    assert json.loads(messages[-1]["content"]) == {
        "event": {
            "type": "live.idle",
            "data": {"topic": "本轮话题", "speech_content": "本轮的新内容"},
        }
    }


def test_events_in_same_second_keep_arrival_order_before_the_response():
    events = [
        {
            "type": "live.reply",
            "data": {"text": text},
            "status": "completed",
            "received_at": "2026-09-12T00:00:00+00:00",
        }
        for text in ("先到", "后到")
    ]
    messages = expression([native(1, content="统一回复")], context={"events": events})
    assert [historical_payload(m) for m in messages[1:3]] == events
    assert messages[3] == {"role": "assistant", "content": "统一回复"}


@pytest.mark.parametrize("received_at", [None, "invalid"])
def test_undated_events_stay_old_without_an_invented_timestamp(received_at):
    event = {
        "type": "example.old",
        "data": {"text": "时间未知的旧事件"},
        "received_at": received_at,
    }
    messages = expression([native(1)], context={"events": [event]})
    assert historical_payload(messages[1]) == event
    assert "message_time_range" not in messages[1]["content"]
    assert "内容1" in messages[2]["content"]


def test_external_history_does_not_displace_the_native_chat_window():
    messages = expression(
        [native(i) for i in range(1, 61)],
        context={
            "events": [
                {
                    "type": "example.past",
                    "data": {"text": "旧事件"},
                    "status": "completed",
                }
            ]
        },
    )
    assert len([m for m in messages if m["role"] == "assistant"]) == 50
    assert "内容11" in messages[2]["content"]
    assert messages[-2]["content"] == "内容60"


def test_trigger_progress_is_not_recursively_replayed():
    historical = {"type": "live.idle", "data": {"recent_host_speech": ["旧模板"] * 6}}
    viewer = {"type": "live.reply", "data": {"text": "真实观众输入"}}
    messages = expression(
        [],
        definition=EventDefinition(
            type="live.idle", schema={"type": "object"}, include_in_context=False
        ),
        context={"events": [historical] * 12 + [viewer]},
    )
    assert "旧模板" not in str(messages)
    assert "真实观众输入" in str(messages)
    assert "真实观众输入" not in messages[-1]["content"]


def test_activity_is_a_duty_and_does_not_replace_personality():
    result = event_messages(
        rules="base",
        profile=SoulProfile("猫", "好奇", "鱼"),
        activity=Activity(
            name="直播", role="主播", audience="观众", objective="表达观点"
        ),
        definition=EventDefinition(type="test.turn", schema={"type": "object"}),
        history=[],
        context={},
        data={},
    )
    assert "好奇" in result[0]["content"] and '"role": "主播"' in result[0]["content"]
    assert set(json.loads(result[-1]["content"])) == {"event"}


def test_merged_events_preserve_each_identity_and_exclude_future_queued_events():
    definition = EventDefinition(
        type="example.enter",
        schema={
            "type": "object",
            "properties": {
                "viewer_id": {"type": "string", "x-fish-context": "identity"},
                "room_id": {"type": "string", "x-fish-context": "metadata"},
                "name": {"type": "string"},
            },
        },
        mergeable=True,
    )
    data = tuple(
        {"viewer_id": viewer, "room_id": "private-room", "name": name}
        for viewer, name in (("raw-a", "小青"), ("raw-b", "小白"))
    )
    messages = expression(
        [],
        definition=definition,
        data=data[0],
        merged_data=data,
        context={
            "events": [
                {
                    "type": "example.enter",
                    "status": "completed",
                    "data": {"name": "之前的观众"},
                },
                {
                    "type": "example.enter",
                    "status": "queued",
                    "data": {"name": "未来观众"},
                },
            ]
        },
    )
    current = json.loads(messages[-1]["content"])["event"]
    assert [value["name"] for value in current["items"]] == ["小青", "小白"]
    assert current["items"][0]["viewer_id"] != current["items"][1]["viewer_id"]
    assert set(current) == {"type", "items"}
    assert historical_payload(messages[-2])["data"] == {"name": "之前的观众"}
    assert all(
        value not in str(messages)
        for value in ("raw-a", "raw-b", "private-room", "未来观众")
    )
