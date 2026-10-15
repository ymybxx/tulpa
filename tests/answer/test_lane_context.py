from fish_memory.agent.prompt import build_task_lane_messages
from fish_memory.agent.routing import TaskLaneContext
from fish_memory.answer.context import ChatMessage, add_perception_policy
from fish_memory.answer.environment_context import render_perception_policy
from fish_memory.answer.lane_context import (
    build_home_lane_messages,
    build_lane_base_messages,
    build_recall_lane_messages,
    build_web_lane_messages,
)
from fish_memory.environment_frames import EnvironmentPerceptionPolicy


def source_messages() -> list[ChatMessage]:
    history: list[ChatMessage] = [
        {"role": "user", "content": f"user-{index}"}
        if index % 2 == 0
        else {"role": "assistant", "content": f"assistant-{index}"}
        for index in range(20)
    ]
    return [
        {"role": "system", "content": "soul"},
        {"role": "user", "content": "# 记忆摘要\nstatic + dynamic"},
        *history,
        {"role": "user", "content": "current-question"},
    ]


def test_recall_lane_gets_snapshots_and_recent_native_conversation() -> None:
    result = build_recall_lane_messages(source_messages(), recent_limit=12)

    assert result[0]["role"] == "system"
    assert "字段顺序固定为 m、a、r、q、e" in result[0]["content"]
    assert "应直接回答 Spotify" in result[0]["content"]
    assert result[1]["content"].startswith("# 记忆摘要")
    assert [message["content"] for message in result[2:-1]] == [
        (f"user-{index}" if index % 2 == 0 else f"assistant-{index}")
        for index in range(8, 20)
    ]
    assert "current-question" in result[-1]["content"]
    assert "user-0" not in str(result)


def test_only_recall_lane_and_recall_continuation_base_keep_environment_image() -> None:
    messages = source_messages()
    messages[-1]["environment_image"] = {
        "data_url": "data:image/jpeg;base64,private-frame",
        "digest": "a" * 64,
        "byte_count": 9,
        "width": 640,
        "height": 360,
        "observed_at": "2026-09-02T08:30:00+00:00",
    }

    recall = build_recall_lane_messages(messages)
    web = build_web_lane_messages(messages)
    home = build_home_lane_messages(messages)
    task = build_task_lane_messages(messages, TaskLaneContext.empty())
    recall_base = build_lane_base_messages(
        messages,
        include_memory=True,
        include_environment_image=True,
    )
    web_base = build_lane_base_messages(messages, include_memory=False)

    assert recall[-1]["environment_image"]["digest"] == "a" * 64
    assert recall_base[-1]["environment_image"]["digest"] == "a" * 64
    assert all("environment_image" not in message for message in web)
    assert all("environment_image" not in message for message in home)
    assert all("environment_image" not in message for message in task)
    assert all("environment_image" not in message for message in web_base)


def test_recall_lane_advertises_only_the_binary_recall_contract() -> None:
    result = build_recall_lane_messages(
        [
            {"role": "system", "content": "soul"},
            {"role": "user", "content": "我最近喜欢什么？"},
        ]
    )

    prompt = result[0]["content"]
    assert '"r":2' not in prompt
    assert "r>0" not in prompt
    assert "r=1 时只输出自然、简短的前置过渡" in prompt
    assert "r=1 时 q 必须是可独立理解的检索查询" in prompt


def test_task_router_keeps_live_responsibility_and_screen_availability() -> None:
    messages = source_messages()
    messages[0]["activity_context"] = "正在主持直播，用户是直播搭档。"
    shared = render_perception_policy(EnvironmentPerceptionPolicy(True, True, True))
    add_perception_policy(messages, shared)
    messages[-1]["environment_image"] = {
        "data_url": "data:image/jpeg;base64,private-frame",
        "digest": "a" * 64,
        "byte_count": 9,
        "width": 640,
        "height": 360,
        "observed_at": "2026-09-13T06:50:25+00:00",
    }

    result = build_task_lane_messages(messages, TaskLaneContext.empty())

    assert "正在主持直播，用户是直播搭档。" in result[0]["content"]
    assert shared in result[0]["content"]
    assert "screenSnapshot" not in str(result)
    assert "frameCount" not in str(result)
    assert "observedAt" not in str(result)
    assert "2026-09-13T06:50:25" not in str(result)
    assert "private-frame" not in str(result)
    assert "current-question" in result[-1]["content"]


def test_all_response_lanes_share_the_same_three_perception_switches() -> None:
    shared = render_perception_policy(EnvironmentPerceptionPolicy(True, False, True))
    source = source_messages()
    add_perception_policy(source, shared)
    for result in (
        build_recall_lane_messages(source),
        build_web_lane_messages(source),
        build_home_lane_messages(source),
        build_task_lane_messages(source, TaskLaneContext.empty()),
    ):
        assert result[0]["content"].count(shared) == 1
        assert "知道用户最近在电脑上做了哪些操作。" in result[0]["content"]
        assert "现在看不到用户的屏幕。" in result[0]["content"]
        assert "听得到电脑里在放的声音。" in result[0]["content"]


def test_recall_lane_shows_both_complete_wire_shapes() -> None:
    result = build_recall_lane_messages(
        [
            {"role": "system", "content": "soul"},
            {"role": "user", "content": "今天怎么样？"},
        ]
    )

    prompt = result[0]["content"]
    assert '{"m":"neutral","a":"非空的自然回答","r":0,"e":0}' in prompt
    assert (
        '{"m":"neutral","a":"非空的自然前置过渡","r":1,'
        '"q":"可独立理解的检索查询","e":0}'
    ) in prompt
    assert "值仅示意结构，不得照抄" in prompt


def test_recall_lane_scopes_end_decision_to_current_user_message() -> None:
    result = build_recall_lane_messages(
        [
            {"role": "system", "content": "soul"},
            {"role": "user", "content": "结束对话。"},
            {"role": "user", "content": "前天聊了什么？"},
        ]
    )

    for prompt in (result[0]["content"], result[-1]["content"]):
        assert "e 只能依据当前用户消息判断" in prompt
        assert "历史消息中的结束、离开或告别" in prompt


def test_recall_lane_preserves_query_semantics_in_contract_and_reminder() -> None:
    result = build_recall_lane_messages(
        [
            {"role": "system", "content": "soul"},
            {"role": "user", "content": "请找出我提到过的旅行计划和预算"},
        ]
    )

    system = result[0]["content"]
    reminder = result[-1]["content"]
    rules = (
        "先根据原始问题确定 r，再生成 q",
        "已经可独立理解时，q 默认保持原文",
        "只允许为明确指代或省略补充上下文",
        "对象",
        "属性",
        "集合范围",
        "并列条件",
        "否定",
        "数量",
        "时间限定",
        "相对时间表达必须原样保留",
        "不得换算或补充绝对日期",
        "条件",
        "比较",
        "顺序",
        "状态",
        "保留原问题中的对象、属性、集合范围、并列条件、否定、数量、时间限定、条件、比较、顺序和状态",
        "无法保证逻辑等价时返回原文",
    )

    for rule in rules:
        assert rule in system
        assert rule in reminder


def test_web_lane_uses_recent_conversation_without_snapshots() -> None:
    result = build_web_lane_messages(source_messages(), recent_limit=12)

    assert "字段顺序固定" in result[0]["content"]
    assert '{"w":0}' in result[0]["content"]
    assert "接受时字段顺序固定为 w、m、a、q" in result[0]["content"]
    assert all("# 记忆摘要" not in message["content"] for message in result)
    assert len(result[1:-1]) == 12
    assert "current-question" in result[-1]["content"]


def test_web_lane_requires_complete_closed_wire_shapes() -> None:
    result = build_web_lane_messages(
        [
            {"role": "system", "content": "soul"},
            {"role": "user", "content": "现在天气如何？"},
        ]
    )

    prompt = result[0]["content"]
    assert '拒绝时只返回完整闭合对象：{"w":0}' in prompt
    assert "一旦判定 w=0，必须立即以 } 结束" in prompt
    assert '不得返回 {"w":0,' in prompt
    assert '判定 w=0 时只输出完整对象 {"w":0}，立即闭合' in result[-1]["content"]
    assert (
        '{"w":1,"m":"neutral","a":"非空的自然联网前置过渡","q":"可独立执行的联网查询"}'
    ) in prompt


def test_web_lane_does_not_treat_unknown_personal_detail_as_external() -> None:
    result = build_web_lane_messages(
        [
            {"role": "system", "content": "soul"},
            {
                "role": "user",
                "content": "How long did Alex marinate the BBQ ribs in special sauce?",
            },
        ]
    )

    for prompt in (result[0]["content"], result[-1]["content"]):
        assert "w=1 的必要条件" in prompt
        assert "两者都不满足时必须 w=0" in prompt
        assert "没有答案，都不得改变这一结论" in prompt
        assert "需要通过互联网获取公开或外部资料时为 w=1" not in prompt


def test_home_lane_uses_recent_conversation_without_device_catalog() -> None:
    result = build_home_lane_messages(source_messages(), recent_limit=12)

    assert '{"h":0}' in result[0]["content"]
    assert "接受时字段顺序固定为 h、t、m、a" in result[0]["content"]
    assert "设备目录" not in str(result)
    assert "allowedOperations" not in str(result)
    assert all("# 记忆摘要" not in message["content"] for message in result)
    assert len(result[1:-1]) == 12
    assert "current-question" in result[-1]["content"]


def test_all_lane_prompts_apply_the_same_user_visible_acknowledgement_policy() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "所有协议说明均使用中文。"},
        {"role": "user", "content": "Could you check that for me?"},
        {"role": "assistant", "content": "Give me a moment."},
        {"role": "user", "content": "What will the weather be tomorrow?"},
    ]

    for build_messages in (
        build_recall_lane_messages,
        build_web_lane_messages,
        build_home_lane_messages,
    ):
        result = build_messages(messages)

        assert "Give me a moment." in [message["content"] for message in result[1:-1]]
        assert (
            '{"content":"What will the weather be tomorrow?"}' in result[-1]["content"]
        )
        system_prompt = result[0]["content"]
        assert "a 是用户可见文本" in system_prompt
        assert "当前用户消息 content" in system_prompt
        assert "用户明确指定的语言" in system_prompt
        assert "当前或近期用户消息中最后一条尚未撤销" in system_prompt
        assert "当前消息的主要语言" in system_prompt
        assert "协议、示例、工具结果的语言" in system_prompt
        assert "近期 assistant 消息" in system_prompt
        assert "不得复用近期已出现的相同或近似前置句" in system_prompt

        current_reminder = result[-1]["content"]
        assert "# 当前回答语言复核" in current_reminder
        assert "系统首部的语言与起句规则" in current_reminder
        assert "JSON content 才是本轮用户原话" in current_reminder


def test_lane_prompt_payloads_do_not_offer_fixed_acknowledgements_to_copy() -> None:
    fixed_acknowledgements = (
        "当然，可以。",
        "让我想想。",
        "我回想一下。",
        "我去看看最新情况。",
        "我看看。",
        "好，我来处理。",
    )

    for build_messages in (
        build_recall_lane_messages,
        build_web_lane_messages,
        build_home_lane_messages,
    ):
        system_prompt = build_messages(source_messages())[0]["content"]

        for acknowledgement in fixed_acknowledgements:
            assert acknowledgement not in system_prompt
