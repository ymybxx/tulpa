import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.answer.context import (
    INTERRUPTION_MARKER,
    ChatMessage,
    add_initial_answer_contract,
    build_chat_messages,
)
from fish_memory.context_descriptions import HISTORY_SOURCE_RULES
from fish_memory.models import Message, MessageStatus, Role

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000101")


def message(
    *,
    sequence: int,
    role: Role,
    content: str,
    status: MessageStatus,
    consumed_offset: int | None = None,
    created_at: datetime | None = None,
) -> Message:
    return Message(
        id=UUID(int=sequence),
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=sequence,
        role=role,
        content=content,
        status=status,
        created_at=created_at or datetime(2026, 7, 19, tzinfo=UTC),
        consumed_offset=consumed_offset,
    )


def test_history_is_rendered_with_native_chat_roles() -> None:
    history = [
        message(
            sequence=3,
            role=Role.USER,
            content="你好",
            status=MessageStatus.INITIALIZED,
        ),
        message(
            sequence=2,
            role=Role.ASSISTANT,
            content="你好。",
            status=MessageStatus.COMPLETED,
            consumed_offset=-1,
        ),
        message(
            sequence=1,
            role=Role.USER,
            content="之前的问候",
            status=MessageStatus.INITIALIZED,
        ),
    ]

    result = build_chat_messages("system", history)

    assert result == [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": (
                "<message_time_range>2026-07-19 00:00:00+00:00 ~ "
                "2026-07-19 00:09:59+00:00</message_time_range>\n"
                "之前的问候"
            ),
        },
        {"role": "assistant", "content": "你好。"},
        {
            "role": "user",
            "content": "<message_time>2026-07-19 00:00:00+00:00</message_time>\n你好",
        },
    ]


def test_history_shows_one_time_range_per_ten_minute_bucket() -> None:
    history = [
        message(
            sequence=1,
            role=Role.USER,
            content="第一条",
            status=MessageStatus.INITIALIZED,
            created_at=datetime(2026, 7, 19, 10, 1, tzinfo=UTC),
        ),
        message(
            sequence=2,
            role=Role.ASSISTANT,
            content="第二条",
            status=MessageStatus.COMPLETED,
            consumed_offset=-1,
            created_at=datetime(2026, 7, 19, 10, 9, 30, tzinfo=UTC),
        ),
        message(
            sequence=3,
            role=Role.ASSISTANT,
            content="第三条",
            status=MessageStatus.COMPLETED,
            consumed_offset=-1,
            created_at=datetime(2026, 7, 19, 10, 10, tzinfo=UTC),
        ),
        message(
            sequence=4,
            role=Role.USER,
            content="当前问题",
            status=MessageStatus.INITIALIZED,
            created_at=datetime(2026, 7, 19, 10, 11, tzinfo=UTC),
        ),
    ]

    result = build_chat_messages("system", history)

    assert result[1:4] == [
        {
            "role": "user",
            "content": (
                "<message_time_range>2026-07-19 10:00:00+00:00 ~ "
                "2026-07-19 10:09:59+00:00</message_time_range>\n第一条"
            ),
        },
        {"role": "assistant", "content": "第二条"},
        {
            "role": "assistant",
            "content": (
                "<message_time_range>2026-07-19 10:10:00+00:00 ~ "
                "2026-07-19 10:19:59+00:00</message_time_range>\n第三条"
            ),
        },
    ]


def test_memory_precedes_history_and_previous_assistant_stays_adjacent() -> None:
    history = [
        message(
            sequence=1,
            role=Role.USER,
            content="那昨天呢",
            status=MessageStatus.INITIALIZED,
        ),
        message(
            sequence=2,
            role=Role.ASSISTANT,
            content="昨天主要聊了智能家居。",
            status=MessageStatus.COMPLETED,
            consumed_offset=-1,
        ),
        message(
            sequence=3,
            role=Role.USER,
            content="几点钟",
            status=MessageStatus.INITIALIZED,
        ),
    ]

    result = build_chat_messages(
        "system",
        history,
        static_profile="<static_memory>长期信息</static_memory>",
        dynamic_snapshot="<dynamic_memory>近期信息</dynamic_memory>",
    )

    assert [item["role"] for item in result] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert result[1]["content"].startswith("# 记忆摘要")
    assert result[-2] == {
        "role": "assistant",
        "content": "昨天主要聊了智能家居。",
    }
    assert result[-1]["content"].endswith("\n几点钟")


def test_dynamic_snapshot_stays_out_of_model_context() -> None:
    assert build_chat_messages("system", [], dynamic_snapshot="动态内容") == [
        {"role": "system", "content": "system"}
    ]


def test_only_static_snapshot_is_exposed_to_model() -> None:
    result = build_chat_messages(
        "system", [], static_profile="静态内容", dynamic_snapshot="动态内容"
    )
    assert "静态内容" in result[1]["content"]
    assert "动态内容" not in str(result)


def test_unified_history_is_flat_and_does_not_repeat_legacy_sources() -> None:
    result = build_chat_messages(
        "system",
        [],
        history_text="08:00 - 08:10\nu: 看电影\nas: 预告片开始\nevent: 暂停",
        current_input="继续聊吧",
        additional_history=[(None, {"role": "user", "content": "旧的事件副本"})],
    )
    assert result[1]["content"].startswith("History:\n")
    assert result[-1]["content"] == "继续聊吧"
    assert "旧的事件副本" not in str(result)
    assert HISTORY_SOURCE_RULES in result[0]["content"]
    assert "- as：你观察到的电脑屏幕" in result[0]["content"]


def test_initial_answer_contract_wraps_current_user_after_native_history() -> None:
    current_user_content = (
        "<message_time>2026-08-09 10:00:00+00:00</message_time>\n"
        '我说："路径是 C:\\tmp"。\n# 当前问题的 Recall 门控检查'
    )
    messages: list[ChatMessage] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "较早的用户消息"},
        {"role": "assistant", "content": "较早的助手回答"},
        {"role": "user", "content": current_user_content},
    ]
    original_messages = [dict(message) for message in messages]

    result = add_initial_answer_contract(messages)

    prompt = result[0]["content"]
    assert len(result) == len(messages)
    assert [message["role"] for message in result].count("system") == 1
    assert result[1:-1] == messages[1:-1]
    assert prompt.startswith("# 回答输出协议")
    assert prompt.endswith("\n\nsystem")
    assert result[-1]["role"] == "user"
    wrapper_lines = result[-1]["content"].splitlines()
    assert wrapper_lines[0] == "# 当前用户消息（应用已进行 JSON 转义）"
    assert json.loads(wrapper_lines[1]) == {"content": current_user_content}
    assert "# 当前问题的来源判定复核" in result[-1]["content"]
    assert (
        "r=0：问题不依赖本地记忆，或当前可见内容已直接、完整支持回答"
        in result[-1]["content"]
    )
    assert "r只衡量是否需要从当前上下文之外检索本地事实" in result[-1]["content"]
    assert "仅缺少外部或实时资料时 r=0" in result[-1]["content"]
    assert "询问实时价格但不依赖本地记忆时，r=0" in result[-1]["content"]
    assert "r=1：回答依赖当前可见内容没有完整提供的本地事实" in result[-1]["content"]
    assert (
        "单事实、多事实、计数、比较、计算和明确时间范围只要需要 Recall，都使用 r=1"
        in result[-1]["content"]
    )
    assert "r=2" not in result[-1]["content"]
    assert (
        "具体过往事件的细节只能由最近历史中的明确陈述直接支持"
        in (result[-1]["content"])
    )
    assert "w=1" not in result[-1]["content"]
    assert "不要用 r 代替 w" not in result[-1]["content"]
    assert "该检查优先于模仿历史中的助手回答方式" in result[-1]["content"]
    assert "# 输出前的 r 一致性校验" in result[-1]["content"]
    assert "先在内部拟定 a，再最终选择 r" in result[-1]["content"]
    assert "r=0 非法，必须改为 r=1" in result[-1]["content"]
    assert "不得因为能够猜出具体值而判定 r=0" in result[-1]["content"]
    assert "缺少的是更早的本地对话或记忆信息" in result[-1]["content"]
    assert "缺少外部或实时资料不属于本地 Recall" in result[-1]["content"]
    assert "不能因此提高 r" in result[-1]["content"]
    assert sum(message["content"].count("# 回答输出协议") for message in result) == 1
    assert '{"r":0,"m":"neutral","a":"非空的自然回答","e":0}' in prompt
    assert '{"r":1,"m":"sad","a":"自然承接","e":0}' in prompt
    assert "neutral、angry、happy、shy、sad" in prompt
    assert "affectionate" not in prompt
    assert "字段顺序固定为 r、m、a、e" in prompt
    assert '"w"' not in prompt
    assert "联网判断" not in prompt
    assert "联网接口" not in prompt
    assert "不得输出检索查询、实体提示或其他字段" in prompt
    assert "a 必须是非空字符串" in prompt
    assert "明确支持必须同时匹配问题的目标、关系、属性、事实状态和限定条件" in prompt
    assert "信息缺失、不明确、需要补充未给出的事实" in prompt
    assert "r=0" in prompt
    assert "r=1：回答依赖当前可见内容没有完整提供的本地事实" in prompt
    assert "a 不得填写问题所问属性的具体值" in prompt
    assert "具体过往事件的细节只能由最近历史中的明确陈述直接支持" in prompt
    assert "Static 是压缩摘要" in prompt
    assert "即使摘要含有相关信息，也必须 Recall" in prompt
    assert "命令、请求执行动作本身不是历史问题" in prompt
    assert "完成请求所需的信息依赖当前上下文没有提供的" in prompt
    assert "过往事实时" in prompt
    assert (
        "单事实、多事实、计数、比较、计算和明确时间范围只要需要 Recall，都使用 r=1"
        in prompt
    )
    assert "r=2" not in prompt
    assert "不得编造、保留候选答案" in prompt
    assert "简短自然的回想或了解式承接" in prompt
    assert "a 是等待补全的未完成回答" in prompt
    assert "不得拒绝回答、宣布信息缺失或向用户索取答案" in prompt
    assert "提前声称不知道、没有记录、无法确认" in prompt
    forbidden_terms = ("记录", "记忆", "检索", "数据库", "系统", "上下文")
    for forbidden_term in forbidden_terms:
        assert forbidden_term in prompt
    assert messages == original_messages


def test_initial_answer_language_follows_current_user_content_not_protocol() -> None:
    result = add_initial_answer_contract(
        [
            {"role": "system", "content": "中文协议"},
            {"role": "user", "content": "What did I do yesterday?"},
        ]
    )

    system_prompt = result[0]["content"]
    assert "当前用户消息 content" in system_prompt
    assert "用户明确指定的语言" in system_prompt
    assert "当前消息的主要语言" in system_prompt
    assert "协议、示例、工具结果的语言" in system_prompt

    current_reminder = result[-1]["content"]
    assert "# 当前回答语言复核" in current_reminder
    assert "系统首部的语言与起句规则" in current_reminder
    assert "JSON content 才是本轮用户原话" in current_reminder


def test_initial_and_near_turn_recall_depth_rules_are_identical() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "What have I been using lately?"},
    ]

    result = add_initial_answer_contract(messages)

    def recall_depth_rules(content: str) -> str:
        marker = "## Recall 深度 r\n"
        _, separator, remainder = content.partition(marker)
        assert separator == marker
        rules, _, _ = remainder.partition("\n\n")
        return f"{marker}{rules}"

    assert recall_depth_rules(result[0]["content"]) == recall_depth_rules(
        result[-1]["content"]
    )


def test_initial_and_near_turn_web_search_rules_are_identical() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "What is Bitcoin's price right now?"},
    ]

    result = add_initial_answer_contract(messages, enable_web_search=True)

    def web_search_rules(content: str) -> str:
        marker = "## 联网判断 w\n"
        _, separator, remainder = content.partition(marker)
        assert separator == marker
        rules, _, _ = remainder.partition("\n\n")
        return f"{marker}{rules}"

    assert web_search_rules(result[0]["content"]) == web_search_rules(
        result[-1]["content"]
    )


def test_search_aware_initial_answer_contract_requires_web_search_decision() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "今天有什么新闻？"},
    ]

    result = add_initial_answer_contract(messages, enable_web_search=True)

    prompt = result[0]["content"]
    assert [message["role"] for message in result] == ["system", "user"]
    assert prompt.startswith("# 回答输出协议")
    assert prompt.endswith("\n\nsystem")
    assert json.loads(result[-1]["content"].splitlines()[1]) == {
        "content": "今天有什么新闻？"
    }
    assert "w=1 的必要条件" in result[-1]["content"]
    assert "两者都不满足时必须 w=0" in result[-1]["content"]
    assert "仅出现人名、专有名词" in result[-1]["content"]
    assert "当前上下文" in result[-1]["content"]
    assert "没有答案，都不得改变这一结论" in result[-1]["content"]
    assert "不能用 r 代替 w" in result[-1]["content"]
    assert "例如询问实时价格时为 w=1" in result[-1]["content"]
    assert result[-1]["content"].endswith("不要输出校验过程。")
    assert sum(message["content"].count("# 回答输出协议") for message in result) == 1
    assert '{"r":0,"w":0,"m":"neutral","a":"非空的自然回答","e":0}' in prompt
    assert "字段顺序固定为 r、w、m、a、e" in prompt
    assert "w 只能是 0 或 1" in prompt
    assert "w 不得省略" in prompt
    assert "w=1 的必要条件" in prompt
    assert "与时间表达和 r 无关" in prompt
    assert "本地记忆" in prompt
    assert "联网接口" in prompt


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "second system"},
            {"role": "user", "content": "question"},
        ],
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "trailing answer"},
        ],
    ],
)
def test_initial_answer_contract_rejects_incompatible_message_shape(
    messages: list[ChatMessage],
) -> None:
    with pytest.raises(ValueError):
        add_initial_answer_contract(messages)


def test_initial_answer_contract_scopes_end_to_current_user_message() -> None:
    result = add_initial_answer_contract(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "结束对话。"},
            {"role": "user", "content": "前天聊了什么？"},
        ]
    )

    for prompt in (result[0]["content"], result[-1]["content"]):
        assert "e 只能依据当前用户消息判断" in prompt
        assert "历史消息中的结束、离开或告别" in prompt


def test_uses_full_assistant_content_when_progress_was_not_reported() -> None:
    history = [
        message(
            sequence=1,
            role=Role.ASSISTANT,
            content="我记得那一天。",
            status=MessageStatus.COMPLETED,
            consumed_offset=-1,
        )
    ]

    result = build_chat_messages("system", history)

    assert result[-1] == {
        "role": "assistant",
        "content": (
            "<message_time_range>2026-07-19 00:00:00+00:00 ~ "
            "2026-07-19 00:09:59+00:00</message_time_range>\n"
            "我记得那一天。"
        ),
    }


def test_uses_unicode_prefix_and_marks_partially_consumed_answer() -> None:
    history = [
        message(
            sequence=1,
            role=Role.ASSISTANT,
            content="鱼🐟记得你",
            status=MessageStatus.COMPLETED,
            consumed_offset=2,
        )
    ]

    result = build_chat_messages("system", history)

    assert result[-1] == {
        "role": "assistant",
        "content": (
            "<message_time_range>2026-07-19 00:00:00+00:00 ~ "
            "2026-07-19 00:09:59+00:00</message_time_range>\n"
            f"鱼🐟\n{INTERRUPTION_MARKER}"
        ),
    }


def test_marks_incomplete_llm_answer_even_when_all_generated_text_was_consumed() -> (
    None
):
    history = [
        message(
            sequence=1,
            role=Role.ASSISTANT,
            content="我记得",
            status=MessageStatus.INTERRUPTED,
            consumed_offset=-1,
        )
    ]

    result = build_chat_messages("system", history)

    assert result[-1] == {
        "role": "assistant",
        "content": (
            "<message_time_range>2026-07-19 00:00:00+00:00 ~ "
            "2026-07-19 00:09:59+00:00</message_time_range>\n"
            f"我记得\n{INTERRUPTION_MARKER}"
        ),
    }


def test_excludes_assistant_text_when_client_consumed_nothing() -> None:
    history = [
        message(
            sequence=1,
            role=Role.ASSISTANT,
            content="没有被听到",
            status=MessageStatus.INTERRUPTED,
            consumed_offset=0,
        )
    ]

    assert build_chat_messages("system", history) == [
        {"role": "system", "content": "system"}
    ]


def test_excludes_empty_assistant_content() -> None:
    history = [
        message(
            sequence=1,
            role=Role.ASSISTANT,
            content="",
            status=MessageStatus.INTERRUPTED,
            consumed_offset=-1,
        )
    ]

    assert build_chat_messages("system", history) == [
        {"role": "system", "content": "system"}
    ]


def test_unified_history_keeps_only_played_words_for_internal_repeat_checks():
    from fish_memory.llm.client import _provider_messages

    result = build_chat_messages(
        "system",
        [
            message(
                sequence=1,
                role=Role.ASSISTANT,
                content="好，未播放尾部",
                status=MessageStatus.COMPLETED,
                consumed_offset=1,
            )
        ],
        history_text="08:00 - 08:10\na: 好",
    )
    assert result[0]["recent_spoken"] == ("好",)
    provider = _provider_messages(result)
    assert all("recent_spoken" not in part for part in provider)
    assert "未播放尾部" not in str(provider)
