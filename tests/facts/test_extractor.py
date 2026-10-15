from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.answer.context import ChatMessage
from fish_memory.facts.extractor import FactExtractor
from fish_memory.facts.models import FactExtractionContext
from fish_memory.llm.client import JsonCompletion
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.models import JsonValue


class FakeJsonCompletionClient:
    def __init__(self, content: str) -> None:
        self.completion = JsonCompletion(
            content=content,
            input_tokens=31,
            output_tokens=17,
        )
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        self.calls.append(messages)
        return self.completion

    def json_request_payload(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonValue:
        return {
            "model": "qwen-plus",
            "messages": [dict(item) for item in messages],
            "parameters": {
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
        }


def make_message(*, role: Role = Role.USER) -> Message:
    return Message(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        fish_id=UUID("00000000-0000-0000-0000-000000000002"),
        session_id=UUID("00000000-0000-0000-0000-000000000003"),
        sequence=3,
        role=role,
        content="stored source",
        status=MessageStatus.INITIALIZED,
        created_at=datetime.fromisoformat("2026-07-23T15:30:00+08:00"),
    )


def fact_response(facts: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "what": fact.get("what", fact.get("text", "")),
            "where": fact.get("where", []),
            "who": fact.get("who", []),
            **fact,
        }
        for fact in facts
    ]
    return json.dumps({"facts": normalized}, ensure_ascii=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [Role.USER, Role.ASSISTANT],
)
async def test_prompt_is_chinese_and_uses_neutral_source_role(
    role: Role,
) -> None:
    llm = FakeJsonCompletionClient('{"facts":[]}')
    extractor = FactExtractor(llm)

    await extractor.extract(make_message(role=role), "完整 source 内容")

    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert "原子化" in messages[0]["content"]
    assert "脱离原消息后仍能独立理解" in messages[0]["content"]
    assert "计划、否定、不确定性、转述" in messages[0]["content"]
    assert "只返回规定的 JSON 对象" in messages[0]["content"]
    assert "不限于传统命名实体" in messages[0]["content"]
    assert "不得因为 sourceRole 添加消息发送者本身" in messages[0]["content"]
    assert "主人" not in messages[0]["content"]
    assert "fish" not in messages[0]["content"]
    assert "fishLabel" not in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {
        "referenceTime": "2026-07-23T07:30:00+00:00",
        "sourceRole": role.value,
        "content": "完整 source 内容",
    }


@pytest.mark.asyncio
async def test_prompt_includes_complete_context_for_reference_resolution() -> None:
    llm = FakeJsonCompletionClient('{"facts":[]}')
    message = make_message(role=Role.USER)
    context = FactExtractionContext(
        role=Role.ASSISTANT,
        content="Target 是我刚才提到的可使用优惠券的地点。" + "x" * 600,
    )

    await FactExtractor(llm).extract(
        message,
        "我实际上兑换了一张 5 美元优惠券",
        context,
    )

    prompt = llm.calls[0][0]["content"]
    payload = json.loads(llm.calls[0][1]["content"])
    assert "当前 content 决定是否生成 Fact，以及 Fact 所描述的事件" in prompt
    assert "补齐同一事件的地点、人物、对象、时间或关系" in prompt
    assert "不得仅根据 conversationContext 创建 Fact" in prompt
    assert "不得把其中未被当前 content 承接的信息另行生成 Fact" in prompt
    assert "I actually redeemed a $5 coupon on coffee creamer" in prompt
    assert "在 Target 兑换" in prompt
    assert payload["conversationContext"]["sourceRole"] == "assistant"
    assert len(payload["conversationContext"]["content"]) == len(context.content)
    assert payload["conversationContext"]["content"].startswith(
        "Target 是我刚才提到的可使用优惠券的地点。"
    )
    assert list(payload) == [
        "referenceTime",
        "sourceRole",
        "conversationContext",
        "content",
    ]


@pytest.mark.asyncio
async def test_prompt_sends_reference_time_at_second_precision() -> None:
    llm = FakeJsonCompletionClient('{"facts":[]}')
    message = Message(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        fish_id=UUID("00000000-0000-0000-0000-000000000002"),
        session_id=UUID("00000000-0000-0000-0000-000000000003"),
        sequence=3,
        role=Role.USER,
        content="stored source",
        status=MessageStatus.INITIALIZED,
        created_at=datetime.fromisoformat("2026-07-30T08:04:12.550564+00:00"),
    )

    await FactExtractor(llm).extract(message, "我上周打游戏了")

    payload = json.loads(llm.calls[0][1]["content"])
    assert payload["referenceTime"] == "2026-07-30T08:04:12+00:00"


@pytest.mark.asyncio
async def test_context_completed_fact_remains_about_the_current_message() -> None:
    expected_text = "该用户于2023年5月21日在Target兑换了一张咖啡奶油的5美元优惠券"
    llm = FakeJsonCompletionClient(
        fact_response(
            [
                {
                    "what": "兑换一张咖啡奶油的5美元优惠券",
                    "where": ["Target"],
                    "who": ["该用户"],
                    "occurredStart": "2023-05-21T08:00:00+08:00",
                    "occurredEnd": "2023-05-21T08:00:00+08:00",
                    "text": expected_text,
                    "entities": ["Target", "咖啡奶油", "5美元优惠券"],
                }
            ]
        )
    )
    context = FactExtractionContext(
        role=Role.ASSISTANT,
        content=(
            "The Cartwheel app is a fantastic tool for saving money "
            "on household items and more at Target."
        ),
    )

    result = await FactExtractor(llm).extract(
        make_message(),
        "I actually redeemed a $5 coupon on coffee creamer last Sunday.",
        context,
    )

    assert [fact.text for fact in result.facts] == [expected_text]
    assert result.facts[0].entities == ("Target", "咖啡奶油", "5美元优惠券")
    assert result.facts[0].what == "兑换一张咖啡奶油的5美元优惠券"
    assert result.facts[0].where == ("Target",)
    assert result.facts[0].who == ("该用户",)


@pytest.mark.asyncio
async def test_prompt_treats_facts_as_lossless_foundation_material() -> None:
    llm = FakeJsonCompletionClient('{"facts":[]}')

    await FactExtractor(llm).extract(make_message(), "source")

    prompt = llm.calls[0][0]["content"]
    assert "高召回的语义原材料" in prompt
    assert "不要根据长期价值、重要性或是否值得记忆进行筛选" in prompt
    assert "请求、命令、问题、承诺、拒绝和回应" in prompt
    assert "问候" in prompt
    assert "简短表达只要传达了态度、情绪、确认或拒绝" in prompt
    assert "不得合并彼此独立的事实" in prompt
    assert (
        "数量、频率、持续时间、计划、否定、不确定性、转述、条件、程度和原因" in prompt
    )
    assert "sourceRole=user 时，“我”指该用户，“你”指该助手" in prompt
    assert "sourceRole=assistant 时，“我”指该助手，“你”指该用户" in prompt
    assert "不得再改写为“对方”" in prompt
    assert "无法确定指代对象时，不得猜测或编造身份" in prompt
    assert "将其改写为带来源的陈述事实" in prompt
    assert "所有输出值必须与原消息使用相同语言" in prompt
    assert "目标不超过 1000 个字符" in prompt
    assert "拆分为多条原子化事实" in prompt
    assert "不得在 text 中照抄 ISO 8601 时间戳" in prompt
    assert "text 中的时间必须使用自然语言" in prompt
    schema = prompt.rsplit("只返回规定的 JSON 对象", maxsplit=1)[1]
    assert schema.index('"entities"') < schema.index('"text"')


@pytest.mark.asyncio
async def test_prompt_separates_context_and_completeness_rules() -> None:
    llm = FakeJsonCompletionClient('{"facts":[]}')

    await FactExtractor(llm).extract(make_message(), "source")

    prompt = llm.calls[0][0]["content"]
    assert "# 任务" in prompt
    assert "# 输出边界" in prompt
    assert "# 角色与指代" in prompt
    assert "# 相邻上下文" in prompt
    assert "# Fact 结构" in prompt
    assert "# 时间" in prompt
    assert "# 实体" in prompt


@pytest.mark.asyncio
async def test_prompt_uses_broad_retrieval_entities_without_sender_entity() -> None:
    llm = FakeJsonCompletionClient('{"facts":[]}')

    await FactExtractor(llm).extract(make_message(), "我每天打篮球")

    prompt = llm.calls[0][0]["content"]
    assert "entities 是用于关联相关事实的检索锚点" in prompt
    assert "不限于传统命名实体" in prompt
    assert "人物、动物、组织、地点、重要物品、活动、兴趣、习惯主题" in prompt
    assert "作品、项目、技术、事件以及抽象概念和主题" in prompt
    assert "名字、别名、昵称、称呼和外号" in prompt
    assert "被赋予的名称必须写入 entities" in prompt
    assert "不得因为 sourceRole 添加消息发送者本身" in prompt
    assert '"entities":["篮球"]' in prompt


@pytest.mark.asyncio
async def test_preserves_fact_order_and_parses_aware_datetimes() -> None:
    llm = FakeJsonCompletionClient(
        fact_response(
            [
                {
                    "text": "该用户于2025年搬到上海",
                    "occurredStart": "2025-01-01T00:00:00+08:00",
                    "occurredEnd": "2025-12-31T23:59:59+08:00",
                    "entities": ["上海"],
                },
                {
                    "text": "该用户计划于2026年7月24日去公园",
                    "occurredStart": "2026-07-24T00:00:00+08:00",
                    "occurredEnd": "2026-07-24T23:59:59+08:00",
                    "entities": ["公园"],
                },
            ]
        )
    )

    result = await FactExtractor(llm).extract(
        make_message(),
        "去年搬到上海，明天去公园",
    )

    assert [fact.text for fact in result.facts] == [
        "该用户于2025年搬到上海",
        "该用户计划于2026年7月24日去公园",
    ]
    assert result.facts[0].occurred_start == datetime.fromisoformat(
        "2025-01-01T00:00:00+08:00"
    )
    assert result.facts[0].occurred_end == datetime.fromisoformat(
        "2025-12-31T23:59:59+08:00"
    )


@pytest.mark.asyncio
async def test_normalizes_entities_without_canonicalizing_names() -> None:
    llm = FakeJsonCompletionClient(
        fact_response(
            [
                {
                    "text": "小明搬到上海",
                    "occurredStart": None,
                    "occurredEnd": None,
                    "entities": [
                        " 小明 ",
                        "",
                        "小明",
                        " 上海 ",
                        "上海",
                        "上海市",
                    ],
                }
            ]
        )
    )

    result = await FactExtractor(llm).extract(make_message(), "小明搬到上海")

    assert result.facts[0].entities == ("小明", "上海", "上海市")


@pytest.mark.asyncio
async def test_returns_request_and_raw_response_payloads() -> None:
    raw_content = '{"facts":[]}'
    llm = FakeJsonCompletionClient(raw_content)

    result = await FactExtractor(llm).extract(make_message(), "你好")

    assert result.request_payload == {
        "model": "qwen-plus",
        "messages": llm.calls[0],
        "parameters": {
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "stream": False,
        },
    }
    assert result.response_payload == {
        "content": raw_content,
        "input_tokens": 31,
        "output_tokens": 17,
    }


@pytest.mark.asyncio
async def test_accepts_fact_item_accidentally_encoded_as_single_json_key() -> None:
    encoded_fact = json.dumps(
        {
            "what": "描述 Roasted Chickpeas 由鹰嘴豆制成",
            "where": [],
            "who": ["该助手"],
            "text": "该助手描述 Roasted Chickpeas 由鹰嘴豆制成",
            "occurredStart": None,
            "occurredEnd": None,
            "entities": ["Roasted Chickpeas", "鹰嘴豆"],
        },
        ensure_ascii=False,
    )
    llm = FakeJsonCompletionClient(
        json.dumps(
            {
                "facts": [
                    {
                        encoded_fact: "ignored accidental value",
                    }
                ]
            },
            ensure_ascii=False,
        )
    )

    result = await FactExtractor(llm).extract(make_message(), "source")

    assert result.facts[0].text == "该助手描述 Roasted Chickpeas 由鹰嘴豆制成"
    assert result.facts[0].entities == ("Roasted Chickpeas", "鹰嘴豆")


@pytest.mark.asyncio
async def test_rejects_unknown_top_level_response_field() -> None:
    extractor = FactExtractor(
        FakeJsonCompletionClient('{"facts":[],"unexpected":true}')
    )

    with pytest.raises(ValidationError):
        await extractor.extract(make_message(), "source")


@pytest.mark.asyncio
async def test_rejects_unknown_field_inside_fact() -> None:
    response = fact_response(
        [
            {
                "text": "小明搬到上海",
                "occurredStart": None,
                "occurredEnd": None,
                "entities": ["小明", "上海"],
                "unexpected": True,
            }
        ]
    )
    extractor = FactExtractor(FakeJsonCompletionClient(response))

    with pytest.raises(ValidationError):
        await extractor.extract(make_message(), "source")


@pytest.mark.asyncio
async def test_rejects_fact_longer_than_2000_characters() -> None:
    response = fact_response(
        [
            {
                "text": "长" * 2001,
                "occurredStart": None,
                "occurredEnd": None,
                "entities": [],
            }
        ]
    )

    with pytest.raises(ValidationError):
        await FactExtractor(FakeJsonCompletionClient(response)).extract(
            make_message(),
            "source",
        )


INVALID_RESPONSES = [
    fact_response(
        [
            {
                "text": "   ",
                "occurredStart": None,
                "occurredEnd": None,
                "entities": [],
            }
        ]
    ),
    fact_response(
        [
            {
                "text": "naive time",
                "occurredStart": "2026-07-23T10:00:00",
                "occurredEnd": None,
                "entities": [],
            }
        ]
    ),
    fact_response(
        [
            {
                "text": "reversed range",
                "occurredStart": "2026-07-24T10:00:00+08:00",
                "occurredEnd": "2026-07-23T10:00:00+08:00",
                "entities": [],
            }
        ]
    ),
    fact_response(
        [
            {
                "text": "end only",
                "occurredStart": None,
                "occurredEnd": "2026-07-23T10:00:00+08:00",
                "entities": [],
            }
        ]
    ),
    fact_response(
        [
            {
                "text": "wrong entity type",
                "occurredStart": None,
                "occurredEnd": None,
                "entities": [1],
            }
        ]
    ),
    "not JSON",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response", INVALID_RESPONSES)
async def test_rejects_invalid_fact_output(response: str) -> None:
    extractor = FactExtractor(FakeJsonCompletionClient(response))

    with pytest.raises(ValidationError):
        await extractor.extract(make_message(), "source")
