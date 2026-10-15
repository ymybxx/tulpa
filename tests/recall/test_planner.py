from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletion
from fish_memory.performance.models import JsonValue
from fish_memory.recall.models import (
    RecallQueryPlan,
    RecallSourceRole,
    RecallTimeRange,
)
from fish_memory.recall.planner import RecallPlanner


class FakeJsonClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        self.calls.append(messages)
        return JsonCompletion(
            content=self.content,
            input_tokens=40,
            output_tokens=20,
        )

    def json_request_payload(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonValue:
        return {
            "model": "qwen-plus",
            "messages": [dict(message) for message in messages],
        }


@pytest.mark.asyncio
async def test_preserves_input_query_and_plans_time_range() -> None:
    client = FakeJsonClient(
        """
        {
          "sourceRole": "user",
          "entityHints": ["  小明  ", "小明"],
          "time": {
            "start": "2026-07-25T00:00:00+08:00",
            "end": "2026-07-26T23:59:59+08:00"
          }
        }
        """
    )
    planner = RecallPlanner(client)
    current_time = datetime.fromisoformat("2026-07-26T12:00:00.123456+08:00")

    conversation_context: tuple[ChatMessage, ...] = (
        {"role": "user", "content": "我昨天遇到一件烦心事"},
        {"role": "assistant", "content": "发生什么了？"},
    )
    result = await planner.plan(
        "我上次不开心是为什么？",
        current_time,
        conversation_context=conversation_context,
    )

    assert result.plan.semantic_query == "我上次不开心是为什么？"
    assert result.plan.entity_hints == ("小明",)
    assert result.plan.source_role is RecallSourceRole.USER
    assert result.plan.time.start == datetime.fromisoformat("2026-07-25T00:00:00+08:00")
    assert result.plan.time.end == datetime.fromisoformat("2026-07-26T23:59:59+08:00")
    assert result.plan.time.has_bounds is True
    assert "field" not in result.plan.model_dump(mode="json", by_alias=True)["time"]
    assert "order" not in result.plan.model_dump(mode="json", by_alias=True)
    assert result.request_payload == {
        "model": "qwen-plus",
        "messages": client.calls[0],
    }
    assert result.response_payload == {
        "content": client.content,
        "input_tokens": 40,
        "output_tokens": 20,
    }
    system_prompt = client.calls[0][0]["content"]
    assert "event_type" in system_prompt
    assert "禁止" in system_prompt
    assert "evidenceMode" not in system_prompt
    assert "sourceRole" in system_prompt
    assert "user、assistant 或 either" in system_prompt
    assert "entityHints" in system_prompt
    assert "明确出现" in system_prompt
    assert (
        "只有问题明确给出日期、时间点、相对日期或时间范围时才能设置 start/end。"
        in system_prompt
    )
    assert (
        "持续多久、花费多少小时、年龄、数量和金额只是 Fact 内容属性，不是时间 Scope。"
        in system_prompt
    )
    assert (
        "最近一次、最早、第一次、最后一次不生成排序字段；"
        "没有明确范围时 start/end 都为 null。" in system_prompt
    )
    assert "occurred" not in system_prompt
    assert "mentioned" not in system_prompt
    assert "latest" not in system_prompt
    assert "earliest" not in system_prompt
    assert "recallDepth" not in system_prompt
    assert client.calls[0][1]["content"] == (
        '{"currentTime":"2026-07-26T04:00:00+00:00",'
        '"conversationContext":[{"role":"user","content":"我昨天遇到一件烦心事"},'
        '{"role":"assistant","content":"发生什么了？"}],'
        '"content":"我上次不开心是为什么？"}'
    )
    assert "recallDepth" not in client.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_prompt_allows_unbounded_time_range_without_depth() -> None:
    client = FakeJsonClient(
        '{"sourceRole":"either","entityHints":["耳机"],'
        '"time":{"start":null,"end":null}}'
    )
    conversation_context: tuple[ChatMessage, ...] = (
        {"role": "user", "content": "我买了一副新耳机"},
    )

    await RecallPlanner(client).plan(
        "它后来怎么了？",
        datetime.fromisoformat("2026-07-26T12:00:00+08:00"),
        conversation_context=conversation_context,
    )

    prompt = client.calls[0][0]["content"]
    assert "只能用于提取结构化检索条件" in prompt
    assert "content 是当前用户的原始问题" in prompt
    assert "content 不一定能脱离 conversationContext 独立理解" in prompt
    assert "向量检索 Query 由调用方另行提供" in prompt
    assert "不得输出或改写向量检索 Query" in prompt
    assert "后续会原样用于向量检索" not in prompt
    assert "recallDepth" not in prompt


@pytest.mark.asyncio
async def test_prompt_supplies_system_month_calendar_for_relative_time() -> None:
    client = FakeJsonClient(
        '{"sourceRole":"user","entityHints":[],"time":{'
        '"start":"2023-05-22T00:00:00Z",'
        '"end":"2023-05-28T23:59:59Z"}}'
    )

    await RecallPlanner(client).plan(
        "How many hours did I exercise last week?",
        datetime.fromisoformat("2023-05-30T21:24:00+00:00"),
    )

    prompt = client.calls[0][0]["content"]
    assert "Calendar reference" in prompt
    assert "Dates use the user's system timezone" in prompt
    assert "A calendar week runs Monday through Sunday" in prompt
    assert "Current time: 2023-05-30 21:24:00+00:00 (Tuesday)" in prompt
    assert '"last week" means the complete previous calendar week' in prompt
    assert "not the previous seven rolling days" in prompt
    assert "May 2023" in prompt
    assert "Mo Tu We Th Fr Sa Su" in prompt
    assert "22 23 24 25 26 27 28" in prompt
    assert "29 30 31" in prompt
    assert (
        "2023-05-29 2023-05-30 2023-05-31 2023-06-01 "
        "2023-06-02 2023-06-03 2023-06-04" in prompt
    )


@pytest.mark.asyncio
async def test_calendar_month_uses_system_zone() -> None:
    client = FakeJsonClient(
        '{"sourceRole":"either","entityHints":[],"time":{"start":null,"end":null}}'
    )

    await RecallPlanner(client).plan(
        "query",
        datetime.fromisoformat("2024-03-01T01:00:00+08:00"),
    )

    prompt = client.calls[0][0]["content"]
    assert "Current time: 2024-02-29 17:00:00+00:00 (Thursday)" in prompt
    assert "February 2024" in prompt
    assert "March 2024" not in prompt
    assert "26 27 28 29" in prompt
    assert (
        "2024-02-26 2024-02-27 2024-02-28 2024-02-29 "
        "2024-03-01 2024-03-02 2024-03-03" in prompt
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_role", "content"),
    [
        (
            "user",
            '{"sourceRole":"user","entityHints":[],"time":{"start":null,"end":null}}',
        ),
        (
            "assistant",
            '{"sourceRole":"assistant",'
            '"entityHints":[],"time":{"start":null,"end":null}}',
        ),
        (
            "either",
            '{"sourceRole":"either","entityHints":[],"time":{"start":null,"end":null}}',
        ),
    ],
)
async def test_accepts_valid_source_roles(
    source_role: str,
    content: str,
) -> None:
    result = await RecallPlanner(FakeJsonClient(content)).plan(
        "query",
        datetime.fromisoformat("2026-07-26T12:00:00+08:00"),
    )

    assert result.plan.source_role.value == source_role


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        ('{"sourceRole":"system","entityHints":[],"time":{"start":null,"end":null}}'),
        ('{"entityHints":[],"time":{"start":null,"end":null}}'),
        (
            '{"sourceRole":"either","semanticQuery":"rewritten query","entityHints":[],'
            '"time":{"start":null,"end":null}}'
        ),
        (
            '{"sourceRole":"either","entityHints":[],"time":{"field":"occurred",'
            '"start":null,"end":null}}'
        ),
        (
            '{"sourceRole":"either","entityHints":[],"time":{'
            '"start":"2026-07-27T00:00:00+08:00",'
            '"end":"2026-07-26T00:00:00+08:00"}}'
        ),
        (
            '{"sourceRole":"either","entityHints":[],"time":{'
            '"start":"2026-07-26T00:00:00","end":null}}'
        ),
        (
            '{"sourceRole":"either","entityHints":[],"time":{"start":null,'
            '"end":null},"order":"latest"}'
        ),
        (
            '{"sourceRole":"either","entityHints":[],"time":{"start":null,'
            '"end":null},'
            '"unexpected":true}'
        ),
        (
            '{"sourceRole":"either","entityHints":[],"time":{"start":null,'
            '"end":null},"evidenceMode":"full"}'
        ),
    ],
)
async def test_rejects_invalid_planner_options(content: str) -> None:
    planner = RecallPlanner(FakeJsonClient(content))

    with pytest.raises(ValidationError):
        await planner.plan(
            "query",
            datetime.fromisoformat("2026-07-26T12:00:00+08:00"),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        ('{"source_role":"either","entityHints":[],"time":{"start":null,"end":null}}'),
        ('{"sourceRole":"either","entity_hints":[],"time":{"start":null,"end":null}}'),
    ],
)
async def test_rejects_snake_case_external_planner_keys(content: str) -> None:
    with pytest.raises(ValidationError):
        await RecallPlanner(FakeJsonClient(content)).plan(
            "query",
            datetime.fromisoformat("2026-07-26T12:00:00+08:00"),
        )


def test_recall_time_range_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="time end must not precede start"):
        RecallTimeRange(
            start=datetime.fromisoformat("2026-07-27T00:00:00+08:00"),
            end=datetime.fromisoformat("2026-07-26T00:00:00+08:00"),
        )


@pytest.mark.parametrize(
    "bound",
    [
        0,
        1.5,
        True,
        False,
        "0",
        "1.5",
        "2026-07-26T00:00:00",
        "2026-07-26",
    ],
)
def test_recall_time_range_rejects_non_timezone_wire_bounds(bound: object) -> None:
    payload = json.dumps({"start": bound, "end": None})

    with pytest.raises(ValidationError):
        RecallTimeRange.model_validate_json(payload)


@pytest.mark.parametrize(
    "bound",
    [
        "2026-07-26T00:00:00Z",
        "2026-07-26T08:00:00+08:00",
    ],
)
def test_recall_time_range_accepts_timezone_qualified_wire_bounds(bound: str) -> None:
    result = RecallTimeRange.model_validate_json(
        json.dumps({"start": bound, "end": None})
    )

    assert result.start == datetime.fromisoformat(bound)


def test_recall_time_range_accepts_internal_aware_datetime() -> None:
    bound = datetime(2026, 7, 26, tzinfo=UTC)

    result = RecallTimeRange(start=bound, end=None)

    assert result.start is bound


def test_recall_query_plan_preserves_semantic_query_whitespace() -> None:
    query = "  exact submitted query  "

    plan = RecallQueryPlan(
        semanticQuery=query,
        sourceRole="either",
        entityHints=[],
        time={"start": None, "end": None},
    )

    assert plan.semantic_query == query


def test_recall_query_plan_rejects_legacy_order() -> None:
    with pytest.raises(ValidationError):
        RecallQueryPlan.model_validate(
            {
                "semanticQuery": "query",
                "sourceRole": "either",
                "entityHints": [],
                "time": {"start": None, "end": None},
                "order": "relevance",
            }
        )
