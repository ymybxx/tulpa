import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletion
from fish_memory.models import Role
from fish_memory.performance.models import JsonValue
from fish_memory.snapshots.atomicity_prompt import ATOMIC_RELATION_ITEM_CONTRACT
from fish_memory.snapshots.confirmer import (
    SnapshotUpdateConfirmer,
    StaticProfileUpdateConfirmer,
)
from fish_memory.snapshots.coverage_prompt import SNAPSHOT_SUPPORT_COVERAGE_CONTRACT
from fish_memory.snapshots.models import (
    ConfirmedCreate,
    ConfirmedIgnore,
    ConfirmedUpdate,
    SnapshotFact,
    SnapshotItemState,
)
from fish_memory.snapshots.static_confirmation_prompt import (
    STATIC_PROFILE_CONFIRMATION_SYSTEM_PROMPT,
)
from fish_memory.snapshots.static_prompt import STATIC_PROFILE_ELIGIBILITY_CONTRACT

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


class FakeJsonCompletionClient:
    def __init__(self, content: str) -> None:
        self._completion = JsonCompletion(
            content=content,
            input_tokens=17,
            output_tokens=9,
        )
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        self.calls.append(messages)
        return self._completion

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return {"messages": [dict(message) for message in messages]}


def make_fact(index: int, text: str | None = None) -> SnapshotFact:
    return SnapshotFact(
        id=UUID(int=100 + index),
        sequence=index + 1,
        fish_id=UUID(int=10),
        source_message_id=UUID(int=20 + index),
        source_role=Role.USER,
        text=text or f"fact-{index}",
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW,
        entities=("用户",),
    )


def make_item(*facts: SnapshotFact) -> SnapshotItemState:
    return SnapshotItemState(
        id=44,
        summary="用户正在退回旧靴。",
        support_fact_ids=tuple(fact.id for fact in facts),
        lineage_fact_ids=tuple(fact.id for fact in facts),
        last_supported_at=NOW,
        item_version=2,
    )


@pytest.mark.asyncio
async def test_confirmer_returns_an_update_and_a_split_create_with_payloads() -> None:
    """Catches a confirmer that loses locally indexed operations or telemetry."""
    old = make_fact(0, "用户正在退回旧靴。")
    returned = make_fact(1, "用户已经退回旧靴。")
    pickup = make_fact(2, "用户正在计划取货。")
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "update",
                        "basis": "旧事项已经完成。",
                        "itemIndex": 17,
                        "newSupportIndexes": [0],
                        "removeSupportIndexes": [0],
                        "summary": "用户已经完成旧靴退回事项。",
                    },
                    {
                        "action": "create",
                        "basis": "取货是独立的新事项。",
                        "newSupportIndexes": [1],
                        "summary": "用户正在计划取货。",
                    },
                ]
            }
        )
    )

    result = await SnapshotUpdateConfirmer(client).confirm(
        item_index=17,
        item=make_item(old),
        support_facts=(old,),
        new_facts=(returned, pickup),
        recent_turns=(
            {"role": "user", "content": "我已经把靴子退了"},
            {"role": "assistant", "content": "好的"},
        ),
    )

    assert result.operations == (
        ConfirmedUpdate(
            basis="旧事项已经完成。",
            item_index=17,
            new_support_indexes=(0,),
            remove_support_indexes=(0,),
            summary="用户已经完成旧靴退回事项。",
        ),
        ConfirmedCreate(
            basis="取货是独立的新事项。",
            new_support_indexes=(1,),
            summary="用户正在计划取货。",
        ),
    )
    assert result.request_payload == {
        "messages": [dict(message) for message in client.calls[0]]
    }
    assert result.response_payload == {
        "content": client._completion.content,
        "input_tokens": 17,
        "output_tokens": 9,
    }

    prompt_payload = json.loads(client.calls[0][1]["content"])
    assert prompt_payload["targetItem"] == {
        "itemIndex": 17,
        "summary": "用户正在退回旧靴。",
    }
    assert prompt_payload["supportFacts"][0]["index"] == 0
    assert prompt_payload["newFacts"][0]["index"] == 0
    assert prompt_payload["contextOnlyRecentTurns"] == [
        {"role": "user", "content": "我已经把靴子退了"},
        {"role": "assistant", "content": "好的"},
    ]
    assert "index" not in prompt_payload["contextOnlyRecentTurns"][0]
    assert "不得把 contextOnlyRecentTurns 当作支持证据" in client.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_static_confirmer_can_veto_an_ineligible_update_marker() -> None:
    old = make_fact(0, "用户长期使用中文。")
    temporary = make_fact(1, "用户这次要求使用英文。")
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "ignore",
                        "basis": "单次语言要求不改变长期偏好。",
                        "newSupportIndexes": [0],
                    }
                ]
            }
        )
    )

    result = await StaticProfileUpdateConfirmer(client).confirm(
        item_index=17,
        item=make_item(old),
        support_facts=(old,),
        new_facts=(temporary,),
        recent_turns=(cast(ChatMessage, {"role": "user", "content": "这次用英文"}),),
    )

    assert result.operations == (
        ConfirmedIgnore(
            basis="单次语言要求不改变长期偏好。",
            new_support_indexes=(0,),
        ),
    )
    prompt = client.calls[0][0]["content"]
    assert prompt == STATIC_PROFILE_CONFIRMATION_SYSTEM_PROMPT
    assert STATIC_PROFILE_ELIGIBILITY_CONTRACT in prompt
    assert ATOMIC_RELATION_ITEM_CONTRACT not in prompt
    assert SNAPSHOT_SUPPORT_COVERAGE_CONTRACT not in prompt
    assert "不能因为第一阶段已经标记它就假定它合格" in prompt
    assert "短期独立事项必须 Ignore" in prompt
    assert "保留仍未被解决的计划、待办、承诺和状态" not in prompt


@pytest.mark.asyncio
async def test_static_confirmer_partitions_long_term_updates_creates_and_ignores() -> (
    None
):
    old = make_fact(0, "用户长期使用中文。")
    language = make_fact(1, "用户以后都使用英文。")
    profession = make_fact(2, "用户是后端工程师。")
    weather = make_fact(3, "用户询问今天的天气。")
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "update",
                        "basis": "长期语言偏好被明确替代。",
                        "itemIndex": 17,
                        "newSupportIndexes": [0],
                        "removeSupportIndexes": [0],
                        "summary": "用户长期使用英文。",
                    },
                    {
                        "action": "create",
                        "basis": "职业是稳定身份属性。",
                        "newSupportIndexes": [1],
                        "summary": "用户是后端工程师。",
                    },
                    {
                        "action": "ignore",
                        "basis": "天气问题仅在短期有用。",
                        "newSupportIndexes": [2],
                    },
                ]
            }
        )
    )

    result = await StaticProfileUpdateConfirmer(client).confirm(
        item_index=17,
        item=make_item(old),
        support_facts=(old,),
        new_facts=(language, profession, weather),
        recent_turns=(),
    )

    assert result.operations == (
        ConfirmedUpdate(
            basis="长期语言偏好被明确替代。",
            item_index=17,
            new_support_indexes=(0,),
            remove_support_indexes=(0,),
            summary="用户长期使用英文。",
        ),
        ConfirmedCreate(
            basis="职业是稳定身份属性。",
            new_support_indexes=(1,),
            summary="用户是后端工程师。",
        ),
        ConfirmedIgnore(
            basis="天气问题仅在短期有用。",
            new_support_indexes=(2,),
        ),
    )


@pytest.mark.asyncio
async def test_static_confirmer_requires_every_assigned_fact_to_be_classified() -> None:
    old = make_fact(0, "用户长期使用中文。")
    first = make_fact(1, "用户这次要求使用英文。")
    second = make_fact(2, "用户询问今天的天气。")
    client = FakeJsonCompletionClient(
        '{"operations":[{"action":"ignore","basis":"临时要求",'
        '"newSupportIndexes":[0]}]}'
    )

    with pytest.raises(ValueError, match="must classify every new fact"):
        await StaticProfileUpdateConfirmer(client).confirm(
            item_index=17,
            item=make_item(old),
            support_facts=(old,),
            new_facts=(first, second),
            recent_turns=(),
        )


@pytest.mark.asyncio
async def test_static_confirmer_rejects_update_without_new_long_term_evidence() -> None:
    old = make_fact(0, "用户长期使用中文。")
    temporary = make_fact(1, "用户这次要求使用英文。")
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "update",
                        "basis": "错误地用无证据更新改写摘要。",
                        "itemIndex": 17,
                        "newSupportIndexes": [],
                        "removeSupportIndexes": [],
                        "summary": "用户这次使用英文。",
                    },
                    {
                        "action": "ignore",
                        "basis": "单次语言要求不属于 Static。",
                        "newSupportIndexes": [0],
                    },
                ]
            }
        )
    )

    with pytest.raises(
        ValueError,
        match="static update must select at least one new fact",
    ):
        await StaticProfileUpdateConfirmer(client).confirm(
            item_index=17,
            item=make_item(old),
            support_facts=(old,),
            new_facts=(temporary,),
            recent_turns=(),
        )


@pytest.mark.asyncio
async def test_dynamic_confirmer_rejects_static_ignore_action() -> None:
    old = make_fact(0)
    new = make_fact(1)
    client = FakeJsonCompletionClient(
        '{"operations":[{"action":"ignore","basis":"Static only",'
        '"newSupportIndexes":[0]}]}'
    )

    with pytest.raises(ValidationError):
        await SnapshotUpdateConfirmer(client).confirm(
            item_index=17,
            item=make_item(old),
            support_facts=(old,),
            new_facts=(new,),
            recent_turns=(),
        )


@pytest.mark.asyncio
async def test_confirmation_prompt_protects_unresolved_current_support() -> None:
    """Catches Summary narrowing being presented as a reason to remove support."""
    old = make_fact(0, "仍待处理的旧事项。")
    new = make_fact(1, "同一主题出现新的当前进展。")
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "update",
                        "basis": "新事实补充同一主题，旧事项仍未解决。",
                        "itemIndex": 0,
                        "newSupportIndexes": [0],
                        "removeSupportIndexes": [],
                        "summary": "旧事项仍待处理，并增加新的当前进展。",
                    }
                ]
            }
        )
    )

    await SnapshotUpdateConfirmer(client).confirm(
        item_index=0,
        item=make_item(old),
        support_facts=(old,),
        new_facts=(new,),
        recent_turns=(),
    )

    prompt = client.calls[0][0]["content"]
    assert ATOMIC_RELATION_ITEM_CONTRACT in prompt
    assert "预更新把不同事项标到同一 Item 时" in prompt
    assert "返回 Create-only" in prompt
    assert "一个 Update 与一个或多个 Create" in prompt
    assert "只有 assigned newFacts 明确证明旧支持已经完成、取消、纠正或被取代" in prompt
    assert "不能因为摘要要缩短、焦点转移、时间较早或相关性较低而移除" in prompt
    assert "必须保留仍未被解决的计划、待办、承诺和状态" in prompt
    assert "每个 action=create 必须包含非空 newSupportIndexes" in prompt
    assert (
        "每个 action=update 必须包含 itemIndex、newSupportIndexes"
        " 和 removeSupportIndexes" in prompt
    )
    assert "两个索引数组即使为空也不能省略" in prompt
    assert '"action":"create"' in prompt
    assert (
        "选择一个 sourceFactIndex 或 newSupportIndex，表示最终 Summary 承诺覆盖该 Fact "
        "仍会影响后续回答的核心命题"
    ) in prompt
    assert "仅提及相同主题或实体不算覆盖" in prompt
    assert "对象、关系、否定、条件、状态、未完成动作、数量、时间和顺序" in prompt
    assert "拆成多个 Create" in prompt
    assert "只有当两组 Facts 描述彼此独立的事项" in prompt
    assert "移除任一组后，另一事项仍然完整" in prompt
    assert "不要拆分同一具体对象的身份细节或必要属性" in prompt
    assert (
        "不要仅因可以分别询问，就拆分同一事件、计划或问题的"
        "进展、结果、纠正、取消、替代或其他阶段"
    ) in prompt
    assert "输出前逐条复核" in prompt
    assert "所有未被合法移除的旧 current support" in prompt
    assert "本次 newSupportIndexes 选择的新命题" in prompt
    assert "Create-only" in prompt


@pytest.mark.asyncio
async def test_confirmer_rejects_duplicate_new_fact_ids_before_the_llm_call() -> None:
    """Catches local indexes disguising reuse of one assigned Fact as two facts."""
    old = make_fact(0)
    new = make_fact(1)
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "update",
                        "basis": "更新旧事项",
                        "itemIndex": 17,
                        "newSupportIndexes": [0],
                        "removeSupportIndexes": [],
                        "summary": "更新后的事项",
                    },
                    {
                        "action": "create",
                        "basis": "另一个事项",
                        "newSupportIndexes": [1],
                        "summary": "另一个事项",
                    },
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate new fact id"):
        await SnapshotUpdateConfirmer(client).confirm(
            item_index=17,
            item=make_item(old),
            support_facts=(old,),
            new_facts=(new, new),
            recent_turns=(),
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_confirmer_downgrades_a_wrong_marker_to_create_only() -> None:
    """Catches a wrong marker that cannot preserve the old item's identity."""
    old = make_fact(0)
    new = make_fact(1, "用户正在安排独立的取货事项。")
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "create",
                        "basis": "取货与旧靴退回事项无关。",
                        "newSupportIndexes": [0],
                        "summary": "用户正在安排取货事项。",
                    }
                ]
            }
        )
    )

    result = await SnapshotUpdateConfirmer(client).confirm(
        item_index=17,
        item=make_item(old),
        support_facts=(old,),
        new_facts=(new,),
        recent_turns=(),
    )

    assert result.operations == (
        ConfirmedCreate(
            basis="取货与旧靴退回事项无关。",
            new_support_indexes=(0,),
            summary="用户正在安排取货事项。",
        ),
    )


@pytest.mark.asyncio
async def test_confirmer_removes_runtime_metadata_from_context_only_turns() -> None:
    """Catches an internal turn index leaking into the model's evidence context."""
    old = make_fact(0)
    new = make_fact(1)
    client = FakeJsonCompletionClient(
        '{"operations":[{"action":"create","basis":"独立事项",'
        '"newSupportIndexes":[0],"summary":"新事项"}]}'
    )
    indexed_turn = cast(
        ChatMessage,
        {"role": "user", "content": "上下文", "index": 99},
    )

    await SnapshotUpdateConfirmer(client).confirm(
        item_index=17,
        item=make_item(old),
        support_facts=(old,),
        new_facts=(new,),
        recent_turns=(indexed_turn,),
    )

    payload = json.loads(client.calls[0][1]["content"])
    assert payload["contextOnlyRecentTurns"] == [{"role": "user", "content": "上下文"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        {
            "action": "update",
            "basis": "a" * 101,
            "itemIndex": 17,
            "newSupportIndexes": [0],
            "removeSupportIndexes": [],
            "summary": "摘要",
        },
        {
            "action": "update",
            "basis": "错误目标",
            "itemIndex": 18,
            "newSupportIndexes": [0],
            "removeSupportIndexes": [],
            "summary": "摘要",
        },
        {
            "action": "create",
            "basis": "错误字段",
            "itemIndex": 17,
            "newSupportIndexes": [0],
            "summary": "摘要",
        },
        {
            "action": "create",
            "basis": "错误字段",
            "newSupportIndexes": [0],
            "removeSupportIndexes": [],
            "summary": "摘要",
        },
        {
            "action": "update",
            "basis": "new support 索引错误",
            "itemIndex": 17,
            "newSupportIndexes": [1],
            "removeSupportIndexes": [],
            "summary": "摘要",
        },
        {
            "action": "update",
            "basis": "remove support 索引错误",
            "itemIndex": 17,
            "newSupportIndexes": [0],
            "removeSupportIndexes": [1],
            "summary": "摘要",
        },
        {
            "action": "create",
            "basis": "旧字段已禁止",
            "newSupportIndexes": [0],
            "summary": "摘要",
            "category": "state",
        },
    ],
)
async def test_confirmer_rejects_invalid_operation_contracts(
    operation: dict[str, object],
) -> None:
    """Catches permissive parsing of invalid indexes and removed legacy fields."""
    old = make_fact(0)
    new = make_fact(1)
    client = FakeJsonCompletionClient(json.dumps({"operations": [operation]}))

    with pytest.raises((ValidationError, ValueError)):
        await SnapshotUpdateConfirmer(client).confirm(
            item_index=17,
            item=make_item(old),
            support_facts=(old,),
            new_facts=(new,),
            recent_turns=(),
        )


@pytest.mark.asyncio
async def test_confirmer_rejects_reusing_a_new_support_fact() -> None:
    """Catches two operations silently claiming the same assigned new fact."""
    old = make_fact(0)
    new = make_fact(1)
    client = FakeJsonCompletionClient(
        json.dumps(
            {
                "operations": [
                    {
                        "action": "update",
                        "basis": "更新旧事项",
                        "itemIndex": 17,
                        "newSupportIndexes": [0],
                        "removeSupportIndexes": [],
                        "summary": "更新后的事项",
                    },
                    {
                        "action": "create",
                        "basis": "另一个事项",
                        "newSupportIndexes": [0],
                        "summary": "另一个事项",
                    },
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="more than once"):
        await SnapshotUpdateConfirmer(client).confirm(
            item_index=17,
            item=make_item(old),
            support_facts=(old,),
            new_facts=(new,),
            recent_turns=(),
        )


@pytest.mark.asyncio
async def test_confirmer_rejects_an_empty_confirmation() -> None:
    """Catches a marker confirmation that would otherwise disappear silently."""
    old = make_fact(0)
    new = make_fact(1)
    client = FakeJsonCompletionClient('{"operations":[]}')

    with pytest.raises(ValidationError):
        await SnapshotUpdateConfirmer(client).confirm(
            item_index=17,
            item=make_item(old),
            support_facts=(old,),
            new_facts=(new,),
            recent_turns=(),
        )
