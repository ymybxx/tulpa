import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import JsonCompletion
from fish_memory.models import Role
from fish_memory.performance.models import JsonValue
from fish_memory.snapshots.atomicity_prompt import ATOMIC_RELATION_ITEM_CONTRACT
from fish_memory.snapshots.models import (
    PreCreateOperation,
    PreUpdateMarker,
    SnapshotFact,
    SnapshotItemState,
    SnapshotPreUpdate,
)
from fish_memory.snapshots.updater import DynamicSnapshotUpdater

NOW = datetime(2026, 7, 24, 12, 0, 0, 123456, tzinfo=UTC)
FISH_ID = UUID(int=10)
MESSAGE_ID = UUID(int=20)


class FakeJsonCompletionClient:
    def __init__(self, content: str) -> None:
        self.completion = JsonCompletion(
            content=content,
            input_tokens=50,
            output_tokens=20,
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
        }


def make_fact(index: int) -> SnapshotFact:
    return SnapshotFact(
        id=UUID(int=100 + index),
        sequence=index + 1,
        fish_id=FISH_ID,
        source_message_id=MESSAGE_ID,
        source_role=Role.USER,
        text=f"fact-{index}",
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW + timedelta(minutes=index),
        entities=(f"entity-{index}",),
    )


def make_item(summary: str = "old") -> SnapshotItemState:
    return SnapshotItemState(
        id=41,
        summary=summary,
        support_fact_ids=(UUID(int=1),),
        lineage_fact_ids=(UUID(int=1),),
        last_supported_at=NOW - timedelta(days=2),
        item_version=3,
    )


@pytest.mark.asyncio
async def test_preupdate_returns_diagnostics_for_normalized_items() -> None:
    """Catches a pre-update call that drops its request or completion payload."""
    raw = json.dumps(
        {
            "creates": [{"summary": "用户正在规划书墙", "sourceFactIndexes": [1]}],
            "updates": [{"itemIndex": 0, "sourceFactIndexes": [0]}],
        }
    )
    client = FakeJsonCompletionClient(raw)

    result = await DynamicSnapshotUpdater(client).preupdate(
        (make_item("用户正在布置书架"),),
        (make_fact(0), make_fact(1)),
        NOW,
    )

    assert result.pre_update == SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="用户正在规划书墙",
                source_fact_indexes=(1,),
            ),
        ),
        updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
    )
    assert result.request_payload == {
        "model": "qwen-plus",
        "messages": [dict(item) for item in client.calls[0]],
    }
    assert result.response_payload == {
        "content": raw,
        "input_tokens": 50,
        "output_tokens": 20,
    }
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["existingItems"] == [
        {
            "index": 0,
            "summary": "用户正在布置书架",
            "lastSupportedAt": (NOW - timedelta(days=2)).isoformat(timespec="seconds"),
        }
    ]


@pytest.mark.asyncio
async def test_preupdate_drops_only_explicit_empty_evidence_operations() -> None:
    """Catches an evidence-free no-op blocking valid sibling operations."""
    raw = json.dumps(
        {
            "creates": [
                {"summary": "无证据主题", "sourceFactIndexes": []},
                {"summary": "有效新主题", "sourceFactIndexes": [1]},
            ],
            "updates": [
                {"itemIndex": 0, "sourceFactIndexes": []},
                {"itemIndex": 0, "sourceFactIndexes": [0]},
            ],
        }
    )

    result = await DynamicSnapshotUpdater(FakeJsonCompletionClient(raw)).preupdate(
        (make_item(),),
        (make_fact(0), make_fact(1)),
        NOW,
    )

    assert result.pre_update == SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="有效新主题",
                source_fact_indexes=(1,),
            ),
        ),
        updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
    )
    assert result.response_payload["dropped_operations"] == [
        {
            "kind": "create",
            "index": 0,
            "reason": "empty_source_fact_indexes",
        },
        {
            "kind": "update",
            "index": 0,
            "reason": "empty_source_fact_indexes",
        },
    ]


@pytest.mark.asyncio
async def test_preupdate_deduplicates_indexes_only_within_each_operation() -> None:
    """Catches idempotent provider duplicates blocking otherwise valid siblings."""
    raw = json.dumps(
        {
            "creates": [
                {"summary": "有效新主题", "sourceFactIndexes": [1, 1]},
            ],
            "updates": [
                {"itemIndex": 0, "sourceFactIndexes": [0, 0]},
            ],
        }
    )

    result = await DynamicSnapshotUpdater(FakeJsonCompletionClient(raw)).preupdate(
        (make_item(),),
        (make_fact(0), make_fact(1)),
        NOW,
    )

    assert result.pre_update == SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="有效新主题",
                source_fact_indexes=(1,),
            ),
        ),
        updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
    )
    assert result.response_payload["normalizations"] == [
        {
            "kind": "create",
            "index": 0,
            "field": "sourceFactIndexes",
            "original": [1, 1],
            "normalized": [1],
            "reason": "duplicate_source_fact_indexes",
        },
        {
            "kind": "update",
            "index": 0,
            "field": "sourceFactIndexes",
            "original": [0, 0],
            "normalized": [0],
            "reason": "duplicate_source_fact_indexes",
        },
    ]


@pytest.mark.asyncio
async def test_preupdate_drops_every_operation_that_claims_an_ambiguous_fact() -> None:
    """Catches one conflicted routing decision poisoning the whole Snapshot batch."""
    raw = json.dumps(
        {
            "creates": [
                {"summary": "受歧义事实影响的主题", "sourceFactIndexes": [0, 1]},
                {"summary": "不受影响的主题", "sourceFactIndexes": [3]},
            ],
            "updates": [
                {"itemIndex": 0, "sourceFactIndexes": [1, 2]},
            ],
        }
    )

    result = await DynamicSnapshotUpdater(FakeJsonCompletionClient(raw)).preupdate(
        (make_item(),),
        tuple(make_fact(index) for index in range(4)),
        NOW,
    )

    assert result.pre_update == SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="不受影响的主题",
                source_fact_indexes=(3,),
            ),
        ),
        updates=(),
    )
    assert result.response_payload["dropped_operations"] == [
        {
            "kind": "create",
            "index": 0,
            "reason": "ambiguous_source_fact_indexes",
            "sourceFactIndexes": [1],
        },
        {
            "kind": "update",
            "index": 0,
            "reason": "ambiguous_source_fact_indexes",
            "sourceFactIndexes": [1],
        },
    ]


@pytest.mark.asyncio
async def test_preupdate_prompt_keeps_dynamic_routing_boundaries() -> None:
    client = FakeJsonCompletionClient('{"creates":[],"updates":[]}')

    await DynamicSnapshotUpdater(client).preupdate(
        (make_item(),),
        (make_fact(0),),
        NOW,
    )

    prompt = client.calls[0][0]["content"]
    assert "近期上下文" in prompt
    assert "未完成的计划、问题、承诺、等待结果" in prompt
    assert "一次性普通提问" in prompt
    assert ATOMIC_RELATION_ITEM_CONTRACT in prompt
    assert "共享契约只适用于已经满足当前 Snapshot 保留资格的 Facts" in prompt
    assert "资格判断先于原子路由" in prompt
    assert "即使它描述独立事项也必须忽略" in prompt
    assert "如果该事项符合当前 Snapshot 的保留资格，必须 Create；否则忽略" in prompt
    assert prompt.index("资格判断先于原子路由") < prompt.index("只共享实体")
    assert "同一个具体记忆事项" in prompt
    assert "相同实体、地点、领域或大主题" in prompt
    assert "可分别回答测试" in prompt
    assert "Amazonia 亚硝酸盐过高" in prompt
    assert "用户想为 Amazonia 购买水草" in prompt
    assert "只有当两组 Facts 描述彼此独立的事项" in prompt
    assert "移除任一组后，另一事项仍然完整" in prompt
    assert "不要拆分同一具体对象的身份细节或必要属性" in prompt
    assert (
        "不要仅因可以分别询问，就拆分同一事件、计划或问题的"
        "进展、结果、纠正、取消、替代或其他阶段"
    ) in prompt
    assert "继续某个 existingItems 的同一主题" not in prompt
    assert "独立新主题" not in prompt
    assert "不得在此阶段重写摘要" in prompt
    assert "不得返回 Fact UUID" in prompt
    assert "分类" in prompt
    assert "废弃操作" in prompt
    assert "sourceRole" in prompt
    assert "每个 create 或 update 的 sourceFactIndexes 必须非空" in prompt
    assert "不得返回 sourceFactIndexes=[]" in prompt
    assert (
        "选择一个 sourceFactIndex 或 newSupportIndex，表示最终 Summary 承诺覆盖该 Fact "
        "仍会影响后续回答的核心命题"
    ) in prompt
    assert "仅提及相同主题或实体不算覆盖" in prompt
    assert "对象、关系、否定、条件、状态、未完成动作、数量、时间和顺序" in prompt
    assert "拆成多个 Create" in prompt
    assert "输出前逐条复核" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"creates":[{"summary":"新主题","category":"state",'
        '"sourceFactIndexes":[0]}],"updates":[]}',
        '{"creates":[{"summary":"新主题","status":"active",'
        '"sourceFactIndexes":[0]}],"updates":[]}',
        '{"creates":[],"updates":[{"itemIndex":0,"summary":"错误摘要",'
        '"sourceFactIndexes":[0]}]}',
        '{"creates":[],"updates":[],"deprecates":[]}',
    ],
)
async def test_preupdate_rejects_legacy_fields(content: str) -> None:
    with pytest.raises(ValidationError):
        await DynamicSnapshotUpdater(FakeJsonCompletionClient(content)).preupdate(
            (make_item(),),
            (make_fact(0),),
            NOW,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"creates":[{"summary":"新主题"}],"updates":[]}',
        '{"creates":[],"updates":[{"itemIndex":0,"sourceFactIndexes":null}]}',
        '{"creates":[{"summary":"新主题","sourceFactIndexes":"0"}],"updates":[]}',
        '{"creates":[{"sourceFactIndexes":[]}],"updates":[]}',
        '{"creates":[{"summary":"新主题","sourceFactIndexes":[],'
        '"extra":true}],"updates":[]}',
        '{"creates":[{"summary":"新主题","sourceFactIndexes":[1,1]}],"updates":[]}',
        '{"creates":[{"summary":" ","sourceFactIndexes":[0]}],"updates":[]}',
        '{"creates":[{"summary":"新主题","sourceFactIndexes":[1]}],"updates":[]}',
        '{"creates":[],"updates":[{"itemIndex":1,"sourceFactIndexes":[0]}]}',
    ],
)
async def test_preupdate_rejects_invalid_or_reused_indexes(content: str) -> None:
    with pytest.raises((ValidationError, ValueError)):
        await DynamicSnapshotUpdater(FakeJsonCompletionClient(content)).preupdate(
            (make_item(),),
            (make_fact(0),),
            NOW,
        )


@pytest.mark.asyncio
async def test_missing_operation_arrays_are_empty() -> None:
    result = await DynamicSnapshotUpdater(
        FakeJsonCompletionClient(
            '{"creates":[{"summary":"新主题","sourceFactIndexes":[0]}]}'
        )
    ).preupdate((), (make_fact(0),), NOW)

    assert len(result.pre_update.creates) == 1
    assert result.pre_update.updates == ()
