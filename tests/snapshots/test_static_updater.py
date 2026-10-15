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
from fish_memory.snapshots.coverage_prompt import SNAPSHOT_SUPPORT_COVERAGE_CONTRACT
from fish_memory.snapshots.models import (
    PreCreateOperation,
    PreUpdateMarker,
    SnapshotFact,
    SnapshotItemState,
    SnapshotPreUpdate,
)
from fish_memory.snapshots.static_prompt import STATIC_PROFILE_ELIGIBILITY_CONTRACT
from fish_memory.snapshots.static_updater import StaticProfileUpdater

NOW = datetime(2026, 7, 25, 10, 0, 0, 123456, tzinfo=UTC)
FISH_ID = UUID(int=1)
MESSAGE_ID = UUID(int=2)


class FakeJsonCompletionClient:
    def __init__(self, content: str) -> None:
        self.completion = JsonCompletion(
            content=content,
            input_tokens=40,
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
        return {"model": "qwen-plus", "messages": [dict(item) for item in messages]}


def fact(index: int, *, source_role: Role = Role.USER) -> SnapshotFact:
    return SnapshotFact(
        id=UUID(int=100 + index),
        sequence=index + 1,
        fish_id=FISH_ID,
        source_message_id=MESSAGE_ID,
        source_role=source_role,
        text="该用户每天打篮球",
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW + timedelta(minutes=index),
        entities=("篮球",),
    )


def item(summary: str = "用户每周运动") -> SnapshotItemState:
    return SnapshotItemState(
        id=51,
        summary=summary,
        support_fact_ids=(UUID(int=10),),
        lineage_fact_ids=(UUID(int=10),),
        last_supported_at=NOW,
        item_version=2,
    )


@pytest.mark.asyncio
async def test_static_preupdate_returns_diagnostics_for_normalized_items() -> None:
    raw = json.dumps(
        {
            "creates": [{"summary": "用户喜欢篮球", "sourceFactIndexes": [1]}],
            "updates": [{"itemIndex": 0, "sourceFactIndexes": [0]}],
        }
    )
    client = FakeJsonCompletionClient(raw)

    result = await StaticProfileUpdater(client).preupdate(
        (item(),),
        (fact(0), fact(1)),
        NOW,
    )

    assert result.pre_update == SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="用户喜欢篮球",
                source_fact_indexes=(1,),
            ),
        ),
        updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
    )
    assert result.request_payload == {
        "model": "qwen-plus",
        "messages": [dict(message) for message in client.calls[0]],
    }
    assert result.response_payload == {
        "content": raw,
        "input_tokens": 40,
        "output_tokens": 20,
    }
    payload = json.loads(client.calls[0][1]["content"])
    assert payload["existingItems"] == [
        {
            "index": 0,
            "summary": "用户每周运动",
            "lastSupportedAt": NOW.isoformat(timespec="seconds"),
        }
    ]


@pytest.mark.asyncio
async def test_static_preupdate_drops_empty_operation_and_keeps_valid_sibling() -> None:
    """Catches a Static no-evidence marker discarding a valid Create sibling."""
    raw = json.dumps(
        {
            "creates": [
                {"summary": "无证据长期主题", "sourceFactIndexes": []},
                {"summary": "用户长期喜欢篮球", "sourceFactIndexes": [0]},
            ],
            "updates": [{"itemIndex": 0, "sourceFactIndexes": []}],
        }
    )

    result = await StaticProfileUpdater(FakeJsonCompletionClient(raw)).preupdate(
        (item(),),
        (fact(0),),
        NOW,
    )

    assert result.pre_update == SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="用户长期喜欢篮球",
                source_fact_indexes=(0,),
            ),
        ),
        updates=(),
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
async def test_static_preupdate_deduplicates_indexes_with_stable_order() -> None:
    """Catches a duplicate Static evidence index causing a terminal retry loop."""
    raw = json.dumps(
        {
            "creates": [
                {
                    "summary": "用户长期喜欢篮球",
                    "sourceFactIndexes": [1, 0, 1],
                }
            ],
            "updates": [],
        }
    )

    result = await StaticProfileUpdater(FakeJsonCompletionClient(raw)).preupdate(
        (),
        (fact(0), fact(1)),
        NOW,
    )

    assert result.pre_update.creates == (
        PreCreateOperation(
            summary="用户长期喜欢篮球",
            source_fact_indexes=(1, 0),
        ),
    )
    assert result.response_payload["normalizations"] == [
        {
            "kind": "create",
            "index": 0,
            "field": "sourceFactIndexes",
            "original": [1, 0, 1],
            "normalized": [1, 0],
            "reason": "duplicate_source_fact_indexes",
        }
    ]


@pytest.mark.asyncio
async def test_static_preupdate_drops_conflicted_operations_not_the_batch() -> None:
    """Catches ambiguous Static routing blocking unrelated valid operations."""
    raw = json.dumps(
        {
            "creates": [
                {
                    "summary": "受歧义事实影响的长期主题",
                    "sourceFactIndexes": [0, 1],
                },
                {"summary": "不受影响的长期主题", "sourceFactIndexes": [3]},
            ],
            "updates": [
                {"itemIndex": 0, "sourceFactIndexes": [1, 2]},
            ],
        }
    )

    result = await StaticProfileUpdater(FakeJsonCompletionClient(raw)).preupdate(
        (item(),),
        tuple(fact(index) for index in range(4)),
        NOW,
    )

    assert result.pre_update == SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="不受影响的长期主题",
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
async def test_static_prompt_owns_long_term_eligibility_and_routing() -> None:
    client = FakeJsonCompletionClient('{"creates":[],"updates":[]}')

    await StaticProfileUpdater(client).preupdate(
        (item(),),
        (fact(0, source_role=Role.ASSISTANT),),
        NOW,
    )

    prompt = client.calls[0][0]["content"]
    assert STATIC_PROFILE_ELIGIBILITY_CONTRACT in prompt
    assert ATOMIC_RELATION_ITEM_CONTRACT not in prompt
    assert SNAPSHOT_SUPPORT_COVERAGE_CONTRACT not in prompt
    assert "不得套用 Dynamic Snapshot" in prompt
    assert "必须先独立判断每条 new Fact 是否具备 Static 资格" in prompt
    assert prompt.index("必须先独立判断") < prompt.index("## 长期主题路由")
    assert "existingItems 可能包含历史误收内容" in prompt
    assert "不能把它们当作准入正例" in prompt
    assert "只服务当前互动的一次性问题、请求、命令、任务、提醒" in prompt
    assert "天气、价格、新闻、搜索、推荐、设备状态" in prompt
    assert "短期状态、短期计划、待办、承诺、等待结果" in prompt
    assert "没有证据表明会长期有效的信息" in prompt
    assert "单次行为不能证明长期习惯或偏好" in prompt
    assert "单个任务无论是否由外部系统持久化，都不属于 Static" in prompt
    assert "今天搬到杭州，今后长期住这里" in prompt
    assert "今天去杭州出差" in prompt
    assert "优先 Update 表达同一长期属性" in prompt
    assert "只有合格 Fact 表达无法归入 existingItems 的独立新长期主题" in prompt
    assert "不得为了 Relation 粒度" in prompt
    assert "用户询问如何换水" not in prompt
    assert "sourceRole=user" in prompt
    assert "不得在此阶段重写摘要" in prompt
    assert "分类" in prompt
    assert "废弃操作" in prompt
    assert "每个 create 或 update 的 sourceFactIndexes 必须非空" in prompt
    assert "不得返回 sourceFactIndexes=[]" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"creates":[{"summary":"用户喜欢篮球","category":"user",'
        '"sourceFactIndexes":[0]}],"updates":[]}',
        '{"creates":[{"summary":"用户喜欢篮球","status":"active",'
        '"sourceFactIndexes":[0]}],"updates":[]}',
        '{"creates":[],"updates":[{"itemIndex":0,"summary":"错误摘要",'
        '"sourceFactIndexes":[0]}]}',
        '{"creates":[],"updates":[],"deprecates":[]}',
    ],
)
async def test_static_preupdate_rejects_legacy_fields(content: str) -> None:
    with pytest.raises(ValidationError):
        await StaticProfileUpdater(FakeJsonCompletionClient(content)).preupdate(
            (item(),),
            (fact(0),),
            NOW,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"creates":[{"summary":"用户喜欢篮球"}],"updates":[]}',
        '{"creates":[],"updates":[{"itemIndex":0,"sourceFactIndexes":null}]}',
        '{"creates":[{"summary":"用户喜欢篮球","sourceFactIndexes":"0"}],"updates":[]}',
        '{"creates":[],"updates":[{"sourceFactIndexes":[]}]}',
        '{"creates":[],"updates":[{"itemIndex":0,"sourceFactIndexes":[],'
        '"extra":true}]}',
        '{"creates":[{"summary":"用户喜欢篮球",'
        '"sourceFactIndexes":[1,1]}],"updates":[]}',
        '{"creates":[{"summary":" ","sourceFactIndexes":[0]}],"updates":[]}',
        '{"creates":[{"summary":"用户喜欢篮球","sourceFactIndexes":[1]}],"updates":[]}',
        '{"creates":[],"updates":[{"itemIndex":1,"sourceFactIndexes":[0]}]}',
    ],
)
async def test_static_preupdate_rejects_invalid_or_reused_indexes(
    content: str,
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        await StaticProfileUpdater(FakeJsonCompletionClient(content)).preupdate(
            (item(),),
            (fact(0),),
            NOW,
        )
