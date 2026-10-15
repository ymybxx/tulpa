import json
from collections.abc import Sequence

from fish_memory.answer.context import ChatMessage
from fish_memory.clock import format_prompt_timestamp
from fish_memory.snapshots.atomicity_prompt import ATOMIC_RELATION_ITEM_CONTRACT
from fish_memory.snapshots.coverage_prompt import SNAPSHOT_SUPPORT_COVERAGE_CONTRACT
from fish_memory.snapshots.models import SnapshotFact, SnapshotItemState

SNAPSHOT_CONFIRMATION_SYSTEM_PROMPT = (
    """你负责确认一次已有快照条目的预更新标记。

targetItem 是唯一可更新的原条目。supportFacts 是该条目当前支持事实，
newFacts 是这次预更新标记专门分配给它的新事实。supportFacts 的 index 只能用于
removeSupportIndexes；newFacts 的 index 只能用于 newSupportIndexes。两种索引空间都从 0
开始且仅在本次确认内有效。不得返回 Fact UUID、数据库条目 ID、category、status
或废弃操作。

contextOnlyRecentTurns 仅用于消解对话指代和理解上下文，绝不能当作支持证据，
也不能从中选择任何 index。只有 supportFacts 与 newFacts 能支持最终摘要；不得把
contextOnlyRecentTurns
当作支持证据。

若新事实延续 targetItem，返回恰好一个 action=update 的操作；其 itemIndex 必须等于
targetItem.itemIndex。update 的 summary 必须保留未通过 removeSupportIndexes 移除的
旧支持内容，并且只能加入由 newSupportIndexes 选择的新事实支持的内容。若标记目标错误，
可只返回一个或多个 action=create；create 不得有 itemIndex 或 removeSupportIndexes。
每个 action=create 必须包含非空 newSupportIndexes，选择真正支持该新 Item 的 assigned
newFacts，并同时包含 basis 和 summary；不能只返回 Summary。
每个 action=update 必须包含 itemIndex、newSupportIndexes 和 removeSupportIndexes，
并同时包含 basis 和 summary；两个索引数组即使为空也不能省略。
每个 new Fact 最多支持一个操作。

确认阶段必须重新检查新旧 Facts 是否属于同一个具体记忆事项，不能盲从预更新标记。
预更新把不同事项标到同一 Item 时，返回 Create-only，把 assigned newFacts 拆成一个或
多个独立 create，保留原 Item 不变。
若部分 newFacts 延续 targetItem、其他 newFacts 是独立事项，可返回恰好一个 Update 与\
一个或多个 Create；\
每个 new Fact 仍最多支持一个操作。

removeSupportIndexes 不是摘要裁剪工具。
只有 assigned newFacts 明确证明旧支持已经完成、取消、纠正或被取代，
才能移除对应 supportFacts。
不能因为摘要要缩短、焦点转移、时间较早或相关性较低而移除。
只要 assigned newFacts 没有明确解决旧内容，必须保留仍未被解决的计划、待办、承诺和状态，
并在 update summary 中继续表达。若新事实只延续 targetItem 的一部分，Update
必须保留其余未解决要素；若无法在一个自包含 Summary 中忠实保留，则将新事实返回为 Create，
不要破坏原 Item。

basis 是不超过 100 个字符的简短决策依据，不是推理过程。每次确认至少返回一个操作。
只返回 JSON 对象，不要解释或 Markdown：
{"operations":[{"action":"update","basis":"简短依据","itemIndex":0,
"newSupportIndexes":[0],"removeSupportIndexes":[0],"summary":"更新摘要"},
{"action":"create","basis":"独立新事项","newSupportIndexes":[1],
"summary":"独立新事项摘要"}]}
"""
    + ATOMIC_RELATION_ITEM_CONTRACT
    + SNAPSHOT_SUPPORT_COVERAGE_CONTRACT
    + (
        "确认 update 时，Summary 必须覆盖所有未被合法移除的旧 current support，\n"
        "以及本次 newSupportIndexes 选择的新命题。若无法同时覆盖，使用 Create-only "
        "操作保留旧 Item。\n"
    )
).replace(
    "不得把\ncontextOnlyRecentTurns\n当作支持证据。",
    "不得把 contextOnlyRecentTurns 当作支持证据。",
)


def build_snapshot_confirmation_messages(
    *,
    item_index: int,
    item: SnapshotItemState,
    support_facts: Sequence[SnapshotFact],
    new_facts: Sequence[SnapshotFact],
    recent_turns: Sequence[ChatMessage],
) -> list[ChatMessage]:
    payload = {
        "targetItem": {"itemIndex": item_index, "summary": item.summary},
        "supportFacts": [
            _fact_payload(index, fact) for index, fact in enumerate(support_facts)
        ],
        "newFacts": [
            _fact_payload(index, fact) for index, fact in enumerate(new_facts)
        ],
        "contextOnlyRecentTurns": [
            {"role": turn["role"], "content": turn["content"]} for turn in recent_turns
        ],
    }
    return [
        {"role": "system", "content": SNAPSHOT_CONFIRMATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _fact_payload(index: int, fact: SnapshotFact) -> dict[str, object]:
    return {
        "index": index,
        "sourceRole": fact.source_role.value,
        "text": fact.text,
        "occurredStart": format_prompt_timestamp(fact.occurred_start)
        if fact.occurred_start is not None
        else None,
        "occurredEnd": format_prompt_timestamp(fact.occurred_end)
        if fact.occurred_end is not None
        else None,
        "mentionedAt": format_prompt_timestamp(fact.mentioned_at),
        "entities": list(fact.entities),
    }
