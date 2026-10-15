import json
from collections.abc import Sequence

from fish_memory.answer.context import ChatMessage
from fish_memory.clock import format_prompt_timestamp
from fish_memory.snapshots.models import SnapshotFact, SnapshotItemState
from fish_memory.snapshots.static_prompt import STATIC_PROFILE_ELIGIBILITY_CONTRACT

STATIC_PROFILE_CONFIRMATION_SYSTEM_PROMPT = (
    """你负责确认 Static Profile 中一个长期条目的预更新标记。

这是 Static 专用确认，不得套用 Dynamic Snapshot、任务生命周期或 Recall Relation 的
准入与拆分规则。必须先重新判断每条 new Fact 的 Static 资格，再决定 Update、Create
或 Ignore；不能因为第一阶段已经标记它就假定它合格。

"""
    + STATIC_PROFILE_ELIGIBILITY_CONTRACT
    + """

## 证据与索引

targetItem 是唯一可更新的原条目。supportFacts 是该条目当前支持事实，newFacts 是这次
预更新标记分配的新事实。supportFacts 的 index 只能用于 removeSupportIndexes；newFacts
的 index 只能用于 newSupportIndexes。两种索引空间都从 0 开始且仅在本次确认内有效。

contextOnlyRecentTurns 只用于消解指代和理解上下文，绝不能作为支持证据，也不能从中选择
任何 index。只有 supportFacts 与 newFacts 能支持最终摘要。

## Static 确认操作

- 不具备 Static 资格的 new Facts 必须 action=ignore；即使它们与 targetItem 共享实体、
  地点或话题，也不得 Update 或 Create。
- 合格 new Facts 延续 targetItem 的同一长期属性、状态、关系、偏好、习惯或规则时，返回
  action=update。itemIndex 必须等于 targetItem.itemIndex。
- 第一阶段标错目标、但 new Facts 自身明确具备 Static 资格并形成独立新长期主题时，才可
  action=create。短期独立事项必须 Ignore，不能因为独立而 Create。
- 一个确认至多包含一个 Update，可以同时包含合格的 Creates 和不合格的 Ignores。
- 每个 new Fact 的 index 必须恰好出现在一个操作的 newSupportIndexes 中，不得遗漏或复用。

Update summary 必须是紧凑的当前长期结论。它应保留仍由旧 supportFacts 支持的长期核心，
并只加入 newSupportIndexes 支持的长期内容。只有 new Facts 明确证明旧长期内容已被纠正、
替代或不再成立时，才能通过 removeSupportIndexes 移除旧支持。单次例外、近期状态、
任务进展或临时指令不得改写已有长期属性。

Create summary 同样只能表达可脱离原对话理解的长期核心结论。Ignore 没有 summary、
itemIndex 或 removeSupportIndexes。basis 是不超过 100 个字符的简短决策依据，
不是推理过程。

不得返回 Fact UUID、数据库条目 ID、category、status、废弃操作、Markdown、解释或
额外字段。
只返回 JSON 对象，结构必须严格如下：
{"operations":[
  {"action":"update","basis":"同一长期状态被明确更新","itemIndex":0,
   "newSupportIndexes":[0],"removeSupportIndexes":[0],"summary":"更新后的长期结论"},
  {"action":"create","basis":"独立且明确的长期属性","newSupportIndexes":[1],
   "summary":"独立长期主题摘要"},
  {"action":"ignore","basis":"一次性问题不属于 Static","newSupportIndexes":[2]}
]}
"""
)


def build_static_profile_confirmation_messages(
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
        {"role": "system", "content": STATIC_PROFILE_CONFIRMATION_SYSTEM_PROMPT},
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
