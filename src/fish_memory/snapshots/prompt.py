import json
from collections.abc import Sequence
from datetime import datetime

from fish_memory.answer.context import ChatMessage
from fish_memory.clock import format_prompt_timestamp
from fish_memory.snapshots.atomicity_prompt import ATOMIC_RELATION_ITEM_CONTRACT
from fish_memory.snapshots.coverage_prompt import SNAPSHOT_SUPPORT_COVERAGE_CONTRACT
from fish_memory.snapshots.models import SnapshotFact, SnapshotItemState

SNAPSHOT_PREUPDATE_SYSTEM_PROMPT = (
    """你负责为 Dynamic 短期记忆快照执行第一阶段预更新。

existingItems 和 newFacts 都显式提供从 0 开始连续递增的零基数组下标 index。
你只能通过 itemIndex 引用 existingItems 中显式提供的 index 字段，
只能通过 sourceFactIndexes 引用 newFacts 中显式提供的 index 字段。
两个数组都按照从老到新的顺序排列。

newFacts 的 sourceRole 是数据库确认的原始消息角色：
- sourceRole=user 表示该事实来自用户的明确表达；
- sourceRole=assistant 默认只证明助手说过或做过什么；
- 助手的提问、猜测、建议、情绪、复述和解释不能证明用户或外部世界的状态；
- 助手明确作出的承诺、拒绝和自我设定可以证明助手自身状态。

sourceFactIndexes 必须是真正支持本次路由的事实证据，不是周边对话上下文。
不得引用只用于帮助理解、但没有提供新信息的 Fact。

Dynamic 快照只保留后续回答需要的近期上下文：正在持续的活动和关注点、
未完成的计划、问题、承诺、等待结果，以及当前仍有关联的经历、状态、
情绪、态度、关系和互动氛围。忽略普通问候、一次性普通提问、无后续影响
的对话过程、助手的普通反应，以及未经确认的猜测。

对每条新事实，先按照下方“同一个具体记忆事项”规则判断：
- 满足 Update 条件时，为已有条目返回 update 标记；update 只包含 itemIndex 和
  sourceFactIndexes，不得在此阶段重写摘要。
- 不满足 Update 条件、但形成值得保留的独立具体事项时，返回 create；create 必须包含
  可脱离原始对话理解的 summary 和 sourceFactIndexes。
仅共享相同实体、地点、领域或大主题时仍必须 Create。

一个新 Fact 至多只能分配给一个 create 或 update 标记。没有合适操作时忽略该 Fact。
每个 create 或 update 的 sourceFactIndexes 必须非空，并且只包含真正支持该操作的新 Fact。
没有可选择的新 Fact 时不要输出该操作；不得返回 sourceFactIndexes=[]。
不得返回 Fact UUID、数据库条目 ID、分类、status、废弃操作，或 update 的摘要。
不得返回 Markdown、解释或额外字段。

只返回 JSON 对象，结构必须严格如下：
{"creates":[{"summary":"新事项摘要","sourceFactIndexes":[0]}],
"updates":[{"itemIndex":0,"sourceFactIndexes":[1]}]}
"""
    + ATOMIC_RELATION_ITEM_CONTRACT
    + SNAPSHOT_SUPPORT_COVERAGE_CONTRACT
)


def build_snapshot_preupdate_messages(
    existing_items: Sequence[SnapshotItemState],
    new_facts: Sequence[SnapshotFact],
    current_time: datetime,
) -> list[ChatMessage]:
    payload = {
        "currentTime": format_prompt_timestamp(current_time),
        "existingItems": [
            {
                "index": index,
                "summary": item.summary,
                "lastSupportedAt": format_prompt_timestamp(item.last_supported_at),
            }
            for index, item in enumerate(existing_items)
        ],
        "newFacts": [
            {
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
            for index, fact in enumerate(new_facts)
        ],
    }
    return [
        {"role": "system", "content": SNAPSHOT_PREUPDATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
