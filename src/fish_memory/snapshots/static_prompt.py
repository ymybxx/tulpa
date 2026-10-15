import json
from collections.abc import Sequence
from datetime import datetime

from fish_memory.answer.context import ChatMessage
from fish_memory.clock import format_prompt_timestamp
from fish_memory.snapshots.models import SnapshotFact, SnapshotItemState

STATIC_PROFILE_ELIGIBILITY_CONTRACT = """## Static Profile 独立准入契约

Static Profile 是每次回答都会常驻的紧凑长期画像，不是 Dynamic 快照、历史日志、
对话摘要、任务列表、搜索结果或近期事件集合。只保留在未来较长时间和跨会话理解中
持续有用的当前核心状态、属性、关系、偏好、习惯、规则与共同认知。

必须先独立判断每条 new Fact 是否具备 Static 资格，再考虑 existingItems 和路由。
existingItems 可能包含历史误收内容，不能把它们当作准入正例；相同人物、地点、实体、
领域或话题也不能降低新 Fact 的 Static 门槛。

只允许以下信息进入：
- 用户或助手的稳定身份、名字、角色和长期设定；
- 明确的长期偏好、习惯、禁忌和反复适用的交互规则；
- 持续有效的人际关系、所有权关系或照护关系；
- 长期居住、工作、学习、健康等稳定状态；
- 长期有效并会持续影响未来理解的重要人物、地点、物品、关系、规则和共同认知。

Fact 必须自身明确表达长期、稳定、持续、反复适用的含义，或者其语义本身就是当前稳定
属性，例如姓名、职业、居住地或亲属关系。明确的长期核心信息首次出现即可进入，
不要求重复出现。一次事件只有在它明确建立、纠正或替代持续状态时才有资格；Summary
只保留持续结果，不保留事件过程。例如“今天搬到杭州，今后长期住这里”可支持
“用户长期居住在杭州”，而“今天去杭州出差”必须忽略。

必须忽略：
- 普通对话、问候，以及只服务当前互动的一次性问题、请求、命令、任务、提醒及其
  执行过程或结果；
- 天气、价格、新闻、搜索、推荐、设备状态等只对当前或短期有用的信息；
- 一次性事件、临时情绪、短期状态、短期计划、待办、承诺、等待结果和互动氛围；
- 仅仅被提到的人、作品、地点、应用或物品；
- 助手的普通回答、提问、建议、反应、复述、解释、工具结果和临时说法；
- 没有证据表明会长期有效的信息，以及未经确认的猜测、可能性、模糊指代或暗示；
- 为了完整而附带的近期场景、风险推断、操作步骤和对话过程。

单次行为不能证明长期习惯或偏好，单次例外也不能更新已有长期习惯或偏好。例如已有
“用户每天跑步”时，“用户今天没跑步”必须忽略；已有“用户长期使用中文”时，
“用户这次使用英文”必须忽略，只有“用户以后都使用英文”才可更新长期规则。
单个任务无论是否由外部系统持久化，都不属于 Static；只有用户明确表达的跨任务长期规则，
例如“以后所有提醒都提前十五分钟”，才有 Static 资格。

sourceRole 是数据库确认的原始消息角色：
- sourceRole=user 可以支持用户、助手设定和外部世界的长期信息；
- sourceRole=assistant 默认只证明助手说过或做过什么；
- assistant Fact 只有在助手明确接受自身长期身份或设定时才可进入；
- 助手的回答、工具结果、猜测、建议、情绪、复述和解释不能证明用户或外部世界的状态。
"""


STATIC_PROFILE_PREUPDATE_SYSTEM_PROMPT = (
    """你负责为 Static Profile 执行第一阶段预更新，并独立决定哪些新增 Facts 可以进入。

资格判断与路由必须完全遵守下方 Static Profile 独立契约，不得套用 Dynamic Snapshot、
任务生命周期或 Recall Relation 的准入与拆分规则。

"""
    + STATIC_PROFILE_ELIGIBILITY_CONTRACT
    + """

## 长期主题路由

existingItems 和 newFacts 中的每个对象都显式提供从 0 开始连续递增的 index。
你只能通过 itemIndex 引用 existingItems 中显式提供的 index，
只能通过 sourceFactIndexes 引用 newFacts 中显式提供的 index。
两个数组都按照从老到新的顺序排列。

sourceFactIndexes 必须是真正支持本次路由的事实证据，不是周边对话上下文。
不得引用只用于帮助理解、但没有提供长期信息的 Fact。

- 不具备 Static 资格的 Fact 必须忽略，不得 Create，也不得 Update，即使它能形成独立事项。
- 对合格 Fact，优先 Update 表达同一长期属性、状态、关系、偏好、习惯或规则的
  existing Item。
  Update 只包含 itemIndex 和 sourceFactIndexes，不得在此阶段重写摘要。
- 只有合格 Fact 表达无法归入 existingItems 的独立新长期主题时才 Create。Create 的
  summary 必须脱离原对话仍可理解，只表达长期核心结论，不得记录近期过程。
- 同义、重叠或属于同一长期主题的信息不能拆成多个 Items。一个 Item 只表达一个长期主题；
  不得为了 Relation 粒度把同一长期属性的身份细节、必要属性或演进拆成多个 Creates。
- 一次事件若建立持续状态，应合并或更新该持续状态；不得同时保存事件过程和持续结果。

一个新 Fact 至多只能分配给一个 create 或 update 标记。没有合适操作时忽略该 Fact。
每个 create 或 update 的 sourceFactIndexes 必须非空，并且只包含真正支持该操作的新 Fact。
没有可选择的新 Fact 时不要输出该操作；不得返回 sourceFactIndexes=[]。
不得返回 Fact UUID、数据库条目 ID、分类、status、废弃操作，或 update 的摘要。
不得返回 Markdown、解释或额外字段。

只返回 JSON 对象，结构必须严格如下：
{"creates":[{"summary":"独立长期主题摘要","sourceFactIndexes":[0]}],
"updates":[{"itemIndex":0,"sourceFactIndexes":[1]}]}
"""
)


def build_static_profile_preupdate_messages(
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
                "occurredStart": (
                    format_prompt_timestamp(fact.occurred_start)
                    if fact.occurred_start is not None
                    else None
                ),
                "occurredEnd": (
                    format_prompt_timestamp(fact.occurred_end)
                    if fact.occurred_end is not None
                    else None
                ),
                "mentionedAt": format_prompt_timestamp(fact.mentioned_at),
                "entities": list(fact.entities),
            }
            for index, fact in enumerate(new_facts)
        ],
    }
    return [
        {"role": "system", "content": STATIC_PROFILE_PREUPDATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
