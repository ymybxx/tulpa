import json

from fish_memory.answer.context import ChatMessage
from fish_memory.clock import format_prompt_timestamp
from fish_memory.facts.models import FactExtractionContext
from fish_memory.models import Message

FACT_OUTPUT_SCHEMA = (
    '{"facts":[{"what":"非空的核心事实",'
    '"where":["明确地点"],'
    '"who":["明确人物或主体"],'
    '"occurredStart":"带时区的 ISO 8601 时间或 null",'
    '"occurredEnd":"带时区的 ISO 8601 时间或 null",'
    '"entities":["用于关联相关事实的实体或主题"],'
    '"text":"基于前述字段生成的非空完整事实文本"}]}'
)


FACT_EXTRACTION_SYSTEM_PROMPT = f"""# 任务

把一条聊天消息展开为高召回的语义原材料，供后续融合、筛选、快照和检索使用。
不要根据长期价值、重要性或是否值得记忆进行筛选。

# 输出边界

- 提取所有具有语义的信息，包括陈述、偏好、习惯和计划。
  还包括请求、命令、问题、承诺、拒绝和回应，
  以及问候、情绪、态度、感受、观察、关系、纠正、否定、推测和不确定性。
- 简短表达只要传达了态度、情绪、确认或拒绝，也应提取。
  只有纯空白或完全无法表达语义的内容才返回零条事实。
- 每条 Fact 只表达一件事，不得合并彼此独立的事实；独立信息必须拆分为多条原子化事实。
- 每条 Fact 必须脱离原消息后仍能独立理解，目标不超过 1000 个字符。

# 角色与指代

sourceRole 只用于判断消息来源：
- sourceRole=user 时，“我”指该用户，“你”指该助手。
- sourceRole=assistant 时，“我”指该助手，“你”指该用户。

原文明确出现“我”或“你”时，必须按上述映射写成“该用户”或“该助手”，
不得再改写为“对方”或“原文中未指明的对象”。
无法确定指代对象时，不得猜测或编造身份。只有不属于上述映射、且确实无法确定时，
才使用“原文中未指明的人”或“原文中未指明的对象”等中性表述。
助手陈述的外部世界信息若无法从消息验证，将其改写为带来源的陈述事实，例如“该助手表示……”或“该助手声称……”。

# 相邻上下文

conversationContext 是当前消息前一条异角色消息。

1. 当前 content 决定是否生成 Fact，以及 Fact 所描述的事件、状态或互动。
2. 不得仅根据 conversationContext 创建 Fact。
   不得把其中未被当前 content 承接的信息另行生成 Fact。
3. 当前 content 通过“这、那、它、那里、上次、这次、也、还是、实际、actually”等表达，
   明确继续、补充、纠正或回答同一话题时，可用 conversationContext
   补齐同一事件的地点、人物、对象、时间或关系。
4. 若关联不明确，宁可不补齐，不得猜测。

示例：conversationContext 为“Cartwheel app 可用于在 Target 节省购物开支”，
当前 content 为 “I actually redeemed a $5 coupon on coffee creamer last Sunday.” 时，
Fact 可以写为“该用户于上周日在 Target 兑换了一张咖啡奶油的 5 美元优惠券”。
不得额外生成“该用户使用 Cartwheel app”等只来自上下文的独立 Fact。

# Fact 结构

每条 Fact 必须按输出 JSON 中的字段顺序生成：先生成 what、where、who、
occurredStart、occurredEnd，再生成 entities，最后才生成 text。

- what：核心动作、状态或陈述，不得为空。写入除人物、地点和主发生时间外的全部信息，
  包括对象、数量、频率、持续时间、计划、否定、不确定性、转述、条件、程度和原因。
- where：明确的地点数组；原文或可明确承接的上下文没有地点时返回 []。
- who：明确涉及的人物、主体或组织数组；没有时返回 []。根据 sourceRole 改写出的“该用户”或
  “该助手”是明确主体，应写入 who，但不要写入 entities。
- occurredStart / occurredEnd：这条 Fact 核心事件的发生区间。无法确定时都为 null。
- text：最后生成。必须只根据同一对象中已生成的 what、where、who、occurredStart、
  occurredEnd 和 entities 组成自然、完整、可独立理解的事实文本；
  不得添加这些字段无法支持的新信息。
  所有非空的 where、who 和已知发生时间都必须在 text 中明确体现。

多个独立事件必须拆成多条 Fact，各自拥有自己的主发生时间。一个事件的前因、计划日期、
截止日期或比较时间等无法作为主发生时间的时间信息，必须完整保留在 what 和 text 中。

# 时间

- 以 referenceTime 为基准解析相对时间。
- 不能确定时，将 occurredStart 和 occurredEnd 都设为 null。
- 按事件发生时间排列 Fact；无法确定时保持其在原文中的出现顺序。
- occurredStart / occurredEnd 是机器使用的时间字段，精度保留到秒并带时区；
  不得在 text 中照抄 ISO 8601 时间戳。
- text 中的时间必须使用自然语言，并且只写到原文能够支持的最小必要时间精度：
  原文仅能确定日期时，写“2026年7月30日”，不要补充时分秒；原文明确到具体时刻时，
  才写“14:30”等相应精度。由“现在”“刚刚”等相对表达依 referenceTime 推得时间时，
  默认只在 text 中写日期。
- 例如 content 为“现在要去吃饭”、referenceTime 为
  “2026-07-30T07:58:39+00:00”时，occurredStart 可以保留该时间，
  但 text 应写“该用户于2026年7月30日要去吃饭”，不得写入“07:58:39.719343”。

# 实体

entities 是用于关联相关事实的检索锚点，不限于传统命名实体。
提取直接支持且有助于关联的人物、动物、组织、地点、重要物品、活动、兴趣、习惯主题，
作品、项目、技术、事件以及抽象概念和主题。
名字、别名、昵称、称呼和外号也是实体。消息明确给某个人或对象命名、改名或使用称呼时，
被赋予的名称必须写入 entities，并在 text 中明确写出该名称指向的对象。
地点、人物、重要对象和主题同时写入 text 与 entities；where 和 who 中的值也应写入
entities，但“用户”“助手”“该用户”“该助手”等中性角色除外。
实体优先保持原文写法；不得因为 sourceRole 添加消息发送者本身。
不要将“用户”“助手”“该用户”“该助手”等中性角色写入 entities。
同一 Fact 内的同名实体最多保留一次。

# 语言与示例

所有输出值必须与原消息使用相同语言；专有名称保持原文写法，不要翻译。

输入 content 为“我每天打篮球”时，应输出：
{{"facts":[{{"what":"每天打篮球","where":[],"who":["该用户"],"occurredStart":null,"occurredEnd":null,"text":"该用户每天打篮球","entities":["篮球"]}}]}}

# 输出格式

只返回规定的 JSON 对象，不要输出解释、Markdown 或其他文本。JSON 结构必须严格如下：
{FACT_OUTPUT_SCHEMA}
"""


def build_fact_extraction_messages(
    message: Message,
    content: str,
    context: FactExtractionContext | None = None,
) -> list[ChatMessage]:
    source = {
        "referenceTime": format_prompt_timestamp(message.created_at),
        "sourceRole": message.role.value,
    }
    if context is not None and context.external_event is not None:
        source["externalEvent"] = context.external_event
        source["sourceAttribution"] = (
            "此 assistant 消息回应插件事件。观众、角色及发言均为外部来源。"
            "不是本机用户或主人的发言。仅提取实际播出内容所支持的事实，"
            "不把事件元数据当作用户偏好。"
        )
    elif context is not None:
        source["conversationContext"] = {
            "sourceRole": context.role.value,
            "content": context.content,
        }
    source["content"] = content
    return [
        {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(source, ensure_ascii=False, separators=(",", ":")),
        },
    ]
