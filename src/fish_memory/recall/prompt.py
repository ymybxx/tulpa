import calendar
import json
from collections.abc import Sequence
from datetime import datetime

from fish_memory.answer.context import ChatMessage
from fish_memory.clock import format_prompt_timestamp, system_local_time

_MONTH_NAMES = (
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

RECALL_PLAN_SYSTEM_PROMPT = """你负责为一个长期记忆查询生成最小结构化检索条件。

输入中的 content 是当前用户的原始问题，只能用于提取结构化检索条件；
content 不一定能脱离 conversationContext 独立理解。conversationContext 是紧邻查询的
最近对话，只能用于消解 content 中明确存在的指代或省略，不得改变检索目标，
也不得把上下文中没有被 content 承接的信息加入条件。
向量检索 Query 由调用方另行提供，不属于本输出协议；不得输出或改写向量检索 Query。
禁止生成 event_type、标签或其他等值分类条件。

sourceRole 只能是 user、assistant 或 either：
- user：回答依赖用户说过、做过、经历过或处于何种状态；
- assistant：回答依赖助手曾说过、推荐过、承诺过或做过什么；
- either：明确需要两边，或者无法可靠判断来源。
sourceRole 只描述证据来源，不得改变 content、Entity 或时间。

entityHints 是问题或上下文中明确出现、可能帮助定位相关记忆的实体名称数组，
包括命名实体、物品、地点、作品、组织和明确主题。不得添加原文没有支持的名称，
不得把“用户”“助手”“我”“你”等通用说话角色作为实体。没有明确实体时返回空数组。

time.start 和 time.end 使用带时区的 ISO 8601。根据 currentTime 理解相对时间，
但不得编造问题无法支持的时间范围。输入时间和日历使用用户当前系统时区，
currentTime 精确到秒并带时区偏移。
只有问题明确给出日期、时间点、相对日期或时间范围时才能设置 start/end。
持续多久、花费多少小时、年龄、数量和金额只是 Fact 内容属性，不是时间 Scope。
最近一次、最早、第一次、最后一次不生成排序字段；没有明确范围时 start/end 都为 null。

只返回以下 JSON 对象，不要输出解释、Markdown 或其他字段：
{
  "sourceRole":"user、assistant 或 either",
  "entityHints":["明确出现的实体名称"],
  "time":{
    "start":"带时区 ISO 8601 或 null",
    "end":"带时区 ISO 8601 或 null"
  }
}
"""


def build_recall_plan_messages(
    content: str,
    current_time: datetime,
    *,
    conversation_context: Sequence[ChatMessage] = (),
) -> list[ChatMessage]:
    calendar_reference = _calendar_reference(current_time)
    payload = {
        "currentTime": format_prompt_timestamp(current_time),
        "conversationContext": [dict(message) for message in conversation_context],
        "content": content,
    }
    return [
        {
            "role": "system",
            "content": f"{RECALL_PLAN_SYSTEM_PROMPT}\n\n{calendar_reference}",
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def _calendar_reference(current_time: datetime) -> str:
    local_time = system_local_time(current_time)
    month_calendar = calendar.Calendar(firstweekday=calendar.MONDAY)
    weeks = month_calendar.monthdayscalendar(
        local_time.year,
        local_time.month,
    )
    dated_weeks = month_calendar.monthdatescalendar(
        local_time.year,
        local_time.month,
    )
    lines = [
        "# Calendar reference",
        "Dates use the user's system timezone. "
        "A calendar week runs Monday through Sunday.",
        (
            f"Current time: {format_prompt_timestamp(local_time, sep=' ')} "
            f"({_WEEKDAY_NAMES[local_time.weekday()]})."
        ),
        (
            '"last week" means the complete previous calendar week, '
            "not the previous seven rolling days."
        ),
        "",
        f"{_MONTH_NAMES[local_time.month]} {local_time.year}",
        "Mo Tu We Th Fr Sa Su",
    ]
    lines.extend(
        " ".join(f"{day:2d}" if day else "  " for day in week).rstrip()
        for week in weeks
    )
    lines.append("")
    lines.append("Boundary week dates, including adjacent months:")
    boundary_weeks = (dated_weeks[0], dated_weeks[-1])
    lines.extend(" ".join(day.isoformat() for day in week) for week in boundary_weeks)
    return "\n".join(lines)
