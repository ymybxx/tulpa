"""Bounded, identifier-free prompt construction for the T lane."""

from __future__ import annotations

import json

from fish_memory.agent.routing import TaskLaneContext
from fish_memory.answer.context import (
    ChatMessage,
    add_perception_policy,
    with_request_time_last,
)
from fish_memory.answer.speech_style import SPOKEN_TEXT_RULES

TASK_LANE_CONTRACT = (
    """# T 通道输出协议
你是 T 通道路由器。最后 user JSON 的 content 是本轮消息。

- t=1：目标超出宿主当前能力与活动职责，需要外部 Agent 执行工作。
- t=0：普通交流、利用已提供的感知回答，或继续、调整宿主正在履行的活动职责。
单纯询问宿主的能力、感知或权限状态属于普通交流。没有要求检查、修复或改变环境时，
不得将这种询问扩写成权限诊断、能力验证或工具调用任务。

严格分两步判断：
1. 先冻结 t。先判断当前目标是否已在宿主能力和活动职责内，属于则 t=0。
不属于时，才借此前对话或 candidates 的 request 消解外部工作目标的省略。
候选存在、唯一、最近、未完成或相邻 Agent 回复都不能改变 t。没有新增或继续外部
工作的目标必须 t=0。
2. 仅当 t=1 后判断关联：与某候选的目标或 request 有明确语义关系才输出 c=i；
无关联或候选为空时省略 c，表示新任务。

严格只输出以下一种 JSON，字段顺序不变：
{"t":0,"m":"neutral","a":""}
{"t":1,"m":"happy","a":"我去检查当前耗电情况。"}
{"t":1,"m":"happy","a":"我继续排查启动错误。","c":0}
示例仅示意结构，禁止照抄目标或措辞。

m 只能是 neutral、angry、happy、shy、sad。
t=0 时 a 必须为空；t=1 时 a 的语言只由当前 content 决定，忽略前文与示例语言；
英文 content 必须用英文，中文 content 必须用中文。a 应确认当前具体目标，
不用无目标信息的通用套话，不能提问或声称任务已完成。
c 只能是 candidates 中的一个 i。不要输出其他字段、Markdown 或解释。"""
    + "\n\n"
    + SPOKEN_TEXT_RULES
)


def build_task_lane_messages(
    messages: list[ChatMessage],
    task_context: TaskLaneContext,
    *,
    recent_limit: int = 12,
) -> list[ChatMessage]:
    if recent_limit < 1:
        raise ValueError("recent lane context limit must be positive")
    if not messages or messages[0]["role"] != "system":
        raise ValueError("lane messages must start with one system message")
    if sum(message["role"] == "system" for message in messages) != 1:
        raise ValueError("lane messages must contain exactly one system message")
    if messages[-1]["role"] != "user":
        raise ValueError("lane messages must end with the current user message")
    recent: list[ChatMessage] = []
    for message in messages[1:-1]:
        if message["role"] == "user" and message["content"].startswith("# 记忆摘要"):
            continue
        copied: ChatMessage = {
            "role": message["role"],
            "content": message["content"],
        }
        recent.append(copied)
    recent = recent[-recent_limit:]
    payload = json.dumps(
        {
            "candidates": task_context.prompt_payload(),
            "content": messages[-1]["content"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    activity = messages[0].get("activity_context", "")
    perception = messages[0].get("perception_context", "")
    host_context = ""
    if activity or perception:
        host_context = (
            "\n\n# 宿主当前上下文\n"
            "宿主能依据已提供的感知交流、履行当前活动职责；"
            "延续或调整这些交流不需要转发外部任务。"
            "外部任务只处理超出这些能力的工作；过去任务对自身能力的声明"
            "不代表宿主的能力。\n" + activity
        )
    result: list[ChatMessage] = [
        {"role": "system", "content": TASK_LANE_CONTRACT + host_context},
        *recent,
        {"role": "user", "content": f"# 当前用户消息与候选\n{payload}"},
    ]
    add_perception_policy(result, perception)
    if perception:
        result[0]["content"] += (
            "\n以上开关是宿主本轮的已知状态，询问这些能力时直接交流即可（t=0），"
            "不能另行委派任务确认这些已知状态。"
        )
    return with_request_time_last(result, source=messages)
