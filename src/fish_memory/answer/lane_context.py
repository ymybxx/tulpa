from __future__ import annotations

import json
from typing import cast

from fish_memory.answer.context import (
    ANSWER_LANGUAGE_RULES,
    ANSWER_LANGUAGE_TURN_REMINDER,
    DEFAULT_RECENT_CONTEXT_LIMIT,
    END_CONVERSATION_DECISION_RULES,
    RECALL_DEPTH_DECISION_RULES,
    TURN_RECALL_OUTPUT_CONSISTENCY_CHECK,
    WEB_SEARCH_DECISION_RULES,
    ChatImageAttachment,
    ChatMessage,
    with_request_time_last,
)

_MEMORY_PREFIX = "# 记忆摘要"

RQUERY_SEMANTIC_PRESERVATION_RULES = (
    "## r 与 q 语义保持\n"
    "- 先根据原始问题确定 r，再生成 q。\n"
    "- 已经可独立理解时，q 默认保持原文；只允许为明确指代或省略补充上下文。\n"
    "- 生成 q 时必须保留原问题中的对象、属性、集合范围、并列条件、否定、数量、"
    "时间限定、条件、比较、顺序和状态；"
    "不得用 Working Memory 的一个候选替换原问题集合。\n"
    "- 相对时间表达必须原样保留；不得换算或补充绝对日期，"
    "除非该日期已明确出现在原问题中；"
    "具体日历范围由后续 Planner 统一解析。\n"
    "- 不得把猜测答案写入 q；无法保证逻辑等价时返回原文。"
)

RECALL_LANE_CONTRACT = f"""# R 通道输出协议
只返回一个紧凑 JSON 对象。
字段顺序固定为 m、a、r、q、e；r=0 时省略 q：
- r=0 的字段依次为 m、a、r、e；
- r=1 的字段依次为 m、a、r、q、e。
完整结构示例（值仅示意结构，不得照抄）：
- r=0：{{"m":"neutral","a":"非空的自然回答","r":0,"e":0}}
- r=1：{{"m":"neutral","a":"非空的自然前置过渡","r":1,"q":"可独立理解的检索查询","e":0}}

{RECALL_DEPTH_DECISION_RULES}

{RQUERY_SEMANTIC_PRESERVATION_RULES}

{END_CONVERSATION_DECISION_RULES}

{ANSWER_LANGUAGE_RULES}

## answer 与 query
- m 必须最先输出，只能是 neutral、angry、happy、shy、sad 之一。
- a 紧随 m。r=0 时直接完整回答；r=1 时只输出自然、简短的前置过渡。
- r 与 a 必须遵守上述 Recall 规则并保持一致。
- 当前上下文或记忆摘要已明确、完整写出答案时必须 r=0 并直接回答，不能为了生成 q
  而忽略可见证据。例如摘要写明“用户最近使用 Spotify 听歌”，询问最近用什么听歌时
  应直接回答 Spotify；“最近”在这里不自动触发 Recall。
- r=1 时 q 必须是可独立理解的检索查询，保留问题的对象、属性和时间限定；
  r=0 时不得输出 q。
不要输出 Markdown、解释或 JSON 之外的内容。"""

WEB_LANE_CONTRACT = f"""# W 通道输出协议
只判断当前问题是否需要互联网中的公开、外部或实时资料。
拒绝时只返回完整闭合对象：{{"w":0}}。一旦判定 w=0，必须立即以 }} 结束，
不得返回 {{"w":0,，也不得追加 m、a、q 或任何其他字段。
接受时字段顺序固定为 w、m、a、q：
{{"w":1,"m":"neutral","a":"非空的自然联网前置过渡","q":"可独立执行的联网查询"}}
示例中的值仅示意结构，不得照抄。

{WEB_SEARCH_DECISION_RULES}

{ANSWER_LANGUAGE_RULES}

- m 只能是 neutral、angry、happy、shy、sad 之一。
- a 只能是自然、简短的联网前置过渡，不得提前回答或声称无法联网。
- q 必须是可独立执行的联网查询，保留当前问题的实时性和外部目标。
不要输出 Markdown、解释或 JSON 之外的内容。"""

HOME_LANE_RULES = """## H 判断规则
- 只有当前消息确实要求查询或操作智能家居设备时 h=1。
- 讨论设备、修改回答、泛指“打开/关闭”但没有家居设备意图时 h=0。
- h=1 时，设备状态读取、盘点或筛选使用 t=query；改变设备状态或触发动作使用 t=control。
- 不判断设备是否存在，不识别设备和参数，不执行动作；这些由后续范围规划与执行层处理。"""

HOME_LANE_CONTRACT = f"""# H 通道输出协议
拒绝时只返回：{{"h":0}}
接受时字段顺序固定为 h、t、m、a：

{HOME_LANE_RULES}

{ANSWER_LANGUAGE_RULES}

- m 只能是 neutral、angry、happy、shy、sad 之一。
- a 只能是自然、简短的查询或执行前置过渡，不得声称查询或操作已经完成。
不要输出 Markdown、解释或 JSON 之外的内容。"""

RECALL_CURRENT_USER_RULES = (
    "# 当前问题的 R 通道复核\n"
    f"{ANSWER_LANGUAGE_TURN_REMINDER}\n\n"
    f"{RECALL_DEPTH_DECISION_RULES}\n\n"
    f"{RQUERY_SEMANTIC_PRESERVATION_RULES}\n\n"
    f"{TURN_RECALL_OUTPUT_CONSISTENCY_CHECK}\n\n"
    f"{END_CONVERSATION_DECISION_RULES}"
)
WEB_CURRENT_USER_RULES = (
    "# 当前问题的 W 通道复核\n"
    f"{ANSWER_LANGUAGE_TURN_REMINDER}\n\n{WEB_SEARCH_DECISION_RULES}\n\n"
    '# W 输出最终复核\n判定 w=0 时只输出完整对象 {"w":0}，立即闭合。'
)
HOME_CURRENT_USER_RULES = (
    f"# 当前问题的 H 通道复核\n{ANSWER_LANGUAGE_TURN_REMINDER}\n\n{HOME_LANE_RULES}"
)
CURRENT_USER_WRAPPER_RULES = (
    RECALL_CURRENT_USER_RULES,
    WEB_CURRENT_USER_RULES,
    HOME_CURRENT_USER_RULES,
)


def build_recall_lane_messages(
    messages: list[ChatMessage],
    *,
    recent_limit: int = DEFAULT_RECENT_CONTEXT_LIMIT,
) -> list[ChatMessage]:
    system, memory, recent, current = _lane_context(
        messages,
        recent_limit=recent_limit,
    )
    result: list[ChatMessage] = [
        {"role": "system", "content": f"{RECALL_LANE_CONTRACT}\n\n{system}"}
    ]
    if memory is not None:
        result.append(memory)
    result.extend(recent)
    wrapped = _wrapped_current(current, RECALL_CURRENT_USER_RULES)
    _copy_current_environment_image(messages, wrapped)
    result.append(wrapped)
    return with_request_time_last(result, source=messages)


def build_lane_base_messages(
    messages: list[ChatMessage],
    *,
    include_memory: bool,
    include_environment_image: bool = False,
    recent_limit: int = DEFAULT_RECENT_CONTEXT_LIMIT,
) -> list[ChatMessage]:
    system, memory, recent, current = _lane_context(
        messages,
        recent_limit=recent_limit,
    )
    result: list[ChatMessage] = [{"role": "system", "content": system}]
    if include_memory and memory is not None:
        result.append(memory)
    result.extend(recent)
    current_message: ChatMessage = {"role": "user", "content": current}
    if include_environment_image:
        _copy_current_environment_image(messages, current_message)
    result.append(current_message)
    return with_request_time_last(result, source=messages)


def build_web_lane_messages(
    messages: list[ChatMessage],
    *,
    recent_limit: int = DEFAULT_RECENT_CONTEXT_LIMIT,
) -> list[ChatMessage]:
    system, _memory, recent, current = _lane_context(
        messages,
        recent_limit=recent_limit,
    )
    result: list[ChatMessage] = [
        {"role": "system", "content": f"{WEB_LANE_CONTRACT}\n\n{system}"},
        *recent,
        _wrapped_current(
            current,
            WEB_CURRENT_USER_RULES,
        ),
    ]
    return with_request_time_last(result, source=messages)


def build_home_lane_messages(
    messages: list[ChatMessage],
    *,
    recent_limit: int = DEFAULT_RECENT_CONTEXT_LIMIT,
) -> list[ChatMessage]:
    system, _memory, recent, current = _lane_context(
        messages,
        recent_limit=recent_limit,
    )
    result: list[ChatMessage] = [
        {"role": "system", "content": f"{HOME_LANE_CONTRACT}\n\n{system}"},
        *recent,
        _wrapped_current(
            current,
            HOME_CURRENT_USER_RULES,
        ),
    ]
    return with_request_time_last(result, source=messages)


def _lane_context(
    messages: list[ChatMessage],
    *,
    recent_limit: int,
) -> tuple[str, ChatMessage | None, list[ChatMessage], str]:
    if recent_limit < 1:
        raise ValueError("recent lane context limit must be positive")
    if not messages or messages[0]["role"] != "system":
        raise ValueError("lane messages must start with one system message")
    if sum(message["role"] == "system" for message in messages) != 1:
        raise ValueError("lane messages must contain exactly one system message")
    if messages[-1]["role"] != "user":
        raise ValueError("lane messages must end with the current user message")
    memory: ChatMessage | None = None
    native: list[ChatMessage] = []
    for message in messages[1:-1]:
        copied: ChatMessage = {
            "role": message["role"],
            "content": message["content"],
        }
        if message.get("context_reference"):
            copied["context_reference"] = True
        if copied["role"] == "user" and copied["content"].startswith(_MEMORY_PREFIX):
            memory = copied
        else:
            native.append(copied)
    # Perception references are separate from the fifty real dialogue messages.
    dialogue = [
        i for i, message in enumerate(native) if not message.get("context_reference")
    ]
    selected = set(dialogue[-recent_limit:])
    return (
        messages[0]["content"],
        memory,
        [
            message
            for i, message in enumerate(native)
            if i in selected or message.get("context_reference")
        ],
        messages[-1]["content"],
    )


def _wrapped_current(content: str, rules: str) -> ChatMessage:
    payload = json.dumps(
        {"content": content},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "role": "user",
        "content": (f"# 当前用户消息（应用已进行 JSON 转义）\n{payload}\n\n{rules}"),
    }


def _copy_current_environment_image(
    source: list[ChatMessage],
    target: ChatMessage,
) -> None:
    image = source[-1].get("environment_image")
    if image is not None:
        target["environment_image"] = cast(ChatImageAttachment, dict(image))
    images = source[-1].get("environment_images")
    if images is not None:
        target["environment_images"] = tuple(
            cast(ChatImageAttachment, dict(item)) for item in images
        )
