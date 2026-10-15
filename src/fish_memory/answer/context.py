from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal, NotRequired, TypedDict

from fish_memory.answer.speech_style import (
    SPOKEN_TEXT_RULES,
    SPOKEN_TEXT_TURN_REMINDER,
)
from fish_memory.clock import format_prompt_timestamp
from fish_memory.context_descriptions import (
    ENVIRONMENT_IMAGE_CAPTION_PREFIX,
    ENVIRONMENT_IMAGE_DEFAULT_DESCRIPTION,
    HISTORY_SOURCE_RULES,
    PERCEPTION_CONTEXT_SYSTEM_RULE,
)
from fish_memory.models import Message, MessageStatus, Role, visible_message_content

INTERRUPTION_MARKER = "[这句话在此处被用户打断]"
DEFAULT_RECENT_CONTEXT_LIMIT = 50
_HISTORY_TIME_BUCKET = timedelta(minutes=10)

RECALL_DEPTH_DECISION_RULES = """## Recall 深度 r
r只衡量是否需要从当前上下文之外检索本地事实，不衡量回顾、整理或概括已有材料的工作。
本轮系统给出的感知开关与输入来源是已知事实，询问这些能力不需要Recall，r=0。
仅缺少外部或实时资料时 r=0，
外部资料由其他能力独立处理。询问实时价格但不依赖本地记忆时，r=0。
先按用户所问的详细程度判断，概况问题只需要有依据地概括，不要求补齐每次点击的细节。
判断回答所需信息是否被当前上下文明确、完整地支持，不要输出判断过程。
明确支持必须同时匹配问题的目标、关系、属性、事实状态和限定条件。
信息缺失、不明确、需要补充未给出的事实，或只有相关主题、对象和候选答案，都不算完整支持。
- r=0：问题不依赖本地记忆，或当前可见内容已直接、完整支持回答。
- r=1：回答依赖当前可见内容没有完整提供的本地事实。
单事实、多事实、计数、比较、计算和明确时间范围只要需要 Recall，都使用 r=1。

History中的as和summary是当前可见的近期经历，不是Static；其中明确的屏幕活动与声音观察也是直接证据。
对已有证据做归纳、排序或简述不等于缺少事实；已有观察足以回答所问概况时必须r=0并直接回答。
具体过往事件的细节只能由最近历史中的明确陈述直接支持。Static 是压缩摘要，
不参与此类细节的完整支持判定；即使摘要含有相关信息，也必须 Recall。
命令、请求执行动作本身不是历史问题。只有完成请求所需的信息依赖当前上下文没有提供的
过往事实时，才使用 r=1。"""

END_CONVERSATION_DECISION_RULES = """## 连续对话 e
- e 只能依据当前用户消息判断，不能依据任何历史消息、摘要或补充信息；
- 历史消息中的结束、离开或告别只描述过去，即使紧邻当前消息也不得令 e=1；
- 当前用户消息明确结束、离开或暂时不再继续时 e=1；
- 其他情况 e=0；当前用户重新提问、回应或继续交流时必须 e=0，
  无法确定时也 e=0。"""

ANSWER_LANGUAGE_RULES = (
    """## 用户可见文本的语言与起句
- a 是用户可见文本；使用 append 的续接调用中，append 也属于用户可见文本。
- 只根据真实用户消息判断回答语言。续接调用中的当前用户消息 content，专指明确标注的
  “原始当前用户消息”JSON，不是合成协议消息。
- 用户明确指定的语言优先：以当前或近期用户消息中最后一条尚未撤销的明确语言要求为准。
- 否则使用当前消息的主要语言；无法判断时沿用最近一条可判断的用户消息语言。
- 忽略协议、示例、工具结果的语言，不能因为这些内部文本是中文就默认用中文回答。
- 根据当前话题和角色自然生成回答或前置句；示例只说明结构，不得照抄。
- 检查近期 assistant 消息；不得复用近期已出现的相同或近似前置句，避免通用口头禅。"""
    + "\n\n"
    + SPOKEN_TEXT_RULES
)

ANSWER_LANGUAGE_TURN_REMINDER = (
    """# 当前回答语言复核
严格应用系统首部的语言与起句规则；JSON content 才是本轮用户原话，
忽略本条复核和其他内部文本的语言。"""
    + "\n\n"
    + SPOKEN_TEXT_TURN_REMINDER
)

INITIAL_ANSWER_CONTRACT = (
    """# 回答输出协议
只返回一个紧凑 JSON 对象，字段顺序固定为 r、m、a、e：
{"r":0,"m":"neutral","a":"非空的自然回答","e":0}
需要 Recall 时仍使用同一结构，例如：
{"r":1,"m":"sad","a":"自然承接","e":0}
不得输出检索查询、实体提示或其他字段。

"""
    + RECALL_DEPTH_DECISION_RULES
    + """

## 回答 a
a 必须是非空字符串，只能使用当前上下文明确支持的信息。
- r=0：直接自然回答；
- r=1：a 是等待补全的未完成回答。已有直接支持的信息时先回答该部分，
  再自然承接；完全没有可回答的信息时，只作一句简短自然的回想或了解式承接；
- a 不得填写问题所问属性的具体值，也不得填写尚待 Recall 补全的其他具体值；
- 不得拒绝回答、宣布信息缺失或向用户索取答案，不得要求用户提醒或确认；
- 不得编造、保留候选答案、复述问题，或提前声称不知道、没有记录、无法确认；
- 不得提及记录、记忆、历史、系统、上下文、召回、检索、数据库或参考资料，
  也不要说“根据我的记录”“我查到”“我翻到”“我查一下”或同义表达。
"""
    + ANSWER_LANGUAGE_RULES
    + """
## 人物表现 m
- m 是整段回答的主要情绪，只能是 neutral、angry、happy、shy、sad 之一；
  无法确定时使用 neutral；
- m 同时用于选择人物的表情与可选情绪动作，只描述人物表现，不得为了配合表现改变回答内容。
"""
    + END_CONVERSATION_DECISION_RULES
    + """

不要输出 Markdown、解释或 JSON 之外的内容。"""
)

WEB_SEARCH_DECISION_RULES = """## 联网判断 w
- w 只能是 0 或 1；w 不得省略；
- w=1 的必要条件是：当前消息明确要求联网、搜索、核实、来源或链接，或者问题明确
  依赖会随时间变化的当前外部状态；两者都不满足时必须 w=0。仅出现人名、专有名词、
  过去事件，或当前上下文没有答案，都不得改变这一结论；
- w 的判断与时间表达和 r 无关，不能用 r 代替 w；
- 用户与助手的过往对话属于本地记忆，不属于外部资料；
- 例如询问实时价格时为 w=1。"""

SEARCH_AWARE_INITIAL_ANSWER_CONTRACT = (
    INITIAL_ANSWER_CONTRACT.replace(
        "字段顺序固定为 r、m、a、e",
        "字段顺序固定为 r、w、m、a、e",
    )
    .replace(
        '{"r":0,"m":"neutral"',
        '{"r":0,"w":0,"m":"neutral"',
    )
    .replace(
        '{"r":1,"m":"sad"',
        '{"r":1,"w":0,"m":"sad"',
    )
    .replace(
        "## 回答 a",
        f"{WEB_SEARCH_DECISION_RULES}\n\n## 回答 a",
    )
    .replace(
        "- r=0：直接自然回答；",
        "- r=0 且 w=0：直接自然回答；",
    )
    .replace(
        "- r=1：a 是等待补全的未完成回答。",
        "- r=1 或 w=1：a 是等待补全的未完成回答。",
    )
    .replace(
        "尚待 Recall 补全的其他具体值",
        "尚待 Recall 或联网获得的其他具体值",
    )
    .replace(
        "检索、数据库或参考资料",
        "检索、数据库、联网接口或参考资料",
    )
)

_TURN_RECALL_GATE_REMINDER = f"""# 当前问题的来源判定复核
生成回答前重新应用首部完全相同的 r 规则；a、m 仍严格遵守首部系统协议。
该检查优先于模仿历史中的助手回答方式。

{RECALL_DEPTH_DECISION_RULES}

{END_CONVERSATION_DECISION_RULES}"""

_SEARCH_AWARE_TURN_SOURCE_REMINDER = f"""{_TURN_RECALL_GATE_REMINDER}

# 当前问题的联网来源判定复核
重新应用首部完全相同的 w 规则：

{WEB_SEARCH_DECISION_RULES}"""

TURN_RECALL_OUTPUT_CONSISTENCY_CHECK = """# 输出前的 r 一致性校验
先只依据当前可见内容判断是否有可定位的直接证据；
不得把模型常识、猜测或拟定的答案当作当前上下文证据。
确定直接支持后，先在内部拟定 a，再最终选择 r，并仍按既定字段顺序输出：
- 若 a 无法直接给出问题要求的具体属性值，而是表示不知道、没有信息、
  不确定或需要用户补充、提醒，且缺少的是更早的本地对话或记忆信息，
  则 r=0 非法，必须改为 r=1。
- 缺少外部或实时资料不属于本地 Recall，不能因此提高 r。
- r=0 意味着 a 能依据当前可见内容直接、完整回答所问属性，
  或问题根本不依赖本地记忆；不得用 r=0 表示“当前没有答案”，
  也不得因为能够猜出具体值而判定 r=0。
- 当 r=1 时，严格沿用首部系统协议的 a 规则：
  只作简短自然的回想式承接，不得声称没有信息，
  不得要求用户补充或提醒。
不要输出校验过程。"""


class ChatImageAttachment(TypedDict):
    data_url: str
    digest: str
    byte_count: int
    width: int
    height: int
    observed_at: str
    description: NotRequired[str]
    detail: NotRequired[Literal["low", "high", "auto"]]


class ChatUserAttachment(TypedDict):
    """A picture or text file the user sent with the message being answered."""

    kind: Literal["image", "text"]
    name: str
    data_url: NotRequired[str]
    text: NotRequired[str]


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    # Internal output capability, never serialized as a provider message field.
    speech_tags: NotRequired[bool]
    context_reference: NotRequired[bool]
    # A lossless alternative view of the same History, used only by experimental S.
    # Providers and Jev ignore this metadata; never send both views together.
    history_messages: NotRequired[tuple[ChatMessage, ...]]
    # Internal repetition checks still consume exact played utterances. Provider
    # serialization only sends role/content and explicit image attachments.
    recent_spoken: NotRequired[tuple[str, ...]]
    activity_context: NotRequired[str]
    perception_context: NotRequired[str]
    # Legacy input marker for experimental views; new contexts keep the clock
    # only in request_time, outside system text and dialogue.
    system_request_time: NotRequired[str]
    # Final response instruction, after observations/images and before the clock.
    response_reminder: NotRequired[str]
    # Render after the message text and all image attachments.
    request_time: NotRequired[str]
    environment_image: NotRequired[ChatImageAttachment]
    environment_images: NotRequired[tuple[ChatImageAttachment, ...]]
    # Sent only with the reply to the message that carried them; History
    # names them in the message text instead.
    user_attachments: NotRequired[tuple[ChatUserAttachment, ...]]


def with_request_time_last(
    messages: Sequence[ChatMessage], *, source: Sequence[ChatMessage] | None = None
) -> list[ChatMessage]:
    """Carry the explicit request clock through rewrites, after all input.

    Only internal metadata moves. Dates in user text, persona and History are
    untouched, and the source stays frozen for speculative generation.
    """
    clock_source = messages if source is None else source
    timestamp = next(
        (
            item["request_time"]
            for item in reversed(clock_source)
            if item.get("request_time")
        ),
        None,
    )
    if timestamp is None or not messages:
        return list(messages)
    result = [message.copy() for message in messages]
    for message in result:
        message.pop("request_time", None)
    result[-1]["request_time"] = timestamp
    return result


def user_attachment_parts(
    message: ChatMessage,
) -> list[tuple[Literal["text", "image"], str]]:
    """The user's files as text and pictures, in the order they were sent."""
    parts: list[tuple[Literal["text", "image"], str]] = []
    for attachment in message.get("user_attachments", ()):
        if attachment["kind"] == "text" and "text" in attachment:
            name, text = attachment["name"], attachment["text"]
            # Fenced so the model can tell where the file stops and the reply
            # rules that follow begin: unfenced, it read them as the file's.
            fence = "`" * max(3, _longest_backtick_run(text) + 1)
            parts.append(
                ("text", f"用户发来的文件《{name}》：\n{fence}\n{text}\n{fence}")
            )
        elif attachment["kind"] == "image" and "data_url" in attachment:
            parts.append(("text", f"用户发来的图片：{attachment['name']}"))
            parts.append(("image", attachment["data_url"]))
    return parts


def _longest_backtick_run(text: str) -> int:
    return max((len(run) for run in re.findall(r"`+", text)), default=0)


def get_environment_images(message: ChatMessage) -> tuple[ChatImageAttachment, ...]:
    images = message.get("environment_images")
    if images:
        return images
    image = message.get("environment_image")
    return () if image is None else (image,)


def environment_image_caption(image: ChatImageAttachment) -> str:
    """Label automatic environment attachments at either provider's wire boundary."""
    description = image.get("description")
    if not description:
        description = ENVIRONMENT_IMAGE_DEFAULT_DESCRIPTION.format(
            observed_at=image["observed_at"]
        )
    return ENVIRONMENT_IMAGE_CAPTION_PREFIX + description


def add_perception_policy(messages: list[ChatMessage], description: str) -> None:
    """Share the same current capability description with every response lane."""
    if description:
        messages[0]["content"] += "\n\n" + description
        messages[0]["perception_context"] = description


def build_chat_messages(
    system_prompt: str,
    history: list[Message],
    *,
    static_profile: str = "",
    dynamic_snapshot: str = "",
    preserve_order: bool = False,
    current_input: str | None = None,
    additional_history: Sequence[tuple[datetime | None, ChatMessage]] = (),
    history_text: str | None = None,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
    ordered_history = sorted(history, key=lambda item: item.sequence)
    current_user = next(
        (message for message in reversed(ordered_history) if message.role is Role.USER),
        None,
    )
    if preserve_order or current_input is not None:
        current_user = None
    recent = [
        message
        for message in ordered_history
        if message is not current_user and visible_message_content(message)
    ][-DEFAULT_RECENT_CONTEXT_LIMIT:]

    # Keep summaries before native-role history so they cannot break the immediate
    # conversational link between the latest assistant turn and the current user turn.
    memory_sections = [static_profile] if static_profile else []
    if memory_sections:
        messages.append(
            {
                "role": "user",
                "context_reference": True,
                "content": (
                    "# 记忆摘要\n"
                    "以下内容是压缩后的内部参考信息，不是原始对话。\n"
                    "它可用于理解长期属性、当前状态和近期主题；"
                    "具体过往事件的细节仍需 Recall 验证。\n"
                    "请像自然了解这些信息一样直接回答，不要提及记录、记忆、历史、系统、"
                    "参考资料、上下文、召回或检索，也不要说“根据我的记录”“我查到”或“我翻到”。\n"
                    "当前角色配置优先于参考信息中的旧身份、旧名称或旧设定。\n\n"
                    + "\n\n".join(memory_sections)
                ),
            }
        )

    if history_text is not None:
        messages[0]["recent_spoken"] = tuple(
            visible_message_content(message)
            for message in recent
            if message.role is Role.ASSISTANT
        )
        messages[0]["content"] += "\n\n" + HISTORY_SOURCE_RULES
        if history_text:
            messages.append(
                {
                    "role": "user",
                    "content": "History:\n" + history_text,
                    "context_reference": True,
                }
            )
        if current_user is not None:
            messages.append(
                {
                    "role": "user",
                    "content": _timestamped_content(current_user, current_user.content),
                }
            )
        if current_input is not None:
            messages.append({"role": "user", "content": current_input})
        return messages

    # Keep the ordinary chat window intact. External history has its own bounded
    # window and is interleaved only after selecting the native messages.
    timeline = list(additional_history)
    for message in recent:
        content = (
            message.content
            if message.role is Role.USER
            else _visible_assistant_content(message)
        )
        if not content:
            continue
        timeline.append(
            (
                message.created_at,
                {
                    "role": "user" if message.role is Role.USER else "assistant",
                    "content": content,
                },
            )
        )
    if additional_history:
        # Stable ties keep received events before their responses. Missing event
        # timestamps stay at the beginning without inventing a displayed date.
        timeline.sort(
            key=lambda item: (
                item[0].astimezone(UTC).replace(microsecond=0)
                if item[0] is not None
                else datetime.min.replace(tzinfo=UTC)
            )
        )
    current_time_bucket: datetime | None = None
    for created_at, historical in timeline:
        content = historical["content"]
        if created_at is not None:
            message_bucket = _history_time_bucket(created_at)
            if current_time_bucket != message_bucket:
                current_time_bucket = message_bucket
                content = f"{_format_history_time_range(message_bucket)}\n{content}"
        messages.append({**historical, "content": content})

    if current_user is not None:
        messages.append(
            {
                "role": "user",
                "content": _timestamped_content(current_user, current_user.content),
            }
        )
    if current_input is not None:
        messages.append({"role": "user", "content": current_input})

    return messages


def add_perception_context(
    messages: list[ChatMessage],
    *,
    summary_context: str = "",
    observation_context: str = "",
    image: ChatImageAttachment | None = None,
    images: tuple[ChatImageAttachment, ...] = (),
) -> None:
    """Attach published perception identically for chat and external triggers."""
    if observation_context or summary_context or image or images:
        messages[0]["content"] += PERCEPTION_CONTEXT_SYSTEM_RULE
    for text in (observation_context.lstrip(), summary_context):
        if text:
            messages.insert(
                len(messages) - 1,
                {
                    "role": "user",
                    "content": text,
                    "context_reference": True,
                },
            )
    if image is not None:
        messages[-1]["environment_image"] = image
    if images:
        messages[-1]["environment_images"] = images


def add_initial_answer_contract(
    messages: list[ChatMessage],
    *,
    enable_web_search: bool = False,
) -> list[ChatMessage]:
    if not messages or messages[0]["role"] != "system":
        raise ValueError("chat messages must start with a system message")
    if sum(message["role"] == "system" for message in messages) != 1:
        raise ValueError("chat messages must contain exactly one system message")
    if messages[-1]["role"] != "user":
        raise ValueError("chat messages must end with the current user message")
    result: list[ChatMessage] = [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]
    contract = (
        SEARCH_AWARE_INITIAL_ANSWER_CONTRACT
        if enable_web_search
        else INITIAL_ANSWER_CONTRACT
    )
    turn_source_reminder = (
        _SEARCH_AWARE_TURN_SOURCE_REMINDER
        if enable_web_search
        else _TURN_RECALL_GATE_REMINDER
    )
    turn_source_reminder = (
        f"{turn_source_reminder}\n\n{ANSWER_LANGUAGE_TURN_REMINDER}"
        f"\n\n{TURN_RECALL_OUTPUT_CONSISTENCY_CHECK}"
    )
    result[0] = {
        "role": "system",
        "content": f"{contract}\n\n{result[0]['content']}",
    }
    current_user_content = result[-1]["content"]
    current_user_payload = json.dumps(
        {"content": current_user_content},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    result[-1] = {
        "role": "user",
        "content": (
            "# 当前用户消息（应用已进行 JSON 转义）\n"
            f"{current_user_payload}\n\n"
            f"{turn_source_reminder}"
        ),
    }
    return with_request_time_last(result, source=messages)


def build_web_search_context_messages(
    messages: list[ChatMessage],
    *,
    question: str | None = None,
) -> list[ChatMessage]:
    result = list(messages)
    current_question = latest_user_content(messages) if question is None else question
    result.append(
        {
            "role": "user",
            "content": (
                "# 外部信息提取协议\n"
                "当前专用调用已经开启并强制执行联网搜索。\n"
                "当前问题（以此确定搜索主题）：\n"
                f"{current_question}\n\n"
                "请根据本次搜索到的外部信息，提炼回答当前问题所需的最少事实。\n"
                "searchContext 只写可靠的外部事实，不要回答用户，"
                "不要提及搜索过程或内部实现；"
                "只有确实没有可靠搜索结果时才返回空字符串。\n"
                "只返回 JSON 对象，不要输出其他文本："
                '{"searchContext":"需要交给下一次回答调用的简短外部事实"}'
            ),
        }
    )
    return with_request_time_last(result)


def parse_search_context(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exception:
        raise ValueError("web search context must be a JSON object") from exception
    if not isinstance(parsed, dict) or set(parsed) != {"searchContext"}:
        raise ValueError("web search context must contain only searchContext")
    search_context = parsed["searchContext"]
    if not isinstance(search_context, str):
        raise ValueError("web search context must be a string")
    return search_context


def latest_user_content(messages: Sequence[ChatMessage]) -> str:
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"]
        if content.startswith("<message_time>"):
            _, separator, remainder = content.partition("\n")
            if separator and content.split("\n", 1)[0].endswith("</message_time>"):
                return remainder
        return content
    raise ValueError("chat messages do not contain a user message")


def _visible_assistant_content(message: Message) -> str:
    offset = message.consumed_offset
    visible = visible_message_content(message)
    if not visible:
        return ""

    was_interrupted = message.status is MessageStatus.INTERRUPTED or (
        offset is not None and offset >= 0 and offset < len(message.content)
    )
    if was_interrupted:
        return f"{visible}\n{INTERRUPTION_MARKER}"
    return visible


def _timestamped_content(message: Message, content: str) -> str:
    timestamp = format_prompt_timestamp(message.created_at, sep=" ")
    return f"<message_time>{timestamp}</message_time>\n{content}"


def _history_time_bucket(created_at: datetime) -> datetime:
    timestamp = created_at.astimezone(UTC)
    return timestamp.replace(
        minute=timestamp.minute - timestamp.minute % 10,
        second=0,
        microsecond=0,
    )


def _format_history_time_range(bucket_start: datetime) -> str:
    bucket_end = bucket_start + _HISTORY_TIME_BUCKET - timedelta(seconds=1)
    start = format_prompt_timestamp(bucket_start, sep=" ")
    end = format_prompt_timestamp(bucket_end, sep=" ")
    return f"<message_time_range>{start} ~ {end}</message_time_range>"
