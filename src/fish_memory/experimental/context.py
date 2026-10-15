"""One role-aware context for models; Jev renders the same sources as flat State."""

import re
from collections.abc import Sequence
from datetime import datetime

from fish_memory.answer.context import INTERRUPTION_MARKER, ChatMessage
from fish_memory.clock import format_prompt_timestamp

# Stored with every recorded thought; the label shown to models omits the
# "not yet expressed" wording, which made old thoughts read as pending tasks.
THOUGHT_PREFIX = "宠物的内心想法（尚未表达）："
_THOUGHT_EVENT = "event: " + THOUGHT_PREFIX
_THOUGHT_SHOWN = "event: 内心想法："


_UNSPACED = re.compile(r"[\s，。？！、,.!?“”‘’…：:；;（）()]")


def trigrams(text: str) -> frozenset[str]:
    text = _UNSPACED.sub("", text)
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))


def context_content(message: ChatMessage) -> str:
    """Normalize experimental text without the loader's injected system clock."""
    content = message["content"]
    if message["role"] == "system" and (
        timestamp := message.get("system_request_time")
    ):
        content = content.replace("\n" + timestamp, "", 1)
    return content.replace("\n  " + INTERRUPTION_MARKER, "(被打断)").replace(
        "\n" + INTERRUPTION_MARKER, "(被打断)"
    )


def model_messages(rules: str, context: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Keep shared persona/environment in system once, then add task rules."""
    system = [message["content"] for message in context if message["role"] == "system"]
    return [
        {"role": "system", "content": "\n\n".join([*system, rules])},
        *(message.copy() for message in context if message["role"] != "system"),
    ]


def _is_history(message: ChatMessage) -> bool:
    return bool(message.get("context_reference")) and message["content"].startswith(
        "History:\n"
    )


def _history_items(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    return [
        item
        for message in messages
        for item in message.get("history_messages", (message,))
        if _is_history(item)
    ]


def recorded_thoughts(messages: Sequence[ChatMessage]) -> list[str]:
    """Thoughts recorded in this History, oldest first."""
    return [
        line.removeprefix(_THOUGHT_EVENT).strip()
        for item in _history_items(messages)
        for line in item["content"].splitlines()
        if line.startswith(_THOUGHT_EVENT)
    ]


# Speaking and deciding see the current state of mind, not every past thought.
SPEECH_THOUGHT_CONTEXT = 3


def with_recent_thoughts(
    messages: Sequence[ChatMessage], keep: int, *, relabel: bool = True
) -> list[ChatMessage]:
    """Keep only the latest thoughts, in place, so they read as a line of thought.

    Every older thought is dropped from History; the kept ones stay at their
    original position between the events that prompted them.
    """
    remaining = len(recorded_thoughts(messages))

    def prune(item: ChatMessage) -> ChatMessage:
        nonlocal remaining
        if not _is_history(item):
            return item
        lines = []
        for line in item["content"].splitlines():
            if line.startswith(_THOUGHT_EVENT):
                remaining -= 1
                if remaining >= keep:
                    continue
                if relabel:
                    line = _THOUGHT_SHOWN + line.removeprefix(_THOUGHT_EVENT)
            lines.append(line)
        return {**item, "content": "\n".join(lines)}

    result: list[ChatMessage] = []
    for message in messages:
        if "history_messages" in message:
            items = tuple(prune(item) for item in message["history_messages"])
            result.append(prune({**message, "history_messages": items}))
        else:
            result.append(prune(message))
    return result


def new_since_last_thought(messages: Sequence[ChatMessage]) -> bool:
    """Whether anything besides thoughts happened after the latest thought."""
    seen_thought, new = False, False
    for message in messages:
        for item in message.get("history_messages", (message,)):
            if item["role"] == "system":
                continue
            if not _is_history(item):
                new = seen_thought
                continue
            for line in item["content"].splitlines()[1:]:
                if line.startswith(_THOUGHT_EVENT):
                    seen_thought, new = True, False
                elif line.strip() and not _is_time_range(line):
                    new = seen_thought
    return new if seen_thought else True


def _is_time_range(line: str) -> bool:
    return len(line) > 20 and line[4] == "-" and " - " in line and line[:4].isdigit()


def expression_context(
    messages: Sequence[ChatMessage], now: datetime, trigger: str
) -> list[ChatMessage]:
    result: list[ChatMessage] = []
    attached = next(
        (item for m in reversed(messages) if (item := m.get("user_attachments"))),
        (),
    )
    for message in messages:
        for item in message.get("history_messages", (message,)):
            copied: ChatMessage = {
                "role": item["role"],
                "content": context_content(item),
            }
            if item.get("context_reference"):
                copied["context_reference"] = True
            if item.get("speech_tags"):
                copied["speech_tags"] = True
            result.append(copied)
    trigger_message: ChatMessage = {
        "role": "user",
        "content": f"本轮触发：{trigger}",
        "request_time": f"当前时间：{format_prompt_timestamp(now)}",
    }
    if attached:
        # What the user sent goes with the turn that answers it, where the
        # screen goes too; History above only names it.
        trigger_message["user_attachments"] = attached
    result.append(trigger_message)
    return result


def timeline(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
    """History lines and native turns, oldest first, as (label, text)."""
    result: list[tuple[str, str]] = []
    for message in messages:
        for item in message.get("history_messages", (message,)):
            if item["role"] == "system":
                continue
            content = context_content(item)
            if _is_history(item):
                for line in content.splitlines()[1:]:
                    if line.startswith("  ") and result:
                        label, text = result[-1]
                        result[-1] = (label, text + "\n" + line.strip())
                    elif ": " in line and not _is_time_range(line):
                        label, _, text = line.partition(": ")
                        result.append((label, text))
            elif not item.get("context_reference"):
                if content.startswith("<message_time>") and "\n" in content:
                    content = content.split("\n", 1)[1]
                result.append(("u" if item["role"] == "user" else "a", content))
    return result


def recent_speech(messages: Sequence[ChatMessage], keep: int) -> list[str]:
    """What TA said most recently, oldest first."""
    spoken = [
        text.removesuffix("(被打断)")
        for label, text in timeline(messages)
        if label == "a"
    ]
    return spoken[-keep:] if keep > 0 else []


# Character-trigram overlap above which a line of TA's copies one said before.
SPEECH_COPY_OVERLAP = 0.5


def without_repeated_speech(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Keep only the first time TA said a line; later copies are left out.

    Seeing its own lines come back again and again, qwen said one of them once
    more word for word. On the real timed wakes where it did, leaving the later
    copies out cut copying from 11/24 to 4/24 (other wakes 5/24 to 1/24),
    where stronger decoding penalties began switching it to English.
    """
    said: list[frozenset[str]] = []

    def repeated(text: str) -> bool:
        grams = trigrams(text.removesuffix("(被打断)"))
        if grams and any(
            len(grams & other) > SPEECH_COPY_OVERLAP * len(grams | other)
            for other in said
        ):
            return True
        said.append(grams)
        return False

    def fold(item: ChatMessage) -> ChatMessage | None:
        if item["role"] == "assistant" and not item.get("context_reference"):
            return None if repeated(context_content(item)) else item
        if not _is_history(item):
            return item
        units: list[list[str]] = []
        for line in item["content"].splitlines():
            if line.startswith("  ") and units:
                units[-1].append(line)
            else:
                units.append([line])
        kept = [
            line
            for unit in units
            if not (
                unit[0].startswith("a: ")
                and repeated("\n".join([unit[0][3:], *unit[1:]]))
            )
            for line in unit
        ]
        return {**item, "content": "\n".join(kept)}

    result: list[ChatMessage] = []
    for message in messages:
        if "history_messages" in message:
            # Models read the split items; the joined text stays as it was.
            items = (fold(item) for item in message["history_messages"])
            result.append(
                {
                    **message,
                    "history_messages": tuple(i for i in items if i is not None),
                }
            )
        elif (folded := fold(message)) is not None:
            result.append(folded)
    return result
