"""Identity and state primitives shared by the voice-draft protocol."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID, uuid5

from fish_memory.answer.lane_context import CURRENT_USER_WRAPPER_RULES

VOICE_DRAFT_ANSWER_NAMESPACE = UUID("f85e118e-7524-5a69-8f71-87f0bbb4d37d")
_DYNAMIC_TIME_TAGS = (
    re.compile(r"\A(<message_time>).*?(</message_time>)", re.DOTALL),
    re.compile(
        r"\A(<message_time_range>).*?(</message_time_range>)",
        re.DOTALL,
    ),
)
_CURRENT_USER_WRAPPER_PREFIX = "# 当前用户消息（应用已进行 JSON 转义）\n"


class VoiceDraftState(StrEnum):
    ACTIVE = "active"
    CANCELLED = "cancelled"
    COMMITTED = "committed"
    EXPIRED = "expired"


def exact_content_hash(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def semantic_initial_input_hash(payloads: object) -> str:
    """Hash lane payloads after removing only generated time-tag contents."""
    normalized = _normalize_dynamic_time_tags(payloads)
    return exact_content_hash(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _normalize_dynamic_time_tags(value: object) -> object:
    if isinstance(value, str):
        return _normalize_current_user_wrapper(_normalize_generated_time_prefix(value))
    if isinstance(value, Mapping):
        return {key: _normalize_dynamic_time_tags(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_normalize_dynamic_time_tags(item) for item in value]
    return value


def _normalize_generated_time_prefix(value: str) -> str:
    normalized = value
    for pattern in _DYNAMIC_TIME_TAGS:
        normalized = pattern.sub(r"\1<dynamic>\2", normalized)
    return normalized


def _normalize_current_user_wrapper(value: str) -> str:
    if not value.startswith(_CURRENT_USER_WRAPPER_PREFIX):
        return value
    json_line, separator, rules = value.removeprefix(
        _CURRENT_USER_WRAPPER_PREFIX
    ).partition("\n\n")
    if not separator or "\n" in json_line or rules not in CURRENT_USER_WRAPPER_RULES:
        return value
    try:
        payload = json.loads(json_line)
    except ValueError:
        return value
    if (
        not isinstance(payload, dict)
        or set(payload) != {"content"}
        or not isinstance(payload.get("content"), str)
        or json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        != json_line
    ):
        return value
    content = payload["content"]
    normalized_content = _normalize_generated_time_prefix(content)
    if normalized_content == content:
        return value
    normalized_payload = json.dumps(
        {"content": normalized_content},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"{_CURRENT_USER_WRAPPER_PREFIX}{normalized_payload}\n\n{rules}"


def voice_answer_message_id(turn_id: UUID) -> UUID:
    return uuid5(VOICE_DRAFT_ANSWER_NAMESPACE, str(turn_id))


@dataclass(frozen=True, slots=True)
class VoiceDraftKey:
    fish_id: UUID
    session_id: UUID
    turn_id: UUID
