"""Strict, bounded terminal outcomes for external Agent runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

_MAX_OUTCOME_MESSAGE_CHARS = 32_000
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[^\s,;\"']+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)"
    r"((?:api[_-]?key|token|secret|password)\s*[\"']?\s*[=:]\s*)"
    r'(?:"(?:\\.|[^"\\\r\n])*"|'
    r"'(?:\\.|[^'\\\r\n])*'|"
    r"[^\s,;}\"']+)"
)
_EMPTY_FALLBACK = "The Agent run completed without a structured result."

AGENT_RUN_OUTCOME_INSTRUCTIONS = (
    "Return the final user-visible outcome as exactly one JSON object with no "
    "additional fields or surrounding prose. Use "
    '{"disposition":"completed","message":"task result or summary"} when '
    "the task is complete, or "
    '{"disposition":"input_required","message":"question the user must answer"} '
    "when more user input is required. The message must be non-empty and no "
    f"longer than {_MAX_OUTCOME_MESSAGE_CHARS} characters."
)


class AgentOutcomeDisposition(StrEnum):
    COMPLETED = "completed"
    INPUT_REQUIRED = "input_required"


@dataclass(frozen=True, slots=True)
class AgentRunOutcome:
    disposition: AgentOutcomeDisposition
    message: str


def parse_agent_run_outcome(value: str) -> AgentRunOutcome:
    """Parse the exact terminal envelope, rejecting ambiguous output."""

    try:
        payload = json.loads(value, object_pairs_hook=_strict_json_object)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Agent run outcome is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"disposition", "message"}:
        raise ValueError("Agent run outcome has unexpected fields")
    disposition_value = payload["disposition"]
    message = payload["message"]
    if not isinstance(disposition_value, str):
        raise ValueError("Agent run disposition must be a string")
    try:
        disposition = AgentOutcomeDisposition(disposition_value)
    except ValueError as error:
        raise ValueError("Agent run disposition is unsupported") from error
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Agent run message must be non-empty")
    if len(message) > _MAX_OUTCOME_MESSAGE_CHARS:
        raise ValueError("Agent run message exceeds the size limit")
    return AgentRunOutcome(disposition=disposition, message=message)


def normalize_agent_run_outcome(value: str | None) -> AgentRunOutcome:
    """Return a deliverable outcome without ever requesting a provider rerun."""

    raw = value if isinstance(value, str) else ""
    try:
        return parse_agent_run_outcome(raw)
    except ValueError:
        message = _redact_text(raw).strip()[:_MAX_OUTCOME_MESSAGE_CHARS]
        return AgentRunOutcome(
            disposition=AgentOutcomeDisposition.COMPLETED,
            message=message or _EMPTY_FALLBACK,
        )


def _redact_text(value: str) -> str:
    redacted = _BEARER_RE.sub(r"\1[REDACTED]", value)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", redacted)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Agent run outcome contains a duplicate field")
        value[key] = item
    return value


__all__ = [
    "AGENT_RUN_OUTCOME_INSTRUCTIONS",
    "AgentOutcomeDisposition",
    "AgentRunOutcome",
    "normalize_agent_run_outcome",
    "parse_agent_run_outcome",
]
