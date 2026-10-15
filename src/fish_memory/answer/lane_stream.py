from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fish_memory.agent.models import AgentTaskLaneDecision
from fish_memory.agent.routing import AgentTaskDecisionKind
from fish_memory.home.models import HomeIntentType
from fish_memory.pet.types import PET_EMOTIONS, PetEmotion

# Longest unknown word tolerated before falling back; longer means a broken stream.
_MAX_ENUM_LENGTH = 32


class BlankFieldError(ValueError):
    """A field the reply must fill came back empty."""

    def __init__(self, key: str):
        super().__init__(f"{key} must not be blank")
        self.key = key


@dataclass(frozen=True, slots=True)
class AnswerLaneDelta:
    accepted: bool | None = None
    home_intent: HomeIntentType | None = None
    emotion: PetEmotion | None = None
    answer: str = ""
    answer_complete: bool = False
    recall_depth: int | None = None
    query: str | None = None
    end: bool | None = None
    task_kind: AgentTaskDecisionKind | None = None
    candidate_indexes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class _Field:
    key: str
    kind: Literal[
        "decision",
        "depth",
        "emotion",
        "home_intent",
        "answer",
        "query",
        "end",
        "task_decision",
        "candidate_index",
    ]


class _CompactLaneParser:
    def __init__(self, *, stop_after_object: bool = False) -> None:
        self._stop_after_object = stop_after_object
        self._buffer = ""
        self._cursor = 0
        self._state = "start"
        self._current: _Field | None = None
        self._values: dict[str, object] = {}
        self._string_parts: list[str] = []

    @property
    def raw_response(self) -> str:
        return self._buffer

    def feed_values(self, chunk: str) -> tuple[dict[str, object], str]:
        self._buffer += chunk
        emitted: dict[str, object] = {}
        emitted_answer: list[str] = []
        while True:
            if self._state == "start":
                if not self._consume_expected("{"):
                    break
                self._current = self._next_field()
                if self._current is None:
                    raise RuntimeError("lane schema has no first field")
                self._state = "key"
                continue
            if self._state == "key":
                assert self._current is not None
                if not self._consume_literal(f'"{self._current.key}"'):
                    break
                self._state = "colon"
                continue
            if self._state == "colon":
                if not self._consume_expected(":"):
                    break
                assert self._current is not None
                self._state = (
                    "string_open"
                    if self._current.kind in {"answer", "query"}
                    else "scalar"
                )
                continue
            if self._state == "scalar":
                assert self._current is not None
                value = self._consume_scalar(self._current)
                if value is None:
                    break
                self._complete_field(value, emitted)
                continue
            if self._state == "string_open":
                if not self._consume_expected('"'):
                    break
                self._string_parts = []
                self._state = "string"
                continue
            if self._state == "string":
                decoded, complete = self._consume_string()
                if self._current is not None and self._current.kind == "answer":
                    emitted_answer.extend(decoded)
                if not complete:
                    break
                value = "".join(self._string_parts)
                assert self._current is not None
                if not value.strip() and not self._allow_blank_string(self._current):
                    raise BlankFieldError(self._current.key)
                self._complete_field(value, emitted)
                continue
            if self._state == "separator":
                next_field = self._next_field()
                if next_field is None:
                    if not self._consume_expected("}"):
                        break
                    self._state = "done"
                    continue
                if self._allows_object_close_before_next_field():
                    self._consume_whitespace()
                    if (
                        self._cursor < len(self._buffer)
                        and self._buffer[self._cursor] == "}"
                    ):
                        self._cursor += 1
                        self._apply_object_close_defaults(emitted)
                        self._state = "done"
                        continue
                if not self._consume_expected(","):
                    break
                self._current = next_field
                self._state = "key"
                continue
            if self._state == "done":
                if self._stop_after_object:
                    break
                self._consume_whitespace()
                if self._cursor != len(self._buffer):
                    raise ValueError("lane response has trailing data")
                break
            raise RuntimeError(f"unknown lane parser state: {self._state}")
        return emitted, "".join(emitted_answer)

    def finish(self) -> None:
        self.feed_values("")
        if self._state != "done":
            raise ValueError("lane response JSON is incomplete")

    def _next_field(self) -> _Field | None:
        raise NotImplementedError

    def _allow_blank_string(self, field: _Field) -> bool:
        del field
        return False

    def _allows_object_close_before_next_field(self) -> bool:
        return False

    def _apply_object_close_defaults(self, emitted: dict[str, object]) -> None:
        del emitted

    def _complete_field(
        self,
        value: object,
        emitted: dict[str, object],
    ) -> None:
        assert self._current is not None
        self._values[self._current.key] = value
        emitted[self._current.key] = value
        self._state = "separator"

    def _consume_scalar(self, field: _Field) -> object | None:
        if field.kind == "emotion":
            # An emotion outside the list only affects the face: fall back to
            # neutral instead of failing the whole answer.
            value = self._consume_enum(
                PET_EMOTIONS, field.key, fallback=PetEmotion.NEUTRAL.value
            )
            return None if value is None else PetEmotion(value)
        if field.kind == "home_intent":
            value = self._consume_enum(
                frozenset(item.value for item in HomeIntentType),
                field.key,
            )
            return None if value is None else HomeIntentType(value)
        if field.kind == "decision":
            value = self._consume_digit({"0", "1"}, field.key)
            return None if value is None else value == 1
        if field.kind == "depth":
            return self._consume_digit({"0", "1"}, field.key)
        if field.kind == "end":
            value = self._consume_digit({"0", "1"}, field.key)
            return None if value is None else value == 1
        if field.kind == "task_decision":
            value = self._consume_digit({"0", "1"}, field.key)
            return None if value is None else AgentTaskLaneDecision(value)
        if field.kind == "candidate_index":
            return self._consume_digit(set("0123456789"), field.key)
        raise RuntimeError(f"unsupported scalar kind: {field.kind}")

    def _consume_whitespace(self) -> None:
        while (
            self._cursor < len(self._buffer) and self._buffer[self._cursor] in " \t\r\n"
        ):
            self._cursor += 1

    def _consume_expected(self, expected: str) -> bool:
        self._consume_whitespace()
        if self._cursor >= len(self._buffer):
            return False
        if self._buffer[self._cursor] != expected:
            raise ValueError(f"expected {expected!r} in lane response")
        self._cursor += 1
        return True

    def _consume_literal(self, literal: str) -> bool:
        self._consume_whitespace()
        remaining = self._buffer[self._cursor :]
        if len(remaining) < len(literal):
            if not literal.startswith(remaining):
                raise ValueError(f"expected {literal!r} in lane response")
            return False
        if not remaining.startswith(literal):
            raise ValueError(f"expected {literal!r} in lane response")
        self._cursor += len(literal)
        return True

    def _consume_enum(
        self, allowed: frozenset[str], field: str, *, fallback: str | None = None
    ) -> str | None:
        self._consume_whitespace()
        remaining = self._buffer[self._cursor :]
        if not remaining:
            return None
        if not remaining.startswith('"'):
            raise ValueError(f"{field} must be a JSON string")
        closing = remaining.find('"', 1)
        if closing < 0:
            partial = remaining[1:]
            if fallback is not None and len(partial) <= _MAX_ENUM_LENGTH:
                return None
            if not any(value.startswith(partial) for value in allowed):
                raise ValueError(f"{field} must be one of {sorted(allowed)}")
            return None
        value = remaining[1:closing]
        if value not in allowed:
            if fallback is None or len(value) > _MAX_ENUM_LENGTH:
                raise ValueError(f"{field} must be one of {sorted(allowed)}")
            value = fallback
        self._cursor += closing + 1
        return value

    def _consume_digit(self, allowed: set[str], field: str) -> int | None:
        self._consume_whitespace()
        if self._cursor >= len(self._buffer):
            return None
        value = self._buffer[self._cursor]
        if value not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}")
        self._cursor += 1
        return int(value)

    def _consume_string(self) -> tuple[list[str], bool]:
        decoded: list[str] = []
        while self._cursor < len(self._buffer):
            value = self._buffer[self._cursor]
            if value == '"':
                self._cursor += 1
                self._state = "separator"
                return decoded, True
            if value == "\\":
                escape = self._decode_escape()
                if escape is None:
                    return decoded, False
                decoded.append(escape)
                self._string_parts.append(escape)
                continue
            if ord(value) < 0x20 or 0xD800 <= ord(value) <= 0xDFFF:
                raise ValueError("lane string contains an invalid JSON character")
            self._cursor += 1
            decoded.append(value)
            self._string_parts.append(value)
        return decoded, False

    def _decode_escape(self) -> str | None:
        if self._cursor + 2 > len(self._buffer):
            return None
        escape = self._buffer[self._cursor + 1]
        simple = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        if escape in simple:
            self._cursor += 2
            return simple[escape]
        if escape != "u":
            raise ValueError("lane string contains an invalid JSON escape")
        if self._cursor + 6 > len(self._buffer):
            return None
        code = _hex_code(self._buffer[self._cursor + 2 : self._cursor + 6])
        if 0xDC00 <= code <= 0xDFFF:
            raise ValueError("lane string contains an unpaired low surrogate")
        if 0xD800 <= code <= 0xDBFF:
            if self._cursor + 12 > len(self._buffer):
                return None
            if self._buffer[self._cursor + 6 : self._cursor + 8] != "\\u":
                raise ValueError("lane string contains an unpaired high surrogate")
            low = _hex_code(self._buffer[self._cursor + 8 : self._cursor + 12])
            if not 0xDC00 <= low <= 0xDFFF:
                raise ValueError("lane string contains an invalid surrogate pair")
            self._cursor += 12
            return chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00)
        self._cursor += 6
        return chr(code)


class ForcedActionStreamParser(_CompactLaneParser):
    """Experimental selected action: wording and parameters, never another vote."""

    def __init__(self, *, query: bool):
        super().__init__()
        self._fields = (
            _Field("m", "emotion"),
            _Field("a", "answer"),
            *((_Field("q", "query"),) if query else ()),
            _Field("e", "end"),
        )

    def _next_field(self) -> _Field | None:
        return next(
            (field for field in self._fields if field.key not in self._values), None
        )

    def feed(self, chunk: str) -> AnswerLaneDelta:
        values, answer = self.feed_values(chunk)
        return AnswerLaneDelta(
            emotion=_emotion(values.get("m")),
            answer=answer,
            answer_complete="a" in values,
            query=_string(values.get("q")),
            end=_bool(values.get("e")),
        )


class SpeechExpressionStreamParser(_CompactLaneParser):
    """Speech already admitted by the host: emotion followed by playable text."""

    def _next_field(self) -> _Field | None:
        if "m" not in self._values:
            return _Field("m", "emotion")
        if "a" not in self._values:
            return _Field("a", "answer")
        return None

    def feed(self, chunk: str) -> AnswerLaneDelta:
        values, answer = self.feed_values(chunk)
        return AnswerLaneDelta(
            emotion=_emotion(values.get("m")),
            answer=answer,
            answer_complete="a" in values,
        )


class RecallLaneStreamParser(_CompactLaneParser):
    def _next_field(self) -> _Field | None:
        if "m" not in self._values:
            return _Field("m", "emotion")
        if "a" not in self._values:
            return _Field("a", "answer")
        if "r" not in self._values:
            return _Field("r", "depth")
        if self._values["r"] == 1 and "q" not in self._values:
            return _Field("q", "query")
        if "e" not in self._values:
            return _Field("e", "end")
        return None

    def _allows_object_close_before_next_field(self) -> bool:
        return (
            "r" in self._values
            and (self._values["r"] != 1 or "q" in self._values)
            and "e" not in self._values
        )

    def _apply_object_close_defaults(self, emitted: dict[str, object]) -> None:
        self._values["e"] = False
        emitted["e"] = False

    def feed(self, chunk: str) -> AnswerLaneDelta:
        values, answer = self.feed_values(chunk)
        return AnswerLaneDelta(
            emotion=_emotion(values.get("m")),
            answer=answer,
            answer_complete="a" in values,
            recall_depth=_int(values.get("r")),
            query=_string(values.get("q")),
            end=_bool(values.get("e")),
        )


class WebLaneStreamParser(_CompactLaneParser):
    def _next_field(self) -> _Field | None:
        if "w" not in self._values:
            return _Field("w", "decision")
        if self._values["w"] is False:
            return None
        if "m" not in self._values:
            return _Field("m", "emotion")
        if "a" not in self._values:
            return _Field("a", "answer")
        if "q" not in self._values:
            return _Field("q", "query")
        return None

    def feed(self, chunk: str) -> AnswerLaneDelta:
        values, answer = self.feed_values(chunk)
        return AnswerLaneDelta(
            accepted=_bool(values.get("w")),
            emotion=_emotion(values.get("m")),
            answer=answer,
            answer_complete="a" in values,
            query=_string(values.get("q")),
        )


class HomeLaneStreamParser(_CompactLaneParser):
    def _next_field(self) -> _Field | None:
        if "h" not in self._values:
            return _Field("h", "decision")
        if self._values["h"] is False:
            return None
        if "t" not in self._values:
            return _Field("t", "home_intent")
        if "m" not in self._values:
            return _Field("m", "emotion")
        if "a" not in self._values:
            return _Field("a", "answer")
        return None

    def feed(self, chunk: str) -> AnswerLaneDelta:
        values, answer = self.feed_values(chunk)
        return AnswerLaneDelta(
            accepted=_bool(values.get("h")),
            home_intent=_home_intent(values.get("t")),
            emotion=_emotion(values.get("m")),
            answer=answer,
            answer_complete="a" in values,
        )


class TaskLaneStreamParser(_CompactLaneParser):
    def __init__(self, *, candidate_count: int) -> None:
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 0
        ):
            raise ValueError("candidate_count must be a non-negative integer")
        super().__init__()
        self._candidate_count = candidate_count

    def _next_field(self) -> _Field | None:
        if "t" not in self._values:
            return _Field("t", "task_decision")
        if "m" not in self._values:
            return _Field("m", "emotion")
        if "a" not in self._values:
            return _Field("a", "answer")
        if self._values["t"] is AgentTaskLaneDecision.TASK and "c" not in self._values:
            return _Field("c", "candidate_index")
        return None

    def _allow_blank_string(self, field: _Field) -> bool:
        return (
            field.key == "a" and self._values.get("t") is AgentTaskLaneDecision.NOT_TASK
        )

    def _allows_object_close_before_next_field(self) -> bool:
        return (
            self._values.get("t") is AgentTaskLaneDecision.TASK
            and "a" in self._values
            and "c" not in self._values
        )

    def feed(self, chunk: str) -> AnswerLaneDelta:
        values, answer = self.feed_values(chunk)
        decision = self._values.get("t")
        if (
            decision is AgentTaskLaneDecision.TASK
            and "a" in values
            and _task_confirmation_is_question(self._values.get("a"))
        ):
            raise ValueError("Task lane confirmation must not ask a question")
        visible = decision is not AgentTaskLaneDecision.NOT_TASK
        # Route decisions may be re-emitted when a later chunk only closes the
        # JSON object. Keep the already parsed candidate in that delta so the
        # lane accumulator cannot replace a valid continuation with ().
        index = self._values.get("c")
        parsed_indexes = (
            (index,)
            if self._candidate_count > 0
            and isinstance(index, int)
            and not isinstance(index, bool)
            else ()
        )
        task_kind = self._task_kind()
        answer_complete = visible and "a" in values
        has_complete_answer = visible and "a" in self._values
        accepted = (
            False
            if decision is AgentTaskLaneDecision.NOT_TASK
            else (True if task_kind is not None and has_complete_answer else None)
        )
        return AnswerLaneDelta(
            accepted=accepted,
            emotion=_emotion(values.get("m")) if visible else None,
            answer=answer if visible else "",
            answer_complete=answer_complete,
            task_kind=task_kind,
            candidate_indexes=parsed_indexes,
        )

    def finish(self) -> None:
        super().finish()
        decision = self._values.get("t")
        if decision is AgentTaskLaneDecision.NOT_TASK:
            if self._values.get("a") != "":
                raise ValueError("not_task answer must be exactly empty")
            return
        if not isinstance(decision, AgentTaskLaneDecision):
            raise ValueError("Task lane is missing its decision")
        if _task_confirmation_is_question(self._values.get("a")):
            raise ValueError("Task lane confirmation must not ask a question")
        index = self._values.get("c")
        if index is None or self._candidate_count == 0:
            parsed_indexes: tuple[int, ...] = ()
        elif isinstance(index, int) and not isinstance(index, bool):
            parsed_indexes = (index,)
        else:
            raise ValueError("Task lane candidate index must be an integer")
        if parsed_indexes and parsed_indexes[0] >= self._candidate_count:
            raise ValueError("Task lane candidate index is outside supplied candidates")
        AgentTaskDecisionKind.from_candidate_index(
            parsed_indexes[0] if parsed_indexes else None
        )

    def _task_kind(self) -> AgentTaskDecisionKind | None:
        decision = self._values.get("t")
        if decision is not AgentTaskLaneDecision.TASK:
            return None
        index = self._values.get("c")
        if index is None:
            if self._state != "done":
                return None
            return AgentTaskDecisionKind.NEW_TASK
        if self._candidate_count == 0:
            return AgentTaskDecisionKind.NEW_TASK
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Task lane candidate index must be an integer")
        return AgentTaskDecisionKind.from_candidate_index(index)


def _emotion(value: object | None) -> PetEmotion | None:
    return value if isinstance(value, PetEmotion) else None


def _task_confirmation_is_question(value: object | None) -> bool:
    return isinstance(value, str) and value.rstrip().endswith(("?", "？"))


def _home_intent(value: object | None) -> HomeIntentType | None:
    return value if isinstance(value, HomeIntentType) else None


def _int(value: object | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bool(value: object | None) -> bool | None:
    return value if isinstance(value, bool) else None


def _string(value: object | None) -> str | None:
    return value if isinstance(value, str) else None


def _hex_code(value: str) -> int:
    if len(value) != 4 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError("lane string contains an invalid Unicode escape")
    return int(value, 16)
