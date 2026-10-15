from dataclasses import dataclass

from fish_memory.pet.types import PET_EMOTIONS, PetEmotion


@dataclass(frozen=True, slots=True)
class StructuredAnswerDelta:
    recall_depth: int | None = None
    need_web_search: bool | None = None
    emotion: PetEmotion | None = None
    answer: str = ""
    end: bool | None = None


class StructuredAnswerStreamParser:
    """Incrementally parses the compact, fixed-order initial answer protocol."""

    def __init__(self, *, enable_web_search: bool) -> None:
        self._buffer = ""
        self._cursor = 0
        self._state = "start"
        self._answer_parts: list[str] = []
        self._enable_web_search = enable_web_search

    @property
    def raw_response(self) -> str:
        return self._buffer

    def feed(self, chunk: str) -> StructuredAnswerDelta:
        self._buffer += chunk
        emitted_depth: int | None = None
        emitted_search: bool | None = None
        emitted_emotion: PetEmotion | None = None
        emitted_answer: list[str] = []
        emitted_end: bool | None = None

        while True:
            if self._state == "start":
                if not self._consume_expected("{"):
                    break
                self._state = "recall_key"
                continue
            if self._state == "recall_key":
                if not self._consume_literal('"r"'):
                    break
                self._state = "recall_colon"
                continue
            if self._state == "recall_colon":
                if not self._consume_expected(":"):
                    break
                self._state = "recall_value"
                continue
            if self._state == "recall_value":
                value = self._consume_digit({"0", "1"}, "r")
                if value is None:
                    break
                emitted_depth = value
                self._state = "recall_comma"
                continue
            if self._state == "recall_comma":
                if not self._consume_expected(","):
                    break
                self._state = (
                    "search_key"
                    if self._enable_web_search
                    else "metadata_or_answer_key"
                )
                continue
            if self._state == "search_key":
                if not self._consume_literal('"w"'):
                    break
                self._state = "search_colon"
                continue
            if self._state == "search_colon":
                if not self._consume_expected(":"):
                    break
                self._state = "search_value"
                continue
            if self._state == "search_value":
                value = self._consume_digit({"0", "1"}, "w")
                if value is None:
                    break
                emitted_search = value == 1
                self._state = "search_comma"
                continue
            if self._state == "search_comma":
                if not self._consume_expected(","):
                    break
                self._state = "metadata_or_answer_key"
                continue
            if self._state == "metadata_or_answer_key":
                key = self._consume_one_of_literals({'"m"', '"a"'})
                if key is None:
                    break
                self._state = "emotion_colon" if key == '"m"' else "answer_colon"
                continue
            if self._state == "emotion_colon":
                if not self._consume_expected(":"):
                    break
                self._state = "emotion_value"
                continue
            if self._state == "emotion_value":
                value = self._consume_enum(PET_EMOTIONS, "m")
                if value is None:
                    break
                emitted_emotion = PetEmotion(value)
                self._state = "emotion_comma"
                continue
            if self._state == "emotion_comma":
                if not self._consume_expected(","):
                    break
                self._state = "answer_key"
                continue
            if self._state == "answer_key":
                if not self._consume_literal('"a"'):
                    break
                self._state = "answer_colon"
                continue
            if self._state == "answer_colon":
                if not self._consume_expected(":"):
                    break
                self._state = "answer_open"
                continue
            if self._state == "answer_open":
                if not self._consume_expected('"'):
                    break
                self._state = "answer"
                continue
            if self._state == "answer":
                emitted_answer.extend(self._consume_answer())
                if self._state == "answer":
                    break
                continue
            if self._state == "answer_comma":
                separator = self._consume_one_of_literals({",", "}"})
                if separator is None:
                    break
                # Some JSON-mode providers occasionally omit the final `e`
                # field even though the answer itself is complete. Treat that
                # valid, closed object as e=0 instead of failing after visible
                # text has already reached ASR/TTS consumers.
                if separator == "}":
                    emitted_end = False
                    self._state = "done"
                else:
                    self._state = "end_key"
                continue
            if self._state == "end_key":
                if not self._consume_literal('"e"'):
                    break
                self._state = "end_colon"
                continue
            if self._state == "end_colon":
                if not self._consume_expected(":"):
                    break
                self._state = "end_value"
                continue
            if self._state == "end_value":
                value = self._consume_digit({"0", "1"}, "e")
                if value is None:
                    break
                emitted_end = value == 1
                self._state = "close"
                continue
            if self._state == "close":
                if not self._consume_expected("}"):
                    break
                self._state = "done"
                continue
            if self._state == "done":
                self._consume_whitespace()
                if self._cursor != len(self._buffer):
                    raise ValueError("structured answer has trailing data")
                break
            raise RuntimeError(f"unknown parser state: {self._state}")

        return StructuredAnswerDelta(
            recall_depth=emitted_depth,
            need_web_search=emitted_search,
            emotion=emitted_emotion,
            answer="".join(emitted_answer),
            end=emitted_end,
        )

    def finish(self) -> None:
        self.feed("")
        if self._state != "done":
            raise ValueError("structured answer JSON is incomplete")

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
            raise ValueError(f"expected {expected!r} in structured answer")
        self._cursor += 1
        return True

    def _consume_literal(self, literal: str) -> bool:
        self._consume_whitespace()
        remaining = self._buffer[self._cursor :]
        if len(remaining) < len(literal):
            if not literal.startswith(remaining):
                raise ValueError(f"expected {literal!r} in structured answer")
            return False
        if not remaining.startswith(literal):
            raise ValueError(f"expected {literal!r} in structured answer")
        self._cursor += len(literal)
        return True

    def _consume_one_of_literals(self, literals: set[str]) -> str | None:
        self._consume_whitespace()
        remaining = self._buffer[self._cursor :]
        for literal in literals:
            if remaining.startswith(literal):
                self._cursor += len(literal)
                return literal
        if any(literal.startswith(remaining) for literal in literals):
            return None
        raise ValueError(f"expected one of {sorted(literals)!r} in structured answer")

    def _consume_enum(self, allowed: frozenset[str], field: str) -> str | None:
        self._consume_whitespace()
        remaining = self._buffer[self._cursor :]
        if not remaining:
            return None
        if not remaining.startswith('"'):
            raise ValueError(f"{field} must be a JSON string")
        closing = remaining.find('"', 1)
        if closing < 0:
            partial = remaining[1:]
            if not any(value.startswith(partial) for value in allowed):
                raise ValueError(f"{field} must be one of {sorted(allowed)}")
            return None
        value = remaining[1:closing]
        if value not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}")
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

    def _consume_answer(self) -> list[str]:
        decoded: list[str] = []
        while self._cursor < len(self._buffer):
            value = self._buffer[self._cursor]
            if value == '"':
                self._cursor += 1
                if not "".join(self._answer_parts).strip():
                    raise ValueError("answer must not be blank")
                self._state = "answer_comma"
                return decoded
            if value == "\\":
                escape = self._decode_escape()
                if escape is None:
                    return decoded
                decoded.append(escape)
                self._answer_parts.append(escape)
                continue
            if ord(value) < 0x20 or 0xD800 <= ord(value) <= 0xDFFF:
                raise ValueError("answer contains an invalid JSON character")
            self._cursor += 1
            decoded.append(value)
            self._answer_parts.append(value)
        return decoded

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
            raise ValueError("answer contains an invalid JSON escape")
        if self._cursor + 6 > len(self._buffer):
            return None
        code = _hex_code(self._buffer[self._cursor + 2 : self._cursor + 6])
        if 0xDC00 <= code <= 0xDFFF:
            raise ValueError("answer contains an unpaired low surrogate")
        if 0xD800 <= code <= 0xDBFF:
            if self._cursor + 12 > len(self._buffer):
                return None
            if self._buffer[self._cursor + 6 : self._cursor + 8] != "\\u":
                raise ValueError("answer contains an unpaired high surrogate")
            low = _hex_code(self._buffer[self._cursor + 8 : self._cursor + 12])
            if not 0xDC00 <= low <= 0xDFFF:
                raise ValueError("answer contains an invalid surrogate pair")
            self._cursor += 12
            return chr(0x10000 + ((code - 0xD800) << 10) + low - 0xDC00)
        self._cursor += 6
        return chr(code)


def _hex_code(value: str) -> int:
    if len(value) != 4 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError("answer contains an invalid Unicode escape")
    return int(value, 16)
