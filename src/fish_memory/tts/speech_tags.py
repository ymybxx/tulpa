"""Provider-specific speech markup and clean-character segment boundaries.

Only explicitly enabled speech uses this module. Ordinary text and non-label
brackets keep their original meaning and byte representation.
"""

import re
from dataclasses import dataclass

TAG_MODELS = frozenset(
    {
        "qwen-audio-3.0-tts-flash",
        "qwen-audio-3.0-tts-plus",
        "qwen-audio-3.1-tts-flash",
    }
)
CONTROL_TAGS = (
    "sad",
    "amazed",
    "deep and loud shouting",
    "trembling",
    "angry",
    "excited",
    "sarcastic",
    "curious",
    "like dracula",
    "bored",
    "tired",
    "scornful",
    "shouting",
    "asmr",
    "panicked",
    "mischievously",
    "empathetic",
    "whispers",
    "reluctantly",
    "crying",
    "serious",
    "very slowly",
    "very fast",
)
EFFECT_TAGS = (
    "gasp",
    "sighing",
    "clears throat",
    "giggles",
    "laughing",
    "cough",
    "snorts",
)
_TAGS = frozenset(f"[{tag}]" for tag in (*CONTROL_TAGS, *EFFECT_TAGS))
_CONTROLS = frozenset(f"[{tag}]" for tag in CONTROL_TAGS)
_EFFECTS = frozenset(f"[{tag}]" for tag in EFFECT_TAGS)
# In speech mode a single lowercase bracketed word is a voice label. An
# unsupported one must never turn into a spoken word or visible dialogue.
_UNSUPPORTED_LABEL = re.compile(r"\[[a-z]{3,32}\]")
_PENDING_LABEL = re.compile(r"\[[a-z]{0,32}$")
_STRONG = frozenset("。！？!?；;\n")
_SOFT = frozenset("，、,:：~～…")
_CLOSERS = frozenset("\"')]})）】》」』”’〉")


def supports_speech_tags(engine: str, model: str) -> bool:
    return engine == "dashscope" and model in TAG_MODELS


@dataclass(frozen=True)
class _Token:
    raw: str
    start: int
    end: int
    tag: bool = False
    unsupported: bool = False


def _tokens(text: str, *, final: bool = False) -> list[_Token]:
    result = []
    index = 0
    while index < len(text):
        if text[index] == "[":
            matched = next((tag for tag in _TAGS if text.startswith(tag, index)), None)
            if matched is not None:
                result.append(_Token(matched, index, index + len(matched), True))
                index += len(matched)
                continue
            unsupported = _UNSUPPORTED_LABEL.match(text, index)
            if unsupported is not None:
                result.append(
                    _Token(unsupported.group(), index, unsupported.end(), True, True)
                )
                index = unsupported.end()
                continue
            if not final and (
                any(tag.startswith(text[index:]) for tag in _TAGS)
                or _PENDING_LABEL.fullmatch(text[index:]) is not None
            ):
                break
        result.append(_Token(text[index], index, index + 1))
        index += 1
    return result


def strip_speech_tags(text: str) -> str:
    return "".join(token.raw for token in _tokens(text, final=True) if not token.tag)


def speech_prefix(text: str, clean_chars: int) -> str:
    """Return only the marked prefix authorized by a clean transcript cursor."""
    if clean_chars <= 0:
        return ""
    count = 0
    for token in _tokens(text):
        count += not token.tag
        if count >= clean_chars:
            return text[: token.end]
    return text


class SpeechTagFilter:
    """Hold split control tokens until they can be identified without flashing."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        self._pending += text
        tokens = _tokens(self._pending)
        consumed = tokens[-1].end if tokens else 0
        self._pending = self._pending[consumed:]
        return "".join(token.raw for token in tokens if not token.tag)

    def finish(self) -> str:
        tail = strip_speech_tags(self._pending)
        self._pending = ""
        return tail


def _is_clause_boundary(tokens: list[_Token], index: int) -> bool:
    character = tokens[index].raw
    if character in _STRONG:
        return True
    if character not in _SOFT and character != ".":
        return False
    before = tokens[index - 1].raw if index else ""
    after = tokens[index + 1].raw if index + 1 < len(tokens) else ""
    if character in ".,:":
        # Decimal/thousands separators, times and addresses are not pauses.
        if before.isdigit() and after.isdigit():
            return False
        if character == "." and before.isascii() and after.isascii():
            if before.isalnum() and after.isalnum():
                return False
        if character == ":" and after == "/":
            return False
    return True


class SpeechTagRepeater:
    """Reassert style inside provider text, without changing source offsets.

    One instance belongs to one utterance. Explicit controls replace the
    previous style; punctuation and provider task boundaries do not. Effects
    are copied once. The original marked transcript remains untouched so
    speculative speech can still promote using its exact raw cursor.
    """

    def __init__(self) -> None:
        self._control = ""

    def prepare(self, text: str) -> tuple[str, str]:
        tokens = _tokens(text, final=True)
        first = next((token for token in tokens if not token.raw.isspace()), None)
        prefix = self._control if first and first.raw not in _CONTROLS else ""
        expression_text = prefix + "".join(
            token.raw for token in tokens if not token.unsupported
        )
        output = []
        clause_start = True
        for index, token in enumerate(tokens):
            if token.raw in _CONTROLS:
                self._control = token.raw
                output.append(token.raw)
                clause_start = False
            elif token.unsupported:
                continue
            elif token.tag:
                output.append(token.raw)
            elif _is_clause_boundary(tokens, index):
                output.append(token.raw)
                clause_start = True
            elif token.raw.isspace() or token.raw in _CLOSERS:
                output.append(token.raw)
            else:
                if clause_start and self._control:
                    output.append(self._control)
                output.append(token.raw)
                clause_start = False
        return expression_text, "".join(output)


class SpeechTextBuffer:
    """One-shot synthesis pieces; offsets remain in the unmarked transcript.

    The first phrase may commit at a comma after eight visible characters;
    subsequent pieces prefer sentence ends. Long sentences may split at a
    soft pause after eighty characters. No extra model is used to segment.
    """

    def __init__(self, *, first_chunk_at_pause: bool = True) -> None:
        self.first_chunk_at_pause = first_chunk_at_pause
        self.reset()

    def reset(self) -> None:
        self._pending = ""
        self._sent_offset = 0
        self._control = ""

    def append(self, text: str) -> list[str]:
        self._pending += text
        chunks = []
        while (end := self._boundary()) is not None:
            chunks.append(self._take(end))
        return chunks

    @staticmethod
    def _playable(text: str) -> bool:
        return bool(strip_speech_tags(text).strip()) or any(
            token.raw in _EFFECTS for token in _tokens(text)
        )

    def _boundary(self) -> int | None:
        tokens = _tokens(self._pending)
        count = 0
        for index, token in enumerate(tokens):
            if token.tag:
                continue
            count += 1
            char = token.raw
            strong = char in _STRONG or (
                char == "."
                and index + 1 < len(tokens)
                and tokens[index + 1].raw.isspace()
            )
            soft = char in _SOFT and (
                (self.first_chunk_at_pause and self._sent_offset == 0 and count >= 8)
                or count >= 80
            )
            if not (strong or soft):
                continue
            end = token.end
            for following in tokens[index + 1 :]:
                if following.raw in _EFFECTS or (
                    not following.tag
                    and (following.raw in _CLOSERS or following.raw in _STRONG)
                ):
                    end = following.end
                else:
                    break
            # A following visible character settles closers and punctuation
            # without waiting for the rest of the response. finish/flush may
            # release a completed final sentence without this lookahead.
            if any(
                not next_token.tag and not next_token.raw.isspace()
                for next_token in tokens
                if next_token.start >= end
            ):
                return end
        return None

    def _take(self, length: int) -> str:
        raw, self._pending = self._pending[:length], self._pending[length:]
        prefix = self._control
        tokens = _tokens(raw, final=True)
        if tokens and tokens[0].raw in _CONTROLS:
            prefix = ""
        for token in tokens:
            if token.raw in _CONTROLS:
                self._control = token.raw
            elif not token.tag:
                self._sent_offset += 1
                if token.raw in _STRONG or token.raw == ".":
                    self._control = ""
        return prefix + raw

    def flush_through(self, end_offset: int) -> list[str]:
        if isinstance(end_offset, bool) or not isinstance(end_offset, int):
            return []
        needed = end_offset - self._sent_offset
        if needed < 0:
            return []
        count = 0
        end = 0
        for token in _tokens(self._pending):
            if (
                count >= needed
                and token.raw not in _EFFECTS
                and not (needed == 0 and token.raw in _CONTROLS)
            ):
                break
            end = token.end
            if not token.tag:
                count += 1
        if not end or not self._playable(self._pending[:end]):
            return []
        return [self._take(end)]

    def finish(self) -> list[str]:
        if not self._pending:
            return []
        text = self._take(len(self._pending))
        return [text] if self._playable(text) else []

    def take_first_pause_flush(self) -> bool:
        # Every marked append is committed once by the server.
        return False

    def adopt_sent_prefix(
        self, content: str, sent_offset: int, *, speech_offset: int | None = None
    ) -> list[str]:
        if not 0 <= sent_offset <= len(strip_speech_tags(content)):
            raise ValueError("speech prefix offset is outside the clean text")
        if speech_offset is not None and (
            not 0 <= speech_offset <= len(content)
            or len(strip_speech_tags(content[:speech_offset])) != sent_offset
        ):
            raise ValueError("marked prefix does not match the clean cursor")
        # Failed adoption must leave no text behind for ordinary TTS to append again.
        self.reset()
        self._pending = content
        if speech_offset is not None:
            self._take(speech_offset)
        elif sent_offset:
            self.flush_through(sent_offset)
        return self.append("")
