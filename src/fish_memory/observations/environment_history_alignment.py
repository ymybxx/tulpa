"""Conservative attribution of an intact ASR transcript to local voice spans.

SenseVoice supplies token points, not reliable word end times. The intervals
below describe supporting evidence around those points; they are not forced
alignment and must not become playback/consumption offsets.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from fish_memory.companion.asr import validated_token_alignment
from fish_memory.observations.environment_history_speakers import (
    SAMPLE_RATE,
    SpeakerSpan,
)

_POINT_GUARD_SECONDS = 0.12
_TIME_TOLERANCE_SECONDS = 0.03
_SILENCE_BREAK_SECONDS = 0.65
_MAX_TEXT_CHARS = 4000
_MAX_PARTIAL_MATCH_CELLS = 1_000_000
_METADATA = re.compile(r"<\|[^<>\r\n]{1,64}\|>|</?Music>", re.IGNORECASE)
# Preserve written numbers, Latin words and their internal punctuation. An
# acoustic boundary is not permission to turn 5500 into two spoken amounts.
_LEXEME = re.compile(r"[0-9]+(?:[.,][0-9]+)+|[A-Za-z0-9]+(?:[.'’_-][A-Za-z0-9]+)*")


@dataclass(frozen=True, slots=True)
class AlignedPart:
    text: str
    start_seconds: float
    end_seconds: float
    span_index: int | None
    reason: str
    overlap: bool = False


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    parts: tuple[AlignedPart, ...]
    status: str


@dataclass(frozen=True, slots=True)
class _Point:
    time: float
    span_index: int | None
    reason: str
    overlap: bool
    # Preserve separate unknown islands even though neither has an identity.
    island: tuple[int, ...]


def align_transcript(
    text: str,
    tokens: tuple[str, ...],
    timestamps: tuple[float, ...],
    spans: tuple[SpeakerSpan, ...],
    duration: float,
) -> AlignmentResult:
    """Preserve every original character; attach identity only with evidence.

    An empty ``parts`` is a request to retain the whole text without speakers,
    never to retry cropped ASR. ``span_index`` is supplied only for a reliable
    time association, including an originally unknown speaker span. Other
    reasons must not borrow any neighbouring span's identity.
    """
    if not text:
        return AlignmentResult((), "empty")
    if not isinstance(duration, (float, int)) or isinstance(duration, bool):
        return AlignmentResult((), "invalid_duration")
    if not math.isfinite(duration) or duration <= 0:
        return AlignmentResult((), "invalid_duration")
    if not tokens and not timestamps:
        return AlignmentResult((), "missing_alignment")
    valid_tokens, valid_times = validated_token_alignment(tokens, timestamps)
    if (
        not valid_tokens
        or len(text) > _MAX_TEXT_CHARS
        or valid_times[-1] > duration + _TIME_TOLERANCE_SECONDS
    ):
        return AlignmentResult((), "invalid_alignment")
    if not _valid_spans(spans, duration):
        return AlignmentResult((), "invalid_spans")
    mapping, exact = _map_text(text, valid_tokens)
    if mapping is None:
        return AlignmentResult((), "unalignable_text")
    points = tuple(
        _point(min(timestamp, duration), spans, duration) for timestamp in valid_times
    )
    characters: list[_Point | None] = [
        points[index] if index is not None else None for index in mapping
    ]
    # Punctuation timestamps can be delayed into silence/the next person's
    # speech. Attach presentation characters to lexical evidence instead.
    previous: _Point | None = None
    for index, character in enumerate(text):
        if _presentation(character):
            characters[index] = previous
        else:
            previous = characters[index]
    following: _Point | None = None
    for index in range(len(text) - 1, -1, -1):
        if _presentation(text[index]):
            if characters[index] is None:
                characters[index] = following
        else:
            following = characters[index]
    return AlignmentResult(
        _parts(text, characters, duration), "aligned" if exact else "partial"
    )


def _valid_spans(spans: tuple[SpeakerSpan, ...], duration: float) -> bool:
    previous_start = 0
    for span in spans:
        if (
            type(span.start_sample) is not int
            or type(span.end_sample) is not int
            or span.start_sample < previous_start
            or span.start_sample < 0
            or span.end_sample <= span.start_sample
            or span.end_sample / SAMPLE_RATE > duration + _TIME_TOLERANCE_SECONDS
        ):
            return False
        previous_start = span.start_sample
    return True


def _canonical_tokens(tokens: tuple[str, ...]) -> tuple[str, list[int]]:
    characters: list[str] = []
    indexes: list[int] = []
    for index, token in enumerate(tokens):
        cleaned = _METADATA.sub("", token).replace("▁", " ")
        if cleaned.startswith("##"):
            cleaned = cleaned[2:]
        for character in cleaned:
            if not character.isspace():
                characters.append(character)
                indexes.append(index)
    return "".join(characters), indexes


def _map_text(
    text: str, tokens: tuple[str, ...]
) -> tuple[list[int | None] | None, bool]:
    source, source_tokens = _canonical_tokens(tokens)
    target_indexes = [index for index, char in enumerate(text) if not char.isspace()]
    target = "".join(text[index] for index in target_indexes)
    if not source or not target:
        return None, False
    mapping: list[int | None] = [None] * len(text)
    if source == target:
        for offset, text_index in enumerate(target_indexes):
            mapping[text_index] = source_tokens[offset]
        return mapping, True
    if len(source) * len(target) > _MAX_PARTIAL_MATCH_CELLS:
        return None, False
    # Do not align equal-looking repeated dialogue by whichever occurrence
    # SequenceMatcher chooses. Only substantial unique literal anchors can
    # survive an ITN/text disagreement; replacements themselves stay unknown.
    for block in SequenceMatcher(
        None, source, target, autojunk=False
    ).get_matching_blocks():
        anchor = source[block.a : block.a + block.size]
        if (
            sum(not _presentation(char) for char in anchor) < 3
            or source.find(anchor) != source.rfind(anchor)
            or target.find(anchor) != target.rfind(anchor)
            or not _whole_ascii_words(source, block.a, block.size)
            or not _whole_ascii_words(target, block.b, block.size)
        ):
            continue
        for offset in range(block.size):
            mapping[target_indexes[block.b + offset]] = source_tokens[block.a + offset]
    return (
        (mapping, False)
        if any(index is not None for index in mapping)
        else (None, False)
    )


def _whole_ascii_words(text: str, start: int, length: int) -> bool:
    end = start + length

    def letter(character: str) -> bool:
        return character.isascii() and character.isalnum()

    return not (
        (start > 0 and letter(text[start - 1]) and letter(text[start]))
        or (end < len(text) and letter(text[end - 1]) and letter(text[end]))
    )


def _presentation(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")


def _point(time: float, spans: tuple[SpeakerSpan, ...], duration: float) -> _Point:
    nearby = tuple(
        index
        for index, span in enumerate(spans)
        if span.start_sample / SAMPLE_RATE <= time + _POINT_GUARD_SECONDS
        and span.end_sample / SAMPLE_RATE >= time - _POINT_GUARD_SECONDS
    )
    containing = tuple(
        index
        for index in nearby
        if spans[index].start_sample / SAMPLE_RATE <= time
        and (
            time < spans[index].end_sample / SAMPLE_RATE
            or time == duration == spans[index].end_sample / SAMPLE_RATE
        )
    )
    if any(spans[index].overlap for index in containing) or len(containing) > 1:
        return _Point(time, None, "overlap", True, containing or nearby)
    if not containing:
        return _Point(time, None, "outside_speech", False, nearby)
    index = containing[0]
    span = spans[index]
    if span.reason in {"low_segmentation_confidence", "low_audio_energy"}:
        return _Point(time, None, "low_segmentation_confidence", False, containing)
    # SenseVoice supplies points, not word durations. Attribute a point only
    # to the interval that actually contains it. The display padding below is
    # not acoustic evidence that adjacent people spoke this token together.
    # Gaps, uncertain spans and actual overlap remain unassigned.
    return _Point(time, index, "token_time", False, containing)


def _parts(
    text: str, points: list[_Point | None], duration: float
) -> tuple[AlignedPart, ...]:
    boundaries = {0, len(text)}
    for index in range(1, len(text)):
        previous, current = points[index - 1], points[index]
        if _separate(previous, current):
            boundaries.add(index)
    ambiguous: set[tuple[int, int]] = set()
    for match in _LEXEME.finditer(text):
        if not boundaries.intersection(range(match.start() + 1, match.end())):
            continue
        end = match.end()
        while end < len(text) and _presentation(text[end]):
            end += 1
        boundaries.difference_update(range(match.start() + 1, end))
        boundaries.update((match.start(), end))
        ambiguous.add((match.start(), end))
    ordered = sorted(boundaries)
    parts: list[AlignedPart] = []
    for start, end in zip(ordered[:-1], ordered[1:], strict=True):
        evidence = [point for point in points[start:end] if point is not None]
        if evidence:
            first = evidence[0]
            uncertain_word = (start, end) in ambiguous
            part_start = max(0.0, first.time - _POINT_GUARD_SECONDS)
            part_end = min(duration, evidence[-1].time + _POINT_GUARD_SECONDS)
            part = AlignedPart(
                text[start:end],
                part_start,
                part_end,
                None if uncertain_word else first.span_index,
                "ambiguous_word_boundary" if uncertain_word else first.reason,
                any(point.overlap for point in evidence)
                if uncertain_word
                else first.overlap,
            )
        else:
            previous = next(
                (point for point in reversed(points[:start]) if point), None
            )
            following = next((point for point in points[end:] if point), None)
            part = AlignedPart(
                text[start:end],
                max(0.0, (previous.time if previous else 0.0) - _POINT_GUARD_SECONDS),
                min(
                    duration,
                    (following.time if following else duration) + _POINT_GUARD_SECONDS,
                ),
                None,
                "text_unaligned",
            )
        parts.append(part)
    return tuple(parts)


def _separate(previous: _Point | None, current: _Point | None) -> bool:
    if previous is None or current is None:
        return previous is not current
    return (
        previous.span_index,
        previous.reason,
        previous.overlap,
        previous.island,
    ) != (
        current.span_index,
        current.reason,
        current.overlap,
        current.island,
    ) or current.time - previous.time > _SILENCE_BREAK_SECONDS
