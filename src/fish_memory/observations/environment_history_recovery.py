"""Bounded recovery of words missing from an otherwise intact transcription.

Speech activity can expose a sparse decode, but it cannot prove that decoded
words are wrong. Recovery therefore retains every existing lexical anchor and
only replaces one whole speech passage after an independently timed decode.
"""

from __future__ import annotations

from dataclasses import dataclass

from fish_memory.companion.asr import validated_token_alignment
from fish_memory.observations.environment_history_alignment import _map_text
from fish_memory.observations.environment_history_speakers import (
    SAMPLE_RATE,
    SpeakerSpan,
)


@dataclass(frozen=True, slots=True)
class RecoveryRegion:
    start: float
    end: float
    first_token: int
    last_token: int  # inclusive, including trailing presentation tokens
    first_character: int
    last_character: int  # exclusive
    anchors: str


@dataclass(frozen=True, slots=True)
class RecoveredTranscript:
    text: str
    tokens: tuple[str, ...]
    timestamps: tuple[float, ...]


def _lexical(text: str) -> str:
    return "".join(character for character in text if character.isalnum())


def sparse_speech_region(
    text: str,
    tokens: tuple[str, ...],
    timestamps: tuple[float, ...],
    spans: tuple[SpeakerSpan, ...],
    duration: float,
) -> RecoveryRegion | None:
    valid_tokens, valid_times = validated_token_alignment(tokens, timestamps)
    if not valid_tokens or valid_times[-1] > duration + 0.03:
        return None
    lexical_times = [
        timestamp
        for token, timestamp in zip(valid_tokens, valid_times, strict=True)
        if _lexical(token)
    ]
    trailing_gap = bool(
        lexical_times
        and not _lexical(valid_tokens[-1])
        and valid_times[-1] - lexical_times[-1] >= 1.0
    )
    mapping, exact = _map_text(text, valid_tokens)
    if not exact or mapping is None:
        return None
    groups: list[list[float]] = []
    for span in spans:
        if not (
            span.local_speaker_id or span.speaker_id or span.track_id or span.overlap
        ):
            continue
        start, end = span.start_sample / SAMPLE_RATE, span.end_sample / SAMPLE_RATE
        if groups and start - groups[-1][1] <= 0.4:
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    candidates: list[tuple[float, RecoveryRegion]] = []
    for start, end in groups:
        length = end - start
        if not 1.2 <= length <= 7.0:
            continue
        indexes = [
            i
            for i, (token, timestamp) in enumerate(
                zip(valid_tokens, valid_times, strict=True)
            )
            if _lexical(token) and start - 0.2 <= timestamp <= end + 0.2
        ]
        anchors = "".join(_lexical(valid_tokens[i]) for i in indexes)
        # A silent or metadata-only decode cannot authorize new dialogue.
        # Two existing lexical characters also exclude isolated sound effects.
        if len(anchors) < 2 or len(anchors) / length >= 2.5:
            continue
        # Missing words can precede the surviving anchors or lie between them,
        # not just precede a trailing punctuation token. Require an actual
        # speech interval without lexical points; short text alone is not a
        # reason to retry. Candidate validation still retains every anchor.
        missing_start = valid_times[indexes[0]] - start >= 1.0
        missing_middle = any(
            valid_times[right] - valid_times[left] >= 1.0
            and not any(
                mark in ".。!?！？"
                for token in valid_tokens[left + 1 : right]
                for mark in token
            )
            for left, right in zip(indexes, indexes[1:], strict=False)
        )
        if not (
            missing_start
            or missing_middle
            or (trailing_gap and len(anchors) / length < 1.8)
        ):
            continue
        first, last = indexes[0], indexes[-1]
        while last + 1 < len(valid_tokens) and not _lexical(valid_tokens[last + 1]):
            last += 1
        characters = [
            i
            for i, token in enumerate(mapping)
            if token is not None and first <= token <= last
        ]
        if not characters:
            continue
        region = RecoveryRegion(
            max(0.0, start - 0.15),
            min(duration, end + 0.15),
            first,
            last,
            characters[0],
            characters[-1] + 1,
            anchors,
        )
        candidates.append((length * 2.5 - len(anchors), region))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def splice_recovered_speech(
    original: RecoveredTranscript,
    candidate: RecoveredTranscript,
    region: RecoveryRegion,
) -> RecoveredTranscript | None:
    words = _lexical(candidate.text)
    if len(words) < max(6, len(region.anchors) + 3) or len(set(words)) < 4:
        return None
    # Missing words may be filled in; this is not a language-model correction
    # of already recognized words or a choice of the more fluent hypothesis.
    remaining = iter(words)
    if not all(
        any(letter == anchor for letter in remaining) for anchor in region.anchors
    ):
        return None
    tokens, times = validated_token_alignment(candidate.tokens, candidate.timestamps)
    if not tokens or times[-1] > region.end - region.start + 0.03:
        return None
    _, exact = _map_text(candidate.text, tokens)
    if not exact:
        return None
    shifted = tuple(time + region.start for time in times)
    before = original.timestamps[: region.first_token]
    after = original.timestamps[region.last_token + 1 :]
    if (before and before[-1] > shifted[0]) or (after and shifted[-1] > after[0]):
        return None
    return RecoveredTranscript(
        original.text[: region.first_character]
        + candidate.text
        + original.text[region.last_character :],
        original.tokens[: region.first_token]
        + tokens
        + original.tokens[region.last_token + 1 :],
        before + shifted + after,
    )
