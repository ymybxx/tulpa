"""Recheck only the join between adjacent system-audio windows.

Model-local channel numbers are not identities across inference calls. A second
segmentation of the previous two seconds and the next two seconds can establish
a continuous turn across the join, then map it back through shared timestamps.
This produces evidence for the caller to check against its voice matches; it
never creates or updates a voice template, nor returns old audio for ASR.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 10 * _SAMPLE_RATE
_SIDE_SAMPLES = 2 * _SAMPLE_RATE
_EDGE_TOLERANCE = int(0.08 * _SAMPLE_RATE)
_MIN_EVIDENCE = int(0.20 * _SAMPLE_RATE)
_MIN_COVERAGE = 0.80


class BoundaryTurn(Protocol):
    """Sample positions and model-local labels from one segmentation call."""

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...

    @property
    def local_id(self) -> int | None: ...

    @property
    def overlap(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class BoundaryCheck:
    proposals: dict[int, str]
    reason: str
    track_proposals: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int
    local_id: int | None
    overlap: bool = False


@dataclass(frozen=True, slots=True)
class _Reference:
    start: int
    end: int
    local_id: int | None
    overlap: bool
    identity: str | None
    track_id: str | None


def _clip(turns: Sequence[BoundaryTurn], start: int, end: int) -> tuple[_Span, ...]:
    return tuple(
        _Span(
            max(turn.start, start) - start,
            min(turn.end, end) - start,
            turn.local_id,
            turn.overlap,
        )
        for turn in turns
        if turn.start < end and turn.end > start
    )


def _coverage(turns: Sequence[BoundaryTurn], start: int, end: int) -> int:
    """Count supported samples once, even if a caller supplied overlapping spans."""
    covered = 0
    through = start
    for turn in sorted(turns, key=lambda turn: turn.start):
        left, right = max(through, turn.start), min(end, turn.end)
        if right > left:
            covered += right - left
            through = right
    return covered


def _continuous(turns: Sequence[BoundaryTurn]) -> tuple[_Span, ...]:
    result: list[_Span] = []
    for turn in turns:
        current = _Span(turn.start, turn.end, turn.local_id, turn.overlap)
        if result:
            previous = result[-1]
            if (
                previous.local_id is not None
                and previous.local_id == current.local_id
                and not previous.overlap
                and not current.overlap
                and 0 <= current.start - previous.end <= _EDGE_TOLERANCE
            ):
                result[-1] = _Span(previous.start, current.end, current.local_id)
                continue
        result.append(current)
    return tuple(result)


class SpeakerBoundaryTracker:
    """One worker, at most 64,000 PCM bytes; resolve is read-only and optional.

    Call remember only after a window is successfully committed. The caller
    clears this cache on stream reset, close, or rejected/reordered input, and
    may retain its independent global voice templates across a small gap.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._pcm = b""
        self._stream_id: str | None = None
        self._window_index: int | None = None
        self._turns: tuple[_Reference, ...] = ()
        self._reliable: tuple[_Span, ...] = ()

    def remember(
        self,
        pcm: bytes,
        *,
        stream_id: str,
        window_index: int,
        turns: Sequence[BoundaryTurn],
        reliable: Sequence[BoundaryTurn],
        identities: Mapping[int, str],
        tracks: Mapping[int, str] | None = None,
    ) -> None:
        # A partial capture cannot establish temporal adjacency to the next
        # ten-second window. Keep no partial tail to stitch over missing audio.
        if len(pcm) != _WINDOW_SAMPLES * 2:
            self.clear()
            return
        start = _WINDOW_SAMPLES - _SIDE_SAMPLES
        self._pcm = pcm[-_SIDE_SAMPLES * 2 :]
        self._stream_id, self._window_index = stream_id, window_index
        self._turns = tuple(
            _Reference(
                max(turn.start, start) - start,
                min(turn.end, _WINDOW_SAMPLES) - start,
                turn.local_id,
                turn.overlap,
                identities.get(index),
                (tracks or {}).get(index),
            )
            for index, turn in enumerate(turns)
            if turn.start < _WINDOW_SAMPLES and turn.end > start
        )
        self._reliable = _clip(reliable, start, _WINDOW_SAMPLES)

    def resolve(
        self,
        pcm: bytes,
        *,
        stream_id: str,
        window_index: int,
        turns: Sequence[BoundaryTurn],
        reliable: Sequence[BoundaryTurn],
        infer: Callable[[NDArray[np.float32]], Sequence[BoundaryTurn]],
    ) -> BoundaryCheck:
        """Return current-island -> previous voice/track continuity proposals.

        infer receives one 10-second float32 array: old 2 s + new <=2 s +
        trailing zero padding. Its output must use reliable-only segmentation;
        low-confidence/overlap spans remain explicit barriers. Only the current
        first island can inherit, and a speaker matcher must veto conflicting
        embeddings or local cannot-link evidence before accepting a proposal.
        """
        if not self._pcm or self._window_index is None:
            return BoundaryCheck({}, "no_previous")
        if self._stream_id != stream_id or window_index != self._window_index + 1:
            return BoundaryCheck({}, "not_contiguous")
        if not turns or not self._turns or len(pcm) % 2:
            return BoundaryCheck({}, "no_boundary_candidate")
        previous, current = self._turns[-1], turns[0]
        if (
            (previous.identity is None and previous.track_id is None)
            or previous.local_id is None
            or previous.overlap
            or previous.end < _SIDE_SAMPLES - _EDGE_TOLERANCE
            or current.local_id is None
            or current.overlap
            or current.start > _EDGE_TOLERANCE
        ):
            return BoundaryCheck({}, "no_boundary_candidate")

        head_samples = min(len(pcm) // 2, _SIDE_SAMPLES)
        current_end = min(current.end, head_samples)
        old_clean = tuple(
            turn
            for turn in self._reliable
            if turn.local_id == previous.local_id and not turn.overlap
        )
        new_clean = tuple(
            turn
            for turn in reliable
            if turn.local_id == current.local_id and not turn.overlap
        )
        if (
            not old_clean
            or not new_clean
            or old_clean[-1].end < _SIDE_SAMPLES - _EDGE_TOLERANCE
            or new_clean[0].start > _EDGE_TOLERANCE
            or _coverage(old_clean, previous.start, previous.end) < _MIN_EVIDENCE
            or _coverage(new_clean, current.start, current_end) < _MIN_EVIDENCE
        ):
            return BoundaryCheck({}, "insufficient_evidence")

        joined = np.zeros(_WINDOW_SAMPLES, dtype=np.float32)
        joined[:_SIDE_SAMPLES] = np.frombuffer(self._pcm, dtype="<i2") / 32768.0
        joined[_SIDE_SAMPLES : _SIDE_SAMPLES + head_samples] = (
            np.frombuffer(pcm[: head_samples * 2], dtype="<i2") / 32768.0
        )
        try:
            spans = _clip(infer(joined), 0, _SIDE_SAMPLES + head_samples)
        except Exception:
            # Boundary verification is supplementary. A failure must not erase
            # successfully obtained current-window segmentation or embeddings.
            return BoundaryCheck({}, "inference_failed")

        for bridge in _continuous(spans):
            if (
                bridge.local_id is None
                or bridge.overlap
                or bridge.start > _SIDE_SAMPLES - _MIN_EVIDENCE
                or bridge.end < _SIDE_SAMPLES + _MIN_EVIDENCE
            ):
                continue
            old_start, old_end = max(previous.start, bridge.start), previous.end
            new_start = current.start
            new_end = min(current_end, bridge.end - _SIDE_SAMPLES)
            if (
                _coverage(old_clean, old_start, old_end) < _MIN_EVIDENCE
                or _coverage(new_clean, new_start, new_end) < _MIN_EVIDENCE
                or old_end - old_start < (previous.end - previous.start) * _MIN_COVERAGE
                or new_end - new_start < (current_end - current.start) * _MIN_COVERAGE
                or any(
                    (
                        (
                            turn.identity is not None
                            and previous.identity is not None
                            and turn.identity != previous.identity
                        )
                        or (
                            turn.track_id is not None
                            and previous.track_id is not None
                            and turn.track_id != previous.track_id
                        )
                    )
                    and min(turn.end, bridge.end) - max(turn.start, bridge.start)
                    > _EDGE_TOLERANCE
                    for turn in self._turns
                )
            ):
                continue
            # Reject contradictory channels, overlap or uncertain regions in
            # either reference region, rather than treating 80% coverage as
            # permission to ignore a brief but real change near the boundary.
            reference_start = max(previous.start, bridge.start)
            reference_end = _SIDE_SAMPLES + current_end
            if any(
                (turn.local_id != bridge.local_id or turn.overlap)
                and min(turn.end, reference_end) - max(turn.start, reference_start)
                > _EDGE_TOLERANCE
                for turn in spans
            ):
                continue
            return BoundaryCheck(
                {0: previous.identity} if previous.identity is not None else {},
                "continuous",
                {0: previous.track_id} if previous.track_id is not None else {},
            )
        return BoundaryCheck({}, "boundary_unconfirmed")
