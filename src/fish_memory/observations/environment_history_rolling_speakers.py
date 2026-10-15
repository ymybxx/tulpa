"""Incremental short-utterance embeddings with bounded retrospective labeling.

Only new five-second segmentation steps and new utterance embeddings run native
inference. Five minutes of small embeddings can be regrouped; PCM is retained
for twenty seconds only. This uses the existing sherpa/ONNX runtime, not the
3D-Speaker Python framework.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from fish_memory.model_catalog import speaker_embedding_urls
from fish_memory.observations.environment_history_speakers import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    SpeakerDiarizationError,
    SpeakerSpan,
    SpeakerWindowOrderError,
    _atomic_write,
    _download,
    _verified,
)
from fish_memory.observations.environment_history_speakers import (
    LocalSpeakerDiarizer as _NativeKernels,
)
from fish_memory.observations.environment_history_voice_clusters import (
    VoiceClusterBank,
    VoiceEvidence,
)

EMBEDDING_MODEL_ID = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
EMBEDDING_MODEL_SIZE = 28_281_164
EMBEDDING_MODEL_SHA256 = (
    "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"
)
EMBEDDING_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    f"speaker-recongition-models/{EMBEDDING_MODEL_ID}"
)
_FRAME_SAMPLES = 320
_CHUNK_SAMPLES = 24_000
_HOP_SAMPLES = 12_000
_PADDING_SAMPLES = 1_280
_MERGE_GAP_SAMPLES = 2_880
_MIN_UTTERANCE_SAMPLES = 4_000
_LOOKAHEAD_SAMPLES = 80_000
_HISTORY_SAMPLES = 300 * SAMPLE_RATE
_MAX_EMBEDDINGS = 512


class _MediaKernels(_NativeKernels):
    def _embedding_path(self) -> Path:
        path = self._directory / EMBEDDING_MODEL_ID
        if not _verified(path, EMBEDDING_MODEL_SIZE, EMBEDDING_MODEL_SHA256):
            data = _download(
                speaker_embedding_urls(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_URL),
                EMBEDDING_MODEL_SIZE,
                EMBEDDING_MODEL_SHA256,
            )
            _atomic_write(path, data)
        return path


@dataclass(frozen=True, slots=True)
class SpeakerWindowRevision:
    window_index: int
    spans: tuple[SpeakerSpan, ...]


@dataclass(frozen=True, slots=True)
class _View:
    start: int
    end: int
    probabilities: NDArray[np.float32]


@dataclass
class _State:
    bank: VoiceClusterBank
    namespace: str = field(default_factory=lambda: uuid4().hex[:6])
    stream_id: str | None = None
    origin_index: int = 0
    origin_time: float | None = None
    last_index: int | None = None
    last_digest: bytes | None = None
    last_spans: tuple[SpeakerSpan, ...] = ()
    pcm: NDArray[np.float32] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    pcm_start: int = 0
    previous_sample_count: int = 0
    views: dict[int, _View] = field(default_factory=dict)
    cursor: int = 0
    run_start: int | None = None
    next_chunk: int = 0
    last_active: int = 0
    run_has_chunks: bool = False
    evidence: dict[tuple[int, int], VoiceEvidence] = field(default_factory=dict)
    windows: dict[int, tuple[int, int]] = field(default_factory=dict)
    published: dict[int, tuple[SpeakerSpan, ...]] = field(default_factory=dict)
    overlap: dict[int, bool] = field(default_factory=dict)
    revisions: tuple[SpeakerWindowRevision, ...] = ()

    def copy(self) -> _State:
        return replace(
            self,
            bank=self.bank.copy(),
            views=self.views.copy(),
            evidence=self.evidence.copy(),
            windows=self.windows.copy(),
            published=self.published.copy(),
            overlap=self.overlap.copy(),
        )


class RollingSpeakerDiarizer:
    def __init__(self, settings: object) -> None:
        self._kernels = _MediaKernels(settings)
        self._capacity = self._kernels._capacity
        self._threshold = self._kernels._threshold
        self._margin = self._kernels._margin
        self.reset()

    def _new_bank(self) -> VoiceClusterBank:
        return VoiceClusterBank(
            self._capacity,
            match_threshold=self._threshold,
            match_margin=self._margin,
        )

    def reset(self) -> None:
        self._state = _State(self._new_bank())

    def close(self) -> None:
        self.reset()
        self._kernels.close()

    @property
    def window_revisions(self) -> tuple[SpeakerWindowRevision, ...]:
        return self._state.revisions

    @property
    def revisions(self) -> tuple[()]:
        return ()

    @property
    def track_aliases(self) -> tuple[()]:
        return ()

    def clear_boundary(self, *, discard_tracks: bool = False) -> None:
        state = self._state.copy()
        state.previous_sample_count = 0
        state.pcm = np.empty(0, dtype=np.float32)
        state.views.clear()
        state.run_start = None
        if discard_tracks:
            state.bank = self._new_bank()
            state.evidence.clear()
        self._state = state

    def observe_silence(self, *, stream_id: str, window_index: int) -> None:
        self.segment(
            bytes(WINDOW_SAMPLES * 2), stream_id=stream_id, window_index=window_index
        )

    def observe_silence_at(
        self, *, stream_id: str, window_index: int, started_at: float, duration: float
    ) -> None:
        self.segment(
            bytes(round(duration * SAMPLE_RATE) * 2),
            stream_id=stream_id,
            window_index=window_index,
            started_at=started_at,
        )

    def segment_at(
        self, pcm: bytes, *, stream_id: str, window_index: int, started_at: float
    ):
        return self.segment(
            pcm, stream_id=stream_id, window_index=window_index, started_at=started_at
        )

    def _infer(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        scores = self._kernels._infer(samples)
        probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _extract(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        return self._kernels._extract(samples)

    def _view(self, samples: NDArray[np.float32], start: int) -> _View:
        if not len(samples) or float(np.max(np.abs(samples))) <= 4 / 32768:
            probabilities = np.zeros((589, 7), dtype=np.float32)
            probabilities[:, 0] = 1.0
        else:
            probabilities = self._infer(samples)
        return _View(start, start + len(samples), probabilities)

    @staticmethod
    def _activity(state: _State, start: int, end: int):
        points = np.arange(start, end, _FRAME_SAMPLES, dtype=np.int64)
        total = np.zeros(len(points))
        speech = np.zeros(len(points))
        overlap = np.zeros(len(points))
        for view in state.views.values():
            indexes = np.flatnonzero((points >= view.start) & (points < view.end))
            if not len(indexes):
                continue
            relative = points[indexes] - view.start
            frames = np.clip(((relative - 495.5) / 270).astype(int), 0, 588)
            weight = np.maximum(
                0.05, np.minimum(relative, WINDOW_SAMPLES - relative) / 80000
            )
            total[indexes] += weight
            speech[indexes] += (1 - view.probabilities[frames, 0]) * weight
            overlap[indexes] += view.probabilities[frames, 4:].sum(axis=1) * weight
        denominator = np.maximum(total, 1e-8)
        return points, speech / denominator > 0.5, overlap / denominator > 0.5

    def _emit(self, state: _State, start: int, end: int) -> None:
        key = (start, end)
        if key in state.evidence or end <= start:
            return
        first, last = start - state.pcm_start, end - state.pcm_start
        if first < 0 or last > len(state.pcm):
            raise SpeakerDiarizationError("声音证据超出有界音频缓存")
        overlapping = sum(
            state.overlap.get(point, False)
            for point in range(
                start // _FRAME_SAMPLES * _FRAME_SAMPLES, end, _FRAME_SAMPLES
            )
        )
        if overlapping * _FRAME_SAMPLES > (end - start) * 0.2:
            # A mixed voice must not train either single-person profile.
            return
        samples = state.pcm[first:last]
        if len(samples) < _CHUNK_SAMPLES:
            # Padding is an input-shape operation, not additional independent
            # speaker evidence. The evidence interval retains its real length.
            samples = np.resize(samples, _CHUNK_SAMPLES)
        vector = self._extract(samples)
        state.evidence[key] = VoiceEvidence(
            start, end, vector, state.run_start or start
        )

    def _finish_run(self, state: _State, available_end: int) -> None:
        if state.run_start is None:
            return
        end = min(available_end, state.last_active + _PADDING_SAMPLES)
        real_start = state.run_start + _PADDING_SAMPLES if state.run_start else 0
        if state.last_active - real_start >= _MIN_UTTERANCE_SAMPLES:
            while (
                not state.run_has_chunks
                or state.next_chunk + _CHUNK_SAMPLES < end + _HOP_SAMPLES
            ):
                self._emit(
                    state, state.next_chunk, min(state.next_chunk + _CHUNK_SAMPLES, end)
                )
                state.next_chunk += _HOP_SAMPLES
                state.run_has_chunks = True
        state.run_start = None

    def _advance(self, state: _State, end: int) -> None:
        points, speech, overlap = self._activity(state, state.cursor, end)
        for point, active, simultaneous in zip(points, speech, overlap, strict=True):
            point = int(point)
            state.overlap[point] = bool(simultaneous)
            if active:
                if state.run_start is None:
                    state.run_start = max(state.pcm_start, point - _PADDING_SAMPLES)
                    state.next_chunk = state.run_start
                    state.run_has_chunks = False
                state.last_active = point + _FRAME_SAMPLES
                while state.next_chunk + _CHUNK_SAMPLES <= state.last_active:
                    self._emit(
                        state, state.next_chunk, state.next_chunk + _CHUNK_SAMPLES
                    )
                    state.next_chunk += _HOP_SAMPLES
                    state.run_has_chunks = True
            elif (
                state.run_start is not None
                and point - state.last_active > _MERGE_GAP_SAMPLES
            ):
                self._finish_run(state, state.pcm_start + len(state.pcm))
        state.cursor = end

    @staticmethod
    def _timeline(state: _State) -> list[tuple[int, int, int | None]]:
        spans: list[tuple[int, int, int | None]] = []
        for item in sorted(state.evidence.values(), key=lambda evidence: evidence.key):
            identity = state.bank.labels.get(item.key)
            if identity not in state.bank.confirmed_ids:
                identity = None
            start, end = item.start, item.end
            if spans and spans[-1][2] == identity and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]), identity)
            else:
                if spans and start < spans[-1][1]:
                    start = (start + spans[-1][1]) // 2
                    spans[-1] = (spans[-1][0], start, spans[-1][2])
                spans.append((start, end, identity))
        return spans

    def _window_spans(
        self, state: _State, index: int, timeline
    ) -> tuple[SpeakerSpan, ...]:
        origin, end = state.windows[index]
        relevant = [
            (max(start, origin), min(stop, end), identity)
            for start, stop, identity in timeline
            if stop > origin and start < end
        ]
        pending_speech: dict[int, bool] = {}
        pending_overlap: dict[int, bool] = {}
        if end > state.cursor:
            points, speech, simultaneous = self._activity(
                state, max(origin, state.cursor), end
            )
            pending_speech = dict(zip(map(int, points), map(bool, speech), strict=True))
            pending_overlap = dict(
                zip(map(int, points), map(bool, simultaneous), strict=True)
            )
        cuts = sorted(
            {
                origin,
                end,
                *range(origin, end, _FRAME_SAMPLES),
                *(point for first, last, _ in relevant for point in (first, last)),
            }
        )
        portions: list[tuple[int, int, int | None, bool, str]] = []
        cursor = 0
        for first, last in zip(cuts[:-1], cuts[1:], strict=True):
            while cursor < len(relevant) and relevant[cursor][1] <= first:
                cursor += 1
            covering = (
                relevant[cursor]
                if cursor < len(relevant)
                and relevant[cursor][0] <= first < relevant[cursor][1]
                else None
            )
            point = first // _FRAME_SAMPLES * _FRAME_SAMPLES
            overlap = pending_overlap.get(point, state.overlap.get(point, False))
            identity = covering[2] if covering else None
            if overlap:
                identity, reason = None, "overlap"
            elif covering:
                reason = (
                    "rolling_acoustic_cluster"
                    if identity is not None
                    else "insufficient_cluster_evidence"
                )
            elif pending_speech.get(point, False):
                reason = "awaiting_acoustic_context"
            else:
                continue
            if (
                portions
                and portions[-1][1] == first
                and portions[-1][2:] == (identity, overlap, reason)
            ):
                portions[-1] = (portions[-1][0], last, identity, overlap, reason)
            else:
                portions.append((first, last, identity, overlap, reason))
        return tuple(
            SpeakerSpan(
                first - origin,
                last - origin,
                f"声音{state.namespace}-{identity:02d}"
                if identity is not None
                else None,
                overlap=overlap,
                local_speaker_id=f"local:{state.namespace}:{index}:{first}",
                reason=reason,
                clean_seconds=0.0 if overlap else (last - first) / SAMPLE_RATE,
                identity_source="acoustic_cluster"
                if identity is not None
                else "unknown",
                track_id=f"track:{state.namespace}:{identity}"
                if identity is not None
                else None,
                track_reason=reason,
            )
            for first, last, identity, overlap, reason in portions
        )

    def segment(
        self,
        pcm: bytes,
        *,
        stream_id: str,
        window_index: int,
        started_at: float | None = None,
    ) -> tuple[SpeakerSpan, ...]:
        if (
            type(pcm) is not bytes
            or len(pcm) % 2
            or len(pcm) > WINDOW_SAMPLES * 2
            or not stream_id
            or type(window_index) is not int
            or window_index < 0
        ):
            raise SpeakerDiarizationError("声音分段输入必须是至多十秒的有序 PCM16")
        old = self._state
        digest = hashlib.sha256(pcm).digest()
        if old.stream_id == stream_id and old.last_index is not None:
            if window_index == old.last_index and digest == old.last_digest:
                return old.last_spans
            if window_index <= old.last_index:
                raise SpeakerWindowOrderError("声音分段窗口重复或顺序无效")
        if old.stream_id != stream_id or (
            old.last_index is not None and window_index - old.last_index > 7
        ):
            state = _State(
                self._new_bank(),
                stream_id=stream_id,
                origin_index=window_index,
            )
        else:
            state = old.copy()
        if started_at is not None:
            if state.origin_time is None:
                state.origin_time = started_at
            start = round((started_at - state.origin_time) * SAMPLE_RATE)
        elif state.last_index is not None and window_index == state.last_index + 1:
            start = state.windows[state.last_index][1]
        else:
            start = (window_index - state.origin_index) * WINDOW_SAMPLES
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        contiguous = (
            state.last_index is not None
            and window_index == state.last_index + 1
            and state.previous_sample_count > 0
            and state.windows[state.last_index][1] == start
        )
        if not contiguous or not len(state.pcm):
            state.pcm_start = start
            state.pcm = samples
            state.cursor = start
            state.run_start = None
            state.views = {}
        else:
            state.pcm = np.concatenate((state.pcm, samples))
            boundary = start - state.pcm_start
            lookback = min(_LOOKAHEAD_SAMPLES, boundary)
            previous = state.pcm[boundary - lookback : boundary]
            shifted = np.concatenate(
                (previous[-_LOOKAHEAD_SAMPLES:], samples[:_LOOKAHEAD_SAMPLES])
            )
            state.views[start - lookback] = self._view(shifted, start - lookback)
        state.views[start] = self._view(samples, start)
        end = start + len(samples)
        state.windows[window_index] = (start, end)
        frontier = max(state.cursor, end - _LOOKAHEAD_SAMPLES)
        self._advance(state, frontier)
        cutoff = end - _HISTORY_SAMPLES
        state.evidence = dict(
            sorted(
                (
                    (key, item)
                    for key, item in state.evidence.items()
                    if item.end > cutoff
                ),
                key=lambda pair: pair[0],
            )[-_MAX_EMBEDDINGS:]
        )
        if any(
            item is not old.evidence.get(key) for key, item in state.evidence.items()
        ):
            state.bank.reconcile(tuple(state.evidence.values()))
        else:
            # Retiring old vectors is bookkeeping, not new acoustic evidence.
            # It must not change people's old labels during silence solely
            # because the five-minute cache's leading edge moved forward.
            state.bank.labels = {
                key: identity
                for key, identity in state.bank.labels.items()
                if key in state.evidence
            }
        state.windows = {
            index: bounds
            for index, bounds in state.windows.items()
            if bounds[1] > cutoff
        }
        state.overlap = {
            point: value for point, value in state.overlap.items() if point > cutoff
        }
        timeline = self._timeline(state)
        revisions = []
        for index in state.windows:
            spans = self._window_spans(state, index, timeline)
            if index != window_index and spans != state.published.get(index, ()):
                revisions.append(SpeakerWindowRevision(index, spans))
            state.published[index] = spans
        state.published = {
            index: spans
            for index, spans in state.published.items()
            if index in state.windows
        }
        state.revisions = tuple(revisions)
        state.last_spans = state.published[window_index]
        state.last_index, state.last_digest, state.previous_sample_count = (
            window_index,
            digest,
            len(samples),
        )
        retain_start = max(state.pcm_start, end - 20 * SAMPLE_RATE)
        state.pcm = state.pcm[retain_start - state.pcm_start :].copy()
        state.pcm_start = retain_start
        state.views = {
            key: view for key, view in state.views.items() if view.end > retain_start
        }
        self._state = state
        return state.last_spans
