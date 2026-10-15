"""Bounded, local speaker turns for system audio, without a second ASR stack.

The MIT pyannote/segmentation-3.0 model is the public k2-fsa ONNX export.
CAM++ embeddings use the existing sherpa CPU runtime and model, never an owner
profile. This is incremental matching inspired by Diart's local/global speaker
constraints, not a reproduction of its overlap-weighted neural architecture.
Thresholds below are conservative starting points, not calibrated guarantees.
No captured PCM or voice templates are written to disk or uploaded.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import tarfile
import tempfile
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from fish_memory.companion.owner_voice import (
    EMBEDDING_DIM,
    _Encoder,
)
from fish_memory.companion.owner_voice import (
    MODEL_ID as EMBEDDING_MODEL_ID,
)
from fish_memory.companion.owner_voice import (
    MODEL_SHA256 as EMBEDDING_MODEL_SHA256,
)
from fish_memory.companion.owner_voice import (
    MODEL_SIZE as EMBEDDING_MODEL_SIZE,
)
from fish_memory.companion.owner_voice import (
    MODEL_URL as EMBEDDING_MODEL_URL,
)
from fish_memory.model_catalog import (
    SEGMENTATION_LICENSE_SHA256,
    SEGMENTATION_LICENSE_SIZE,
    segmentation_license_urls,
    segmentation_urls,
    speaker_embedding_urls,
)
from fish_memory.observations.environment_history_speaker_boundary import (
    SpeakerBoundaryTracker,
)
from fish_memory.observations.environment_history_speaker_tracks import (
    SpeakerTrackTable,
    TrackMatch,
    reference_score,
)

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 160_000
MODEL_ID = "pyannote-segmentation-3.0.onnx"
MODEL_SIZE = 5_992_913
# The upstream INT8 export missed clear speaker changes on public AISHELL-3
# turns that this FP32 export distinguished. Keep the paired-verified model.
MODEL_SHA256 = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
MODEL_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
MODEL_ARCHIVE_SIZE = 6_958_444
MODEL_ARCHIVE_SHA256 = (
    "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
)
_ARCHIVE_ROOT = "sherpa-onnx-pyannote-segmentation-3-0"
_FRAME_SIZE = 991
_FRAME_SHIFT = 270
_FRAME_COUNT = (WINDOW_SAMPLES - _FRAME_SIZE) // _FRAME_SHIFT + 1
_MIN_CLEAN_SAMPLES = int(1.2 * SAMPLE_RATE)
_MIN_MATCH_SAMPLES = int(0.5 * SAMPLE_RATE)
_MAX_EMBEDDING_SAMPLES = 5 * SAMPLE_RATE
_BOUNDARY_GUARD_SAMPLES = int(0.05 * SAMPLE_RATE)
_CONFIDENCE = 0.60
_UNINFORMATIVE_CONFIDENCE = 0.35
_MAX_SAME_VOICE_GAP = int(0.25 * SAMPLE_RATE)
_MAX_GAP_WINDOWS = 6
_MIN_RMS = 1e-5
_MAX_REFERENCES = 6
_RECENT_WINDOWS = 6
_MAX_PENDING_TURNS = 64
_POWERSET = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2))


class SpeakerDiarizationError(RuntimeError):
    """Local inference failure, safe for callers to fall back to unknown speech."""


class SpeakerWindowOrderError(ValueError):
    """Caller supplied a stale/changed window; existing speaker state is intact."""


@dataclass(frozen=True, slots=True)
class SpeakerSpan:
    """Half-open sample offsets in the supplied 16 kHz mono PCM16 window."""

    start_sample: int
    end_sample: int
    speaker_id: str | None
    overlap: bool = False
    local_speaker_id: str | None = None
    reason: str = "unavailable"
    clean_seconds: float = 0.0
    best_similarity: float | None = None
    second_similarity: float | None = None
    identity_source: str = "unknown"
    boundary_reason: str | None = None
    track_id: str | None = None
    track_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SpeakerRevision:
    """A label correction for an already returned interval, never new words."""

    window_index: int
    start_sample: int
    end_sample: int
    speaker_id: str | None
    track_id: str | None
    reason: str = "retrospective_match"


@dataclass(frozen=True, slots=True)
class _LocalSpan:
    start: int
    end: int
    local_id: int | None
    overlap: bool = False


@dataclass(frozen=True, slots=True)
class _Voice:
    anchor: NDArray[np.float32]
    centroid: NDArray[np.float32]
    updates: int = 1
    members: tuple[NDArray[np.float32], ...] = ()
    roots: tuple[NDArray[np.float32], ...] = ()


@dataclass(frozen=True, slots=True)
class _PendingVoice:
    vector: NDArray[np.float32]
    window_index: int
    island: int
    channel: int
    clean_samples: int = _MIN_CLEAN_SAMPLES
    quality: float = 1.0


@dataclass(frozen=True, slots=True)
class _RecentWindow:
    spans: tuple[SpeakerSpan, ...]
    vectors: dict[int, NDArray[np.float32]]
    channels: dict[int, int]


@dataclass(frozen=True, slots=True)
class _Evidence:
    samples: NDArray[np.float32]
    clean_samples: int
    reason: str
    span_samples: int = 0


@dataclass(frozen=True, slots=True)
class _Decision:
    identity: str | None = None
    reason: str = "short_evidence"
    best: float | None = None
    second: float | None = None
    source: str = "unknown"
    boundary_reason: str | None = None


def _verified(path: Path, size: int, digest: str) -> bool:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            return False
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest() == digest
    except OSError:
        return False


def _download(urls: str | tuple[str, ...], size: int, digest: str) -> bytes:
    """Fetch a pinned public artifact from the first source that serves it intact."""
    candidates = (urls,) if isinstance(urls, str) else urls
    failure: SpeakerDiarizationError | None = None
    for url in candidates:
        try:
            request = Request(url, headers={"User-Agent": "Fish-Memory/1.0"})
            with urlopen(request, timeout=30) as response:
                body = response.read(size + 1)
        except Exception as error:
            failure = SpeakerDiarizationError("无法下载本地声音分段模型")
            failure.__cause__ = error
            continue
        if len(body) == size and hashlib.sha256(body).hexdigest() == digest:
            return body
        failure = SpeakerDiarizationError("本地声音模型下载校验失败")
    raise failure or SpeakerDiarizationError("无法下载本地声音分段模型")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SpeakerDiarizationError("本地声音模型路径无效")
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
            name = target.name
            target.write(data)
        os.replace(name, path)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _prepare_segmentation(directory: Path) -> Path:
    path = directory / MODEL_ID
    if _verified(path, MODEL_SIZE, MODEL_SHA256):
        return path
    # Mainland mirror first: the same pinned model file plus its license.
    try:
        raw = _download(segmentation_urls(), MODEL_SIZE, MODEL_SHA256)
        license_text = _download(
            segmentation_license_urls(),
            SEGMENTATION_LICENSE_SIZE,
            SEGMENTATION_LICENSE_SHA256,
        )
        if (
            hashlib.sha256(raw).hexdigest() == MODEL_SHA256
            and hashlib.sha256(license_text).hexdigest() == SEGMENTATION_LICENSE_SHA256
        ):
            _atomic_write(directory / "segmentation-3.0.LICENSE", license_text)
            _atomic_write(path, raw)
            return path
    except SpeakerDiarizationError:
        pass
    data = _download(MODEL_ARCHIVE_URL, MODEL_ARCHIVE_SIZE, MODEL_ARCHIVE_SHA256)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:bz2") as archive:
            model = archive.getmember(f"{_ARCHIVE_ROOT}/model.onnx")
            license_member = archive.getmember(f"{_ARCHIVE_ROOT}/LICENSE")
            if (
                not model.isfile()
                or model.size != MODEL_SIZE
                or not license_member.isfile()
                or license_member.size > 4096
            ):
                raise ValueError("invalid members")
            source = archive.extractfile(model)
            license_source = archive.extractfile(license_member)
            if source is None or license_source is None:
                raise ValueError("missing model")
            raw = source.read(MODEL_SIZE + 1)
            license_text = license_source.read(4097)
        if len(raw) != MODEL_SIZE or hashlib.sha256(raw).hexdigest() != MODEL_SHA256:
            raise ValueError("invalid model digest")
        # Preserve the license shipped in the pinned upstream export.
        _atomic_write(directory / "segmentation-3.0.LICENSE", license_text)
        _atomic_write(path, raw)
        return path
    except (OSError, tarfile.TarError, KeyError, ValueError) as error:
        raise SpeakerDiarizationError("本地声音分段模型准备失败") from error


def _prepare_embedding(directory: Path, owner_directory: Path) -> Path:
    # Reuse only the public model file. Never load owner-voice.json or templates.
    owner_model = owner_directory / EMBEDDING_MODEL_ID
    if _verified(owner_model, EMBEDDING_MODEL_SIZE, EMBEDDING_MODEL_SHA256):
        return owner_model
    path = directory / EMBEDDING_MODEL_ID
    if not _verified(path, EMBEDDING_MODEL_SIZE, EMBEDDING_MODEL_SHA256):
        body = _download(
            speaker_embedding_urls(EMBEDDING_MODEL_ID, EMBEDDING_MODEL_URL),
            EMBEDDING_MODEL_SIZE,
            EMBEDDING_MODEL_SHA256,
        )
        _atomic_write(path, body)
    return path


def _unit_vector(value: object) -> NDArray[np.float32]:
    try:
        vector = np.asarray(value, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
    except (TypeError, ValueError, OverflowError) as error:
        raise SpeakerDiarizationError("声音特征格式无效") from error
    if (
        vector.shape != (EMBEDDING_DIM,)
        or not np.isfinite(vector).all()
        or not math.isfinite(norm)
        or norm <= 1e-8
    ):
        raise SpeakerDiarizationError("声音特征数值无效")
    return vector / norm


class LocalSpeakerDiarizer:
    """One caller/worker, one bounded capture stream; no background work here.

    A repeated identical latest window is idempotent. Older or altered repeats
    fail without mutating templates. Gaps of at most six windows retain only
    voice templates (never PCM/local labels); larger gaps or stream changes
    start a new random identity namespace. Least recently matched templates are
    evicted at capacity; IDs only increase and are never recycled. A returning
    evicted voice gets a new ID rather than being assigned another character's.
    """

    def __init__(self, settings: object) -> None:
        self._owner_directory = Path(
            getattr(settings, "owner_voice_dir", ".cache/fish-memory/owner-voice")
        )
        self._directory = self._owner_directory.parent / "speaker-diarization"
        self._threshold = float(
            getattr(settings, "environment_history_speaker_threshold", 0.65)
        )
        self._margin = float(
            getattr(settings, "environment_history_speaker_margin", 0.10)
        )
        self._capacity = int(
            getattr(settings, "environment_history_speaker_max_speakers", 16)
        )
        if (
            not math.isfinite(self._threshold)
            or not 0 < self._threshold < 1
            or not math.isfinite(self._margin)
            or not 0 < self._margin < self._threshold
            or not 1 <= self._capacity <= 64
        ):
            raise ValueError("invalid local speaker matching configuration")
        self._session: Any = None
        self._encoder: Any = None
        self.reset()

    def reset(self) -> None:
        self._boundary = SpeakerBoundaryTracker()
        self._namespace = uuid4().hex[:6]
        self._stream_id: str | None = None
        self._last_index: int | None = None
        self._last_digest: bytes | None = None
        self._last_spans: tuple[SpeakerSpan, ...] = ()
        self._voices: dict[str, _Voice] = {}
        self._next_identity = 1
        self._pending: tuple[_PendingVoice, ...] = ()
        self._recent: dict[int, _RecentWindow] = {}
        self._revisions: tuple[SpeakerRevision, ...] = ()
        self._track_aliases: tuple[tuple[str, str], ...] = ()
        self._voice_aliases: dict[str, str] = {}
        self._cannot_link: set[frozenset[str]] = set()
        self._processing_index = 0
        self._tracks = SpeakerTrackTable(
            self._namespace, self._capacity, self._threshold, self._margin
        )

    def clear_boundary(self, *, discard_tracks: bool = False) -> None:
        """Discard PCM continuity while retaining independent voice templates."""
        self._boundary.clear()
        if discard_tracks:
            self._tracks.clear()

    @property
    def revisions(self) -> tuple[SpeakerRevision, ...]:
        return self._revisions

    @property
    def track_aliases(self) -> tuple[tuple[str, str], ...]:
        return self._track_aliases

    def observe_silence(self, *, stream_id: str, window_index: int) -> None:
        """Commit a complete silent window without loading either model."""
        self.segment(
            bytes(WINDOW_SAMPLES * 2), stream_id=stream_id, window_index=window_index
        )

    def close(self) -> None:
        self.reset()
        self._session = None
        self._encoder = None

    def _load(self) -> None:
        if self._session is not None and self._encoder is not None:
            return
        try:
            ort = import_module("onnxruntime")
            path = _prepare_segmentation(self._directory)
            embedding_path = self._embedding_path()
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            expected = {
                "sample_rate": "16000",
                "window_size": "160000",
                "receptive_field_size": "991",
                "receptive_field_shift": "270",
                "num_speakers": "3",
                "num_classes": "7",
                "powerset_max_classes": "2",
            }
            metadata = session.get_modelmeta().custom_metadata_map
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise SpeakerDiarizationError("本地声音分段模型参数不兼容")
            encoder = _Encoder(embedding_path)
        except SpeakerDiarizationError:
            raise
        except Exception as error:
            raise SpeakerDiarizationError("本地声音分段组件不可用") from error
        self._session, self._encoder = session, encoder

    def _embedding_path(self) -> Path:
        return _prepare_embedding(self._directory, self._owner_directory)

    def _infer(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        self._load()
        padded = np.zeros((1, 1, WINDOW_SAMPLES), dtype=np.float32)
        padded[0, 0, : len(samples)] = samples
        try:
            scores = np.asarray(
                self._session.run(["y"], {"x": padded})[0], dtype=np.float32
            )
        except Exception as error:
            raise SpeakerDiarizationError("本地声音分段推理失败") from error
        if scores.shape != (1, _FRAME_COUNT, 7) or not np.isfinite(scores).all():
            raise SpeakerDiarizationError("本地声音分段结果无效")
        return scores[0]

    @staticmethod
    def _local_spans(
        scores: NDArray[np.float32],
        samples_count: int,
        *,
        reliable_only: bool = False,
    ) -> tuple[_LocalSpan, ...]:
        if scores.shape != (_FRAME_COUNT, 7) or not np.isfinite(scores).all():
            raise SpeakerDiarizationError("本地声音分段结果无效")
        # The export returns log-softmax powerset probabilities. Normalize once
        # defensively; never interpret the seven columns as seven individuals.
        probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        labels = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)
        spans: list[_LocalSpan] = []
        for index, powerset in enumerate(labels):
            start = 0 if index == 0 else int(_FRAME_SIZE / 2 + index * _FRAME_SHIFT)
            end = (
                WINDOW_SAMPLES
                if index + 1 == _FRAME_COUNT
                else int(_FRAME_SIZE / 2 + (index + 1) * _FRAME_SHIFT)
            )
            end = min(end, samples_count)
            if start >= end:
                break
            speakers = _POWERSET[int(powerset)]
            minimum = _CONFIDENCE if reliable_only else _UNINFORMATIVE_CONFIDENCE
            if not speakers and confidence[index] >= minimum:
                continue
            overlap = len(speakers) > 1
            local_id = (
                speakers[0]
                if len(speakers) == 1 and confidence[index] >= minimum
                else None
            )
            if (
                spans
                and spans[-1].end == start
                and spans[-1].local_id == local_id
                and spans[-1].overlap == overlap
            ):
                previous = spans[-1]
                spans[-1] = _LocalSpan(previous.start, end, local_id, overlap)
            else:
                spans.append(_LocalSpan(start, end, local_id, overlap))
        return tuple(spans)

    @staticmethod
    def _join_spans(spans: tuple[SpeakerSpan, ...]) -> tuple[SpeakerSpan, ...]:
        """Join one verified turn, retaining local identity and diagnostics.

        Unknown is an absence of a global match, not a shared speaker identity.
        Different local islands must never become one ASR input just because
        both global IDs are None. Segmentation confidence dips are already
        handled before producing these islands.
        """
        joined: list[SpeakerSpan] = []
        for span in spans:
            if joined:
                previous = joined[-1]
                gap = span.start_sample - previous.end_sample
                if (
                    previous.speaker_id == span.speaker_id
                    and previous.local_speaker_id == span.local_speaker_id
                    and previous.overlap == span.overlap
                    and previous.reason == span.reason
                    and previous.identity_source == span.identity_source
                    and previous.best_similarity == span.best_similarity
                    and previous.second_similarity == span.second_similarity
                    and previous.boundary_reason == span.boundary_reason
                    and previous.track_id == span.track_id
                    and previous.track_reason == span.track_reason
                    and 0 <= gap <= (0 if span.overlap else _MAX_SAME_VOICE_GAP)
                    and (
                        span.speaker_id is not None or span.local_speaker_id is not None
                    )
                ):
                    joined[-1] = SpeakerSpan(
                        previous.start_sample,
                        span.end_sample,
                        span.speaker_id,
                        span.overlap,
                        span.local_speaker_id,
                        span.reason,
                        previous.clean_seconds + span.clean_seconds,
                        span.best_similarity,
                        span.second_similarity,
                        span.identity_source,
                        span.boundary_reason,
                        span.track_id,
                        span.track_reason,
                    )
                    continue
            joined.append(span)
        return tuple(joined)

    @staticmethod
    def _clean_evidence(
        samples: NDArray[np.float32],
        spans: tuple[_LocalSpan, ...],
        island_count: int,
    ) -> dict[int, _Evidence]:
        evidence: dict[int, _Evidence] = {}
        for island in range(island_count):
            clean = []
            has_reliable_span = False
            for span in spans:
                if span.local_id != island or span.overlap:
                    continue
                has_reliable_span = span.end > span.start or has_reliable_span
                start = span.start + _BOUNDARY_GUARD_SAMPLES
                end = span.end - _BOUNDARY_GUARD_SAMPLES
                if end > start:
                    clean.append(samples[start:end])
            clean_samples = sum(len(part) for part in clean)
            joined = (
                np.concatenate(clean)[:_MAX_EMBEDDING_SAMPLES]
                if clean
                else np.empty(0, dtype=np.float32)
            )
            reason = "ready"
            if not clean_samples:
                reason = (
                    "short_evidence"
                    if has_reliable_span
                    else "low_segmentation_confidence"
                )
            elif float(np.sqrt(np.mean(joined * joined))) < _MIN_RMS:
                reason = "low_audio_energy"
            elif clean_samples < _MIN_MATCH_SAMPLES:
                reason = "short_evidence"
            evidence[island] = _Evidence(joined, clean_samples, reason)
        return evidence

    def _extract(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        try:
            return _unit_vector(self._encoder.extract(samples))
        except SpeakerDiarizationError:
            raise
        except Exception as error:
            raise SpeakerDiarizationError("本地声音特征提取失败") from error

    @staticmethod
    def _speech_islands(spans: tuple[_LocalSpan, ...]) -> tuple[_LocalSpan, ...]:
        """A model-local channel is not proof of identity after a long pause.

        Independently verify each speech island. Otherwise a model-missed
        change would mix two people in one embedding and attach a short earlier
        tail to whoever talks for longer later in this ten-second window.
        """
        joined: list[_LocalSpan] = []
        for span in spans:
            if joined:
                previous = joined[-1]
                if (
                    previous.local_id == span.local_id
                    and previous.overlap == span.overlap
                    and 0
                    <= span.start - previous.end
                    <= (0 if span.overlap else _MAX_SAME_VOICE_GAP)
                ):
                    joined[-1] = _LocalSpan(
                        previous.start, span.end, span.local_id, span.overlap
                    )
                    continue
            joined.append(span)
        return tuple(joined)

    def _update_voice(self, identity: str, vector: NDArray[np.float32]) -> None:
        voice = self._voices[identity]
        # This turn already passed identity and cannot-link checks. Keep its
        # independently obtained reference even when its score is only .65:
        # the previous .75 update gate discarded precisely those useful modes.
        # An indirect match cannot recursively move the original evidence.
        if (
            vector is not voice.anchor
            and all(vector is not member for member in voice.members)
            and float(np.dot(vector, voice.anchor)) >= self._threshold
        ):
            count = min(voice.updates, 19)
            centroid = _unit_vector(voice.centroid * count + vector)
            voice = replace(
                voice,
                centroid=centroid,
                updates=count + 1,
                members=(*voice.members[-(_MAX_REFERENCES - 1) :], vector),
            )
        self._voices.pop(identity)
        self._voices[identity] = voice

    def _scores_for(self, vector: NDArray[np.float32]) -> list[tuple[float, str]]:
        return sorted(
            (
                (reference_score(vector, voice.anchor, voice.members), identity)
                for identity, voice in self._voices.items()
            ),
            reverse=True,
        )

    def _identify(
        self,
        vectors: dict[int, NDArray[np.float32]],
        local_ids: dict[int, int],
        *,
        short: bool = False,
        assigned: dict[int, _Decision] | None = None,
        audio_evidence: dict[int, _Evidence] | None = None,
    ) -> dict[int, _Decision]:
        # Short speech may recognize a known voice, but cannot create or update
        # one. Require stronger absolute similarity and candidate separation.
        # The +.05 candidate was checked on a small public four-speaker sample;
        # it is not a universal speaker-verification threshold.
        threshold = (
            max(self._threshold, min(0.95, self._threshold + 0.05))
            if short
            else self._threshold
        )
        margin = (
            max(self._margin, min(0.40, self._margin + 0.05)) if short else self._margin
        )
        proposals: dict[int, tuple[str, float]] = {}
        decisions: dict[int, _Decision] = {}
        new_voices: list[int] = []
        for island, vector in vectors.items():
            scores = self._scores_for(vector)
            best = scores[0][0] if scores else None
            second = scores[1][0] if len(scores) > 1 else None
            if not scores or scores[0][0] < self._threshold - self._margin:
                reason = "match_below_threshold" if short else "pending_voice"
                decisions[island] = _Decision(reason=reason, best=best, second=second)
                new_voices.append(island)
            elif scores[0][0] < threshold:
                decisions[island] = _Decision(
                    reason="match_below_threshold", best=best, second=second
                )
                if not short:
                    new_voices.append(island)
            elif len(scores) > 1 and scores[0][0] - scores[1][0] < margin:
                decisions[island] = _Decision(
                    reason="match_ambiguous", best=best, second=second
                )
            else:
                proposals[island] = (scores[0][1], scores[0][0])
                decisions[island] = _Decision(
                    reason="local_conflict", best=best, second=second
                )

        protected = {
            decision.identity
            for decision in (assigned or {}).values()
            if decision.identity is not None
        }
        occupied = {
            decision.identity: local_ids[island]
            for island, decision in (assigned or {}).items()
            if decision.identity is not None
        }
        # Distinct model-local speakers cannot acquire one global ID. Tied
        # candidates both remain unassigned. Long evidence already established
        # in this window takes precedence over a later short-utterance match.
        for identity in sorted({candidate[0] for candidate in proposals.values()}):
            matches = [
                (score, island)
                for island, (candidate, score) in proposals.items()
                if candidate == identity
            ]
            by_local: dict[int, float] = {}
            for score, island in matches:
                channel = local_ids[island]
                by_local[channel] = max(by_local.get(channel, -1.0), score)
            strongest = sorted(
                ((score, channel) for channel, score in by_local.items()), reverse=True
            )
            if identity in occupied:
                chosen_local = occupied[identity]
            elif len(strongest) > 1 and strongest[0][0] - strongest[1][0] < margin:
                continue
            else:
                chosen_local = strongest[0][1]
            for _, island in matches:
                if local_ids[island] != chosen_local:
                    continue
                evidence = decisions[island]
                decisions[island] = _Decision(
                    identity, "matched", evidence.best, evidence.second, "embedding"
                )
                if short:
                    # Refresh recency without letting a noisy short embedding
                    # change the anchor, centroid, or update count.
                    self._voices[identity] = self._voices.pop(identity)
                else:
                    self._update_voice(identity, vectors[island])
                protected.add(identity)

        created: dict[str, int] = {}
        for island in new_voices:
            vector = vectors[island]
            channel = local_ids[island]
            audio = (audio_evidence or {}).get(island)
            clean_samples = audio.clean_samples if audio else _MIN_CLEAN_SAMPLES
            quality = (
                min(1.0, clean_samples / audio.span_samples)
                if audio and audio.span_samples
                else 0.0
            )
            fresh = [
                (score, identity)
                for score, identity in self._scores_for(vector)
                if created.get(identity) == channel
            ]
            if (
                fresh
                and fresh[0][0] >= self._threshold
                and (len(fresh) == 1 or fresh[0][0] - fresh[1][0] >= self._margin)
            ):
                identity = fresh[0][1]
                decisions[island] = _Decision(
                    identity,
                    "matched",
                    fresh[0][0],
                    fresh[1][0] if len(fresh) > 1 else None,
                    "embedding",
                )
                self._update_voice(identity, vector)
                continue
            # A local channel only rules out contradictions. Two separate long
            # turns must independently support a new voice; a single low score
            # against history no longer allocates another public person.
            candidates = sorted(
                (
                    (float(np.dot(vector, pending.vector)), index)
                    for index, pending in enumerate(self._pending)
                    if (
                        pending.window_index != self._processing_index
                        or pending.channel == channel
                    )
                    and clean_samples + pending.clean_samples >= _MIN_CLEAN_SAMPLES
                    and (not short or min(quality, pending.quality) >= 0.80)
                ),
                reverse=True,
            )
            witness = None
            if candidates:
                candidate = self._pending[candidates[0][1]]
                gate = (
                    max(0.80, self._threshold + 0.10)
                    if short or candidate.clean_samples < _MIN_CLEAN_SAMPLES
                    else max(0.70, self._threshold)
                )
                conflicting = any(
                    self._pending[index].window_index == candidate.window_index
                    and self._pending[index].channel != candidate.channel
                    and candidates[0][0] - score < self._margin
                    for score, index in candidates[1:]
                )
                if candidates[0][0] >= gate and not conflicting:
                    witness = candidate
            decision = decisions[island]
            clean_new_voice = (
                not short
                and quality >= 0.90
                and clean_samples >= _MIN_CLEAN_SAMPLES
                and (
                    decision.best is None
                    or decision.best < self._threshold - self._margin
                )
            )
            if witness is None and not clean_new_voice:
                self._pending = (
                    *self._pending[-(_MAX_PENDING_TURNS - 1) :],
                    _PendingVoice(
                        vector,
                        self._processing_index,
                        island,
                        channel,
                        clean_samples,
                        quality,
                    ),
                )
                continue
            if len(self._voices) >= self._capacity:
                victim = next(
                    (
                        identity
                        for identity in self._voices
                        if identity not in protected
                    ),
                    None,
                )
                if victim is None:
                    evidence = decisions[island]
                    decisions[island] = replace(evidence, reason="capacity")
                    continue
                del self._voices[victim]
            identity = f"声音{self._namespace}-{self._next_identity:02d}"
            self._next_identity += 1
            anchor = witness.vector if witness is not None else vector
            self._voices[identity] = _Voice(anchor, anchor)
            if witness is not None:
                self._update_voice(identity, vector)
                self._pending = tuple(
                    item for item in self._pending if item is not witness
                )
            evidence = decisions[island]
            decisions[island] = _Decision(
                identity,
                "confirmed_voice" if witness else "quality_anchor",
                evidence.best,
                evidence.second,
                "embedding",
            )
            if witness is not None and witness.window_index == self._processing_index:
                previous = decisions.get(witness.island, _Decision())
                decisions[witness.island] = _Decision(
                    identity,
                    "confirmed_voice",
                    previous.best,
                    previous.second,
                    "embedding",
                )
            protected.add(identity)
            created[identity] = channel
        return decisions

    def segment(
        self, pcm: bytes, *, stream_id: str, window_index: int
    ) -> tuple[SpeakerSpan, ...]:
        if (
            not isinstance(pcm, bytes)
            or len(pcm) % 2
            or len(pcm) > WINDOW_SAMPLES * 2
            or not isinstance(stream_id, str)
            or not stream_id
            or type(window_index) is not int
            or window_index < 0
        ):
            raise SpeakerDiarizationError("声音分段输入必须是至多十秒的有序 PCM16")
        digest = hashlib.sha256(pcm).digest()
        if self._stream_id == stream_id and self._last_index is not None:
            if window_index == self._last_index and digest == self._last_digest:
                return self._last_spans
            if window_index <= self._last_index:
                self.clear_boundary()
                raise SpeakerWindowOrderError("声音分段窗口重复或顺序无效")
        # Voice updates, LRU order, namespace, and replay state commit together.
        # In particular, short embedding failure after long-island identification
        # must not leave a half-updated table or consume identity numbers.
        snapshot = (
            self._namespace,
            self._stream_id,
            self._last_index,
            self._last_digest,
            self._last_spans,
            self._voices.copy(),
            self._next_identity,
            self._boundary,
            self._tracks,
            self._tracks.snapshot(),
            self._pending,
            self._recent.copy(),
            self._revisions,
            self._track_aliases,
            self._voice_aliases.copy(),
            self._cannot_link.copy(),
            self._processing_index,
        )
        try:
            result = self._segment_window(pcm, stream_id, window_index)
        except Exception:
            (
                self._namespace,
                self._stream_id,
                self._last_index,
                self._last_digest,
                self._last_spans,
                self._voices,
                self._next_identity,
                self._boundary,
                self._tracks,
                track_state,
                self._pending,
                self._recent,
                self._revisions,
                self._track_aliases,
                self._voice_aliases,
                self._cannot_link,
                self._processing_index,
            ) = snapshot
            self._tracks.restore(track_state)
            raise
        self._last_index, self._last_digest, self._last_spans = (
            window_index,
            digest,
            result,
        )
        return result

    def _segment_window(
        self, pcm: bytes, stream_id: str, window_index: int
    ) -> tuple[SpeakerSpan, ...]:
        if self._stream_id != stream_id or (
            self._last_index is not None
            and window_index - self._last_index > _MAX_GAP_WINDOWS + 1
        ):
            self.reset()
            self._stream_id = stream_id
        elif self._last_index is not None and window_index != self._last_index + 1:
            # Do not mutate the previous PCM cache before this transaction has
            # committed; failed inference must restore that exact old state.
            self._boundary = SpeakerBoundaryTracker()
            self._tracks.clear(keep_canonical=True)
            self._pending = ()
            self._recent.clear()
        self._processing_index = window_index
        self._revisions, self._track_aliases = (), ()
        self._pending = tuple(
            item
            for item in self._pending
            if window_index - item.window_index < _RECENT_WINDOWS
        )
        self._recent = {
            index: item
            for index, item in self._recent.items()
            if window_index - index < _RECENT_WINDOWS
        }
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        if not len(samples) or float(np.sqrt(np.mean(samples * samples))) < _MIN_RMS:
            self.clear_boundary()
            return ()
        scores = self._infer(samples)
        local = self._speech_islands(self._local_spans(scores, len(samples)))
        reliable = self._local_spans(scores, len(samples), reliable_only=True)
        local_ids = {
            index: turn.local_id
            for index, turn in enumerate(local)
            if turn.local_id is not None
        }
        # Keep each island's clean evidence separate even if the neural output
        # reuses one local channel after a pause or a model-missed voice change.
        embedding_spans = tuple(
            _LocalSpan(max(turn.start, part.start), min(turn.end, part.end), index)
            for index, turn in enumerate(local)
            if turn.local_id is not None and not turn.overlap
            for part in reliable
            if part.local_id == turn.local_id
            and not part.overlap
            and part.start < turn.end
            and part.end > turn.start
        )
        evidence = self._clean_evidence(samples, embedding_spans, len(local))
        evidence = {
            index: replace(item, span_samples=local[index].end - local[index].start)
            for index, item in evidence.items()
        }
        long_vectors = {
            index: self._extract(item.samples)
            for index, item in evidence.items()
            if item.reason == "ready" and item.clean_samples >= _MIN_CLEAN_SAMPLES
        }
        decisions = self._identify(long_vectors, local_ids, audio_evidence=evidence)
        # A short first utterance may match a long utterance later in this same
        # window, but only after independently extracting and matching its voice.
        # Keep an independently usable short reference even before the first
        # person is confirmed, so a later long turn can resolve it retroactively.
        short_vectors = {
            index: self._extract(item.samples)
            for index, item in evidence.items()
            if item.reason == "ready"
            and _MIN_MATCH_SAMPLES <= item.clean_samples < _MIN_CLEAN_SAMPLES
        }
        decisions.update(
            self._identify(
                short_vectors,
                local_ids,
                short=True,
                assigned=decisions,
                audio_evidence=evidence,
            )
        )
        vectors = {**long_vectors, **short_vectors}
        for identity in sorted(
            {
                decision.identity
                for decision in decisions.values()
                if decision.identity is not None
            }
        ):
            voice = self._voices[identity]
            self._tracks.install_identity(
                identity, voice.anchor, voice.members, window_index=window_index
            )
        tracks = self._tracks.assign(
            vectors,
            {index: item.clean_samples for index, item in evidence.items()},
            local_ids,
            {
                index: decision.identity
                for index, decision in decisions.items()
                if decision.identity is not None
            },
            window_index=window_index,
        )
        self._apply_boundary(
            pcm,
            stream_id,
            window_index,
            local,
            reliable,
            decisions,
            vectors,
            evidence,
            tracks,
        )
        result = self._join_spans(
            tuple(
                self._speaker_span(
                    span,
                    index,
                    window_index,
                    evidence[index],
                    decisions.get(index),
                    tracks.get(index),
                )
                for index, span in enumerate(local)
            )
        )
        self._recent[window_index] = _RecentWindow(result, vectors, local_ids)
        result = self._reconcile_recent(window_index)
        self._boundary.remember(
            pcm,
            stream_id=stream_id,
            window_index=window_index,
            turns=local,
            reliable=reliable,
            identities={
                index: span.speaker_id
                for index, span in enumerate(result)
                if span.speaker_id is not None
            },
            tracks={
                index: span.track_id
                for index, span in enumerate(result)
                if span.track_id is not None
            },
        )
        return result

    def _canonical_voice(self, identity: str | None) -> str | None:
        while identity in self._voice_aliases:
            identity = self._voice_aliases[identity]
        return identity

    def _reconcile_recent(self, current_index: int) -> tuple[SpeakerSpan, ...]:
        """Reconsider cached embeddings, never re-decode or extend old audio."""
        original = self._recent.copy()
        for window_index, record in tuple(self._recent.items()):
            spans = list(record.spans)
            for short in (False, True):
                proposals: dict[str, list[tuple[float, int, float | None]]] = {}
                threshold = (
                    max(self._threshold, min(0.95, self._threshold + 0.05))
                    if short
                    else self._threshold
                )
                margin = (
                    max(self._margin, min(0.40, self._margin + 0.05))
                    if short
                    else self._margin
                )
                for index, vector in record.vectors.items():
                    span = spans[index]
                    if (
                        span.speaker_id
                        or span.overlap
                        or (span.clean_seconds < 1.2) != short
                    ):
                        continue
                    scores = self._scores_for(vector)
                    if not scores or scores[0][0] < threshold:
                        continue
                    second = scores[1][0] if len(scores) > 1 else None
                    if second is not None and scores[0][0] - second < margin:
                        continue
                    proposals.setdefault(scores[0][1], []).append(
                        (scores[0][0], index, second)
                    )
                for identity, candidates in proposals.items():
                    occupied = {
                        record.channels[index]
                        for index, span in enumerate(spans)
                        if self._canonical_voice(span.speaker_id) == identity
                        and index in record.channels
                    }
                    by_channel: dict[int, float] = {}
                    for score, index, _ in candidates:
                        channel = record.channels[index]
                        by_channel[channel] = max(score, by_channel.get(channel, -1))
                    ranked = sorted(
                        ((score, channel) for channel, score in by_channel.items()),
                        reverse=True,
                    )
                    if len(occupied) > 1:
                        continue
                    if occupied:
                        chosen = next(iter(occupied))
                    elif len(ranked) > 1 and ranked[0][0] - ranked[1][0] < margin:
                        continue
                    else:
                        chosen = ranked[0][1]
                    track_id = self._tracks.canonical.get(identity)
                    for score, index, second in candidates:
                        if record.channels[index] != chosen:
                            continue
                        spans[index] = replace(
                            spans[index],
                            speaker_id=identity,
                            track_id=track_id,
                            reason="retrospective_match",
                            track_reason="retrospective_match",
                            identity_source="embedding",
                            best_similarity=score,
                            second_similarity=second,
                        )
            self._recent[window_index] = replace(record, spans=tuple(spans))
        # Retain cannot-link facts beyond the short transcript cache, bounded
        # by the live voice table. A later similar excerpt cannot erase an
        # independently observed distinction between two speakers.
        for record in self._recent.values():
            assigned = [
                (self._canonical_voice(span.speaker_id), record.channels[index])
                for index, span in enumerate(record.spans)
                if span.speaker_id is not None and index in record.channels
            ]
            for left, left_channel in assigned:
                for right, right_channel in assigned:
                    if (
                        left is not None
                        and right is not None
                        and left != right
                        and left_channel != right_channel
                    ):
                        self._cannot_link.add(frozenset((left, right)))
        self._cannot_link = {
            pair for pair in self._cannot_link if pair <= self._voices.keys()
        }
        self._merge_voices()
        revisions = []
        for window_index, record in tuple(self._recent.items()):
            spans = []
            for before, span in zip(
                original[window_index].spans, record.spans, strict=True
            ):
                identity = self._canonical_voice(span.speaker_id)
                if identity is not None and identity != span.speaker_id:
                    span = replace(
                        span,
                        speaker_id=identity,
                        track_id=self._tracks.canonical.get(identity),
                        reason="retrospective_merge",
                        track_reason="retrospective_merge",
                    )
                spans.append(span)
                if window_index != current_index and (
                    before.speaker_id,
                    before.track_id,
                ) != (span.speaker_id, span.track_id):
                    revisions.append(
                        SpeakerRevision(
                            window_index,
                            span.start_sample,
                            span.end_sample,
                            span.speaker_id,
                            span.track_id,
                            span.track_reason or "retrospective_match",
                        )
                    )
            self._recent[window_index] = replace(record, spans=tuple(spans))
        self._revisions = tuple(revisions)
        self._pending = tuple(
            item
            for item in self._pending
            if item.window_index not in self._recent
            or self._recent[item.window_index].spans[item.island].speaker_id is None
        )
        return self._recent[current_index].spans

    def _merge_voices(self) -> None:
        """Merge only mutually supported templates without past local conflict."""
        keys = sorted(
            self._voices,
            key=lambda key: (
                (0, int(key.rsplit("-", 1)[-1]))
                if key.rsplit("-", 1)[-1].isdigit()
                else (1, key)
            ),
        )
        for offset, left_id in enumerate(keys):
            for right_id in keys[offset + 1 :]:
                if left_id not in self._voices or right_id not in self._voices:
                    continue
                if frozenset((left_id, right_id)) in self._cannot_link:
                    continue
                left, right = self._voices[left_id], self._voices[right_id]
                if len(left.members) < 2 or len(right.members) < 2:
                    continue
                # Both original anchors need support in the other person's
                # accepted references. One lucky pair cannot merge identities.
                if (
                    min(
                        reference_score(left.anchor, right.anchor, right.members),
                        reference_score(right.anchor, left.anchor, left.members),
                    )
                    < self._threshold
                ):
                    continue
                gate = max(0.75, self._threshold + self._margin)
                links = [
                    (i, j)
                    for i, a in enumerate((left.anchor, *left.members))
                    for j, b in enumerate((right.anchor, *right.members))
                    if float(np.dot(a, b)) >= gate
                ]
                if not any(
                    i != k and j != other_j for i, j in links for k, other_j in links
                ):
                    continue
                left_roots, right_roots = (
                    (left.anchor, *left.roots),
                    (right.anchor, *right.roots),
                )
                if len(left_roots) + len(right_roots) > 8 or any(
                    float(np.dot(a, b)) < self._threshold - self._margin
                    for a in left_roots
                    for b in right_roots
                ):
                    # Preserve every original anchor as a merge constraint.
                    # A -> B -> C similarity chain cannot gradually erase A.
                    continue
                # Do not resolve a tie involving a third established voice by
                # arbitrarily merging the first pair encountered.
                pair_score = min(
                    reference_score(left.anchor, right.anchor, right.members),
                    reference_score(right.anchor, left.anchor, left.members),
                )
                rivals = [
                    max(
                        reference_score(left.anchor, other.anchor, other.members),
                        reference_score(right.anchor, other.anchor, other.members),
                    )
                    for key, other in self._voices.items()
                    if key not in (left_id, right_id)
                ]
                if rivals and pair_score - max(rivals) < self._margin:
                    continue
                alias = self._tracks.merge_identities(right_id, left_id)
                if alias is not None:
                    self._track_aliases = (*self._track_aliases, alias)
                self._voice_aliases[right_id] = left_id
                for key, value in tuple(self._voice_aliases.items()):
                    if value == right_id:
                        self._voice_aliases[key] = left_id
                while len(self._voice_aliases) > self._capacity * 2:
                    del self._voice_aliases[next(iter(self._voice_aliases))]
                del self._voices[right_id]
                self._voices[left_id] = replace(left, roots=(*left.roots, *right_roots))
                for reference in (right.anchor, *right.members):
                    self._update_voice(left_id, reference)
                self._cannot_link = {
                    frozenset(left_id if item == right_id else item for item in pair)
                    for pair in self._cannot_link
                }

    def _apply_boundary(
        self,
        pcm: bytes,
        stream_id: str,
        window_index: int,
        local: tuple[_LocalSpan, ...],
        reliable: tuple[_LocalSpan, ...],
        decisions: dict[int, _Decision],
        vectors: dict[int, NDArray[np.float32]],
        evidence: dict[int, _Evidence],
        tracks: dict[int, TrackMatch],
    ) -> None:
        # A confidently identified current turn needs no extra segmentation.
        # In particular, continuity cannot overwrite a contradictory embedding.
        first = decisions.get(0)
        if (
            not local
            or local[0].overlap
            or local[0].local_id is None
            or evidence[0].clean_samples >= _MIN_CLEAN_SAMPLES
            or evidence[0].reason == "low_audio_energy"
        ):
            return
        first_track = tracks.get(0, TrackMatch())
        if first_track.track_id is not None and first is not None and first.identity:
            return
        check = self._boundary.resolve(
            pcm,
            stream_id=stream_id,
            window_index=window_index,
            turns=local,
            reliable=reliable,
            infer=lambda samples: self._local_spans(
                self._infer(samples), WINDOW_SAMPLES, reliable_only=True
            ),
        )
        # Preserve why a supplementary check did not recover an identity; an
        # inference failure must remain distinguishable from an unconfirmed seam.
        decisions[0] = replace(
            decisions.get(
                0,
                _Decision(
                    reason=evidence[0].reason
                    if evidence[0].reason != "ready"
                    else "short_evidence"
                ),
            ),
            boundary_reason=check.reason,
        )
        for island, identity in check.proposals.items():
            decision = decisions.get(island, _Decision())
            if identity not in self._voices:
                decisions[island] = replace(
                    decision, boundary_reason="identity_unavailable"
                )
                continue
            if decision.identity is not None:
                continue
            if any(
                other.identity == identity
                and local[index].local_id != local[island].local_id
                for index, other in decisions.items()
            ):
                decisions[island] = _Decision(
                    reason="local_conflict",
                    best=decision.best,
                    second=decision.second,
                    boundary_reason="local_conflict",
                )
                continue
            vector = vectors.get(island)
            if vector is not None:
                voice = self._voices[identity]
                candidate_score = reference_score(vector, voice.anchor, voice.members)
                alternatives = [
                    score
                    for score, candidate in self._scores_for(vector)
                    if candidate != identity
                ]
                if candidate_score < self._threshold or (
                    alternatives and candidate_score - alternatives[0] < self._margin
                ):
                    # Spatial continuity can supplement the stricter short
                    # thresholds only when ordinary matching also selects this
                    # same candidate unambiguously. It cannot overrule a better
                    # competing voice or a close tie between known templates.
                    decisions[island] = replace(
                        decision, boundary_reason="embedding_conflict"
                    )
                    continue
            decisions[island] = _Decision(
                identity,
                "boundary_continuity",
                decision.best,
                decision.second,
                "boundary",
                check.reason,
            )
            self._voices[identity] = self._voices.pop(identity)
        for island, track_id in check.track_proposals.items():
            if tracks.get(island, TrackMatch()).track_id is not None:
                continue
            decision = decisions.get(island, _Decision())
            if decision.boundary_reason in {
                "embedding_conflict",
                "local_conflict",
            }:
                tracks[island] = TrackMatch(reason=decision.boundary_reason)
                continue
            # A provisional track may outlive recognition ambiguity, but may
            # never bypass an independently established contradictory identity.
            tracks[island] = self._tracks.inherit(
                track_id,
                island=island,
                channels={
                    index: turn.local_id
                    for index, turn in enumerate(local)
                    if turn.local_id is not None
                },
                matches=tracks,
                vector=vectors.get(island),
                identity=decision.identity,
                window_index=window_index,
            )

    def _speaker_span(
        self,
        span: _LocalSpan,
        index: int,
        window_index: int,
        evidence: _Evidence,
        decision: _Decision | None,
        track: TrackMatch | None = None,
    ) -> SpeakerSpan:
        if decision is None:
            reason = (
                "overlap"
                if span.overlap
                else "low_segmentation_confidence"
                if span.local_id is None
                else "short_evidence"
                if evidence.reason == "ready"
                else evidence.reason
            )
            decision = _Decision(reason=reason)
        local_id = (
            f"local:{self._namespace}:{window_index}:{span.local_id}:{index}"
            if span.local_id is not None and not span.overlap
            else None
        )
        return SpeakerSpan(
            span.start,
            span.end,
            decision.identity,
            span.overlap,
            local_id,
            decision.reason,
            evidence.clean_samples / SAMPLE_RATE,
            decision.best,
            decision.second,
            decision.source
            if decision.identity is not None
            else "local"
            if local_id
            else "unknown",
            decision.boundary_reason,
            track.track_id if track is not None else None,
            track.reason if track is not None else None,
        )
