"""Opt-in, local-only diagnostics for the native audio companion.

The diagnostic observer is deliberately independent from the audio and answer
state machines. It keeps a small in-memory window while audio is running and
only writes snapshots asynchronously after a user question is complete, an
interruption has been confirmed, or a TTS playback interval has ended.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import shutil
import time
import wave
from array import array
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4

from fish_memory.companion.audio_processing import (
    SAMPLE_RATE,
    speech_pre_roll_ms_to_frames,
)
from fish_memory.companion.latency import summarize_turn_latency

logger = logging.getLogger(__name__)

VOICE_DRAFT_EVENT_PREFIX = "voice_draft_"
_VOICE_DRAFT_INTEGER_FIELDS = frozenset(
    {
        "revision",
        "attemptRevision",
        "candidateEpoch",
        "speechGeneration",
        "bindingGeneration",
        "charCount",
        "utf8Bytes",
        "tokenCount",
        "pcmBytes",
        "discardedBytes",
        "queueDepth",
        "droppedFrames",
        "cleanupCount",
        "throughRevision",
        "submittedChars",
        "flushOffset",
        "gateFrames",
        "gateSpeechFrames",
        "gateSilentFrames",
        "gateLongestSilenceFrames",
        "gateRequiredSilenceFrames",
        "gateCandidateCount",
        "gateResumedCount",
    }
)
_VOICE_DRAFT_UUID_FIELDS = frozenset({"turnId"})
_VOICE_DRAFT_NUMBER_FIELDS = frozenset({"queueAgeMs"})
_VOICE_DRAFT_BOOLEAN_FIELDS = frozenset(
    {"promoted", "accepted", "providerReady", "firstContent", "decodeStarted"}
)
_VOICE_DRAFT_REASONS = frozenset(
    {
        "binding_changed",
        "audio_frame_drop",
        "audio_queue_age",
        "buffer_limit",
        "closed",
        "empty_transcript",
        "invalid_event",
        "promoted",
        "promotion_miss",
        "provider_error",
        "range_mismatch",
        "rejected",
        "non_speech_event",
        "nonlexical_vocalization",
        "request_rejected",
        "revision_mismatch",
        "shutdown",
        "speech_resumed",
        "stale_owner",
        "stream_ended",
        "decode_busy",
        "fast_vad",
        "primary_start",
        "primary_inference",
        "primary_inference_tail",
        "speech_end",
        "committed",
        "committed_tts_active",
        "abandoned",
        "snapshot_unavailable",
        "empty_pcm",
        "promotion_disabled",
        "snapshot_type",
        "no_matching_decode",
        "pcm_prefix_mismatch",
        "decode_not_started",
        "decode_invalidated",
        "reuse_invalidated",
        "reuse_not_available",
        "speech_generation_changed",
        "frame_in_flight",
        "capture_vad_unavailable",
        "owner_pending_speech",
        "capture_speech",
        "pcm_track_changed",
        "primary_vad_tail",
        "fast_vad_tail",
    }
)


def privacy_safe_event_fields(
    name: str,
    *,
    text: str | None,
    metadata: Mapping[str, object] | None,
) -> tuple[str | None, Mapping[str, object] | None]:
    """Remove speculative content before any diagnostic observer can retain it."""
    if not name.startswith(VOICE_DRAFT_EVENT_PREFIX):
        return text, metadata
    safe: dict[str, object] = {}
    for key, value in (metadata or {}).items():
        if key in _VOICE_DRAFT_UUID_FIELDS and _strict_uuid_string(value):
            safe[key] = value
        elif (
            key in _VOICE_DRAFT_INTEGER_FIELDS
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            safe[key] = value
        elif (
            key in _VOICE_DRAFT_NUMBER_FIELDS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0
        ):
            safe[key] = float(value)
        elif key in _VOICE_DRAFT_BOOLEAN_FIELDS and isinstance(value, bool):
            safe[key] = value
        elif (
            key == "reason" and isinstance(value, str) and value in _VOICE_DRAFT_REASONS
        ):
            safe[key] = value
    return None, safe or None


def _strict_uuid_string(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def record_voice_draft_event(
    observer: AudioDiagnosticsObserver | None,
    name: str,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Send one privacy-sanitized draft lifecycle event to the shared observer."""
    if observer is None:
        return
    if not name.startswith(VOICE_DRAFT_EVENT_PREFIX):
        raise ValueError("voice draft event name must use the voice_draft_ prefix")
    _, safe_metadata = privacy_safe_event_fields(
        name,
        text=None,
        metadata=metadata,
    )
    try:
        observer.record_event(
            name,
            source="voice_draft",
            metadata=safe_metadata,
        )
    except Exception:
        logger.debug("Unable to record voice draft diagnostic event", exc_info=True)


class AudioDiagnosticsObserver(Protocol):
    def record_frame(
        self,
        capture_pcm: bytes,
        render_pcm: bytes,
        cleaned_pcm: bytes,
        *,
        voice_detected: bool | None = None,
        processing_ms: float | None = None,
        dropped_frames: int | None = None,
    ) -> int: ...

    def record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> None: ...


class _AudioDiagnosticsSettings(Protocol):
    audio_diagnostics_enabled: bool
    audio_diagnostics_dir: str
    audio_diagnostics_retention_days: int
    audio_diagnostics_pre_seconds: float
    audio_diagnostics_post_seconds: float
    companion_aec_delay_ms: int
    companion_aec_delay_mode: Literal["fixed", "auto"]
    companion_vad_start_frames: int
    companion_vad_threshold: float
    companion_vad_min_silence_ms: int
    companion_vad_min_speech_ms: int
    companion_vad_window_size: int
    companion_vad_max_speech_seconds: float
    companion_vad_num_threads: int

    @property
    def asr_model(self) -> str: ...

    asr_num_threads: int
    companion_audio_block_size: int


@dataclass(frozen=True, slots=True)
class AudioDiagnosticsConfig:
    enabled: bool = False
    directory: Path = Path(".cache/fish-memory/audio-diagnostics")
    retention_days: int = 7
    pre_seconds: float = 5.0
    post_seconds: float = 5.0
    sample_rate: int = 16_000
    frame_ms: int = 10
    queue_size: int = 8
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if self.retention_days < 1:
            raise ValueError("retention_days must be positive")
        if self.pre_seconds < 0 or self.post_seconds < 0:
            raise ValueError("diagnostic windows cannot be negative")
        if self.sample_rate < 1 or self.frame_ms < 1:
            raise ValueError("audio format must be positive")
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")

    @classmethod
    def from_settings(
        cls, settings: _AudioDiagnosticsSettings
    ) -> AudioDiagnosticsConfig:
        return cls(
            enabled=settings.audio_diagnostics_enabled,
            directory=Path(settings.audio_diagnostics_dir),
            retention_days=settings.audio_diagnostics_retention_days,
            pre_seconds=settings.audio_diagnostics_pre_seconds,
            post_seconds=settings.audio_diagnostics_post_seconds,
            parameters={
                "aecDelayMode": settings.companion_aec_delay_mode,
                "aecDelayMs": (
                    settings.companion_aec_delay_ms
                    if settings.companion_aec_delay_mode == "fixed"
                    else None
                ),
                "vadBackend": "silero",
                "vadStartFrames": 1,
                "vadThreshold": settings.companion_vad_threshold,
                "vadMinSilenceMs": settings.companion_vad_min_silence_ms,
                "vadMinSpeechMs": settings.companion_vad_min_speech_ms,
                "vadSpeechPreRollFrames": speech_pre_roll_ms_to_frames(
                    settings.companion_vad_min_speech_ms,
                    window_ms=math.ceil(
                        settings.companion_vad_window_size * 1000 / SAMPLE_RATE
                    ),
                ),
                "vadWindowSize": settings.companion_vad_window_size,
                "vadMaxSpeechSeconds": settings.companion_vad_max_speech_seconds,
                "vadNumThreads": settings.companion_vad_num_threads,
                "asrModel": settings.asr_model,
                "asrNumThreads": settings.asr_num_threads,
                "audioBlockSize": settings.companion_audio_block_size,
            },
        )


@dataclass(frozen=True, slots=True)
class DiagnosticFrame:
    frame_index: int
    at_ns: int
    capture_pcm: bytes
    render_pcm: bytes
    cleaned_pcm: bytes
    voice_detected: bool | None
    processing_ms: float | None
    dropped_frames: int | None
    capture_rms: float
    capture_peak: int
    render_rms: float
    render_peak: int
    cleaned_rms: float
    cleaned_peak: int


@dataclass(slots=True)
class _ActiveSnapshot:
    event_id: str
    start_ns: int
    kind: str
    source_message_id: str | None
    frames: list[DiagnosticFrame]
    timeline: list[dict[str, object]]
    remaining_post_frames: int
    question_start_frame_index: int | None = None
    question_end_frame_index: int | None = None
    has_asr_text: bool = False


@dataclass(frozen=True, slots=True)
class _Snapshot:
    event_id: str
    start_ns: int
    kind: str
    source_message_id: str | None
    frames: tuple[DiagnosticFrame, ...]
    timeline: tuple[dict[str, object], ...]
    question_start_frame_index: int | None = None
    question_end_frame_index: int | None = None


class AudioDiagnostics:
    """Non-blocking observer and bounded local audio-event writer."""

    def __init__(self, config: AudioDiagnosticsConfig) -> None:
        self.config = config
        self._run_id = str(uuid4())
        ring_frames = max(
            1,
            math.ceil(config.pre_seconds * 1000 / config.frame_ms),
        )
        self._ring: deque[DiagnosticFrame] = deque(maxlen=ring_frames)
        self._frame_index = 0
        self._timeline_start_ns: int | None = None
        self._timeline: list[dict[str, object]] = []
        self._active_snapshot: _ActiveSnapshot | None = None
        self._ready_snapshots: deque[_Snapshot] = deque()
        self._queue: asyncio.Queue[_Snapshot | None] = asyncio.Queue(
            maxsize=config.queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._runtime_failed = False
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    @property
    def run_id(self) -> str:
        return self._run_id

    async def start(self) -> None:
        if not self.config.enabled or self._worker_task is not None:
            return
        try:
            self.config.directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._runtime_failed = True
            logger.exception("Unable to create audio diagnostics directory")
            return
        self._cleanup_task = asyncio.create_task(
            asyncio.to_thread(self._cleanup_expired)
        )
        self._worker_task = asyncio.create_task(self._writer_loop())
        self._drain_ready_snapshots()

    async def close(self) -> None:
        if not self.config.enabled:
            return
        if self._active_snapshot is not None:
            if (
                self._active_snapshot.kind != "question"
                or self._active_snapshot.has_asr_text
            ):
                self._finalize_active_snapshot()
            else:
                self._active_snapshot = None
        self._drain_ready_snapshots()
        worker = self._worker_task
        if worker is not None:
            try:
                await self._queue.put(None)
                await worker
            except Exception:
                logger.exception("Unable to close audio diagnostics writer")
            finally:
                self._worker_task = None
        cleanup = self._cleanup_task
        self._cleanup_task = None
        if cleanup is not None:
            try:
                await cleanup
            except Exception:
                logger.exception("Unable to finish audio diagnostics cleanup")

    def record_frame(
        self,
        capture_pcm: bytes,
        render_pcm: bytes,
        cleaned_pcm: bytes,
        *,
        voice_detected: bool | None = None,
        processing_ms: float | None = None,
        dropped_frames: int | None = None,
    ) -> int:
        if not self.config.enabled or self._runtime_failed:
            return -1
        now_ns = time.monotonic_ns()
        frame = DiagnosticFrame(
            frame_index=self._frame_index,
            at_ns=now_ns,
            capture_pcm=bytes(capture_pcm),
            render_pcm=bytes(render_pcm),
            cleaned_pcm=bytes(cleaned_pcm),
            voice_detected=voice_detected,
            processing_ms=processing_ms,
            dropped_frames=dropped_frames,
            capture_rms=_pcm_rms(capture_pcm),
            capture_peak=_pcm_peak(capture_pcm),
            render_rms=_pcm_rms(render_pcm),
            render_peak=_pcm_peak(render_pcm),
            cleaned_rms=_pcm_rms(cleaned_pcm),
            cleaned_peak=_pcm_peak(cleaned_pcm),
        )
        self._frame_index += 1
        self._ring.append(frame)
        active = self._active_snapshot
        if active is not None:
            active.frames.append(frame)
            if active.kind == "interruption":
                active.remaining_post_frames -= 1
                if active.remaining_post_frames <= 0:
                    self._finalize_active_snapshot()
        return frame.frame_index

    def record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not self.config.enabled or self._runtime_failed:
            return
        text, metadata = privacy_safe_event_fields(
            name,
            text=text,
            metadata=metadata,
        )
        now_ns = time.monotonic_ns()
        if name == "speech_start" and self._active_snapshot is None:
            self._timeline = []
            self._timeline_start_ns = now_ns
        if self._timeline_start_ns is None:
            self._timeline_start_ns = now_ns
        event: dict[str, object] = {
            "name": name,
            "atMs": round((now_ns - self._timeline_start_ns) / 1_000_000, 3),
            "source": source,
        }
        if frame_index is not None and frame_index >= 0:
            event["frameIndex"] = frame_index
        if text:
            event["text"] = text
        if metadata:
            event["metadata"] = dict(metadata)
        self._timeline.append(event)

        if name == "tts_playback_started":
            if self._active_snapshot is not None:
                if (
                    self._active_snapshot.kind == "question"
                    and not self._active_snapshot.has_asr_text
                ):
                    self._active_snapshot = None
                else:
                    self._finalize_active_snapshot()
            self._start_snapshot(
                kind="playback",
                start_ns=now_ns,
                source_message_id=(
                    self._source_message_id(metadata)
                    or self._latest_answer_message_id()
                ),
            )
            return

        active = self._active_snapshot
        if active is not None:
            active.timeline.append(event)
            if active.kind == "playback":
                if name == "tts_playback_stopped":
                    self._finalize_active_snapshot()
                return
            if name == "speech_end" and active.question_start_frame_index is not None:
                active.question_end_frame_index = frame_index
            if name == "interruption_confirmed" and active.kind == "question":
                active.kind = "interruption"
                active.remaining_post_frames = math.ceil(
                    self.config.post_seconds * 1000 / self.config.frame_ms
                )
                if active.remaining_post_frames <= 0:
                    self._finalize_active_snapshot()
            elif name == "asr_final" and active.kind == "question":
                if text and text.strip():
                    active.has_asr_text = True
                else:
                    # A VAD segment without recognized text never becomes an
                    # ask, so it should not create a diagnostic snapshot.
                    self._active_snapshot = None
            elif (
                name == "ask_submitted"
                and active.kind == "question"
                and active.has_asr_text
            ):
                self._finalize_active_snapshot()
            return

        if name == "speech_start":
            self._start_snapshot(
                kind="question",
                start_ns=now_ns,
                question_start_frame_index=self._question_start_frame_index(
                    frame_index,
                    metadata,
                ),
            )
            return
        if name not in {"interruption_confirmed", "tts_playback_stopped"}:
            return
        if name == "tts_playback_stopped":
            return
        self._start_snapshot(
            kind="interruption",
            start_ns=now_ns,
            source_message_id=None,
            remaining_post_frames=math.ceil(
                self.config.post_seconds * 1000 / self.config.frame_ms
            ),
        )
        active = self._active_snapshot
        if active is not None and active.remaining_post_frames <= 0:
            self._finalize_active_snapshot()

    def _latest_answer_message_id(self) -> str | None:
        for event in reversed(self._timeline):
            if event.get("name") != "answer_event_received":
                continue
            metadata = event.get("metadata")
            if isinstance(metadata, Mapping):
                message_id = metadata.get("messageId")
                if isinstance(message_id, str) and message_id:
                    return message_id
        return None

    def _start_snapshot(
        self,
        *,
        kind: str,
        start_ns: int,
        source_message_id: str | None = None,
        remaining_post_frames: int = 0,
        question_start_frame_index: int | None = None,
    ) -> None:
        self._active_snapshot = _ActiveSnapshot(
            event_id=str(uuid4()),
            start_ns=start_ns,
            kind=kind,
            source_message_id=source_message_id,
            frames=list(self._ring),
            timeline=list(self._timeline),
            remaining_post_frames=remaining_post_frames,
            question_start_frame_index=question_start_frame_index,
        )

    def _question_start_frame_index(
        self,
        frame_index: int | None,
        metadata: Mapping[str, object] | None,
    ) -> int | None:
        if frame_index is None or frame_index < 0:
            return None
        pre_roll_frames = 1
        if metadata is not None:
            value = metadata.get("preRollFrames")
            if isinstance(value, int) and value > 0:
                pre_roll_frames = value
        start = frame_index - pre_roll_frames + 1
        if self._ring:
            start = max(start, self._ring[0].frame_index)
        return start

    @staticmethod
    def _source_message_id(
        metadata: Mapping[str, object] | None,
    ) -> str | None:
        if metadata is None:
            return None
        value = metadata.get("messageId")
        return value if isinstance(value, str) and value else None

    def _finalize_active_snapshot(self) -> None:
        active = self._active_snapshot
        if active is None:
            return
        self._active_snapshot = None
        self._ready_snapshots.append(
            _Snapshot(
                event_id=active.event_id,
                start_ns=active.start_ns,
                kind=active.kind,
                source_message_id=active.source_message_id,
                frames=tuple(active.frames),
                timeline=tuple(active.timeline),
                question_start_frame_index=active.question_start_frame_index,
                question_end_frame_index=active.question_end_frame_index,
            )
        )
        self._drain_ready_snapshots()

    def _drain_ready_snapshots(self) -> None:
        if self._worker_task is None:
            return
        while self._ready_snapshots:
            snapshot = self._ready_snapshots[0]
            try:
                self._queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                self._ready_snapshots.popleft()
                self._dropped_events += 1
            else:
                self._ready_snapshots.popleft()

    async def _writer_loop(self) -> None:
        while True:
            snapshot = await self._queue.get()
            try:
                if snapshot is None:
                    return
                await asyncio.to_thread(self._write_snapshot, snapshot)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unable to write audio diagnostics snapshot")
            finally:
                self._queue.task_done()

    def _write_snapshot(self, snapshot: _Snapshot) -> None:
        event_dir = self.config.directory / self._run_id / snapshot.event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(
            event_dir / "capture.wav",
            self._pcm_bytes(snapshot.frames, "capture_pcm"),
            self.config,
        )
        _write_wav(
            event_dir / "render.wav",
            self._pcm_bytes(snapshot.frames, "render_pcm"),
            self.config,
        )
        _write_wav(
            event_dir / "cleaned.wav",
            self._pcm_bytes(snapshot.frames, "cleaned_pcm"),
            self.config,
        )
        question_frames = self._question_frames(snapshot)
        if question_frames:
            _write_wav(
                event_dir / "question.wav",
                self._pcm_bytes(question_frames, "capture_pcm"),
                self.config,
            )
            _write_wav(
                event_dir / "question-cleaned.wav",
                self._pcm_bytes(question_frames, "cleaned_pcm"),
                self.config,
            )
        question_manifest = None
        if question_frames:
            question_manifest = {
                "startFrame": snapshot.question_start_frame_index,
                "endFrame": snapshot.question_end_frame_index,
                "captureFile": "question.wav",
                "cleanedFile": "question-cleaned.wav",
            }
        manifest = {
            "runId": self._run_id,
            "eventId": snapshot.event_id,
            "kind": snapshot.kind,
            "sourceMessageId": snapshot.source_message_id,
            "createdAt": datetime.now(UTC).isoformat(),
            "audio": {
                "sampleRate": self.config.sample_rate,
                "channels": 1,
                "sampleWidth": 2,
                "frameMs": self.config.frame_ms,
                "preSeconds": self.config.pre_seconds,
                "postSeconds": self.config.post_seconds,
                "captureFile": "capture.wav",
                "renderFile": "render.wav",
                "cleanedFile": "cleaned.wav",
            },
            "config": dict(self.config.parameters),
            "question": question_manifest,
            "frames": [
                _frame_manifest(frame, snapshot.start_ns) for frame in snapshot.frames
            ],
            "timeline": list(snapshot.timeline),
            "latencyMs": summarize_turn_latency(snapshot.timeline),
            "metrics": _aggregate_metrics(snapshot.frames),
        }
        temporary = event_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        temporary.replace(event_dir / "manifest.json")

    @staticmethod
    def _pcm_bytes(frames: tuple[DiagnosticFrame, ...], field_name: str) -> bytes:
        return b"".join(getattr(frame, field_name) for frame in frames)

    @staticmethod
    def _question_frames(snapshot: _Snapshot) -> tuple[DiagnosticFrame, ...]:
        start = snapshot.question_start_frame_index
        if start is None:
            return ()
        end = snapshot.question_end_frame_index
        return tuple(
            frame
            for frame in snapshot.frames
            if frame.frame_index >= start and (end is None or frame.frame_index <= end)
        )

    def _cleanup_expired(self) -> None:
        root = self.config.directory
        if not root.exists():
            return
        cutoff = time.time() - self.config.retention_days * 86_400
        for path in root.iterdir():
            if not path.is_dir() or path.name == self._run_id:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    shutil.rmtree(path)
            except Exception:
                logger.exception(
                    "Unable to clean audio diagnostics directory: %s", path
                )


def _pcm_rms(pcm: bytes) -> float:
    if len(pcm) < 2 or len(pcm) % 2:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _pcm_peak(pcm: bytes) -> int:
    if len(pcm) < 2 or len(pcm) % 2:
        return 0
    samples = array("h")
    samples.frombytes(pcm)
    return max((abs(sample) for sample in samples), default=0)


def _frame_manifest(frame: DiagnosticFrame, start_ns: int) -> dict[str, object]:
    return {
        "frameIndex": frame.frame_index,
        "atMs": round((frame.at_ns - start_ns) / 1_000_000, 3),
        "voiceDetected": frame.voice_detected,
        "processingMs": frame.processing_ms,
        "droppedFrames": frame.dropped_frames,
        "captureRms": round(frame.capture_rms, 3),
        "capturePeak": frame.capture_peak,
        "renderRms": round(frame.render_rms, 3),
        "renderPeak": frame.render_peak,
        "cleanedRms": round(frame.cleaned_rms, 3),
        "cleanedPeak": frame.cleaned_peak,
    }


def _aggregate_metrics(frames: tuple[DiagnosticFrame, ...]) -> dict[str, object]:
    return {
        "frameCount": len(frames),
        "droppedFrames": max(
            (frame.dropped_frames or 0 for frame in frames),
            default=0,
        ),
        "maxFrameProcessingMs": max(
            (frame.processing_ms or 0.0 for frame in frames),
            default=0.0,
        ),
        "maxCaptureRms": max((frame.capture_rms for frame in frames), default=0.0),
        "maxRenderRms": max((frame.render_rms for frame in frames), default=0.0),
        "maxCleanedRms": max((frame.cleaned_rms for frame in frames), default=0.0),
    }


def _write_wav(path: Path, pcm: bytes, config: AudioDiagnosticsConfig) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(config.sample_rate)
        output.writeframes(pcm)
