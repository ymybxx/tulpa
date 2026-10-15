"""Audiovisual observations with continuous local speaker context."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError

from fish_memory.clock import (
    format_prompt_timestamp,
    system_local_time,
)
from fish_memory.environment_frames import EnvironmentPerceptionPolicy
from fish_memory.observations.computer_history import ComputerHistoryContextSummary
from fish_memory.observations.environment_history_audio import (
    SystemAudioRevision,
    SystemAudioSegment,
    SystemAudioTranscript,
    _effectively_silent_pcm16,
    format_system_audio_text,
    format_system_audio_transcript,
)
from fish_memory.observations.environment_history_audio import (
    system_audio_words as _audio_words,
)
from fish_memory.observations.environment_history_hierarchy import (
    parent_level_for_period,
    snapshot_levels,
)
from fish_memory.observations.environment_history_models import (
    EnvironmentHistoryWindow,
    SemanticDocument,
    SemanticSnapshot,
    snapshot_id,
)
from fish_memory.observations.environment_history_store import (
    EnvironmentHistoryConflict,
)
from fish_memory.observations.media_questions import requests_media_recap
from fish_memory.observations.models import redact_sensitive_observation_text

logger = logging.getLogger(__name__)

_RECENT_AUDIO_CONTEXT_SECONDS = 60
_RECENT_AUDIO_CACHE_SECONDS = 310
_MAX_RECENT_AUDIO_WINDOWS = 31
_MEDIA_RECAP_SECONDS = 10 * 60
_MEDIA_RECAP_MAX_CHARACTERS = 24_000
_HOUR_CONTEXT_SECONDS = 60 * 60
_HOUR_CACHE_SECONDS = 70 * 60
_HOUR_DETAIL_SECONDS = 60
_ACTIVE_CAPTURE_ANCHOR_SECONDS = 20


class HistoryRepository(Protocol):
    async def get(
        self, fish_id: UUID, snapshot_id: UUID
    ) -> SemanticSnapshot | None: ...
    async def save(
        self, value: SemanticSnapshot, *, evidence: dict[str, object] | None = None
    ) -> SemanticSnapshot: ...
    async def child_snapshots(
        self,
        fish_id: UUID,
        capture_session_id: UUID,
        first_index: int,
        *,
        child_period_seconds: int,
        parent_period_seconds: int,
    ) -> tuple[SemanticSnapshot, ...]: ...
    async def recent_hour_snapshots(
        self,
        fish_id: UUID,
        *,
        since: datetime,
        through: datetime,
    ) -> tuple[SemanticSnapshot, ...]: ...
    async def delete_compacted_windows(
        self,
        fish_id: UUID,
        *,
        ended_before: datetime,
    ) -> int: ...
    async def pending_minute_windows(
        self,
        fish_id: UUID,
        limit: int = 4,
        *,
        since: datetime | None = None,
        through: datetime | None = None,
    ) -> tuple[SemanticSnapshot, ...]: ...


class HistorySummarizer(Protocol):
    async def summarize_window(
        self,
        window: EnvironmentHistoryWindow,
        transcript: SystemAudioTranscript,
        *,
        previous_window: EnvironmentHistoryWindow | None = None,
        continuity_context: str = "",
    ) -> SemanticDocument: ...
    async def merge_snapshots(
        self,
        windows: Sequence[SemanticSnapshot],
        *,
        parent_period_seconds: int,
        previous_document: SemanticDocument | None = None,
    ) -> SemanticDocument: ...
    async def close(self) -> None: ...


class AudioTranscriber(Protocol):
    async def transcribe(self, pcm: bytes) -> SystemAudioTranscript: ...
    async def close(self) -> None: ...


class SummaryLoader(Protocol):
    async def get_latest_context_summary(
        self, fish_id: UUID
    ) -> ComputerHistoryContextSummary | None: ...


class WindowProactivity(Protocol):
    """The companion (Jev) decides whether a finished summary is worth reacting to."""

    async def admit_window_summary(
        self, fish_id: UUID, session_id: UUID | None, document: SemanticDocument
    ) -> bool: ...


class HistoryQueueFull(RuntimeError):
    pass


class HistoryDisabled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Job:
    fish_id: UUID
    window: EnvironmentHistoryWindow
    digest: str
    generation: int
    previous_window: EnvironmentHistoryWindow | None = None


@dataclass(frozen=True, slots=True)
class _QuietVisualState:
    capture_session_id: UUID
    window_index: int
    ended_at: float
    digest: str
    source_id: UUID


def _quiet_visual_digest(window: EnvironmentHistoryWindow) -> str | None:
    """Reuse requires fresh, dense observations of identical pixels and silence.

    Audio may be explicitly disabled; capture failures still cannot qualify.
    Missing frames, subtitle changes and cross-session timestamps never qualify.
    """
    if (
        (
            window.audio_status != "complete"
            and window.audio_gap_reason != "system_audio_disabled"
        )
        or window.audio_gap_reason not in {None, "system_audio_disabled"}
        or len(window.frames)
        < min(5, max(1, round(window.ended_at - window.started_at)))
        or (
            not window.audio_silence
            and not _effectively_silent_pcm16(window.decode_pcm())
        )
    ):
        return None
    first, last = window.frames[0], window.frames[-1]
    if (
        first.observed_at - window.started_at > 1.5
        or window.ended_at - last.observed_at > 1.5
        or any(
            right.observed_at - left.observed_at > 2.5
            for left, right in zip(window.frames, window.frames[1:], strict=False)
        )
    ):
        return None
    identity = (
        first.display_id,
        first.pixel_width,
        first.pixel_height,
        first.jpeg_base64,
    )
    if any(
        (f.display_id, f.pixel_width, f.pixel_height, f.jpeg_base64) != identity
        for f in window.frames[1:]
    ):
        return None
    return hashlib.sha256(repr(identity).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _RecentSystemAudioTranscript:
    """A short-lived ASR result used only by the current process."""

    fish_id: UUID
    capture_session_id: UUID
    window_index: int
    started_at: datetime
    ended_at: datetime
    text: str
    event: str
    completed_at: datetime
    segments: tuple[SystemAudioSegment, ...] = ()


def _empty_counts() -> dict[str, int]:
    return {
        field: 0
        for level in snapshot_levels()
        for field in (level.completed_field, level.failed_field)
    }


class EnvironmentHistoryService:
    def __init__(
        self,
        repository: HistoryRepository,
        summarizer: HistorySummarizer,
        transcriber: AudioTranscriber,
        *,
        now: Callable[[], datetime],
        concurrency: int = 2,
        max_pending: int = 12,
        retry_delays: tuple[float, ...] = (1, 3),
        image_input: Callable[[], Awaitable[str]] | None = None,
    ) -> None:
        self._repository = repository
        self._image_input = image_input
        self._summarizer = summarizer
        self._transcriber = transcriber
        self._now = now
        self._max_pending = max_pending
        self._retry_delays = retry_delays
        self._semaphore = asyncio.Semaphore(concurrency)
        self._quiet_visual: dict[UUID, _QuietVisualState] = {}
        self._last_windows: dict[UUID, EnvironmentHistoryWindow] = {}
        self._enabled: set[UUID] = set()
        self._closed = False
        self._generations: dict[UUID, int] = {}
        self._jobs: dict[UUID, tuple[_Job, asyncio.Task[None]]] = {}
        self._recovery_tasks: dict[UUID, asyncio.Task[None]] = {}
        self._release_task: asyncio.Task[None] | None = None
        # Retired generations remain owned until cancellation has settled.
        self._tasks: set[asyncio.Task[None]] = set()
        self._outcomes: OrderedDict[UUID, tuple[UUID, str, str]] = OrderedDict()
        self._counts: dict[UUID, dict[str, int]] = {}
        self._recent_audio: dict[
            UUID,
            OrderedDict[tuple[UUID, int], _RecentSystemAudioTranscript],
        ] = {}
        self._recent_snapshots: dict[
            UUID,
            OrderedDict[UUID, SemanticSnapshot],
        ] = {}
        self._context_views: dict[UUID, ComputerHistoryContextSummary] = {}
        self._proactivity: WindowProactivity | None = None

    def set_proactivity(self, proactivity: WindowProactivity) -> None:
        self._proactivity = proactivity

    def set_enabled(self, fish_id: UUID, enabled: bool) -> None:
        if self._closed:
            return
        if enabled:
            # EnvironmentFrameCache owns one native capture target. Retire any
            # previous Fish through the same cancellation/cache-release path.
            for other_fish_id in tuple(self._enabled):
                if other_fish_id != fish_id:
                    self.set_enabled(other_fish_id, False)
            if fish_id in self._enabled:
                return
            self._enabled.add(fish_id)
            if self._release_task is not None:
                release_task = self._release_task
                self._release_task = None
                release_task.cancel()
            if fish_id not in self._recovery_tasks:
                generation = self._generations.get(fish_id, 0)
                task = asyncio.create_task(
                    self._recover_history_once(fish_id, generation),
                    name=f"environment-history-recovery-scan-{fish_id}",
                )
                self._recovery_tasks[fish_id] = task
                self._tasks.add(task)
                task.add_done_callback(
                    lambda finished: self._finish_recovery(fish_id, finished)
                )
            return
        self._enabled.discard(fish_id)
        self._quiet_visual.pop(fish_id, None)
        self._last_windows.pop(fish_id, None)
        self._recent_audio.pop(fish_id, None)
        self._recent_snapshots.pop(fish_id, None)
        self._context_views.pop(fish_id, None)
        self._generations[fish_id] = self._generations.get(fish_id, 0) + 1
        recovery = self._recovery_tasks.pop(fish_id, None)
        if recovery is not None:
            recovery.cancel()
        retired_jobs: list[asyncio.Task[None]] = []
        for key, (job, task) in tuple(self._jobs.items()):
            if job.fish_id == fish_id:
                self._jobs.pop(key)
                task.cancel()
                retired_jobs.append(task)
        if not self._enabled and self._release_task is None:
            release = getattr(self._transcriber, "release", None)
            if callable(release):
                task = asyncio.create_task(
                    self._release_transcriber_if_unused(release, tuple(retired_jobs)),
                    name="environment-history-asr-release",
                )
                self._release_task = task
                self._tasks.add(task)
                task.add_done_callback(self._finish_release)

    async def _release_transcriber_if_unused(
        self,
        release: Callable[[], Awaitable[None]],
        retired_jobs: tuple[asyncio.Task[None], ...],
    ) -> None:
        # Let cancellation settle before releasing native resources. The
        # transcriber serializes this behind native work that survived asyncio
        # cancellation and guards a queued release against newer decoding.
        await asyncio.gather(*retired_jobs, return_exceptions=True)
        await asyncio.sleep(0)  # keeps an immediate disable -> enable warm
        if self._closed or self._enabled:
            return
        await release()

    def _finish_release(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if self._release_task is task:
            self._release_task = None

    async def _recover_history_once(self, fish_id: UUID, generation: int) -> None:
        as_of = self._now()
        snapshots: tuple[SemanticSnapshot, ...] = ()
        try:
            snapshots = await self._repository.recent_hour_snapshots(
                fish_id,
                since=as_of - timedelta(seconds=_HOUR_CACHE_SECONDS),
                through=as_of,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self._generation_current(fish_id, generation):
                _log_failure("recovery_hydration", error)
        if not self._generation_current(fish_id, generation):
            return
        for snapshot in snapshots:
            self._remember_snapshot(snapshot, publish=False)
        self._publish_context_view(fish_id, as_of=as_of)
        # Old aggregate records remain readable, but fixed-period summaries and
        # their cleanup/recovery jobs are retired. Session History owns compaction.

    def _finish_recovery(self, fish_id: UUID, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if self._recovery_tasks.get(fish_id) is task:
            self._recovery_tasks.pop(fish_id)

    async def enqueue(self, fish_id: UUID, window: EnvironmentHistoryWindow) -> str:
        generation = self._generations.get(fish_id, 0)
        if not self._generation_current(fish_id, generation):
            raise HistoryDisabled()
        if not window.frames and window.audio_captured_samples == 0:
            # Permission loss or a disconnected capture source is missing
            # evidence, not a scene for the model to invent a summary of.
            return "skipped"
        window = window.model_copy(deep=True)
        key = snapshot_id(
            fish_id,
            window.capture_session_id,
            window.window_index,
            window.period_seconds,
        )
        digest = window.input_digest()
        existing = await self._repository.get(fish_id, key)
        if not self._generation_current(fish_id, generation):
            raise HistoryDisabled()
        if existing is not None:
            if existing.input_digest != digest:
                raise EnvironmentHistoryConflict("snapshot input identity changed")
            self._remember_snapshot(existing)
            return "completed"
        pending = self._jobs.get(key)
        if pending is not None:
            if pending[0].digest != digest:
                raise EnvironmentHistoryConflict("snapshot input identity changed")
            return "pending"
        if len(self._jobs) >= self._max_pending:
            raise HistoryQueueFull()
        previous_window = self._last_windows.get(fish_id)
        if previous_window is not None and (
            previous_window.capture_session_id != window.capture_session_id
            or previous_window.session_id != window.session_id
            or previous_window.window_index + 1 != window.window_index
            or abs(previous_window.ended_at - window.started_at) > 0.001
        ):
            previous_window = None
        self._last_windows[fish_id] = window
        job = _Job(fish_id, window, digest, generation, previous_window)
        task = asyncio.create_task(
            self._run(key, job), name=f"environment-window-{key}"
        )
        self._jobs[key] = (job, task)
        self._tasks.add(task)
        task.add_done_callback(lambda finished: self._finish_job(key, finished))
        return "pending"

    def status(self, fish_id: UUID) -> dict[str, object]:
        return {
            "enabled": fish_id in self._enabled,
            "pendingWindows": sum(
                job.fish_id == fish_id for job, _ in self._jobs.values()
            ),
            **self._counts.get(
                fish_id,
                _empty_counts(),
            ),
        }

    def _current(self, job: _Job) -> bool:
        return self._generation_current(job.fish_id, job.generation)

    def _generation_current(self, fish_id: UUID, generation: int) -> bool:
        return (
            not self._closed
            and fish_id in self._enabled
            and self._generations.get(fish_id, 0) == generation
        )

    def _finish_job(self, key: UUID, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        current = self._jobs.get(key)
        if current is not None and current[1] is task:
            self._jobs.pop(key)

    async def _run(self, key: UUID, job: _Job) -> None:
        try:
            quiet_digest = _quiet_visual_digest(job.window)
            # Coalesce adjacent quiet windows behind a still-running baseline
            # without occupying a worker slot or blocking later speech jobs.
            if quiet_digest is not None and job.window.window_index > 0:
                predecessor = self._jobs.get(
                    snapshot_id(
                        job.fish_id,
                        job.window.capture_session_id,
                        job.window.window_index - 1,
                        job.window.period_seconds,
                    )
                )
                if predecessor is not None:
                    await asyncio.shield(predecessor[1])
            async with self._semaphore:
                if not self._current(job):
                    return
                transcribe_window = getattr(
                    self._transcriber, "transcribe_window", None
                )
                if (
                    job.window.audio_status == "unavailable"
                    and job.window.audio_gap_reason == "system_audio_disabled"
                ):
                    transcript = SystemAudioTranscript(
                        "", "silence", diarization_status="disabled"
                    )
                elif callable(transcribe_window):
                    transcript = await transcribe_window(
                        job.window.decode_pcm(),
                        stream_id=f"{job.fish_id}:{job.window.capture_session_id}",
                        window_index=job.window.window_index,
                        started_at=job.window.started_at,
                        audio_complete=job.window.audio_status == "complete",
                    )
                else:
                    transcript = await self._transcriber.transcribe(
                        job.window.decode_pcm()
                    )
                if not self._current(job):
                    return
                transcript = replace(
                    _redacted_audio_transcript(transcript),
                    revisions=tuple(
                        _redacted_audio_revision(revision)
                        for revision in transcript.revisions
                    ),
                )
                self._remember_audio_transcript(job, transcript)
                if transcript.text.strip():
                    quiet_digest = None
                document: SemanticDocument | None = None
                previous = self._quiet_visual.get(job.fish_id)
                if (
                    quiet_digest is not None
                    and previous is not None
                    and previous.capture_session_id == job.window.capture_session_id
                    and previous.window_index + 1 == job.window.window_index
                    and abs(previous.ended_at - job.window.started_at) <= 0.001
                    and previous.digest == quiet_digest
                ):
                    document = SemanticDocument(
                        summary_version=4, unchanged_from=previous.source_id
                    )
                window = job.window
                if (
                    document is None
                    and window.frames
                    and self._image_input is not None
                    and await self._image_input() == "unsupported"
                ):
                    # A model that refuses images failed every window with the
                    # screen in it; what was heard can still be summarised.
                    window = window.model_copy(update={"frames": ()})
                for attempt in range(len(self._retry_delays) + 1):
                    if document is not None:
                        break
                    if not self._current(job):
                        return
                    try:
                        document = await self._summarizer.summarize_window(
                            window,
                            transcript,
                            **self._visual_reference(job, transcript),
                        )
                        break
                    except Exception as error:
                        if (
                            attempt == len(self._retry_delays)
                            or not self._current(job)
                            or not _transient_model_failure(error)
                        ):
                            raise
                        _log_failure("window_retry", error)
                        await asyncio.sleep(self._retry_delays[attempt])
                if document is None or not self._current(job):
                    return
                if (
                    self._proactivity is not None
                    and not await self._proactivity.admit_window_summary(
                        job.fish_id, job.window.session_id, document
                    )
                ):
                    document = SemanticDocument(summary_version=4)
                # Keep attribution writes off the first-expression path.
                await self._persist_attribution_revisions(job, transcript)
                value = SemanticSnapshot(
                    id=key,
                    fish_id=job.fish_id,
                    capture_session_id=job.window.capture_session_id,
                    window_index=job.window.window_index,
                    period_seconds=job.window.period_seconds,
                    session_id=job.window.session_id,
                    started_at=datetime.fromtimestamp(job.window.started_at, UTC),
                    ended_at=datetime.fromtimestamp(job.window.ended_at, UTC),
                    document=document,
                    input_digest=job.digest,
                    created_at=self._now(),
                    audio_unavailable=transcript.event == "unavailable",
                )
                # A later acoustic window may have resolved this one while its
                # independent visual summary was in flight.
                value = self._resolved_audio_snapshot(value)
                latest_audio = self._recent_audio.get(job.fish_id, {}).get(
                    (job.window.capture_session_id, job.window.window_index)
                )
                attribution_revised = (
                    latest_audio is not None
                    and latest_audio.started_at == value.started_at
                    and latest_audio.segments != transcript.segments
                    and value.document.audio == latest_audio.text
                    and _audio_words(latest_audio.text)
                    == _audio_words(format_system_audio_transcript(transcript))
                )
                if attribution_revised and latest_audio is not None:
                    transcript = replace(transcript, segments=latest_audio.segments)
                stored = await self._repository.save(
                    value,
                    evidence={
                        **(
                            {"sessionId": str(job.window.session_id)}
                            if job.window.session_id
                            else {}
                        ),
                        "frameTimes": [
                            frame.observed_at for frame in job.window.frames
                        ],
                        "audioStatus": job.window.audio_status,
                        "audioCapturedSamples": job.window.audio_captured_samples,
                        "audioGapReason": job.window.audio_gap_reason,
                        "audioSilence": job.window.audio_silence,
                        "visualReuseFrom": str(document.unchanged_from)
                        if document.unchanged_from
                        else None,
                        "audioSource": "system_playback_local_asr",
                        "diarizationStatus": transcript.diarization_status,
                        "alignmentStatus": transcript.alignment_status,
                        "transcriptionTextStatus": (
                            "unavailable"
                            if transcript.event == "unavailable"
                            else "recognized"
                            if transcript.text.strip()
                            else "no_recognized_words"
                        ),
                        **(
                            {
                                "transcriptionUnavailableReason": (
                                    transcript.unavailable_reason
                                )
                            }
                            if transcript.unavailable_reason is not None
                            else {}
                        ),
                        "transcriptionRecoveryStatus": transcript.recovery_status,
                        "recoveredSpeechIntervals": [
                            {"startSeconds": start, "endSeconds": end}
                            for start, end in transcript.recovered_intervals
                        ],
                        "speakerAttributionRevised": attribution_revised,
                        "speakerRevisedWindows": [
                            revision.window_index for revision in transcript.revisions
                        ],
                        "audioSegments": [
                            {
                                "startSeconds": segment.start_seconds,
                                "endSeconds": segment.end_seconds,
                                "speakerId": segment.speaker_id,
                                "overlap": segment.overlap,
                                "localSpeakerId": segment.local_speaker_id,
                                "reason": segment.reason,
                                "cleanSeconds": round(segment.clean_seconds, 3),
                                "bestSimilarity": (
                                    round(segment.best_similarity, 4)
                                    if segment.best_similarity is not None
                                    else None
                                ),
                                "secondSimilarity": (
                                    round(segment.second_similarity, 4)
                                    if segment.second_similarity is not None
                                    else None
                                ),
                                "identitySource": segment.identity_source,
                                "boundaryReason": segment.boundary_reason,
                                "trackId": segment.track_id,
                                "trackReason": segment.track_reason,
                                "alignmentReason": segment.alignment_reason,
                            }
                            for segment in transcript.segments
                        ],
                    },
                )
                if not self._current(job):
                    return
                previous = self._quiet_visual.get(job.fish_id)
                if previous is None or job.window.ended_at >= previous.ended_at:
                    if quiet_digest is not None:
                        self._quiet_visual[job.fish_id] = _QuietVisualState(
                            job.window.capture_session_id,
                            job.window.window_index,
                            job.window.ended_at,
                            quiet_digest,
                            document.unchanged_from or stored.id,
                        )
                    else:
                        self._quiet_visual.pop(job.fish_id, None)
                self._count(job.fish_id, "completedWindows")
                self._remember(key, job, "completed")
                self._remember_snapshot(stored)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not self._current(job):
                return
            self._count(job.fish_id, "failedWindows")
            self._remember(key, job, "failed")
            _log_failure("window", error)

    def _visual_reference(self, job: _Job, transcript: SystemAudioTranscript) -> dict:
        result: dict[str, object] = {}
        context = self._continuity_context(job)
        if context:
            result["continuity_context"] = context
        if job.previous_window is not None and transcript.event == "silence":
            result["previous_window"] = job.previous_window
        return result

    def _continuity_context(self, job: _Job) -> str:
        previous = sorted(
            (
                self._resolved_audio_snapshot(value)
                for value in self._recent_snapshots.get(job.fish_id, {}).values()
                if value.capture_session_id == job.window.capture_session_id
                and value.session_id == job.window.session_id
                and 0 <= job.window.started_at - value.ended_at.timestamp() <= 300
                and value.period_seconds in (5, 10)
                and not value.document.unchanged_from
                and (value.document.audiovisual_summary or value.document.audio)
            ),
            key=lambda value: value.started_at,
        )
        parts: list[str] = []
        for value in reversed(previous[-6:]):
            raw = self._recent_audio.get(job.fish_id, {}).get(
                (value.capture_session_id, value.window_index)
            )
            heard = raw.text if raw is not None else value.document.audio
            text = (
                f"{format_prompt_timestamp(value.started_at)} - "
                f"{format_prompt_timestamp(value.ended_at)}\n"
                f"as: {value.document.audiovisual_summary}\n"
                f"转写（声源标签连续）：{heard}"
            )
            if sum(map(len, parts)) + len(text) > 6_000:
                break
            parts.insert(0, text)
        # ASR may be ready before the previous visual summary. Its original
        # timestamp and labels still provide continuity without serializing LLMs.
        window = job.previous_window
        if window is not None and not any(
            value.window_index == window.window_index for value in previous
        ):
            raw = self._recent_audio.get(job.fish_id, {}).get(
                (window.capture_session_id, window.window_index)
            )
            if raw is not None and raw.text:
                text = (
                    f"{format_prompt_timestamp(raw.started_at)} - "
                    f"{format_prompt_timestamp(raw.ended_at)}\n"
                    f"转写（声源标签连续）：{raw.text}"
                )
                while parts and sum(map(len, parts)) + len(text) > 6_000:
                    parts.pop(0)
                if len(text) <= 6_000:
                    parts.append(text)
        return "\n\n".join(parts)

    def _remember(self, key: UUID, job: _Job, outcome: str) -> None:
        self._outcomes[key] = (job.fish_id, job.digest, outcome)
        while len(self._outcomes) > 256:
            self._outcomes.popitem(last=False)

    def window_status(self, fish_id: UUID, key: UUID) -> str:
        pending = self._jobs.get(key)
        if pending is not None and pending[0].fish_id == fish_id:
            return "pending"
        outcome = self._outcomes.get(key)
        return (
            outcome[2] if outcome is not None and outcome[0] == fish_id else "unknown"
        )

    def _count(self, fish_id: UUID, field: str) -> None:
        counts = self._counts.setdefault(fish_id, _empty_counts())
        counts[field] = counts.get(field, 0) + 1

    async def get_latest_context_summary(
        self, fish_id: UUID
    ) -> ComputerHistoryContextSummary | None:
        generation = self._generations.get(fish_id, 0)
        if not self._generation_current(fish_id, generation):
            return None
        # Rebuild from bounded memory so a quiet source cannot keep an expired
        # head forever. The oldest complete block may cross the nominal cutoff.
        self._publish_context_view(fish_id)
        return self._context_views.get(fish_id)

    async def get_latest_media_context_summary(
        self, fish_id: UUID
    ) -> ComputerHistoryContextSummary | None:
        """Read media channels without legacy/ACS activity or changing Ask's view."""
        return self._build_context_view(fish_id, include_activity=False)

    async def get_context_for_question(
        self, fish_id: UUID, question: str
    ) -> ComputerHistoryContextSummary | None:
        context = await self.get_latest_context_summary(fish_id)
        if context is None or not requests_media_recap(question):
            return context
        now = self._now()
        snapshots = tuple(self._recent_snapshots.get(fish_id, {}).values())
        latest = max(
            (value.ended_at for value in snapshots if value.ended_at <= now),
            default=now,
        )
        as_of = (
            latest
            if (now - latest).total_seconds() <= _ACTIVE_CAPTURE_ANCHOR_SECONDS
            else now
        )
        selected = _select_hour_snapshots(snapshots, as_of=as_of)
        sections = []
        if selected:
            rendered = _render_snapshot_context(selected, include_activity=False)
            if rendered:
                sections.append(rendered)
        # Read original words from the existing bounded snapshot cache. No new
        # model call, PCM replay, or second context load during draft promotion.
        raw = _media_recap_audio(
            snapshots,
            tuple(self._recent_audio.get(fish_id, {}).values()),
            as_of=now,
        )
        if raw:
            sections.append(
                _render_recent_audio_context(raw, title="你听到的原话，供核对前后经过")
            )
        gaps = _render_audio_gaps(snapshots, as_of=now, seconds=_MEDIA_RECAP_SECONDS)
        if gaps:
            sections.append(gaps)
        return replace(context, summary="\n\n".join(sections)) if sections else None

    def _remember_snapshot(
        self,
        value: SemanticSnapshot,
        *,
        publish: bool = True,
    ) -> None:
        value = self._resolved_audio_snapshot(value)
        cache = self._recent_snapshots.setdefault(value.fish_id, OrderedDict())
        cache[value.id] = value
        cache.move_to_end(value.id)
        cutoff = self._now() - timedelta(seconds=_HOUR_CACHE_SECONDS)
        for key, snapshot in tuple(cache.items()):
            if snapshot.ended_at <= cutoff:
                cache.pop(key, None)
        if publish:
            self._publish_context_view(value.fish_id)

    def _publish_context_view(
        self,
        fish_id: UUID,
        *,
        as_of: datetime | None = None,
    ) -> None:
        view = self._build_context_view(fish_id, as_of=as_of)
        if view is None:
            self._context_views.pop(fish_id, None)
        else:
            self._context_views[fish_id] = view

    def _build_context_view(
        self,
        fish_id: UUID,
        *,
        as_of: datetime | None = None,
        include_activity: bool = True,
    ) -> ComputerHistoryContextSummary | None:
        if fish_id not in self._enabled:
            return None
        effective_as_of = as_of or self._now()
        cached_snapshots = tuple(self._recent_snapshots.get(fish_id, {}).values())
        latest_snapshot = max(
            (value for value in cached_snapshots if value.ended_at <= effective_as_of),
            key=lambda value: value.ended_at,
            default=None,
        )
        snapshot_as_of = effective_as_of
        if latest_snapshot is not None:
            age = (effective_as_of - latest_snapshot.ended_at).total_seconds()
            if 0 <= age <= _ACTIVE_CAPTURE_ANCHOR_SECONDS:
                # During active capture, use the immutable completed-window edge.
                # This avoids shredding aligned hierarchy nodes merely because
                # Ask arrived a few milliseconds after a ten-second boundary.
                snapshot_as_of = latest_snapshot.ended_at
        snapshot_values = _select_hour_snapshots(
            cached_snapshots,
            as_of=snapshot_as_of,
        )
        recent_audio = self._recent_audio_context(
            fish_id,
            since=effective_as_of - timedelta(seconds=_RECENT_AUDIO_CONTEXT_SECONDS),
            through=effective_as_of,
        )
        if not snapshot_values and not recent_audio:
            return None
        sections: list[str] = []
        if snapshot_values:
            sections.append(
                _render_snapshot_context(
                    snapshot_values, include_activity=include_activity
                )
            )
        if recent_audio:
            sections.append(_render_recent_audio_context(recent_audio))
        gaps = _render_audio_gaps(
            cached_snapshots, as_of=effective_as_of, seconds=_HOUR_CONTEXT_SECONDS
        )
        if gaps:
            sections.append(gaps)
        summary = "\n\n".join(sections)
        latest_ended_at = max(
            tuple(value.ended_at for value in snapshot_values)
            + tuple(value.ended_at for value in recent_audio)
        )
        latest_completed_at = max(
            tuple(value.created_at for value in snapshot_values)
            + tuple(value.completed_at for value in recent_audio)
        )
        latest_capture_session = max(
            recent_audio,
            key=lambda value: value.ended_at,
            default=None,
        )
        return ComputerHistoryContextSummary(
            summary=summary,
            segment_id=(
                latest_capture_session.capture_session_id
                if latest_capture_session is not None
                and (
                    not snapshot_values
                    or latest_capture_session.ended_at >= snapshot_values[-1].ended_at
                )
                else snapshot_values[-1].capture_session_id
            ),
            through_event_count=len(snapshot_values) + len(recent_audio),
            ended_at=latest_ended_at,
            last_successful_at=latest_completed_at,
            source="observation",
        )

    def _remember_audio_transcript(
        self,
        job: _Job,
        transcript: SystemAudioTranscript,
    ) -> _RecentSystemAudioTranscript:
        cache = self._recent_audio.setdefault(job.fish_id, OrderedDict())
        for revision in transcript.revisions:
            key = (job.window.capture_session_id, revision.window_index)
            previous = cache.get(key)
            if (
                previous is None
                or not math.isfinite(revision.started_at)
                or not 1 <= job.window.window_index - revision.window_index <= 30
                or abs(previous.started_at.timestamp() - revision.started_at) > 0.001
            ):
                continue
            revised_text = format_system_audio_transcript(
                SystemAudioTranscript(revision.text, revision.event, revision.segments)
            )
            # Only speaker attribution may change retrospectively. Never let a
            # correction alter recognized words or cross a capture identity.
            if _audio_words(previous.text) != _audio_words(revised_text):
                continue
            cache[key] = replace(
                previous,
                text=revised_text,
                segments=revision.segments,
                completed_at=self._now(),
            )
        value = _RecentSystemAudioTranscript(
            fish_id=job.fish_id,
            capture_session_id=job.window.capture_session_id,
            window_index=job.window.window_index,
            started_at=datetime.fromtimestamp(job.window.started_at, UTC),
            ended_at=datetime.fromtimestamp(job.window.ended_at, UTC),
            text=format_system_audio_transcript(transcript),
            event=transcript.event,
            completed_at=self._now(),
            segments=transcript.segments,
        )
        if _useful_audio_window(value):
            cache[(value.capture_session_id, value.window_index)] = value
        cutoff = value.completed_at - timedelta(seconds=_RECENT_AUDIO_CACHE_SECONDS)
        for key, item in tuple(cache.items()):
            if item.ended_at <= cutoff:
                cache.pop(key, None)
        while len(cache) > _MAX_RECENT_AUDIO_WINDOWS:
            cache.popitem(last=False)
        snapshots = self._recent_snapshots.get(job.fish_id, {})
        for key, snapshot in tuple(snapshots.items()):
            if snapshot.capture_session_id == job.window.capture_session_id:
                snapshots[key] = self._resolved_audio_snapshot(snapshot)
        self._publish_context_view(job.fish_id)
        return value

    async def _persist_attribution_revisions(
        self, job: _Job, transcript: SystemAudioTranscript
    ) -> None:
        persist = getattr(self._repository, "revise_attribution", None)
        if not callable(persist) or not transcript.revisions:
            return
        changed = {revision.window_index for revision in transcript.revisions}
        cache = self._recent_audio.get(job.fish_id, {})
        for value in tuple(self._recent_snapshots.get(job.fish_id, {}).values()):
            if (
                not self._current(job)
                or value.capture_session_id != job.window.capture_session_id
                or value.period_seconds > 300
                or not any(
                    value.window_index
                    <= index
                    < value.window_index + value.period_seconds // 10
                    for index in changed
                )
            ):
                continue
            recent = cache.get((value.capture_session_id, value.window_index))
            segments = (
                [
                    {
                        "startSeconds": s.start_seconds,
                        "endSeconds": s.end_seconds,
                        "speakerId": s.speaker_id,
                        "trackId": s.track_id,
                        "overlap": s.overlap,
                        "reason": s.reason,
                        "alignmentReason": s.alignment_reason,
                    }
                    for s in recent.segments
                ]
                if value.period_seconds in (5, 10) and recent is not None
                else None
            )
            try:
                await persist(
                    self._resolved_audio_snapshot(value),
                    through_index=job.window.window_index,
                    segments=segments,
                )
            except Exception as error:
                # Current bounded context already contains the correction.
                # A storage failure must not discard this window's speech.
                _log_failure("speaker_revision", error)

    def _resolved_audio_snapshot(self, value: SemanticSnapshot) -> SemanticSnapshot:
        """Apply recent speaker corrections to views and pending merges only.

        Original words and inferred prose stay fixed. Updated source ownership
        reaches the next merge without re-running old PCM or transcription.
        """
        if value.document.summary_version < 2 or value.period_seconds > 300:
            return value
        cache = self._recent_audio.get(value.fish_id, {})
        document = value.document
        if value.period_seconds in (5, 10):
            recent = cache.get((value.capture_session_id, value.window_index))
            if recent is None or recent.started_at != value.started_at:
                return value
            if _audio_words(document.audio) != _audio_words(recent.text):
                return value
            document = _replace_document_audio(document, recent.text, leaf=True)
        elif document.audio or document.narrative_sources:
            replacements = {
                _audio_range_header(recent.started_at, recent.ended_at): recent.text
                for recent in cache.values()
                if recent.capture_session_id == value.capture_session_id
                and value.started_at
                <= recent.started_at
                < recent.ended_at
                <= value.ended_at
            }
            if not replacements:
                return value
            blocks = []
            for block in document.audio.split("\n\n"):
                header, separator, body = block.partition("\n")
                replacement = replacements.get(header)
                if replacement is not None and _audio_words(body) == _audio_words(
                    replacement
                ):
                    block = header + separator + replacement
                blocks.append(block)
            document = _replace_document_audio(document, "\n\n".join(blocks))
            sources = {}
            for source_id, words in document.narrative_sources.items():
                header, separator, body = words.partition("\n")
                replacement = replacements.get(header)
                if (
                    not source_id.endswith((":media", ":summary"))
                    and replacement is not None
                    and _audio_words(body) == _audio_words(replacement)
                ):
                    words = header + separator + replacement
                sources[source_id] = words
            if sources != document.narrative_sources:
                try:
                    document = SemanticDocument.model_validate(
                        {
                            **document.model_dump(),
                            "narrativeSources": sources,
                        }
                    )
                except ValidationError:
                    # Keep the original auditable evidence on size overflow.
                    pass
        return replace(value, document=document)

    def heard_between(self, fish_id: UUID, since: datetime, through: datetime) -> str:
        """Freeze ready original words inside the requested screen interval.

        Never wait for ASR or substitute an older window. A phrase crossing the
        interval boundary is omitted because its words cannot be split reliably.
        """
        if fish_id not in self._enabled or since >= through:
            return ""
        current = self._last_windows.get(fish_id)
        if current is None:
            return ""
        lines: list[str] = []
        for value in self._recent_audio_context(fish_id, since=since, through=through):
            if value.capture_session_id != current.capture_session_id:
                continue
            if value.segments:
                for segment in value.segments:
                    start = value.started_at + timedelta(seconds=segment.start_seconds)
                    end = value.started_at + timedelta(seconds=segment.end_seconds)
                    if not since <= start < end <= through or not segment.text.strip():
                        continue
                    words = format_system_audio_transcript(
                        SystemAudioTranscript(
                            segment.text,
                            value.event,
                            (
                                replace(
                                    segment,
                                    start_seconds=(start - since).total_seconds(),
                                    end_seconds=(end - since).total_seconds(),
                                ),
                            ),
                        )
                    )
                    lines.append(words)
            elif (
                since <= value.started_at < value.ended_at <= through
                and value.text.strip()
            ):
                lines.append(
                    f"{format_prompt_timestamp(value.started_at)} - "
                    f"{format_prompt_timestamp(value.ended_at)}\n{value.text}"
                )
        if not lines:
            return ""
        return (
            f"{format_prompt_timestamp(since)} - {format_prompt_timestamp(through)}\n"
            + "\n".join(lines)
        )

    def _recent_audio_context(
        self,
        fish_id: UUID,
        *,
        since: datetime,
        through: datetime,
    ) -> tuple[_RecentSystemAudioTranscript, ...]:
        cache = self._recent_audio.get(fish_id)
        if cache is not None:
            for key, value in tuple(cache.items()):
                if value.ended_at <= through - timedelta(
                    seconds=_RECENT_AUDIO_CACHE_SECONDS
                ):
                    cache.pop(key, None)
        return tuple(
            sorted(
                (
                    value
                    for value in (cache.values() if cache is not None else ())
                    if _useful_audio_window(value)
                    and _audio_window_in_range(value, since=since, through=through)
                ),
                key=lambda value: (
                    value.started_at,
                    value.ended_at,
                    str(value.capture_session_id),
                    value.window_index,
                ),
            )
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._enabled.clear()
        self._last_windows.clear()
        self._recent_audio.clear()
        self._recent_snapshots.clear()
        self._context_views.clear()
        tasks = tuple(self._tasks)
        self._jobs.clear()
        self._quiet_visual.clear()
        self._recovery_tasks.clear()
        self._release_task = None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._transcriber.close()
        await self._summarizer.close()


def _redacted_audio_transcript(
    transcript: SystemAudioTranscript,
) -> SystemAudioTranscript:
    text = redact_sensitive_observation_text(transcript.text)
    segments = tuple(
        replace(segment, text=redact_sensitive_observation_text(segment.text))
        for segment in transcript.segments
    )
    original_parts = "".join(segment.text for segment in transcript.segments)
    expected_parts = redact_sensitive_observation_text(original_parts)
    if "".join(segment.text for segment in segments) != expected_parts:
        # A sensitive expression can cross an attribution boundary. Never
        # reassemble its unredacted halves from individually clean segments.
        return replace(
            transcript,
            text=expected_parts,
            segments=(),
            tokens=(),
            timestamps=(),
            alignment_status="redacted_alignment",
        )
    redacted = text != transcript.text or original_parts != expected_parts
    return replace(
        transcript,
        text=text,
        segments=segments,
        tokens=() if redacted else transcript.tokens,
        timestamps=() if redacted else transcript.timestamps,
    )


def _redacted_audio_revision(revision: SystemAudioRevision) -> SystemAudioRevision:
    redacted = _redacted_audio_transcript(
        SystemAudioTranscript(revision.text, revision.event, revision.segments)
    )
    return replace(revision, text=redacted.text, segments=redacted.segments)


def _audio_range_header(started_at: datetime, ended_at: datetime) -> str:
    return (
        f"{started_at.astimezone(UTC):%Y-%m-%d %H:%M:%S} — "
        f"{ended_at.astimezone(UTC):%Y-%m-%d %H:%M:%S} UTC"
    )


def _replace_document_audio(
    document: SemanticDocument, text: str, *, leaf: bool = False
) -> SemanticDocument:
    if text == document.audio:
        return document
    update = {}
    if leaf and document.audiovisual_summary == f"听到的话：\n{document.audio}":
        update["audiovisual_summary"] = f"听到的话：\n{text}"
    # Revalidate storage bounds; never truncate corrected words to make room.
    try:
        return SemanticDocument.model_validate(
            {
                **document.model_dump(),
                "audio": text,
                **(
                    {"audiovisualSummary": update["audiovisual_summary"]}
                    if "audiovisual_summary" in update
                    else {}
                ),
            }
        )
    except ValidationError:
        # The bounded recent-audio view still holds the full correction.
        # A larger display label must not fail a completed capture window.
        return document


def _audio_window_in_range(
    value: _RecentSystemAudioTranscript,
    *,
    since: datetime,
    through: datetime,
) -> bool:
    return (
        value.ended_at > since
        and value.ended_at <= through
        and value.completed_at <= through
    )


def _useful_audio_window(value: _RecentSystemAudioTranscript) -> bool:
    return bool(format_system_audio_text(value.text).strip())


def _missing_parent_representatives(
    values: Sequence[SemanticSnapshot],
) -> tuple[SemanticSnapshot, ...]:
    existing_ids = {value.id for value in values}
    representatives: dict[UUID, SemanticSnapshot] = {}
    for value in values:
        if value.period_seconds in (5, 10):
            continue
        spec = parent_level_for_period(value.period_seconds)
        if spec is None:
            continue
        parent_width = spec.period_seconds // 10
        first_index = (value.window_index // parent_width) * parent_width
        parent_id = snapshot_id(
            value.fish_id,
            value.capture_session_id,
            first_index,
            spec.period_seconds,
        )
        if parent_id not in existing_ids:
            representatives.setdefault(parent_id, value)
    return tuple(
        sorted(
            representatives.values(),
            key=lambda value: (
                value.started_at,
                value.period_seconds,
                value.window_index,
                str(value.id),
            ),
        )
    )


def _select_hour_snapshots(
    values: Sequence[SemanticSnapshot],
    *,
    as_of: datetime,
) -> tuple[SemanticSnapshot, ...]:
    since = as_of - timedelta(seconds=_HOUR_CONTEXT_SECONDS)
    detail_since = as_of - timedelta(seconds=_HOUR_DETAIL_SECONDS)
    retained_since = as_of - timedelta(seconds=_HOUR_CACHE_SECONDS)
    completed = tuple(
        value
        for value in values
        if value.ended_at >= retained_since and value.ended_at <= as_of
    )
    # Anchor before the nominal cutoff on the nearest widest complete block.
    # This also covers a restart gap just after a half-hour block; an old tiny
    # tail must not become the head and force the entire view down to 10s.
    head = min(
        (value for value in completed if value.started_at < since),
        key=lambda item: (-item.period_seconds, -item.started_at.timestamp()),
        default=None,
    )
    starts_at = head.started_at if head is not None else since
    candidates = tuple(value for value in completed if value.started_at >= starts_at)
    recent = tuple(value for value in candidates if value.ended_at > detail_since)
    selected: list[SemanticSnapshot] = []
    for value in sorted(
        candidates,
        key=lambda item: (
            item.started_at,
            -item.period_seconds,
            -item.created_at.timestamp(),
            str(item.id),
        ),
    ):
        if selected and value.started_at < selected[-1].ended_at:
            continue
        if value.ended_at > detail_since and any(
            child.period_seconds < value.period_seconds
            and _snapshots_overlap(value, child)
            for child in recent
        ):
            continue
        selected.append(value)
    return tuple(selected)


def _snapshot_order_key(
    item: SemanticSnapshot,
) -> tuple[datetime, datetime, int, str]:
    return (item.started_at, item.ended_at, -item.period_seconds, str(item.id))


def _snapshots_overlap(left: SemanticSnapshot, right: SemanticSnapshot) -> bool:
    return left.started_at < right.ended_at and right.started_at < left.ended_at


@dataclass(frozen=True, slots=True)
class _SnapshotContextBlock:
    """A display bucket; its sources keep their actual coverage and resolution."""

    period_seconds: int
    snapshots: tuple[SemanticSnapshot, ...]

    @property
    def started_at(self) -> datetime:
        return self.snapshots[0].started_at

    @property
    def ended_at(self) -> datetime:
        return self.snapshots[-1].ended_at

    @property
    def capture_session_id(self) -> UUID:
        return self.snapshots[0].capture_session_id


def _snapshot_context_blocks(
    values: Sequence[SemanticSnapshot],
) -> tuple[_SnapshotContextBlock, ...]:
    # Keep every available coarse summary. Fold an earlier incomplete tail
    # into that layer instead of forcing all later captures down to its detail
    # level. This is local, lossless presentation, not a fabricated Observation
    # or another Ask-time model call.
    periods: list[int] = []
    period = 10
    for value in reversed(values):
        period = max(period, value.period_seconds)
        periods.append(period)
    blocks: list[_SnapshotContextBlock] = []
    for value, period in zip(values, reversed(periods), strict=True):
        width = period // 10
        if (
            blocks
            and blocks[-1].period_seconds == period
            and blocks[-1].capture_session_id == value.capture_session_id
            and blocks[-1].ended_at == value.started_at
            and blocks[-1].snapshots[0].window_index // width
            == value.window_index // width
        ):
            previous = blocks[-1]
            blocks[-1] = replace(previous, snapshots=(*previous.snapshots, value))
        else:
            blocks.append(_SnapshotContextBlock(period, (value,)))
    return tuple(blocks)


def _render_snapshot_context(
    values: Sequence[SemanticSnapshot],
    *,
    include_activity: bool = True,
) -> str:
    values = tuple(sorted(values, key=_snapshot_order_key))
    started_at = values[0].started_at
    ended_at = values[-1].ended_at
    prefix = (
        "## 你之前看到和听到的事"
        f"（{format_prompt_timestamp(started_at)} — "
        f"{format_prompt_timestamp(ended_at)}）。"
        "时间使用用户当前系统时区。"
    )
    groups: list[list[_SnapshotContextBlock]] = []
    for value in _snapshot_context_blocks(values):
        if (
            not groups
            or groups[-1][-1].period_seconds != value.period_seconds
            or groups[-1][-1].ended_at != value.started_at
            or groups[-1][-1].capture_session_id != value.capture_session_id
        ):
            groups.append([])
        groups[-1].append(value)
    headers = [_snapshot_group_header(group, reference=started_at) for group in groups]
    sections = [prefix]
    for header, group in zip(headers, groups, strict=True):
        channels = []
        for field, label in (
            ("audiovisual_summary", "影音内容"),
            ("visual_summary", "看到的画面"),
            ("activity_summary", "电脑活动"),
            ("legacy_summary", "未分类内容"),
        ):
            if field in {"activity_summary", "legacy_summary"} and not include_activity:
                continue
            lines = []
            for value in group:
                summary = "；".join(
                    text
                    for source in value.snapshots
                    if not (
                        not include_activity
                        and field == "audiovisual_summary"
                        and source.document.summary_version < 2
                    )
                    if (text := getattr(source.document, field).strip())
                )
                if not summary:
                    continue
                if value.ended_at - value.started_at < timedelta(
                    seconds=value.period_seconds
                ):
                    duration = f"{value.period_seconds // 60}分钟"
                    summary = (
                        f"（{_snapshot_timestamp(value.started_at, started_at)} — "
                        f"{_snapshot_timestamp(value.ended_at, started_at)}，"
                        f"未满{duration}）{summary}"
                    )
                lines.append("- " + summary.replace("\n", "\n  "))
            if lines:
                channels.append("\n".join((f"{label}：", *lines)))
        quiet_runs: list[list[SemanticSnapshot]] = []
        for block in group:
            for source in block.snapshots:
                if source.document.unchanged_from is None:
                    continue
                if (
                    quiet_runs
                    and quiet_runs[-1][-1].ended_at == source.started_at
                    and quiet_runs[-1][-1].document.unchanged_from
                    == source.document.unchanged_from
                ):
                    quiet_runs[-1].append(source)
                else:
                    quiet_runs.append([source])
        if quiet_runs:
            channels.append(
                "\n".join(
                    f"{_snapshot_timestamp(run[0].started_at, started_at)} — "
                    f"{_snapshot_timestamp(run[-1].ended_at, started_at)}："
                    "画面保持不变，未听到新的声音。"
                    for run in quiet_runs
                )
            )
        if channels:
            sections.append("\n\n".join((header, *channels)))
    return "\n\n".join(sections) if len(sections) > 1 else ""


def _snapshot_group_header(
    values: Sequence[_SnapshotContextBlock], *, reference: datetime
) -> str:
    period = values[0].period_seconds
    label = f"{period // 60}min" if period >= 60 else f"{period}s"
    return (
        f"{_snapshot_timestamp(values[0].started_at, reference)} — "
        f"{_snapshot_timestamp(values[-1].ended_at, reference)}"
        f"｜{label} 粒度"
    )


def _snapshot_timestamp(value: datetime, reference: datetime) -> str:
    value = system_local_time(value)
    reference = system_local_time(reference)
    if value.utcoffset() != reference.utcoffset():
        return format_prompt_timestamp(value, sep=" ")
    # The overall range supplies the date; repeat it only across midnight.
    return value.strftime(
        "%H:%M:%S" if value.date() == reference.date() else "%Y-%m-%d %H:%M:%S"
    )


def _render_recent_audio_context(
    values: Sequence[_RecentSystemAudioTranscript],
    *,
    title: str = "你刚才听到的声音",
) -> str:
    started_at = min(value.started_at for value in values)
    ended_at = max(value.ended_at for value in values)
    prefix = (
        f"## {title}"
        f"（{format_prompt_timestamp(started_at)} — "
        f"{format_prompt_timestamp(ended_at)}）"
    )
    lines = []
    for value in values:
        transcript = format_system_audio_text(value.text).strip()
        lines.append(
            f"\n{_snapshot_timestamp(value.started_at, started_at)} — "
            f"{_snapshot_timestamp(value.ended_at, started_at)}"
        )
        lines.append("- " + transcript.replace("\n", "\n  "))
    return "\n".join((prefix, *lines))


def _render_audio_gaps(
    snapshots: Sequence[SemanticSnapshot], *, as_of: datetime, seconds: int
) -> str:
    since = as_of - timedelta(seconds=seconds)
    missing = sorted(
        (value.started_at, value.ended_at)
        for value in snapshots
        if value.audio_unavailable
        and value.period_seconds in (5, 10)
        and since < value.ended_at <= as_of
    )
    if not missing:
        return ""
    ranges: list[tuple[datetime, datetime]] = []
    for start, end in missing:
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    # A quiet window is not a failure. Only explicitly failed transcription is
    # described here; do not infer missing dialogue from an empty transcript.
    times = "、".join(
        f"{_snapshot_timestamp(start, as_of)} — {_snapshot_timestamp(end, as_of)}"
        for start, end in ranges[-12:]
    )
    return (
        f"听觉记录不完整：{times} 期间的声音没有听清。"
        "已记录的原话只覆盖标注的时段，不能代表整段经过；"
        "这些缺失时段若有剧情描述，来自看到的画面，不能当作听到的对白。"
    )


def _media_recap_audio(
    snapshots: Sequence[SemanticSnapshot],
    pending_audio: Sequence[_RecentSystemAudioTranscript],
    *,
    as_of: datetime,
    max_characters: int = _MEDIA_RECAP_MAX_CHARACTERS,
) -> tuple[_RecentSystemAudioTranscript, ...]:
    """Use whole audio-bearing windows, avoiding minute/leaf duplicates."""
    since = as_of - timedelta(seconds=_MEDIA_RECAP_SECONDS)
    candidates = [
        _RecentSystemAudioTranscript(
            value.fish_id,
            value.capture_session_id,
            value.window_index,
            value.started_at,
            value.ended_at,
            value.document.audio,
            "Speech",
            value.created_at,
        )
        for value in snapshots
        if value.period_seconds <= 60
        and since <= value.started_at < value.ended_at <= as_of
        and format_system_audio_text(value.document.audio)
    ]
    candidates.extend(
        passage
        for snapshot in snapshots
        for passage in _original_audio_from_sources(snapshot)
        if since <= passage.started_at < passage.ended_at <= as_of
    )
    candidates.extend(
        value
        for value in pending_audio
        if since <= value.started_at < value.ended_at <= as_of
        and format_system_audio_text(value.text)
    )
    # Prefer a complete minute's original transcript when available. The minute
    # contains the source window ranges; it is not its generated story summary.
    candidates.sort(
        key=lambda value: (
            value.ended_at - value.started_at,
            value.completed_at,
        ),
        reverse=True,
    )
    selected: list[_RecentSystemAudioTranscript] = []
    for value in candidates:
        if not any(
            other.fish_id == value.fish_id
            and other.capture_session_id == value.capture_session_id
            and other.started_at <= value.started_at
            and value.ended_at <= other.ended_at
            for other in selected
        ):
            selected.append(value)
    result = []
    length = 0
    for value in sorted(
        selected, key=lambda value: (value.started_at, value.ended_at), reverse=True
    ):
        cost = len(value.text) + 80
        if length + cost > max_characters:
            break
        result.append(value)
        length += cost
    return tuple(reversed(result))


_ORIGINAL_PASSAGE_KEY = re.compile(
    r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}(?::\d+)?"
)
_ORIGINAL_PASSAGE_TIME = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) — "
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC"
)


def _original_audio_from_sources(
    snapshot: SemanticSnapshot,
) -> tuple[_RecentSystemAudioTranscript, ...]:
    """Recover stored words after leaves have been compacted into a parent.

    Media/model summaries and compressed header-only provenance are not speech.
    Source timestamps were formatted to seconds; recover their exact native
    ten-second grid from the enclosing snapshot instead of shifting boundaries.
    """
    groups: dict[tuple[datetime, datetime], list[tuple[int, str]]] = {}
    for key, value in snapshot.document.narrative_sources.items():
        if not _ORIGINAL_PASSAGE_KEY.fullmatch(key):
            continue
        header, _, words = value.partition("\n")
        match = _ORIGINAL_PASSAGE_TIME.fullmatch(header)
        if not match or not format_system_audio_text(words).strip():
            continue
        try:
            start, end = (
                datetime.fromisoformat(part).replace(tzinfo=UTC)
                for part in match.groups()
            )
        except ValueError:
            continue
        offset = round((start - snapshot.started_at).total_seconds() / 10)
        duration = round((end - start).total_seconds() / 10) * 10
        exact_start = snapshot.started_at + timedelta(seconds=offset * 10)
        exact_end = exact_start + timedelta(seconds=duration)
        if (
            duration <= 0
            or duration > 60
            or abs((exact_start - start).total_seconds()) >= 1
            or abs((exact_end - end).total_seconds()) >= 1
            or not snapshot.started_at <= exact_start < exact_end <= snapshot.ended_at
        ):
            continue
        part = int(key.rsplit(":", 1)[1]) if ":" in key else 0
        groups.setdefault((exact_start, exact_end), []).append((part, words.strip()))
    return tuple(
        _RecentSystemAudioTranscript(
            snapshot.fish_id,
            snapshot.capture_session_id,
            snapshot.window_index
            + round((start - snapshot.started_at).total_seconds() / 10),
            start,
            end,
            "\n\n".join(text for _, text in sorted(parts)),
            "Speech",
            snapshot.created_at,
        )
        for (start, end), parts in sorted(groups.items())
    )


def _log_failure(phase: str, error: Exception) -> None:
    # Provider exception messages may contain screenshot/transcript fragments.
    status = getattr(error, "status_code", None)
    logger.warning(
        "Environment history failed: phase=%s type=%s status=%s",
        phase,
        type(error).__name__,
        status if isinstance(status, int) else None,
    )


def _transient_model_failure(error: Exception) -> bool:
    """Retry transport pressure and server faults, never invalid payloads."""
    if isinstance(error, (ValidationError, ValueError)):
        return False
    if isinstance(error, (APIConnectionError, APITimeoutError, TimeoutError, OSError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 429} or error.status_code >= 500
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and (status in {408, 429} or status >= 500)


class CombinedEnvironmentSummaryLoader:
    def __init__(
        self,
        history: SummaryLoader,
        legacy: SummaryLoader,
        *,
        policy: Callable[[UUID], EnvironmentPerceptionPolicy],
    ) -> None:
        self._history = history
        self._legacy = legacy
        self._policy = policy

    async def get_context_for_question(
        self, fish_id: UUID, question: str
    ) -> ComputerHistoryContextSummary | None:
        if requests_media_recap(question) and self._policy(fish_id).history_enabled:
            contextual = getattr(self._history, "get_context_for_question", None)
            if callable(contextual):
                return await contextual(fish_id, question)
            return await self._history.get_latest_context_summary(fish_id)
        return await self.get_latest_context_summary(fish_id)

    async def get_latest_context_summary(
        self, fish_id: UUID
    ) -> ComputerHistoryContextSummary | None:
        policy = self._policy(fish_id)
        sources = []
        if policy.operation_enabled:
            sources.append(("操作感知", self._legacy))
        if policy.history_enabled:
            sources.append(("影音总结", self._history))
        results = await asyncio.gather(
            *(loader.get_latest_context_summary(fish_id) for _, loader in sources),
            return_exceptions=True,
        )
        current_policy = self._policy(fish_id)
        allowed_sources = {
            "操作感知": current_policy.operation_enabled,
            "影音总结": current_policy.history_enabled,
        }
        summaries: list[tuple[str, ComputerHistoryContextSummary]] = []
        for (label, _), result in zip(sources, results, strict=True):
            if not allowed_sources[label]:
                continue
            if isinstance(result, BaseException):
                logger.warning(
                    "Environment summary read failed: %s", type(result).__name__
                )
            elif result is not None:
                summaries.append((label, result))
        if not summaries:
            return None
        latest = max((item for _, item in summaries), key=lambda item: item.ended_at)
        sections = []
        for label, item in summaries:
            if label == "影音总结":
                sections.append(item.summary)
            else:
                through = min(item.ended_at, item.last_successful_at)
                sections.append(
                    "## 最近的电脑操作（截至 "
                    f"{format_prompt_timestamp(through)}）\n{item.summary}"
                )
        return replace(latest, summary="\n\n".join(sections))
