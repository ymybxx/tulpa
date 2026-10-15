"""Always-on, PCM-free performance tracking for native companion turns."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast
from uuid import UUID

from fish_memory.companion.diagnostics import (
    VOICE_DRAFT_EVENT_PREFIX,
    AudioDiagnosticsObserver,
    privacy_safe_event_fields,
)
from fish_memory.companion.latency import summarize_turn_latency
from fish_memory.performance.models import JsonValue

logger = logging.getLogger(__name__)


class PetVoiceStatus(StrEnum):
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class _TurnTrace:
    user_message_id: UUID
    answer_message_id: UUID | None = None
    events: list[dict[str, object]] = field(default_factory=list)


class CompanionLatencyTracker:
    """Correlate lightweight timing events across overlapping voice turns."""

    _INPUT_EVENTS = frozenset(
        {
            "speech_start",
            "speech_end",
            "acoustic_end",
            "asr_final",
            "smart_turn_started",
            "smart_turn_result",
        }
    )

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_turns: int = 32,
        max_draft_turns: int = 32,
        max_draft_events_per_turn: int = 64,
    ) -> None:
        if min(max_turns, max_draft_turns, max_draft_events_per_turn) < 1:
            raise ValueError("tracker bounds must be positive")
        self._clock_ns = clock_ns
        self._pending_input: list[dict[str, object]] = []
        self._turns: deque[_TurnTrace] = deque(maxlen=max_turns)
        self._active_answer_id: UUID | None = None
        self._max_draft_turns = max_draft_turns
        self._max_draft_events_per_turn = max_draft_events_per_turn
        self._unmatched_draft_events: OrderedDict[
            UUID,
            deque[dict[str, object]],
        ] = OrderedDict()

    @property
    def unmatched_draft_turn_count(self) -> int:
        return len(self._unmatched_draft_events)

    def record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        text, metadata = privacy_safe_event_fields(
            name,
            text=text,
            metadata=metadata,
        )
        event: dict[str, object] = {
            "name": name,
            "atMs": self._clock_ns() / 1_000_000,
            "source": source,
        }
        if frame_index is not None and frame_index >= 0:
            event["frameIndex"] = frame_index
        if text:
            event["text"] = text
        if metadata:
            event["metadata"] = dict(metadata)

        if name.startswith(VOICE_DRAFT_EVENT_PREFIX):
            turn_id = _event_turn_id(event)
            if turn_id is None:
                return
            trace = self._trace_for_user(turn_id)
            if trace is not None:
                trace.events.append(event)
            else:
                self._buffer_unmatched_draft_event(turn_id, event)
            return

        if name in self._INPUT_EVENTS:
            if name == "speech_start":
                self._pending_input = [event]
            else:
                self._pending_input.append(event)
            return

        message_id = _event_message_id(event)
        if name == "ask_submitted":
            if message_id is None:
                return
            self._turns.append(
                _TurnTrace(
                    user_message_id=message_id,
                    events=[
                        *self._pending_input,
                        *self._unmatched_draft_events.pop(message_id, ()),
                        event,
                    ],
                )
            )
            self._pending_input = []
            return

        if name == "answer_event_received":
            if message_id is None:
                return
            trace = next(
                (
                    turn
                    for turn in reversed(self._turns)
                    if turn.answer_message_id is None
                ),
                None,
            )
            if trace is None:
                return
            trace.answer_message_id = message_id
            trace.events.append(event)
            self._active_answer_id = message_id
            return

        answer_id = message_id or self._active_answer_id
        if answer_id is None:
            return
        trace = self._trace_for_answer(answer_id)
        if trace is None:
            return
        if message_id is None:
            event_metadata = dict(cast(Mapping[str, object], event.get("metadata", {})))
            event_metadata["messageId"] = str(answer_id)
            event["metadata"] = event_metadata
        trace.events.append(event)
        if name == "tts_playback_stopped" and self._active_answer_id == answer_id:
            self._active_answer_id = None

    def metrics_for(
        self,
        answer_message_id: UUID,
        status: PetVoiceStatus,
    ) -> dict[str, JsonValue] | None:
        trace = self._trace_for_answer(answer_message_id)
        if trace is None:
            return None
        summary = summarize_turn_latency(trace.events)
        native_ms = summary.get("nativeMs")
        if not isinstance(native_ms, dict):
            return None
        ready_metadata = _event_metadata(trace.events, "tts_ready")
        metrics: dict[str, JsonValue] = {
            "status": status.value,
            "userMessageId": str(trace.user_message_id),
            "answerMessageId": str(answer_message_id),
            "provider": _optional_string(ready_metadata.get("provider")),
            "model": _optional_string(ready_metadata.get("model")),
            "voice": _optional_string(ready_metadata.get("voice")),
            "connectionReused": _optional_bool(ready_metadata.get("connectionReused")),
            "nativeMs": cast(dict[str, JsonValue], native_ms),
        }
        draft_metrics = _summarize_voice_draft_events(trace.events)
        if draft_metrics is not None:
            metrics["voiceDraft"] = draft_metrics
        facial_metrics = _summarize_facial_events(trace.events)
        if facial_metrics is not None:
            metrics["facial"] = facial_metrics
        acoustic_end_ms = _first_event_ms(trace.events, "acoustic_end")
        if acoustic_end_ms is not None:
            first_content_ms = _first_event_ms(trace.events, "answer_first_content")
            playback_ms = _first_event_ms(trace.events, "tts_playback_started")
            native_ms["acousticEndToFirstContent"] = _nonnegative_difference(
                acoustic_end_ms,
                first_content_ms,
            )
            native_ms["acousticEndToPlayback"] = _nonnegative_difference(
                acoustic_end_ms,
                playback_ms,
            )
        return metrics

    def _trace_for_answer(self, answer_message_id: UUID) -> _TurnTrace | None:
        return next(
            (
                turn
                for turn in reversed(self._turns)
                if turn.answer_message_id == answer_message_id
            ),
            None,
        )

    def _trace_for_user(self, user_message_id: UUID) -> _TurnTrace | None:
        return next(
            (
                turn
                for turn in reversed(self._turns)
                if turn.user_message_id == user_message_id
            ),
            None,
        )

    def _buffer_unmatched_draft_event(
        self,
        turn_id: UUID,
        event: dict[str, object],
    ) -> None:
        events = self._unmatched_draft_events.get(turn_id)
        if events is None:
            events = deque(maxlen=self._max_draft_events_per_turn)
            self._unmatched_draft_events[turn_id] = events
        events.append(event)
        self._unmatched_draft_events.move_to_end(turn_id)
        while len(self._unmatched_draft_events) > self._max_draft_turns:
            self._unmatched_draft_events.popitem(last=False)


class PetVoicePerformanceClient(Protocol):
    async def persist_pet_voice_performance(
        self,
        fish_id: UUID,
        message_id: UUID,
        metrics: dict[str, JsonValue],
    ) -> None: ...


class PetVoicePerformanceReporter:
    """Persist completed snapshots without blocking the realtime voice path."""

    def __init__(
        self,
        client: PetVoicePerformanceClient,
        tracker: CompanionLatencyTracker,
    ) -> None:
        self._client = client
        self._tracker = tracker
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        status: PetVoiceStatus,
    ) -> None:
        metrics = self._tracker.metrics_for(answer_message_id, status)
        if metrics is None:
            return
        task = asyncio.create_task(self._persist(fish_id, answer_message_id, metrics))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks))

    async def _persist(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        metrics: dict[str, JsonValue],
    ) -> None:
        try:
            await self._client.persist_pet_voice_performance(
                fish_id,
                answer_message_id,
                metrics,
            )
        except Exception as error:
            logger.warning(
                "宠物语音性能上报失败: answer_message_id=%s error=%s",
                answer_message_id,
                error,
            )


class CompositeAudioDiagnostics:
    """Keep optional audio diagnostics and always-on timings in one observer."""

    def __init__(
        self,
        audio: AudioDiagnosticsObserver,
        latency: CompanionLatencyTracker,
    ) -> None:
        self._audio = audio
        self._latency = latency

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
        return self._audio.record_frame(
            capture_pcm,
            render_pcm,
            cleaned_pcm,
            voice_detected=voice_detected,
            processing_ms=processing_ms,
            dropped_frames=dropped_frames,
        )

    def record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        text, metadata = privacy_safe_event_fields(
            name,
            text=text,
            metadata=metadata,
        )
        self._audio.record_event(
            name,
            frame_index=frame_index,
            text=text,
            source=source,
            metadata=metadata,
        )
        self._latency.record_event(
            name,
            frame_index=frame_index,
            text=text,
            source=source,
            metadata=metadata,
        )


def _event_message_id(event: Mapping[str, object]) -> UUID | None:
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("messageId")
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _event_turn_id(event: Mapping[str, object]) -> UUID | None:
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("turnId")
    if not isinstance(value, str):
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _event_metadata(
    events: list[dict[str, object]],
    name: str,
) -> Mapping[str, object]:
    for event in events:
        if event.get("name") != name:
            continue
        metadata = event.get("metadata")
        if isinstance(metadata, Mapping):
            return cast(Mapping[str, object], metadata)
    return {}


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _summarize_facial_events(
    events: list[dict[str, object]],
) -> dict[str, JsonValue] | None:
    waits: list[float] = []
    failures = 0
    for event in events:
        if event.get("name") != "tts_facial_ready":
            continue
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        wait = metadata.get("waitMs")
        if (
            not isinstance(wait, (int, float))
            or isinstance(wait, bool)
            or not math.isfinite(wait)
            or wait < 0
        ):
            continue
        waits.append(float(wait))
        failures += bool(metadata.get("errorType"))
    if not waits:
        return None
    return {
        "sentenceCount": len(waits),
        "waitedSentenceCount": sum(wait > 0 for wait in waits),
        "totalWaitMs": round(sum(waits), 3),
        "maxWaitMs": round(max(waits), 3),
        "failedSentenceCount": failures,
    }


def summarize_voice_draft_attempt(
    *,
    started_ms: float,
    commit_ms: float,
    first_content_ms: float | None,
    promoted: bool,
    discarded_pcm_bytes: int,
) -> dict[str, JsonValue]:
    """Summarize one attempt using timing and aggregate counts only."""
    lead_ms = max(0.0, commit_ms - started_ms)
    first_content_lead_ms = (
        None if first_content_ms is None else commit_ms - first_content_ms
    )
    return {
        "draftLeadMs": round(lead_ms, 3),
        "draftFirstContentLeadMs": (
            None if first_content_lead_ms is None else round(first_content_lead_ms, 3)
        ),
        "draftUsefulHit": bool(
            promoted
            and first_content_lead_ms is not None
            and first_content_lead_ms >= 0
        ),
        "draftPromoted": promoted,
        "draftDiscardedPcmBytes": max(0, discarded_pcm_bytes),
    }


def _summarize_voice_draft_events(
    events: list[dict[str, object]],
) -> dict[str, JsonValue] | None:
    draft_events = [
        event
        for event in events
        if isinstance(event.get("name"), str)
        and cast(str, event["name"]).startswith(VOICE_DRAFT_EVENT_PREFIX)
    ]
    commit_ms = _first_event_ms(events, "ask_submitted")
    if not draft_events or commit_ms is None:
        return None
    by_revision: dict[int, list[dict[str, object]]] = {}
    for event in draft_events:
        revision = _event_attempt_revision(event)
        if revision is None or revision < 1:
            continue
        by_revision.setdefault(revision, []).append(event)
    if not by_revision:
        return None
    promoted_revisions = [
        revision
        for revision, attempt_events in by_revision.items()
        if _event_count(attempt_events, "voice_draft_promoted") > 0
    ]
    terminal_names = {
        "voice_draft_canceled",
        "voice_draft_fallback",
        "voice_draft_cleanup",
    }
    terminal_revisions = [
        revision
        for revision, attempt_events in by_revision.items()
        if any(event.get("name") in terminal_names for event in attempt_events)
    ]
    selected_revision = max(
        promoted_revisions or terminal_revisions or list(by_revision)
    )
    selected_events = by_revision[selected_revision]
    started_ms = _first_event_ms(selected_events, "voice_draft_started")
    first_content_ms = _first_event_ms(
        selected_events,
        "voice_draft_first_content",
    )
    promoted_count = _event_count(selected_events, "voice_draft_promoted")
    summary = summarize_voice_draft_attempt(
        started_ms=commit_ms if started_ms is None else started_ms,
        commit_ms=commit_ms,
        first_content_ms=first_content_ms,
        promoted=promoted_count > 0,
        discarded_pcm_bytes=_metadata_total_for_event(
            draft_events,
            "voice_draft_tts_pcm_discarded",
            "pcmBytes",
        ),
    )
    summary.update(
        {
            "selectedRevision": selected_revision,
            "startedCount": _event_count(draft_events, "voice_draft_started"),
            "transcriptReadyCount": _event_count(
                draft_events,
                "voice_draft_transcript_ready",
            ),
            "requestStartedCount": _event_count(
                draft_events,
                "voice_draft_request_started",
            ),
            "firstContentCount": _event_count(
                draft_events,
                "voice_draft_first_content",
            ),
            "promotedCount": promoted_count,
            "canceledCount": _event_count(draft_events, "voice_draft_canceled"),
            "fallbackCount": _event_count(draft_events, "voice_draft_fallback"),
            "lateEventDroppedCount": _event_count(
                draft_events,
                "voice_draft_late_event_dropped",
            ),
            "cleanupCount": _event_count(draft_events, "voice_draft_cleanup"),
            "ttsProviderReadyCount": _event_count(
                draft_events,
                "voice_draft_tts_provider_ready",
            ),
            "ttsTextSubmittedCount": _event_count(
                draft_events,
                "voice_draft_tts_text_submitted",
            ),
            "ttsFirstAudioBufferedCount": _event_count(
                draft_events,
                "voice_draft_tts_first_audio_buffered",
            ),
            "ttsPcmPromotedCount": _event_count(
                draft_events,
                "voice_draft_tts_pcm_promoted",
            ),
            "ttsPcmDiscardedCount": _event_count(
                draft_events,
                "voice_draft_tts_pcm_discarded",
            ),
            "attempts": [
                _summarize_revision_attempt(
                    revision,
                    attempt_events,
                    commit_ms,
                )
                for revision, attempt_events in sorted(by_revision.items())
            ],
        }
    )
    return summary


def _first_event_ms(
    events: list[dict[str, object]],
    name: str,
) -> float | None:
    for event in events:
        if event.get("name") != name:
            continue
        value = event.get("atMs")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _event_count(events: list[dict[str, object]], name: str) -> int:
    return sum(event.get("name") == name for event in events)


def _metadata_total(events: list[dict[str, object]], key: str) -> int:
    total = 0
    for event in events:
        metadata = event.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            total += value
    return total


def _metadata_total_for_event(
    events: list[dict[str, object]],
    name: str,
    key: str,
) -> int:
    return _metadata_total(
        [event for event in events if event.get("name") == name],
        key,
    )


def _nonnegative_difference(start_ms: float, end_ms: float | None) -> float | None:
    if end_ms is None or end_ms < start_ms:
        return None
    return round(end_ms - start_ms, 3)


def _event_attempt_revision(event: Mapping[str, object]) -> int | None:
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("attemptRevision", metadata.get("revision"))
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _summarize_revision_attempt(
    revision: int,
    events: list[dict[str, object]],
    commit_ms: float,
) -> dict[str, JsonValue]:
    first_content_ms = _first_event_ms(events, "voice_draft_first_content")
    first_content_lead_ms = (
        None if first_content_ms is None else round(commit_ms - first_content_ms, 3)
    )
    promoted_count = _event_count(events, "voice_draft_promoted")
    return {
        "revision": revision,
        "startedCount": _event_count(events, "voice_draft_started"),
        "firstContentCount": _event_count(events, "voice_draft_first_content"),
        "promotedCount": promoted_count,
        "canceledCount": _event_count(events, "voice_draft_canceled"),
        "fallbackCount": _event_count(events, "voice_draft_fallback"),
        "lateEventDroppedCount": _event_count(
            events,
            "voice_draft_late_event_dropped",
        ),
        "draftFirstContentLeadMs": first_content_lead_ms,
        "draftUsefulHit": bool(
            promoted_count > 0
            and first_content_lead_ms is not None
            and first_content_lead_ms >= 0
        ),
    }
