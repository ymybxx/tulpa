"""Content-free, bounded runtime logs for diagnosing native voice stalls.

This wraps the existing diagnostics observer without retaining PCM or changing
its events. Levels are sampled locally; no recording, transcription, network
request, timer task, or voice decision is introduced here.
"""

from __future__ import annotations

import json
import logging
import math
import re
import struct
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from uuid import UUID, uuid4

from fish_memory.companion.diagnostics import (
    AudioDiagnosticsObserver,
    privacy_safe_event_fields,
)

logger = logging.getLogger(__name__)

_DRAFT_EVENTS = frozenset(
    "voice_draft_asr_prepare_triggered voice_draft_preparation_started "
    "voice_draft_preparation_queued "
    "voice_draft_started voice_draft_transcript_ready voice_draft_request_started "
    "voice_draft_first_content voice_draft_promoted voice_draft_fallback "
    "voice_draft_canceled voice_draft_cleanup voice_draft_late_event_dropped "
    "voice_draft_tts_provider_ready voice_draft_tts_text_submitted "
    "voice_draft_tts_skipped "
    "voice_draft_tts_first_audio_buffered voice_draft_tts_pcm_promoted "
    "voice_draft_tts_pcm_discarded voice_draft_candidate_triggered "
    "voice_draft_gate_summary voice_draft_resumed voice_draft_asr_reuse_missed".split()
)
_EVENTS = _DRAFT_EVENTS | frozenset(
    "voice_trace_started wake_word_detected session_state_changed speech_start "
    "speech_end vad_reset asr_first_partial asr_decode_started asr_decode_completed "
    "asr_final asr_final_rejected asr_draft_reused near_end_confirmed finish_scheduled "
    "finish_cancelled finish_waiting finish_discarded smart_turn_result "
    "smart_turn_failed submit_turn_rejected ask_submit_started ask_submitted "
    "ask_submit_failed answer_event_received answer_first_content "
    "tts_start_requested tts_ready tts_first_audio_received tts_first_pcm_queued "
    "tts_facial_ready tts_first_audio_timing "
    "tts_playback_started tts_playback_paused tts_playback_resumed "
    "tts_playback_ducked tts_playback_restored tts_playback_stopped "
    "interruption_confirmed answer_playback_completed owner_voice_decision".split()
)
_STATES = frozenset(
    {"unbound", "standby", "acknowledging", "listening", "thinking", "speaking"}
)
_REASONS = frozenset(
    "invalid_voice_revision missing_turn_binding empty_content reply_context_blocked "
    "binding_changed voice_conversation_changed agent_progress_unsettled "
    "agent_reply_finalize_failed binding_changed_after_wake_ack session_changed "
    "state_not_listening punctuation_only short_single_character non_speech_event "
    "nonlexical_vocalization "
    "no_near_end likely_echo speech_resumed speech_end wake_word stale_generation "
    "snapshot_not_consumed smart_turn_incomplete commit_delay cancelled closed "
    "replaced generation_changed speech_restart inference near_end_not_confirmed "
    "smart_turn_inference matched not_owner buffer_limit insufficient_speech "
    "inference_busy invalid_score inference_timeout inference_failed stale_result "
    "owner_voice_rejected".split()
)
_ENUM_FIELDS = {
    "state": _STATES,
    "previousState": _STATES,
    "reason": _REASONS,
    "route": frozenset({"ask", "voice_commit"}),
    "detector": frozenset({"silero_vad"}),
    "decodeKind": frozenset({"snapshot", "finish", "finish_update"}),
    "confirmationSource": frozenset({"partial", "final", "early_final"}),
    "audioBackend": frozenset({"livekit", "sounddevice"}),
    "vadBackend": frozenset({"livekit_silero", "sherpa_silero"}),
    "captureStage": frozenset({"post_vpio", "microphone"}),
    "playbackOwnerKind": frozenset({"answer", "home", "agent", "proactive", "wake"}),
}
_NUMBER_FIELDS = frozenset(
    "speechGeneration expectedGeneration currentGeneration currentSpeechGeneration "
    "preRollFrames commitDelayMs timeoutMs "
    "durationMs remainingMs delayMs speechMs inferenceMs probability charCount "
    "compactChars inputChars pcmBytes playedOffset frameQueueAgeMs frameQueueDepth "
    "droppedFrames revision score releasedFrames waitMs endOffset "
    "providerFirstPacketMs facialWaitMs".split()
)
_BOOL_FIELDS = frozenset(
    "complete requiresConfirmation requiresNearEnd nearEndConfirmed confirmed "
    "answerExpected outputAttenuated frameFinishPending frameFinishClaimed "
    "smartTurnEnabled revoked ownerRecognitionEnabled".split()
)
_UUID_FIELDS = frozenset({"fishId", "sessionId", "messageId", "turnId"})
_ASR_LABELS = frozenset({"speech", "bgm", "breath", "cough", "sneeze"})


def _safe_fields(metadata: Mapping[str, object] | None) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in (metadata or {}).items():
        if key in _BOOL_FIELDS and isinstance(value, bool):
            result[key] = value
        elif (
            key in _NUMBER_FIELDS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            result[key] = value
        elif key in _ENUM_FIELDS and isinstance(value, str):
            if value in _ENUM_FIELDS[key]:
                result[key] = value
        elif key in _UUID_FIELDS and isinstance(value, str) and len(value) == 36:
            with suppress(ValueError):
                if str(UUID(value)) == value:
                    result[key] = value
        elif key == "errorType" and isinstance(value, str):
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", value):
                result[key] = value
        elif key == "asrEvent" and isinstance(value, str):
            # ASR metadata is model output too; never forward arbitrary text.
            result[key] = sorted(
                (
                    set(re.findall(r"<\|([a-z]+)\|>", value.lower()))
                    | {value.strip().lower()}
                )
                & _ASR_LABELS
            )
    return result


class VoiceRuntimeTrace:
    """One five-second input summary plus allowlisted lifecycle events."""

    def __init__(
        self,
        observer: AudioDiagnosticsObserver,
        *,
        context: Callable[[], Mapping[str, object]],
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._observer = observer
        self._context = context
        self._clock_ns = clock_ns
        self._run_id = str(uuid4())
        self._started_ns = clock_ns()
        self._window_started_ns: int | None = None
        self._event_window_ns = self._started_ns
        self._event_count = 0
        self._suppressed_events = 0
        self._disabled = False
        self._speech_started_ns: int | None = None
        self._speech_generation: object = None
        self._session_state: object = None
        self._frames = 0
        self._voiced_frames = 0
        self._vad_evaluated_frames = 0
        self._sample_count = 0
        self._sum_squares = 0
        self._peak = 0
        self._processing_ms_max = 0.0
        self._last_event: str | None = None

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
        index = self._observer.record_frame(
            capture_pcm,
            render_pcm,
            cleaned_pcm,
            voice_detected=voice_detected,
            processing_ms=processing_ms,
            dropped_frames=dropped_frames,
        )
        if not self._disabled and logger.isEnabledFor(logging.INFO):
            try:
                self._observe_frame(cleaned_pcm, voice_detected, processing_ms)
            except Exception:
                self._disable()
        return index

    def record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._observer.record_event(
            name, frame_index=frame_index, text=text, source=source, metadata=metadata
        )
        if (
            self._disabled
            or name not in _EVENTS
            or not logger.isEnabledFor(logging.INFO)
        ):
            return
        try:
            now = self._clock_ns()
            if name in _DRAFT_EVENTS:
                _, safe_metadata = privacy_safe_event_fields(
                    name, text=text, metadata=metadata
                )
                fields = dict(safe_metadata or {})
            else:
                fields = _safe_fields(metadata)
            if name == "smart_turn_failed" and metadata is not None:
                fields.update(_safe_fields({"errorType": metadata.get("error")}))
            if "state" in fields:
                self._session_state = fields["state"]
            if name == "speech_start":
                self._speech_started_ns = now
                self._speech_generation = fields.get("speechGeneration")
            elif name in {"speech_end", "vad_reset"}:
                self._speech_started_ns = None
            self._last_event = name
            if now - self._event_window_ns >= 5_000_000_000:
                self._event_window_ns = now
                self._event_count = 0
            if self._event_count >= 80:
                self._suppressed_events += 1
                return
            self._event_count += 1
            if text is not None and name.startswith("asr_"):
                fields.setdefault("charCount", len(text))
            self._emit(name, now, fields)
        except Exception:
            self._disable()

    def _observe_frame(
        self, pcm: bytes, voiced: bool | None, processing_ms: float | None
    ) -> None:
        now = self._clock_ns()
        if self._window_started_ns is None:
            self._window_started_ns = now
        self._frames += 1
        # The processor's standby fast path supplies False without running
        # VAD. Do not report that placeholder as a negative VAD decision.
        if voiced is not None and (
            self._session_state != "standby"
            or self._context().get("ownerRecognitionEnabled") is True
        ):
            self._vad_evaluated_frames += 1
            self._voiced_frames += int(voiced is True)
        if self._frames % 10 == 1 and len(pcm) == 320:
            samples = struct.unpack("<160h", pcm)
            self._sample_count += len(samples)
            self._sum_squares += sum(sample * sample for sample in samples)
            self._peak = max(self._peak, max(abs(sample) for sample in samples))
        if processing_ms is not None and math.isfinite(processing_ms):
            self._processing_ms_max = max(self._processing_ms_max, processing_ms)
        if now - self._window_started_ns < 5_000_000_000:
            return
        rms = math.sqrt(self._sum_squares / max(1, self._sample_count))
        self._emit(
            "input_window",
            now,
            {
                "windowMs": round((now - self._window_started_ns) / 1_000_000, 1),
                "frames": self._frames,
                "vadEvaluatedFrames": self._vad_evaluated_frames,
                "voicedFrames": self._voiced_frames,
                "candidateActiveMs": (
                    round((now - self._speech_started_ns) / 1_000_000, 1)
                    if self._speech_started_ns is not None
                    else 0
                ),
                "speechGeneration": self._speech_generation,
                "sampledPcmSource": "processor_cleaned_pcm",
                "sampledRmsDbfs": round(20 * math.log10(max(rms, 1) / 32768), 1),
                "sampledPeak": self._peak,
                "processingMaxMs": round(self._processing_ms_max, 3),
                "suppressedEvents": self._suppressed_events,
                "lastEvent": self._last_event,
            },
        )
        self._window_started_ns = now
        self._frames = self._voiced_frames = self._sample_count = 0
        self._vad_evaluated_frames = 0
        self._sum_squares = self._peak = self._suppressed_events = 0
        self._processing_ms_max = 0.0

    def _emit(self, event: str, now: int, fields: Mapping[str, object]) -> None:
        payload = {
            "runId": self._run_id,
            "atMs": round((now - self._started_ns) / 1_000_000, 1),
            **_safe_fields(self._context()),
            **fields,
        }
        self._session_state = payload.get("state", self._session_state)
        logger.info(
            "voice_trace event=%s %s",
            event,
            json.dumps(
                payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
            ),
        )

    def _disable(self) -> None:
        self._disabled = True
        with suppress(Exception):
            logger.warning("Voice runtime trace disabled after a diagnostic error")
