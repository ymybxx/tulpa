"""Converts microphone PCM blocks into KWS and ASR state transitions."""

import asyncio
import logging
import math
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.audio_processing import (
    FRAME_DURATION_MS,
    ProcessedAudioFrame,
    SpeechEvent,
    SpeechTurnDetector,
)
from fish_memory.companion.diagnostics import (
    AudioDiagnosticsObserver,
    record_voice_draft_event,
)
from fish_memory.companion.owner_voice_gate import OwnerVoiceFrame, OwnerVoiceGate
from fish_memory.companion.playback import ForegroundReservation
from fish_memory.companion.session import (
    CompanionSession,
    CompanionState,
    FrozenVoiceTurn,
)
from fish_memory.companion.utterance_admission import (
    low_evidence_utterance_reason,
)
from fish_memory.companion.voice_drafts import (
    CandidateEvent,
    SpeculativeSilenceGate,
    VoiceDraftOwner,
    VoiceDraftTurnController,
)

logger = logging.getLogger(__name__)

_VOICE_DRAFT_MAX_QUEUE_AGE_MS = 30.0


class KwsInput(Protocol):
    def accept_pcm(self, pcm: bytes) -> Sequence[object]: ...

    def reset(self) -> None: ...


class AsrInput(Protocol):
    def accept_pcm(self, pcm: bytes) -> AsrUpdate | None: ...

    def reset(self) -> None: ...

    def finish_utterance(self) -> str: ...

    def snapshot_utterance(self) -> bytes: ...

    def buffered_pcm_bytes(self) -> int: ...

    def recognize_snapshot(self, pcm: bytes) -> str | None: ...

    def consume_snapshot(self, pcm: bytes) -> bool: ...


class SmartTurnResultInput(Protocol):
    complete: bool
    probability: float
    inference_ms: float


class SmartTurnInput(Protocol):
    def predict_pcm16(self, pcm: bytes) -> SmartTurnResultInput: ...


@dataclass(frozen=True, slots=True)
class _FinalizedTurn:
    content: str
    asr_event: str
    speech_ms: int
    requires_confirmation: bool
    confirmed: bool
    requires_near_end: bool
    near_end_confirmed: bool
    frozen_turn: FrozenVoiceTurn | None
    owner_epoch: int = 0
    owner_binding_generation: int = 0


@dataclass(frozen=True, slots=True)
class _PendingTurnPart:
    content: str
    frozen_turn: FrozenVoiceTurn | None


class FrameProcessor(Protocol):
    def process(
        self,
        *,
        capture_pcm: bytes,
        render_pcm: bytes,
    ) -> ProcessedAudioFrame: ...

    def reset_vad(self) -> None: ...

    def set_stream_delay(self, delay_ms: int) -> None: ...


class VadInput(Protocol):
    def is_speech(self, pcm: bytes) -> bool: ...

    def reset(self) -> None: ...


class VoiceDraftVadInput(Protocol):
    def is_speech(self, pcm: bytes) -> bool: ...


class CompanionAudioProcessor:
    def __init__(
        self,
        session: CompanionSession,
        kws: KwsInput,
        asr: AsrInput,
        *,
        # The real-time synchronized path sets this to zero because Silero VAD
        # already applies temporal speech/silence stabilization. The parameter
        # remains for legacy frame callers and unit scenarios that deliberately
        # exercise delayed boundary behavior.
        utterance_commit_delay_seconds: float = 0.0,
        smart_turn: SmartTurnInput | None = None,
        smart_turn_reopen_seconds: float = 0.8,
        smart_turn_timeout_seconds: float = 0.5,
        early_interrupt_snapshot_seconds: float = 0.3,
        early_interrupt_min_chars: int = 2,
        single_char_min_speech_ms: int = 200,
        vad_trailing_silence_ms: int = 0,
        frame_processor: FrameProcessor | None = None,
        interruption_vad: VadInput | None = None,
        defer_interruption_output_until_near_end: bool = False,
        speech_start_frames: int = 1,
        speech_pre_roll_frames: int = 1,
        speech_confirmation_frames: int = 1,
        speech_end_frames: int = 20,
        diagnostics: AudioDiagnosticsObserver | None = None,
        dropped_frames: Callable[[], int] | None = None,
        frame_queue_age_ms: Callable[[], float] | None = None,
        frame_queue_depth: Callable[[], int] | None = None,
        voice_draft_vad: VoiceDraftVadInput | None = None,
        voice_draft_gate: SpeculativeSilenceGate | None = None,
        voice_draft_controller: VoiceDraftTurnController | None = None,
        voice_draft_capture_vad: VoiceDraftVadInput | None = None,
        voice_draft_max_pcm_bytes: int | None = None,
        owner_voice_gate: OwnerVoiceGate | None = None,
        notify_voice_activity: Callable[[], None] | None = None,
    ) -> None:
        self._session = session
        self._kws = kws
        self._asr = asr
        self._utterance_commit_delay_seconds = utterance_commit_delay_seconds
        self._smart_turn = smart_turn
        self._smart_turn_reopen_seconds = smart_turn_reopen_seconds
        if smart_turn_timeout_seconds <= 0:
            raise ValueError("Smart Turn timeout must be positive")
        self._smart_turn_timeout_seconds = smart_turn_timeout_seconds
        self._smart_turn_inference_task: asyncio.Task[SmartTurnResultInput] | None = (
            None
        )
        if early_interrupt_snapshot_seconds < 0:
            raise ValueError("early interrupt snapshot delay cannot be negative")
        if early_interrupt_min_chars < 1:
            raise ValueError("early interrupt minimum characters must be positive")
        self._early_interrupt_snapshot_frames = max(
            1,
            math.ceil(early_interrupt_snapshot_seconds / 0.01),
        )
        self._early_interrupt_min_chars = early_interrupt_min_chars
        if single_char_min_speech_ms < 0:
            raise ValueError("single-character minimum speech cannot be negative")
        if vad_trailing_silence_ms < 0:
            raise ValueError("VAD trailing silence cannot be negative")
        self._single_char_min_speech_ms = single_char_min_speech_ms
        self._vad_trailing_silence_ms = vad_trailing_silence_ms
        self._pending_parts: list[_PendingTurnPart] = []
        self._foreground_reservation: ForegroundReservation | None = None
        self._commit_task: asyncio.Task[None] | None = None
        self._frame_finish_task: asyncio.Task[None] | None = None
        self._frame_finish_error: Exception | None = None
        self._frame_finish_claimed = False
        self._frame_processor = frame_processor
        if owner_voice_gate is not None and frame_processor is None:
            raise ValueError("Owner recognition requires synchronized processed audio")
        self._owner_voice_gate = owner_voice_gate
        self._notify_voice_activity = notify_voice_activity
        self._owner_binding_generation = (
            session.binding_generation if owner_voice_gate is not None else 0
        )
        self._owner_epoch = 0
        self._closed = False
        self._microphone_enabled = True
        self._interruption_vad = interruption_vad
        if defer_interruption_output_until_near_end and interruption_vad is None:
            raise ValueError(
                "deferred interruption output requires an independent near-end VAD"
            )
        self._defer_interruption_output_until_near_end = (
            defer_interruption_output_until_near_end
        )
        self._diagnostics = diagnostics
        self._dropped_frames = dropped_frames
        self._frame_queue_age_ms = frame_queue_age_ms
        self._frame_queue_depth = frame_queue_depth
        self._speech_detector = SpeechTurnDetector(
            start_frames=speech_start_frames,
            end_silence_frames=speech_end_frames,
        )
        if speech_pre_roll_frames < 1:
            raise ValueError("speech_pre_roll_frames must be positive")
        if speech_confirmation_frames < 1:
            raise ValueError("speech_confirmation_frames must be positive")
        self._speech_confirmation_frames = speech_confirmation_frames
        self._speech_pre_roll: deque[tuple[bytes, bytes]] = deque(
            maxlen=speech_pre_roll_frames
        )
        self._speech_candidate_requires_confirmation = False
        self._speech_interruption_confirmed = False
        self._asr_partial_recorded = False
        self._interruption_partial_text: str | None = None
        self._interruption_partial_repeats = 0
        self._confirmed_interruption_text: str | None = None
        self._interruption_requires_near_end = False
        self._interruption_near_end_confirmed = False
        self._stream_delay_frames = 0
        self._post_pause_echo_frames_remaining = 0
        self._active_speech_frames = 0
        self._early_interruption_attempted = False
        self._early_interruption_task: asyncio.Task[None] | None = None
        self._speech_generation = 0
        draft_parts = (
            voice_draft_vad,
            voice_draft_gate,
            voice_draft_controller,
        )
        if any(part is not None for part in draft_parts) and not all(
            part is not None for part in draft_parts
        ):
            raise ValueError("voice draft VAD, gate, and controller must be paired")
        self._voice_draft_vad = voice_draft_vad
        self._voice_draft_gate = voice_draft_gate
        self._voice_draft_controller = voice_draft_controller
        self._voice_draft_capture_vad = voice_draft_capture_vad
        if voice_draft_max_pcm_bytes is not None and voice_draft_max_pcm_bytes < 1:
            raise ValueError("voice draft PCM budget must be positive")
        self._voice_draft_max_pcm_bytes = voice_draft_max_pcm_bytes
        self._voice_draft_pcm_limit_reached = False
        self._voice_draft_owner: VoiceDraftOwner | None = None
        # Freeze the reply target at the first synchronized speech START. This
        # remains authoritative even when an early interruption changes live
        # session state before final ASR is delivered.
        self._frame_frozen_turn: FrozenVoiceTurn | None = None
        self._voice_draft_dropped_frames_baseline: int | None = None
        self._voice_draft_transcript_reusable = False
        self._voice_draft_inflight_frames: set[asyncio.Task[object]] = set()

    @property
    def _input_blocked(self) -> bool:
        return self._closed or not self._microphone_enabled

    async def _discard_invalidated_frame(
        self, *, input_epoch: int, generation: int, binding_generation: int
    ) -> bool:
        if (
            not self._input_blocked
            and input_epoch == self._owner_epoch
            and binding_generation == self._session.binding_generation
        ):
            return False
        if (
            generation == self._speech_generation
            and binding_generation == self._session.binding_generation
        ):
            # Input can be closed while a reversible output control awaits I/O.
            # Undo its late effect, but never touch a newer speech/binding owner.
            await self._session.resume_output_after_speech()
        return True

    async def set_microphone_enabled(self, enabled: bool) -> None:
        if self._closed or enabled == self._microphone_enabled:
            return
        if not enabled:
            self._microphone_enabled = False
            await self._clear_input(discard_claimed=True)
        else:
            await asyncio.to_thread(self._kws.reset)
            await asyncio.to_thread(self._asr.reset)
            await self._reset_vad("microphone_enabled")
            self._microphone_enabled = True

    async def process_pcm(self, pcm: bytes) -> None:
        if self._input_blocked:
            return
        if self._owner_voice_gate is not None:
            if not pcm or len(pcm) % 320:
                raise ValueError("Owner recognition requires complete 10 ms PCM frames")
            for offset in range(0, len(pcm), 320):
                await self.process_frame(pcm[offset : offset + 320], bytes(320))
            return
        state = self._session.state
        if state is CompanionState.STANDBY:
            detections = await asyncio.to_thread(self._kws.accept_pcm, pcm)
            if self._input_blocked:
                return
            if detections:
                self._notify_confirmed_voice_activity()
            if detections and await self._session.on_wake_word():
                await asyncio.to_thread(self._kws.reset)
                await asyncio.to_thread(self._asr.reset)
            return

        if state not in {
            CompanionState.ACKNOWLEDGING,
            CompanionState.LISTENING,
            CompanionState.THINKING,
            CompanionState.SPEAKING,
        }:
            return

        input_epoch = self._owner_epoch
        binding_generation = self._session.binding_generation
        update = await asyncio.to_thread(self._asr.accept_pcm, pcm)
        if (
            self._input_blocked
            or input_epoch != self._owner_epoch
            or binding_generation != self._session.binding_generation
        ):
            return
        if update is None or not update.text.strip():
            return
        if not update.is_final:
            if self._pending_parts:
                self._cancel_pending_commit()
            return
        content = update.text.strip()
        reply_context = getattr(self._session, "reply_context", None)
        rejection_reason = low_evidence_utterance_reason(
            content,
            update.event,
            allow_nonlexical=(
                reply_context is not None
                and reply_context.subject_kind == "interaction"
            ),
        )
        if rejection_reason is None and not _compact_text(content):
            rejection_reason = "punctuation_only"
        if rejection_reason is not None:
            rejection_metadata: dict[str, object] = {
                "reason": rejection_reason,
                "compactChars": len(_compact_text(content)),
            }
            if update.event:
                rejection_metadata["asrEvent"] = update.event
            self._record_event(
                "asr_final_rejected",
                text=content,
                source="asr",
                metadata=rejection_metadata,
            )
            logger.info(
                "忽略低证据 ASR final: reason=%s, event=%r, text=%r",
                rejection_reason,
                update.event,
                content,
            )
            self._release_foreground_reservation()
            return
        if self._session.state is not CompanionState.LISTENING:
            if self._session.is_likely_echo(content):
                self._record_event(
                    "asr_final_rejected",
                    source="asr",
                    metadata={
                        "reason": "likely_echo",
                        "charCount": len(content),
                        "compactChars": len(_compact_text(content)),
                    },
                )
                return
            self._ensure_foreground_reservation()
            frozen_turn = self._pending_frozen_turn()
            if frozen_turn is None or not self._pending_binding_is_current():
                frozen_turn = self._session.freeze_turn_binding()
            if self._pending_parts and not self._pending_binding_is_current():
                await self._flush_pending_utterance()
                self._ensure_foreground_reservation()
            await self._session.interrupt_from_speech()
        else:
            self._ensure_foreground_reservation()
            frozen_turn = self._pending_frozen_turn()
            if frozen_turn is None or not self._pending_binding_is_current():
                frozen_turn = self._session.freeze_turn_binding()
            if self._pending_parts and not self._pending_binding_is_current():
                await self._flush_pending_utterance()
                self._ensure_foreground_reservation()
        frozen_turn = self._session.refresh_pending_agent_reply(frozen_turn)
        if self._session.state is CompanionState.LISTENING:
            self._pending_parts.append(_PendingTurnPart(content, frozen_turn))
            if self._utterance_commit_delay_seconds <= 0:
                await self._flush_pending_utterance()
            else:
                self._schedule_pending_commit()

    async def process_frame(self, capture_pcm: bytes, render_pcm: bytes) -> None:
        """Process one synchronized capture/render frame through AEC and VAD."""
        if self._input_blocked:
            return
        if self._frame_processor is None:
            raise RuntimeError("frame processor is required for synchronized audio")
        # A claimed final may reset the stateful Silero detector. Keep the next
        # frame out of AEC/VAD until that reset and transcript handoff finish.
        await self._wait_for_claimed_frame_finish()
        if self._frame_finish_error is not None:
            error, self._frame_finish_error = self._frame_finish_error, None
            # Route detached finalization failures through the ordinary audio
            # callback so the runner's existing recovery closes these streams.
            raise error
        task = asyncio.current_task()
        assert task is not None
        self._voice_draft_inflight_frames.add(task)
        try:
            await self._process_frame(capture_pcm, render_pcm)
        finally:
            self._voice_draft_inflight_frames.discard(task)

    async def _process_frame(self, capture_pcm: bytes, render_pcm: bytes) -> None:
        assert self._frame_processor is not None
        if (
            self._owner_voice_gate is None
            and self._session.state is CompanionState.STANDBY
            and not any(render_pcm)
        ):
            # Standby only consumes KWS output; its VAD result is discarded and
            # there is no Fish Memory speaker reference to cancel. Avoid running
            # native WebRTC AEC and Silero ONNX 100 times per second until the
            # wake word fires. If playback is present, keep the full path so an
            # acknowledgement/transition frame can never leak echo into KWS.
            processed = ProcessedAudioFrame(
                cleaned_pcm=capture_pcm,
                voice_detected=False,
            )
            frame_index = self._record_frame(
                capture_pcm,
                render_pcm,
                processed,
                processing_ms=0.0,
            )
            await self._process_clean_frame(
                processed,
                capture_pcm=capture_pcm,
                frame_index=frame_index,
            )
            return
        started_ns = time.perf_counter_ns()
        processed = await asyncio.to_thread(
            self._frame_processor.process,
            capture_pcm=capture_pcm,
            render_pcm=render_pcm,
        )
        frame_index = self._record_frame(
            capture_pcm,
            render_pcm,
            processed,
            processing_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
        )
        if self._owner_voice_gate is None:
            await self._process_clean_frame(
                processed, capture_pcm=capture_pcm, frame_index=frame_index
            )
        else:
            await self._process_owner_frame(processed, frame_index=frame_index)

    async def _process_owner_frame(
        self, processed: ProcessedAudioFrame, *, frame_index: int | None
    ) -> None:
        gate = self._owner_voice_gate
        assert gate is not None
        # Owner admission may buffer speech or replace unchecked frames with
        # zeroes. Observe the original capture before those replacements.
        capture_vad = self._voice_draft_capture_vad
        capture_voice = processed.candidate_voice_detected
        if capture_voice is None:
            capture_voice = (
                capture_vad.is_speech(processed.cleaned_pcm)
                if capture_vad is not None
                else False
            )
        if capture_voice:
            self._invalidate_voice_draft_transcript("capture_speech")
        binding = self._session.binding_generation
        if binding != self._owner_binding_generation:
            gate.reset()
            await self._discard_owner_candidate()
            self._owner_binding_generation = binding
        epoch = self._owner_epoch
        if gate.pending_speech:
            self._invalidate_voice_draft_transcript("owner_pending_speech")
        admission = await gate.accept(
            OwnerVoiceFrame(processed, frame_index),
            finalization_pending=(
                self._frame_finish_task is not None
                and not self._frame_finish_task.done()
            ),
        )
        if (
            self._input_blocked
            or binding != self._session.binding_generation
            or epoch != self._owner_epoch
        ):
            gate.reset()
            return
        if gate.pending_speech:
            self._invalidate_voice_draft_transcript("owner_pending_speech")
        if admission.reason is not None:
            self._record_event(
                "owner_voice_decision",
                source="owner_voice",
                frame_index=frame_index,
                metadata={
                    "reason": admission.reason,
                    "score": admission.score,
                    "inferenceMs": admission.inference_ms,
                    "revoked": admission.revoked,
                    "releasedFrames": len(admission.frames),
                },
            )
        if admission.revoked:
            await self._discard_owner_candidate()
            return
        for frame in admission.frames:
            if self._input_blocked or binding != self._session.binding_generation:
                gate.reset()
                if not self._input_blocked:
                    await self._discard_owner_candidate()
                return
            # Never validate cleaned speech and then feed different, raw
            # speaker audio to ASR/drafts. The disabled path stays unchanged.
            await self._process_clean_frame(
                frame.processed,
                capture_pcm=frame.processed.cleaned_pcm,
                frame_index=frame.frame_index,
            )
            if self._input_blocked or epoch != self._owner_epoch:
                gate.reset()
                if binding == self._session.binding_generation:
                    # close() may have resumed before an in-flight pause had
                    # set its flag. Undo that late reversible side effect.
                    await self._session.resume_output_after_speech()
                return
        if capture_voice:
            self._invalidate_voice_draft_transcript("capture_speech")

    async def _discard_owner_candidate(self) -> None:
        self._owner_epoch += 1
        self._speech_generation += 1
        self._cancel_frame_finish(reason="owner_voice_rejected")
        self._cancel_pending_commit()
        self._retire_early_interruption()
        self._abandon_voice_draft("rejected")
        for part in self._pending_parts:
            self._session.release_frozen_turn(part.frozen_turn)
        self._pending_parts.clear()
        await asyncio.to_thread(self._asr.reset)
        await asyncio.to_thread(self._kws.reset)
        self._speech_detector.reset()
        self._speech_pre_roll.clear()
        self._active_speech_frames = 0
        self._speech_candidate_requires_confirmation = False
        self._speech_interruption_confirmed = False
        self._asr_partial_recorded = False
        self._confirmed_interruption_text = None
        self._reset_interruption_partial()
        self._reset_near_end_confirmation()
        self._release_foreground_reservation()
        await self._session.resume_output_after_speech()

    def set_stream_delay(self, delay_ms: int) -> None:
        if delay_ms < 0:
            raise ValueError("stream delay cannot be negative")
        frame_processor = self._frame_processor
        if frame_processor is None:
            raise RuntimeError("frame processor is required for stream-delay updates")
        frame_processor.set_stream_delay(delay_ms)
        self._stream_delay_frames = (delay_ms + 9) // 10

    async def _process_clean_frame(
        self,
        processed: ProcessedAudioFrame,
        *,
        capture_pcm: bytes,
        frame_index: int | None,
    ) -> None:
        if self._input_blocked:
            return
        pcm = processed.cleaned_pcm
        state = self._session.state
        if state is CompanionState.STANDBY:
            detections = await asyncio.to_thread(self._kws.accept_pcm, pcm)
            if self._input_blocked:
                return
            if detections:
                self._notify_confirmed_voice_activity()
            if detections and await self._session.on_wake_word():
                await asyncio.to_thread(self._kws.reset)
                await asyncio.to_thread(self._asr.reset)
                await self._reset_vad("wake_word")
                self._speech_detector.reset()
                self._speech_pre_roll.clear()
                self._speech_candidate_requires_confirmation = False
                self._speech_interruption_confirmed = False
                self._reset_interruption_partial()
                self._reset_near_end_confirmation()
            return

        if state not in {
            CompanionState.ACKNOWLEDGING,
            CompanionState.LISTENING,
            CompanionState.THINKING,
            CompanionState.SPEAKING,
        }:
            return

        input_epoch = self._owner_epoch
        binding_generation = self._session.binding_generation
        echo_tail_active = (
            self._speech_detector.active
            and self._session.output_attenuated_for_speech
            and self._post_pause_echo_frames_remaining > 0
        )
        if self._speech_detector.active and self._interruption_requires_near_end:
            # AEC alignment delay can be hundreds of milliseconds on devices
            # that resample in PortAudio. It is valid for echo alignment, but
            # must not become a blind interval for short interruptions. Confirm
            # sustained near-end speech immediately on the AEC-cleaned track.
            generation = self._speech_generation
            await self._observe_near_end(pcm)
            if await self._discard_invalidated_frame(
                input_epoch=input_epoch,
                generation=generation,
                binding_generation=binding_generation,
            ):
                return
        boundary_voice = processed.voice_detected
        # The independently reset VAD is only a one-way proof that this is
        # near-end speech. It starts observing after the primary VAD and can
        # therefore report silence before the full utterance has ended. Keep
        # the continuously running primary VAD as the sole turn boundary.
        event = self._speech_detector.accept(boundary_voice)
        self._speech_pre_roll.append((capture_pcm, pcm))
        if event is SpeechEvent.START:
            self._notify_confirmed_voice_activity()
            # While TA is talking, a sound is not yet an interruption: reserving
            # the foreground drops proactive speech at once, so a cough or an
            # echo would silence TA mid-sentence. Reserve when it is confirmed.
            if self._session.state is CompanionState.LISTENING:
                self._ensure_foreground_reservation()
            self._speech_generation += 1
            generation = self._speech_generation
            if self._frame_frozen_turn is None:
                self._frame_frozen_turn = self._session.freeze_turn_binding()
            self._open_or_reopen_voice_draft()
            # A brief VAD gap must not close the ASR stream. Cancel only the
            # delayed boundary; the recognizer still contains the prior audio.
            self._cancel_frame_finish(
                reason="speech_restart",
                generation=self._speech_generation - 1,
            )
            self._cancel_pending_commit()
            self._retire_early_interruption()
            self._active_speech_frames = self._speech_confirmation_frames
            self._early_interruption_attempted = False
            self._record_event(
                "speech_start",
                frame_index=frame_index,
                source="vad",
                metadata={
                    "preRollFrames": len(self._speech_pre_roll),
                    "requiresConfirmation": (
                        self._session.state is not CompanionState.LISTENING
                    ),
                },
            )
            self._speech_candidate_requires_confirmation = (
                self._session.state is not CompanionState.LISTENING
            )
            self._speech_interruption_confirmed = False
            self._asr_partial_recorded = False
            self._reset_interruption_partial()
            self._confirmed_interruption_text = None
            self._reset_near_end_confirmation()
            if self._speech_candidate_requires_confirmation:
                if (
                    self._defer_interruption_output_until_near_end
                    and self._interruption_vad is not None
                ):
                    await asyncio.to_thread(self._interruption_vad.reset)
                    self._interruption_requires_near_end = True
                    # LiveKit/VPIO provides platform-cleaned PCM. Keep TTS at
                    # full volume while that track receives independent
                    # near-end confirmation. A primary-VAD candidate alone
                    # must not attenuate or pause playback.
                    await self._observe_near_end(pcm)
                else:
                    attenuated = await self._session.attenuate_output_for_speech()
                    if attenuated:
                        self._post_pause_echo_frames_remaining = (
                            self._stream_delay_frames
                        )
                    if attenuated and self._interruption_vad is not None:
                        await asyncio.to_thread(self._interruption_vad.reset)
                        self._interruption_requires_near_end = True
                    else:
                        self._interruption_near_end_confirmed = True
                if await self._discard_invalidated_frame(
                    input_epoch=input_epoch,
                    generation=generation,
                    binding_generation=binding_generation,
                ):
                    return
            # The AEC-cleaned track decides whether speech started, but it can
            # suppress the first phoneme during double-talk. Preserve the raw
            # capture pre-roll for ASR; echo-only candidates are still rejected
            # by the independent near-end VAD after playback is attenuated.
            for frame_pair in tuple(self._speech_pre_roll):
                await self._accept_asr_frame(frame_pair[0])
            self._speech_pre_roll.clear()
        elif self._speech_detector.active:
            self._active_speech_frames += 1
            asr_pcm = (
                pcm
                if echo_tail_active
                else capture_pcm
                if (
                    self._session.state is CompanionState.LISTENING
                    or (
                        self._session.output_attenuated_for_speech
                        and self._interruption_near_end_confirmed
                    )
                )
                else pcm
            )
            # Silence was classified on the cleaned track. If ASR received
            # different tail samples, keep its existing full decode instead.
            if self._voice_draft_transcript_reusable and asr_pcm != pcm:
                self._invalidate_voice_draft_transcript("pcm_track_changed")
            await self._accept_asr_frame(asr_pcm)

        if self._speech_detector.active:
            self._maybe_start_early_interruption()
            self._process_voice_draft_frame(
                pcm, candidate_voice_detected=processed.candidate_voice_detected
            )
        elif self._voice_draft_owner is not None:
            self._observe_voice_draft_tail(
                pcm,
                boundary_voice=boundary_voice,
                candidate_voice_detected=processed.candidate_voice_detected,
            )

        if echo_tail_active:
            self._post_pause_echo_frames_remaining -= 1

        if event is SpeechEvent.END:
            self._close_voice_draft_gate("speech_end")
            self._record_event(
                "speech_end",
                frame_index=frame_index,
                source="vad",
                metadata={
                    "requiresConfirmation": (
                        self._speech_candidate_requires_confirmation
                    ),
                },
            )
            if self._smart_turn is None and self._utterance_commit_delay_seconds <= 0:
                await self._finish_utterance(record_speech_end=False)
            else:
                self._schedule_frame_finish()

    def _notify_confirmed_voice_activity(self) -> None:
        notify = self._notify_voice_activity
        if notify is not None:
            notify()

    async def _accept_asr_frame(self, pcm: bytes) -> None:
        # Endpoint flags from ASR are intentionally ignored. VAD owns the
        # utterance boundary so a short pause cannot split one user turn.
        if self._input_blocked:
            return
        input_epoch = self._owner_epoch
        binding_generation = self._session.binding_generation
        update = await asyncio.to_thread(self._asr.accept_pcm, pcm)
        if (
            self._input_blocked
            or input_epoch != self._owner_epoch
            or binding_generation != self._session.binding_generation
        ):
            return
        if update is None or not update.text.strip():
            return
        if not self._asr_partial_recorded:
            self._record_event(
                "asr_first_partial",
                text=update.text,
                source="asr",
            )
            self._asr_partial_recorded = True
        if self._session.state is CompanionState.LISTENING:
            show_hearing = getattr(self._session, "show_hearing", None)
            if callable(show_hearing):
                show_hearing(update.text)
            return
        if not self._speech_candidate_requires_confirmation:
            return
        if self._session.is_likely_echo(update.text):
            logger.debug("忽略疑似回声 ASR partial: text=%r", update.text)
            self._reset_interruption_partial()
            return
        if (
            self._interruption_requires_near_end
            and not self._interruption_near_end_confirmed
        ):
            return
        # Do not let a one-character ASR fragment stop an answer. It is both
        # too little evidence of user speech and the commonest echo artifact.
        if not update.is_final and len(_compact_text(update.text)) < 2:
            self._reset_interruption_partial()
            return
        if not update.is_final and not self._observe_interruption_partial(update.text):
            return
        await self._confirm_user_speech(update.text, source="partial")

    async def _confirm_user_speech(self, text: str, *, source: str) -> None:
        if self._session.state is CompanionState.LISTENING:
            return
        # With owner recognition on, ASR only ever receives speech that already
        # passed the owner check, so words heard here are the owner's: stop now
        # rather than after the whole sentence is checked.
        self._record_event(
            "interruption_confirmed",
            text=text,
            source="asr",
            metadata={"confirmationSource": source},
        )
        logger.info("确认用户插话并停止当前回答: source=%s, text=%r", source, text)
        if source == "partial":
            self._confirmed_interruption_text = text.strip()
        self._ensure_foreground_reservation()
        await self._session.interrupt_from_speech()
        self._speech_interruption_confirmed = (
            self._session.state is CompanionState.LISTENING
        )
        self._reset_interruption_partial()

    async def _finish_utterance(self, *, record_speech_end: bool = True) -> None:
        await self._finish_utterance_snapshot(
            record_speech_end=record_speech_end,
            pcm_snapshot=None,
            expected_generation=None,
        )

    async def _finish_utterance_snapshot(
        self,
        *,
        record_speech_end: bool,
        pcm_snapshot: bytes | None,
        expected_generation: int | None,
    ) -> None:
        if self._owner_voice_gate is not None and (
            self._input_blocked
            or self._owner_binding_generation != self._session.binding_generation
        ):
            # A binding may change while Smart Turn waits without another
            # microphone frame. Never re-label its old PCM with the new Fish.
            self._abandon_voice_draft("binding_changed")
            self._release_foreground_reservation()
            await self._session.resume_output_after_speech()
            return
        owner_epoch = self._owner_epoch
        owner_binding = self._session.binding_generation
        self._retire_early_interruption()
        if record_speech_end:
            self._record_event(
                "speech_end",
                source="vad",
                metadata={
                    "requiresConfirmation": (
                        self._speech_candidate_requires_confirmation
                    ),
                },
            )
        finish_update = getattr(self._asr, "finish_utterance_update", None)
        finish = getattr(self._asr, "finish_utterance", None)
        finish_call = finish_update if callable(finish_update) else finish
        if not callable(finish_call):
            self._abandon_voice_draft("rejected")
            await self._session.resume_output_after_speech()
            await self._reset_vad("speech_end")
            self._speech_detector.reset()
            self._speech_pre_roll.clear()
            self._asr_partial_recorded = False
            self._reset_interruption_partial()
            self._confirmed_interruption_text = None
            self._speech_candidate_requires_confirmation = False
            self._speech_interruption_confirmed = False
            self._reset_near_end_confirmation()
            self._release_foreground_reservation()
            return
        raw_update: AsrUpdate
        recognize_update = getattr(self._asr, "recognize_snapshot_update", None)
        recognize = (
            recognize_update
            if callable(recognize_update)
            else getattr(self._asr, "recognize_snapshot", None)
        )
        consume = getattr(self._asr, "consume_snapshot", None)
        decode_generation = (
            self._speech_generation
            if expected_generation is None
            else expected_generation
        )
        reusable_pcm = pcm_snapshot
        reusable_update: AsrUpdate | None = None
        controller = self._voice_draft_controller
        draft_owner = self._voice_draft_owner
        if (
            pcm_snapshot is None
            and controller is not None
            and draft_owner is not None
            and callable(recognize)
            and callable(consume)
        ):
            # Keep the live PCM intact while registration can still be revoked
            # by resumed speech. SenseVoice can decode this detached snapshot.
            pcm_snapshot = self._asr.snapshot_utterance()
            reusable_pcm = pcm_snapshot
        if (
            controller is not None
            and draft_owner is not None
            and self._can_reuse_voice_draft_transcript(draft_owner)
            and callable(consume)
        ):
            if reusable_pcm is None:
                reusable_pcm = self._asr.snapshot_utterance()
            reusable_update = await controller.wait_for_transcript(
                draft_owner, reusable_pcm
            )
            if not self._can_reuse_voice_draft_transcript(draft_owner):
                reusable_update = None
        if reusable_update is not None:
            if not await self._wait_voice_draft_registration(
                draft_owner, generation=decode_generation
            ):
                return
            if draft_owner is None or not self._can_reuse_voice_draft_transcript(
                draft_owner
            ):
                reusable_update = None
        if reusable_update is not None and reusable_pcm is not None:
            assert callable(consume)
            if not self._claim_frame_finalization(expected_generation):
                return
            if not consume(reusable_pcm):
                self._frame_finish_claimed = False
                self._record_event(
                    "finish_discarded",
                    metadata={
                        "reason": "snapshot_not_consumed",
                        "speechGeneration": decode_generation,
                        "currentSpeechGeneration": self._speech_generation,
                    },
                )
                return
            raw_update = reusable_update
            self._voice_draft_transcript_reusable = False
            self._record_event(
                "asr_draft_reused",
                source="asr",
                metadata={
                    "speechGeneration": decode_generation,
                    "pcmBytes": len(reusable_pcm),
                    "charCount": len(raw_update.text),
                    "asrEvent": raw_update.event,
                },
            )
        elif pcm_snapshot is not None and callable(recognize) and callable(consume):
            decode_started_ns = self._record_asr_decode_started(
                generation=decode_generation,
                decode_kind="snapshot",
            )
            decoded = await asyncio.to_thread(recognize, pcm_snapshot)
            decoded_update = _coerce_asr_update(decoded)
            self._record_asr_decode_completed(
                decoded_update,
                generation=decode_generation,
                decode_kind="snapshot",
                started_ns=decode_started_ns,
            )
            if decoded_update is not None:
                if not await self._wait_voice_draft_registration(
                    draft_owner, generation=decode_generation
                ):
                    return
                if not self._claim_frame_finalization(expected_generation):
                    return
                if not consume(pcm_snapshot):
                    self._frame_finish_claimed = False
                    self._record_event(
                        "finish_discarded",
                        metadata={
                            "reason": "snapshot_not_consumed",
                            "speechGeneration": decode_generation,
                            "currentSpeechGeneration": self._speech_generation,
                        },
                    )
                    return
                raw_update = decoded_update
            else:
                # Online recognizers cannot decode a detached snapshot. Claim
                # the turn before their destructive reset, so a new START
                # waits for the old final instead of cancelling its coroutine.
                if not await self._wait_voice_draft_registration(
                    draft_owner, generation=decode_generation
                ):
                    return
                if not self._claim_frame_finalization(expected_generation):
                    return
                decode_started_ns = self._record_asr_decode_started(
                    generation=decode_generation,
                    decode_kind="finish",
                )
                raw_update = _coerce_asr_update(
                    await asyncio.to_thread(finish_call)
                ) or AsrUpdate(text="", is_final=True)
                self._record_asr_decode_completed(
                    raw_update,
                    generation=decode_generation,
                    decode_kind="finish",
                    started_ns=decode_started_ns,
                )
        else:
            if not await self._wait_voice_draft_registration(
                draft_owner, generation=decode_generation
            ):
                return
            if not self._claim_frame_finalization(expected_generation):
                return
            decode_started_ns = self._record_asr_decode_started(
                generation=decode_generation,
                decode_kind="finish",
            )
            raw_update = _coerce_asr_update(
                await asyncio.to_thread(finish_call)
            ) or AsrUpdate(text="", is_final=True)
            self._record_asr_decode_completed(
                raw_update,
                generation=decode_generation,
                decode_kind="finish",
                started_ns=decode_started_ns,
            )
        if self._input_blocked or (
            self._owner_voice_gate is not None
            and (
                owner_epoch != self._owner_epoch
                or owner_binding != self._session.binding_generation
            )
        ):
            return
        turn = self._finalize_turn_state(raw_update.text, raw_update.event)
        try:
            await self._deliver_finalized_turn(turn)
        except BaseException:
            self._session.release_frozen_turn(turn.frozen_turn)
            raise
        finally:
            if self._frame_processor is not None:
                self._release_foreground_reservation()

    def _claim_frame_finalization(self, expected_generation: int | None) -> bool:
        if expected_generation is None:
            return True
        if expected_generation != self._speech_generation:
            self._record_event(
                "finish_discarded",
                metadata={
                    "reason": "stale_generation",
                    "speechGeneration": expected_generation,
                    "currentSpeechGeneration": self._speech_generation,
                },
            )
            return False
        self._frame_finish_claimed = True
        return True

    def _finalize_turn_state(
        self,
        raw_content: str,
        asr_event: str = "",
    ) -> _FinalizedTurn:
        content = _prefer_complete_transcript(
            raw_content,
            self._confirmed_interruption_text
            if self._speech_interruption_confirmed
            else None,
        )
        final_metadata: dict[str, object] = {}
        if raw_content != content:
            final_metadata["rawText"] = raw_content
        if asr_event:
            final_metadata["asrEvent"] = asr_event
        self._record_event(
            "asr_final",
            text=content or None,
            source="asr",
            metadata=final_metadata or None,
        )
        self._reset_vad_now("speech_end")
        frozen_turn = self._frame_frozen_turn
        self._frame_frozen_turn = None
        turn = _FinalizedTurn(
            content=content,
            asr_event=asr_event,
            speech_ms=max(
                0,
                self._active_speech_frames * FRAME_DURATION_MS
                - self._vad_trailing_silence_ms,
            ),
            requires_confirmation=self._speech_candidate_requires_confirmation,
            confirmed=self._speech_interruption_confirmed,
            requires_near_end=self._interruption_requires_near_end,
            near_end_confirmed=self._interruption_near_end_confirmed,
            frozen_turn=frozen_turn or self._session.freeze_turn_binding(),
            owner_epoch=self._owner_epoch,
            owner_binding_generation=self._session.binding_generation,
        )
        self._speech_detector.reset()
        self._speech_pre_roll.clear()
        self._asr_partial_recorded = False
        self._reset_interruption_partial()
        self._speech_candidate_requires_confirmation = False
        self._speech_interruption_confirmed = False
        self._confirmed_interruption_text = None
        self._reset_near_end_confirmation()
        self._active_speech_frames = 0
        self._early_interruption_attempted = False
        return turn

    async def _deliver_finalized_turn(self, turn: _FinalizedTurn) -> None:
        if not self._owner_turn_is_current(turn):
            self._session.release_frozen_turn(turn.frozen_turn)
            return
        content = turn.content
        compact_content = _compact_text(content)
        reply_context = (
            turn.frozen_turn.reply_context if turn.frozen_turn is not None else None
        )
        # Only a pending interaction may relax short-input admission. Result
        # playback also carries reply context for routing substantive follow-ups.
        allow_nonlexical = (
            reply_context is not None and reply_context.subject_kind == "interaction"
        )
        rejection_reason = low_evidence_utterance_reason(
            content,
            turn.asr_event,
            allow_nonlexical=allow_nonlexical,
        )
        if rejection_reason is None and not compact_content:
            rejection_reason = "punctuation_only"
        elif (
            rejection_reason is None
            and len(compact_content) == 1
            and turn.speech_ms < self._single_char_min_speech_ms
            and not allow_nonlexical
        ):
            # Preserve the existing protection against very short one-character
            # hallucinations. The new nonlexical gate is additive and must not
            # broaden admission for unrelated noise.
            rejection_reason = "short_single_character"
        if rejection_reason is not None:
            self._abandon_voice_draft("rejected")
            rejection_metadata: dict[str, object] = {
                "reason": rejection_reason,
                "speechMs": turn.speech_ms,
                "compactChars": len(compact_content),
            }
            if turn.asr_event:
                rejection_metadata["asrEvent"] = turn.asr_event
            self._record_event(
                "asr_final_rejected",
                text=content or None,
                source="asr",
                metadata=rejection_metadata,
            )
            logger.info(
                "忽略低证据 ASR final: reason=%s, event=%r, speech_ms=%s, text=%r",
                rejection_reason,
                turn.asr_event,
                turn.speech_ms,
                content,
            )
            await self._session.resume_output_after_speech()
            self._session.release_frozen_turn(turn.frozen_turn)
            return
        if (
            turn.requires_confirmation
            and not turn.confirmed
            and turn.requires_near_end
            and not turn.near_end_confirmed
        ):
            self._abandon_voice_draft("rejected")
            self._record_event(
                "asr_final_rejected",
                source="asr",
                metadata={
                    "reason": "near_end_not_confirmed",
                    "speechMs": turn.speech_ms,
                    "charCount": len(content),
                    "compactChars": len(compact_content),
                    "requiresConfirmation": turn.requires_confirmation,
                    "requiresNearEnd": turn.requires_near_end,
                    "nearEndConfirmed": turn.near_end_confirmed,
                    "detector": "silero_vad",
                },
            )
            logger.info("忽略没有近端人声证据的回声候选: text=%r", content)
            await self._session.resume_output_after_speech()
            self._session.release_frozen_turn(turn.frozen_turn)
            return
        if content and turn.requires_confirmation and not turn.confirmed:
            if self._session.is_likely_echo(content):
                self._abandon_voice_draft("rejected")
                self._record_event(
                    "asr_final_rejected",
                    source="asr",
                    metadata={
                        "reason": "likely_echo",
                        "speechMs": turn.speech_ms,
                        "charCount": len(content),
                        "compactChars": len(compact_content),
                    },
                )
                logger.info("忽略疑似回声 ASR final: text=%r", content)
                await self._session.resume_output_after_speech()
                self._session.release_frozen_turn(turn.frozen_turn)
                return
            if self._session.state is not CompanionState.LISTENING:
                self._record_event(
                    "interruption_confirmed",
                    text=content,
                    source="asr",
                    metadata={"confirmationSource": "final"},
                )
                logger.info(
                    "确认用户插话并停止当前回答: source=final, text=%r",
                    content,
                )
                await self._session.interrupt_from_speech()
        elif turn.requires_confirmation and not turn.confirmed:
            await self._session.resume_output_after_speech()
        frozen_turn = self._session.refresh_pending_agent_reply(turn.frozen_turn)
        if not self._owner_turn_is_current(turn):
            self._session.release_frozen_turn(frozen_turn)
            return
        if content and self._session.state is CompanionState.LISTENING:
            if self._frame_processor is not None:
                if self._voice_draft_owner is None:
                    accepted = await self._session.submit_transcript(
                        content,
                        frozen_turn=frozen_turn,
                    )
                else:
                    accepted = await self._submit_voice_draft(
                        content,
                        frozen_turn,
                    )
                if not accepted:
                    self._session.release_frozen_turn(frozen_turn)
            else:
                self._pending_parts.append(_PendingTurnPart(content, frozen_turn))
                if self._utterance_commit_delay_seconds <= 0:
                    await self._flush_pending_utterance()
                else:
                    self._schedule_pending_commit()
        elif self._voice_draft_owner is not None:
            self._abandon_voice_draft("rejected")
            self._session.release_frozen_turn(frozen_turn)
        elif self._session.state is not CompanionState.LISTENING:
            self._session.release_frozen_turn(frozen_turn)

    def _owner_turn_is_current(self, turn: _FinalizedTurn) -> bool:
        return not self._input_blocked and (
            self._owner_voice_gate is None
            or (
                turn.owner_epoch == self._owner_epoch
                and turn.owner_binding_generation == self._session.binding_generation
            )
        )

    async def _observe_near_end(self, capture_pcm: bytes) -> bool | None:
        if not self._interruption_requires_near_end or self._interruption_vad is None:
            return None
        voice_detected = await asyncio.to_thread(
            self._interruption_vad.is_speech,
            capture_pcm,
        )
        if voice_detected and not self._interruption_near_end_confirmed:
            self._interruption_near_end_confirmed = True
            self._record_event(
                "near_end_confirmed",
                source="vad",
                metadata={"detector": "silero_vad"},
            )
            paused = await self._session.pause_output_for_speech()
            if paused and self._defer_interruption_output_until_near_end:
                self._post_pause_echo_frames_remaining = self._stream_delay_frames
        return voice_detected

    def _maybe_start_early_interruption(self) -> None:
        if (
            self._early_interruption_attempted
            or not self._speech_candidate_requires_confirmation
            or not self._interruption_requires_near_end
            or not self._interruption_near_end_confirmed
            or self._active_speech_frames < self._early_interrupt_snapshot_frames
            or self._session.state is CompanionState.LISTENING
        ):
            return
        snapshot = getattr(self._asr, "snapshot_utterance", None)
        recognize_update = getattr(self._asr, "recognize_snapshot_update", None)
        recognize = (
            recognize_update
            if callable(recognize_update)
            else getattr(self._asr, "recognize_snapshot", None)
        )
        if not callable(snapshot) or not callable(recognize):
            return
        self._early_interruption_attempted = True
        self._early_interruption_task = asyncio.create_task(
            self._run_early_interruption(
                snapshot,
                recognize,
                generation=self._speech_generation,
                binding_generation=self._session.binding_generation,
            )
        )

    async def _run_early_interruption(
        self,
        snapshot: Callable[[], bytes],
        recognize: Callable[[bytes], object],
        *,
        generation: int,
        binding_generation: int,
    ) -> None:
        task = asyncio.current_task()
        restore_output = False

        def is_current() -> bool:
            return (
                self._early_interruption_task is task
                and generation == self._speech_generation
                and binding_generation == self._session.binding_generation
                and not self._input_blocked
            )

        try:
            pcm = await asyncio.to_thread(snapshot)
            decoded = _coerce_asr_update(await asyncio.to_thread(recognize, pcm))
            # None means this adapter cannot decode a detached snapshot; it is
            # not an ASR rejection of the still-valid, buffered utterance.
            restore_output = decoded is not None
            decoded = decoded or AsrUpdate(text="", is_final=True)
            text = decoded.text.strip()
            if not is_current():
                return
            self._record_event(
                "asr_early_final",
                text=text or None,
                source="asr",
                metadata=({"asrEvent": decoded.event} if decoded.event else None),
            )
            reply_context = getattr(self._session, "reply_context", None)
            allow_nonlexical = (
                reply_context is not None
                and reply_context.subject_kind == "interaction"
            )
            rejection_reason = low_evidence_utterance_reason(
                text,
                decoded.event,
                allow_nonlexical=allow_nonlexical,
            )
            if rejection_reason is None and not _compact_text(text):
                rejection_reason = "punctuation_only"
            if rejection_reason is not None:
                rejection_metadata: dict[str, object] = {
                    "reason": rejection_reason,
                    "compactChars": len(_compact_text(text)),
                }
                if decoded.event:
                    rejection_metadata["asrEvent"] = decoded.event
                self._record_event(
                    "asr_early_final_rejected",
                    text=text or None,
                    source="asr",
                    metadata=rejection_metadata,
                )
                logger.info(
                    "忽略低证据 ASR early final: reason=%s, event=%r, text=%r",
                    rejection_reason,
                    decoded.event,
                    text,
                )
                return
            if (
                len(_compact_text(text)) < self._early_interrupt_min_chars
                and not allow_nonlexical
            ):
                return
            if self._session.is_likely_echo(text):
                logger.debug("忽略疑似回声 ASR early final: text=%r", text)
                return
            await self._confirm_user_speech(text, source="early_final")
        except asyncio.CancelledError:
            restore_output = True
            raise
        except Exception:
            restore_output = True
            logger.exception("SenseVoice 提前快照识别失败")
        finally:
            try:
                if (
                    restore_output
                    and is_current()
                    and not self._speech_interruption_confirmed
                ):
                    # A rejected/failed snapshot is not permission to keep TA
                    # paused until VAD eventually ends. Restore this candidate's
                    # reversible attenuation, retaining PCM for the final decode.
                    # A retired generation must never resume a newer candidate.
                    await self._session.resume_output_after_speech()
            finally:
                if self._early_interruption_task is task:
                    self._early_interruption_task = None

    def _cancel_early_interruption(self) -> None:
        task = self._early_interruption_task
        self._early_interruption_task = None
        if task is not None and not task.done():
            task.cancel()

    def _retire_early_interruption(self) -> None:
        # Snapshot decoding is non-destructive. Let an already-running worker
        # finish, but revoke its right to confirm this speech generation.
        self._early_interruption_task = None

    async def _reset_vad(self, reason: str) -> None:
        frame_processor = self._frame_processor
        if frame_processor is None:
            return
        reset = getattr(frame_processor, "reset_vad", None)
        if not callable(reset):
            return
        await asyncio.to_thread(reset)
        self._record_event(
            "vad_reset",
            source="vad",
            metadata={"reason": reason},
        )

    def _reset_vad_now(self, reason: str) -> None:
        frame_processor = self._frame_processor
        if frame_processor is None:
            return
        reset = getattr(frame_processor, "reset_vad", None)
        if not callable(reset):
            return
        reset()
        self._record_event(
            "vad_reset",
            source="vad",
            metadata={"reason": reason},
        )

    def _record_frame(
        self,
        capture_pcm: bytes,
        render_pcm: bytes,
        processed: ProcessedAudioFrame,
        *,
        processing_ms: float,
    ) -> int | None:
        diagnostics = self._diagnostics
        if diagnostics is None:
            return None
        try:
            return diagnostics.record_frame(
                capture_pcm,
                render_pcm,
                processed.cleaned_pcm,
                voice_detected=processed.voice_detected,
                processing_ms=processing_ms,
                dropped_frames=(
                    self._dropped_frames() if self._dropped_frames is not None else None
                ),
            )
        except Exception:
            logger.debug("Unable to record companion diagnostic frame", exc_info=True)
            return None

    def _record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "processor",
        metadata: dict[str, object] | None = None,
    ) -> None:
        diagnostics = self._diagnostics
        if diagnostics is None:
            return
        try:
            event_metadata: dict[str, object] = {
                "speechGeneration": self._speech_generation,
                "state": self._session.state.value,
            }
            if metadata is not None:
                event_metadata.update(metadata)
            diagnostics.record_event(
                name,
                frame_index=frame_index,
                text=text,
                source=source,
                metadata=event_metadata,
            )
        except Exception:
            logger.debug("Unable to record companion diagnostic event", exc_info=True)

    def _record_asr_decode_started(
        self,
        *,
        generation: int,
        decode_kind: str,
    ) -> int:
        self._record_event(
            "asr_decode_started",
            source="asr",
            metadata={
                "speechGeneration": generation,
                "decodeKind": decode_kind,
            },
        )
        return time.perf_counter_ns()

    def _record_asr_decode_completed(
        self,
        update: AsrUpdate | None,
        *,
        generation: int,
        decode_kind: str,
        started_ns: int,
    ) -> None:
        text = update.text if update is not None else ""
        self._record_event(
            "asr_decode_completed",
            source="asr",
            metadata={
                "speechGeneration": generation,
                "decodeKind": decode_kind,
                "durationMs": (time.perf_counter_ns() - started_ns) / 1_000_000,
                "charCount": len(text),
                "compactChars": len(_compact_text(text)),
                "asrEvent": update.event if update is not None else "",
            },
        )

    async def discard_input(self) -> None:
        """Drop what was being heard: the person stopped the listening."""
        await self._clear_input(discard_claimed=True)

    async def close(self) -> None:
        self._closed = True
        await self._clear_input(discard_claimed=False)

    async def _clear_input(self, *, discard_claimed: bool) -> None:
        self._owner_epoch += 1
        if self._owner_voice_gate is not None:
            self._owner_voice_gate.reset()
        frame_finish_task = self._frame_finish_task
        self._frame_finish_task = None
        try:
            if frame_finish_task is not None and not frame_finish_task.done():
                if self._frame_finish_claimed and not discard_claimed:
                    await asyncio.shield(frame_finish_task)
                else:
                    frame_finish_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await frame_finish_task
        finally:
            self._frame_finish_claimed = False
            self._frame_finish_error = None
            # A claimed finalization owns the frozen turn until it has either
            # transferred or released it. Abandoning first would clear that
            # owner and let the finishing coroutine freeze it a second time.
            self._abandon_voice_draft("shutdown")
            await self._session.resume_output_after_speech()
            task = self._commit_task
            self._commit_task = None
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            early_interruption_task = self._early_interruption_task
            self._early_interruption_task = None
            if (
                early_interruption_task is not None
                and not early_interruption_task.done()
            ):
                early_interruption_task.cancel()
                with suppress(asyncio.CancelledError):
                    await early_interruption_task
            pending_frozen_turn = self._pending_frozen_turn()
            self._pending_parts.clear()
            self._session.release_frozen_turn(pending_frozen_turn)
            self._release_foreground_reservation()
            self._speech_detector.reset()
            self._speech_pre_roll.clear()
            self._speech_candidate_requires_confirmation = False
            self._speech_interruption_confirmed = False
            self._asr_partial_recorded = False
            self._reset_interruption_partial()
            self._confirmed_interruption_text = None
            self._reset_near_end_confirmation()
            self._active_speech_frames = 0
            self._early_interruption_attempted = False

    def _open_or_reopen_voice_draft(self) -> None:
        self._voice_draft_transcript_reusable = False
        controller = self._voice_draft_controller
        gate = self._voice_draft_gate
        if controller is None or gate is None:
            return
        current = self._voice_draft_owner
        if current is None:
            self._voice_draft_pcm_limit_reached = False
            frozen_turn = self._frame_frozen_turn
            if frozen_turn is None:
                frozen_turn = self._session.freeze_turn_binding()
            if frozen_turn is None:
                gate.close_turn()
                return
            self._frame_frozen_turn = frozen_turn
            self._voice_draft_owner = controller.open_turn(
                fish_id=frozen_turn.fish_id,
                session_id=frozen_turn.session_id,
                binding_generation=frozen_turn.binding_generation,
                speech_generation=self._speech_generation,
                speech_started_at=datetime.now(UTC),
                reply_context=(
                    frozen_turn.reply_context.client_context
                    if frozen_turn.reply_context is not None
                    else None
                ),
            )
            self._voice_draft_dropped_frames_baseline = self._current_dropped_frames()
        else:
            # A new authoritative START after Smart Turn's END is resumed
            # speech in the same logical voice turn. Revision fences the old
            # candidate; speech generation fences only the endpoint attempt.
            controller.resume(current, source="primary_start")
            self._voice_draft_owner = controller.reopen(
                speech_generation=self._speech_generation,
            )
        gate.open_turn()

    def _process_voice_draft_frame(
        self, pcm: bytes, *, candidate_voice_detected: bool | None = None
    ) -> None:
        vad = self._voice_draft_vad
        gate = self._voice_draft_gate
        controller = self._voice_draft_controller
        owner = self._voice_draft_owner
        if gate is None or controller is None or owner is None:
            return
        source = "primary_inference"
        if candidate_voice_detected is None:
            if vad is None:
                return
            candidate_voice_detected = vad.is_speech(pcm)
            source = "fast_vad"
        # The primary model's per-window activity excludes its endpoint hold.
        # It lets the candidate's shorter silence window anticipate that same
        # model's END without a second classifier disagreeing about silence.
        event = gate.accept(candidate_voice_detected)
        if event is CandidateEvent.RESUMED:
            self._voice_draft_transcript_reusable = False
            self._voice_draft_owner = controller.resume(owner, source=source)
            return
        if event not in {CandidateEvent.PREPARE, CandidateEvent.CANDIDATE}:
            return
        if self._voice_draft_pcm_limit_reached:
            return
        max_pcm_bytes = self._voice_draft_max_pcm_bytes
        buffered_pcm_bytes = getattr(self._asr, "buffered_pcm_bytes", None)
        if max_pcm_bytes is not None and callable(buffered_pcm_bytes):
            pcm_bytes = buffered_pcm_bytes()
            if pcm_bytes > max_pcm_bytes:
                self._voice_draft_pcm_limit_reached = True
                self._invalidate_voice_draft_transcript("buffer_limit")
                self._record_voice_draft_status(
                    "voice_draft_fallback",
                    owner,
                    {"reason": "buffer_limit", "pcmBytes": pcm_bytes},
                )
                return
        self._record_voice_draft_status(
            (
                "voice_draft_asr_prepare_triggered"
                if event is CandidateEvent.PREPARE
                else "voice_draft_candidate_triggered"
            ),
            owner,
            {"reason": source},
        )
        snapshot = getattr(self._asr, "snapshot_utterance", None)
        if not callable(snapshot):
            self._record_voice_draft_status(
                "voice_draft_fallback", owner, {"reason": "snapshot_unavailable"}
            )
            return
        candidate_pcm = snapshot()
        if type(candidate_pcm) is bytes and candidate_pcm:
            if max_pcm_bytes is not None and len(candidate_pcm) > max_pcm_bytes:
                # Compatibility fallback for ASR implementations that do not
                # expose the O(1) size probe. Production StreamingAsr takes
                # the branch above and avoids this full-buffer copy.
                self._voice_draft_pcm_limit_reached = True
                self._invalidate_voice_draft_transcript("buffer_limit")
                self._record_voice_draft_status(
                    "voice_draft_fallback",
                    owner,
                    {"reason": "buffer_limit", "pcmBytes": len(candidate_pcm)},
                )
                return
            overload = self._voice_draft_overload()
            if overload is not None:
                reason, queue_age_ms, queue_depth, dropped_frames = overload
                reuse_preparation = (
                    reason == "audio_queue_age"
                    and event is CandidateEvent.CANDIDATE
                    and self._voice_draft_transcript_reusable
                    and controller.has_preparation(owner, pcm=candidate_pcm)
                )
                if not reuse_preparation:
                    if reason == "audio_frame_drop":
                        self._invalidate_voice_draft_transcript(reason)
                    controller.skip_candidate(
                        owner,
                        reason=reason,
                        queue_age_ms=queue_age_ms,
                        queue_depth=queue_depth,
                        dropped_frames=dropped_frames,
                    )
                    return
            # Preparation starts recognition only. The later candidate event
            # releases the same result/task to the server once silence holds.
            # Watch every subsequent frame from this earlier PCM cutoff, so a
            # continuation or an audio-track change revokes reuse immediately.
            start_stage = (
                controller.prepare
                if event is CandidateEvent.PREPARE
                else controller.candidate
            )
            self._voice_draft_transcript_reusable = (
                start_stage(owner, pcm=candidate_pcm) is not None
            )
        else:
            self._record_voice_draft_status(
                "voice_draft_fallback", owner, {"reason": "empty_pcm"}
            )

    def _observe_voice_draft_tail(
        self,
        pcm: bytes,
        *,
        boundary_voice: bool,
        candidate_voice_detected: bool | None = None,
    ) -> None:
        # Keep observing during END/Smart Turn's wait. A short continuation
        # may not yet satisfy the primary VAD's START confirmation threshold.
        # It revokes transcript reuse without changing the authoritative turn.
        reason = "primary_inference_tail"
        if candidate_voice_detected is None:
            vad = self._voice_draft_vad
            if vad is None:
                return
            candidate_voice_detected = vad.is_speech(pcm)
            reason = "fast_vad_tail"
        if boundary_voice or candidate_voice_detected:
            self._invalidate_voice_draft_transcript(
                "primary_vad_tail" if boundary_voice else reason
            )

    def _invalidate_voice_draft_transcript(
        self, reason: str = "reuse_invalidated"
    ) -> None:
        if not self._voice_draft_transcript_reusable:
            return
        self._voice_draft_transcript_reusable = False
        controller = self._voice_draft_controller
        owner = self._voice_draft_owner
        if controller is not None and owner is not None:
            controller.invalidate_transcript(owner, reason=reason)

    def _can_reuse_voice_draft_transcript(self, owner: VoiceDraftOwner) -> bool:
        reason = self._voice_draft_reuse_block_reason(owner)
        if reason is not None and self._voice_draft_controller is not None:
            self._voice_draft_controller.record_reuse_miss(owner, reason)
        return reason is None

    def _voice_draft_reuse_block_reason(self, owner: VoiceDraftOwner) -> str | None:
        current_frame = asyncio.current_task()
        owner_gate = self._owner_voice_gate
        if not self._voice_draft_transcript_reusable:
            return "reuse_not_available"
        if self._input_blocked:
            return "closed"
        if owner.binding_generation != self._session.binding_generation:
            return "binding_changed"
        if owner.speech_generation != self._speech_generation:
            return "speech_generation_changed"
        # Queue age limits starting optional work. Reusing a decode that is
        # already running or complete adds no decoder load, and queue age alone
        # does not invalidate its PCM prefix. Missing audio still does.
        baseline = self._voice_draft_dropped_frames_baseline
        if baseline is not None and self._current_dropped_frames() > baseline:
            return "audio_frame_drop"
        if any(task is not current_frame for task in self._voice_draft_inflight_frames):
            return "frame_in_flight"
        if owner_gate is not None:
            if self._voice_draft_capture_vad is None:
                return "capture_vad_unavailable"
            if owner_gate.pending_speech:
                return "owner_pending_speech"
        return None

    def _record_voice_draft_status(
        self,
        name: str,
        owner: VoiceDraftOwner,
        metadata: dict[str, object] | None = None,
    ) -> None:
        controller = self._voice_draft_controller
        current = controller.current_owner(owner) if controller is not None else None
        current = current or owner
        record_voice_draft_event(
            self._diagnostics,
            name,
            metadata={
                "turnId": str(current.turn_id),
                "revision": current.revision,
                "speechGeneration": current.speech_generation,
                "candidateEpoch": current.candidate_epoch,
                **(metadata or {}),
            },
        )

    def _close_voice_draft_gate(
        self, reason: str, owner: VoiceDraftOwner | None = None
    ) -> None:
        gate = self._voice_draft_gate
        if gate is None:
            return
        summary = gate.take_summary()
        owner = owner or self._voice_draft_owner
        if summary is not None and owner is not None:
            self._record_voice_draft_status(
                "voice_draft_gate_summary", owner, {"reason": reason, **summary}
            )
        gate.close_turn()

    async def _submit_voice_draft(
        self,
        content: str,
        frozen_turn: FrozenVoiceTurn | None,
    ) -> bool:
        controller = self._voice_draft_controller
        owner = self._voice_draft_owner
        if controller is None or owner is None:
            return await self._session.submit_transcript(
                content,
                frozen_turn=frozen_turn,
            )
        commit_owner = controller.current_owner(owner)
        if commit_owner is None:
            return False
        if (
            self._input_blocked
            or self._voice_draft_owner != owner
            or not controller.is_current(commit_owner)
            or commit_owner.speech_generation != self._speech_generation
            or commit_owner.binding_generation != self._session.binding_generation
        ):
            return False
        submission = controller.commit(commit_owner, content=content)
        self._voice_draft_transcript_reusable = False
        self._voice_draft_pcm_limit_reached = False
        self._voice_draft_owner = None
        self._voice_draft_dropped_frames_baseline = None
        self._close_voice_draft_gate("committed", commit_owner)
        try:
            accepted = await self._session.submit_voice_turn(
                submission.key.turn_id,
                submission.revision,
                submission.promote_revision,
                submission.content,
                frozen_turn=frozen_turn,
            )
        except BaseException:
            controller.cancel_submission(submission, reason="binding_changed")
            raise
        if not accepted:
            controller.cancel_submission(submission, reason="binding_changed")
        return accepted

    async def _wait_voice_draft_registration(
        self, owner: VoiceDraftOwner | None, *, generation: int
    ) -> bool:
        """Wait before consuming PCM or handing off the frozen reply reservation."""

        controller = self._voice_draft_controller
        if controller is None or owner is None:
            return True
        exact_owner = controller.current_owner(owner)
        if exact_owner is None:
            return False
        await controller.wait_for_registration(exact_owner)
        return (
            not self._input_blocked
            and self._voice_draft_owner == owner
            and controller.is_current(exact_owner)
            and generation == self._speech_generation
            and exact_owner.binding_generation == self._session.binding_generation
        )

    def _abandon_voice_draft(self, reason: str) -> None:
        self._voice_draft_transcript_reusable = False
        self._voice_draft_pcm_limit_reached = False
        controller = self._voice_draft_controller
        owner = self._voice_draft_owner
        frozen_turn = self._frame_frozen_turn
        self._frame_frozen_turn = None
        if frozen_turn is not None:
            self._session.release_frozen_turn(frozen_turn)
        # Retire local ownership before the best-effort HTTP cancel can yield.
        self._voice_draft_owner = None
        self._voice_draft_dropped_frames_baseline = None
        self._close_voice_draft_gate("abandoned", owner)
        if controller is None or owner is None:
            return
        try:
            controller.abandon(owner, reason=reason)
        except RuntimeError:
            logger.debug("Voice draft owner was already retired", exc_info=True)

    def _current_dropped_frames(self) -> int:
        callback = self._dropped_frames
        if callback is None:
            return 0
        try:
            return max(0, int(callback()))
        except Exception:
            logger.debug("Unable to read audio dropped-frame telemetry", exc_info=True)
            return 0

    def _voice_draft_overload(self) -> tuple[str, float, int, int] | None:
        try:
            queue_age_ms = (
                max(0.0, float(self._frame_queue_age_ms()))
                if self._frame_queue_age_ms is not None
                else 0.0
            )
            queue_depth = (
                max(0, int(self._frame_queue_depth()))
                if self._frame_queue_depth is not None
                else 0
            )
        except Exception:
            logger.debug("Unable to read audio queue telemetry", exc_info=True)
            queue_age_ms = 0.0
            queue_depth = 0
        dropped_frames = self._current_dropped_frames()
        baseline = self._voice_draft_dropped_frames_baseline
        if baseline is not None and dropped_frames > baseline:
            return "audio_frame_drop", queue_age_ms, queue_depth, dropped_frames
        if queue_age_ms > _VOICE_DRAFT_MAX_QUEUE_AGE_MS:
            return "audio_queue_age", queue_age_ms, queue_depth, dropped_frames
        return None

    def _observe_interruption_partial(self, text: str) -> bool:
        candidate = _compact_text(text)
        if candidate == self._interruption_partial_text:
            self._interruption_partial_repeats += 1
        else:
            self._interruption_partial_text = candidate
            self._interruption_partial_repeats = 1
        # A single unstable partial is common while the recognizer is still
        # decoding. Require the same meaningful partial twice before it can
        # stop an answer; final ASR results still confirm immediately.
        return self._interruption_partial_repeats >= 2

    def _reset_interruption_partial(self) -> None:
        self._interruption_partial_text = None
        self._interruption_partial_repeats = 0

    def _reset_near_end_confirmation(self) -> None:
        self._interruption_requires_near_end = False
        self._interruption_near_end_confirmed = False
        self._post_pause_echo_frames_remaining = 0

    def _schedule_pending_commit(self) -> None:
        self._cancel_pending_commit()
        self._commit_task = asyncio.create_task(self._commit_pending_utterance())

    def _schedule_frame_finish(self) -> None:
        self._cancel_frame_finish()
        self._frame_finish_claimed = False
        self._record_event(
            "finish_scheduled",
            metadata={
                "commitDelayMs": self._utterance_commit_delay_seconds * 1_000,
                "smartTurnEnabled": self._smart_turn is not None,
                "requiresConfirmation": self._speech_candidate_requires_confirmation,
            },
        )
        self._frame_finish_task = asyncio.create_task(
            self._finish_frame_utterance_after_delay(self._speech_generation)
        )

    def _cancel_frame_finish(
        self,
        *,
        reason: str | None = None,
        generation: int | None = None,
    ) -> None:
        task = self._frame_finish_task
        if self._frame_finish_claimed and task is not None and not task.done():
            return
        self._frame_finish_task = None
        self._frame_finish_claimed = False
        if task is not None and not task.done():
            if reason is not None:
                self._record_event(
                    "finish_cancelled",
                    metadata={
                        "reason": reason,
                        "speechGeneration": (
                            self._speech_generation
                            if generation is None
                            else generation
                        ),
                        "currentSpeechGeneration": self._speech_generation,
                    },
                )
            task.cancel()

    async def _wait_for_claimed_frame_finish(self) -> None:
        task = self._frame_finish_task
        if (
            not self._frame_finish_claimed
            or task is None
            or task.done()
            or task is asyncio.current_task()
        ):
            return
        await asyncio.shield(task)

    def _cancel_pending_commit(self) -> None:
        task = self._commit_task
        self._commit_task = None
        if task is not None and not task.done():
            task.cancel()

    async def _commit_pending_utterance(self) -> None:
        try:
            await asyncio.sleep(self._utterance_commit_delay_seconds)
            await self._flush_pending_utterance()
        except asyncio.CancelledError:
            raise
        finally:
            if self._commit_task is asyncio.current_task():
                self._commit_task = None

    async def _finish_frame_utterance_after_delay(self, generation: int) -> None:
        binding_generation = self._session.binding_generation
        try:
            started = time.monotonic()
            delay = self._utterance_commit_delay_seconds
            wait_reason = "commit_delay"
            pcm = await asyncio.to_thread(self._asr.snapshot_utterance)
            smart_turn = self._smart_turn
            if smart_turn is not None:
                self._record_event(
                    "finish_waiting",
                    metadata={
                        "speechGeneration": generation,
                        "reason": "smart_turn_inference",
                        "timeoutMs": self._smart_turn_timeout_seconds * 1_000,
                    },
                )
                result = await self._predict_smart_turn(smart_turn, pcm)
                if result is not None:
                    self._record_event(
                        "smart_turn_result",
                        source="smart_turn",
                        metadata={
                            "speechGeneration": generation,
                            "complete": result.complete,
                            "probability": result.probability,
                            "inferenceMs": result.inference_ms,
                        },
                    )
                    if not result.complete:
                        delay = max(delay, self._smart_turn_reopen_seconds)
                        wait_reason = "smart_turn_incomplete"
            remaining = delay - (time.monotonic() - started)
            if remaining > 0:
                self._record_event(
                    "finish_waiting",
                    metadata={
                        "speechGeneration": generation,
                        "reason": wait_reason,
                        "remainingMs": remaining * 1_000,
                    },
                )
                await asyncio.sleep(remaining)
            owner_gate = self._owner_voice_gate
            if owner_gate is not None and owner_gate.pending_speech:
                # Speaker admission buffers untrusted speech. Give an already
                # started short reopen one bounded chance to identify itself;
                # background dialogue must never postpone an answer forever.
                await owner_gate.wait_until_settled()
            if generation != self._speech_generation:
                self._record_event(
                    "finish_discarded",
                    metadata={
                        "reason": "stale_generation",
                        "speechGeneration": generation,
                        "currentSpeechGeneration": self._speech_generation,
                    },
                )
                return
            await self._finish_utterance_snapshot(
                record_speech_end=False,
                pcm_snapshot=pcm,
                expected_generation=generation,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if (
                generation == self._speech_generation
                and binding_generation == self._session.binding_generation
                and not self._input_blocked
            ):
                self._frame_finish_error = error
                await self._session.resume_output_after_speech()
                self._record_event(
                    "asr_final_failed",
                    source="asr",
                    metadata={"errorType": type(error).__name__},
                )
                logger.exception("ASR finalization failed; restarting the audio stream")
        finally:
            if self._frame_finish_task is asyncio.current_task():
                self._frame_finish_task = None
                self._frame_finish_claimed = False

    async def _predict_smart_turn(
        self,
        smart_turn: SmartTurnInput,
        pcm: bytes,
    ) -> SmartTurnResultInput | None:
        in_flight = self._smart_turn_inference_task
        if in_flight is not None and not in_flight.done():
            self._disable_smart_turn(RuntimeError("previous inference still running"))
            return None
        task = asyncio.create_task(asyncio.to_thread(smart_turn.predict_pcm16, pcm))
        self._smart_turn_inference_task = task
        task.add_done_callback(self._on_smart_turn_inference_done)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._smart_turn_timeout_seconds,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            self._disable_smart_turn(error)
            return None
        except Exception as error:
            self._disable_smart_turn(error)
            return None

    def _disable_smart_turn(self, error: Exception) -> None:
        logger.warning("Smart Turn 推理失败，本次会话退回 VAD 端点: %s", error)
        self._smart_turn = None
        self._record_event(
            "smart_turn_failed",
            source="smart_turn",
            metadata={"error": type(error).__name__},
        )

    def _on_smart_turn_inference_done(
        self,
        task: asyncio.Task[SmartTurnResultInput],
    ) -> None:
        if self._smart_turn_inference_task is task:
            self._smart_turn_inference_task = None
        with suppress(asyncio.CancelledError, Exception):
            task.result()

    async def _flush_pending_utterance(self) -> None:
        parts = tuple(self._pending_parts)
        self._pending_parts.clear()
        content = _join_transcript_parts([part.content for part in parts])
        frozen_turn = parts[0].frozen_turn if parts else None
        try:
            if content:
                await self._session.submit_transcript(
                    content,
                    frozen_turn=frozen_turn,
                )
        finally:
            self._release_foreground_reservation()

    def _pending_frozen_turn(self) -> FrozenVoiceTurn | None:
        return self._pending_parts[0].frozen_turn if self._pending_parts else None

    def _pending_binding_is_current(self) -> bool:
        frozen = self._pending_frozen_turn()
        if frozen is None:
            return not self._pending_parts
        return (
            frozen.fish_id == self._session.fish_id
            and frozen.session_id == self._session.session_id
            and frozen.voice_conversation_id == self._session.voice_conversation_id
            and frozen.binding_generation == self._session.binding_generation
        )

    def _ensure_foreground_reservation(self) -> None:
        if self._foreground_reservation is None:
            self._foreground_reservation = self._session.reserve_foreground_speech()

    def _release_foreground_reservation(self) -> None:
        reservation = self._foreground_reservation
        self._foreground_reservation = None
        if reservation is not None:
            self._session.release_foreground_speech(reservation)


def _join_transcript_parts(parts: Sequence[str]) -> str:
    content = ""
    for part in parts:
        if (
            content
            and content[-1].isascii()
            and content[-1].isalnum()
            and part[0].isascii()
            and part[0].isalnum()
        ):
            content += " "
        content += part
    return content


def _compact_text(value: str) -> str:
    return "".join(
        character.casefold()
        for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _coerce_asr_update(value: object) -> AsrUpdate | None:
    if value is None:
        return None
    if isinstance(value, AsrUpdate):
        return value
    return AsrUpdate(text=str(value).strip(), is_final=True)


def _prefer_complete_transcript(final_text: str, stable_partial: str | None) -> str:
    final = final_text.strip()
    partial = stable_partial.strip() if stable_partial else ""
    if not final:
        return partial
    if not partial:
        return final
    compact_final = _compact_text(final)
    compact_partial = _compact_text(partial)
    if len(compact_final) < len(compact_partial) and compact_final in compact_partial:
        return partial
    return final
