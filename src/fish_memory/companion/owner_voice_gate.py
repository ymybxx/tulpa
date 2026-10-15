"""Bounded, local speaker admission before any conversational side effects."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fish_memory.companion.audio_processing import ProcessedAudioFrame

_SILENCE = bytes(320)
SAMPLE_BYTES_PER_SECOND = 32_000
# A near miss worth offering to the owner. Live, the owner's own misses scored
# 0.43-0.49 while other voices and noise stayed at or below 0.16.
UNRECOGNIZED_MIN_SCORE = 0.3
UNRECOGNIZED_MAX_BYTES = 20 * SAMPLE_BYTES_PER_SECOND
UNRECOGNIZED_KEEP_SECONDS = 300.0


class OwnerVoiceVerifier(Protocol):
    def score_pcm(self, pcm: bytes) -> float: ...


@dataclass(frozen=True, slots=True)
class OwnerVoiceFrame:
    processed: ProcessedAudioFrame
    frame_index: int | None = None


@dataclass(frozen=True, slots=True)
class OwnerVoiceAdmission:
    frames: tuple[OwnerVoiceFrame, ...] = ()
    revoked: bool = False
    reason: str | None = None
    score: float | None = None
    inference_ms: float | None = None


class OwnerVoiceGate:
    """Verify each small audio block; a match never unlocks an entire session.

    Only checked PCM is released, in order, to the existing KWS/ASR/VAD turn
    machine. Short tails may use at most one recent checked block as context.
    These initial thresholds are product defaults, not calibrated accuracy
    claims. No audio, embeddings, or recognized text are persisted here; near
    misses stay in memory for a few minutes so the owner can add them.
    """

    def __init__(
        self,
        verifier: OwnerVoiceVerifier,
        *,
        threshold: float = 0.5,
        check_speech_ms: int = 800,
        minimum_speech_ms: int = 300,
        max_buffer_ms: int = 3000,
        context_speech_ms: int = 2200,
        inference_timeout: float = 0.4,
        on_unrecognized: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Owner voice threshold must be finite and in (0, 1)")
        if not 10 <= minimum_speech_ms <= check_speech_ms <= max_buffer_ms:
            raise ValueError("Invalid owner voice audio window")
        if context_speech_ms < check_speech_ms:
            raise ValueError("Owner voice context must cover a check block")
        if not math.isfinite(inference_timeout) or inference_timeout <= 0:
            raise ValueError("Owner voice timeout must be positive")
        self._verifier = verifier
        self._threshold = threshold
        self._check_frames = math.ceil(check_speech_ms / 10)
        self._minimum_frames = math.ceil(minimum_speech_ms / 10)
        self._max_frames = math.ceil(max_buffer_ms / 10)
        self._context_frames = math.ceil(context_speech_ms / 10)
        self._timeout = inference_timeout
        # Signal frames the next check waits for; a miss before any match
        # waits for more speech instead of dropping what was said.
        self._next_check = self._check_frames
        self._missed_at = 0
        self._pending: deque[tuple[OwnerVoiceFrame, bool]] = deque()
        self._prefix: deque[OwnerVoiceFrame] = deque(maxlen=22)
        self._context = b""
        self._signal_frames = 0
        self._active = False
        self._admitted = False
        self._generation = 0
        self._inference: asyncio.Task[float] | None = None
        self._settled = asyncio.Event()
        self._settled.set()
        self._last_score: float | None = None
        # Near misses the owner may choose to add to the voiceprint. Memory
        # only, bounded, and forgotten after a few minutes.
        # (kept at, speech frames for the voiceprint, the utterance as heard).
        self._unrecognized: deque[tuple[float, bytes, bytes]] = deque()
        self._on_unrecognized = on_unrecognized
        self._clock = clock

    def replace_verifier(self, verifier: OwnerVoiceVerifier) -> None:
        """Use a voiceprint that was just extended, from the next check on."""
        self._verifier = verifier

    def unrecognized_seconds(self) -> float:
        self._forget_old_unrecognized()
        return (
            sum(len(pcm) for _, pcm, _ in self._unrecognized) / SAMPLE_BYTES_PER_SECOND
        )

    def take_unrecognized(self) -> tuple[bytes, list[bytes]]:
        """Hand over the kept near misses once: the speech for the voiceprint,
        and each utterance whole (pauses included) for the owner to listen to."""
        self._forget_old_unrecognized()
        speech = b"".join(pcm for _, pcm, _ in self._unrecognized)
        heard = [utterance for _, _, utterance in self._unrecognized]
        self._unrecognized.clear()
        self._notify_unrecognized()
        return speech, heard

    def _keep_unrecognized(self) -> None:
        """Keep this utterance's unreleased speech when it only just missed."""
        score = self._last_score
        if score is None or score < UNRECOGNIZED_MIN_SCORE:
            return
        speech = b"".join(
            item.processed.cleaned_pcm for item, signal in self._pending if signal
        )
        if not speech:
            return
        self._forget_old_unrecognized()
        heard = b"".join(item.processed.cleaned_pcm for item, _ in self._pending)
        self._unrecognized.append((self._clock(), speech, heard))
        while (
            sum(len(pcm) for _, pcm, _ in self._unrecognized) > UNRECOGNIZED_MAX_BYTES
        ):
            self._unrecognized.popleft()
        self._notify_unrecognized()

    def _forget_old_unrecognized(self) -> None:
        cutoff = self._clock() - UNRECOGNIZED_KEEP_SECONDS
        while self._unrecognized and self._unrecognized[0][0] < cutoff:
            self._unrecognized.popleft()

    def _notify_unrecognized(self) -> None:
        if self._on_unrecognized is not None:
            self._on_unrecognized(self.unrecognized_seconds())

    @property
    def buffered_frames(self) -> int:
        return len(self._pending)

    @property
    def pending_speech(self) -> bool:
        return self._active and self._signal_frames > 0

    async def wait_until_settled(self) -> None:
        try:
            await asyncio.wait_for(self._settled.wait(), 0.4)
        except TimeoutError:
            pass

    def reset(self) -> None:
        # A thread cannot be forcibly canceled. Fence its result and keep its
        # task until done so a timeout/reset cannot start unbounded inference.
        self._generation += 1
        self._pending.clear()
        self._prefix.clear()
        self._context = b""
        self._signal_frames = 0
        self._next_check = self._check_frames
        self._missed_at = 0
        self._last_score = None
        self._active = False
        self._admitted = False
        self._settled.set()

    async def accept(
        self, frame: OwnerVoiceFrame, *, finalization_pending: bool = False
    ) -> OwnerVoiceAdmission:
        pcm = frame.processed.cleaned_pcm
        if len(pcm) != 320:
            raise ValueError("Owner voice requires 10 ms mono PCM16 frames")
        speech = frame.processed.voice_detected
        signal = speech and _has_signal(pcm)
        if not self._active and not speech:
            self._prefix.append(frame)
            # Preserve silence boundaries without feeding unchecked capture to
            # an ASR tail or a KWS decoder still containing an earlier prefix.
            return OwnerVoiceAdmission(frames=(_silence(frame),))
        if not self._active:
            self._active = True
            self._pending.extend((item, False) for item in self._prefix)
            self._prefix.clear()
        self._pending.append((frame, signal))
        self._signal_frames += int(signal)
        if self._signal_frames:
            self._settled.clear()
        if len(self._pending) > self._max_frames:
            # Discard the entire uncertain block rather than splice together
            # disconnected fragments or retain stale speech indefinitely.
            revoked = self._admitted
            self.reset()
            return OwnerVoiceAdmission(revoked=revoked, reason="buffer_limit")
        ended = not speech
        # A short verified reopen must reach Smart Turn before its existing
        # deadline; do not hide it behind the full first-block window.
        check_frames = (
            self._minimum_frames if finalization_pending else self._next_check
        )
        if not ended and self._signal_frames < check_frames:
            return OwnerVoiceAdmission()
        if ended and self._signal_frames == 0 and self._admitted:
            frames = tuple(
                OwnerVoiceFrame(
                    ProcessedAudioFrame(
                        _SILENCE,
                        item.processed.voice_detected,
                        candidate_voice_detected=False
                        if item.processed.candidate_voice_detected is not None
                        else None,
                    ),
                    item.frame_index,
                )
                for item, _ in self._pending
            )
            self.reset()
            return OwnerVoiceAdmission(frames=frames)
        if self._signal_frames < self._minimum_frames and not self._context:
            self.reset()
            return OwnerVoiceAdmission(reason="insufficient_speech")
        if ended and self._missed_at and self._signal_frames == self._missed_at:
            # Nothing said since the last miss: the same audio would score the same.
            self._keep_unrecognized()
            self.reset()
            return OwnerVoiceAdmission(reason="not_owner")
        frames = tuple(item for item, _ in self._pending)
        candidate = b"".join(item.processed.cleaned_pcm for item in frames)
        if self._context:
            # Every block is still checked, together with the speech of this
            # utterance already checked: live, 0.8 s blocks of the owner's own
            # voice scored from -0.04 to 0.74 against a 0.5 threshold, and a
            # single low one dropped the whole sentence. Bounded, so a long
            # owner monologue cannot dominate a new voice. This is still
            # verification, not source separation.
            candidate = self._context[-self._context_frames * 320 :] + candidate
        generation = self._generation
        started = time.perf_counter_ns()
        score: float | None = None
        reason = "matched"
        pending = self._inference
        try:
            if pending is not None and not pending.done():
                reason = "inference_busy"
            else:
                pending = asyncio.create_task(
                    asyncio.to_thread(self._verifier.score_pcm, candidate)
                )
                pending.add_done_callback(_consume_inference_exception)
                self._inference = pending
                score = float(
                    await asyncio.wait_for(asyncio.shield(pending), self._timeout)
                )
                if not math.isfinite(score) or not -1 <= score <= 1:
                    reason = "invalid_score"
                    score = None
                elif score < self._threshold:
                    reason = "not_owner"
        except TimeoutError:
            reason = "inference_timeout"
        except asyncio.CancelledError:
            self.reset()
            raise
        except Exception:
            reason = "inference_failed"
        if generation != self._generation:
            return OwnerVoiceAdmission(reason="stale_result")
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if score is not None:
            self._last_score = score
        if reason == "not_owner" and not self._admitted and not ended:
            # Nothing released yet: wait for more of this utterance and check
            # the longer stretch, rather than dropping it on a short block.
            self._next_check = self._signal_frames + self._check_frames
            self._missed_at = self._signal_frames
            self._settled.set()
            return OwnerVoiceAdmission(reason=reason, score=score, inference_ms=elapsed)
        if reason != "matched":
            revoked = self._admitted
            if reason == "not_owner":
                self._keep_unrecognized()
            self.reset()
            return OwnerVoiceAdmission(
                revoked=revoked, reason=reason, score=score, inference_ms=elapsed
            )
        self._admitted = True
        self._context = candidate[-self._context_frames * 320 :]
        self._pending.clear()
        self._signal_frames = 0
        self._next_check = self._check_frames
        self._settled.set()
        if ended:
            self.reset()
        return OwnerVoiceAdmission(
            frames=frames, reason="matched", score=score, inference_ms=elapsed
        )


def _has_signal(pcm: bytes) -> bool:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    # Exclude near-digital silence only; this is not a replacement VAD.
    return float(np.sqrt(np.mean(samples * samples))) >= 0.001


def _silence(frame: OwnerVoiceFrame) -> OwnerVoiceFrame:
    return OwnerVoiceFrame(
        ProcessedAudioFrame(
            _SILENCE,
            False,
            candidate_voice_detected=False
            if frame.processed.candidate_voice_detected is not None
            else None,
        ),
        frame.frame_index,
    )


def _consume_inference_exception(task: asyncio.Task[float]) -> None:
    if not task.cancelled():
        task.exception()
