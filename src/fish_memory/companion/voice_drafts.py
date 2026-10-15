"""Speculative Companion voice-turn primitives with local ownership fencing."""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Protocol, TypeVar, cast
from uuid import UUID, uuid4

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.client import CompanionReplyContext
from fish_memory.companion.diagnostics import (
    AudioDiagnosticsObserver,
    record_voice_draft_event,
)
from fish_memory.companion.utterance_admission import (
    low_evidence_utterance_reason,
)

logger = logging.getLogger(__name__)


class CandidateEvent(Enum):
    NONE = auto()
    PREPARE = auto()
    CANDIDATE = auto()
    RESUMED = auto()


class SpeculativeSilenceGate:
    """Debounce a separate fast-VAD stream without owning a speech boundary."""

    def __init__(
        self, silence_frames: int = 20, *, prepare_silence_frames: int | None = None
    ) -> None:
        if silence_frames < 1:
            raise ValueError("silence_frames must be positive")
        if (
            prepare_silence_frames is not None
            and not 1 <= prepare_silence_frames <= silence_frames
        ):
            raise ValueError(
                "prepare_silence_frames must be within the candidate window"
            )
        self._silence_frames = silence_frames
        self._prepare_silence_frames = prepare_silence_frames
        self._turn_open = False
        self._silent_count = 0
        self._preparation_open = False
        self._candidate_open = False
        self._summary_pending = False
        self._observed_frames = 0
        self._voiced_frames = 0
        self._observed_silence = 0
        self._longest_silence = 0
        self._candidate_count = 0
        self._resumed_count = 0

    def open_turn(self) -> None:
        self._turn_open = True
        self._silent_count = 0
        self._preparation_open = False
        self._candidate_open = False
        self._summary_pending = True
        self._observed_frames = self._voiced_frames = 0
        self._observed_silence = self._longest_silence = 0
        self._candidate_count = self._resumed_count = 0

    def close_turn(self) -> None:
        self._turn_open = False
        self._silent_count = 0
        self._preparation_open = False
        self._candidate_open = False

    def accept(self, speech: bool) -> CandidateEvent:
        if not self._turn_open:
            return CandidateEvent.NONE
        self._observed_frames += 1
        if speech:
            self._voiced_frames += 1
            self._observed_silence = 0
        else:
            self._observed_silence += 1
            self._longest_silence = max(self._longest_silence, self._observed_silence)
        if speech:
            self._silent_count = 0
            if self._candidate_open or self._preparation_open:
                self._preparation_open = False
                self._candidate_open = False
                self._resumed_count += 1
                return CandidateEvent.RESUMED
            return CandidateEvent.NONE
        if self._candidate_open:
            return CandidateEvent.NONE
        self._silent_count += 1
        if self._silent_count == self._silence_frames:
            self._candidate_open = True
            self._candidate_count += 1
            return CandidateEvent.CANDIDATE
        if self._silent_count == self._prepare_silence_frames:
            self._preparation_open = True
            return CandidateEvent.PREPARE
        return CandidateEvent.NONE

    def take_summary(self) -> dict[str, int] | None:
        """Read one content-free summary without changing gate decisions."""
        if not self._summary_pending:
            return None
        self._summary_pending = False
        return {
            "gateFrames": self._observed_frames,
            "gateSpeechFrames": self._voiced_frames,
            "gateSilentFrames": self._observed_silence,
            "gateLongestSilenceFrames": self._longest_silence,
            "gateRequiredSilenceFrames": self._silence_frames,
            "gateCandidateCount": self._candidate_count,
            "gateResumedCount": self._resumed_count,
        }


class VoiceDraftDecoder(Protocol):
    def recognize(self, pcm: bytes) -> str: ...


class VoiceDraftClient(Protocol):
    async def upsert_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        reply_context: CompanionReplyContext | None = None,
        speech_started_at: datetime | None = None,
        frame_snapshot_at: datetime | None = None,
    ) -> bool: ...

    async def cancel_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        reason: str,
    ) -> bool: ...


class VoiceDraftTts(Protocol):
    async def start(self, key: VoiceDraftKey, *, revision: int) -> bool: ...

    def mark_committed(self, key: VoiceDraftKey, *, revision: int) -> bool: ...

    async def cancel(
        self,
        key: VoiceDraftKey,
        *,
        through_revision: int,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class VoiceDraftOwner:
    turn_id: UUID
    revision: int
    candidate_epoch: int
    speech_generation: int
    binding_generation: int
    fish_id: UUID
    session_id: UUID
    reply_context: CompanionReplyContext | None = None
    speech_started_at: datetime | None = None

    @property
    def key(self) -> VoiceDraftKey:
        return VoiceDraftKey(self.fish_id, self.session_id, self.turn_id)


@dataclass(frozen=True, slots=True)
class VoiceDraftSubmission:
    """Immutable, single-use parameters for the authoritative session handoff."""

    key: VoiceDraftKey
    revision: int
    content: str
    promote_revision: int
    draft_request_started: bool = False


@dataclass(frozen=True, slots=True)
class _NativeDecodeSlot:
    started: threading.Event
    finished: threading.Event


@dataclass(frozen=True, slots=True)
class _CachedDraftTranscript:
    owner: VoiceDraftOwner
    pcm: bytes
    update: AsrUpdate


@dataclass(frozen=True, slots=True)
class _PendingDraftTranscript:
    owner: VoiceDraftOwner
    pcm: bytes
    slot: _NativeDecodeSlot
    ready: asyncio.Future[bool]


@dataclass(frozen=True, slots=True)
class _DraftRegistration:
    owner: VoiceDraftOwner
    ready: asyncio.Future[bool]


@dataclass(slots=True, eq=False)
class _DraftPreparation:
    owner: VoiceDraftOwner
    pcm: bytes
    slot: _NativeDecodeSlot
    decoded: asyncio.Future[AsrUpdate | None]
    reusable: bool = True
    authorized: bool = False
    request_started: bool = False
    frame_snapshot_at: datetime | None = None
    prepare_task: asyncio.Task[VoiceDraftOwner | None] | None = None
    candidate_task: asyncio.Task[VoiceDraftOwner | None] | None = None


_TaskResult = TypeVar("_TaskResult")


class VoiceDraftTurnController:
    """Own one local voice turn and fence every speculative completion."""

    def __init__(
        self,
        *,
        client: VoiceDraftClient,
        decoder: VoiceDraftDecoder,
        draft_tts: VoiceDraftTts | None = None,
        diagnostics: AudioDiagnosticsObserver | None = None,
        allow_promotion: bool = True,
        before_context: Callable[[], Awaitable[bool]] | None = None,
        turn_id_factory: Callable[[], UUID] = uuid4,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._decoder = decoder
        self._draft_tts = draft_tts
        self._diagnostics = diagnostics
        self._allow_promotion = allow_promotion
        self._before_context = before_context
        self._turn_id_factory = turn_id_factory
        self._now = now or (lambda: datetime.now(UTC))
        self._owner: VoiceDraftOwner | None = None
        self._committed_tts_owner: VoiceDraftOwner | None = None
        self._terminal = False
        self._promote_revision = 0
        self._resume_cancel_pending = False
        self._cancel_sent_revisions: set[int] = set()
        self._scheduled_cancels: dict[
            int, tuple[VoiceDraftOwner, asyncio.Task[None]]
        ] = {}
        self._tasks: set[asyncio.Task[object]] = set()
        self._decode_slot: _NativeDecodeSlot | None = None
        self._preparation: _DraftPreparation | None = None
        self._active_preparation: _DraftPreparation | None = None
        self._cached_transcript: _CachedDraftTranscript | None = None
        self._pending_transcript: _PendingDraftTranscript | None = None
        self._registration: _DraftRegistration | None = None
        self._request_started_revisions: set[int] = set()
        self._reuse_miss_reasons: set[tuple[int, str]] = set()
        self._closed = False

    def open_turn(
        self,
        *,
        fish_id: UUID,
        session_id: UUID,
        binding_generation: int,
        speech_generation: int,
        reply_context: CompanionReplyContext | None = None,
        speech_started_at: datetime | None = None,
    ) -> VoiceDraftOwner:
        if self._closed:
            raise RuntimeError("voice draft controller is closed")
        if speech_started_at is not None and speech_started_at.utcoffset() is None:
            raise ValueError("speech_started_at must be timezone-aware")
        owner = VoiceDraftOwner(
            turn_id=self._turn_id_factory(),
            revision=0,
            candidate_epoch=0,
            speech_generation=speech_generation,
            binding_generation=binding_generation,
            fish_id=fish_id,
            session_id=session_id,
            reply_context=reply_context,
            speech_started_at=speech_started_at,
        )
        self._owner = owner
        self._committed_tts_owner = None
        self._retire_preparation()
        self._clear_cached_transcript()
        self._retire_registration()
        self._terminal = False
        self._promote_revision = 0
        self._resume_cancel_pending = False
        self._cancel_sent_revisions.clear()
        self._scheduled_cancels.clear()
        self._request_started_revisions.clear()
        self._reuse_miss_reasons.clear()
        return owner

    def prepare(
        self,
        owner: VoiceDraftOwner,
        *,
        pcm: bytes,
    ) -> asyncio.Task[VoiceDraftOwner | None] | None:
        """Prepare recognition privately without authorizing a draft request."""
        current = self._require_lineage(owner)
        if type(pcm) is not bytes:
            raise TypeError("draft PCM must be immutable bytes")
        preparation = self._matching_preparation(current, pcm)
        if preparation is None:
            preparation = self._create_preparation(current, pcm)
            if preparation is None:
                return None
            self._record_event("voice_draft_preparation_started", preparation.owner)
        task = preparation.prepare_task
        if task is None:
            task = self._track(
                asyncio.create_task(
                    self._await_preparation(preparation),
                    name=f"voice-draft-prepare-{preparation.owner.turn_id}",
                )
            )
            preparation.prepare_task = task
        return task

    def has_preparation(self, owner: VoiceDraftOwner, *, pcm: bytes) -> bool:
        """Whether this snapshot can reuse registered private recognition work."""
        if type(pcm) is not bytes:
            return False
        try:
            current = self._require_lineage(owner)
        except RuntimeError:
            return False
        return self._matching_preparation(current, pcm) is not None

    def candidate(
        self,
        owner: VoiceDraftOwner,
        *,
        pcm: bytes,
    ) -> asyncio.Task[VoiceDraftOwner | None] | None:
        """Authorize one PUT using existing or newly started recognition."""
        current = self._require_lineage(owner)
        if type(pcm) is not bytes:
            raise TypeError("draft PCM must be immutable bytes")
        preparation = self._matching_preparation(current, pcm)
        if preparation is None:
            previous = self._preparation
            if (
                previous is not None
                and previous.owner == current
                and previous.request_started
                and pcm.startswith(previous.pcm)
            ):
                # Revoking transcript reuse must not register an already
                # authorized pause twice or restart its existing LLM request.
                return previous.candidate_task
            preparation = self._create_preparation(current, pcm)
            if preparation is None:
                return None
        if preparation.candidate_task is not None:
            return preparation.candidate_task
        preparation.authorized = True
        candidate_owner = preparation.owner
        if candidate_owner.speech_started_at is not None:
            snapshot_at = self._now()
            if (
                snapshot_at.utcoffset() is None
                or snapshot_at < candidate_owner.speech_started_at
            ):
                raise ValueError("voice frame snapshot clock must follow speech start")
            preparation.frame_snapshot_at = snapshot_at
        self._promote_revision = (
            candidate_owner.revision if self._allow_promotion else 0
        )
        # Register the ACK waiter before yielding. A completed early transcript
        # must not let endpoint confirmation race ahead of this same PUT.
        self._registration = _DraftRegistration(
            owner=candidate_owner, ready=asyncio.get_running_loop().create_future()
        )
        self._record_event(
            "voice_draft_started",
            candidate_owner,
            metadata={
                "candidateEpoch": candidate_owner.candidate_epoch,
                "speechGeneration": candidate_owner.speech_generation,
                "bindingGeneration": candidate_owner.binding_generation,
            },
        )
        task = self._track(
            asyncio.create_task(
                self._register_preparation(preparation),
                name=f"voice-draft-register-{candidate_owner.turn_id}",
            )
        )
        preparation.candidate_task = task
        return task

    def _matching_preparation(
        self, current: VoiceDraftOwner, pcm: bytes
    ) -> _DraftPreparation | None:
        preparation = self._preparation
        if (
            preparation is None
            or preparation.owner != current
            or not preparation.reusable
            or not preparation.pcm
            or not pcm.startswith(preparation.pcm)
            or preparation.decoded.cancelled()
            or (preparation.decoded.done() and preparation.decoded.result() is None)
        ):
            return None
        return preparation

    def _create_preparation(
        self,
        current: VoiceDraftOwner,
        pcm: bytes,
    ) -> _DraftPreparation | None:
        if self._active_preparation is not None:
            self._record_decode_busy(current)
            return None
        self._retire_preparation()
        slot = _NativeDecodeSlot(threading.Event(), threading.Event())
        candidate_owner = replace(current, revision=current.revision + 1)
        self._owner = candidate_owner
        self._clear_cached_transcript()
        self._retire_registration()
        self._promote_revision = 0
        preparation = _DraftPreparation(
            owner=candidate_owner,
            pcm=pcm,
            slot=slot,
            decoded=asyncio.get_running_loop().create_future(),
        )
        self._preparation = preparation
        if self._allow_promotion:
            self._pending_transcript = _PendingDraftTranscript(
                owner=candidate_owner,
                pcm=pcm,
                slot=slot,
                ready=asyncio.get_running_loop().create_future(),
            )
        self._start_preparation(preparation)
        return preparation

    def _start_preparation(self, preparation: _DraftPreparation) -> None:
        self._active_preparation = preparation
        self._decode_slot = preparation.slot
        self._track(
            asyncio.create_task(
                self._decode_preparation(preparation),
                name=f"voice-draft-decode-{preparation.owner.turn_id}",
            )
        )

    def _retire_preparation(self) -> None:
        preparation = self._preparation
        self._preparation = None
        if preparation is not None:
            preparation.reusable = False
            if not preparation.decoded.done():
                preparation.decoded.set_result(None)

    async def _await_preparation(
        self, preparation: _DraftPreparation
    ) -> VoiceDraftOwner | None:
        update = await asyncio.shield(preparation.decoded)
        if (
            update is None
            or not preparation.reusable
            or not self.is_current(preparation.owner)
        ):
            return None
        return preparation.owner

    def cached_transcript(
        self,
        owner: VoiceDraftOwner,
        pcm_snapshot: bytes,
    ) -> AsrUpdate | None:
        """Return a completed decode covering the current snapshot's prefix.

        The caller must separately establish that any remaining PCM contains no
        new speech. Prefix equality alone does not authorize finalization.
        """
        if not self._allow_promotion or type(pcm_snapshot) is not bytes:
            return None
        try:
            current = self._require_lineage(owner)
        except RuntimeError:
            return None
        cached = self._cached_transcript
        if (
            cached is None
            or cached.owner != current
            or not cached.pcm
            or not pcm_snapshot.startswith(cached.pcm)
        ):
            return None
        return cached.update

    async def wait_for_transcript(
        self,
        owner: VoiceDraftOwner,
        pcm_snapshot: bytes,
    ) -> AsrUpdate | None:
        """Share matching scheduled recognition without waiting for upload.

        This only deduplicates recognition. As with ``cached_transcript``, the
        caller still owns the no-new-speech and finalization checks.
        """
        cached = self.cached_transcript(owner, pcm_snapshot)
        if cached is not None:
            return cached
        if not self._allow_promotion or type(pcm_snapshot) is not bytes:
            self.record_reuse_miss(
                owner,
                "promotion_disabled" if not self._allow_promotion else "snapshot_type",
            )
            return None
        try:
            current = self._require_lineage(owner)
        except RuntimeError:
            self.record_reuse_miss(owner, "stale_owner")
            return None
        pending = self._pending_transcript
        if (
            pending is None
            or pending.owner != current
            or not pending.pcm
            or not pcm_snapshot.startswith(pending.pcm)
        ):
            cached = self._cached_transcript
            if (
                cached is not None
                and cached.owner == current
                and not pcm_snapshot.startswith(cached.pcm)
            ):
                reason = "pcm_prefix_mismatch"
            elif pending is None:
                reason = "no_matching_decode"
            elif pending.owner != current:
                reason = "stale_owner"
            else:
                reason = "pcm_prefix_mismatch"
            self.record_reuse_miss(owner, reason)
            return None
        if not await asyncio.shield(pending.ready):
            self.record_reuse_miss(owner, "decode_invalidated")
            return None
        if not self.is_current(pending.owner):
            self.record_reuse_miss(owner, "stale_owner")
            return None
        result = self.cached_transcript(owner, pcm_snapshot)
        if result is None:
            self.record_reuse_miss(owner, "pcm_prefix_mismatch")
        return result

    async def wait_for_registration(self, owner: VoiceDraftOwner) -> bool:
        """Join the existing PUT acknowledgement, never context/model execution."""

        try:
            current = self._require_lineage(owner)
        except RuntimeError:
            return False
        registration = self._registration
        if (
            registration is None
            or registration.owner != current
            or not self._allow_promotion
        ):
            return False
        accepted = await asyncio.shield(registration.ready)
        return (
            accepted
            and self.is_current(registration.owner)
            and self._promote_revision == registration.owner.revision
        )

    def current_owner(self, owner: VoiceDraftOwner) -> VoiceDraftOwner | None:
        """Freeze the exact candidate revision before an asynchronous handoff."""

        try:
            return self._require_lineage(owner)
        except RuntimeError:
            return None

    def invalidate_transcript(
        self, owner: VoiceDraftOwner, *, reason: str = "reuse_invalidated"
    ) -> None:
        """Revoke ASR reuse without cancelling or revising the LLM draft."""
        try:
            current = self._require_lineage(owner)
        except RuntimeError:
            return
        self.record_reuse_miss(current, reason)
        preparation = self._preparation
        if preparation is not None and preparation.owner == current:
            preparation.reusable = False
            if not preparation.request_started:
                if not preparation.decoded.done():
                    preparation.decoded.set_result(None)
                self._complete_registration(current, accepted=False)
                self._promote_revision = 0
        self._clear_cached_transcript(current)

    def record_reuse_miss(self, owner: VoiceDraftOwner, reason: str) -> None:
        """Record each reason at most once for a candidate revision."""
        current = self.current_owner(owner) or owner
        identity = (current.revision, reason)
        if identity in self._reuse_miss_reasons:
            return
        self._reuse_miss_reasons.add(identity)
        self._record_event(
            "voice_draft_asr_reuse_missed",
            current,
            metadata={
                "reason": reason,
                "speechGeneration": current.speech_generation,
                "candidateEpoch": current.candidate_epoch,
            },
        )

    def _record_decode_busy(self, owner: VoiceDraftOwner) -> None:
        """Report one same-revision fallback without scheduling more native work."""
        identity = (owner.revision, "decode_busy")
        if identity in self._reuse_miss_reasons:
            return
        self._reuse_miss_reasons.add(identity)
        self._record_event(
            "voice_draft_fallback",
            owner,
            metadata={
                "reason": "decode_busy",
                "speechGeneration": owner.speech_generation,
                "candidateEpoch": owner.candidate_epoch,
            },
        )

    def skip_candidate(
        self,
        owner: VoiceDraftOwner,
        *,
        reason: str,
        queue_age_ms: float,
        queue_depth: int,
        dropped_frames: int,
    ) -> None:
        current = self._require_lineage(owner)
        self._record_event(
            "voice_draft_fallback",
            current,
            metadata={
                "reason": reason,
                "queueAgeMs": max(0.0, queue_age_ms),
                "queueDepth": max(0, queue_depth),
                "droppedFrames": max(0, dropped_frames),
            },
        )

    def resume(
        self, owner: VoiceDraftOwner, *, source: str = "speech_resumed"
    ) -> VoiceDraftOwner:
        current = self._require_lineage(owner)
        attempt_revision = self._promote_revision or current.revision
        resumed = replace(
            current,
            revision=current.revision + 1,
            candidate_epoch=current.candidate_epoch + 1,
        )
        self._owner = resumed
        self._retire_preparation()
        self._clear_cached_transcript()
        self._retire_registration()
        self._promote_revision = 0
        self._resume_cancel_pending = True
        self._record_event(
            "voice_draft_resumed",
            resumed,
            metadata={
                "reason": source,
                "speechGeneration": resumed.speech_generation,
                "candidateEpoch": resumed.candidate_epoch,
            },
            attempt_revision=attempt_revision,
        )
        self._schedule_cancel(
            resumed,
            reason="speech_resumed",
            attempt_revision=attempt_revision,
        )
        return resumed

    def reopen(self, *, speech_generation: int) -> VoiceDraftOwner:
        current = self._require_active()
        if speech_generation <= current.speech_generation:
            raise ValueError("speech_generation must increase on reopen")
        reopened = replace(current, speech_generation=speech_generation)
        self._owner = reopened
        self._retire_preparation()
        self._clear_cached_transcript()
        self._retire_registration()
        if self._resume_cancel_pending:
            self._schedule_cancel(
                reopened,
                reason="speech_resumed",
                attempt_revision=max(0, reopened.revision - 1),
            )
        return reopened

    def commit(
        self,
        owner: VoiceDraftOwner,
        *,
        content: str,
    ) -> VoiceDraftSubmission:
        current = self._require_lineage(owner)
        attempt_revision = self._promote_revision or current.revision
        draft_request_started = self._has_started_request(attempt_revision)
        committed = replace(current, revision=current.revision + 1)
        promote_revision = self._promote_revision
        self._mark_tts_committed(current, promote_revision)
        # Retire speculative ASR/draft ownership before the session can yield
        # while starting the one authoritative HTTP commit.
        self._owner = None
        self._retire_preparation()
        self._clear_cached_transcript()
        self._retire_registration()
        self._terminal = True
        self._promote_revision = 0
        self._resume_cancel_pending = False
        self._request_started_revisions.clear()
        if not self._allow_promotion and current.revision > 0:
            self._track(
                asyncio.create_task(
                    self._retire_shadow_tts(
                        committed,
                        attempt_revision=attempt_revision,
                    ),
                    name=f"voice-draft-shadow-cleanup-{committed.turn_id}",
                )
            )
        return VoiceDraftSubmission(
            key=committed.key,
            revision=committed.revision,
            content=content,
            promote_revision=promote_revision,
            draft_request_started=draft_request_started,
        )

    def abandon(
        self,
        owner: VoiceDraftOwner,
        *,
        reason: str,
    ) -> VoiceDraftOwner:
        current = self._require_lineage(owner)
        attempt_revision = self._promote_revision or current.revision
        abandoned = replace(
            current,
            revision=current.revision + 1,
            candidate_epoch=current.candidate_epoch + 1,
        )
        self._owner = abandoned
        self._retire_preparation()
        self._clear_cached_transcript()
        self._retire_registration()
        self._terminal = True
        self._promote_revision = 0
        self._resume_cancel_pending = False
        self._schedule_cancel(
            abandoned,
            reason=reason,
            attempt_revision=attempt_revision,
        )
        return abandoned

    def cancel_submission(
        self,
        submission: VoiceDraftSubmission,
        *,
        reason: str,
    ) -> asyncio.Task[None]:
        """Cancel a finalized handoff that the session could not accept."""
        committed_owner = self._committed_tts_owner
        if committed_owner is not None and committed_owner.key == submission.key:
            self._committed_tts_owner = None
        return self._track(
            asyncio.create_task(
                self._cancel_finalized_submission(submission, reason),
                name=f"voice-draft-cancel-finalized-{submission.key.turn_id}",
            )
        )

    def is_current(self, owner: VoiceDraftOwner) -> bool:
        return self._owner == owner

    async def wait_idle(self) -> None:
        while self._tasks:
            tasks = tuple(self._tasks)
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.difference_update(tasks)

    async def close(self) -> None:
        cancel_owner: VoiceDraftOwner | None = None
        if not self._closed:
            self._closed = True
            self._committed_tts_owner = None
            current = self._owner
            if current is not None:
                cancel_owner = (
                    current
                    if self._terminal
                    else replace(
                        current,
                        revision=current.revision + 1,
                        candidate_epoch=current.candidate_epoch + 1,
                    )
                )
            self._owner = cancel_owner
            self._retire_preparation()
            self._clear_cached_transcript()
            self._retire_registration()
            self._terminal = True
            self._promote_revision = 0
            self._resume_cancel_pending = False
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        await self.wait_idle()
        if cancel_owner is not None:
            await self._cancel(
                cancel_owner,
                "shutdown",
                attempt_revision=max(0, cancel_owner.revision - 1),
            )
        self._owner = None
        self._cancel_sent_revisions.clear()
        self._scheduled_cancels.clear()

    async def _decode_preparation(self, preparation: _DraftPreparation) -> None:
        owner = preparation.owner
        try:
            update = await asyncio.to_thread(
                self._recognize, preparation.pcm, preparation.slot
            )
            if not update.is_final:
                raise ValueError("draft decoder must return a completed transcript")
            content = update.text.strip()
            if not content:
                self._clear_cached_transcript(owner)
                if self.is_current(owner):
                    self._record_event(
                        "voice_draft_fallback",
                        owner,
                        metadata={
                            "reason": "empty_transcript",
                        },
                    )
                return
            if not self.is_current(owner) or self._preparation is not preparation:
                self._record_late_drop(owner, "stale_owner")
                return
            if not preparation.reusable:
                return
            update = replace(update, text=content)
            pending = self._pending_transcript
            if (
                self._allow_promotion
                and preparation.reusable
                and pending is not None
                and pending.owner == owner
            ):
                self._cached_transcript = _CachedDraftTranscript(
                    owner=owner,
                    pcm=preparation.pcm,
                    update=update,
                )
                self._complete_transcript_decode(owner, ready=True)
            self._record_event(
                "voice_draft_transcript_ready",
                owner,
                metadata={
                    "charCount": len(content),
                    "utf8Bytes": len(content.encode("utf-8")),
                },
            )
            if not preparation.decoded.done():
                preparation.decoded.set_result(update)
        except asyncio.CancelledError:
            self._discard_failed_candidate(owner)
            raise
        except Exception:
            self._discard_failed_candidate(owner)
            logger.exception("Companion voice draft recognition failed")
        finally:
            if not preparation.decoded.done():
                preparation.decoded.set_result(None)
            self._complete_transcript_decode(owner, ready=False)
            if self._active_preparation is preparation:
                self._active_preparation = None
                self._decode_slot = None

    async def _register_preparation(
        self, preparation: _DraftPreparation
    ) -> VoiceDraftOwner | None:
        owner = preparation.owner
        try:
            update = await asyncio.shield(preparation.decoded)
            if update is None:
                self._discard_failed_candidate(owner)
                return None
            if not self.is_current(owner) or self._preparation is not preparation:
                self._record_late_drop(owner, "stale_owner")
                return None
            if not preparation.reusable:
                self._discard_failed_candidate(owner)
                return None
            content = update.text
            rejection_reason = low_evidence_utterance_reason(
                content,
                update.event,
                allow_nonlexical=(
                    owner.reply_context is not None
                    and owner.reply_context.kind == "interaction"
                ),
            )
            if rejection_reason is not None:
                # Recognition remains authoritative for finalization, but a
                # low-evidence pause must not start remote LLM or TTS work.
                # Keep the cached AsrUpdate intact so the final path can reuse
                # this exact decode without paying for recognition again.
                self._complete_registration(owner, accepted=False)
                if self.is_current(owner):
                    self._promote_revision = 0
                    self._record_event(
                        "voice_draft_fallback",
                        owner,
                        metadata={"reason": rejection_reason},
                    )
                return None
            if self._before_context is not None:
                if not await self._before_context():
                    self._complete_registration(owner, accepted=False)
                    return None
                if not self.is_current(owner) or self._preparation is not preparation:
                    return None
            self._record_event(
                "voice_draft_request_started",
                owner,
            )
            preparation.request_started = True
            self._request_started_revisions.add(owner.revision)
            if owner.speech_started_at is not None:
                accepted = await self._client.upsert_voice_draft(
                    owner.key,
                    revision=owner.revision,
                    content=content,
                    reply_context=owner.reply_context,
                    speech_started_at=owner.speech_started_at,
                    frame_snapshot_at=preparation.frame_snapshot_at,
                )
            elif owner.reply_context is None:
                accepted = await self._client.upsert_voice_draft(
                    owner.key, revision=owner.revision, content=content
                )
            else:
                accepted = await self._client.upsert_voice_draft(
                    owner.key,
                    revision=owner.revision,
                    content=content,
                    reply_context=owner.reply_context,
                )
            self._complete_registration(owner, accepted=accepted)
            if not accepted:
                self._discard_failed_candidate(owner)
                if self.is_current(owner):
                    self._record_event(
                        "voice_draft_fallback",
                        owner,
                        metadata={
                            "reason": "request_rejected",
                        },
                    )
                return None
            if not self.is_current(owner):
                self._record_late_drop(owner, "stale_owner")
                return None
            draft_tts = self._draft_tts
            if draft_tts is not None:
                try:
                    await draft_tts.start(owner.key, revision=owner.revision)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("Companion draft TTS start missed", exc_info=True)
            # A committed turn owns its existing TTS startup until readiness;
            # completing that wait is not a stale speculative callback.
            if not self.is_current(owner) and self._committed_tts_owner != owner:
                self._record_late_drop(owner, "stale_owner")
                return None
            return owner
        except asyncio.CancelledError:
            self._discard_failed_candidate(owner)
            raise
        except Exception:
            self._discard_failed_candidate(owner)
            logger.exception("Companion voice draft candidate failed")
            return None

    def _mark_tts_committed(
        self, owner: VoiceDraftOwner, promote_revision: int
    ) -> None:
        if promote_revision <= 0 or promote_revision != owner.revision:
            return
        mark_committed = getattr(self._draft_tts, "mark_committed", None)
        if not callable(mark_committed):
            return
        try:
            if mark_committed(owner.key, revision=promote_revision):
                self._committed_tts_owner = owner
        except Exception:
            # TTS remains optional; this cannot prevent the authoritative turn.
            logger.debug("Companion draft TTS handoff missed", exc_info=True)

    def _discard_failed_candidate(self, owner: VoiceDraftOwner) -> None:
        preparation = self._preparation
        if preparation is not None and preparation.owner == owner:
            preparation.reusable = False
            if not preparation.decoded.done():
                preparation.decoded.set_result(None)
        self._clear_cached_transcript(owner)
        self._complete_registration(owner, accepted=False)
        if self.is_current(owner):
            self._promote_revision = 0

    def _complete_registration(self, owner: VoiceDraftOwner, *, accepted: bool) -> None:
        registration = self._registration
        if (
            registration is not None
            and registration.owner == owner
            and not registration.ready.done()
        ):
            registration.ready.set_result(accepted)

    def _retire_registration(self) -> None:
        registration = self._registration
        self._registration = None
        if registration is not None and not registration.ready.done():
            registration.ready.set_result(False)

    def _recognize(self, pcm: bytes, slot: _NativeDecodeSlot) -> AsrUpdate:
        slot.started.set()
        try:
            recognize_update = getattr(self._decoder, "recognize_update", None)
            if callable(recognize_update):
                update = recognize_update(pcm)
                if not isinstance(update, AsrUpdate):
                    raise TypeError("draft decoder must return an AsrUpdate")
                return update
            return AsrUpdate(text=self._decoder.recognize(pcm), is_final=True)
        finally:
            slot.finished.set()

    def _clear_cached_transcript(self, owner: VoiceDraftOwner | None = None) -> None:
        cached = self._cached_transcript
        if owner is None or (cached is not None and cached.owner == owner):
            self._cached_transcript = None
        self._complete_transcript_decode(owner, ready=False)

    def _complete_transcript_decode(
        self,
        owner: VoiceDraftOwner | None,
        *,
        ready: bool,
    ) -> None:
        pending = self._pending_transcript
        if pending is None or (owner is not None and pending.owner != owner):
            return
        self._pending_transcript = None
        if not pending.ready.done():
            pending.ready.set_result(ready)

    def _schedule_cancel(
        self,
        owner: VoiceDraftOwner,
        *,
        reason: str,
        attempt_revision: int,
    ) -> None:
        if not self._has_started_request(attempt_revision):
            self._resume_cancel_pending = False
            if self._terminal and self.is_current(owner):
                self._owner = None
            return
        if owner.revision in self._cancel_sent_revisions:
            return
        scheduled = self._scheduled_cancels.get(owner.revision)
        if scheduled is not None:
            scheduled_owner, scheduled_task = scheduled
            if scheduled_owner == owner and not scheduled_task.done():
                return
            if not scheduled_task.done():
                # Reopen can retain the same revision while changing speech
                # generation. Its new owner must replace the stale task whose
                # current-owner fence would otherwise suppress cancellation.
                scheduled_task.cancel()
        task = asyncio.create_task(
            self._cancel(
                owner,
                reason,
                attempt_revision=attempt_revision,
            ),
            name=f"voice-draft-cancel-{owner.turn_id}",
        )
        self._scheduled_cancels[owner.revision] = (owner, task)
        task.add_done_callback(
            lambda completed, revision=owner.revision: self._release_scheduled_cancel(
                revision, completed
            )
        )
        self._track(task)

    async def _cancel(
        self,
        owner: VoiceDraftOwner,
        reason: str,
        *,
        attempt_revision: int,
    ) -> None:
        try:
            if not self._has_started_request(attempt_revision):
                self._resume_cancel_pending = False
                if self._terminal and self.is_current(owner):
                    self._owner = None
                return
            if not self.is_current(owner):
                return
            await self._cancel_draft_tts(owner.key, owner.revision)
            if not self.is_current(owner):
                return
            if owner.revision in self._cancel_sent_revisions:
                return
            if not self.is_current(owner):
                return
            sent = await self._client.cancel_voice_draft(
                owner.key,
                revision=owner.revision,
                reason=reason,
            )
            if not sent:
                return
            if not self.is_current(owner):
                return
            self._cancel_sent_revisions.add(owner.revision)
            self._resume_cancel_pending = False
            self._record_event(
                "voice_draft_canceled",
                owner,
                metadata={"reason": reason},
                attempt_revision=attempt_revision,
            )
            self._record_event(
                "voice_draft_cleanup",
                owner,
                metadata={"reason": reason},
                attempt_revision=attempt_revision,
            )
            if self._terminal:
                self._owner = None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Companion voice draft cancellation failed")

    async def _cancel_finalized_submission(
        self,
        submission: VoiceDraftSubmission,
        reason: str,
    ) -> None:
        try:
            if not submission.draft_request_started:
                return
            attempt_revision = submission.promote_revision or max(
                0, submission.revision - 1
            )
            await self._cancel_draft_tts(
                submission.key,
                submission.revision + 1,
            )
            await self._client.cancel_voice_draft(
                submission.key,
                revision=submission.revision + 1,
                reason=reason,
            )
            self._record_event(
                "voice_draft_canceled",
                submission.key,
                submission.revision + 1,
                metadata={
                    "reason": reason,
                },
                attempt_revision=attempt_revision,
            )
            self._record_event(
                "voice_draft_cleanup",
                submission.key,
                submission.revision + 1,
                metadata={
                    "reason": reason,
                },
                attempt_revision=attempt_revision,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Companion finalized voice draft cancellation failed")

    def _has_started_request(self, through_revision: int) -> bool:
        return any(
            revision <= through_revision for revision in self._request_started_revisions
        )

    def _release_scheduled_cancel(
        self,
        revision: int,
        task: asyncio.Task[None],
    ) -> None:
        scheduled = self._scheduled_cancels.get(revision)
        if scheduled is not None and scheduled[1] is task:
            self._scheduled_cancels.pop(revision, None)

    async def _cancel_draft_tts(
        self,
        key: VoiceDraftKey,
        through_revision: int,
    ) -> None:
        draft_tts = self._draft_tts
        if draft_tts is None:
            return
        try:
            await draft_tts.cancel(
                key,
                through_revision=through_revision,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Companion draft TTS cancellation missed", exc_info=True)

    async def _retire_shadow_tts(
        self,
        owner: VoiceDraftOwner,
        *,
        attempt_revision: int,
    ) -> None:
        await self._cancel_draft_tts(owner.key, owner.revision)
        self._record_event(
            "voice_draft_cleanup",
            owner,
            metadata={"reason": "promotion_miss"},
            attempt_revision=attempt_revision,
        )

    def _require_active(self) -> VoiceDraftOwner:
        if self._closed:
            raise RuntimeError("voice draft controller is closed")
        if self._owner is None or self._terminal:
            raise RuntimeError("no active voice draft turn")
        return self._owner

    def _require_lineage(self, owner: VoiceDraftOwner) -> VoiceDraftOwner:
        current = self._require_active()
        if (
            owner.turn_id != current.turn_id
            or owner.fish_id != current.fish_id
            or owner.session_id != current.session_id
            or owner.binding_generation != current.binding_generation
            or owner.speech_generation != current.speech_generation
            or owner.candidate_epoch != current.candidate_epoch
            or owner.reply_context != current.reply_context
            or owner.speech_started_at != current.speech_started_at
        ):
            raise RuntimeError("stale voice draft owner")
        return current

    def _record_late_drop(self, owner: VoiceDraftOwner, reason: str) -> None:
        self._record_event(
            "voice_draft_late_event_dropped",
            owner,
            metadata={"reason": reason},
        )

    def _record_event(
        self,
        name: str,
        owner: VoiceDraftOwner | VoiceDraftKey,
        revision: int | None = None,
        *,
        metadata: dict[str, object] | None = None,
        attempt_revision: int | None = None,
    ) -> None:
        key = owner.key if isinstance(owner, VoiceDraftOwner) else owner
        effective_revision = (
            owner.revision if isinstance(owner, VoiceDraftOwner) else revision
        )
        if effective_revision is None:
            raise ValueError("voice draft diagnostic revision is required")
        safe_metadata = {
            "turnId": str(key.turn_id),
            "revision": effective_revision,
            **(metadata or {}),
        }
        if attempt_revision is not None:
            safe_metadata["attemptRevision"] = attempt_revision
        record_voice_draft_event(
            self._diagnostics,
            name,
            metadata=safe_metadata,
        )

    def _track(
        self,
        task: asyncio.Task[_TaskResult],
    ) -> asyncio.Task[_TaskResult]:
        tracked = cast(asyncio.Task[object], task)
        self._tasks.add(tracked)
        task.add_done_callback(self._observe_task_completion)
        return task

    def _observe_task_completion(self, task: asyncio.Task[object]) -> None:
        try:
            if task.cancelled():
                return
            error = task.exception()
            if error is None:
                return
            logger.error(
                "Companion voice draft background task failed: %s",
                task.get_name(),
                exc_info=(type(error), error, error.__traceback__),
            )
        finally:
            self._tasks.discard(task)
