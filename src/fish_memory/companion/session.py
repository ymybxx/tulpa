"""Platform-independent state machine for the native voice companion."""

from __future__ import annotations

import asyncio
import logging
import time
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Protocol, cast, overload
from uuid import UUID, uuid4

from fish_memory import failures
from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.answer_progress import AnswerPlaybackProgress
from fish_memory.companion.client import (
    CompanionAskResult,
    CompanionMessage,
    CompanionReplyContext,
)
from fish_memory.companion.diagnostics import (
    AudioDiagnosticsObserver,
    record_voice_draft_event,
)
from fish_memory.companion.performance import PetVoiceStatus
from fish_memory.companion.playback import (
    BoundTtsOutput,
    ForegroundReservation,
    PlaybackArbiter,
    PlaybackOwner,
)
from fish_memory.companion.playback_caption import PlaybackCaption
from fish_memory.companion.text_buffer import (
    TTS_TEXT_BOUNDARIES as TTS_TEXT_BOUNDARIES,
)
from fish_memory.companion.text_buffer import (
    TTS_TEXT_MIN_CHARS as TTS_TEXT_MIN_CHARS,
)
from fish_memory.companion.text_buffer import (
    TtsTextBuffer as TtsTextBuffer,
)
from fish_memory.companion.tts import DraftTtsPending
from fish_memory.pet.publisher import NoopPetEventSink, PetEventSink
from fish_memory.pet.types import PetEmotion
from fish_memory.tts.speech_tags import SpeechTextBuffer, strip_speech_tags

logger = logging.getLogger(__name__)

WAKE_ACK_TEXT = "我在呢。"
_AGENT_SEEN_MESSAGE_LIMIT = 2048
_AGENT_RECONNECT_DELAYS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
_ANSWER_PROGRESS_TIMEOUT_SECONDS = 0.5
# How long a pre-synthesized draft may wait for matching answer text.
_DRAFT_TTS_STALL_SECONDS = 8.0
_SubmitTurnRejectionReason = Literal[
    "invalid_voice_revision",
    "missing_turn_binding",
    "empty_content",
    "reply_context_blocked",
    "binding_changed",
    "voice_conversation_changed",
    "agent_progress_unsettled",
    "agent_reply_finalize_failed",
    "binding_changed_after_wake_ack",
    "session_changed",
    "state_not_listening",
]


@dataclass(frozen=True, slots=True)
class _AudioRecovery:
    fish_id: UUID
    binding_generation: int
    resume_listening: bool


@dataclass(frozen=True, slots=True)
class VoiceReplyContext:
    fish_id: UUID
    session_id: UUID
    voice_conversation_id: UUID
    subject_kind: Literal["interaction", "run_result"]
    subject_id: UUID
    source_message_id: UUID
    expires_at: float

    @property
    def client_context(self) -> CompanionReplyContext:
        return CompanionReplyContext(self.subject_kind, self.subject_id)


@dataclass(frozen=True, slots=True)
class FrozenVoiceTurn:
    fish_id: UUID
    session_id: UUID
    voice_conversation_id: UUID
    binding_generation: int
    reply_context: VoiceReplyContext | None
    context_reservation_id: UUID | None
    reply_blocked: bool
    agent_reply_finalize: FrozenAgentReply | None


@dataclass(frozen=True, slots=True)
class FrozenAgentReply:
    message: CompanionMessage
    consumed_offset: int


@dataclass(frozen=True, slots=True)
class _VoiceTtsCandidate:
    key: VoiceDraftKey
    revision: int
    voice_conversation_id: UUID
    reply_context: VoiceReplyContext | None


@dataclass(frozen=True, slots=True)
class _AgentProgressBarrier:
    message: CompanionMessage
    consumed_offset: int
    resume_playback: bool
    activate_context: bool
    persist_required: bool
    retry_delay_seconds: float = 0.0


class CompanionState(StrEnum):
    UNBOUND = "unbound"
    STANDBY = "standby"
    ACKNOWLEDGING = "acknowledging"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class CompanionApi(Protocol):
    async def resolve_session(self, fish_id: UUID) -> UUID: ...

    async def ask(
        self,
        fish_id: UUID,
        content: str,
        *,
        session_id: UUID,
        reply_context: CompanionReplyContext | None = None,
        input_mode: Literal["typed", "spoken"] | None = None,
        attachment_ids: tuple[UUID, ...] = (),
    ) -> CompanionAskResult: ...

    async def commit_voice_turn(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        promote_revision: int,
        content: str,
        reply_context: CompanionReplyContext | None = None,
    ) -> CompanionAskResult: ...

    def iter_answer(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
    ) -> AsyncIterator[dict[str, object]]: ...

    def iter_home_actions(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
    ) -> AsyncIterator[dict[str, object]]: ...

    async def update_progress(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None: ...

    async def interrupt_answer(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None: ...

    def iter_agent_messages(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        after_sequence: int,
    ) -> AsyncIterator[CompanionMessage]: ...

    async def consume_agent_message(
        self,
        fish_id: UUID,
        message_id: UUID,
        *,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> None: ...

    def iter_companion_messages(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        after_sequence: int,
    ) -> AsyncIterator[CompanionMessage]: ...

    async def consume_companion_message(
        self,
        fish_id: UUID,
        message_id: UUID,
        *,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
        source: Literal["agent", "proactive_observation", "plugin"],
    ) -> None: ...


class TtsOutput(Protocol):
    @property
    def played_offset(self) -> int: ...

    async def start(self, *, text_offset: int = 0) -> None: ...

    async def append(self, text: str) -> None: ...

    async def flush(self) -> None: ...

    async def finish(self) -> None: ...

    async def stop(self) -> None: ...

    async def interrupt(self) -> None: ...

    async def pause(self) -> None: ...

    async def resume(self) -> None: ...

    async def duck(self) -> None: ...

    async def unduck(self) -> None: ...


class DraftTtsPromotionInput(Protocol):
    @property
    def output(self) -> TtsOutput: ...

    @property
    def submitted_offset(self) -> int: ...


class DraftTtsSessionInput(Protocol):
    async def promote(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        flush_offset: int,
        playback_owner: PlaybackOwner | None = None,
        before_playback: Callable[[], None] | None = None,
    ) -> DraftTtsPromotionInput | DraftTtsPending | None: ...

    async def cancel(
        self,
        key: VoiceDraftKey,
        *,
        through_revision: int,
    ) -> None: ...

    async def close(self) -> None: ...


class VoicePerformanceReporter(Protocol):
    def submit(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        status: PetVoiceStatus,
    ) -> None: ...


class CompanionSession:
    """Coordinates one bound Fish without creating or selecting Fish records."""

    def __init__(
        self,
        api: CompanionApi,
        tts: TtsOutput,
        *,
        diagnostics: AudioDiagnosticsObserver | None = None,
        performance_reporter: VoicePerformanceReporter | None = None,
        pet_events: PetEventSink | None = None,
        draft_tts: DraftTtsSessionInput | None = None,
        pet_audio_sample_rate: int = 16_000,
        home_actions_enabled: bool = False,
        home_action_timeout_seconds: float = 8.0,
        playback_arbiter: PlaybackArbiter | None = None,
        agent_reply_context_ttl_seconds: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
        agent_reconnect_delays_seconds: tuple[float, ...] = (
            _AGENT_RECONNECT_DELAYS_SECONDS
        ),
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if agent_reply_context_ttl_seconds <= 0:
            raise ValueError("Agent reply context TTL must be positive")
        if not agent_reconnect_delays_seconds or any(
            delay < 0 for delay in agent_reconnect_delays_seconds
        ):
            raise ValueError("Agent reconnect delays must be nonnegative")
        self._api = api
        self._tts = tts
        self._playback = playback_arbiter or PlaybackArbiter()
        self._owns_playback = playback_arbiter is None
        self._agent_reply_context_ttl_seconds = agent_reply_context_ttl_seconds
        self._monotonic = monotonic
        self._now = now
        self._agent_reconnect_delays_seconds = agent_reconnect_delays_seconds
        self._draft_tts = draft_tts
        self._turn_tts: TtsOutput | None = None
        self._voice_tts_owner: _VoiceTtsCandidate | None = None
        self._diagnostics = diagnostics
        self._performance_reporter = performance_reporter
        self._pet_events = pet_events or NoopPetEventSink()
        self._pet_audio_sample_rate = pet_audio_sample_rate
        self._pet_generation: UUID | None = None
        self._home_actions_enabled = home_actions_enabled
        self._home_action_timeout_seconds = home_action_timeout_seconds
        self._binding_generation = 0
        self._fish_id: UUID | None = None
        self._session_id: UUID | None = None
        self._state = CompanionState.UNBOUND
        self._audio_recovery: _AudioRecovery | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._wake_ack_task: asyncio.Task[None] | None = None
        self._interrupt_cleanup_task: asyncio.Task[None] | None = None
        self._tts_started = False
        self._draft_tts_waiting = False
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        self._answer_message_id: UUID | None = None
        self._answer_progress: AnswerPlaybackProgress | None = None
        self._agent_caption: PlaybackCaption | None = None
        self._pending_answer_progress: list[AnswerPlaybackProgress] = []
        self._answer_event_recorded = False
        self._answer_first_content_recorded = False
        self._answer_offset = 0
        self._answer_flush_offset = 0
        self._assistant_text = ""
        self._assistant_speech = ""
        self._speech_tags = False
        self._wake_ack_text = ""
        self._tts_text_buffer: TtsTextBuffer | SpeechTextBuffer = TtsTextBuffer(
            first_chunk_at_pause=True
        )
        self._home_tasks: set[asyncio.Task[None]] = set()
        self._status_speech_queue: asyncio.Queue[tuple[int, UUID, UUID, str]] = (
            asyncio.Queue()
        )
        self._status_speech_task: asyncio.Task[None] | None = None
        self._turn_idle = asyncio.Event()
        self._turn_idle.set()
        self._foreground_reservation: ForegroundReservation | None = None
        self._playback_owner: PlaybackOwner | None = None
        # Keep the proven Agent ordering/reconnect/progress lane as the one
        # background-message lane. The legacy ``_agent_*`` names remain so the
        # Agent resume contract is changed as little as possible.
        self._agent_listener_task: asyncio.Task[None] | None = None
        self._agent_listener_generation = 0
        self._agent_listener_session_id: UUID | None = None
        self._agent_after_cursor = 0
        self._agent_seen_ids: set[UUID] = set()
        self._agent_seen_order: list[UUID] = []
        self._agent_queue: list[CompanionMessage] = []
        self._agent_drain_task: asyncio.Task[None] | None = None
        self._agent_playback_task: asyncio.Task[None] | None = None
        self._agent_pending_message: CompanionMessage | None = None
        self._agent_active_message: CompanionMessage | None = None
        self._agent_tts: BoundTtsOutput | None = None
        self._agent_tts_ranges: tuple[tuple[int, int], ...] = ()
        self._reply_context: VoiceReplyContext | None = None
        self._reply_context_expiry_task: asyncio.Task[None] | None = None
        self._reply_context_reservation_id: UUID | None = None
        self._reply_context_finalize_failed = False
        self._voice_conversation_id: UUID | None = None
        self._pending_agent_reply: FrozenAgentReply | None = None
        self._agent_progress_barrier: _AgentProgressBarrier | None = None
        self._agent_progress_retry_task: asyncio.Task[None] | None = None
        self._agent_progress_lock = asyncio.Lock()
        self._agent_playback_retry_message_id: UUID | None = None
        self._agent_playback_retry_offset = 0
        self._agent_playback_retry_attempt = 0
        self._proactive_expiry_tasks: dict[UUID, asyncio.Task[None]] = {}
        self._proactive_settlement_tasks: set[asyncio.Task[None]] = set()
        self._proactive_terminal_ids: set[UUID] = set()
        self._proactive_unsettled_ids: set[UUID] = set()
        self._agent_focus_idle = asyncio.Event()
        self._agent_focus_idle.set()
        self._binding_quiesced = False
        self._binding_quiesce_task: asyncio.Task[None] | None = None
        self._plugin_playback_sessions: frozenset[tuple[UUID, UUID]] = frozenset()
        self._plugin_binding_task: asyncio.Task[None] | None = None
        bind_interrupt = getattr(self._pet_events, "set_answer_interrupt_handler", None)
        if callable(bind_interrupt):
            bind_interrupt(self._handle_answer_interrupt_request)
        bind_plugins = getattr(self._pet_events, "set_plugin_playback_handler", None)
        if callable(bind_plugins):
            bind_plugins(self._handle_plugin_playback_sessions)
        bind_text = getattr(self._pet_events, "set_text_input_handler", None)
        if callable(bind_text):
            bind_text(self.submit_text)

    async def submit_text(
        self, fish_id: UUID, content: str, attachment_ids: tuple[UUID, ...] = ()
    ) -> dict[str, object]:
        """Typed input shares the native turn and playback owner, never ASR.

        Acknowledgement means /ask persisted the message, not merely that a
        background coroutine started. Microphone configuration is untouched.
        """
        if fish_id != self._fish_id or not content.strip():
            raise ValueError("宠物已切换或输入为空，请重试")
        reservation = self.reserve_foreground_speech()
        try:
            if self._state is CompanionState.STANDBY:
                await self.on_wake_word(play_ack=False)
            frozen = self.freeze_turn_binding()
            if frozen is None:
                raise ValueError("对话尚未就绪，请稍后重试；文字尚未发送")
            try:
                owner = self._playback.current_owner
                playback_task = self._agent_playback_task
                await self.interrupt_from_speech()
                if owner is not None and owner.kind == "proactive" and playback_task:
                    # Proactive interruption retires its owner asynchronously.
                    # Wait for that handoff, not the Agent reply-context TTL:
                    # this input is precisely what consumes the frozen context.
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(
                                asyncio.gather(playback_task, return_exceptions=True)
                            ),
                            timeout=3.0,
                        )
                    except TimeoutError as error:
                        raise ValueError(
                            "上一段播报尚未停止，请稍后重试；文字尚未发送"
                        ) from error
                frozen = self.refresh_pending_agent_reply(frozen)
                submitted = asyncio.get_running_loop().create_future()
                accepted = await self._submit_turn(
                    content,
                    voice_turn=None,
                    frozen_turn=frozen,
                    submitted=submitted,
                    attachment_ids=attachment_ids,
                    spoken=False,
                )
                if not accepted:
                    raise ValueError("上一段对话尚未结束，请稍后重试；文字尚未发送")
                result = await submitted
                return {
                    "messageId": str(result.message_id),
                    "sessionId": str(frozen.session_id),
                    "sequence": result.sequence,
                    "answerExpected": result.answer_expected,
                }
            finally:
                if frozen is not None:
                    self.release_frozen_turn(frozen)
        finally:
            self.release_foreground_speech(reservation)

    async def _handle_plugin_playback_sessions(
        self, sessions: frozenset[tuple[UUID, UUID]]
    ) -> None:
        self._plugin_playback_sessions = sessions
        self._ensure_plugin_session_binding()

    def _plugin_session_for_fish(self, fish_id: UUID) -> UUID | None:
        sessions = {
            session
            for fish, session in self._plugin_playback_sessions
            if fish == fish_id
        }
        # Keep an already served plugin stable. Conflicting plugin sessions do
        # not get to move the single native voice conversation arbitrarily.
        if self._session_id in sessions:
            return self._session_id
        return next(iter(sessions)) if len(sessions) == 1 else None

    def _ensure_plugin_session_binding(self) -> bool:
        task = self._plugin_binding_task
        if task is not None and not task.done():
            return True
        fish = self._fish_id
        if fish is None or self._binding_quiesced or self._audio_recovery is not None:
            return False
        target = self._plugin_session_for_fish(fish)
        if target is None or target == self._session_id:
            return False
        if not self._plugin_session_binding_idle():
            # Playback/state completion calls this again; no polling or audio
            # restart, and an in-flight user turn finishes in its own Session.
            return False
        self._plugin_binding_task = asyncio.create_task(
            self._bind_plugin_session(fish, target), name="companion-plugin-session"
        )
        return True

    def _plugin_session_binding_idle(self) -> bool:
        return not (
            self._binding_quiesced
            or self._audio_recovery is not None
            or self._playback.foreground_reserved
            or self._playback.current_owner is not None
            or self._turn_task is not None
            or (
                self._agent_playback_task is not None
                and not self._agent_playback_task.done()
            )
            or self._state not in {CompanionState.STANDBY, CompanionState.LISTENING}
        )

    async def _bind_plugin_session(self, fish_id: UUID, target: UUID) -> None:
        try:
            if (
                fish_id != self._fish_id
                or target == self._session_id
                or target != self._plugin_session_for_fish(fish_id)
                or not self._plugin_session_binding_idle()
            ):
                return
            listening = self._state is CompanionState.LISTENING
            previous = self._session_id
            self.quiesce_binding(fish_id)
            generation = self._binding_generation
            await self._stop_agent_side_effects(persist_partial=True)
            await self._stop_binding_side_effects()
            if generation != self._binding_generation or fish_id != self._fish_id:
                return
            self._session_id = (
                target
                if (fish_id, target) in self._plugin_playback_sessions
                else previous
            )
            self._voice_conversation_id = uuid4() if listening else None
            self._binding_quiesced = False
            self._publish_companion_activity()
            self._start_agent_listener()
            logger.info(
                "Plugin playback Session rebound: %s -> %s", previous, self._session_id
            )
        finally:
            if self._plugin_binding_task is asyncio.current_task():
                self._plugin_binding_task = None
            self._ensure_agent_drain()

    async def _cancel_plugin_session_binding(self) -> None:
        task = self._plugin_binding_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._plugin_binding_task = None

    @property
    def fish_id(self) -> UUID | None:
        return self._fish_id

    @property
    def state(self) -> CompanionState:
        return self._state

    @property
    def session_id(self) -> UUID | None:
        return self._session_id

    @property
    def binding_generation(self) -> int:
        return self._binding_generation

    @property
    def reply_context(self) -> VoiceReplyContext | None:
        self._expire_reply_context_if_needed()
        return self._reply_context

    @property
    def voice_conversation_id(self) -> UUID | None:
        return self._voice_conversation_id

    @property
    def output_paused_for_speech(self) -> bool:
        return self._tts_paused_for_speech

    @property
    def output_attenuated_for_speech(self) -> bool:
        return self._tts_paused_for_speech or self._tts_ducked_for_speech

    def quiesce_binding(self, fish_id: UUID | None) -> bool:
        """Synchronously fence an old Fish before Runner awaits audio teardown."""
        if fish_id is None or fish_id != self._fish_id:
            return False
        if self._binding_quiesced:
            return True
        self._binding_quiesced = True
        # Runner retires draft providers and the physical sink before close().
        # Freeze their last acknowledged boundary while they still own audio.
        self._finish_answer_progress("interrupted")
        self._binding_generation += 1
        self._agent_listener_generation += 1
        listener = self._agent_listener_task
        if listener is not None and not listener.done():
            listener.cancel()
        expiry = self._reply_context_expiry_task
        if expiry is not None and not expiry.done():
            expiry.cancel()
        if self._agent_active_message is not None:
            self._binding_quiesce_task = asyncio.create_task(
                self._stop_agent_playback_for_reset(persist_partial=True),
                name="companion-binding-quiesce",
            )
        return True

    def attach_playback_arbiter(self, arbiter: PlaybackArbiter) -> None:
        if (
            self._playback.current_owner is not None
            or self._foreground_reservation is not None
            or self._turn_task is not None
            or self._agent_playback_task is not None
        ):
            raise RuntimeError("cannot replace an active playback arbiter")
        attach_draft = getattr(self._draft_tts, "attach_playback_arbiter", None)
        if callable(attach_draft):
            attach_draft(arbiter)
        self._playback = arbiter
        self._owns_playback = False

    def reserve_foreground_speech(self) -> ForegroundReservation:
        reservation = self._foreground_reservation
        if reservation is None:
            reservation = self._playback.reserve_foreground()
            self._foreground_reservation = reservation
            self._drop_proactive_for_priority_speech()
            self._publish_companion_activity()
        return reservation

    def release_foreground_speech(
        self,
        reservation: ForegroundReservation | None = None,
    ) -> bool:
        current = self._foreground_reservation
        if current is None or (reservation is not None and reservation != current):
            return False
        self._foreground_reservation = None
        released = self._playback.release_foreground(current)
        self._publish_companion_activity()
        self._ensure_agent_drain()
        return released

    def freeze_turn_binding(self) -> FrozenVoiceTurn | None:
        fish_id = self._fish_id
        self._expire_reply_context_if_needed()
        conversation_id = self._voice_conversation_id
        context = self._reply_context
        if fish_id is None or conversation_id is None:
            return None
        if context is not None and (
            context.fish_id != fish_id
            or context.voice_conversation_id != conversation_id
        ):
            self._clear_reply_context(context)
            context = None
        reservation_id: UUID | None = None
        if context is not None:
            if self._reply_context_reservation_id is not None:
                context = None
            else:
                reservation_id = uuid4()
                self._reply_context_reservation_id = reservation_id
        session_id = context.session_id if context is not None else self._session_id
        if session_id is None:
            return None
        return FrozenVoiceTurn(
            fish_id=fish_id,
            session_id=session_id,
            voice_conversation_id=conversation_id,
            binding_generation=self._binding_generation,
            reply_context=context,
            context_reservation_id=reservation_id,
            reply_blocked=(
                context is not None
                and self._reply_context_finalize_failed
                and self._pending_agent_reply is None
            ),
            agent_reply_finalize=(
                self._pending_agent_reply if context is not None else None
            ),
        )

    def release_frozen_turn(self, frozen_turn: FrozenVoiceTurn | None) -> None:
        if frozen_turn is not None:
            self._release_context_reservation(frozen_turn, consumed=False)

    @overload
    def refresh_pending_agent_reply(
        self, frozen_turn: FrozenVoiceTurn
    ) -> FrozenVoiceTurn: ...

    @overload
    def refresh_pending_agent_reply(
        self, frozen_turn: FrozenVoiceTurn | None
    ) -> FrozenVoiceTurn | None: ...

    def refresh_pending_agent_reply(
        self,
        frozen_turn: FrozenVoiceTurn | None,
    ) -> FrozenVoiceTurn | None:
        """Attach a post-freeze barge-in finalize without rebinding Context."""
        if frozen_turn is None or frozen_turn.agent_reply_finalize is not None:
            return frozen_turn
        context = frozen_turn.reply_context
        pending = self._pending_agent_reply
        if (
            context is None
            or pending is None
            or pending.message.message_id != context.source_message_id
            or self._reply_context != context
            or frozen_turn.fish_id != self._fish_id
            or frozen_turn.binding_generation != self._binding_generation
            or frozen_turn.voice_conversation_id != self._voice_conversation_id
            or frozen_turn.context_reservation_id != self._reply_context_reservation_id
        ):
            return frozen_turn
        return replace(frozen_turn, agent_reply_finalize=pending)

    def on_playback_started(self, owner: PlaybackOwner) -> None:
        if not self._playback.is_current(owner):
            return
        message = self._agent_active_message
        if owner.kind == "proactive" and message is not None:
            if message.message_id != owner.message_id:
                return
            publish = getattr(self._pet_events, "proactive_started", None)
            if callable(publish) and self._fish_id is not None:
                publish(
                    {
                        "fishId": str(self._fish_id),
                        "sessionId": str(message.session_id),
                        "messageId": str(message.message_id),
                        "startedAt": self._now().timestamp(),
                    }
                )
            return
        if owner.kind != "agent":
            return
        if (
            message is None
            or owner.message_id != message.message_id
            or message.activation_policy != "playback_start"
        ):
            return
        self._activate_reply_context(message)

    def _set_state(self, state: CompanionState) -> None:
        if self._state is state:
            return
        previous_state = self._state
        self._state = state
        self._pet_events.state(state.value, self._pet_generation)
        self._publish_companion_activity()
        self._record_event(
            "session_state_changed",
            metadata=self._voice_runtime_metadata(
                {"previousState": previous_state.value, "state": state.value}
            ),
        )

    def show_hearing(self, text: str) -> None:
        """Let the chat page show what is being heard before the turn is sent."""
        if self._state is not CompanionState.LISTENING:
            return
        publish = getattr(self._pet_events, "hearing", None)
        if callable(publish):
            publish(text.strip())

    def _publish_companion_activity(self) -> None:
        self._ensure_plugin_session_binding()
        publish = getattr(self._pet_events, "companion_activity", None)
        if not callable(publish) or self._fish_id is None or self._session_id is None:
            return
        owner = self._playback.current_owner
        busy = (
            self._foreground_reservation is not None
            or self._state in {CompanionState.ACKNOWLEDGING, CompanionState.THINKING}
            or (owner is not None and owner.kind != "proactive")
        )
        publish(
            {
                "fishId": str(self._fish_id),
                "sessionId": str(self._session_id),
                "busy": busy,
                "playbackBusy": owner is not None,
            }
        )

    def _begin_pet_generation(self) -> UUID:
        generation = uuid4()
        self._pet_generation = generation
        self._pet_events.expression(
            PetEmotion.NEUTRAL,
            generation,
        )
        return generation

    def _start_pet_audio(self) -> None:
        generation = self._pet_generation or self._begin_pet_generation()
        self._pet_events.audio_start(generation, self._pet_audio_sample_rate)

    def on_rendered_audio(self, pcm: bytes) -> None:
        # This callback runs after device consumption updates display_offset.
        # Draft audio uses the very same output object after exact adoption;
        # unpromoted draft cues can never reach the visible pet through here.
        output = self._playback.current_tts
        if output is not None and self._pet_generation is not None:
            cue = output.pop_facial_cue()
            if cue is not None:
                self._pet_events.facial(cue, self._pet_generation)
        self._pet_events.pcm(pcm)

    def _stop_pet_audio(self, *, cancel: bool = False) -> None:
        generation = self._pet_generation
        if cancel:
            self._pet_events.cancel(generation)
        self._pet_events.audio_stop(generation)
        self._pet_generation = None

    def _apply_pet_metadata(self, data: dict[str, object]) -> None:
        raw_emotion = data.get("emotion")
        try:
            emotion = (
                PetEmotion(raw_emotion)
                if isinstance(raw_emotion, str)
                else PetEmotion.NEUTRAL
            )
        except ValueError:
            return
        self._pet_events.expression(emotion, self._pet_generation)

    async def set_binding(self, fish_id: UUID | None) -> None:
        self._audio_recovery = None
        if fish_id == self._fish_id and self._turn_task is None:
            if (
                self._agent_playback_task is not None
                and not self._agent_playback_task.done()
            ):
                self._start_agent_listener()
                return
            self._set_state(
                CompanionState.LISTENING
                if self.reply_context is not None
                else CompanionState.STANDBY
                if fish_id is not None
                else CompanionState.UNBOUND
            )
            if fish_id is not None:
                await self._resolve_agent_session_for_binding(
                    fish_id,
                    self._binding_generation,
                )
            self._start_agent_listener()
            return
        if fish_id != self._fish_id:
            # Invalidate all delayed work before its cancellation is awaited.
            # A Home Assistant stream may deliver its final event concurrently
            # with cancellation; the generation prevents that old Fish result
            # from entering the replacement Fish's speech queue.
            self._binding_generation += 1
            self._binding_quiesced = True
            await self._cancel_plugin_session_binding()
            await self._stop_agent_side_effects(persist_partial=True)
            await self._stop_binding_side_effects()
        await self._stop_turn()
        if fish_id != self._fish_id:
            # Catch a Home Action watch created by the retiring turn between
            # the first task snapshot and that turn's cancellation.
            await self._stop_binding_side_effects()
        self._fish_id = fish_id
        self._session_id = None
        self._voice_conversation_id = None
        self._binding_quiesced = False
        self._set_state(
            CompanionState.STANDBY if fish_id is not None else CompanionState.UNBOUND
        )
        if fish_id is not None:
            await self._resolve_agent_session_for_binding(
                fish_id,
                self._binding_generation,
            )
        self._start_agent_listener()

    async def reset_for_audio_recovery(self, fish_id: UUID) -> bool:
        """Reset a failed audio turn without briefly unbinding the Fish.

        Audio recovery must not use ``set_binding`` as a reset shortcut: its
        same-binding fast path intentionally leaves an active acknowledgement
        and the current voice session alone. A replacement native audio chain
        cannot safely inherit either of those states.

        Return ``False`` when the runner is recovering an obsolete binding so
        it can stop rebuilding audio for a Fish that is no longer current.
        """
        if self._fish_id != fish_id:
            return False
        if self._audio_recovery is None:
            self._audio_recovery = _AudioRecovery(
                fish_id,
                self._binding_generation,
                self._state not in {CompanionState.STANDBY, CompanionState.UNBOUND},
            )
        await self._cancel_plugin_session_binding()
        await self._reset_agent_conversation(persist_partial=True)
        await self._stop_turn()
        if self._fish_id != fish_id:
            return False
        self._session_id = None
        self._voice_conversation_id = None
        self._binding_quiesced = False
        self._set_state(CompanionState.STANDBY)
        self._ensure_agent_drain()
        return True

    async def resume_after_audio_recovery(self, fish_id: UUID) -> bool:
        """Reconnect after the replacement backend is ready, without another wake."""
        recovery = self._audio_recovery
        if self._fish_id != fish_id:
            return False
        if recovery is None:
            return True
        if recovery.binding_generation != self._binding_generation:
            return False
        session_id = await self._resolve_agent_session_for_binding(
            fish_id, recovery.binding_generation
        )
        if self._audio_recovery is not recovery or self._fish_id != fish_id:
            return False
        if session_id is None:
            logger.warning("音频已恢复，但会话暂不可用；保留恢复状态")
            return True
        self._audio_recovery = None
        if recovery.resume_listening and self._state is CompanionState.STANDBY:
            self._voice_conversation_id = self._voice_conversation_id or uuid4()
            self._set_state(CompanionState.LISTENING)
        self._start_agent_listener()
        self._publish_companion_activity()
        logger.info("音频恢复完成，已重连后台消息，状态=%s", self._state.value)
        return True

    async def on_wake_word(self, *, play_ack: bool = True) -> bool:
        if self._fish_id is None or self._state is not CompanionState.STANDBY:
            return False
        fish_id = self._fish_id
        binding_generation = self._binding_generation
        session_id = await self._resolve_bound_session(
            fish_id,
            binding_generation,
        )
        if session_id is None:
            return False
        if (
            binding_generation != self._binding_generation
            or self._fish_id != fish_id
            or self._state is not CompanionState.STANDBY
        ):
            return False
        self._voice_conversation_id = uuid4()
        self._start_agent_listener()
        self._record_event("wake_word_detected", source="kws")
        if not play_ack:
            self._begin_pet_generation()
            self._set_state(CompanionState.LISTENING)
            self._record_event("wake_word_ack_skipped", source="replay")
            return True
        self._begin_pet_generation()
        self._set_state(CompanionState.ACKNOWLEDGING)
        self._wake_ack_text = WAKE_ACK_TEXT
        self._wake_ack_task = asyncio.create_task(self._run_wake_ack())
        return True

    async def submit_transcript(
        self,
        text: str,
        *,
        frozen_turn: FrozenVoiceTurn | None = None,
    ) -> bool:
        return await self._submit_turn(
            text,
            voice_turn=None,
            frozen_turn=frozen_turn or self.freeze_turn_binding(),
            spoken=True,
        )

    async def submit_voice_turn(
        self,
        turn_id: UUID,
        revision: int,
        promote_revision: int,
        content: str,
        *,
        frozen_turn: FrozenVoiceTurn | None = None,
    ) -> bool:
        if revision < 1 or promote_revision < 0:
            self._record_submit_turn_rejected(
                "invalid_voice_revision",
                input_chars=len(content),
                turn_id=turn_id,
            )
            return False
        return await self._submit_turn(
            content,
            voice_turn=(turn_id, revision, promote_revision),
            frozen_turn=frozen_turn or self.freeze_turn_binding(),
        )

    async def _submit_turn(
        self,
        text: str,
        *,
        voice_turn: tuple[UUID, int, int] | None,
        frozen_turn: FrozenVoiceTurn | None,
        submitted: asyncio.Future[CompanionAskResult] | None = None,
        spoken: bool | None = None,
        attachment_ids: tuple[UUID, ...] = (),
    ) -> bool:
        content = text.strip()
        turn_id = voice_turn[0] if voice_turn is not None else None
        if frozen_turn is None:
            self._record_submit_turn_rejected(
                "missing_turn_binding",
                input_chars=len(content),
                turn_id=turn_id,
            )
            return False
        if not content or frozen_turn.reply_blocked:
            self.release_frozen_turn(frozen_turn)
            self._record_submit_turn_rejected(
                "empty_content" if not content else "reply_context_blocked",
                input_chars=len(content),
                turn_id=turn_id,
            )
            return False
        fish_id = frozen_turn.fish_id
        session_id = frozen_turn.session_id
        binding_generation = frozen_turn.binding_generation
        if (
            fish_id != self._fish_id
            or binding_generation != self._binding_generation
            or frozen_turn.voice_conversation_id != self._voice_conversation_id
        ):
            self._release_context_reservation(frozen_turn, consumed=False)
            self._record_submit_turn_rejected(
                "binding_changed"
                if (
                    fish_id != self._fish_id
                    or binding_generation != self._binding_generation
                )
                else "voice_conversation_changed",
                input_chars=len(content),
                turn_id=turn_id,
            )
            return False
        if not await self._settle_frozen_agent_progress(frozen_turn):
            self._release_context_reservation(frozen_turn, consumed=False)
            self._record_submit_turn_rejected(
                "agent_progress_unsettled",
                input_chars=len(content),
                turn_id=turn_id,
            )
            return False
        pending_reply = frozen_turn.agent_reply_finalize
        if pending_reply is not None:
            finalized = await self._persist_agent_consumption(
                pending_reply.message,
                pending_reply.consumed_offset,
                interrupted=True,
            )
            if not finalized:
                self._reply_context_finalize_failed = True
                self._release_context_reservation(frozen_turn, consumed=False)
                self._record_submit_turn_rejected(
                    "agent_reply_finalize_failed",
                    input_chars=len(content),
                    turn_id=turn_id,
                )
                return False
            if self._pending_agent_reply == pending_reply:
                self._pending_agent_reply = None
            self._reply_context_finalize_failed = False
        if self._state is CompanionState.ACKNOWLEDGING:
            await self._stop_wake_ack()
            if (
                binding_generation != self._binding_generation
                or fish_id != self._fish_id
                or session_id != self._session_id
            ):
                self._release_context_reservation(frozen_turn, consumed=False)
                self._record_submit_turn_rejected(
                    "binding_changed_after_wake_ack",
                    input_chars=len(content),
                    turn_id=turn_id,
                )
                return False
            self._set_state(CompanionState.LISTENING)
        if (
            binding_generation != self._binding_generation
            or fish_id != self._fish_id
            or session_id != self._session_id
            or self._state is not CompanionState.LISTENING
        ):
            self._release_context_reservation(frozen_turn, consumed=False)
            self._record_submit_turn_rejected(
                "binding_changed"
                if (
                    binding_generation != self._binding_generation
                    or fish_id != self._fish_id
                )
                else "session_changed"
                if session_id != self._session_id
                else "state_not_listening",
                input_chars=len(content),
                turn_id=turn_id,
            )
            return False

        if not await self._settle_answer_progress():
            self._release_context_reservation(frozen_turn, consumed=False)
            logger.warning(
                "Waiting for prior answer playback progress before submitting next turn"
            )
            return False
        self._begin_pet_generation()
        self._set_state(CompanionState.THINKING)
        self._tts_started = False
        self._draft_tts_waiting = False
        self._turn_tts = None
        self._voice_tts_owner = (
            _VoiceTtsCandidate(
                VoiceDraftKey(fish_id, session_id, voice_turn[0]),
                voice_turn[2],
                frozen_turn.voice_conversation_id,
                frozen_turn.reply_context,
            )
            if voice_turn is not None and voice_turn[2] > 0
            else None
        )
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        self._answer_message_id = None
        self._answer_event_recorded = False
        self._answer_first_content_recorded = False
        self._answer_offset = 0
        self._answer_flush_offset = 0
        self._assistant_text = ""
        self._assistant_speech = ""
        self._speech_tags = False
        self._tts_text_buffer = TtsTextBuffer(first_chunk_at_pause=True)
        self._turn_idle.clear()
        self._turn_task = asyncio.create_task(
            self._run_turn(
                fish_id,
                session_id,
                content,
                binding_generation,
                voice_turn=voice_turn,
                frozen_turn=frozen_turn,
                submitted=submitted,
                spoken=spoken,
                attachment_ids=attachment_ids,
            )
        )
        return True

    def is_likely_echo(self, text: str) -> bool:
        """Avoid treating a short ASR transcription of our voice as an interruption."""
        candidate = _compact_text(text)
        answer = _compact_text(self._playback.current_spoken_text)
        if not candidate or not answer:
            return False
        if len(candidate) == 1:
            return answer.startswith(candidate)
        prefix = answer[: max(24, len(candidate) * 2)]
        return answer.startswith(candidate) or candidate in prefix

    async def pause_output_for_speech(self) -> bool:
        if not self._tts_started or self._tts_paused_for_speech:
            owner = self._playback.current_owner
            logger.info(
                "插话候选未暂停播放: tts_started=%s, already_paused=%s, owner_kind=%s",
                self._tts_started,
                self._tts_paused_for_speech,
                owner.kind if owner is not None else None,
            )
            return False
        output = self._playback.current_tts
        if output is None:
            owner = self._playback.current_owner
            logger.warning(
                "插话候选无法定位当前 TTS: owner_kind=%s",
                owner.kind if owner is not None else None,
            )
            return False
        binding_generation = self._binding_generation
        playback_owner = self._playback.current_owner
        await output.pause()
        if (
            binding_generation != self._binding_generation
            or playback_owner != self._playback.current_owner
        ):
            return False
        self._tts_paused_for_speech = True
        self._pet_events.audio_pause(self._pet_generation)
        if self._tts_ducked_for_speech:
            self._tts_ducked_for_speech = False
            await output.unduck()
        self._record_event("tts_playback_paused", source="tts")
        owner = self._playback.current_owner
        logger.info(
            "已暂停当前 TTS 等待插话确认: owner_kind=%s",
            owner.kind if owner is not None else None,
        )
        return True

    async def attenuate_output_for_speech(self) -> bool:
        if not self._tts_started or self.output_attenuated_for_speech:
            return False
        output = self._playback.current_tts
        if output is None:
            return False
        binding_generation = self._binding_generation
        playback_owner = self._playback.current_owner
        await output.duck()
        if (
            binding_generation != self._binding_generation
            or playback_owner != self._playback.current_owner
        ):
            return False
        self._tts_ducked_for_speech = True
        self._pet_events.audio_pause(self._pet_generation)
        self._record_event("tts_playback_ducked", source="tts")
        return True

    async def resume_output_after_speech(self) -> None:
        output = self._playback.current_tts
        if self._tts_ducked_for_speech:
            self._tts_ducked_for_speech = False
            if output is not None:
                await output.unduck()
                self._record_event("tts_playback_restored", source="tts")
            self._pet_events.audio_resume(self._pet_generation)
            return
        if not self._tts_paused_for_speech:
            return
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        if output is not None:
            await output.resume()
            self._record_event("tts_playback_resumed", source="tts")
            owner = self._playback.current_owner
            logger.info(
                "插话候选未成立，已恢复当前 TTS: owner_kind=%s",
                owner.kind if owner is not None else None,
            )
        self._pet_events.audio_resume(self._pet_generation)

    async def _resolve_agent_session_for_binding(
        self,
        fish_id: UUID,
        binding_generation: int,
    ) -> UUID | None:
        if not any(
            callable(getattr(self._api, name, None))
            for name in ("iter_companion_messages", "iter_agent_messages")
        ):
            return None
        return await self._resolve_bound_session(fish_id, binding_generation)

    async def _resolve_bound_session(
        self,
        fish_id: UUID,
        binding_generation: int,
    ) -> UUID | None:
        session_id = self._session_id
        if session_id is not None:
            return session_id
        try:
            resolved_session_id = self._plugin_session_for_fish(fish_id)
            if resolved_session_id is None:
                resolved_session_id = await self._api.resolve_session(fish_id)
        except Exception:
            logger.exception("Unable to resolve Session for Fish %s", fish_id)
            return None
        if (
            binding_generation != self._binding_generation
            or fish_id != self._fish_id
            or self._binding_quiesced
        ):
            return None
        # A concurrent wake or same-Fish binding may have completed resolution
        # first. Keep that established Session so listener ownership cannot
        # switch underneath either caller.
        session_id = self._session_id
        if session_id is None:
            session_id = resolved_session_id
            self._session_id = session_id
            # Binding enters standby before the session is known. Republish now
            # so a replacement audio process supersedes the old busy snapshot.
            self._publish_companion_activity()
        return session_id

    def _background_message_iterator(
        self,
    ) -> Callable[..., AsyncIterator[CompanionMessage]] | None:
        for name in ("iter_companion_messages", "iter_agent_messages"):
            iterator = getattr(self._api, name, None)
            if callable(iterator):
                return cast(
                    Callable[..., AsyncIterator[CompanionMessage]],
                    iterator,
                )
        return None

    def _start_agent_listener(self) -> None:
        fish_id = self._fish_id
        session_id = self._session_id
        iterator = self._background_message_iterator()
        if (
            fish_id is None
            or session_id is None
            or self._binding_quiesced
            or not callable(iterator)
        ):
            return
        task = self._agent_listener_task
        if task is not None and not task.done():
            if self._agent_listener_session_id == session_id:
                return
            logger.error("Agent listener session changed without reset")
            return
        self._agent_listener_generation += 1
        generation = self._agent_listener_generation
        self._agent_listener_session_id = session_id
        self._agent_listener_task = asyncio.create_task(
            self._run_agent_listener(fish_id, session_id, generation),
            name=f"companion-agent-listener-{fish_id}-{session_id}",
        )

    async def _run_agent_listener(
        self,
        fish_id: UUID,
        session_id: UUID,
        generation: int,
    ) -> None:
        reconnect_attempt = 0
        try:
            while (
                fish_id == self._fish_id
                and session_id == self._session_id
                and generation == self._agent_listener_generation
            ):
                try:
                    iterator = self._background_message_iterator()
                    if iterator is None:
                        return
                    async for message in iterator(
                        fish_id,
                        session_id=session_id,
                        after_sequence=self._agent_after_cursor,
                    ):
                        if (
                            fish_id != self._fish_id
                            or session_id != self._session_id
                            or generation != self._agent_listener_generation
                        ):
                            return
                        reconnect_attempt = 0
                        self._accept_agent_message(message)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "Agent Message listener disconnected for Fish %s",
                        fish_id,
                        exc_info=True,
                    )
                if (
                    fish_id != self._fish_id
                    or session_id != self._session_id
                    or generation != self._agent_listener_generation
                ):
                    return
                delay = self._agent_reconnect_delays_seconds[
                    min(
                        reconnect_attempt,
                        len(self._agent_reconnect_delays_seconds) - 1,
                    )
                ]
                reconnect_attempt += 1
                await asyncio.sleep(delay)
        finally:
            if self._agent_listener_task is asyncio.current_task():
                self._agent_listener_task = None
                self._agent_listener_session_id = None

    def _accept_agent_message(self, message: CompanionMessage) -> None:
        if message.session_id != self._session_id:
            logger.warning(
                "Ignoring Agent Message for another Session: message_id=%s",
                message.message_id,
            )
            return
        self._agent_after_cursor = max(self._agent_after_cursor, message.sequence)
        if message.message_id in self._agent_seen_ids:
            return
        self._agent_seen_ids.add(message.message_id)
        self._agent_seen_order.append(message.message_id)
        while len(self._agent_seen_order) > _AGENT_SEEN_MESSAGE_LIMIT:
            expired = self._agent_seen_order.pop(0)
            self._agent_seen_ids.discard(expired)
        if message.consumed_offset == -1 or message.consumed_offset >= len(
            message.content
        ):
            return
        if message.source == "plugin" and (
            self._playback.current_owner is not None or self._agent_queue
        ):
            self._schedule_proactive_drop(message, message.consumed_offset)
            return
        if message.source in {"proactive_observation", "plugin"}:
            if self._proactive_expired(message):
                self._schedule_proactive_drop(message, message.consumed_offset)
                return
            if self._playback.foreground_reserved and not (
                message.supersession_key or ""
            ).startswith("codex-completion:"):
                self._schedule_proactive_drop(message, message.consumed_offset)
                return
            self._supersede_proactive_message(message)
        else:
            self._drop_proactive_for_priority_speech()
        self._agent_queue.append(message)
        self._agent_queue.sort(key=lambda item: (item.sequence, item.message_id.int))
        if message.source in {"proactive_observation", "plugin"}:
            self._schedule_proactive_expiry(message)
        self._ensure_agent_drain()

    def _proactive_expired(self, message: CompanionMessage) -> bool:
        expires_at = message.expires_at
        return (
            not message.streaming
            and expires_at is not None
            and self._now() >= expires_at
        )

    def _schedule_proactive_expiry(self, message: CompanionMessage) -> None:
        expires_at = message.expires_at
        if (
            message.streaming
            or expires_at is None
            or message.message_id in self._proactive_terminal_ids
        ):
            return
        existing = self._proactive_expiry_tasks.pop(message.message_id, None)
        if existing is not None:
            existing.cancel()

        async def expire() -> None:
            try:
                delay = max(0.0, (expires_at - self._now()).total_seconds())
                await asyncio.sleep(delay)
                self._schedule_proactive_drop(
                    message,
                    max(0, message.consumed_offset),
                )
            except asyncio.CancelledError:
                raise
            finally:
                current = self._proactive_expiry_tasks.get(message.message_id)
                if current is asyncio.current_task():
                    self._proactive_expiry_tasks.pop(message.message_id, None)

        self._proactive_expiry_tasks[message.message_id] = asyncio.create_task(
            expire(),
            name=f"companion-proactive-expiry-{message.message_id}",
        )

    def _supersede_proactive_message(self, message: CompanionMessage) -> None:
        key = message.supersession_key
        if key is None:
            return
        superseded = [
            queued
            for queued in self._agent_queue
            if queued.source in {"proactive_observation", "plugin"}
            and queued.supersession_key == key
        ]
        if superseded:
            superseded_ids = {item.message_id for item in superseded}
            self._agent_queue[:] = [
                item
                for item in self._agent_queue
                if item.message_id not in superseded_ids
            ]
            for item in superseded:
                self._schedule_proactive_drop(item, max(0, item.consumed_offset))
        active = self._agent_active_message
        if (
            active is not None
            and active.source in {"proactive_observation", "plugin"}
            and active.supersession_key == key
        ):
            self._schedule_proactive_drop(active, max(0, active.consumed_offset))
        pending = self._agent_pending_message
        if (
            pending is not None
            and pending.source in {"proactive_observation", "plugin"}
            and pending.supersession_key == key
        ):
            self._schedule_proactive_drop(pending, max(0, pending.consumed_offset))

    def _drop_proactive_for_priority_speech(self) -> None:
        queued = [
            item
            for item in self._agent_queue
            if item.source in {"proactive_observation", "plugin"}
            and not (item.supersession_key or "").startswith("codex-completion:")
        ]
        if queued:
            queued_ids = {item.message_id for item in queued}
            self._agent_queue[:] = [
                item for item in self._agent_queue if item.message_id not in queued_ids
            ]
            for item in queued:
                self._schedule_proactive_drop(item, max(0, item.consumed_offset))
        active = self._agent_active_message
        if active is not None and active.source in {"proactive_observation", "plugin"}:
            self._schedule_proactive_drop(active, max(0, active.consumed_offset))
        pending = self._agent_pending_message
        if pending is not None and pending.source in {
            "proactive_observation",
            "plugin",
        }:
            self._schedule_proactive_drop(pending, max(0, pending.consumed_offset))

    async def _prepare_foreground_answer_playback(self) -> None:
        self._drop_proactive_for_priority_speech()
        while True:
            owner = self._playback.current_owner
            active = self._agent_active_message
            if (owner is None or owner.kind != "proactive") and (
                active is None
                or active.source not in {"proactive_observation", "plugin"}
            ):
                return
            await asyncio.sleep(0)

    def _schedule_proactive_drop(
        self,
        message: CompanionMessage,
        consumed_offset: int,
    ) -> None:
        if message.message_id in self._proactive_terminal_ids:
            return
        self._mark_proactive_terminal(message.message_id)
        task = asyncio.create_task(
            self._drop_proactive_message(message, consumed_offset),
            name=f"companion-proactive-drop-{message.message_id}",
        )
        self._track_proactive_settlement(task)

    async def _drop_proactive_message(
        self,
        message: CompanionMessage,
        consumed_offset: int,
    ) -> None:
        active = self._agent_active_message
        if active is not None and active.message_id == message.message_id:
            # The producer may have appended text since expiry/supersession was
            # scheduled. Settle against the live prefix and stop the same owner.
            message = active
        self._agent_queue[:] = [
            item for item in self._agent_queue if item.message_id != message.message_id
        ]
        owner = self._playback.current_owner
        output = self._agent_tts
        if (
            owner is not None
            and owner.kind == "proactive"
            and owner.message_id == message.message_id
            and self._agent_active_message == message
            and output is not None
        ):
            consumed_offset = self._proactive_interruption_offset(message, output)
            await self._halt_agent_playback(owner, output)
        await self._persist_proactive_terminal(
            message,
            consumed_offset,
            interrupted=True,
        )
        # Cancelling the active playback also unwinds the queue worker that
        # awaited it. Let that worker clear its slot before starting the
        # replacement delivery.
        await asyncio.sleep(0)
        self._ensure_agent_drain()

    def _mark_proactive_terminal(self, message_id: UUID) -> None:
        self._proactive_terminal_ids.add(message_id)
        expiry = self._proactive_expiry_tasks.pop(message_id, None)
        if expiry is not None and expiry is not asyncio.current_task():
            expiry.cancel()

    def _clear_proactive_terminal_if_inactive(self, message_id: UUID) -> None:
        if message_id in self._proactive_unsettled_ids:
            return
        pending = self._agent_pending_message
        active = self._agent_active_message
        if (
            (pending is not None and pending.message_id == message_id)
            or (active is not None and active.message_id == message_id)
            or any(item.message_id == message_id for item in self._agent_queue)
        ):
            return
        self._proactive_terminal_ids.discard(message_id)

    def _track_proactive_settlement(self, task: asyncio.Task[None]) -> None:
        self._proactive_settlement_tasks.add(task)
        task.add_done_callback(self._proactive_settlement_tasks.discard)

    async def _cancel_proactive_background_tasks(self) -> None:
        current = asyncio.current_task()
        tasks: list[asyncio.Task[None]] = []
        for expiry in self._proactive_expiry_tasks.values():
            if expiry is not current and not expiry.done():
                expiry.cancel()
                tasks.append(expiry)
        self._proactive_expiry_tasks.clear()
        for settlement in tuple(self._proactive_settlement_tasks):
            if settlement is not current and not settlement.done():
                settlement.cancel()
                tasks.append(settlement)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._proactive_settlement_tasks.difference_update(tasks)

    async def _persist_proactive_terminal(
        self,
        message: CompanionMessage,
        consumed_offset: int,
        *,
        interrupted: bool,
    ) -> None:
        if await self._persist_agent_consumption(
            message,
            consumed_offset,
            interrupted=interrupted,
        ):
            self._proactive_unsettled_ids.discard(message.message_id)
            self._clear_proactive_terminal_if_inactive(message.message_id)
            return
        self._proactive_unsettled_ids.add(message.message_id)
        self._proactive_terminal_ids.add(message.message_id)
        binding_generation = self._binding_generation

        async def retry() -> None:
            attempt = 0
            while binding_generation == self._binding_generation:
                delay = self._agent_reconnect_delays_seconds[
                    min(attempt, len(self._agent_reconnect_delays_seconds) - 1)
                ]
                attempt += 1
                await asyncio.sleep(delay)
                if await self._persist_agent_consumption(
                    message,
                    consumed_offset,
                    interrupted=interrupted,
                ):
                    self._proactive_unsettled_ids.discard(message.message_id)
                    self._clear_proactive_terminal_if_inactive(message.message_id)
                    return

        task = asyncio.create_task(
            retry(),
            name=f"companion-proactive-consumption-{message.message_id}",
        )
        self._track_proactive_settlement(task)

    def _ensure_agent_drain(self) -> None:
        if self._ensure_plugin_session_binding():
            return
        if self._fish_id is None or not self._agent_queue:
            return
        task = self._agent_drain_task
        if task is None or task.done():
            self._agent_drain_task = asyncio.create_task(
                self._run_agent_queue(),
                name="companion-agent-playback-queue",
            )

    def _agent_playback_blocked(self) -> bool:
        self._expire_reply_context_if_needed()
        return (
            self._agent_focus_blocked()
            or self._binding_quiesced
            or self._playback.foreground_reserved
            or self._playback.current_owner is not None
            or self._turn_task is not None
            or self._state
            in {
                CompanionState.ACKNOWLEDGING,
                CompanionState.THINKING,
                CompanionState.SPEAKING,
            }
        )

    def _agent_focus_blocked(self) -> bool:
        return (
            self._reply_context is not None or self._agent_progress_barrier is not None
        )

    def _refresh_agent_focus_idle(self) -> None:
        if self._agent_focus_blocked():
            self._agent_focus_idle.clear()
        else:
            self._agent_focus_idle.set()

    async def _wait_for_agent_focus_idle(
        self,
        binding_generation: int,
        fish_id: UUID,
    ) -> bool:
        while True:
            self._expire_reply_context_if_needed()
            self._refresh_agent_focus_idle()
            await self._agent_focus_idle.wait()
            if (
                binding_generation != self._binding_generation
                or fish_id != self._fish_id
            ):
                return False
            if not self._agent_focus_blocked():
                return True

    async def _run_agent_queue(self) -> None:
        try:
            # Coalesce one delivery batch so an out-of-order reconnect page is
            # sorted before any item claims live playback.
            await asyncio.sleep(0)
            while self._agent_queue and not self._agent_playback_blocked():
                message = self._agent_queue.pop(0)
                self._agent_pending_message = message
                task = asyncio.create_task(
                    self._play_agent_message(message),
                    name=f"companion-agent-message-{message.message_id}",
                )
                self._agent_playback_task = task
                try:
                    await task
                finally:
                    if self._agent_playback_task is task:
                        self._agent_playback_task = None
                    if self._agent_pending_message == message:
                        self._agent_pending_message = None
                    self._clear_proactive_terminal_if_inactive(message.message_id)
                if self._reply_context is not None:
                    return
        finally:
            if self._agent_drain_task is asyncio.current_task():
                self._agent_drain_task = None
            self._ensure_plugin_session_binding()

    async def _play_agent_message(self, message: CompanionMessage) -> None:
        fish_id = self._fish_id
        session_id = self._session_id
        binding_generation = self._binding_generation
        if fish_id is None or session_id is None or message.session_id != session_id:
            return
        playback_kind: Literal["agent", "proactive"] = (
            "proactive"
            if message.source in {"proactive_observation", "plugin"}
            else "agent"
        )
        if message.source == "plugin":
            owner = self._playback.try_acquire(playback_kind, message.message_id)
            if owner is None:
                await self._persist_proactive_terminal(
                    message, max(0, message.consumed_offset), interrupted=True
                )
                return
        else:
            owner = await self._playback.acquire(playback_kind, message.message_id)
        if message.message_id in self._proactive_terminal_ids:
            await self._playback.release(owner)
            return
        if message.source in {
            "proactive_observation",
            "plugin",
        } and self._proactive_expired(message):
            self._mark_proactive_terminal(message.message_id)
            await self._playback.release(owner)
            await self._persist_proactive_terminal(
                message,
                max(0, message.consumed_offset),
                interrupted=True,
            )
            return
        if (
            fish_id != self._fish_id
            or session_id != self._session_id
            or message.session_id != session_id
            or binding_generation != self._binding_generation
            or self._agent_post_acquire_blocked(owner)
        ):
            if message.source in {"proactive_observation", "plugin"}:
                self._mark_proactive_terminal(message.message_id)
                await self._persist_proactive_terminal(
                    message,
                    max(0, message.consumed_offset),
                    interrupted=True,
                )
            else:
                self._requeue_agent_message(message, message.consumed_offset)
            await self._playback.release(owner)
            return
        self._agent_active_message = message
        self._playback_owner = owner
        output = self._playback.bind_tts(owner, self._tts)
        self._agent_tts = output
        self._voice_conversation_id = self._voice_conversation_id or uuid4()
        self._begin_pet_generation()
        self._pet_events.expression(PetEmotion(message.emotion), self._pet_generation)
        self._set_state(CompanionState.SPEAKING)
        self._playback.set_spoken_text(owner, message.content)
        self._tts_started = False
        self._draft_tts_waiting = False
        start_offset = max(0, message.consumed_offset)
        buffer: TtsTextBuffer | SpeechTextBuffer = (
            SpeechTextBuffer() if message.speech_tags else TtsTextBuffer()
        )
        speech_content = message.speech_content
        provider_started = False
        progress_settled = False
        caption: PlaybackCaption | None = None
        try:
            output.set_speech_tags(message.speech_tags)
            await output.start(text_offset=start_offset)
            if not self._playback.is_current(owner):
                return
            provider_started = True
            self._tts_started = True
            self._start_pet_audio()
            publish_caption = getattr(self._pet_events, "playback_progress", None)
            if callable(publish_caption):

                def sample_caption() -> tuple[str, int, int, str]:
                    consumed = max(start_offset, output.played_offset)
                    display = getattr(output, "display_offset", consumed)
                    status = "paused" if self._tts_paused_for_speech else "playing"
                    return message.content, display, consumed, status

                caption = PlaybackCaption(
                    fish_id,
                    session_id,
                    message.message_id,
                    sample_caption,
                    publish_caption,
                    message_sequence=message.sequence,
                )
                self._agent_caption = caption
            suffix = message.content[start_offset:]
            next_offset = start_offset
            ranges: list[tuple[int, int]] = []

            async def append_chunks(chunks: list[str]) -> None:
                nonlocal next_offset
                for chunk in chunks:
                    end_offset = next_offset + len(
                        strip_speech_tags(chunk) if message.speech_tags else chunk
                    )
                    ranges.append((next_offset, end_offset))
                    next_offset = end_offset
                    self._agent_tts_ranges = tuple(ranges)
                    await output.append(chunk)

            if message.speech_tags:
                await append_chunks(
                    buffer.adopt_sent_prefix(speech_content, start_offset)
                )
            else:
                await append_chunks(buffer.append(suffix))
            # Proactive delivery commits only complete playable chunks. A short
            # completed answer must not wait for the producer's JSON tail.
            if message.streaming:
                await append_chunks(buffer.finish())
            if message.streaming:
                continuation = getattr(
                    self._api,
                    "iter_plugin_expression"
                    if message.source == "plugin"
                    else "iter_proactive_expression",
                    None,
                )
                if not callable(continuation):
                    raise RuntimeError("Proactive stream transport is unavailable")
                async with aclosing(
                    continuation(
                        fish_id,
                        message.message_id,
                        session_id=session_id,
                        after_offset=len(message.content),
                    )
                ) as response:
                    async for item in response:
                        if isinstance(item, dict):
                            text = str(item.get("text", ""))
                            speech = str(item.get("speechContent", ""))
                            if not message.speech_tags or not speech.startswith(
                                speech_content
                            ):
                                raise ValueError("Proactive speech protocol mismatch")
                            delta = speech[len(speech_content) :]
                            speech_content = speech
                        else:
                            text = item
                            delta = text
                        message = replace(message, content=message.content + text)
                        self._agent_active_message = message
                        self._playback.set_spoken_text(owner, message.content)
                        await append_chunks(buffer.append(delta))
            await append_chunks(buffer.finish())
            await output.finish()
            if not self._playback.is_current(owner):
                return
            played_offset = min(
                len(message.content),
                max(start_offset, output.played_offset),
            )
            await self._settle_agent_playback_progress(
                message,
                played_offset,
            )
            progress_settled = True
        except asyncio.CancelledError:
            if caption is not None:
                caption.finish("interrupted")
            raise
        except Exception:
            if caption is not None:
                caption.finish("failed")
            logger.exception(
                "Unable to speak Agent Message %s",
                message.message_id,
            )
            if not progress_settled and self._playback.is_current(owner):
                played_offset = (
                    self._agent_interruption_offset(message, output)
                    if provider_started
                    else start_offset
                )
                with suppress(Exception):
                    await output.stop()
                await self._settle_agent_playback_progress(
                    message,
                    played_offset,
                    interrupted=message.streaming,
                )
        finally:
            if caption is not None:
                caption.finish("completed" if progress_settled else "interrupted")
            if self._agent_caption is caption:
                self._agent_caption = None
            if self._agent_active_message == message:
                self._agent_active_message = None
            self._agent_tts = None
            self._agent_tts_ranges = ()
            self._tts_started = False
            self._draft_tts_waiting = False
            self._tts_paused_for_speech = False
            self._tts_ducked_for_speech = False
            if self._pet_generation is not None:
                self._stop_pet_audio()
            await self._playback.release(owner)
            if self._playback_owner == owner:
                self._playback_owner = None
            if fish_id == self._fish_id:
                self._set_state(
                    CompanionState.LISTENING
                    if self._voice_conversation_id is not None
                    else CompanionState.STANDBY
                )

    def _agent_interruption_offset(
        self,
        message: CompanionMessage,
        output: TtsOutput,
    ) -> int:
        start_offset = max(0, message.consumed_offset)
        played_offset = min(
            len(message.content),
            max(start_offset, output.played_offset),
        )
        if getattr(output, "consumption_is_confirmed", False):
            return played_offset
        if played_offset <= start_offset:
            return start_offset
        for segment_start, segment_end in self._agent_tts_ranges:
            if segment_start < played_offset <= segment_end:
                return max(start_offset, segment_start)
        return start_offset

    @staticmethod
    def _proactive_interruption_offset(
        message: CompanionMessage,
        output: TtsOutput,
    ) -> int:
        return min(
            len(message.content),
            max(max(0, message.consumed_offset), output.played_offset),
        )

    def _agent_post_acquire_blocked(self, owner: PlaybackOwner) -> bool:
        self._expire_reply_context_if_needed()
        return (
            not self._playback.is_current(owner)
            or self._agent_focus_blocked()
            or self._binding_quiesced
            or self._playback.foreground_reserved
            or self._turn_task is not None
            or self._state
            in {
                CompanionState.ACKNOWLEDGING,
                CompanionState.THINKING,
                CompanionState.SPEAKING,
            }
        )

    async def _settle_agent_playback_progress(
        self,
        message: CompanionMessage,
        consumed_offset: int,
        *,
        interrupted: bool = False,
    ) -> None:
        full = not interrupted and consumed_offset >= len(message.content)
        if message.source in {"proactive_observation", "plugin"}:
            if message.message_id in self._proactive_terminal_ids:
                return
            self._mark_proactive_terminal(message.message_id)
            await self._persist_proactive_terminal(
                message,
                consumed_offset,
                interrupted=not full,
            )
            return
        if full:
            self._clear_agent_playback_retry(message.message_id)
        context = self._reply_context
        if (
            not full
            and context is not None
            and context.source_message_id == message.message_id
        ):
            self._clear_reply_context(context)
        persisted = await self._persist_agent_consumption(
            message,
            consumed_offset,
            interrupted=False,
        )
        if full and persisted:
            if message.activation_policy == "playback_complete":
                self._activate_reply_context(message)
            return
        self._block_on_agent_progress(
            _AgentProgressBarrier(
                message=message,
                consumed_offset=consumed_offset,
                resume_playback=not full,
                activate_context=(
                    full and message.activation_policy == "playback_complete"
                ),
                persist_required=not persisted,
            )
        )

    async def _persist_agent_consumption(
        self,
        message: CompanionMessage,
        consumed_offset: int,
        *,
        interrupted: bool,
    ) -> bool:
        fish_id = self._fish_id
        session_id = self._session_id
        if fish_id is None or session_id is None or message.session_id != session_id:
            return False
        try:
            consume = getattr(self._api, "consume_companion_message", None)
            if callable(consume):
                await consume(
                    fish_id,
                    message.message_id,
                    session_id=session_id,
                    consumed_offset=consumed_offset,
                    interrupted=interrupted,
                    source=message.source,
                )
            elif message.source == "agent":
                await self._api.consume_agent_message(
                    fish_id,
                    message.message_id,
                    session_id=session_id,
                    consumed_offset=consumed_offset,
                    interrupted=interrupted,
                )
            else:
                return False
            return True
        except Exception:
            logger.exception(
                "Unable to persist Agent Message consumption: message_id=%s",
                message.message_id,
            )
            return False

    def _activate_reply_context(self, message: CompanionMessage) -> None:
        owner = self._playback.current_owner
        conversation_id = self._voice_conversation_id
        fish_id = self._fish_id
        if (
            owner is None
            or owner.kind != "agent"
            or owner.message_id != message.message_id
            or self._agent_active_message != message
            or fish_id is None
            or conversation_id is None
        ):
            return
        self._install_reply_context(message)

    def _install_reply_context(self, message: CompanionMessage) -> None:
        conversation_id = self._voice_conversation_id
        fish_id = self._fish_id
        if (
            fish_id is None
            or conversation_id is None
            or message.source != "agent"
            or message.subject_kind is None
            or message.subject_id is None
        ):
            return
        context = VoiceReplyContext(
            fish_id=fish_id,
            session_id=message.session_id,
            voice_conversation_id=conversation_id,
            subject_kind=message.subject_kind,
            subject_id=message.subject_id,
            source_message_id=message.message_id,
            expires_at=self._monotonic() + self._agent_reply_context_ttl_seconds,
        )
        self._reply_context = context
        self._reply_context_finalize_failed = False
        self._reply_context_reservation_id = None
        self._refresh_agent_focus_idle()
        expiry_task = self._reply_context_expiry_task
        if expiry_task is not None and expiry_task is not asyncio.current_task():
            expiry_task.cancel()
        self._reply_context_expiry_task = asyncio.create_task(
            self._expire_reply_context(context),
            name=f"companion-agent-context-{message.message_id}",
        )

    def _block_on_agent_progress(self, barrier: _AgentProgressBarrier) -> None:
        current = self._agent_progress_barrier
        if (
            current is not None
            and current.message.message_id != barrier.message.message_id
        ):
            return
        if (
            current is not None
            and replace(
                current,
                retry_delay_seconds=barrier.retry_delay_seconds,
            )
            == barrier
        ):
            barrier = current
        elif barrier.resume_playback and barrier.consumed_offset < len(
            barrier.message.content
        ):
            barrier = replace(
                barrier,
                retry_delay_seconds=self._next_agent_playback_retry_delay(
                    barrier.message.message_id,
                    barrier.consumed_offset,
                ),
            )
        else:
            self._clear_agent_playback_retry(barrier.message.message_id)
            if barrier.persist_required:
                barrier = replace(
                    barrier,
                    retry_delay_seconds=self._agent_reconnect_delays_seconds[0],
                )
        self._agent_progress_barrier = barrier
        self._refresh_agent_focus_idle()
        self._ensure_agent_progress_retry()

    def _ensure_agent_progress_retry(self) -> None:
        barrier = self._agent_progress_barrier
        if barrier is None:
            return
        task = self._agent_progress_retry_task
        if task is None or task.done():
            self._agent_progress_retry_task = asyncio.create_task(
                self._retry_agent_progress(
                    barrier.message.message_id,
                    self._binding_generation,
                ),
                name=f"companion-agent-progress-{barrier.message.message_id}",
            )

    def _next_agent_playback_retry_delay(
        self,
        message_id: UUID,
        consumed_offset: int,
    ) -> float:
        if (
            self._agent_playback_retry_message_id != message_id
            or consumed_offset > self._agent_playback_retry_offset
        ):
            self._agent_playback_retry_message_id = message_id
            self._agent_playback_retry_offset = consumed_offset
            self._agent_playback_retry_attempt = 0
        delay = self._agent_reconnect_delays_seconds[
            min(
                self._agent_playback_retry_attempt,
                len(self._agent_reconnect_delays_seconds) - 1,
            )
        ]
        self._agent_playback_retry_attempt += 1
        return delay

    def _clear_agent_playback_retry(self, message_id: UUID | None = None) -> None:
        if (
            message_id is not None
            and self._agent_playback_retry_message_id != message_id
        ):
            return
        self._agent_playback_retry_message_id = None
        self._agent_playback_retry_offset = 0
        self._agent_playback_retry_attempt = 0

    async def _settle_frozen_agent_progress(
        self,
        frozen_turn: FrozenVoiceTurn,
    ) -> bool:
        context = frozen_turn.reply_context
        barrier = self._agent_progress_barrier
        if (
            context is None
            or barrier is None
            or barrier.message.message_id != context.source_message_id
        ):
            return True
        if barrier.consumed_offset < len(barrier.message.content):
            return False
        async with self._agent_progress_lock:
            current = self._agent_progress_barrier
            if current is None:
                return True
            if current != barrier:
                return False
            persisted = (
                await self._persist_agent_consumption(
                    barrier.message,
                    barrier.consumed_offset,
                    interrupted=False,
                )
                if barrier.persist_required
                else True
            )
            if not persisted:
                return False
            self._resolve_agent_progress_barrier(barrier)
            return True

    async def _retry_agent_progress(
        self,
        message_id: UUID,
        binding_generation: int,
    ) -> None:
        barrier: _AgentProgressBarrier | None = None
        persistence_attempt = 0
        try:
            while binding_generation == self._binding_generation:
                current = self._agent_progress_barrier
                if current is None or current.message.message_id != message_id:
                    return
                if current != barrier:
                    barrier = current
                    persistence_attempt = 0
                    delay = barrier.retry_delay_seconds
                else:
                    delay = self._agent_reconnect_delays_seconds[
                        min(
                            persistence_attempt,
                            len(self._agent_reconnect_delays_seconds) - 1,
                        )
                    ]
                    persistence_attempt += 1
                await asyncio.sleep(delay)
                async with self._agent_progress_lock:
                    current = self._agent_progress_barrier
                    if (
                        current is None
                        or current.message.message_id != message_id
                        or binding_generation != self._binding_generation
                    ):
                        return
                    if current != barrier:
                        continue
                    persisted = (
                        await self._persist_agent_consumption(
                            current.message,
                            current.consumed_offset,
                            interrupted=False,
                        )
                        if current.persist_required
                        else True
                    )
                    if persisted:
                        self._resolve_agent_progress_barrier(current)
                        return
        except asyncio.CancelledError:
            raise
        finally:
            if self._agent_progress_retry_task is asyncio.current_task():
                self._agent_progress_retry_task = None
                self._ensure_agent_progress_retry()

    def _resolve_agent_progress_barrier(
        self,
        barrier: _AgentProgressBarrier,
    ) -> None:
        if self._agent_progress_barrier != barrier:
            return
        self._agent_progress_barrier = None
        if barrier.resume_playback and barrier.consumed_offset < len(
            barrier.message.content
        ):
            self._requeue_agent_message(
                barrier.message,
                barrier.consumed_offset,
            )
        elif barrier.activate_context and self._reply_context is None:
            self._clear_agent_playback_retry(barrier.message.message_id)
            self._install_reply_context(barrier.message)
        else:
            self._clear_agent_playback_retry(barrier.message.message_id)
        task = self._agent_progress_retry_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        self._refresh_agent_focus_idle()
        self._ensure_agent_drain()

    def _requeue_agent_message(
        self,
        message: CompanionMessage,
        consumed_offset: int,
    ) -> None:
        if message.source in {"proactive_observation", "plugin"}:
            self._schedule_proactive_drop(message, consumed_offset)
            return
        self._agent_queue.append(replace(message, consumed_offset=consumed_offset))
        self._agent_queue.sort(key=lambda item: (item.sequence, item.message_id.int))

    async def _expire_reply_context(self, context: VoiceReplyContext) -> None:
        try:
            delay = max(0.0, context.expires_at - self._monotonic())
            await asyncio.sleep(delay)
            if self._reply_context == context:
                self._clear_reply_context(context)
                self._ensure_agent_drain()
        except asyncio.CancelledError:
            raise
        finally:
            if self._reply_context_expiry_task is asyncio.current_task():
                self._reply_context_expiry_task = None

    def _expire_reply_context_if_needed(self) -> None:
        context = self._reply_context
        if context is not None and self._monotonic() >= context.expires_at:
            self._clear_reply_context(context)

    def _clear_reply_context(
        self,
        context: VoiceReplyContext | None = None,
    ) -> None:
        if context is not None and self._reply_context != context:
            return
        self._reply_context = None
        self._reply_context_reservation_id = None
        self._reply_context_finalize_failed = False
        self._pending_agent_reply = None
        task = self._reply_context_expiry_task
        self._reply_context_expiry_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._refresh_agent_focus_idle()

    def _release_context_reservation(
        self,
        frozen_turn: FrozenVoiceTurn,
        *,
        consumed: bool,
    ) -> None:
        reservation_id = frozen_turn.context_reservation_id
        if (
            reservation_id is None
            or self._reply_context_reservation_id != reservation_id
        ):
            return
        self._reply_context_reservation_id = None
        if consumed:
            self._clear_reply_context(frozen_turn.reply_context)
            self._ensure_agent_drain()

    async def _interrupt_agent_from_speech(self, owner: PlaybackOwner) -> None:
        message = self._agent_active_message
        output = self._playback.current_tts
        if message is None or output is None or owner.message_id != message.message_id:
            return
        consumed_offset = (
            self._proactive_interruption_offset(message, output)
            if message.source in {"proactive_observation", "plugin"}
            else self._agent_interruption_offset(message, output)
        )
        context = self._reply_context
        contextual_reply = (
            context is not None
            and context.source_message_id == message.message_id
            and message.activation_policy == "playback_start"
        )
        await self._halt_agent_playback(owner, output)
        if contextual_reply:
            # Early partial/near-end confirmation only stops audio. The
            # interrupted=true transition is frozen and attempted after the
            # authoritative ASR final, immediately before the contextual ask.
            self._pending_agent_reply = FrozenAgentReply(
                message,
                consumed_offset,
            )
            self._reply_context_finalize_failed = False
        else:
            persisted = await self._persist_agent_consumption(
                message,
                consumed_offset,
                interrupted=False,
            )
            if persisted and consumed_offset < len(message.content):
                self._block_on_agent_progress(
                    _AgentProgressBarrier(
                        message=message,
                        consumed_offset=consumed_offset,
                        resume_playback=True,
                        activate_context=False,
                        persist_required=False,
                    )
                )
            elif not persisted:
                self._block_on_agent_progress(
                    _AgentProgressBarrier(
                        message=message,
                        consumed_offset=consumed_offset,
                        resume_playback=consumed_offset < len(message.content),
                        activate_context=False,
                        persist_required=True,
                    )
                )
        if self._fish_id is not None:
            self._set_state(CompanionState.LISTENING)

    async def _halt_agent_playback(
        self,
        owner: PlaybackOwner,
        output: BoundTtsOutput,
    ) -> None:
        if self._agent_caption is not None:
            self._agent_caption.finish("interrupted")
        with suppress(Exception):
            await output.interrupt()
        task = self._agent_playback_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._playback.release(owner)
        if self._playback_owner == owner:
            self._playback_owner = None
        self._agent_active_message = None
        self._agent_tts = None
        self._agent_tts_ranges = ()
        self._tts_started = False
        self._draft_tts_waiting = False
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        if self._pet_generation is not None:
            self._stop_pet_audio(cancel=True)

    async def _stop_agent_side_effects(self, *, persist_partial: bool) -> None:
        quiesce_task = self._binding_quiesce_task
        self._binding_quiesce_task = None
        if quiesce_task is not None and quiesce_task is not asyncio.current_task():
            await asyncio.gather(quiesce_task, return_exceptions=True)
        self._agent_listener_generation += 1
        listener = self._agent_listener_task
        self._agent_listener_task = None
        self._agent_listener_session_id = None
        if listener is not None and listener is not asyncio.current_task():
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
        await self._stop_agent_playback_for_reset(persist_partial=persist_partial)
        drain = self._agent_drain_task
        self._agent_drain_task = None
        if (
            drain is not None
            and drain is not asyncio.current_task()
            and not drain.done()
        ):
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)
        progress_retry = self._agent_progress_retry_task
        self._agent_progress_retry_task = None
        if (
            progress_retry is not None
            and progress_retry is not asyncio.current_task()
            and not progress_retry.done()
        ):
            progress_retry.cancel()
            await asyncio.gather(progress_retry, return_exceptions=True)
        await self._cancel_proactive_background_tasks()
        self._agent_queue.clear()
        self._agent_pending_message = None
        self._agent_seen_ids.clear()
        self._agent_seen_order.clear()
        self._agent_after_cursor = 0
        self._clear_reply_context()
        self._pending_agent_reply = None
        self._agent_progress_barrier = None
        self._clear_agent_playback_retry()
        self._proactive_terminal_ids.clear()
        self._proactive_unsettled_ids.clear()
        self._refresh_agent_focus_idle()
        self._voice_conversation_id = None

    async def _reset_agent_conversation(self, *, persist_partial: bool) -> None:
        await self._stop_agent_side_effects(persist_partial=persist_partial)

    async def _stop_agent_playback_for_reset(
        self,
        *,
        persist_partial: bool,
    ) -> None:
        owner = self._playback.current_owner
        message = self._agent_active_message
        output = self._agent_tts
        if (
            owner is None
            or owner.kind not in {"agent", "proactive"}
            or message is None
            or output is None
        ):
            return
        consumed_offset = (
            self._proactive_interruption_offset(message, output)
            if message.source in {"proactive_observation", "plugin"}
            else self._agent_interruption_offset(message, output)
        )
        await self._halt_agent_playback(owner, output)
        if persist_partial:
            if message.source in {"proactive_observation", "plugin"}:
                self._mark_proactive_terminal(message.message_id)
                await self._persist_proactive_terminal(
                    message,
                    consumed_offset,
                    interrupted=True,
                )
                return
            persisted = await self._persist_agent_consumption(
                message,
                consumed_offset,
                interrupted=False,
            )
            if persisted and consumed_offset < len(message.content):
                self._block_on_agent_progress(
                    _AgentProgressBarrier(
                        message=message,
                        consumed_offset=consumed_offset,
                        resume_playback=True,
                        activate_context=False,
                        persist_required=False,
                    )
                )
            elif not persisted:
                self._block_on_agent_progress(
                    _AgentProgressBarrier(
                        message=message,
                        consumed_offset=consumed_offset,
                        resume_playback=consumed_offset < len(message.content),
                        activate_context=False,
                        persist_required=True,
                    )
                )

    async def _handle_answer_interrupt_request(
        self, fish_id: UUID, message_id: UUID
    ) -> None:
        if self._fish_id == fish_id:
            candidates = [
                self._agent_active_message,
                self._agent_pending_message,
                *self._agent_queue,
            ]
            for message in candidates:
                if (
                    message is not None
                    and message.message_id == message_id
                    and message.source in {"proactive_observation", "plugin"}
                ):
                    self._agent_queue[:] = [
                        item
                        for item in self._agent_queue
                        if item.message_id != message_id
                    ]
                    self._schedule_proactive_drop(
                        message, max(0, message.consumed_offset)
                    )
                    return
        if self._fish_id == fish_id and self._answer_message_id == message_id:
            await self.interrupt_from_speech()

    async def interrupt_from_speech(self) -> None:
        if self._state not in {
            CompanionState.ACKNOWLEDGING,
            CompanionState.THINKING,
            CompanionState.SPEAKING,
        }:
            return
        owner = self._playback.current_owner
        if owner is not None and owner.kind == "wake":
            # finish() holds the playback control lock while awaiting the
            # provider. Stop its owning task before releasing the audio slot.
            await self._stop_wake_ack()
            return
        if owner is not None and owner.kind == "proactive":
            message = self._agent_active_message
            if message is not None:
                self._schedule_proactive_drop(
                    message,
                    self._proactive_interruption_offset(
                        message,
                        self._playback.current_tts or self._tts,
                    ),
                )
            return
        if owner is not None and owner.kind == "agent":
            await self._interrupt_agent_from_speech(owner)
            return
        fish_id = self._fish_id
        message_id = self._answer_message_id
        turn_tts = self._playback.current_tts or self._current_tts()
        consumed_offset = max(0, turn_tts.played_offset) if self._tts_started else 0
        progress = self._answer_progress
        self._finish_answer_progress("interrupted")
        turn_task = self._turn_task
        self._turn_task = None
        if turn_task is not None and not turn_task.done():
            turn_task.cancel()
        tts_started = self._tts_started
        playback_owner = self._playback_owner
        self._tts_started = False
        self._draft_tts_waiting = False
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        self._answer_message_id = None
        self._answer_offset = 0
        self._answer_flush_offset = 0
        self._assistant_text = ""
        self._assistant_speech = ""
        self._speech_tags = False
        self._tts_text_buffer = TtsTextBuffer(first_chunk_at_pause=True)
        self._speech_tags = False
        self._assistant_speech = ""
        self._wake_ack_text = ""
        if self._pet_generation is not None:
            self._stop_pet_audio(cancel=True)
        if self._fish_id is not None:
            self._set_state(CompanionState.LISTENING)
        cleanup_operations: list[Awaitable[object]] = []
        release_playback_in_cleanup = False
        if turn_task is not None:
            cleanup_operations.append(turn_task)
        if tts_started:
            self._record_event(
                "tts_interrupt_requested",
                source="session",
                metadata={
                    "messageId": str(message_id) if message_id else None,
                    "playedOffset": consumed_offset,
                },
            )
            interrupt_tts = getattr(turn_tts, "interrupt", None)
            if callable(interrupt_tts) and getattr(
                turn_tts,
                "supports_interrupt",
                True,
            ):
                # ApiTtsOutput detaches the provider connection synchronously
                # and retires it in the background. This keeps the next TTS
                # task independent from the old provider acknowledgement.
                await interrupt_tts()
                self._record_event(
                    "tts_playback_stopped",
                    source="tts",
                    metadata={
                        "messageId": str(message_id) if message_id else None,
                        "playedOffset": consumed_offset,
                    },
                )
            else:
                # Preserve compatibility with simple test/third-party outputs
                # that only implement stop(). They remain best effort here.
                stop_task = asyncio.create_task(turn_tts.stop())
                if playback_owner is not None:
                    cleanup_operations.append(
                        self._release_playback_after_cleanup(
                            playback_owner,
                            stop_task,
                        )
                    )
                    release_playback_in_cleanup = True
                else:
                    cleanup_operations.append(stop_task)
                # Give the compatibility stop one scheduling turn while the
                # owner is still current. Production ApiTtsOutput implements
                # interrupt() and never enters this slower fallback.
                await asyncio.sleep(0)
        if playback_owner is not None and not release_playback_in_cleanup:
            await self._release_playback_owner(playback_owner)
        await self._retire_voice_tts()
        self._report_pet_voice(fish_id, message_id, PetVoiceStatus.STOPPED)
        if progress is not None:
            cleanup_operations.append(asyncio.create_task(progress.settle()))
        elif fish_id is not None and message_id is not None:
            cleanup_operations.append(
                asyncio.create_task(
                    self._persist_speech_interrupt(
                        fish_id,
                        message_id,
                        consumed_offset,
                    )
                )
            )
        # Do not block microphone processing or the next answer on provider /
        # WebSocket cleanup. The old turn is retired independently after the
        # new turn has already been allowed to start.
        self._interrupt_cleanup_task = asyncio.create_task(
            self._cleanup_speech_interrupt(cleanup_operations)
        )

    async def _release_playback_after_cleanup(
        self,
        owner: PlaybackOwner,
        cleanup: asyncio.Task[None],
    ) -> None:
        await asyncio.gather(cleanup, return_exceptions=True)
        await self._release_playback_owner(owner)

    async def _cleanup_speech_interrupt(
        self,
        operations: list[Awaitable[object]],
    ) -> None:
        try:
            if operations:
                await asyncio.gather(*operations, return_exceptions=True)
        finally:
            if self._interrupt_cleanup_task is asyncio.current_task():
                self._interrupt_cleanup_task = None

    async def _persist_speech_interrupt(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None:
        try:
            self._record_event(
                "answer_interrupt_submitted",
                source="api",
                metadata={
                    "messageId": str(message_id),
                    "consumedOffset": consumed_offset,
                },
            )
            await self._api.interrupt_answer(
                fish_id,
                message_id,
                consumed_offset,
            )
        except Exception:
            logger.exception(
                "Unable to persist speech interruption: fish_id=%s, message_id=%s",
                fish_id,
                message_id,
            )

    async def wait_for_idle(self) -> None:
        tasks = (self._wake_ack_task, self._turn_task, self._interrupt_cleanup_task)
        for task in tasks:
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._home_tasks:
            await asyncio.gather(*tuple(self._home_tasks), return_exceptions=True)
        if self._status_speech_task is not None:
            await self._status_speech_task
        for task in (self._agent_playback_task, self._agent_drain_task):
            if task is not None and task is not asyncio.current_task():
                with suppress(asyncio.CancelledError):
                    await task

    async def _run_wake_ack(self) -> None:
        tts_started = False
        owner: PlaybackOwner | None = None
        output: BoundTtsOutput | None = None
        try:
            owner = await self._playback.acquire("wake", None)
            if self._state is not CompanionState.ACKNOWLEDGING:
                return
            output = self._playback.bind_tts(owner, self._tts)
            self._playback_owner = owner
            self._playback.set_spoken_text(owner, WAKE_ACK_TEXT)
            self._record_event("tts_start_requested", source="session")
            await output.start()
            self._tts_started = True
            tts_started = True
            self._start_pet_audio()
            if self._state is not CompanionState.ACKNOWLEDGING:
                await output.stop()
                return
            await output.append(WAKE_ACK_TEXT)
            await output.finish()
        except asyncio.CancelledError:
            if self._tts_started and output is not None:
                with suppress(Exception):
                    await output.stop()
            raise
        except Exception:
            logger.exception("Unable to play wake-word acknowledgement")
            if self._tts_started and output is not None:
                await output.stop()
        finally:
            if tts_started and self._tts_started:
                self._record_event("tts_playback_stopped", source="tts")
            if tts_started:
                self._stop_pet_audio()
            self._tts_started = False
            self._draft_tts_waiting = False
            self._tts_paused_for_speech = False
            self._tts_ducked_for_speech = False
            self._wake_ack_text = ""
            if owner is not None:
                await self._release_playback_owner(owner)
            if (
                self._state is CompanionState.ACKNOWLEDGING
                and self._fish_id is not None
            ):
                self._set_state(CompanionState.LISTENING)
            if self._wake_ack_task is asyncio.current_task():
                self._wake_ack_task = None

    async def _end_voice_conversation(self) -> None:
        ended_binding = (self._fish_id, self._session_id)
        ended_generation = self._binding_generation
        await self._stop_agent_side_effects(persist_partial=True)
        self._session_id = None
        if (
            ended_binding in self._plugin_playback_sessions
            and ended_generation == self._binding_generation
            and ended_binding[0] == self._fish_id
            and not self._binding_quiesced
        ):
            # End microphone conversation state, but keep the playback
            # subscription needed by the user's running plugin.
            self._session_id = ended_binding[1]
            self._publish_companion_activity()
            self._start_agent_listener()
            return
        # The conversation ends, not TA's day: as after binding, the listener
        # rejoins the character's current Session. Its subscription is what keeps
        # 内心, 主动陪伴 and screen understanding running between conversations.
        fish_id = self._fish_id
        if (
            fish_id is not None
            and fish_id == ended_binding[0]
            and ended_generation == self._binding_generation
            and not self._binding_quiesced
        ):
            await self._resolve_agent_session_for_binding(fish_id, ended_generation)
            self._start_agent_listener()

    async def start_new_session(self, fish_id: UUID, session_id: UUID) -> bool:
        """「新对话」: what TA is saying stops and the next turn joins ``session_id``.

        The API has already created the session. Background messages of the
        old session are no longer played; long-term memory is per Fish and stays.
        """
        if self._fish_id != fish_id:
            return False
        await self._cancel_plugin_session_binding()
        await self._reset_agent_conversation(persist_partial=True)
        await self._stop_turn()
        if self._fish_id != fish_id:
            return False
        self._session_id = session_id
        self._voice_conversation_id = None
        self._binding_quiesced = False
        self._set_state(CompanionState.STANDBY)
        self._start_agent_listener()
        self._publish_companion_activity()
        logger.info("已开始新对话：Fish %s，会话 %s", fish_id, session_id)
        return True

    async def stop_listening(self) -> bool:
        """The person closed the mic: the conversation ends as when TA ends it."""
        if self._state not in {CompanionState.ACKNOWLEDGING, CompanionState.LISTENING}:
            return False
        await self._stop_wake_ack()
        self._set_state(CompanionState.STANDBY)
        await self._end_voice_conversation()
        self._record_event("listening_stopped", source="user")
        return True

    async def _stop_wake_ack(self) -> None:
        task = self._wake_ack_task
        self._wake_ack_task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        self._binding_quiesced = True
        self._binding_generation += 1
        await self._cancel_plugin_session_binding()
        await self._stop_agent_side_effects(persist_partial=True)
        self.release_foreground_speech()
        await self._stop_binding_side_effects()
        await self._stop_turn()
        await self._stop_binding_side_effects()
        draft_tts = self._draft_tts
        if draft_tts is not None:
            await draft_tts.close()
        close_tts = getattr(self._tts, "close", None)
        if callable(close_tts):
            await close_tts()
        for progress in self._pending_answer_progress:
            await progress.close()
        self._pending_answer_progress.clear()
        if self._owns_playback:
            await self._playback.close()
        self._set_state(
            CompanionState.STANDBY
            if self._fish_id is not None
            else CompanionState.UNBOUND
        )

    async def _run_turn(
        self,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        binding_generation: int,
        *,
        voice_turn: tuple[UUID, int, int] | None,
        frozen_turn: FrozenVoiceTurn,
        submitted: asyncio.Future[CompanionAskResult] | None = None,
        spoken: bool | None = None,
        attachment_ids: tuple[UUID, ...] = (),
    ) -> None:
        route = "ask" if voice_turn is None else "voice_commit"
        submission_metadata: dict[str, object] = {
            "route": route,
            "fishId": str(fish_id),
            "sessionId": str(session_id),
        }
        if voice_turn is not None:
            submission_metadata["turnId"] = str(voice_turn[0])
        submit_succeeded = False
        try:
            self._record_event(
                "ask_submit_started",
                source="api",
                metadata=self._voice_runtime_metadata(
                    {**submission_metadata, "inputChars": len(content)}
                ),
            )
            if voice_turn is None:
                result = await self._api.ask(
                    fish_id,
                    content,
                    session_id=session_id,
                    reply_context=(
                        frozen_turn.reply_context.client_context
                        if frozen_turn.reply_context is not None
                        else None
                    ),
                    input_mode=(
                        None if spoken is None else "spoken" if spoken else "typed"
                    ),
                    # Only when there are some: other clients keep their shape.
                    **({"attachment_ids": attachment_ids} if attachment_ids else {}),
                )
            else:
                turn_id, revision, promote_revision = voice_turn
                result = await self._api.commit_voice_turn(
                    VoiceDraftKey(fish_id, session_id, turn_id),
                    revision=revision,
                    promote_revision=promote_revision,
                    content=content,
                    reply_context=(
                        frozen_turn.reply_context.client_context
                        if frozen_turn.reply_context is not None
                        else None
                    ),
                )
            submit_succeeded = True
            if submitted is not None and not submitted.done():
                submitted.set_result(result)
            message_id = result.message_id
            self._record_event(
                "ask_submitted",
                source="api",
                metadata=self._voice_runtime_metadata(
                    {
                        **submission_metadata,
                        "messageId": str(message_id),
                        "answerExpected": result.answer_expected,
                    }
                ),
            )
            self._release_context_reservation(frozen_turn, consumed=True)
            if not result.answer_expected:
                await self._retire_voice_tts()
                if self._pet_generation is not None:
                    self._stop_pet_audio(cancel=True)
                if (
                    binding_generation == self._binding_generation
                    and fish_id == self._fish_id
                ):
                    self._set_state(CompanionState.LISTENING)
                return
            candidate = self._voice_tts_owner
            if candidate is not None and (
                candidate.voice_conversation_id != frozen_turn.voice_conversation_id
                or candidate.reply_context != frozen_turn.reply_context
            ):
                await self._retire_voice_tts()
            if (
                self._home_actions_enabled
                and binding_generation == self._binding_generation
                and fish_id == self._fish_id
                and session_id == self._session_id
            ):
                self._start_home_action_watch(
                    fish_id,
                    session_id,
                    message_id,
                    binding_generation,
                )
            events = aiter(self._api.iter_answer(fish_id, session_id=session_id))
            while True:
                try:
                    if self._draft_tts_waiting:
                        # A pre-synthesized draft waits for text the answer
                        # stream may never publish. Past the deadline, speak
                        # normally so the reply is always heard.
                        pending = asyncio.ensure_future(anext(events))
                        await asyncio.wait({pending}, timeout=_DRAFT_TTS_STALL_SECONDS)
                        if not pending.done() and self._draft_tts_waiting:
                            await self._fall_back_from_stalled_draft()
                        event = await pending
                    else:
                        event = await anext(events)
                except StopAsyncIteration:
                    break
                if self._fish_id != fish_id or self._session_id != session_id:
                    return
                if await self._consume_event(event):
                    return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not submit_succeeded:
                self._record_event(
                    "ask_submit_failed",
                    source="api",
                    metadata=self._voice_runtime_metadata(
                        {
                            **submission_metadata,
                            "errorType": type(error).__name__,
                        }
                    ),
                )
            logger.exception(
                "语音回答处理失败: fish_id=%s, content=%r",
                fish_id,
                content,
            )
            if self._fish_id == fish_id and self._session_id == session_id:
                answer_message_id = self._answer_message_id
                failure = failures.failure_of(error)
                code = failure.code if failure is not None else None
                self._finish_answer_progress("failed", code)
                self._record_failure(fish_id, answer_message_id, code)
                await self._settle_answer_progress()
                if self._tts_started:
                    playback_owner = self._playback_owner
                    await self._current_tts().stop()
                    self._record_event(
                        "tts_playback_stopped",
                        source="tts",
                        metadata={"messageId": str(self._answer_message_id)}
                        if self._answer_message_id is not None
                        else None,
                    )
                    self._tts_started = False
                    self._draft_tts_waiting = False
                    self._tts_paused_for_speech = False
                    self._tts_ducked_for_speech = False
                    self._stop_pet_audio(cancel=True)
                    if playback_owner is not None:
                        await self._release_playback_owner(playback_owner)
                elif self._pet_generation is not None:
                    self._stop_pet_audio(cancel=True)
                self._report_pet_voice(
                    fish_id,
                    answer_message_id,
                    PetVoiceStatus.FAILED,
                )
                self._answer_message_id = None
                self._tts_text_buffer = TtsTextBuffer(first_chunk_at_pause=True)
                self._speech_tags = False
                self._assistant_speech = ""
                await self._retire_voice_tts()
                self._set_state(CompanionState.LISTENING)
        finally:
            if submitted is not None and not submitted.done():
                submitted.set_exception(
                    RuntimeError("发送状态未确认，请先查看聊天记录；草稿已保留")
                )
            self._release_context_reservation(frozen_turn, consumed=False)
            current = asyncio.current_task()
            if self._turn_task is current:
                self._turn_task = None
            self._turn_idle.set()
            self._ensure_agent_drain()

    def _start_home_action_watch(
        self,
        fish_id: UUID,
        session_id: UUID,
        source_message_id: UUID,
        binding_generation: int,
    ) -> None:
        task = asyncio.create_task(
            self._watch_home_action(
                fish_id,
                session_id,
                source_message_id,
                binding_generation,
            ),
            name=f"companion-home-action-{source_message_id}",
        )
        self._home_tasks.add(task)
        task.add_done_callback(self._home_tasks.discard)

    async def _watch_home_action(
        self,
        fish_id: UUID,
        session_id: UUID,
        source_message_id: UUID,
        binding_generation: int,
    ) -> None:
        try:
            async with asyncio.timeout(self._home_action_timeout_seconds):
                async for event in self._api.iter_home_actions(
                    fish_id,
                    session_id=session_id,
                ):
                    data = event.get("data")
                    if not isinstance(data, dict):
                        continue
                    data = cast(dict[str, object], data)
                    if data.get("sourceMessageId") != str(source_message_id):
                        continue
                    speech = _home_action_speech(data)
                    status = data.get("status")
                    if status in {"succeeded", "partial", "failed", "ambiguous"}:
                        if (
                            speech
                            and binding_generation == self._binding_generation
                            and fish_id == self._fish_id
                        ):
                            await self._status_speech_queue.put(
                                (
                                    binding_generation,
                                    fish_id,
                                    source_message_id,
                                    speech,
                                )
                            )
                            self._ensure_status_speech_worker()
                        return
        except TimeoutError:
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Unable to receive Home Assistant result for message %s",
                source_message_id,
            )

    def _ensure_status_speech_worker(self) -> None:
        self._drop_proactive_for_priority_speech()
        if self._status_speech_task is None or self._status_speech_task.done():
            self._status_speech_task = asyncio.create_task(
                self._run_status_speech(),
                name="companion-home-status-speech",
            )

    async def _run_status_speech(self) -> None:
        try:
            while not self._status_speech_queue.empty():
                binding_generation, fish_id, source_message_id, text = (
                    self._status_speech_queue.get_nowait()
                )
                if (
                    binding_generation != self._binding_generation
                    or fish_id != self._fish_id
                ):
                    continue
                await self._turn_idle.wait()
                if (
                    binding_generation != self._binding_generation
                    or fish_id != self._fish_id
                ):
                    continue
                owner: PlaybackOwner | None = None
                while owner is None:
                    if not await self._wait_for_agent_focus_idle(
                        binding_generation,
                        fish_id,
                    ):
                        break
                    await self._turn_idle.wait()
                    if (
                        binding_generation != self._binding_generation
                        or fish_id != self._fish_id
                    ):
                        break
                    if self._agent_focus_blocked():
                        continue
                    candidate = await self._playback.acquire(
                        "home",
                        source_message_id,
                    )
                    self._expire_reply_context_if_needed()
                    if (
                        binding_generation != self._binding_generation
                        or fish_id != self._fish_id
                    ):
                        await self._release_playback_owner(candidate)
                        break
                    if self._home_post_acquire_blocked(candidate):
                        await self._release_playback_owner(candidate)
                        continue
                    owner = candidate
                if owner is None:
                    continue
                return_state = self._state
                self._begin_pet_generation()
                self._set_state(CompanionState.SPEAKING)
                output: BoundTtsOutput | None = None
                try:
                    output = self._playback.bind_tts(owner, self._tts)
                    self._playback_owner = owner
                    self._playback.set_spoken_text(owner, text)
                    await output.start()
                    self._tts_started = True
                    self._start_pet_audio()
                    await output.append(text)
                    await output.finish()
                except asyncio.CancelledError:
                    # start() may already have allocated a provider stream even
                    # when cancellation lands before it returns.
                    if output is not None:
                        with suppress(Exception):
                            await output.stop()
                    raise
                except Exception:
                    logger.exception("Unable to speak Home Assistant result")
                    if output is not None:
                        with suppress(Exception):
                            await output.stop()
                finally:
                    self._stop_pet_audio()
                    self._tts_started = False
                    self._draft_tts_waiting = False
                    if owner is not None:
                        await self._release_playback_owner(owner)
                    if (
                        binding_generation == self._binding_generation
                        and fish_id == self._fish_id
                    ):
                        self._set_state(
                            CompanionState.STANDBY
                            if return_state is CompanionState.STANDBY
                            else CompanionState.LISTENING
                        )
        finally:
            if self._status_speech_task is asyncio.current_task():
                self._status_speech_task = None

    def _home_post_acquire_blocked(self, owner: PlaybackOwner) -> bool:
        return (
            not self._playback.is_current(owner)
            or self._agent_focus_blocked()
            or self._binding_quiesced
            or self._playback.foreground_reserved
            or self._turn_task is not None
            or self._state
            in {
                CompanionState.ACKNOWLEDGING,
                CompanionState.THINKING,
                CompanionState.SPEAKING,
            }
        )

    async def _stop_binding_side_effects(self) -> None:
        """Retire work whose result belongs to the previous bound Fish."""
        home_tasks = tuple(self._home_tasks)
        for task in home_tasks:
            task.cancel()
        if home_tasks:
            await asyncio.gather(*home_tasks, return_exceptions=True)

        status_task = self._status_speech_task
        self._status_speech_task = None
        if (
            status_task is not None
            and status_task is not asyncio.current_task()
            and not status_task.done()
        ):
            status_task.cancel()
            await asyncio.gather(status_task, return_exceptions=True)

        while not self._status_speech_queue.empty():
            self._status_speech_queue.get_nowait()

    async def _consume_event(self, event: dict[str, object]) -> bool:
        event_name = event.get("event")
        data = event.get("data")
        if not isinstance(data, dict):
            return False
        data = cast(dict[str, object], data)

        message_id = data.get("messageId")
        if isinstance(message_id, str):
            try:
                self._answer_message_id = UUID(message_id)
            except ValueError:
                pass
            else:
                if not self._answer_event_recorded:
                    self._answer_event_recorded = True
                    self._record_event(
                        "answer_event_received",
                        source="sse",
                        metadata={"messageId": message_id},
                    )
                if event_name in {
                    "answer.metadata",
                    "answer.snapshot",
                    "answer.delta",
                }:
                    self._drop_proactive_for_priority_speech()

        sequence = data.get("sequence") if isinstance(message_id, str) else None
        message_sequence = (
            sequence
            if isinstance(sequence, int)
            and not isinstance(sequence, bool)
            and sequence >= 0
            else None
        )
        if (
            self._answer_message_id is not None
            and self._answer_progress is None
            and not self._binding_quiesced
        ):
            self._start_answer_progress(message_sequence)
        elif self._answer_progress is not None and message_sequence is not None:
            if self._answer_progress.message_sequence is None:
                self._answer_progress.message_sequence = message_sequence
                self._answer_progress.refresh()

        speech_delta = ""
        if data.get("speechTags") is True:
            speech = data.get("speechContent")
            if isinstance(speech, str):
                if not self._speech_tags:
                    if self._assistant_text or self._tts_started:
                        raise ValueError("speech protocol changed after playback began")
                    self._speech_tags = True
                    self._tts_text_buffer = SpeechTextBuffer()
                if not speech.startswith(self._assistant_speech):
                    raise ValueError("speech stream rewrote submitted text")
                speech_delta = speech[len(self._assistant_speech) :]
                self._assistant_speech = speech
        if (
            speech_delta
            and self._tts_started
            and event_name not in {"answer.snapshot", "answer.delta"}
        ):
            await self._append_answer_chunks(self._tts_text_buffer.append(speech_delta))

        if event_name in {"answer.snapshot", "answer.metadata"}:
            self._apply_pet_metadata(data)
        if event_name == "answer.metadata":
            return False
        if event_name == "answer.segment":
            await self._flush_tts_through(data.get("flushOffset"))
            return False

        if event_name in {"answer.snapshot", "answer.delta"}:
            content = data.get("content")
            if not isinstance(content, str) or (not content and not speech_delta):
                if event_name == "answer.snapshot":
                    await self._flush_tts_through(data.get("flushOffset"))
                return False
            if event_name == "answer.snapshot":
                new_text = content[self._answer_offset :]
                self._answer_offset = len(content)
                self._assistant_text = content
            else:
                new_text = content
                self._answer_offset += len(content)
                self._assistant_text += content
            if event_name == "answer.snapshot":
                self._observe_answer_flush(data.get("flushOffset"))
            active_owner = self._playback_owner
            if active_owner is not None and active_owner.kind == "answer":
                self._playback.set_spoken_text(
                    active_owner,
                    self._assistant_text,
                )
            if new_text or speech_delta:
                await self._prepare_foreground_answer_playback()
                if not self._answer_first_content_recorded:
                    self._answer_first_content_recorded = True
                    self._record_event(
                        "answer_first_content",
                        source="sse",
                        metadata={"messageId": message_id},
                    )
                if not self._tts_started:
                    if not await self._start_answer_tts():
                        return False
                else:
                    await self._append_answer_chunks(
                        self._tts_text_buffer.append(
                            speech_delta if self._speech_tags else new_text
                        )
                    )
                    self._set_state(CompanionState.SPEAKING)
            if event_name == "answer.snapshot":
                await self._flush_tts_through(data.get("flushOffset"))
            return False

        if event_name not in {
            "answer.completed",
            "answer.interrupted",
            "answer.failed",
        }:
            return False

        if event_name in {"answer.completed", "answer.failed"}:
            generation_failed = event_name == "answer.failed"
            reported = failures.ServiceFailure.parse(data.get("failure"))
            failure = reported.code if generation_failed and reported else None
            playback_failed = bool(self._assistant_text)
            if (
                not self._tts_started
                and self._assistant_text
                and self._voice_tts_owner is not None
            ):
                # Completion authorizes flushing all published text. A private
                # prefix that still extends beyond it cannot be adopted.
                await self._flush_tts_through(len(self._assistant_text))
                if not self._tts_started:
                    await self._retire_voice_tts()
                    await self._start_answer_tts()
            playback_owner = self._playback_owner
            if self._tts_started:
                # Generation failure ends the text stream, not valid audio
                # already in flight. Drain it with the normal interruptible
                # finish path; only an actual interruption should stop it.
                output = self._current_tts()
                await self._append_answer_chunks(self._tts_text_buffer.finish(), output)
                await output.finish()
                played_offset = max(0, output.played_offset)
                playback_failed = played_offset < len(self._assistant_text)
                self._finish_answer_progress(
                    "failed" if generation_failed or playback_failed else "completed",
                    failure,
                )
                self._record_event(
                    "answer_playback_completed",
                    source="tts",
                    metadata=self._voice_runtime_metadata(
                        {
                            "messageId": message_id,
                            "playedOffset": played_offset,
                        }
                    ),
                )
                self._record_event(
                    "tts_playback_stopped",
                    source="tts",
                    metadata={"messageId": message_id},
                )
            self._finish_answer_progress(
                "failed" if generation_failed or playback_failed else "completed",
                failure,
            )
            await self._settle_answer_progress()
            if playback_owner is not None and playback_owner.kind == "answer":
                await self._release_playback_owner(playback_owner)
            if self._pet_generation is not None:
                self._stop_pet_audio()
            self._report_pet_voice(
                self._fish_id,
                self._answer_message_id,
                PetVoiceStatus.FAILED
                if generation_failed or playback_failed
                else PetVoiceStatus.COMPLETED,
            )
            should_end = (
                not (generation_failed or playback_failed) and data.get("end") is True
            )
            self._set_state(
                CompanionState.STANDBY if should_end else CompanionState.LISTENING
            )
            if should_end:
                await self._end_voice_conversation()
            self._tts_started = False
            self._draft_tts_waiting = False
            self._tts_paused_for_speech = False
            self._tts_ducked_for_speech = False
            self._answer_message_id = None
            self._tts_text_buffer = TtsTextBuffer(first_chunk_at_pause=True)
            self._speech_tags = False
            self._assistant_speech = ""
            await self._retire_voice_tts()
        else:
            logger.info(
                "语音回答被中断: event=%s, message_id=%s, played_offset=%s",
                event_name,
                message_id,
                self._current_tts().played_offset,
            )
            playback_owner = self._playback_owner
            self._finish_answer_progress("interrupted")
            await self._settle_answer_progress()
            await self._current_tts().stop()
            if playback_owner is not None and playback_owner.kind == "answer":
                await self._release_playback_owner(playback_owner)
            self._stop_pet_audio(cancel=True)
            self._record_event(
                "tts_playback_stopped",
                source="tts",
                metadata={"messageId": message_id},
            )
            self._report_pet_voice(
                self._fish_id,
                self._answer_message_id,
                PetVoiceStatus.STOPPED
                if event_name == "answer.interrupted"
                else PetVoiceStatus.FAILED,
            )
            self._tts_started = False
            self._draft_tts_waiting = False
            self._tts_paused_for_speech = False
            self._tts_ducked_for_speech = False
            self._set_state(CompanionState.LISTENING)
            self._answer_message_id = None
            self._tts_text_buffer = TtsTextBuffer(first_chunk_at_pause=True)
            self._speech_tags = False
            self._assistant_speech = ""
            await self._retire_voice_tts()
        return True

    def _observe_answer_flush(self, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return
        self._answer_flush_offset = max(
            self._answer_flush_offset, min(value, len(self._assistant_text))
        )

    async def _flush_tts_through(self, value: object) -> None:
        self._observe_answer_flush(value)
        if not self._tts_started:
            if (
                not self._assistant_text
                or self._voice_tts_owner is None
                or not await self._start_answer_tts()
            ):
                return
        output = self._current_tts()
        await self._append_answer_chunks(
            self._tts_text_buffer.flush_through(self._answer_flush_offset), output
        )
        if self._answer_flush_offset > 0:
            await output.flush()

    async def _append_answer_chunks(
        self, chunks: list[str], output: TtsOutput | None = None
    ) -> None:
        if not chunks:
            return
        if output is None:
            output = self._current_tts()
        flush_first = self._tts_text_buffer.take_first_pause_flush()
        for index, chunk in enumerate(chunks):
            await output.append(chunk)
            if index == 0 and flush_first:
                # A short prefix alone can still wait inside the provider.
                # Commit its pause without ending this answer or connection.
                await output.flush()

    async def _start_answer_tts(self) -> bool:
        """Adopt or start speech from cumulative public text, across SSE forms."""
        if self._tts_started:
            return True
        if self._voice_tts_owner is not None:
            promoted = await self._promote_voice_tts(
                {
                    "messageId": str(self._answer_message_id),
                    "flushOffset": self._answer_flush_offset,
                },
                self._assistant_text,
            )
            if isinstance(promoted, DraftTtsPending):
                self._draft_tts_waiting = True
                return False
            if promoted:
                self._set_state(CompanionState.SPEAKING)
                return True
        answer_owner = await self._playback.acquire("answer", self._answer_message_id)
        if self._tts_started:
            await self._release_playback_owner(answer_owner)
            return True
        self._playback_owner = answer_owner
        self._turn_tts = self._playback.bind_tts(answer_owner, self._tts)
        self._playback.set_spoken_text(answer_owner, self._assistant_text)
        self._record_event(
            "tts_start_requested",
            source="session",
            metadata={"messageId": str(self._answer_message_id)},
        )
        configure_tags = getattr(self._current_tts(), "set_speech_tags", None)
        if callable(configure_tags):
            configure_tags(self._speech_tags)
        await self._current_tts().start()
        self._tts_started = True
        self._draft_tts_waiting = False
        self._start_pet_audio()
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        self._set_state(CompanionState.SPEAKING)
        # Earlier deltas may have waited for a private prefix. Start the
        # ordinary provider with their complete text if adoption becomes invalid.
        await self._append_answer_chunks(
            self._tts_text_buffer.append(
                self._assistant_speech if self._speech_tags else self._assistant_text
            )
        )
        return True

    async def _stop_turn(self) -> None:
        await self._wait_for_interrupt_cleanup()
        await self._stop_wake_ack()
        task = self._turn_task
        self._turn_task = None
        fish_id = self._fish_id
        answer_message_id = self._answer_message_id
        playback_owner = self._playback_owner
        self._finish_answer_progress("interrupted")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._tts_started:
            await self._current_tts().stop()
            self._stop_pet_audio(cancel=True)
            self._record_event(
                "tts_playback_stopped",
                source="tts",
                metadata={"messageId": str(answer_message_id)}
                if answer_message_id is not None
                else None,
            )
        if playback_owner is not None:
            await self._release_playback_owner(playback_owner)
        self._report_pet_voice(
            fish_id,
            answer_message_id,
            PetVoiceStatus.STOPPED,
        )
        self._tts_started = False
        self._draft_tts_waiting = False
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        self._answer_message_id = None
        self._answer_event_recorded = False
        self._answer_first_content_recorded = False
        self._answer_offset = 0
        self._answer_flush_offset = 0
        self._assistant_text = ""
        self._assistant_speech = ""
        self._speech_tags = False
        self._tts_text_buffer = TtsTextBuffer(first_chunk_at_pause=True)
        self._speech_tags = False
        self._assistant_speech = ""
        self._wake_ack_text = ""
        await self._retire_voice_tts()
        if self._pet_generation is not None:
            self._stop_pet_audio(cancel=True)
        self._turn_idle.set()
        await self._settle_answer_progress()

    def _start_answer_progress(self, message_sequence: int | None = None) -> None:
        if (
            self._fish_id is None
            or self._session_id is None
            or self._answer_message_id is None
        ):
            return

        def sample() -> tuple[str, int, int, str]:
            output = self._current_tts()
            consumed = max(0, output.played_offset) if self._tts_started else 0
            display = (
                getattr(output, "display_offset", consumed) if self._tts_started else 0
            )
            status = (
                "paused"
                if self._tts_paused_for_speech
                else "playing"
                if display
                else "buffering"
            )
            return self._assistant_text, display, consumed, status

        publish = getattr(self._pet_events, "playback_progress", lambda payload: None)
        self._answer_progress = AnswerPlaybackProgress(
            self._api,
            self._fish_id,
            self._session_id,
            self._answer_message_id,
            sample,
            publish,
            message_sequence=message_sequence,
        )

    def _finish_answer_progress(self, status: str, failure: str | None = None) -> None:
        progress = self._answer_progress
        if progress is None:
            return
        progress.finish(status, failure)
        self._answer_progress = None
        self._pending_answer_progress.append(progress)

    async def settle_playback_before_context(self) -> bool:
        """Gate draft context as well as final Ask on the prior frozen cutoff."""
        return await self._settle_answer_progress()

    async def _settle_answer_progress(self) -> bool:
        pending = tuple(self._pending_answer_progress)
        for progress in pending:
            if not await progress.settle(_ANSWER_PROGRESS_TIMEOUT_SECONDS):
                return False
            if progress in self._pending_answer_progress:
                self._pending_answer_progress.remove(progress)
        return True

    def _current_tts(self) -> TtsOutput:
        return self._agent_tts or self._turn_tts or self._tts

    async def _release_playback_owner(self, owner: PlaybackOwner) -> None:
        await self._playback.release(owner)
        if self._playback_owner == owner:
            self._playback_owner = None
        self._ensure_agent_drain()

    async def _promote_voice_tts(
        self,
        data: dict[str, object],
        content: str,
    ) -> bool | DraftTtsPending:
        draft_owner = self._voice_tts_owner
        draft_tts = self._draft_tts
        flush_offset = data.get("flushOffset")
        if (
            draft_owner is None
            or draft_tts is None
            or isinstance(flush_offset, bool)
            or not isinstance(flush_offset, int)
        ):
            await self._retire_voice_tts()
            return False
        key, revision = draft_owner.key, draft_owner.revision
        playback_owner = await self._playback.acquire(
            "answer",
            self._answer_message_id,
        )
        try:
            promotion = await draft_tts.promote(
                key,
                revision=revision,
                content=content,
                flush_offset=flush_offset,
                playback_owner=playback_owner,
                before_playback=self._start_pet_audio,
                **(
                    {"speech_content": self._assistant_speech}
                    if self._speech_tags
                    else {}
                ),
            )
        except asyncio.CancelledError:
            await self._retire_voice_tts()
            await self._release_playback_owner(playback_owner)
            raise
        except Exception:
            logger.debug("Draft TTS promotion missed", exc_info=True)
            record_voice_draft_event(
                self._diagnostics,
                "voice_draft_fallback",
                metadata={
                    "turnId": str(key.turn_id),
                    "revision": revision,
                    "reason": "provider_error",
                },
            )
            await self._retire_voice_tts()
            await self._release_playback_owner(playback_owner)
            return False
        if isinstance(promotion, DraftTtsPending):
            await self._release_playback_owner(playback_owner)
            return promotion
        if promotion is None:
            record_voice_draft_event(
                self._diagnostics,
                "voice_draft_fallback",
                metadata={
                    "turnId": str(key.turn_id),
                    "revision": revision,
                    "reason": "promotion_miss",
                },
            )
            self._voice_tts_owner = None
            await self._release_playback_owner(playback_owner)
            return False
        try:
            chunks = self._tts_text_buffer.adopt_sent_prefix(
                self._assistant_speech if self._speech_tags else content,
                promotion.submitted_offset,
                **(
                    {"speech_offset": getattr(promotion, "speech_offset", None)}
                    if self._speech_tags
                    else {}
                ),
            )
        except ValueError:
            record_voice_draft_event(
                self._diagnostics,
                "voice_draft_fallback",
                metadata={
                    "turnId": str(key.turn_id),
                    "revision": revision,
                    "reason": "range_mismatch",
                },
            )
            await self._retire_voice_tts()
            await self._release_playback_owner(playback_owner)
            return False
        record_voice_draft_event(
            self._diagnostics,
            "voice_draft_promoted",
            metadata={
                "turnId": str(key.turn_id),
                "revision": revision,
                "promoted": True,
            },
        )
        self._record_event(
            "tts_start_requested",
            source="session",
            metadata={"messageId": data.get("messageId")},
        )
        self._playback_owner = playback_owner
        turn_tts = self._playback.bind_tts(
            playback_owner,
            promotion.output,
        )
        turn_tts.activate_diagnostics(self._diagnostics)
        self._turn_tts = turn_tts
        self._playback.set_spoken_text(playback_owner, content)
        self._tts_started = True
        # The adoption hook already started the pet stream before replaying
        # private PCM. Restarting here would reset the first sentence's clock.
        self._tts_paused_for_speech = False
        self._tts_ducked_for_speech = False
        await self._append_answer_chunks(chunks, self._turn_tts)
        return True

    async def _fall_back_from_stalled_draft(self) -> None:
        self._record_event(
            "voice_draft_stall_fallback",
            source="session",
            metadata={"messageId": str(self._answer_message_id)},
        )
        self._draft_tts_waiting = False
        if self._tts_started:
            return
        await self._retire_voice_tts()
        if self._assistant_text:
            await self._start_answer_tts()

    async def _retire_voice_tts(self) -> None:
        self._draft_tts_waiting = False
        owner = self._voice_tts_owner
        self._voice_tts_owner = None
        self._turn_tts = None
        draft_tts = self._draft_tts
        if owner is None or draft_tts is None:
            return
        key, revision = owner.key, owner.revision
        try:
            await draft_tts.cancel(key, through_revision=revision)
        except Exception:
            logger.debug("Unable to retire draft TTS", exc_info=True)

    def _record_failure(
        self, fish_id: UUID, message_id: UUID | None, failure: str | None
    ) -> None:
        """Keep why this reply let the user down on the message itself."""
        record = getattr(self._api, "record_failure", None)
        if failure is None or message_id is None or record is None:
            return
        task = asyncio.create_task(record(fish_id, message_id, failure))
        _FAILURE_REPORTS.add(task)
        task.add_done_callback(_failure_reported)

    def _report_pet_voice(
        self,
        fish_id: UUID | None,
        answer_message_id: UUID | None,
        status: PetVoiceStatus,
    ) -> None:
        if (
            self._performance_reporter is None
            or fish_id is None
            or answer_message_id is None
        ):
            return
        try:
            self._performance_reporter.submit(fish_id, answer_message_id, status)
        except Exception:
            logger.exception(
                "Unable to queue pet voice performance: message_id=%s",
                answer_message_id,
            )

    async def _wait_for_interrupt_cleanup(self) -> None:
        task = self._interrupt_cleanup_task
        if task is None or task is asyncio.current_task():
            return
        await task

    def _voice_runtime_metadata(
        self,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        context: dict[str, object] = {"state": self._state.value}
        for name, value in (
            ("fishId", self._fish_id),
            ("sessionId", self._session_id),
            ("messageId", self._answer_message_id),
        ):
            if value is not None:
                context[name] = str(value)
        return {**context, **metadata}

    def _record_submit_turn_rejected(
        self,
        reason: _SubmitTurnRejectionReason,
        *,
        input_chars: int,
        turn_id: UUID | None,
    ) -> None:
        metadata: dict[str, object] = {"reason": reason, "inputChars": input_chars}
        if turn_id is not None:
            metadata["turnId"] = str(turn_id)
        self._record_event(
            "submit_turn_rejected",
            metadata=self._voice_runtime_metadata(metadata),
        )

    def _record_event(
        self,
        name: str,
        *,
        text: str | None = None,
        source: str = "session",
        metadata: dict[str, object] | None = None,
    ) -> None:
        diagnostics = self._diagnostics
        if diagnostics is None:
            return
        try:
            diagnostics.record_event(
                name,
                text=text,
                source=source,
                metadata=metadata,
            )
        except Exception:
            logger.debug("Unable to record companion diagnostic event", exc_info=True)


def _compact_text(value: str) -> str:
    return "".join(
        character.casefold()
        for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def _home_action_speech(data: dict[str, object]) -> str | None:
    action = data.get("action")
    if isinstance(action, dict) and action.get("operation") == "query":
        return None
    status = data.get("status")
    if status == "ambiguous":
        return "我还不能确定你指的是哪个设备。"
    if status == "failed":
        return "设备操作没有成功。"
    if status == "partial":
        return "部分设备已经操作，其他设备不支持这个操作。"
    if status != "succeeded" or not isinstance(action, dict):
        return None
    if action.get("operation") != "query":
        return None
    result = data.get("result")
    if not isinstance(result, dict):
        return "设备状态已经更新。"
    state = result.get("state")
    labels = {"on": "开启", "off": "关闭"}
    state_text = labels.get(state, state) if isinstance(state, str) else None
    return f"设备当前是{state_text}的。" if state_text else "设备状态已经更新。"


_FAILURE_REPORTS: set[asyncio.Task[None]] = set()


def _failure_reported(task: asyncio.Task[None]) -> None:
    _FAILURE_REPORTS.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Unable to record reply failure", exc_info=task.exception())
