"""Speaker admission integration, without recordings or model downloads."""

import asyncio
import threading
from collections.abc import Callable
from typing import cast

import pytest

from fish_memory.companion.audio_processing import ProcessedAudioFrame
from fish_memory.companion.owner_voice_gate import OwnerVoiceGate
from fish_memory.companion.playback import BoundTtsOutput, PlaybackArbiter
from fish_memory.companion.processor import CompanionAudioProcessor
from fish_memory.companion.session import CompanionSession, CompanionState
from fish_memory.companion.voice_drafts import (
    SpeculativeSilenceGate,
    VoiceDraftTurnController,
)

from .test_processor import (
    FISH_ID,
    BlockingOnlineAsr,
    FakeDraftDecoder,
    FakeDraftTransport,
    FakeDraftVad,
    FakeFrameProcessor,
    FakeInterruptionVad,
    FakeKws,
    FakeSmartTurn,
    FrameAsr,
    FrozenTurnSessionHarness,
    frozen_turn,
)
from .test_session import FakeApi, FakeTts

SILENCE = bytes(320)
OWNER = b"a" * 320
OTHER = b"b" * 320
RAW = b"r" * 320
CLEANED = b"x" * 320


class RecordingVerifier:
    def __init__(self, *scores: float) -> None:
        self.scores = iter(scores)
        self.accepted: list[bytes] = []

    def score_pcm(self, pcm: bytes) -> float:
        self.accepted.append(pcm)
        score = next(self.scores, None)
        assert score is not None, "Unexpected additional owner verification"
        return score


class BlockingVerifier(RecordingVerifier):
    def __init__(self) -> None:
        super().__init__(0.9)
        self.started = threading.Event()
        self.release = threading.Event()

    def score_pcm(self, pcm: bytes) -> float:
        self.started.set()
        assert self.release.wait(timeout=2)
        return super().score_pcm(pcm)


class OwnerSessionHarness(FrozenTurnSessionHarness):
    def __init__(self, state: CompanionState = CompanionState.LISTENING) -> None:
        super().__init__(frozen_turn(generation=1))
        self.state = state
        self.wake_calls = 0
        self.pause_calls = 0
        self.duck_calls = 0
        self.resume_calls = 0
        self.interrupt_calls = 0
        self.attenuated = False

    @property
    def output_attenuated_for_speech(self) -> bool:
        return self.attenuated

    async def on_wake_word(self) -> bool:
        self.wake_calls += 1
        self.state = CompanionState.LISTENING
        return True

    async def pause_output_for_speech(self) -> bool:
        self.pause_calls += 1
        self.attenuated = True
        return True

    async def attenuate_output_for_speech(self) -> bool:
        self.duck_calls += 1
        self.attenuated = True
        return True

    async def resume_output_after_speech(self) -> None:
        if self.attenuated:
            self.resume_calls += 1
        self.attenuated = False

    async def interrupt_from_speech(self) -> None:
        self.interrupt_calls += 1
        self.attenuated = False
        await super().interrupt_from_speech()

    def change_binding(self) -> None:
        self.binding_generation += 1
        self.current_frozen_turn = frozen_turn(
            generation=self.binding_generation,
            session_value=202,
        )
        self.session_id = self.current_frozen_turn.session_id


class CleanedFrameProcessor(FakeFrameProcessor):
    def process(self, *, capture_pcm: bytes, render_pcm: bytes) -> ProcessedAudioFrame:
        del render_pcm
        self.processed_captures.append(capture_pcm)
        speech = next(self._voice_values)
        return ProcessedAudioFrame(CLEANED if speech else SILENCE, speech)


async def feed(processor: CompanionAudioProcessor, pcm: bytes, frames: int) -> None:
    for _ in range(frames):
        await processor.process_frame(pcm, SILENCE)


async def wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    pytest.fail("Timed out waiting for background work")


def make_processor(
    session: OwnerSessionHarness,
    verifier: RecordingVerifier,
    voices: list[bool],
    *,
    asr: FrameAsr | None = None,
    kws: FakeKws | None = None,
    smart_turn: FakeSmartTurn | None = None,
    frame_processor: FakeFrameProcessor | None = None,
    draft_controller: VoiceDraftTurnController | None = None,
) -> CompanionAudioProcessor:
    return CompanionAudioProcessor(
        cast(CompanionSession, session),
        kws if kws is not None else FakeKws(),
        asr if asr is not None else FrameAsr("主人的问题"),
        frame_processor=(
            frame_processor
            if frame_processor is not None
            else FakeFrameProcessor(voices)
        ),
        owner_voice_gate=OwnerVoiceGate(verifier, inference_timeout=1),
        interruption_vad=FakeInterruptionVad([True] * 300),
        defer_interruption_output_until_near_end=True,
        smart_turn=smart_turn,
        smart_turn_reopen_seconds=0.05,
        voice_draft_vad=(
            FakeDraftVad([True] + [False] * 300)
            if draft_controller is not None
            else None
        ),
        voice_draft_gate=(
            SpeculativeSilenceGate(silence_frames=1)
            if draft_controller is not None
            else None
        ),
        voice_draft_controller=draft_controller,
        speech_start_frames=1,
        speech_pre_roll_frames=1,
        speech_end_frames=1,
    )


@pytest.mark.parametrize(
    "state",
    [CompanionState.STANDBY, CompanionState.LISTENING, CompanionState.SPEAKING],
)
async def test_non_owner_cannot_wake_submit_or_pause(state: CompanionState) -> None:
    session = OwnerSessionHarness(state)
    verifier = RecordingVerifier(0.1)
    kws = FakeKws()
    asr = FrameAsr("不应提交")
    frame_processor = FakeFrameProcessor([True] * 80 + [False])
    processor = make_processor(
        session, verifier, [], asr=asr, kws=kws, frame_processor=frame_processor
    )
    try:
        await feed(processor, OTHER, 80)
        await feed(processor, SILENCE, 1)

        assert verifier.accepted == [OTHER * 80]
        assert len(frame_processor.processed_captures) == 81
        assert not any(any(pcm) for pcm in kws.accepted)
        assert asr.accepted == []
        assert session.wake_calls == session.interrupt_calls == 0
        assert session.pause_calls == session.duck_calls == session.resume_calls == 0
        assert session.submissions == []
    finally:
        await processor.close()


async def test_owner_block_can_wake_and_submit_after_verified_end() -> None:
    session = OwnerSessionHarness(CompanionState.STANDBY)
    kws = FakeKws()
    kws.detect = True
    asr = FrameAsr("主人的问题")
    verifier = RecordingVerifier(0.9)
    processor = make_processor(
        session, verifier, [True] * 80 + [False], asr=asr, kws=kws
    )
    try:
        await feed(processor, OWNER, 79)
        assert kws.accepted == []
        assert session.wake_calls == 0

        await feed(processor, OWNER, 1)
        assert session.wake_calls == 1
        assert kws.accepted == [OWNER]
        assert asr.accepted == [OWNER] * 79
        assert session.submissions == []

        await feed(processor, SILENCE, 1)
        assert [text for text, _ in session.submissions] == ["主人的问题"]
    finally:
        await processor.close()


async def test_owner_stops_ta_once_verified_words_are_heard() -> None:
    session = OwnerSessionHarness(CompanionState.SPEAKING)
    asr = FrameAsr("主人的问题", partial_text="请听我说", snapshot_text="请听我说")
    processor = make_processor(
        session, RecordingVerifier(0.9), [True] * 80 + [False], asr=asr
    )
    try:
        await feed(processor, OWNER, 80)
        assert session.pause_calls == 1
        assert session.interrupt_calls == 1
        assert session.submissions == []

        await feed(processor, SILENCE, 1)
        assert session.interrupt_calls == 1
        assert [text for text, _ in session.submissions] == ["主人的问题"]
    finally:
        await processor.close()


@pytest.mark.parametrize("tail_frames", [20, 80])
async def test_rejected_later_block_after_the_stop_submits_nothing(
    tail_frames: int,
) -> None:
    session = OwnerSessionHarness(CompanionState.SPEAKING)
    asr = FrameAsr("不可提交", partial_text="请听我说")
    verifier = RecordingVerifier(0.9, 0.1)
    processor = make_processor(
        session,
        verifier,
        [True] * (80 + tail_frames) + [False],
        asr=asr,
    )
    try:
        await feed(processor, OWNER, 80)
        assert session.interrupt_calls == 1

        await feed(processor, OTHER, tail_frames)
        await feed(processor, SILENCE, 1)
        assert len(verifier.accepted) == 2
        assert asr.accepted == [OWNER] * 80
        assert asr.reset_calls == 1
        assert session.interrupt_calls == 1
        assert session.submissions == []
    finally:
        await processor.close()


@pytest.mark.parametrize("tail_frames", [20, 80])
async def test_rejected_later_block_restores_tts_before_any_words(
    tail_frames: int,
) -> None:
    session = OwnerSessionHarness(CompanionState.SPEAKING)
    asr = FrameAsr("不可提交", partial_text=None)
    verifier = RecordingVerifier(0.9, 0.1)
    processor = make_processor(
        session,
        verifier,
        [True] * (80 + tail_frames) + [False],
        asr=asr,
    )
    try:
        await feed(processor, OWNER, 80)
        assert session.pause_calls == 1
        assert session.interrupt_calls == 0

        await feed(processor, OTHER, tail_frames)
        await feed(processor, SILENCE, 1)
        assert len(verifier.accepted) == 2
        assert asr.accepted == [OWNER] * 80
        assert session.resume_calls == 1
        assert not session.attenuated
        assert session.interrupt_calls == 0
        assert session.submissions == []
    finally:
        await processor.close()


async def test_asr_receives_only_the_cleaned_pcm_that_was_verified() -> None:
    session = OwnerSessionHarness()
    verifier = RecordingVerifier(0.9)
    asr = FrameAsr("清理后的问题")
    frame_processor = CleanedFrameProcessor([True] * 80 + [False])
    processor = make_processor(
        session, verifier, [], asr=asr, frame_processor=frame_processor
    )
    try:
        await feed(processor, RAW, 80)
        await feed(processor, SILENCE, 1)

        assert verifier.accepted == [CLEANED * 80]
        assert asr.accepted == [CLEANED] * 80
        assert frame_processor.processed_captures[:80] == [RAW] * 80
        assert [text for text, _ in session.submissions] == ["清理后的问题"]
    finally:
        await processor.close()


async def test_legacy_pcm_still_uses_owner_admission_and_frame_processing() -> None:
    session = OwnerSessionHarness(CompanionState.STANDBY)
    verifier = RecordingVerifier(0.1)
    kws = FakeKws()
    kws.detect = True
    frame_processor = FakeFrameProcessor([True] * 80)
    processor = make_processor(
        session, verifier, [], kws=kws, frame_processor=frame_processor
    )
    try:
        await processor.process_pcm(OTHER * 80)
        assert verifier.accepted == [OTHER * 80]
        assert frame_processor.processed_captures == [OTHER] * 80
        assert kws.accepted == []
        assert session.wake_calls == 0
        assert session.submissions == []
    finally:
        await processor.close()


@pytest.mark.parametrize("pcm", [b"", b"x", bytes(321)])
async def test_legacy_pcm_rejects_incomplete_frames(pcm: bytes) -> None:
    processor = make_processor(OwnerSessionHarness(), RecordingVerifier(), [])
    try:
        with pytest.raises(ValueError, match="complete 10 ms PCM frames"):
            await processor.process_pcm(pcm)
    finally:
        await processor.close()


@pytest.mark.parametrize("invalidate", ["close", "binding"])
async def test_late_owner_result_has_no_effect_after_invalidation(
    invalidate: str,
) -> None:
    session = OwnerSessionHarness(CompanionState.STANDBY)
    verifier = BlockingVerifier()
    kws = FakeKws()
    kws.detect = True
    asr = FrameAsr("不应提交")
    processor = make_processor(session, verifier, [True] * 80, asr=asr, kws=kws)
    task: asyncio.Task[None] | None = None
    try:
        await feed(processor, OWNER, 79)
        task = asyncio.create_task(processor.process_frame(OWNER, SILENCE))
        await wait_until(verifier.started.is_set)
        if invalidate == "close":
            await processor.close()
        else:
            session.change_binding()
        verifier.release.set()
        await asyncio.wait_for(task, timeout=1)

        assert kws.accepted == []
        assert asr.accepted == []
        assert session.wake_calls == session.pause_calls == session.interrupt_calls == 0
        assert session.submissions == []
        assert processor._owner_voice_gate is not None
        assert processor._owner_voice_gate.buffered_frames == 0
    finally:
        verifier.release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await processor.close()


@pytest.mark.parametrize("invalidate", ["close", "binding"])
async def test_late_final_decode_cannot_interrupt_or_submit_after_invalidation(
    invalidate: str,
) -> None:
    session = OwnerSessionHarness(CompanionState.SPEAKING)
    asr = BlockingOnlineAsr()
    processor = make_processor(
        session, RecordingVerifier(0.9), [True] * 80 + [False], asr=asr
    )
    task: asyncio.Task[None] | None = None
    try:
        await feed(processor, OWNER, 80)
        assert session.pause_calls == 1
        task = asyncio.create_task(processor.process_frame(SILENCE, SILENCE))
        await wait_until(asr.finish_started.is_set)
        if invalidate == "close":
            await processor.close()
        else:
            session.change_binding()
        asr.release_finish.set()
        await asyncio.wait_for(task, timeout=1)

        assert session.interrupt_calls == 0
        assert session.submissions == []
    finally:
        asr.release_finish.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await processor.close()


async def test_close_during_reversible_pause_cannot_leave_output_paused() -> None:
    class BlockingPauseSession(OwnerSessionHarness):
        def __init__(self) -> None:
            super().__init__(CompanionState.SPEAKING)
            self.pause_started = asyncio.Event()
            self.release_pause = asyncio.Event()

        async def pause_output_for_speech(self) -> bool:
            self.pause_started.set()
            await self.release_pause.wait()
            return await super().pause_output_for_speech()

    session = BlockingPauseSession()
    processor = make_processor(session, RecordingVerifier(0.9), [True] * 80)
    task: asyncio.Task[None] | None = None
    try:
        await feed(processor, OWNER, 79)
        task = asyncio.create_task(processor.process_frame(OWNER, SILENCE))
        await asyncio.wait_for(session.pause_started.wait(), timeout=1)
        await processor.close()
        session.release_pause.set()
        await asyncio.wait_for(task, timeout=1)

        assert not session.attenuated
        assert session.resume_calls == 1
        assert session.interrupt_calls == 0
        assert session.submissions == []
    finally:
        session.release_pause.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await processor.close()


async def test_pending_audio_cannot_cross_a_binding_change_between_frames() -> None:
    session = OwnerSessionHarness(CompanionState.STANDBY)
    verifier = RecordingVerifier(0.9)
    kws = FakeKws()
    kws.detect = True
    processor = make_processor(session, verifier, [True] * 80, kws=kws)
    try:
        await feed(processor, OWNER, 79)
        session.change_binding()
        await feed(processor, OTHER, 1)

        assert verifier.accepted == []
        assert kws.accepted == []
        assert session.wake_calls == 0
        assert processor._owner_voice_gate is not None
        assert processor._owner_voice_gate.buffered_frames <= 1
    finally:
        await processor.close()


async def test_smart_turn_binding_change_without_another_frame_drops_old_audio() -> (
    None
):
    session = OwnerSessionHarness()
    asr = FrameAsr("旧会话的问题", partial_text=None)
    smart_turn = FakeSmartTurn(False)
    processor = make_processor(
        session,
        RecordingVerifier(0.9),
        [True] * 80 + [False],
        asr=asr,
        smart_turn=smart_turn,
    )
    try:
        await feed(processor, OWNER, 80)
        await feed(processor, SILENCE, 1)
        await wait_until(lambda: len(smart_turn.accepted) == 1)
        session.change_binding()
        finish = processor._frame_finish_task
        assert finish is not None
        await asyncio.wait_for(asyncio.shield(finish), timeout=1)

        assert session.submissions == []
        assert session.interrupt_calls == 0
    finally:
        await processor.close()


async def test_buffered_resumed_speech_holds_smart_turn_until_admission() -> None:
    session = OwnerSessionHarness()
    asr = FrameAsr("合并后的完整问题", partial_text=None)
    smart_turn = FakeSmartTurn(False, True)
    processor = make_processor(
        session,
        RecordingVerifier(0.9, 0.9, 0.9),
        [True] * 80 + [False] + [True] * 80 + [False],
        asr=asr,
        smart_turn=smart_turn,
    )
    try:
        await feed(processor, OWNER, 80)
        await feed(processor, SILENCE, 1)
        await wait_until(lambda: len(smart_turn.accepted) == 1)

        await feed(processor, OTHER, 1)
        await asyncio.sleep(0.08)
        assert asr.finish_calls == 0
        assert session.submissions == []
        assert asr.accepted == [OWNER] * 80

        await feed(processor, OTHER, 79)
        await feed(processor, SILENCE, 1)
        await wait_until(lambda: len(session.submissions) == 1)
        assert asr.finish_calls == 1
        assert smart_turn.accepted[-1] == OWNER * 80 + OTHER * 80
        assert [text for text, _ in session.submissions] == ["合并后的完整问题"]
    finally:
        await processor.close()


async def test_rejected_reopen_keeps_the_verified_previous_owner_turn() -> None:
    session = OwnerSessionHarness(CompanionState.SPEAKING)
    asr = FrameAsr("主人的上一段", partial_text=None)
    smart_turn = FakeSmartTurn(False)
    verifier = RecordingVerifier(0.9, 0.1)
    processor = make_processor(
        session,
        verifier,
        [True] * 80 + [False] + [True] * 30 + [False],
        asr=asr,
        smart_turn=smart_turn,
    )
    try:
        await feed(processor, OWNER, 80)
        await feed(processor, SILENCE, 1)
        await wait_until(lambda: len(smart_turn.accepted) == 1)
        await feed(processor, OTHER, 30)
        await feed(processor, SILENCE, 1)
        await wait_until(lambda: len(session.submissions) == 1)

        # The prior segment already passed its END verification. Rejecting a
        # different, unadmitted speaker must not erase that valid question.
        assert [text for text, _ in session.submissions] == ["主人的上一段"]
        assert verifier.accepted == [OWNER * 80, OTHER * 30]
        assert asr.accepted == [OWNER] * 80
        assert asr.finish_calls == 1
        assert session.interrupt_calls == 1
        assert session.resume_calls == 0
    finally:
        await processor.close()


async def test_each_block_is_checked_with_this_utterance_checked_so_far() -> None:
    session = OwnerSessionHarness()
    verifier = RecordingVerifier(0.9, 0.9, 0.9)
    asr = FrameAsr("含有短尾巴的问题", partial_text=None)
    processor = make_processor(session, verifier, [True] * 180 + [False], asr=asr)
    try:
        await feed(processor, OWNER, 160)
        await feed(processor, OTHER, 20)
        await feed(processor, SILENCE, 1)

        assert verifier.accepted[:2] == [OWNER * 80, OWNER * 160]
        assert verifier.accepted[2] == OWNER * 160 + OTHER * 20 + SILENCE
        assert asr.accepted == [OWNER] * 160 + [OTHER] * 20
        assert len(session.submissions) == 1
    finally:
        await processor.close()


async def test_unmatched_speech_never_creates_a_voice_draft() -> None:
    session = OwnerSessionHarness()
    transport = FakeDraftTransport()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=FakeDraftDecoder(),
    )
    processor = make_processor(
        session,
        RecordingVerifier(0.1),
        [True] * 80 + [False],
        draft_controller=controller,
    )
    try:
        await feed(processor, OTHER, 80)
        await feed(processor, SILENCE, 1)
        await controller.wait_idle()

        assert processor._voice_draft_owner is None
        assert transport.upserts == []
        assert transport.cancels == []
        assert session.submissions == []
    finally:
        await processor.close()
        await controller.close()


async def test_rejected_block_abandons_already_started_verified_voice_draft() -> None:
    session = OwnerSessionHarness(CompanionState.SPEAKING)
    transport = FakeDraftTransport()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=FakeDraftDecoder(),
    )
    processor = make_processor(
        session,
        RecordingVerifier(0.9, 0.1),
        [True] * 160,
        draft_controller=controller,
    )
    try:
        await feed(processor, OWNER, 80)
        await controller.wait_idle()
        assert len(transport.upserts) == 1
        assert processor._voice_draft_owner is not None

        await feed(processor, OTHER, 80)
        await controller.wait_idle()
        assert processor._voice_draft_owner is None
        assert len(transport.upserts) == 1
        assert len(transport.cancels) == 1
        assert transport.cancels[0][2] == "rejected"
        assert session.interrupt_calls == 1
        assert session.submissions == []
    finally:
        await processor.close()
        await controller.close()


@pytest.mark.parametrize("operation", ["pause", "duck"])
@pytest.mark.parametrize("invalidate", ["binding", "playback_owner"])
async def test_real_session_late_output_control_cannot_set_new_owner_flags(
    operation: str,
    invalidate: str,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class DelayedCompletionOutput(BoundTtsOutput):
        async def pause(self) -> None:
            await super().pause()
            started.set()
            await release.wait()

        async def duck(self) -> None:
            await super().duck()
            started.set()
            await release.wait()

    playback = PlaybackArbiter()
    original_output = FakeTts()
    original_owner = await playback.acquire("answer", None)
    playback.bind_tts(
        original_owner,
        DelayedCompletionOutput(playback, original_owner, original_output),
    )
    session = CompanionSession(FakeApi(), original_output, playback_arbiter=playback)
    session._tts_started = True
    next_output = FakeTts()
    control = (
        session.pause_output_for_speech
        if operation == "pause"
        else session.attenuate_output_for_speech
    )
    task = asyncio.create_task(control())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        if invalidate == "binding":
            await session.set_binding(FISH_ID)
            # This isolates the binding fence: no session-owned turn existed
            # to release the separately acquired playback owner.
            assert playback.current_owner == original_owner
        else:
            assert await playback.release(original_owner)
            next_owner = await playback.acquire("answer", None)
            playback.bind_tts(next_owner, next_output)
            assert session.binding_generation == 0
        release.set()

        assert not await asyncio.wait_for(task, timeout=1)
        assert not session.output_attenuated_for_speech
        assert not session._tts_paused_for_speech
        assert not session._tts_ducked_for_speech
        await session.resume_output_after_speech()
        assert next_output.calls == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await session.close()
        await playback.close()
