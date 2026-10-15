import asyncio
import threading
from collections.abc import Callable
from typing import Literal, cast
from uuid import UUID

import pytest

from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.audio_processing import ProcessedAudioFrame
from fish_memory.companion.client import CompanionAskResult, CompanionReplyContext
from fish_memory.companion.playback import PlaybackArbiter, PlaybackOwner
from fish_memory.companion.processor import AsrInput, CompanionAudioProcessor
from fish_memory.companion.session import (
    CompanionSession,
    CompanionState,
    FrozenVoiceTurn,
    TtsOutput,
)
from fish_memory.companion.voice_drafts import (
    SpeculativeSilenceGate,
    VoiceDraftDecoder,
    VoiceDraftTurnController,
)
from fish_memory.pet.publisher import NoopPetEventSink

from .test_session import (
    AgentFakeApi,
    BlockingAgentTts,
    FakeApi,
    FakeTts,
    RecordingDiagnostics,
    agent_message,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
ANSWER_ID = UUID("00000000-0000-0000-0000-000000000101")


async def wait_for_condition(condition: Callable[[], bool]) -> None:
    """Wait for asynchronous work without assuming a sub-16 ms Windows timer."""
    async with asyncio.timeout(1):
        while not condition():  # noqa: ASYNC110 - thread-owned counters have no Event
            await asyncio.sleep(0.001)


class FakeKws:
    def __init__(self) -> None:
        self.detect = False
        self.reset_calls = 0
        self.accepted: list[bytes] = []

    def accept_pcm(self, pcm: bytes) -> list[object]:
        self.accepted.append(pcm)
        if self.detect:
            self.detect = False
            return [object()]
        return []

    def reset(self) -> None:
        self.reset_calls += 1


class FakeAsr:
    def __init__(self) -> None:
        self.update: AsrUpdate | None = None

    def accept_pcm(self, pcm: bytes) -> AsrUpdate | None:
        del pcm
        update, self.update = self.update, None
        return update

    def reset(self) -> None:
        return None


class FrameAsr:
    def __init__(
        self,
        finished_text: str,
        partial_text: str | None = "partial",
        snapshot_text: str | None = None,
    ) -> None:
        self.accepted: list[bytes] = []
        self.finished_text = finished_text
        self.partial_text = partial_text
        self.snapshot_text = snapshot_text
        self.finish_calls = 0
        self.reset_calls = 0
        self.snapshot_calls = 0

    def accept_pcm(self, pcm: bytes) -> AsrUpdate | None:
        self.accepted.append(pcm)
        if self.partial_text is None:
            return None
        return AsrUpdate(self.partial_text, is_final=False)

    def finish_utterance(self) -> str:
        self.finish_calls += 1
        return self.finished_text

    def snapshot_utterance(self) -> bytes:
        return b"".join(self.accepted)

    def recognize_snapshot(self, _pcm: bytes) -> str | None:
        self.snapshot_calls += 1
        return self.snapshot_text

    def reset(self) -> None:
        self.reset_calls += 1


class EventFrameAsr(FrameAsr):
    def __init__(
        self,
        finished_text: str,
        *,
        event: str,
        snapshot_text: str | None = None,
        snapshot_event: str | None = None,
    ) -> None:
        super().__init__(
            finished_text,
            partial_text=None,
            snapshot_text=snapshot_text,
        )
        self.event = event
        self.snapshot_event = snapshot_event if snapshot_event is not None else event

    def finish_utterance_update(self) -> AsrUpdate:
        self.finish_calls += 1
        return AsrUpdate(self.finished_text, is_final=True, event=self.event)

    def recognize_snapshot_update(self, _pcm: bytes) -> AsrUpdate:
        self.snapshot_calls += 1
        return AsrUpdate(
            self.snapshot_text or "",
            is_final=True,
            event=self.snapshot_event,
        )


class SequenceFrameAsr(FrameAsr):
    def __init__(self, updates: list[AsrUpdate]) -> None:
        super().__init__(finished_text="停止对话", partial_text=None)
        self._updates = iter(updates)

    def accept_pcm(self, pcm: bytes) -> AsrUpdate | None:
        self.accepted.append(pcm)
        return next(self._updates, None)


class StablePartialAsr(FrameAsr):
    def __init__(self, *, final_text: str) -> None:
        super().__init__(finished_text=final_text, partial_text=None)
        self._updates = iter(
            [
                AsrUpdate("停止对话", is_final=False),
                AsrUpdate("停止对话", is_final=False),
            ]
        )

    def accept_pcm(self, pcm: bytes) -> AsrUpdate | None:
        self.accepted.append(pcm)
        return next(self._updates, None)


class FakeFrameProcessor:
    def __init__(self, voice_values: list[bool]) -> None:
        self._voice_values = iter(voice_values)
        self.reset_vad_calls = 0
        self.stream_delays: list[int] = []
        self.processed_captures: list[bytes] = []

    def process(self, *, capture_pcm: bytes, render_pcm: bytes) -> ProcessedAudioFrame:
        del render_pcm
        self.processed_captures.append(capture_pcm)
        return ProcessedAudioFrame(
            cleaned_pcm=capture_pcm,
            voice_detected=next(self._voice_values),
        )

    def reset_vad(self) -> None:
        self.reset_vad_calls += 1

    def set_stream_delay(self, delay_ms: int) -> None:
        self.stream_delays.append(delay_ms)


class CleaningFrameProcessor(FakeFrameProcessor):
    def process(self, *, capture_pcm: bytes, render_pcm: bytes) -> ProcessedAudioFrame:
        del capture_pcm, render_pcm
        return ProcessedAudioFrame(
            cleaned_pcm=b"x" * 320,
            voice_detected=next(self._voice_values),
        )


class FakeInterruptionVad:
    def __init__(self, voice_values: list[bool]) -> None:
        self._voice_values = iter(voice_values)
        self.accepted: list[bytes] = []
        self.reset_calls = 0

    def is_speech(self, pcm: bytes) -> bool:
        self.accepted.append(pcm)
        return next(self._voice_values, False)

    def reset(self) -> None:
        self.reset_calls += 1


class FakeDraftVad:
    def __init__(self, voice_values: list[bool]) -> None:
        self._voice_values = iter(voice_values)
        self.accepted: list[bytes] = []

    def is_speech(self, pcm: bytes) -> bool:
        self.accepted.append(pcm)
        return next(self._voice_values, False)


class FakeDraftDecoder:
    def recognize(self, pcm: bytes) -> str:
        assert type(pcm) is bytes
        return "候选问题"


class FakeDraftTransport:
    def __init__(self) -> None:
        self.upserts: list[tuple[object, int, str]] = []
        self.cancels: list[tuple[object, int, str]] = []

    async def upsert_voice_draft(
        self,
        key: object,
        *,
        revision: int,
        content: str,
        **_metadata: object,
    ) -> bool:
        self.upserts.append((key, revision, content))
        return True

    async def cancel_voice_draft(
        self,
        key: object,
        *,
        revision: int,
        reason: str,
    ) -> bool:
        self.cancels.append((key, revision, reason))
        return True


class FakeSmartTurn:
    def __init__(self, *results: bool | Exception) -> None:
        self._results = iter(results)
        self.accepted: list[bytes] = []

    def predict_pcm16(self, pcm: bytes) -> object:
        self.accepted.append(pcm)
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return type(
            "SmartTurnResult",
            (),
            {
                "complete": result,
                "probability": 0.9 if result else 0.1,
                "inference_ms": 1.0,
            },
        )()


class BlockingSnapshotAsr(FrameAsr):
    def __init__(self) -> None:
        super().__init__("", partial_text=None)
        self.decode_started = threading.Event()
        self.release_first_decode = threading.Event()
        self._decode_calls = 0

    def recognize_snapshot(self, pcm: bytes) -> str | None:
        self._decode_calls += 1
        if self._decode_calls == 1:
            self.decode_started.set()
            assert self.release_first_decode.wait(timeout=1)
        return "合并问题" if len(pcm) >= 640 else "前半句"

    def consume_snapshot(self, pcm: bytes) -> bool:
        joined = b"".join(self.accepted)
        if not joined.startswith(pcm):
            return False
        consumed_frames = len(pcm) // 320
        del self.accepted[:consumed_frames]
        self.finish_calls += 1
        return True


@pytest.mark.asyncio
async def test_standby_skips_unused_aec_and_vad_when_render_is_silent() -> None:
    session = CompanionSession(FakeApi(), FakeTts())
    await session.set_binding(FISH_ID)
    kws = FakeKws()
    frame_processor = FakeFrameProcessor([])
    processor = CompanionAudioProcessor(
        session,
        kws,
        FakeAsr(),
        frame_processor=frame_processor,
    )
    capture = b"a" * 320

    await processor.process_frame(capture, b"\x00" * 320)

    assert kws.accepted == [capture]
    assert frame_processor.processed_captures == []


@pytest.mark.asyncio
async def test_standby_keeps_full_processing_when_render_audio_is_present() -> None:
    session = CompanionSession(FakeApi(), FakeTts())
    await session.set_binding(FISH_ID)
    kws = FakeKws()
    frame_processor = FakeFrameProcessor([False])
    processor = CompanionAudioProcessor(
        session,
        kws,
        FakeAsr(),
        frame_processor=frame_processor,
    )
    capture = b"a" * 320

    await processor.process_frame(capture, b"r" * 320)

    assert kws.accepted == [capture]
    assert frame_processor.processed_captures == [capture]


class BlockingSmartTurn:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def predict_pcm16(self, _pcm: bytes) -> object:
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=1)
        return type(
            "SmartTurnResult",
            (),
            {"complete": True, "probability": 0.9, "inference_ms": 1.0},
        )()


class BlockingSubmitSession(CompanionSession):
    def __init__(self, api: FakeApi, tts: FakeTts) -> None:
        super().__init__(api, tts)
        self.submit_started = asyncio.Event()
        self.release_submit = asyncio.Event()

    async def submit_transcript(
        self,
        text: str,
        *,
        frozen_turn: FrozenVoiceTurn | None = None,
    ) -> bool:
        self.submit_started.set()
        await self.release_submit.wait()
        return await super().submit_transcript(text, frozen_turn=frozen_turn)


class FrozenTurnSessionHarness:
    def __init__(self, frozen_turn: FrozenVoiceTurn) -> None:
        self.state = CompanionState.LISTENING
        self.fish_id = frozen_turn.fish_id
        self.session_id = frozen_turn.session_id
        self.voice_conversation_id = frozen_turn.voice_conversation_id
        self.binding_generation = frozen_turn.binding_generation
        self.current_frozen_turn = frozen_turn
        self.freeze_calls: list[FrozenVoiceTurn] = []
        self.submissions: list[tuple[str, FrozenVoiceTurn | None]] = []
        self.interrupt_started = asyncio.Event()
        self.release_interrupt = asyncio.Event()
        self.block_interrupt = False
        self.reservations = 0
        self.releases = 0
        self.frozen_releases: list[FrozenVoiceTurn] = []
        self.submit_error: Exception | None = None

    @property
    def output_attenuated_for_speech(self) -> bool:
        return False

    def freeze_turn_binding(self) -> FrozenVoiceTurn:
        self.freeze_calls.append(self.current_frozen_turn)
        return self.current_frozen_turn

    def reserve_foreground_speech(self) -> object:
        self.reservations += 1
        return object()

    def release_foreground_speech(self, _reservation: object = None) -> bool:
        self.releases += 1
        return True

    def is_likely_echo(self, _text: str) -> bool:
        return False

    async def interrupt_from_speech(self) -> None:
        self.interrupt_started.set()
        if self.block_interrupt:
            await self.release_interrupt.wait()
        self.state = CompanionState.LISTENING

    async def submit_transcript(
        self,
        text: str,
        *,
        frozen_turn: FrozenVoiceTurn | None = None,
    ) -> bool:
        self.submissions.append((text, frozen_turn))
        if self.submit_error is not None:
            raise self.submit_error
        return True

    def release_frozen_turn(self, value: FrozenVoiceTurn | None) -> None:
        if value is not None:
            self.frozen_releases.append(value)

    def refresh_pending_agent_reply(
        self,
        value: FrozenVoiceTurn | None,
    ) -> FrozenVoiceTurn | None:
        return value

    async def submit_voice_turn(self, *_: object, **__: object) -> bool:
        return True

    async def resume_output_after_speech(self) -> None:
        return None

    async def attenuate_output_for_speech(self) -> bool:
        return False

    async def pause_output_for_speech(self) -> bool:
        return False


def frozen_turn(*, generation: int, session_value: int = 201) -> FrozenVoiceTurn:
    return FrozenVoiceTurn(
        fish_id=FISH_ID,
        session_id=UUID(int=session_value),
        voice_conversation_id=UUID(int=301),
        binding_generation=generation,
        reply_context=None,
        context_reservation_id=None,
        reply_blocked=False,
        agent_reply_finalize=None,
    )


class ClaimingSnapshotAsr(FrameAsr):
    def __init__(self) -> None:
        super().__init__("", partial_text=None)

    def recognize_snapshot(self, _pcm: bytes) -> str | None:
        return "前半句"

    def consume_snapshot(self, pcm: bytes) -> bool:
        joined = b"".join(self.accepted)
        if not joined.startswith(pcm):
            return False
        del self.accepted[: len(pcm) // 320]
        self.finish_calls += 1
        return True


class BlockingOnlineAsr(FrameAsr):
    def __init__(self) -> None:
        super().__init__("", partial_text=None)
        self.finish_started = threading.Event()
        self.release_finish = threading.Event()

    def recognize_snapshot(self, _pcm: bytes) -> str | None:
        return None

    def finish_utterance(self) -> str:
        self.finish_started.set()
        assert self.release_finish.wait(timeout=1)
        self.accepted.clear()
        self.finish_calls += 1
        return "在线上一段"


@pytest.mark.asyncio
async def test_processor_switches_from_wake_word_to_transcript_submission() -> None:
    api = FakeApi()
    session = CompanionSession(api, cast(TtsOutput, FakeTts()))
    await session.set_binding(FISH_ID)
    kws = FakeKws()
    asr = FakeAsr()
    processor = CompanionAudioProcessor(
        session,
        kws,
        asr,
        utterance_commit_delay_seconds=0,
    )

    kws.detect = True
    await processor.process_pcm(b"wake")
    assert session.state in {
        CompanionState.ACKNOWLEDGING,
        CompanionState.LISTENING,
    }
    assert kws.reset_calls == 1

    asr.update = AsrUpdate("你好", is_final=True)
    await processor.process_pcm(b"speech")
    await asyncio.sleep(0)

    assert api.ask_calls == [(FISH_ID, "你好")]


@pytest.mark.asyncio
async def test_wake_notifies_voice_processing_before_session_transition() -> None:
    events: list[str] = []

    class Session:
        state = CompanionState.STANDBY

        async def on_wake_word(self) -> bool:
            events.append("wake")
            return False

    kws = FakeKws()
    kws.detect = True
    processor = CompanionAudioProcessor(
        cast(CompanionSession, Session()),
        kws,
        FakeAsr(),
        frame_processor=FakeFrameProcessor([]),
        notify_voice_activity=lambda: events.append("notify"),
    )

    await processor.process_frame(b"w" * 320, bytes(320))

    assert events == ["notify", "wake"]


@pytest.mark.asyncio
async def test_speech_start_notifies_once_without_repeating_on_active_frames() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    notifications: list[None] = []
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("", partial_text=None),
        frame_processor=FakeFrameProcessor([True, True, True]),
        speech_start_frames=1,
        speech_end_frames=1,
        notify_voice_activity=lambda: notifications.append(None),
    )

    await processor.process_frame(b"a" * 320, bytes(320))
    await processor.process_frame(b"b" * 320, bytes(320))
    await processor.process_frame(b"c" * 320, bytes(320))

    assert notifications == [None]
    await processor.close()


@pytest.mark.asyncio
async def test_legacy_processor_rejects_punctuation_only_final() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FakeAsr()
    asr.update = AsrUpdate("。", is_final=True)
    processor = CompanionAudioProcessor(session, FakeKws(), asr)

    await processor.process_pcm(b"noise")
    await asyncio.sleep(0)

    assert api.ask_calls == []


@pytest.mark.asyncio
async def test_legacy_processor_rejects_explicit_sensevoice_cough_event() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FakeAsr()
    asr.update = AsrUpdate("呵呵。", is_final=True, event="<|Cough|>")
    processor = CompanionAudioProcessor(session, FakeKws(), asr)

    await processor.process_pcm(b"cough")
    await asyncio.sleep(0)

    assert api.ask_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transcript",
    ["嗯。", "哼。", "嗯嗯。", "嗯哼。", "嗯哼哼嗯。", "咳咳。"],
)
async def test_legacy_processor_rejects_standalone_nonlexical_vocalization(
    transcript: str,
) -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FakeAsr()
    asr.update = AsrUpdate(transcript, is_final=True, event="<|Speech|>")
    processor = CompanionAudioProcessor(session, FakeKws(), asr)

    await processor.process_pcm(b"vocalization")
    await asyncio.sleep(0)

    assert api.ask_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event", "expected_text"),
    [
        ("<|Cough|>", None),
        ("<|Speech|>", "停止"),
        ("<|Unknown|>", "停止"),
        ("", "停止"),
    ],
)
async def test_frame_processor_only_rejects_known_non_speech_events(
    event: str,
    expected_text: str | None,
) -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = EventFrameAsr("停止", event=event)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    await processor.process_frame(b"\0" * 320, b"\0" * 320)
    await asyncio.sleep(0)

    expected_calls = [] if expected_text is None else [(FISH_ID, expected_text)]
    assert api.ask_calls == expected_calls
    assert asr.finish_calls == 1


@pytest.mark.asyncio
async def test_processor_merges_asr_endpoints_from_one_continuous_utterance() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    kws = FakeKws()
    asr = FakeAsr()
    processor = CompanionAudioProcessor(
        session,
        kws,
        asr,
        utterance_commit_delay_seconds=0.03,
    )

    asr.update = AsrUpdate("一二三", is_final=True)
    await processor.process_pcm(b"first endpoint")
    asr.update = AsrUpdate("四", is_final=False)
    await processor.process_pcm(b"continued speech")
    asr.update = AsrUpdate("四五", is_final=True)
    await processor.process_pcm(b"second endpoint")

    await wait_for_condition(lambda: bool(api.ask_calls))

    assert api.ask_calls == [(FISH_ID, "一二三四五")]
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_processor_ignores_asr_text_that_is_echoed_from_current_answer() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("开始")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "收到，稍等一下",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    asr = FakeAsr()
    asr.update = AsrUpdate("收到", is_final=True)
    processor = CompanionAudioProcessor(session, FakeKws(), asr)
    await processor.process_pcm(b"echo")

    assert api.ask_calls == [(FISH_ID, "开始")]
    assert session.state is CompanionState.SPEAKING


@pytest.mark.asyncio
async def test_processor_drops_agent_echo_without_finalizing_agent_message() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    message = agent_message(1, content="任务已经完成，可以查看结果。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    assert session.state is CompanionState.SPEAKING

    asr = FakeAsr()
    asr.update = AsrUpdate("任务已经完成", is_final=True)
    processor = CompanionAudioProcessor(session, FakeKws(), asr)
    await processor.process_pcm(b"agent echo")

    assert api.agent_consumption_calls == []
    assert api.interrupt_calls == []
    assert arbiter.current_owner is not None
    assert arbiter.current_owner.kind == "agent"
    tts.release_finish.set()
    await session.close()
    await processor.close()


@pytest.mark.asyncio
async def test_processor_waits_for_asr_confirmation_before_interrupting() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    # VAD alone is not enough to interrupt an answer. The frame may still be
    # residual speaker echo after AEC, so wait for ASR evidence.
    asr = FrameAsr("新的问题", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, False, False]),
        speech_start_frames=2,
        speech_end_frames=2,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)

    assert tts.calls[-1] == ("duck", None)
    assert tts.calls[-1] != ("stop", None)
    assert session.state is CompanionState.SPEAKING
    assert all(offset == 0 for _, offset in api.progress_calls)

    await processor.process_frame(b"c" * 320, b"r" * 320)
    await processor.process_frame(b"d" * 320, b"r" * 320)
    await asyncio.sleep(0)

    assert tts.calls[-1] == ("stop", None)
    assert session.state is CompanionState.THINKING
    assert api.interrupt_calls == [(ANSWER_ID, 0)]
    assert api.ask_calls == [(FISH_ID, "上一句"), (FISH_ID, "新的问题")]


@pytest.mark.asyncio
async def test_processor_preserves_capture_pre_roll_for_interruption_onset() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    asr = FrameAsr("新的问题", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=CleaningFrameProcessor([True, True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)

    assert asr.accepted == [b"a" * 320, b"b" * 320]


@pytest.mark.asyncio
async def test_processor_keeps_cleaned_audio_during_post_pause_echo_tail() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    asr = FrameAsr("新的问题", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=CleaningFrameProcessor([True, True, True, True]),
        speech_start_frames=1,
    )
    processor.set_stream_delay(20)

    for value in (b"a", b"b", b"c", b"d"):
        await processor.process_frame(value * 320, b"r" * 320)

    assert asr.accepted == [b"a" * 320, b"x" * 320, b"x" * 320, b"d" * 320]
    await processor.close()


@pytest.mark.asyncio
async def test_echo_candidate_ducks_and_restores_tts_without_near_end_voice() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    near_end_vad = FakeInterruptionVad([])
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("不", partial_text=None),
        frame_processor=CleaningFrameProcessor([True, False]),
        interruption_vad=near_end_vad,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"\x00" * 320, b"\x00" * 320)

    assert near_end_vad.reset_calls == 1
    assert api.interrupt_calls == []
    assert api.ask_calls == [(FISH_ID, "上一句")]
    assert ("pause", None) not in tts.calls
    assert tts.calls[-2:] == [("duck", None), ("unduck", None)]
    assert session.state is CompanionState.SPEAKING


@pytest.mark.asyncio
async def test_preprocessed_echo_candidate_keeps_tts_at_full_volume() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    near_end_vad = FakeInterruptionVad([False, False])
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("不", partial_text=None),
        frame_processor=CleaningFrameProcessor([True, False]),
        interruption_vad=near_end_vad,
        defer_interruption_output_until_near_end=True,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    assert ("duck", None) not in tts.calls
    assert ("pause", None) not in tts.calls

    await processor.process_frame(b"\x00" * 320, b"\x00" * 320)

    assert near_end_vad.reset_calls == 1
    assert near_end_vad.accepted == [b"x" * 320, b"x" * 320]
    assert ("duck", None) not in tts.calls
    assert ("pause", None) not in tts.calls
    assert ("unduck", None) not in tts.calls
    assert api.interrupt_calls == []
    assert api.ask_calls == [(FISH_ID, "上一句")]
    assert session.state is CompanionState.SPEAKING


@pytest.mark.asyncio
async def test_preprocessed_near_end_confirmation_pauses_without_ducking() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    near_end_vad = FakeInterruptionVad([False, True])
    asr = FrameAsr("停止对话", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=CleaningFrameProcessor([True, True, True, False]),
        interruption_vad=near_end_vad,
        defer_interruption_output_until_near_end=True,
        speech_start_frames=1,
        speech_end_frames=1,
    )
    processor.set_stream_delay(20)

    await processor.process_frame(b"a" * 320, b"r" * 320)
    assert ("duck", None) not in tts.calls
    assert ("pause", None) not in tts.calls

    await processor.process_frame(b"b" * 320, b"r" * 320)
    assert ("duck", None) not in tts.calls
    assert ("pause", None) in tts.calls

    await processor.process_frame(b"c" * 320, b"r" * 320)
    await processor.process_frame(b"\x00" * 320, b"\x00" * 320)
    await asyncio.sleep(0)

    assert asr.accepted == [b"a" * 320, b"b" * 320, b"x" * 320]
    assert api.interrupt_calls == [(ANSWER_ID, 0)]
    assert api.ask_calls == [(FISH_ID, "上一句"), (FISH_ID, "停止对话")]


@pytest.mark.asyncio
async def test_processor_confirms_interruption_from_post_pause_near_end_voice() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    near_end_vad = FakeInterruptionVad([True])
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("停止对话", partial_text=None),
        frame_processor=CleaningFrameProcessor([True, True, False]),
        interruption_vad=near_end_vad,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"\x00" * 320)
    await processor.process_frame(b"\x00" * 320, b"\x00" * 320)
    await asyncio.sleep(0)

    assert near_end_vad.accepted == [b"x" * 320, b"x" * 320]
    assert ("duck", None) in tts.calls
    assert ("pause", None) in tts.calls
    assert api.interrupt_calls == [(ANSWER_ID, 0)]
    assert api.ask_calls == [(FISH_ID, "上一句"), (FISH_ID, "停止对话")]


@pytest.mark.asyncio
async def test_early_snapshot_cancels_before_speech_end_and_keeps_full_final() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    asr = FrameAsr(
        "等一下我换个问题",
        partial_text=None,
        snapshot_text="等一下",
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, True, True, False]),
        interruption_vad=FakeInterruptionVad([True]),
        speech_start_frames=1,
        speech_end_frames=1,
        early_interrupt_snapshot_seconds=0.03,
        early_interrupt_min_chars=2,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    assert ("pause", None) in tts.calls
    assert asr.snapshot_calls == 0

    await processor.process_frame(b"c" * 320, b"r" * 320)
    await wait_for_condition(lambda: bool(api.interrupt_calls))

    assert asr.snapshot_calls == 1
    assert api.interrupt_calls == [(ANSWER_ID, 0)]
    assert api.ask_calls == [(FISH_ID, "上一句")]

    await processor.process_frame(b"d" * 320, b"r" * 320)
    await processor.process_frame(b"e" * 320, b"r" * 320)
    await wait_for_condition(lambda: len(api.ask_calls) == 2)

    assert asr.finish_calls == 1
    assert api.ask_calls == [
        (FISH_ID, "上一句"),
        (FISH_ID, "等一下我换个问题"),
    ]
    await processor.close()
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_text", "event"),
    [
        ("呵呵。", "<|Cough|>"),
        ("嗯哼。", "<|Speech|>"),
        ("咳咳。", "<|Speech|>"),
    ],
)
async def test_early_snapshot_does_not_interrupt_for_low_evidence_vocalization(
    snapshot_text: str,
    event: str,
) -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    asr = EventFrameAsr(
        snapshot_text,
        event=event,
        snapshot_text=snapshot_text,
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, True]),
        interruption_vad=FakeInterruptionVad([True]),
        speech_start_frames=1,
        speech_end_frames=1,
        early_interrupt_snapshot_seconds=0.03,
        early_interrupt_min_chars=2,
    )

    for value in (b"a", b"b", b"c"):
        await processor.process_frame(value * 320, b"r" * 320)
    await wait_for_condition(
        lambda: asr.snapshot_calls == 1 and processor._early_interruption_task is None
    )

    assert asr.snapshot_calls == 1
    assert api.interrupt_calls == []
    assert api.ask_calls == [(FISH_ID, "上一句")]
    assert session.state is CompanionState.SPEAKING
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_early_snapshot_delay_includes_silero_confirmation_time() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    asr = FrameAsr("完整问题", partial_text=None, snapshot_text="等一下")
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True] * 19),
        interruption_vad=FakeInterruptionVad([True]),
        speech_start_frames=1,
        speech_confirmation_frames=12,
        early_interrupt_snapshot_seconds=0.3,
    )

    for _ in range(18):
        await processor.process_frame(b"a" * 320, b"r" * 320)
    await asyncio.sleep(0)
    assert asr.snapshot_calls == 0

    await processor.process_frame(b"b" * 320, b"r" * 320)
    await wait_for_condition(
        lambda: asr.snapshot_calls == 1 and processor._early_interruption_task is None
    )
    assert asr.snapshot_calls == 1
    await processor.close()
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_text", ["等", "正在"])
async def test_early_snapshot_does_not_cancel_for_one_character_or_echo(
    snapshot_text: str,
) -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    asr = FrameAsr("最终文本", partial_text=None, snapshot_text=snapshot_text)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, True]),
        interruption_vad=FakeInterruptionVad([True]),
        speech_start_frames=1,
        speech_end_frames=1,
        early_interrupt_snapshot_seconds=0.03,
        early_interrupt_min_chars=2,
    )

    for value in (b"a", b"b", b"c"):
        await processor.process_frame(value * 320, b"r" * 320)
    await wait_for_condition(
        lambda: asr.snapshot_calls == 1 and processor._early_interruption_task is None
    )

    assert asr.snapshot_calls == 1
    assert api.interrupt_calls == []
    assert session.state is CompanionState.SPEAKING
    assert session.output_attenuated_for_speech is False
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_processor_does_not_hide_short_interruption_behind_aec_delay() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    near_end_vad = FakeInterruptionVad([True])
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("知道了知道了", partial_text=None),
        frame_processor=CleaningFrameProcessor([True, True, False]),
        interruption_vad=near_end_vad,
        speech_start_frames=1,
        speech_end_frames=1,
    )
    processor.set_stream_delay(600)

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    await processor.process_frame(b"\x00" * 320, b"\x00" * 320)
    await asyncio.sleep(0)

    assert near_end_vad.accepted == [b"x" * 320, b"x" * 320]
    assert api.interrupt_calls == [(ANSWER_ID, 0)]
    assert api.ask_calls == [(FISH_ID, "上一句"), (FISH_ID, "知道了知道了")]


@pytest.mark.asyncio
async def test_processor_keeps_primary_vad_as_interruption_end_boundary() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    near_end_vad = FakeInterruptionVad([True, False])
    asr = FrameAsr("what are you doing", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=CleaningFrameProcessor([True, True, True, False]),
        interruption_vad=near_end_vad,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    for value in (b"a", b"b", b"c", b"d"):
        await processor.process_frame(value * 320, b"r" * 320)

    assert near_end_vad.accepted == [b"x" * 320, b"x" * 320, b"x" * 320]
    assert asr.accepted == [b"a" * 320, b"b" * 320, b"c" * 320]


@pytest.mark.asyncio
async def test_processor_close_restores_output_ducked_for_speech_candidate() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("", partial_text=None),
        frame_processor=FakeFrameProcessor([True]),
        interruption_vad=FakeInterruptionVad([]),
        speech_start_frames=1,
    )
    await processor.process_frame(b"a" * 320, b"r" * 320)

    await processor.close()

    assert tts.calls[-2:] == [("duck", None), ("unduck", None)]


@pytest.mark.asyncio
async def test_processor_requires_stable_partial_before_interrupting() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    # A changing partial is not enough evidence. The same partial must be
    # observed again before it is allowed to stop the current answer.
    asr = SequenceFrameAsr(
        [
            AsrUpdate("停", is_final=False),
            AsrUpdate("停止", is_final=False),
            AsrUpdate("停止", is_final=False),
        ]
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, True]),
        speech_start_frames=1,
        speech_end_frames=2,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    assert session.state is CompanionState.SPEAKING
    await processor.process_frame(b"b" * 320, b"r" * 320)
    assert session.state is CompanionState.SPEAKING
    await processor.process_frame(b"c" * 320, b"r" * 320)
    await asyncio.sleep(0)

    assert tts.calls[-1] == ("stop", None)
    assert session.state is CompanionState.LISTENING
    await session.wait_for_idle()
    assert api.interrupt_calls == [(ANSWER_ID, 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("final_text", ["", "话"])
async def test_processor_keeps_stable_partial_when_final_is_truncated(
    final_text: str,
) -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        StablePartialAsr(final_text=final_text),
        frame_processor=FakeFrameProcessor([True, True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
    )
    for index in range(3):
        await processor.process_frame(bytes([index]) * 320, b"r" * 320)
    await asyncio.sleep(0)

    assert api.interrupt_calls == [(ANSWER_ID, 0)]
    assert api.ask_calls == [(FISH_ID, "上一句"), (FISH_ID, "停止对话")]


@pytest.mark.asyncio
async def test_processor_discards_echo_after_vad_end_without_interrupting() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    asr = FrameAsr("正", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, False, False]),
        speech_start_frames=2,
        speech_end_frames=2,
    )

    for frame in range(4):
        await processor.process_frame(bytes([frame]) * 320, b"r" * 320)

    assert tts.calls[-2:] == [("duck", None), ("unduck", None)]
    assert session.state is CompanionState.SPEAKING
    assert all(offset == 0 for _, offset in api.progress_calls)
    assert api.ask_calls == [(FISH_ID, "上一句")]


@pytest.mark.asyncio
async def test_processor_ignores_one_character_final_fragment_during_answer() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    asr = FrameAsr("正", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, False, False]),
        speech_start_frames=2,
        speech_end_frames=2,
    )

    for index in range(4):
        await processor.process_frame(bytes([index]) * 320, b"r" * 320)

    assert tts.calls[-1] != ("stop", None)
    assert session.state is CompanionState.SPEAKING
    assert api.interrupt_calls == []
    assert api.ask_calls == [(FISH_ID, "上一句")]


@pytest.mark.asyncio
async def test_processor_keeps_long_one_character_final_during_answer() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    asr = FrameAsr("十", partial_text=None)
    voice_values = [True] * 21 + [False, False]
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor(voice_values),
        speech_start_frames=2,
        speech_end_frames=2,
    )

    for index in range(len(voice_values)):
        await processor.process_frame(bytes([index]) * 320, b"r" * 320)

    await asyncio.sleep(0)

    assert tts.calls[-1] == ("stop", None)
    assert api.interrupt_calls == [(ANSWER_ID, 0)]
    assert api.ask_calls == [(FISH_ID, "上一句"), (FISH_ID, "十")]


@pytest.mark.asyncio
async def test_processor_merges_adjacent_vad_segments_before_submitting() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    asr = FrameAsr("一二三四五十", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor(
            [True, True, False, False, True, True, False, False]
        ),
        speech_start_frames=2,
        speech_end_frames=2,
        utterance_commit_delay_seconds=0.05,
    )

    for index in range(4):
        await processor.process_frame(bytes([index]) * 320, b"r" * 320)

    assert api.ask_calls == []
    assert asr.finish_calls == 0

    for index in range(4, 8):
        await processor.process_frame(bytes([index]) * 320, b"r" * 320)

    await wait_for_condition(lambda: bool(api.ask_calls))

    assert api.ask_calls == [(FISH_ID, "一二三四五十")]
    assert asr.finish_calls == 1
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_processor_submits_one_transcript_after_vad_speech_end() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await asyncio.sleep(0)
    asr = FrameAsr("一二三四五")
    frame_processor = FakeFrameProcessor([True, True, True, False, False])
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=frame_processor,
        speech_start_frames=2,
        speech_end_frames=2,
    )

    for index in range(5):
        await processor.process_frame(bytes([index]) * 320, b"r" * 320)

    await asyncio.sleep(0)
    assert api.ask_calls == [(FISH_ID, "一二三四五")]
    assert asr.finish_calls == 1
    assert asr.reset_calls == 0
    assert frame_processor.reset_vad_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "active_frames"),
    [
        ("。", 60),
        ("嗯。", 119),
        ("哼。", 61),
        ("咳咳。", 61),
        ("对。", 45),
    ],
)
async def test_processor_rejects_punctuation_and_nonlexical_finals(
    transcript: str,
    active_frames: int,
) -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr(transcript, partial_text=None),
        frame_processor=FakeFrameProcessor([True] * active_frames + [False]),
        speech_start_frames=1,
        speech_end_frames=1,
        single_char_min_speech_ms=200,
        vad_trailing_silence_ms=350,
    )

    for _ in range(active_frames + 1):
        await processor.process_frame(b"a" * 320, b"r" * 320)
    await asyncio.sleep(0)

    assert api.ask_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "active_frames"),
    [
        ("停。", 55),
        ("好。", 55),
        ("嗯，我想看下一集。", 1),
    ],
)
async def test_processor_keeps_actionable_single_character_and_substantive_finals(
    transcript: str,
    active_frames: int,
) -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr(transcript, partial_text=None),
        frame_processor=FakeFrameProcessor([True] * active_frames + [False]),
        speech_start_frames=1,
        speech_end_frames=1,
        single_char_min_speech_ms=200,
        vad_trailing_silence_ms=350,
    )

    for _ in range(active_frames + 1):
        await processor.process_frame(b"a" * 320, b"r" * 320)
    await asyncio.sleep(0)

    assert api.ask_calls == [(FISH_ID, transcript)]


@pytest.mark.asyncio
async def test_complete_smart_turn_commits_immediately() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FrameAsr("完整问题", partial_text=None)
    smart_turn = FakeSmartTurn(True)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False]),
        smart_turn=smart_turn,
        speech_start_frames=1,
        speech_pre_roll_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    await wait_for_condition(lambda: bool(api.ask_calls))

    assert smart_turn.accepted == [b"a" * 320]
    assert asr.finish_calls == 1
    assert api.ask_calls == [(FISH_ID, "完整问题")]
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_incomplete_smart_turn_reopens_the_same_utterance() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FrameAsr("合并后的完整问题", partial_text=None)
    smart_turn = FakeSmartTurn(False, True)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False, True, False]),
        smart_turn=smart_turn,
        smart_turn_reopen_seconds=0.05,
        speech_start_frames=1,
        speech_pre_roll_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    await wait_for_condition(lambda: len(smart_turn.accepted) == 1)
    assert asr.finish_calls == 0

    await processor.process_frame(b"c" * 320, b"r" * 320)
    await processor.process_frame(b"d" * 320, b"r" * 320)
    await wait_for_condition(lambda: bool(api.ask_calls))

    assert smart_turn.accepted == [b"a" * 320, b"a" * 320 + b"c" * 320]
    assert asr.finish_calls == 1
    assert api.ask_calls == [(FISH_ID, "合并后的完整问题")]
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_incomplete_smart_turn_commits_after_reopen_grace() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FrameAsr("没有后续", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False]),
        smart_turn=FakeSmartTurn(False),
        smart_turn_reopen_seconds=0.02,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    await wait_for_condition(lambda: bool(api.ask_calls))

    assert asr.finish_calls == 1
    assert api.ask_calls == [(FISH_ID, "没有后续")]
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_smart_turn_failure_falls_back_to_vad_final() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FrameAsr("降级提交", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False]),
        smart_turn=FakeSmartTurn(RuntimeError("inference failed")),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    await wait_for_condition(lambda: bool(api.ask_calls))

    assert asr.finish_calls == 1
    assert api.ask_calls == [(FISH_ID, "降级提交")]
    assert processor._smart_turn is None
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_reopen_during_sensevoice_final_decode_keeps_the_full_utterance() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = BlockingSnapshotAsr()
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False, True, False]),
        smart_turn=FakeSmartTurn(True, True),
        speech_start_frames=1,
        speech_pre_roll_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    assert await asyncio.to_thread(asr.decode_started.wait, 1)

    await processor.process_frame(b"c" * 320, b"r" * 320)
    asr.release_first_decode.set()
    await processor.process_frame(b"d" * 320, b"r" * 320)
    await asyncio.sleep(0.02)

    assert asr.finish_calls == 1
    assert asr.accepted == []
    assert api.ask_calls == [(FISH_ID, "合并问题")]


@pytest.mark.asyncio
async def test_new_speech_waits_after_sensevoice_snapshot_is_claimed() -> None:
    api = FakeApi()
    session = BlockingSubmitSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = ClaimingSnapshotAsr()
    frame_processor = FakeFrameProcessor([True, False, True])
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=frame_processor,
        smart_turn=FakeSmartTurn(True),
        speech_start_frames=1,
        speech_pre_roll_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    await asyncio.wait_for(session.submit_started.wait(), timeout=1)
    assert asr.accepted == []

    new_start = asyncio.create_task(processor.process_frame(b"c" * 320, b"r" * 320))
    await asyncio.sleep(0.01)
    assert not new_start.done()
    assert frame_processor.processed_captures == [b"a" * 320, b"b" * 320]
    assert asr.accepted == []

    session.release_submit.set()
    await asyncio.wait_for(new_start, timeout=1)

    assert api.ask_calls == [(FISH_ID, "前半句")]
    assert asr.accepted == [b"c" * 320]


@pytest.mark.asyncio
async def test_new_speech_waits_for_destructive_online_final() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = BlockingOnlineAsr()
    frame_processor = FakeFrameProcessor([True, False, True])
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=frame_processor,
        smart_turn=FakeSmartTurn(True),
        speech_start_frames=1,
        speech_pre_roll_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    assert await asyncio.to_thread(asr.finish_started.wait, 1)

    new_start = asyncio.create_task(processor.process_frame(b"c" * 320, b"r" * 320))
    await asyncio.sleep(0.01)
    assert not new_start.done()
    assert frame_processor.processed_captures == [b"a" * 320, b"b" * 320]

    asr.release_finish.set()
    await asyncio.wait_for(new_start, timeout=1)

    assert asr.accepted == [b"c" * 320]
    assert api.ask_calls == [(FISH_ID, "在线上一段")]


@pytest.mark.asyncio
async def test_smart_turn_timeout_falls_back_once_and_disables_more_inference() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    smart_turn = BlockingSmartTurn()
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("超时降级", partial_text=None),
        frame_processor=FakeFrameProcessor([True, False]),
        smart_turn=smart_turn,
        smart_turn_timeout_seconds=0.01,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    assert await asyncio.to_thread(smart_turn.started.wait, 1)
    await asyncio.sleep(0.02)

    assert smart_turn.calls == 1
    assert processor._smart_turn is None
    assert api.ask_calls == [(FISH_ID, "超时降级")]
    smart_turn.release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_processor_uses_independent_speech_pre_roll_with_one_frame_gate() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FrameAsr("你好", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([False, False, True, False]),
        speech_start_frames=1,
        speech_pre_roll_frames=3,
        speech_end_frames=1,
    )

    for index in range(4):
        await processor.process_frame(bytes([index]) * 320, b"r" * 320)

    assert asr.accepted == [
        b"\x00" * 320,
        b"\x01" * 320,
        b"\x02" * 320,
    ]


@pytest.mark.asyncio
async def test_processor_uses_capture_audio_for_listening_transcript() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = FrameAsr("what are you doing", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=CleaningFrameProcessor([True, True, False]),
        speech_start_frames=1,
        speech_pre_roll_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)

    assert asr.accepted == [b"a" * 320, b"b" * 320]


@pytest.mark.asyncio
async def test_processor_shows_what_it_hears_while_listening() -> None:
    class HearingEvents(NoopPetEventSink):
        def __init__(self) -> None:
            self.heard: list[str] = []

        def hearing(self, text: str) -> None:
            self.heard.append(text)

    events = HearingEvents()
    session = CompanionSession(FakeApi(), FakeTts(), pet_events=events)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("今天天气不错", partial_text=" 今天天气 "),
        frame_processor=FakeFrameProcessor([True, True]),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"r" * 320)

    assert events.heard == ["今天天气"]


@pytest.mark.asyncio
async def test_processor_reports_vad_and_asr_boundaries_to_diagnostics() -> None:
    api = FakeApi()
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(api, FakeTts(), diagnostics=diagnostics)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await asyncio.sleep(0)

    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("你好", partial_text="你"),
        frame_processor=FakeFrameProcessor([True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
        diagnostics=diagnostics,
    )
    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)

    names = [
        event[0]
        for event in diagnostics.events
        if event[0]
        not in {
            "session_state_changed",
            "wake_word_detected",
            "tts_start_requested",
            "tts_playback_started",
            "tts_playback_stopped",
        }
    ]
    assert names[:2] == ["speech_start", "asr_first_partial"]
    assert "speech_end" in names
    assert "asr_final" in names


@pytest.mark.asyncio
async def test_voice_draft_candidate_http_never_blocks_frame_processing() -> None:
    class BlockingTransport(FakeDraftTransport):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: object,
            *,
            revision: int,
            content: str,
            **metadata: object,
        ) -> bool:
            self.started.set()
            await self.release.wait()
            return await super().upsert_voice_draft(
                key,
                revision=revision,
                content=content,
                **metadata,
            )

    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    transport = BlockingTransport()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=FakeDraftDecoder(),
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("最终问题", partial_text=None),
        frame_processor=FakeFrameProcessor([True, True]),
        voice_draft_vad=FakeDraftVad([True, False]),
        voice_draft_gate=SpeculativeSilenceGate(silence_frames=1),
        voice_draft_controller=controller,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    await asyncio.wait_for(
        processor.process_frame(b"b" * 320, b"\0" * 320),
        timeout=0.05,
    )
    await asyncio.wait_for(transport.started.wait(), timeout=0.2)

    transport.release.set()
    await controller.wait_idle()
    assert len(transport.upserts) == 1
    await processor.close()
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("queue_age_ms", "drop_after_start", "reason"),
    [
        (30.001, 0, "audio_queue_age"),
        (0.0, 1, "audio_frame_drop"),
    ],
)
async def test_voice_draft_overload_skips_decode_and_network(
    queue_age_ms: float,
    drop_after_start: int,
    reason: str,
) -> None:
    class CountingDecoder:
        def __init__(self) -> None:
            self.calls = 0

        def recognize(self, _pcm: bytes) -> str:
            self.calls += 1
            return "候选问题"

    api = FakeApi()
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(api, cast(TtsOutput, FakeTts()))
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    transport = FakeDraftTransport()
    decoder = CountingDecoder()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=cast(VoiceDraftDecoder, decoder),
        diagnostics=diagnostics,
    )
    drops = [0]
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        cast(AsrInput, FrameAsr("最终问题", partial_text=None)),
        frame_processor=FakeFrameProcessor([True, True]),
        voice_draft_vad=FakeDraftVad([True, False]),
        voice_draft_gate=SpeculativeSilenceGate(silence_frames=1),
        voice_draft_controller=controller,
        speech_start_frames=1,
        speech_end_frames=1,
        diagnostics=diagnostics,
        dropped_frames=lambda: drops[0],
        frame_queue_age_ms=lambda: queue_age_ms,
        frame_queue_depth=lambda: 4,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    drops[0] = drop_after_start
    await processor.process_frame(b"b" * 320, b"\0" * 320)
    await controller.wait_idle()

    assert decoder.calls == 0
    assert transport.upserts == []
    fallback = [
        kwargs
        for name, kwargs in cast(
            list[tuple[str, dict[str, object]]], diagnostics.events
        )
        if name == "voice_draft_fallback"
    ]
    assert fallback[-1]["metadata"]["reason"] == reason
    await processor.close()
    await controller.close()
    summaries = [
        kwargs["metadata"]
        for name, kwargs in diagnostics.events
        if name == "voice_draft_gate_summary"
    ]
    assert len(summaries) == 1
    assert summaries[0]["reason"] == "abandoned"
    assert summaries[0]["gateFrames"] == 2
    assert summaries[0]["gateRequiredSilenceFrames"] == 1
    assert summaries[0]["gateSilentFrames"] == 1
    assert summaries[0]["gateCandidateCount"] == 1
    assert [name for name, _ in diagnostics.events].count(
        "voice_draft_candidate_triggered"
    ) == 1
    assert "voice_draft_started" not in [name for name, _ in diagnostics.events]


@pytest.mark.asyncio
async def test_voice_draft_queue_age_threshold_allows_exactly_thirty_ms() -> None:
    class CountingDecoder:
        def __init__(self) -> None:
            self.calls = 0

        def recognize(self, _pcm: bytes) -> str:
            self.calls += 1
            return "候选问题"

    api = FakeApi()
    session = CompanionSession(api, cast(TtsOutput, FakeTts()))
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    transport = FakeDraftTransport()
    decoder = CountingDecoder()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=cast(VoiceDraftDecoder, decoder),
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        cast(AsrInput, FrameAsr("最终问题", partial_text=None)),
        frame_processor=FakeFrameProcessor([True, True]),
        voice_draft_vad=FakeDraftVad([True, False]),
        voice_draft_gate=SpeculativeSilenceGate(silence_frames=1),
        voice_draft_controller=controller,
        speech_start_frames=1,
        speech_end_frames=1,
        dropped_frames=lambda: 0,
        frame_queue_age_ms=lambda: 30.0,
        frame_queue_depth=lambda: 3,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    await processor.process_frame(b"b" * 320, b"\0" * 320)
    await controller.wait_idle()

    assert decoder.calls == 1
    assert len(transport.upserts) == 1
    await processor.close()
    await controller.close()


@pytest.mark.asyncio
async def test_smart_turn_reopen_retains_logical_turn_drop_baseline() -> None:
    api = FakeApi()
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(api, cast(TtsOutput, FakeTts()))
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    transport = FakeDraftTransport()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=FakeDraftDecoder(),
        diagnostics=diagnostics,
    )
    drops = [4]
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        cast(AsrInput, FrameAsr("最终问题", partial_text=None)),
        frame_processor=FakeFrameProcessor([]),
        voice_draft_vad=FakeDraftVad([False]),
        voice_draft_gate=SpeculativeSilenceGate(silence_frames=1),
        voice_draft_controller=controller,
        dropped_frames=lambda: drops[0],
        frame_queue_age_ms=lambda: 0.0,
        frame_queue_depth=lambda: 0,
    )

    processor._speech_generation = 1
    processor._open_or_reopen_voice_draft()
    drops[0] = 5
    processor._speech_generation = 2
    processor._open_or_reopen_voice_draft()
    processor._process_voice_draft_frame(b"a" * 320)
    await controller.wait_idle()

    assert processor._voice_draft_dropped_frames_baseline == 4
    assert transport.upserts == []
    resumed = [
        kwargs["metadata"]
        for name, kwargs in diagnostics.events
        if name == "voice_draft_resumed"
    ]
    assert len(resumed) == 1
    assert resumed[0]["reason"] == "primary_start"
    assert resumed[0]["attemptRevision"] == 0
    await processor.close()
    await controller.close()


@pytest.mark.asyncio
async def test_voice_draft_smart_turn_reopen_keeps_turn_id_until_final_commit() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    transport = FakeDraftTransport()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=FakeDraftDecoder(),
    )
    smart_turn = FakeSmartTurn(False, True)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("合并后的完整问题", partial_text=None),
        frame_processor=FakeFrameProcessor([True, True, False, True, True, False]),
        smart_turn=smart_turn,
        smart_turn_reopen_seconds=0.1,
        voice_draft_vad=FakeDraftVad([True, False, False, True, False, False]),
        voice_draft_gate=SpeculativeSilenceGate(silence_frames=1),
        voice_draft_controller=controller,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    for frame in (b"a", b"b", b"c"):
        await processor.process_frame(frame * 320, b"\0" * 320)
    for _ in range(100):
        if smart_turn.accepted:
            break
        await asyncio.sleep(0.001)
    for frame in (b"d", b"e", b"f"):
        await processor.process_frame(frame * 320, b"\0" * 320)
    for _ in range(200):
        if api.voice_commit_calls:
            break
        await asyncio.sleep(0.001)
    await controller.wait_idle()

    assert len(transport.upserts) == 2
    draft_keys = {call[0] for call in transport.upserts}
    assert len(draft_keys) == 1
    assert [call[1] for call in transport.upserts] == [1, 3]
    assert len(api.voice_commit_calls) == 1
    commit_key, revision, promote_revision, content = api.voice_commit_calls[0]
    assert commit_key in draft_keys
    assert (revision, promote_revision, content) == (4, 3, "合并后的完整问题")
    assert api.ask_calls == []

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)
    await processor.close()
    await controller.close()


@pytest.mark.asyncio
async def test_voice_draft_rejection_cancels_without_commit_or_state_change() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    transport = FakeDraftTransport()
    controller = VoiceDraftTurnController(
        client=transport,
        decoder=FakeDraftDecoder(),
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("。", partial_text=None),
        frame_processor=FakeFrameProcessor([True, True, False]),
        voice_draft_vad=FakeDraftVad([True, False]),
        voice_draft_gate=SpeculativeSilenceGate(silence_frames=1),
        voice_draft_controller=controller,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    for frame in (b"a", b"b", b"c"):
        await processor.process_frame(frame * 320, b"\0" * 320)
    await controller.wait_idle()

    assert len(transport.upserts) == 1
    assert [(call[1], call[2]) for call in transport.cancels] == [(2, "rejected")]
    assert api.voice_commit_calls == []
    assert api.ask_calls == []
    assert session.state is CompanionState.LISTENING
    await processor.close()
    await controller.close()


@pytest.mark.asyncio
async def test_voice_draft_disabled_keeps_ordinary_transcript_submission() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("普通问题", partial_text=None),
        frame_processor=FakeFrameProcessor([True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    await processor.process_frame(b"b" * 320, b"\0" * 320)
    for _ in range(100):
        if api.ask_calls:
            break
        await asyncio.sleep(0.001)

    assert api.ask_calls == [(FISH_ID, "普通问题")]
    assert api.voice_commit_calls == []
    await session.close()
    await processor.close()


@pytest.mark.asyncio
async def test_legacy_final_freezes_binding_before_interrupt_await() -> None:
    first = frozen_turn(generation=1)
    second = frozen_turn(generation=1, session_value=202)
    session = FrozenTurnSessionHarness(first)
    session.state = CompanionState.SPEAKING
    session.block_interrupt = True
    asr = FakeAsr()
    asr.update = AsrUpdate("这是对A的回复", is_final=True)
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        asr,
    )

    processing = asyncio.create_task(processor.process_pcm(b"final"))
    await asyncio.wait_for(session.interrupt_started.wait(), timeout=1)
    assert session.freeze_calls == [first]
    session.current_frozen_turn = second
    session.session_id = second.session_id
    session.release_interrupt.set()
    await processing

    assert session.submissions == [("这是对A的回复", first)]
    assert session.reservations == 1
    assert session.releases == 1
    await processor.close()


@pytest.mark.asyncio
async def test_frame_finalization_freezes_binding_in_finalized_turn() -> None:
    first = frozen_turn(generation=1)
    session = FrozenTurnSessionHarness(first)
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        FrameAsr("frame final", partial_text=None),
        frame_processor=FakeFrameProcessor([True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    await processor.process_frame(b"b" * 320, b"\0" * 320)

    assert session.freeze_calls == [first]
    assert session.submissions == [("frame final", first)]
    assert session.reservations == 1
    assert session.releases == 1
    await processor.close()


@pytest.mark.asyncio
async def test_close_releases_frame_frozen_turn_before_finalization() -> None:
    first = frozen_turn(generation=1)
    session = FrozenTurnSessionHarness(first)
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        FrameAsr("unused", partial_text=None),
        frame_processor=FakeFrameProcessor([True]),
        speech_start_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    await processor.close()

    assert session.freeze_calls == [first]
    assert session.frozen_releases == [first]
    assert session.submissions == []
    assert session.reservations == 1
    assert session.releases == 1


@pytest.mark.asyncio
async def test_close_during_claimed_frame_finish_does_not_refreeze_or_submit() -> None:
    first = frozen_turn(generation=1)
    session = FrozenTurnSessionHarness(first)
    asr = BlockingOnlineAsr()
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False]),
        utterance_commit_delay_seconds=0.001,
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    await processor.process_frame(b"b" * 320, b"\0" * 320)
    assert await asyncio.to_thread(asr.finish_started.wait, 1)

    closing = asyncio.create_task(processor.close())
    await asyncio.sleep(0)
    assert not closing.done()
    asr.release_finish.set()
    await asyncio.wait_for(closing, timeout=1)

    assert session.freeze_calls == [first]
    assert session.frozen_releases == [first]
    assert session.submissions == []
    assert session.reservations == 1
    assert session.releases == 1


@pytest.mark.asyncio
async def test_close_during_immediate_frame_finish_does_not_refreeze_or_submit() -> (
    None
):
    first = frozen_turn(generation=1)
    session = FrozenTurnSessionHarness(first)
    asr = BlockingOnlineAsr()
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    finishing = asyncio.create_task(processor.process_frame(b"b" * 320, b"\0" * 320))
    assert await asyncio.to_thread(asr.finish_started.wait, 1)

    await processor.close()
    asr.release_finish.set()
    await asyncio.wait_for(finishing, timeout=1)

    assert session.freeze_calls == [first]
    assert session.frozen_releases == [first]
    assert session.submissions == []
    assert session.reservations == 1
    assert session.releases == 1


@pytest.mark.asyncio
async def test_close_releases_legacy_pending_frozen_turn_once() -> None:
    first = frozen_turn(generation=1)
    session = FrozenTurnSessionHarness(first)
    asr = FakeAsr()
    asr.update = AsrUpdate("pending final", is_final=True)
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        asr,
        utterance_commit_delay_seconds=60,
    )

    await processor.process_pcm(b"final")
    await processor.close()

    assert session.freeze_calls == [first]
    assert session.frozen_releases == [first]
    assert session.submissions == []
    assert session.reservations == 1
    assert session.releases == 1


@pytest.mark.asyncio
async def test_frame_delivery_failure_releases_frozen_context_and_reservation() -> None:
    first = frozen_turn(generation=1)
    session = FrozenTurnSessionHarness(first)
    session.submit_error = RuntimeError("delivery failed")
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        FrameAsr("frame final", partial_text=None),
        frame_processor=FakeFrameProcessor([True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
    )

    await processor.process_frame(b"a" * 320, b"\0" * 320)
    with pytest.raises(RuntimeError, match="delivery failed"):
        await processor.process_frame(b"b" * 320, b"\0" * 320)

    assert session.frozen_releases == [first]
    assert session.releases == 1
    await processor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_path", "transcript", "speech_frames", "reason"),
    [
        ("frame", "嗯。", 30, "nonlexical_vocalization"),
        ("frame", "对。", 10, "short_single_character"),
        ("legacy", "嗯。", 30, "nonlexical_vocalization"),
    ],
)
async def test_completed_agent_result_does_not_exempt_low_evidence_final(
    final_path: str,
    transcript: str,
    speech_frames: int,
    reason: str,
) -> None:
    api = AgentFakeApi()
    api.answer_expected = False
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="现在是20:41。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.release_finish.set()
    await session.wait_for_idle()
    context = session.reply_context
    assert context is not None and context.subject_kind == "run_result"
    assert session.state is CompanionState.LISTENING
    consumption_calls = list(api.agent_consumption_calls)

    if final_path == "frame":
        processor = CompanionAudioProcessor(
            session,
            FakeKws(),
            FrameAsr(transcript, partial_text=None),
            frame_processor=FakeFrameProcessor([True] * speech_frames + [False]),
            speech_start_frames=1,
            speech_end_frames=1,
            vad_trailing_silence_ms=0,
            diagnostics=diagnostics,
        )
        for _ in range(speech_frames + 1):
            await processor.process_frame(b"a" * 320, bytes(320))
    else:
        asr = FakeAsr()
        asr.update = AsrUpdate(transcript, is_final=True, event="<|Speech|>")
        processor = CompanionAudioProcessor(
            session, FakeKws(), asr, diagnostics=diagnostics
        )
        await processor.process_pcm(b"final")
    await session.wait_for_idle()

    assert api.ask_calls == []
    assert api.agent_consumption_calls == consumption_calls
    assert session.reply_context == context
    assert [
        kwargs["metadata"]["reason"]
        for name, kwargs in diagnostics.events
        if name == "asr_final_rejected"
    ] == [reason]
    await processor.close()
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("final_path", ["frame", "legacy"])
@pytest.mark.parametrize("subject_kind", ["interaction", "run_result"])
async def test_final_only_agent_barge_in_consumes_safe_checkpoint_before_ask(
    final_path: str,
    subject_kind: Literal["interaction", "run_result"],
) -> None:
    order: list[tuple[str, object]] = []

    class OrderedApi(AgentFakeApi):
        async def consume_agent_message(
            self,
            fish_id: UUID,
            message_id: UUID,
            *,
            session_id: UUID,
            consumed_offset: int,
            interrupted: bool,
        ) -> None:
            order.append(("consume", (consumed_offset, interrupted)))
            await super().consume_agent_message(
                fish_id,
                message_id,
                session_id=session_id,
                consumed_offset=consumed_offset,
                interrupted=interrupted,
            )

        async def ask(
            self,
            fish_id: UUID,
            content: str,
            *,
            session_id: UUID,
            reply_context: CompanionReplyContext | None = None,
            input_mode: str | None = None,
        ) -> CompanionAskResult:
            order.append(("ask", reply_context))
            return await super().ask(
                fish_id,
                content,
                session_id=session_id,
                reply_context=reply_context,
            )

    api = OrderedApi()
    api.answer_expected = False
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="请确认是否继续。", subject_kind=subject_kind)
    transcript = "嗯。" if subject_kind == "interaction" else "继续吧。"
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    assert session.reply_context is not None
    tts.played_offset = 4

    if final_path == "frame":
        processor = CompanionAudioProcessor(
            session,
            FakeKws(),
            FrameAsr(transcript, partial_text=None),
            frame_processor=FakeFrameProcessor([True, False]),
            speech_start_frames=1,
            speech_end_frames=1,
        )
        await processor.process_frame(b"a" * 320, b"\0" * 320)
        await processor.process_frame(b"b" * 320, b"\0" * 320)
    else:
        asr = FakeAsr()
        asr.update = AsrUpdate(transcript, is_final=True)
        processor = CompanionAudioProcessor(session, FakeKws(), asr)
        await processor.process_pcm(b"final")
    await session.wait_for_idle()

    assert order == [
        ("consume", (0, True)),
        ("ask", CompanionReplyContext(subject_kind, message.subject_id)),
    ]
    await processor.close()
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_kind", ["interaction", "run_result"])
async def test_early_nonlexical_snapshot_only_interrupts_for_agent_interaction(
    subject_kind: Literal["interaction", "run_result"],
) -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="请确认是否继续。", subject_kind=subject_kind)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    assert session.reply_context is not None

    asr = EventFrameAsr(
        "嗯。",
        event="<|Speech|>",
        snapshot_text="嗯。",
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True, True]),
        interruption_vad=FakeInterruptionVad([True]),
        speech_start_frames=1,
        speech_end_frames=1,
        early_interrupt_snapshot_seconds=0.03,
        early_interrupt_min_chars=2,
        diagnostics=diagnostics,
    )

    for value in (b"a", b"b", b"c"):
        await processor.process_frame(value * 320, b"r" * 320)
    await wait_for_condition(
        lambda: asr.snapshot_calls == 1 and processor._early_interruption_task is None
    )

    assert asr.snapshot_calls == 1
    if subject_kind == "interaction":
        assert ("interrupt", None) in tts.calls
        assert session.state is CompanionState.LISTENING
        assert "interruption_confirmed" in [name for name, _ in diagnostics.events]
    else:
        assert ("interrupt", None) not in tts.calls
        assert session.state is CompanionState.SPEAKING
        assert "interruption_confirmed" not in [name for name, _ in diagnostics.events]
        assert api.ask_calls == []
        tts.release_finish.set()
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_sensevoice_cough_overrides_explicit_agent_reply_context() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="请确认是否继续。", subject_kind="interaction")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    assert session.reply_context is not None

    asr = FakeAsr()
    asr.update = AsrUpdate("嗯。", is_final=True, event="<|Cough|>")
    processor = CompanionAudioProcessor(session, FakeKws(), asr)
    await processor.process_pcm(b"cough")
    await asyncio.sleep(0)

    assert api.ask_calls == []
    assert api.agent_consumption_calls == []
    assert ("interrupt", None) not in tts.calls
    tts.release_finish.set()
    await processor.close()
    await session.close()


@pytest.mark.asyncio
async def test_pending_legacy_parts_keep_the_first_frozen_binding() -> None:
    first = frozen_turn(generation=1)
    later_context = FrozenVoiceTurn(
        fish_id=first.fish_id,
        session_id=first.session_id,
        voice_conversation_id=first.voice_conversation_id,
        binding_generation=first.binding_generation,
        reply_context=None,
        context_reservation_id=UUID(int=999),
        reply_blocked=False,
        agent_reply_finalize=None,
    )
    session = FrozenTurnSessionHarness(first)
    asr = FakeAsr()
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        asr,
        utterance_commit_delay_seconds=0.02,
    )

    asr.update = AsrUpdate("第一段", is_final=True)
    await processor.process_pcm(b"first")
    session.current_frozen_turn = later_context
    asr.update = AsrUpdate("继续", is_final=False)
    await processor.process_pcm(b"partial")
    asr.update = AsrUpdate("第二段", is_final=True)
    await processor.process_pcm(b"second")
    await asyncio.sleep(0.03)

    assert session.submissions == [("第一段第二段", first)]
    await processor.close()


@pytest.mark.asyncio
async def test_pending_legacy_parts_do_not_merge_across_binding_generation() -> None:
    first = frozen_turn(generation=1)
    second = frozen_turn(generation=2, session_value=202)
    session = FrozenTurnSessionHarness(first)
    asr = FakeAsr()
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        asr,
        utterance_commit_delay_seconds=0.02,
    )

    asr.update = AsrUpdate("旧绑定", is_final=True)
    await processor.process_pcm(b"old")
    session.current_frozen_turn = second
    session.fish_id = second.fish_id
    session.session_id = second.session_id
    session.binding_generation = second.binding_generation
    asr.update = AsrUpdate("新绑定", is_final=True)
    await processor.process_pcm(b"new")
    await asyncio.sleep(0.03)

    assert session.submissions == [
        ("旧绑定", first),
        ("新绑定", second),
    ]
    await processor.close()


@pytest.mark.asyncio
async def test_unconfirmed_sound_while_speaking_does_not_take_the_foreground() -> None:
    # Reserving the foreground drops proactive speech (such as TA's first
    # words), so a cough or an echo must not do it before it is confirmed.
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": "00000000-0000-0000-0000-000000000101",
                "content": "正在回答",
            },
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FrameAsr("嗯", partial_text=None),
        frame_processor=CleaningFrameProcessor([True, False]),
        interruption_vad=FakeInterruptionVad([False, False]),
        defer_interruption_output_until_near_end=True,
        speech_start_frames=1,
        speech_end_frames=1,
    )
    await processor.process_frame(b"a" * 320, b"r" * 320)

    assert processor._foreground_reservation is None
    assert session._foreground_reservation is None
    assert session.state is CompanionState.SPEAKING


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_text", ["。", "正在", "嗯。", "等", "", "error"])
async def test_rejected_early_candidate_restores_output_and_keeps_later_final(
    snapshot_text: str,
) -> None:
    class SnapshotAsr(FrameAsr):
        def recognize_snapshot(self, pcm: bytes) -> str | None:
            result = super().recognize_snapshot(pcm)
            if result == "error":
                raise RuntimeError("snapshot decode failed")
            return result

    api, tts = FakeApi(), FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    await wait_for_condition(lambda: session.state is CompanionState.SPEAKING)
    asr = SnapshotAsr(
        "我现在要问另一个问题", partial_text=None, snapshot_text=snapshot_text
    )
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True] * 103 + [False]),
        interruption_vad=FakeInterruptionVad([True] * 103),
        speech_start_frames=1,
        speech_end_frames=1,
        early_interrupt_snapshot_seconds=0.03,
    )
    try:
        for _ in range(3):
            await processor.process_frame(b"a" * 320, bytes(320))
        await wait_for_condition(
            lambda: (
                asr.snapshot_calls == 1 and processor._early_interruption_task is None
            )
        )
        assert not session.output_attenuated_for_speech
        assert api.interrupt_calls == []
        assert api.ask_calls == [(FISH_ID, "上一句")]
        assert tts.calls[-1] == ("resume", None)

        # A continuously false-positive VAD must not re-pause or spin the ASR.
        for _ in range(100):
            await processor.process_frame(b"b" * 320, bytes(320))
        assert not session.output_attenuated_for_speech
        assert asr.snapshot_calls == 1
        assert tts.calls.count(("pause", None)) == 1
        assert len(asr.accepted) == 103

        # A weak early sample must not discard a later genuine utterance.
        await processor.process_frame(bytes(320), bytes(320))
        await wait_for_condition(lambda: len(api.ask_calls) == 2)
        assert api.interrupt_calls == [(ANSWER_ID, 0)]
        assert api.ask_calls[-1] == (FISH_ID, "我现在要问另一个问题")
    finally:
        await processor.close()
        await session.close()


@pytest.mark.asyncio
async def test_retired_early_rejection_cannot_resume_new_candidate() -> None:
    class BlockingSnapshots(FrameAsr):
        def __init__(self) -> None:
            super().__init__("。", partial_text=None)
            self.release = [threading.Event(), threading.Event()]

        def recognize_snapshot(self, _pcm: bytes) -> str:
            index = self.snapshot_calls
            self.snapshot_calls += 1
            assert self.release[index].wait(2)
            return "。"

    api, tts = FakeApi(), FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    await wait_for_condition(lambda: session.state is CompanionState.SPEAKING)
    asr = BlockingSnapshots()
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True] * 3 + [False] + [True] * 3),
        interruption_vad=FakeInterruptionVad([True] * 7),
        speech_start_frames=1,
        speech_end_frames=1,
        early_interrupt_snapshot_seconds=0.03,
    )
    old_task = None
    try:
        for _ in range(3):
            await processor.process_frame(b"a" * 320, bytes(320))
        await wait_for_condition(lambda: asr.snapshot_calls == 1)
        old_task = processor._early_interruption_task
        assert old_task is not None
        await processor.process_frame(bytes(320), bytes(320))
        assert not session.output_attenuated_for_speech

        for _ in range(3):
            await processor.process_frame(b"b" * 320, bytes(320))
        await wait_for_condition(lambda: asr.snapshot_calls == 2)
        current_task = processor._early_interruption_task
        assert current_task is not old_task
        assert session.output_paused_for_speech
        resume_count = tts.calls.count(("resume", None))
        asr.release[0].set()
        await asyncio.wait_for(old_task, timeout=1)
        assert session.output_paused_for_speech
        assert tts.calls.count(("resume", None)) == resume_count
        assert processor._early_interruption_task is current_task

        asr.release[1].set()
        await wait_for_condition(lambda: processor._early_interruption_task is None)
        assert not session.output_attenuated_for_speech
    finally:
        for release in asr.release:
            release.set()
        if old_task is not None:
            await asyncio.gather(old_task, return_exceptions=True)
        await processor.close()
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["duck", "pause"])
@pytest.mark.parametrize("stop", ["microphone", "close"])
async def test_input_stop_during_output_control_does_not_leave_late_attenuation(
    control: str,
    stop: str,
) -> None:
    class BlockingTts(FakeTts):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def duck(self) -> None:
            if control == "duck":
                self.started.set()
                await self.release.wait()
            await super().duck()

        async def pause(self) -> None:
            if control == "pause":
                self.started.set()
                await self.release.wait()
            await super().pause()

    api, tts = FakeApi(), BlockingTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    await wait_for_condition(lambda: session.state is CompanionState.SPEAKING)
    asr = FrameAsr("不应提交", partial_text=None)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True, True]),
        interruption_vad=FakeInterruptionVad([True]),
        speech_start_frames=1,
        speech_end_frames=1,
    )
    inflight = None
    try:
        if control == "pause":
            await processor.process_frame(b"a" * 320, bytes(320))
        inflight = asyncio.create_task(processor.process_frame(b"b" * 320, bytes(320)))
        await asyncio.wait_for(tts.started.wait(), timeout=1)
        stopping = asyncio.create_task(
            processor.set_microphone_enabled(False)
            if stop == "microphone"
            else processor.close()
        )
        await asyncio.sleep(0)
        accepted_before_release = len(asr.accepted)
        tts.release.set()
        await asyncio.wait_for(asyncio.gather(stopping, inflight), timeout=1)
        assert not session.output_attenuated_for_speech
        assert len(asr.accepted) == accepted_before_release
        assert api.interrupt_calls == []
    finally:
        tts.release.set()
        if inflight is not None:
            await asyncio.gather(inflight, return_exceptions=True)
        await processor.close()
        await session.close()


@pytest.mark.asyncio
async def test_deferred_asr_failure_restores_output_and_reaches_audio_supervisor() -> (
    None
):
    class FailedAsr(FrameAsr):
        def finish_utterance(self) -> str:
            raise RuntimeError("authoritative ASR failed")

    api, tts = FakeApi(), FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    await wait_for_condition(lambda: session.state is CompanionState.SPEAKING)
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        FailedAsr("", partial_text=None),
        frame_processor=FakeFrameProcessor([True, True, False]),
        interruption_vad=FakeInterruptionVad([True, False]),
        speech_start_frames=1,
        speech_end_frames=1,
        utterance_commit_delay_seconds=0.001,
    )
    try:
        for _ in range(3):
            await processor.process_frame(b"a" * 320, bytes(320))
        assert session.output_paused_for_speech
        finishing = processor._frame_finish_task
        assert finishing is not None
        await asyncio.wait_for(asyncio.gather(finishing, return_exceptions=True), 1)
        assert not session.output_attenuated_for_speech
        # A background final must propagate through the next audio callback,
        # where the existing runner closes and rebuilds the native streams.
        with pytest.raises(RuntimeError, match="authoritative ASR failed"):
            await processor.process_frame(bytes(320), bytes(320))
        assert api.interrupt_calls == []
        assert api.ask_calls == [(FISH_ID, "上一句")]
    finally:
        await processor.close()
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["legacy", "frame"])
async def test_muted_input_rejects_asr_result_returning_from_worker(path: str) -> None:
    class BlockedAsr(FrameAsr):
        def __init__(self) -> None:
            super().__init__("不应提交", partial_text=None)
            self.started = threading.Event()
            self.release = threading.Event()

        def accept_pcm(self, pcm: bytes) -> AsrUpdate:
            self.started.set()
            assert self.release.wait(2)
            return AsrUpdate("不应提交", is_final=True)

    api, tts = FakeApi(), FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    asr = BlockedAsr()
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=FakeFrameProcessor([True]),
        speech_start_frames=1,
    )
    inflight = asyncio.create_task(
        processor.process_pcm(b"a" * 320)
        if path == "legacy"
        else processor.process_frame(b"a" * 320, bytes(320))
    )
    try:
        await wait_for_condition(asr.started.is_set)
        await processor.set_microphone_enabled(False)
        asr.release.set()
        await asyncio.wait_for(inflight, timeout=1)
        await asyncio.sleep(0)
        assert api.ask_calls == []
        assert api.interrupt_calls == []
    finally:
        asr.release.set()
        await asyncio.gather(inflight, return_exceptions=True)
        await processor.close()
        await session.close()
