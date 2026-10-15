import asyncio

import pytest

from fish_memory.companion.answer_progress import AnswerPlaybackProgress
from fish_memory.companion.audio import PcmPlayback, PlayedOffsetTracker
from fish_memory.companion.draft_pcm import PromotablePcmSink
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.session import CompanionSession
from fish_memory.pet.publisher import NoopPetEventSink

from .test_session import ANSWER_ID, FISH_ID, SESSION_ID, FakeApi, FakeTts, wait_until


def test_sentence_start_is_visible_but_only_end_mark_confirms_consumption():
    tracker = PlayedOffsetTracker()
    tracker.add_segment(byte_length=4, start_offset=0, end_offset=4)
    tracker.add_segment(byte_length=6, start_offset=0, end_offset=4)
    tracker.mark(4)
    tracker.add_segment(byte_length=8, start_offset=4, end_offset=9)
    tracker.mark(9)
    assert (tracker.display_offset, tracker.played_offset) == (0, 0)
    tracker.consume(2)
    assert (tracker.display_offset, tracker.played_offset) == (4, 0)
    tracker.consume(6)
    assert tracker.played_offset == 0
    tracker.consume(2)
    assert tracker.played_offset == 4
    tracker.consume(2)
    assert (tracker.display_offset, tracker.played_offset) == (9, 4)
    tracker.consume(6)
    assert tracker.played_offset == 9


def test_late_mark_confirms_already_rendered_audio_but_unmarked_packets_do_not():
    tracker = PlayedOffsetTracker(initial_offset=4)
    tracker.add_segment(byte_length=4, start_offset=4, end_offset=7)
    tracker.consume(4)
    assert (tracker.display_offset, tracker.played_offset) == (7, 4)
    tracker.mark(7)
    assert tracker.played_offset == 7
    with pytest.raises(ValueError):
        tracker.mark(8)


@pytest.mark.asyncio
async def test_draft_handoff_preserves_marks_without_claiming_buffered_audio():
    class Stream:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    live = PcmPlayback(stream_factory=Stream)
    arbiter = PlaybackArbiter()
    sink = PromotablePcmSink(live, 100, playback_arbiter=arbiter)
    await sink.append(b"\0" * 4, start_offset=0, end_offset=4)
    await sink.append(b"\0" * 4, start_offset=0, end_offset=4)
    await sink.mark(4)
    await sink.append(b"\0" * 4, start_offset=4, end_offset=7)
    await sink.mark(7)
    assert sink.played_offset == sink.display_offset == 0
    owner = await arbiter.acquire("answer", ANSWER_ID)
    assert await sink.promote(expected_end_offset=7, owner=owner)
    assert sink.played_offset == sink.display_offset == 0
    live._callback(bytearray(2), 1)
    assert (sink.display_offset, sink.played_offset) == (4, 0)
    live._callback(bytearray(8), 4)
    assert (sink.display_offset, sink.played_offset) == (7, 4)
    live._callback(bytearray(2), 1)
    assert sink.played_offset == 7
    await sink.stop()
    await live.stop()


@pytest.mark.asyncio
async def test_progress_retries_frozen_identity_and_never_samples_next_turn():
    class Api(FakeApi):
        attempts = 0

        async def update_progress(self, fish_id, message_id, consumed_offset):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary failure")
            await super().update_progress(fish_id, message_id, consumed_offset)

    api = Api()
    value = ["第一句。第二句。", 4, 0, "playing"]
    events = []
    progress = AnswerPlaybackProgress(
        api,
        FISH_ID,
        SESSION_ID,
        ANSWER_ID,
        lambda: tuple(value),
        events.append,
        message_sequence=10025,
    )
    await wait_until(lambda: api.attempts == 1)
    value[1:3] = [8, 4]
    progress.finish("interrupted")
    value[:] = ["下一轮文本", 6, 6, "playing"]
    assert await progress.settle()
    assert api.interrupt_calls == [(ANSWER_ID, 4)]
    assert events[-1]["visibleContent"] == "第一句。"
    assert events[-1]["displayOffset"] == events[-1]["consumedOffset"] == 4
    assert len({e["playbackId"] for e in events}) == 1
    assert {e["messageSequence"] for e in events} == {10025}


class ManualTts(FakeTts):
    def __init__(self):
        super().__init__()
        self.display_offset = 0
        self.finishing = asyncio.Event()
        self.release = asyncio.Event()

    async def append(self, text):
        self.calls.append(("append", text))

    async def finish(self):
        self.finishing.set()
        await self.release.wait()

    async def stop(self):
        self.played_offset = self.display_offset = 0


class Events(NoopPetEventSink):
    def __init__(self):
        self.progress = []

    def playback_progress(self, payload):
        self.progress.append(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", ["completed", "interrupted", "shutdown", "runner_shutdown"]
)
async def test_session_persists_device_progress_before_reset_and_retracts_preview(
    terminal,
):
    api, tts, events = FakeApi(), ManualTts(), Events()
    session = CompanionSession(api, tts, pet_events=events)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    text = "第一句。第二句。第三句。"
    await api.answer_events.put(
        {
            "event": "answer.snapshot",
            "data": {"messageId": str(ANSWER_ID), "content": text, "sequence": 10025},
        }
    )
    await api.answer_events.put(
        {"event": "answer.completed", "data": {"messageId": str(ANSWER_ID)}}
    )
    await tts.finishing.wait()
    assert events.progress and all(
        p["messageSequence"] == 10025 for p in events.progress
    )
    await wait_until(lambda: bool(api.progress_calls))
    assert api.progress_calls[-1] == (ANSWER_ID, 0)
    tts.display_offset, tts.played_offset = 8, 4
    await wait_until(lambda: events.progress[-1]["displayOffset"] == 8)
    await wait_until(lambda: api.progress_calls[-1] == (ANSWER_ID, 4))
    if terminal == "completed":
        tts.display_offset = tts.played_offset = len(text)
        tts.release.set()
        await session.wait_for_idle()
        assert api.progress_calls[-1] == (ANSWER_ID, len(text))
        assert events.progress[-1]["status"] == "completed"
    elif terminal == "interrupted":
        await session.interrupt_from_speech()
        await session.wait_for_idle()
        assert api.interrupt_calls[-1] == (ANSWER_ID, 4)
        assert events.progress[-1]["visibleContent"] == "第一句。"
    else:
        if terminal == "runner_shutdown":
            session.quiesce_binding(FISH_ID)
            await tts.stop()
        await session.close()
        assert api.interrupt_calls[-1] == (ANSWER_ID, 4)
        assert events.progress[-1]["visibleContent"] == "第一句。"
    await session.close()


@pytest.mark.asyncio
async def test_draft_context_waits_for_prior_consumption_before_any_llm_request():
    from fish_memory.companion.voice_drafts import VoiceDraftTurnController

    from .test_voice_drafts import FakeVoiceDraftClient, ImmediateDecoder

    entered, release = asyncio.Event(), asyncio.Event()

    async def before_context():
        entered.set()
        await release.wait()
        return True

    client = FakeVoiceDraftClient()
    controller = VoiceDraftTurnController(
        client=client, decoder=ImmediateDecoder(), before_context=before_context
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    task = controller.candidate(owner, pcm=bytes(320))
    await entered.wait()
    assert client.upserts == []
    release.set()
    assert await task is not None
    assert len(client.upserts) == 1
    await controller.close()


@pytest.mark.asyncio
async def test_unconfirmed_audio_is_not_reported_as_successful_playback():
    from fish_memory.companion.performance import PetVoiceStatus

    from .test_session import RecordingPerformanceReporter

    api, tts, events = FakeApi(), ManualTts(), Events()
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(
        api, tts, pet_events=events, performance_reporter=reporter
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "未知音频边界"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": True},
        }
    )
    tts.release.set()
    await session.wait_for_idle()
    assert api.progress_calls[-1] == (ANSWER_ID, 0)
    assert events.progress[-1]["status"] == "failed"
    assert reporter.calls[-1][2] is PetVoiceStatus.FAILED
    await session.close()
