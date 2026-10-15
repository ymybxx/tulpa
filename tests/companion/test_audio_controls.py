import asyncio
import struct

import pytest

from fish_memory.companion.processor import CompanionAudioProcessor
from fish_memory.companion.session import CompanionSession, CompanionState
from tests.companion.test_livekit_audio import (
    FakeProcess,
    decoded_packets,
    make_input,
    start_with_first_frame,
)
from tests.companion.test_processor import (
    FISH_ID,
    FakeFrameProcessor,
    FakeKws,
    FrameAsr,
)
from tests.companion.test_session import (
    AgentFakeApi,
    CallbackTts,
    proactive_message,
    wait_until,
)


@pytest.mark.asyncio
async def test_muted_microphone_does_not_run_kws_asr_vad_and_keeps_playback():
    api, tts = AgentFakeApi(), CallbackTts()
    session = CompanionSession(api, tts)
    kws, asr, frames = FakeKws(), FrameAsr("不应发送"), FakeFrameProcessor([False])
    processor = CompanionAudioProcessor(session, kws, asr, frame_processor=frames)
    try:
        await session.set_binding(FISH_ID)
        kws.detect = True
        await processor.set_microphone_enabled(False)
        await processor.process_frame(b"a" * 320, b"r" * 320)
        await processor.process_pcm(b"a" * 320)
        assert not kws.accepted and not asr.accepted and not frames.processed_captures
        assert session.state is CompanionState.STANDBY
        await api.agent_events.put(proactive_message(301, content="继续播放。"))
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert api.agent_consumption_calls[0][1] == len("继续播放。")
        await processor.set_microphone_enabled(True)
        kws.detect = False
        await processor.process_frame(b"a" * 320, bytes(320))
        assert frames.processed_captures == [b"a" * 320]
        assert asr.reset_calls == 1 and frames.reset_vad_calls == 1
    finally:
        await processor.close()
        await session.close()


@pytest.mark.asyncio
async def test_mute_discards_pending_utterance_and_releases_foreground():
    api = AgentFakeApi()
    session = CompanionSession(api, CallbackTts())
    kws, asr = FakeKws(), FrameAsr("不应发送")
    processor = CompanionAudioProcessor(
        session, kws, asr, utterance_commit_delay_seconds=10
    )
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        from fish_memory.companion.asr import AsrUpdate

        class FinalAsr:
            def accept_pcm(self, pcm):
                return AsrUpdate("关闭前半句话", is_final=True)

            def reset(self):
                pass

        processor._asr = FinalAsr()
        await processor.process_pcm(bytes(320))
        await processor.set_microphone_enabled(False)
        assert not api.ask_calls
        assert not processor._pending_parts
        assert not session._playback.foreground_reserved
    finally:
        await processor.close()
        await session.close()


@pytest.mark.asyncio
async def test_livekit_controls_are_independent_and_survive_playback_reset():
    from fish_memory.companion.livekit_audio import CONTROL_MAGIC

    process = FakeProcess()
    audio, spawns = make_input(process)
    audio.set_microphone_enabled(False)
    try:
        await start_with_first_frame(audio, process)
        assert spawns[0][1]["env"]["FISH_MEMORY_MICROPHONE_ENABLED"] == "false"
        audio.set_microphone_enabled(False)
        audio.set_playback_enabled(False)
        generation = audio._generation
        await audio.stop()
        await asyncio.sleep(0.01)
        ops = [
            struct.unpack("<BQ", body)[0]
            for magic, body in decoded_packets(process.stdin.written)
            if magic == CONTROL_MAGIC
        ]
        assert 9 in ops and 11 in ops
        assert not audio._microphone_enabled and not audio._playback_enabled
        assert len(spawns) == 1
        audio.set_playback_enabled(True)
        await asyncio.sleep(0.01)
        assert not audio._microphone_enabled
        assert audio._generation >= generation
    finally:
        await audio.close()
