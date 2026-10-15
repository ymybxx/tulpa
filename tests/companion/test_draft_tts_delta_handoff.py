import asyncio
from collections.abc import AsyncIterator

import pytest

from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.sse import iter_answer_events
from fish_memory.companion.draft_tts import DraftTtsSession, PromotablePcmSink
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.session import CompanionSession
from fish_memory.companion.voice_drafts import VoiceDraftKey
from fish_memory.models import MessageStatus

from .test_draft_tts import (
    TURN_ID,
    FakeDraftClient,
    FakeDraftOutput,
    FakeSink,
    draft_snapshot,
    load_configuration,
    wait_for_condition,
)
from .test_session import ANSWER_ID, FISH_ID, FakeApi, FakeTts


class StreamingApi(FakeApi):
    def __init__(self, runtime: FishRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self.events: list[str] = []

    async def iter_answer(self, fish_id, *, session_id) -> AsyncIterator[dict]:
        del fish_id, session_id
        async for event in iter_answer_events(self.runtime):
            self.events.append(event.event)
            yield {"event": event.event, "data": event.data}


class MappedOutput(FakeDraftOutput):
    def __init__(self, sink: PromotablePcmSink) -> None:
        super().__init__(sink)
        self.offset = 0

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))
        end = self.offset + len(text)
        await self.sink.append(
            b"\x01\x00" * len(text), start_offset=self.offset, end_offset=end
        )
        self.offset = end


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("private_ahead", "outcome"),
    [
        (False, "matched"),
        (True, "matched"),
        (True, "conflict"),
        (True, "short_completion"),
        (True, "interrupted"),
    ],
)
async def test_real_empty_snapshot_delta_segment_stream_reuses_private_tts(
    private_ahead: bool,
    outcome: str,
) -> None:
    runtime = FishRuntime()
    runtime.answer_message_id = ANSWER_ID
    runtime.status = MessageStatus.GENERATING
    api = StreamingApi(runtime)
    ordinary = FakeTts()
    private = FakeDraftClient()
    live = FakeSink()
    arbiter = PlaybackArbiter()
    outputs: list[FakeDraftOutput] = []

    def factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = MappedOutput(sink)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=private,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=factory,
        load_configuration=load_configuration,
        playback_arbiter=arbiter,
    )
    session = CompanionSession(api, ordinary, draft_tts=draft, playback_arbiter=arbiter)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    assert session.session_id is not None
    key = VoiceDraftKey(FISH_ID, session.session_id, TURN_ID)
    prefix = "已经生成的第一段。"
    suffix = "接下来继续回答。"
    try:
        assert await draft.start(key, revision=4)
        if private_ahead:
            await private.events.put(draft_snapshot(prefix, flush_offset=len(prefix)))
            await wait_for_condition(lambda: ("append", prefix) in outputs[0].calls)
        assert await session.submit_voice_turn(
            TURN_ID, revision=5, promote_revision=4, content="测试问题"
        )
        await wait_for_condition(lambda: api.events == ["answer.snapshot"])
        assert ordinary.calls == []
        assert live.calls == []

        async with runtime.condition:
            runtime.content = prefix[:2]
            runtime.condition.notify_all()
        await wait_for_condition(lambda: session._answer_offset == 2)
        for _ in range(10):
            await asyncio.sleep(0)
        assert ordinary.calls == []
        assert ("interrupt", None) not in outputs[0].calls
        if private_ahead:
            assert not session._tts_started
            assert live.calls == []

        if outcome in {"short_completion", "interrupted"}:
            async with runtime.condition:
                runtime.status = (
                    MessageStatus.COMPLETED
                    if outcome == "short_completion"
                    else MessageStatus.INTERRUPTED
                )
                runtime.condition.notify_all()
            await asyncio.wait_for(session.wait_for_idle(), 1)
            spoken = [text for name, text in ordinary.calls if name == "append"]
            assert spoken == ([prefix[:2]] if outcome == "short_completion" else [])
            assert live.calls == []
            assert ("interrupt", None) in outputs[0].calls
            return

        full_content = (
            prefix[:2] + "改成另一种回答。"
            if outcome == "conflict"
            else prefix + suffix
        )
        async with runtime.condition:
            runtime.content = full_content
            runtime.condition.notify_all()
        await wait_for_condition(lambda: session._answer_offset == len(runtime.content))
        if private_ahead and outcome == "matched":
            for _ in range(10):
                await asyncio.sleep(0)
            assert not session._tts_started
            assert live.calls == []
        async with runtime.condition:
            runtime.flush_offset = len(runtime.content)
            runtime.condition.notify_all()
        await wait_for_condition(lambda: session._tts_started)
        async with runtime.condition:
            runtime.status = MessageStatus.COMPLETED
            runtime.condition.notify_all()
        await asyncio.wait_for(session.wait_for_idle(), 1)

        assert api.events == [
            "answer.snapshot",
            "answer.delta",
            "answer.delta",
            "answer.segment",
            "answer.completed",
        ]
        assert len(outputs) == 1
        assert outputs[0].calls.count(("start", None)) == 1
        if outcome == "conflict":
            spoken = [text for name, text in ordinary.calls if name == "append"]
            assert "".join(spoken) == full_content
            assert ordinary.calls.count(("start", None)) == 1
            assert live.calls == []
            return
        assert ordinary.calls == []
        appended = [text for name, text in outputs[0].calls if name == "append"]
        assert "".join(appended) == prefix + suffix
        if private_ahead:
            assert appended.count(prefix) == 1
    finally:
        await session.close()
        await draft.close()
