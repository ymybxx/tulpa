import asyncio

import pytest

from fish_memory.companion.draft_tts import DraftTtsSession, PromotablePcmSink
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.session import CompanionSession
from fish_memory.companion.voice_drafts import VoiceDraftKey

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending_stage",
    ["start", "append", "first_pcm", "public_ahead", "start_no_text", "ready_no_text"],
)
async def test_formal_session_adopts_existing_synthesis_and_appends_only_the_suffix(
    pending_stage: str,
) -> None:
    """Public snapshot/delta/completion cannot replace or duplicate private work."""

    start_entered = asyncio.Event()
    release_start = asyncio.Event()
    append_entered = asyncio.Event()
    release_append = asyncio.Event()
    start_pending = pending_stage.startswith("start")
    has_private_text = not pending_stage.endswith("no_text")
    public_flush_ahead = pending_stage == "public_ahead" or not has_private_text
    if not start_pending:
        release_start.set()
    if pending_stage != "append":
        release_append.set()

    class PendingOutput(FakeDraftOutput):
        def __init__(self, sink: PromotablePcmSink) -> None:
            # Deliberately withhold all PCM: lack of a first audio packet must
            # not authorize starting another synthesis stream.
            super().__init__(sink, emit_pcm=False)
            self.append_active = False

        async def start(self) -> None:
            self.calls.append(("start", None))
            start_entered.set()
            await release_start.wait()
            await self.sink.start()

        async def append(self, text: str) -> None:
            assert not self.append_active
            self.append_active = True
            self.calls.append(("append", text))
            try:
                if not append_entered.is_set():
                    append_entered.set()
                    await release_append.wait()
                self.calls.append(("append_done", text))
            finally:
                self.append_active = False

        async def finish(self) -> None:
            assert not self.append_active
            await super().finish()

    outputs: list[PendingOutput] = []

    def output_factory(sink: PromotablePcmSink) -> PendingOutput:
        output = PendingOutput(sink)
        outputs.append(output)
        return output

    api = FakeApi()
    ordinary_tts = FakeTts()
    private_events = FakeDraftClient()
    arbiter = PlaybackArbiter()
    live_sink = FakeSink()
    draft = DraftTtsSession(
        client=private_events,
        live_sink=live_sink,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
        playback_arbiter=arbiter,
    )
    session = CompanionSession(
        api, ordinary_tts, draft_tts=draft, playback_arbiter=arbiter
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    assert session.session_id is not None
    key = VoiceDraftKey(FISH_ID, session.session_id, TURN_ID)
    starter = asyncio.create_task(draft.start(key, revision=4))
    prefix = "第一段。"
    snapshot_content = prefix + "接管后。"
    suffix = "最后一句。"
    full_content = snapshot_content + suffix
    try:
        await wait_for_condition(lambda: start_entered.is_set() or starter.done())
        if starter.done():
            starter.result()
        assert start_entered.is_set()
        if has_private_text:
            await private_events.events.put(
                draft_snapshot(prefix, flush_offset=len(prefix))
            )
        if has_private_text and not start_pending:
            await asyncio.wait_for(append_entered.wait(), 1)
        if pending_stage in {"first_pcm", "public_ahead"}:
            await wait_for_condition(
                lambda: ("append_done", prefix) in outputs[0].calls
            )
        if pending_stage == "ready_no_text":
            await asyncio.wait_for(asyncio.shield(starter), 1)
        assert not [call for call in live_sink.calls if call[0] == "append"]

        assert await session.submit_voice_turn(
            TURN_ID, revision=5, promote_revision=4, content="最终问题"
        )
        await api.answer_events.put(
            {
                "event": "answer.snapshot",
                "data": {
                    "messageId": str(ANSWER_ID),
                    "content": snapshot_content,
                    "flushOffset": len(snapshot_content)
                    if public_flush_ahead
                    else len(prefix),
                },
            }
        )
        await wait_for_condition(lambda: session._answer_first_content_recorded)
        if start_pending or pending_stage == "append":
            # The formal event reader must stay behind the existing readiness
            # or append operation instead of starting ordinary synthesis.
            for _ in range(20):
                await asyncio.sleep(0)
            assert ordinary_tts.calls == []
            assert ("finish", None) not in outputs[0].calls
        await api.answer_events.put(
            {
                "event": "answer.delta",
                "data": {"messageId": str(ANSWER_ID), "content": suffix},
            }
        )
        await api.answer_events.put(
            {
                "event": "answer.segment",
                "data": {
                    "messageId": str(ANSWER_ID),
                    "flushOffset": len(full_content),
                },
            }
        )
        await api.answer_events.put(
            {
                "event": "answer.completed",
                "data": {"messageId": str(ANSWER_ID), "end": False},
            }
        )
        release_start.set()
        release_append.set()
        await asyncio.wait_for(session.wait_for_idle(), 1)
        await asyncio.wait_for(starter, 1)

        assert len(outputs) == 1
        output = outputs[0]
        assert output.calls.count(("start", None)) == 1
        assert ordinary_tts.calls == []
        appended = [text for name, text in output.calls if name == "append"]
        # An unsent snapshot now releases its short first sentence too. Both
        # paths must continue after that same prefix without sending it twice.
        assert appended[0] == prefix
        assert appended.count(prefix) == 1
        assert "".join(appended) == full_content
        assert output.calls.count(("finish", None)) == 1
        finish_index = output.calls.index(("finish", None))
        assert all(
            index < finish_index
            for index, (name, _text) in enumerate(output.calls)
            if name == "append_done"
        )
    finally:
        release_start.set()
        release_append.set()
        await session.close()
        await asyncio.gather(starter, return_exceptions=True)
        await draft.close()
