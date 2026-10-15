"""A speculative next turn cannot evict TTS owned by a committed answer."""

import asyncio
from uuid import UUID

import pytest

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.draft_tts import (
    DraftTtsPromotion,
    DraftTtsSession,
    PromotablePcmSink,
)
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.voice_drafts import VoiceDraftTurnController

from .test_draft_tts import (
    KEY,
    FakeDraftClient,
    FakeSink,
    RecordingDiagnostics,
    load_configuration,
    wait_for_condition,
)
from .test_draft_tts_commit import DelayedStartOutput
from .test_draft_tts_handoff import TEXT, handoff_case
from .test_voice_drafts import (
    FISH_ID,
    SESSION_ID,
    FakeVoiceDraftClient,
    ImmediateDecoder,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connecting", "ready", "promoted"])
async def test_rejected_next_candidate_keeps_committed_provider(phase: str) -> None:
    client = FakeVoiceDraftClient()
    events = FakeDraftClient()
    live = FakeSink()
    arbiter = PlaybackArbiter()
    diagnostics = RecordingDiagnostics()
    outputs: list[DelayedStartOutput] = []

    def factory(sink: PromotablePcmSink) -> DelayedStartOutput:
        output = DelayedStartOutput(sink, blocked=not outputs)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=events,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=factory,
        load_configuration=load_configuration,
        playback_arbiter=arbiter,
        diagnostics=diagnostics,
    )
    decoder = ImmediateDecoder("旧问题")
    controller = VoiceDraftTurnController(
        client=client, decoder=decoder, draft_tts=draft, diagnostics=diagnostics
    )
    try:
        old_owner = controller.open_turn(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            binding_generation=1,
            speech_generation=9,
        )
        old_candidate = controller.candidate(old_owner, pcm=bytes(320))
        assert old_candidate is not None
        await wait_for_condition(lambda: len(outputs) == 1)
        old_output = outputs[0]
        await old_output.start_entered.wait()
        assert await controller.wait_for_registration(old_owner)
        if phase != "connecting":
            old_output.start_release.set()
            await asyncio.wait_for(old_candidate, timeout=0.5)
        submission = controller.commit(old_owner, content="旧问题")
        playback_owner = await arbiter.acquire("answer", UUID(int=991))
        promotion = None
        if phase == "promoted":
            promotion = await draft.promote(
                old_owner.key,
                revision=submission.promote_revision,
                content=TEXT,
                flush_offset=len(TEXT),
                playback_owner=playback_owner,
            )
            assert isinstance(promotion, DraftTtsPromotion)

        # This candidate reaches its ASR result and draft PUT, but is not an
        # accepted interruption. In production this single punctuation is rejected.
        decoder.text = "。"
        new_owner = controller.open_turn(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            binding_generation=1,
            speech_generation=10,
        )
        next_candidate = controller.candidate(new_owner, pcm=bytes(320))
        assert next_candidate is not None
        assert await asyncio.wait_for(next_candidate, timeout=0.5) is not None
        assert [call.key for call in client.upserts] == [old_owner.key, new_owner.key]
        assert len(outputs) == 1
        assert ("interrupt", None) not in old_output.calls
        if phase == "connecting":
            assert not old_candidate.done()
            assert not old_output.start_release.is_set()
        skipped = [
            data
            for name, data in diagnostics.events
            if name == "voice_draft_tts_skipped"
        ]
        assert len(skipped) == 1
        assert skipped[0]["metadata"] == {
            "turnId": str(new_owner.turn_id),
            "revision": 1,
            "reason": "committed_tts_active",
        }

        controller.abandon(new_owner, reason="rejected")
        await wait_for_condition(lambda: bool(client.cancels))
        assert client.cancels[0][0] == new_owner.key
        assert ("interrupt", None) not in old_output.calls
        old_output.start_release.set()
        await asyncio.wait_for(old_candidate, timeout=0.5)
        if promotion is None:
            promotion = await draft.promote(
                old_owner.key,
                revision=submission.promote_revision,
                content=TEXT,
                flush_offset=len(TEXT),
                playback_owner=playback_owner,
            )
        assert isinstance(promotion, DraftTtsPromotion)
        await promotion.output.append(TEXT)
        assert old_output.calls.count(("start", None)) == 1
        assert old_output.calls.count(("append", TEXT)) == 1
        assert len([call for call in live.calls if call[0] == "append"]) == 1

        # Real retirement releases the slot; a later candidate can warm TTS.
        await draft.cancel(old_owner.key, through_revision=submission.revision)
        await wait_for_condition(lambda: ("interrupt", None) in old_output.calls)
        later_key = VoiceDraftKey(FISH_ID, SESSION_ID, UUID(int=993))
        assert await draft.start(later_key, revision=1)
        assert len(outputs) == 2
        assert outputs[1].calls.count(("start", None)) == 1
    finally:
        for output in outputs:
            output.start_release.set()
        await controller.close()
        await draft.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connecting", "ready", "promoted"])
async def test_same_committed_identity_start_is_an_idempotent_join(phase: str) -> None:
    async with handoff_case(block_start=phase == "connecting") as case:
        starting = asyncio.create_task(case.draft.start(KEY, revision=4))
        case.tasks.append(starting)
        await case.created.wait()
        output = case.outputs[0]
        await output.start_entered.wait()
        if phase != "connecting":
            assert await starting
        assert case.draft.mark_committed(KEY, revision=4)
        if phase == "promoted":
            owner = await case.arbiter.acquire("answer", UUID(int=994))
            result = await case.draft.promote(
                KEY,
                revision=4,
                content=TEXT,
                flush_offset=len(TEXT),
                playback_owner=owner,
            )
            assert isinstance(result, DraftTtsPromotion)
        duplicate = asyncio.create_task(case.draft.start(KEY, revision=4))
        case.tasks.append(duplicate)
        for _ in range(3):
            await asyncio.sleep(0)
        if phase == "connecting":
            assert not duplicate.done()
            assert not output.start_cancelled
            output.start_release.set()
        assert await asyncio.wait_for(duplicate, timeout=0.5)
        assert await starting
        assert len(case.outputs) == 1
        assert output.calls.count(("start", None)) == 1
        assert ("interrupt", None) not in output.calls


@pytest.mark.asyncio
async def test_close_still_retires_a_committed_provider() -> None:
    async with handoff_case() as case:
        assert await case.draft.start(KEY, revision=4)
        output = case.outputs[0]
        assert case.draft.mark_committed(KEY, revision=4)
        await case.draft.close()
        assert ("interrupt", None) in output.calls
        assert not await case.draft.start(KEY, revision=4)
        await output.emit_late_pcm()
        assert case.live.calls == []


@pytest.mark.asyncio
async def test_cancelled_duplicate_start_does_not_cancel_committed_startup() -> None:
    async with handoff_case(block_start=True) as case:
        starting = asyncio.create_task(case.draft.start(KEY, revision=4))
        case.tasks.append(starting)
        await case.created.wait()
        output = case.outputs[0]
        await output.start_entered.wait()
        assert case.draft.mark_committed(KEY, revision=4)
        duplicate = asyncio.create_task(case.draft.start(KEY, revision=4))
        case.tasks.append(duplicate)
        for _ in range(3):
            await asyncio.sleep(0)
        duplicate.cancel()
        with pytest.raises(asyncio.CancelledError):
            await duplicate
        assert not starting.done()
        assert not output.start_cancelled
        output.start_release.set()
        assert await asyncio.wait_for(starting, timeout=0.5)
        assert len(case.outputs) == 1
        assert output.calls.count(("start", None)) == 1
        assert ("interrupt", None) not in output.calls
