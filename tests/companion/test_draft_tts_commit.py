"""Commit may retire the private SSE before its TTS provider becomes ready."""

import asyncio
from collections.abc import AsyncIterator
from typing import cast
from uuid import UUID

import pytest

from fish_memory.answer.voice_drafts import PrepareVoiceDraft, VoiceDraftCoordinator
from fish_memory.companion.draft_tts import DraftTtsSession, PromotablePcmSink
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.voice_drafts import VoiceDraftTurnController
from tests.answer.test_voice_drafts import FakePrepared

from .test_draft_tts import (
    FakeDraftClient,
    FakeDraftOutput,
    FakeSink,
    RecordingDiagnostics,
    load_configuration,
    wait_for_condition,
)
from .test_voice_drafts import (
    FISH_ID,
    SESSION_ID,
    FakeVoiceDraftClient,
    ImmediateDecoder,
)

TEXT = "第一段。"


class ServerDraftClient:
    def __init__(self, server: VoiceDraftCoordinator, *, block_sse: bool) -> None:
        self.server = server
        self.sse_entered = asyncio.Event()
        self.sse_release = asyncio.Event()
        self.sse_closed = asyncio.Event()
        self.events: list[str] = []
        self.cancel_calls = 0
        if not block_sse:
            self.sse_release.set()

    async def upsert_voice_draft(self, key, *, revision: int, content: str) -> bool:
        await self.server.upsert(key, revision=revision, content=content)
        return True

    async def cancel_voice_draft(self, key, *, revision: int, reason: str) -> bool:
        del reason
        self.cancel_calls += 1
        await self.server.cancel(key, revision=revision)
        return True

    async def iter_voice_draft_events(
        self, key, *, revision: int
    ) -> AsyncIterator[dict[str, object]]:
        self.sse_entered.set()
        await self.sse_release.wait()
        try:
            async for event in self.server.iter_events(key, revision=revision):
                self.events.append(event.event)
                yield {"event": event.event, "data": event.data}
        finally:
            self.sse_closed.set()


class DelayedStartOutput(FakeDraftOutput):
    def __init__(self, sink: PromotablePcmSink, *, blocked: bool) -> None:
        super().__init__(sink)
        self.start_entered = asyncio.Event()
        self.start_release = asyncio.Event()
        if not blocked:
            self.start_release.set()

    async def start(self) -> None:
        self.start_entered.set()
        await self.start_release.wait()
        await super().start()


class LegacyDraftTts:
    """Preserve the old controller contract for a paired race reproduction."""

    def __init__(self, draft: DraftTtsSession) -> None:
        self.draft = draft

    async def start(self, key, *, revision: int) -> bool:
        return await self.draft.start(key, revision=revision)

    async def cancel(self, key, *, through_revision: int) -> None:
        await self.draft.cancel(key, through_revision=through_revision)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["provider_start", "sse_subscription"])
@pytest.mark.parametrize("commit_notification", [False, True])
async def test_committed_provider_survives_late_empty_private_sse(
    phase: str, commit_notification: bool
) -> None:
    prepared = FakePrepared(1)
    await prepared.sink.append(TEXT)
    await prepared.sink.flush()
    prepared.ready.set()

    async def prepare(*args: object) -> FakePrepared:
        del args
        return prepared

    server = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare), ttl_seconds=5
    )
    client = ServerDraftClient(server, block_sse=phase == "sse_subscription")
    live = FakeSink()
    arbiter = PlaybackArbiter()
    diagnostics = RecordingDiagnostics()
    outputs: list[DelayedStartOutput] = []

    def factory(sink: PromotablePcmSink) -> DelayedStartOutput:
        output = DelayedStartOutput(sink, blocked=phase == "provider_start")
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=factory,
        load_configuration=load_configuration,
        playback_arbiter=arbiter,
        diagnostics=diagnostics,
    )
    controller = VoiceDraftTurnController(
        client=client,
        decoder=ImmediateDecoder("候选问题"),
        draft_tts=draft if commit_notification else LegacyDraftTts(draft),
        diagnostics=diagnostics,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    candidate = controller.candidate(owner, pcm=bytes(320))
    assert candidate is not None
    lease = None
    try:
        await wait_for_condition(lambda: bool(outputs))
        output = outputs[0]
        await output.start_entered.wait()
        if phase == "sse_subscription":
            await asyncio.wait_for(client.sse_entered.wait(), timeout=0.5)
        assert await controller.wait_for_registration(owner)
        submission = controller.commit(owner, content="候选问题")
        lease = await server.finalize(
            submission.key,
            revision=submission.revision,
            promote_revision=submission.promote_revision,
            content=submission.content,
        )
        assert lease is not None
        assert not prepared.cancelled
        output.start_release.set()
        client.sse_release.set()
        await candidate
        await asyncio.wait_for(client.sse_closed.wait(), timeout=0.5)

        assert client.events == []
        assert client.cancel_calls == 0
        assert live.calls == []
        if not commit_notification:
            await wait_for_condition(lambda: ("interrupt", None) in output.calls)
        playback_owner = await arbiter.acquire("answer", UUID(int=901))
        promotion = await draft.promote(
            owner.key,
            revision=submission.promote_revision,
            content=TEXT,
            flush_offset=len(TEXT),
            playback_owner=playback_owner,
        )

        if commit_notification:
            assert promotion is not None
            assert promotion.submitted_offset == 0
            assert ("interrupt", None) not in output.calls
            await promotion.output.append(TEXT)
            assert output.calls.count(("start", None)) == 1
            assert output.calls.count(("append", TEXT)) == 1
            assert len([call for call in live.calls if call[0] == "append"]) == 1
            assert "voice_draft_late_event_dropped" not in [
                name for name, _ in diagnostics.events
            ]
        else:
            assert promotion is None
            assert live.calls == []
        assert not prepared.cancelled
    finally:
        for output in outputs:
            output.start_release.set()
        client.sse_release.set()
        await controller.close()
        await draft.close()
        if lease is not None:
            await lease.prepared.cancel()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["resume", "rejected_commit", "close"])
async def test_handoff_never_overrides_real_cancellation(
    action: str,
) -> None:
    client = FakeVoiceDraftClient()
    events = FakeDraftClient()
    live = FakeSink()
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
    )
    controller = VoiceDraftTurnController(
        client=client, decoder=ImmediateDecoder(), draft_tts=draft
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    candidate = controller.candidate(owner, pcm=bytes(320))
    assert candidate is not None
    try:
        await wait_for_condition(lambda: bool(outputs))
        old = outputs[0]
        await old.start_entered.wait()
        if action == "resume":
            controller.resume(owner)
        else:
            submission = controller.commit(owner, content="候选问题")
            if action == "rejected_commit":
                await controller.cancel_submission(submission, reason="binding_changed")
            else:
                await controller.close()
        await asyncio.wait_for(controller.wait_idle(), timeout=0.5)
        assert ("interrupt", None) in old.calls
        assert live.calls == []
        for output in outputs:
            await output.sink.append(b"late", start_offset=0, end_offset=4)
        assert live.calls == []
        assert (
            await draft.promote(owner.key, revision=1, content=TEXT, flush_offset=4)
            is None
        )
    finally:
        for output in outputs:
            output.start_release.set()
        await controller.close()
        await draft.close()
