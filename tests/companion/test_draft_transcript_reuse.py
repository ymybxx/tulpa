import asyncio
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from uuid import UUID

import pytest

from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.audio_processing import ProcessedAudioFrame
from fish_memory.companion.client import CompanionAskResult, CompanionReplyContext
from fish_memory.companion.owner_voice_gate import OwnerVoiceGate
from fish_memory.companion.processor import CompanionAudioProcessor
from fish_memory.companion.session import CompanionSession, VoiceReplyContext
from fish_memory.companion.voice_drafts import (
    SpeculativeSilenceGate,
    VoiceDraftTurnController,
)

from .test_processor import (
    ANSWER_ID,
    FISH_ID,
    FakeApi,
    FakeDraftTransport,
    FakeDraftVad,
    FakeFrameProcessor,
    FakeKws,
    FakeTts,
    FrameAsr,
    RecordingDiagnostics,
)

SILENCE = bytes(320)
SPEECH = b"a" * 320


class BufferedAsr(FrameAsr):
    def __init__(self) -> None:
        super().__init__("正式识别", partial_text=None)
        self.consumed: list[bytes] = []

    def recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate:
        self.snapshot_calls += 1
        return AsrUpdate("正式识别", is_final=True, event="Speech")

    def consume_snapshot(self, pcm: bytes) -> bool:
        if not self.snapshot_utterance().startswith(pcm):
            return False
        self.consumed.append(pcm)
        del self.accepted[: len(pcm) // 320]
        return True


class CandidateDecoder:
    def __init__(self, event: str = "Speech") -> None:
        self.event = event
        self.calls = 0
        self.inputs: list[bytes] = []

    def recognize(self, pcm: bytes) -> str:
        return self.recognize_update(pcm).text

    def recognize_update(self, pcm: bytes) -> AsrUpdate:
        self.calls += 1
        self.inputs.append(pcm)
        return AsrUpdate("预识别结果", is_final=True, event=self.event)


class SharedVadFrameProcessor(FakeFrameProcessor):
    def __init__(self, primary: list[bool], candidate: list[bool]) -> None:
        super().__init__(primary)
        self._candidate_values = iter(candidate)

    def process(self, *, capture_pcm: bytes, render_pcm: bytes) -> ProcessedAudioFrame:
        frame = super().process(capture_pcm=capture_pcm, render_pcm=render_pcm)
        return replace(frame, candidate_voice_detected=next(self._candidate_values))


class PausingSmartTurn:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def predict_pcm16(self, pcm: bytes) -> SimpleNamespace:
        self.started.set()
        assert self.release.wait(1)
        return SimpleNamespace(complete=True, probability=1.0, inference_ms=1.0)


class PausingCandidateDecoder(CandidateDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def recognize_update(self, pcm: bytes) -> AsrUpdate:
        self.started.set()
        assert self.release.wait(1)
        return super().recognize_update(pcm)


class OwnerVerifier:
    def score_pcm(self, pcm: bytes) -> float:
        return 1.0


async def wait_for_commit(c: SimpleNamespace) -> None:
    for _ in range(100):
        if c.api.voice_commit_calls:
            return
        await asyncio.sleep(0.001)
    pytest.fail("voice commit did not arrive")


@asynccontextmanager
async def case(
    primary: list[bool],
    fast: list[bool],
    *,
    silence_frames: int = 1,
    event: str = "Speech",
    smart_turn: PausingSmartTurn | None = None,
    owner: bool = False,
    capture: list[bool] | None = None,
    candidate_decoder: CandidateDecoder | None = None,
    draft_transport: FakeDraftTransport | None = None,
    api_client: FakeApi | None = None,
    candidate: list[bool] | None = None,
) -> AsyncIterator[SimpleNamespace]:
    api = api_client or FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    decoder = candidate_decoder or CandidateDecoder(event)
    diagnostics = RecordingDiagnostics()
    controller = VoiceDraftTurnController(
        client=draft_transport or FakeDraftTransport(),
        decoder=decoder,
        diagnostics=diagnostics,
    )
    asr = BufferedAsr()
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=(
            FakeFrameProcessor(primary)
            if candidate is None
            else SharedVadFrameProcessor(primary, candidate)
        ),
        smart_turn=smart_turn,
        voice_draft_vad=FakeDraftVad(fast),
        voice_draft_gate=SpeculativeSilenceGate(silence_frames=silence_frames),
        voice_draft_controller=controller,
        voice_draft_capture_vad=FakeDraftVad(capture or []) if owner else None,
        owner_voice_gate=(
            OwnerVoiceGate(OwnerVerifier(), check_speech_ms=10, minimum_speech_ms=10)
            if owner
            else None
        ),
        speech_start_frames=1,
        speech_end_frames=1,
        diagnostics=diagnostics,
    )
    try:
        yield SimpleNamespace(
            api=api,
            session=session,
            asr=asr,
            decoder=decoder,
            controller=controller,
            processor=processor,
            diagnostics=diagnostics,
        )
    finally:
        if smart_turn is not None:
            smart_turn.release.set()
        if isinstance(decoder, PausingCandidateDecoder):
            decoder.release.set()
        await processor.close()
        await controller.close()
        await api.answer_events.put(
            {
                "event": "answer.completed",
                "data": {"messageId": str(ANSWER_ID), "end": False},
            }
        )
        await asyncio.wait_for(session.wait_for_idle(), 1)


@pytest.mark.asyncio
async def test_200ms_candidate_reuses_after_350ms_silence_without_second_asr() -> None:
    async with case(
        [True] * 54 + [False], [True] * 20 + [False] * 35, silence_frames=20
    ) as c:
        for pcm in [SPEECH] * 20 + [SILENCE] * 20:
            await c.processor.process_frame(pcm, SILENCE)
        await c.controller.wait_idle()
        assert c.api.voice_commit_calls == []
        for _ in range(15):
            await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)

        assert c.decoder.calls == 1
        assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        assert c.api.voice_commit_calls[0][3] == "预识别结果"
        assert c.asr.consumed == [SPEECH * 20 + SILENCE * 34]
        assert c.asr.snapshot_utterance() == b""
        assert "asr_draft_reused" in [e[0] for e in c.diagnostics.events]


@pytest.mark.asyncio
async def test_shared_vad_starts_draft_while_legacy_vad_stays_active() -> None:
    async with case(
        [True] * 54 + [False],
        [True] * 55,
        silence_frames=20,
        candidate=[True] * 20 + [False] * 35,
    ) as c:
        for pcm in [SPEECH] * 20 + [SILENCE] * 20:
            await c.processor.process_frame(pcm, SILENCE)
        await c.controller.wait_idle()
        assert c.decoder.calls == 1
        assert c.api.voice_commit_calls == []

        for _ in range(15):
            await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)
        assert c.api.voice_commit_calls[0][3] == "预识别结果"
        assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        assert c.processor._voice_draft_vad.accepted == []


@pytest.mark.asyncio
async def test_shared_vad_continuation_restarts_with_complete_pcm() -> None:
    async with case(
        [True] * 5 + [False],
        [False] * 6,
        candidate=[True, False, True, True, False, False],
    ) as c:
        for pcm in [SPEECH, SILENCE]:
            await c.processor.process_frame(pcm, SILENCE)
        await c.controller.wait_idle()
        for pcm in [SPEECH, SPEECH, SILENCE]:
            await c.processor.process_frame(pcm, SILENCE)
        await c.controller.wait_idle()
        await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)

        assert c.decoder.inputs == [
            SPEECH + SILENCE,
            SPEECH + SILENCE + SPEECH * 2 + SILENCE,
        ]
        assert c.api.voice_commit_calls[0][2] == 3
        assert c.asr.snapshot_calls == 0
        resumed = [
            kwargs["metadata"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_resumed"
        ]
        assert len(resumed) == 1
        assert resumed[0]["reason"] == "primary_inference"
        assert c.processor._voice_draft_vad.accepted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_age_ms", [39.505, 59.327])
async def test_completed_draft_reuse_is_not_a_new_queue_budgeted_decode(
    queue_age_ms: float,
) -> None:
    async with case([True, True, False], [True, False, False]) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await c.controller.wait_idle()
        # These ages occurred in the packaged reproduction after the candidate
        # was decoded. No speech or missing PCM invalidates its frozen prefix.
        c.processor._frame_queue_age_ms = lambda: queue_age_ms
        c.processor._frame_queue_depth = lambda: 0
        await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)

        assert c.decoder.calls == 1
        assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        assert c.api.voice_commit_calls[0][3] == "预识别结果"
        assert "asr_draft_reused" in [name for name, _ in c.diagnostics.events]


@pytest.mark.asyncio
async def test_dropped_audio_still_invalidates_completed_draft_reuse() -> None:
    async with case([True, True, False], [True, False, False]) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await c.controller.wait_idle()
        c.processor._dropped_frames = lambda: 1
        await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)

        assert c.asr.snapshot_calls == 1
        assert c.api.voice_commit_calls[0][3] == "正式识别"
        reasons = [
            kwargs["metadata"]["reason"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_asr_reuse_missed"
        ]
        assert "audio_frame_drop" in reasons


@pytest.mark.asyncio
async def test_gate_summary_explains_a_turn_without_enough_fast_vad_silence() -> None:
    async with case(
        [True] * 6 + [False],
        [True, False, True, False, True, False, False],
        silence_frames=3,
    ) as c:
        for _ in range(7):
            await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)
        await c.processor.close()
        summaries = [
            kwargs["metadata"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_gate_summary"
        ]
        assert len(summaries) == 1
        assert summaries[0]["reason"] == "speech_end"
        assert summaries[0]["gateFrames"] == 6
        assert summaries[0]["gateSilentFrames"] == 1
        assert summaries[0]["gateLongestSilenceFrames"] == 1
        assert summaries[0]["gateRequiredSilenceFrames"] == 3
        assert summaries[0]["gateCandidateCount"] == 0
        assert summaries[0]["gateResumedCount"] == 0
        assert c.decoder.calls == 0
        assert "voice_draft_candidate_triggered" not in [
            name for name, _ in c.diagnostics.events
        ]


@pytest.mark.asyncio
async def test_endpoint_joins_running_candidate_without_another_decode() -> None:
    decoder = PausingCandidateDecoder()
    async with case(
        [True, True, False],
        [True, False, False],
        candidate_decoder=decoder,
    ) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(decoder.started.wait, 1)
        finishing = asyncio.create_task(c.processor.process_frame(SILENCE, SILENCE))
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            assert c.asr.finish_calls == c.asr.snapshot_calls == 0
            assert c.api.voice_commit_calls == []
            decoder.release.set()
            await finishing
            await wait_for_commit(c)
            assert decoder.calls == 1
            assert c.asr.finish_calls == c.asr.snapshot_calls == 0
            assert c.api.voice_commit_calls[0][3] == "预识别结果"
        finally:
            decoder.release.set()
            await finishing


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse_transcript", [False, True])
async def test_resumption_during_registration_keeps_new_revision_and_frozen_reply(
    reuse_transcript: bool,
) -> None:
    class ReplyApi(FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.contexts: list[CompanionReplyContext | None] = []

        async def commit_voice_turn(
            self,
            key: object,
            *,
            revision: int,
            promote_revision: int,
            content: str,
            reply_context: CompanionReplyContext | None = None,
        ) -> CompanionAskResult:
            self.contexts.append(reply_context)
            return await super().commit_voice_turn(
                key,
                revision=revision,
                promote_revision=promote_revision,
                content=content,
                reply_context=reply_context,
            )

    class PendingFirstUpload(FakeDraftTransport):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.second_registered = asyncio.Event()
            self.contexts: list[CompanionReplyContext | None] = []

        async def upsert_voice_draft(
            self,
            key: object,
            *,
            revision: int,
            content: str,
            reply_context: CompanionReplyContext | None = None,
            **_metadata: object,
        ) -> bool:
            self.upserts.append((key, revision, content))
            self.contexts.append(reply_context)
            if revision == 1:
                self.started.set()
                await self.release.wait()
            else:
                self.second_registered.set()
            return True

    decoder = PausingCandidateDecoder()
    transport = PendingFirstUpload()
    api = ReplyApi()
    async with case(
        [True, True, False, True, True, False],
        [True, False, False, True, False, False],
        candidate_decoder=decoder,
        draft_transport=transport,
        api_client=api,
    ) as c:
        conversation_id = c.session.voice_conversation_id
        assert conversation_id is not None
        session_id = c.session.session_id
        assert session_id is not None
        context = VoiceReplyContext(
            fish_id=FISH_ID,
            session_id=session_id,
            voice_conversation_id=conversation_id,
            subject_kind="interaction",
            subject_id=UUID(int=902),
            source_message_id=UUID(int=903),
            expires_at=time.monotonic() + 60,
        )
        c.session._reply_context = context
        await c.processor.process_frame(SPEECH, SILENCE)
        frozen = c.processor._frame_frozen_turn
        assert frozen is not None and frozen.reply_context == context
        reservation_id = frozen.context_reservation_id
        assert reservation_id is not None
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(decoder.started.wait, 1)
        if not reuse_transcript:
            decoder.release.set()
            await asyncio.wait_for(transport.started.wait(), 1)
            c.processor._invalidate_voice_draft_transcript()
        finishing = asyncio.create_task(c.processor.process_frame(SILENCE, SILENCE))
        try:
            decoder.release.set()
            await asyncio.wait_for(transport.started.wait(), 1)
            for _ in range(20):
                await asyncio.sleep(0)
            assert c.processor._frame_frozen_turn is frozen
            assert c.asr.consumed == []
            assert c.api.voice_commit_calls == []

            # A new START revokes the endpoint waiting on the old PUT.
            await c.processor.process_frame(SPEECH, SILENCE)
            await asyncio.wait_for(finishing, 1)
            assert c.api.voice_commit_calls == []
            assert c.processor._frame_frozen_turn is frozen
            assert c.session._reply_context_reservation_id == reservation_id
            assert c.asr.consumed == []

            await c.processor.process_frame(SILENCE, SILENCE)
            await asyncio.wait_for(transport.second_registered.wait(), 1)
            transport.release.set()
            await c.controller.wait_idle()
            active = c.controller.current_owner(c.processor._voice_draft_owner)
            assert active is not None and active.revision == 3
            await c.processor.process_frame(SILENCE, SILENCE)
            await wait_for_commit(c)

            assert transport.contexts == [context.client_context] * 2
            assert api.contexts == [context.client_context]
            assert api.voice_commit_calls[0][0].session_id == context.session_id
            assert api.voice_commit_calls[0][2] == 3
            assert decoder.inputs[1].startswith(decoder.inputs[0])
            assert len(decoder.inputs[1]) > len(decoder.inputs[0])
            assert c.asr.consumed == [decoder.inputs[1]]
            assert c.processor._frame_frozen_turn is None
            assert c.session._reply_context_reservation_id is None
        finally:
            decoder.release.set()
            transport.release.set()
            await finishing


@pytest.mark.asyncio
async def test_abandon_releases_early_frozen_reply_reservation() -> None:
    async with case([True], [True]) as c:
        conversation_id = c.session.voice_conversation_id
        assert conversation_id is not None
        c.session._reply_context = VoiceReplyContext(
            FISH_ID,
            UUID(int=901),
            conversation_id,
            "interaction",
            UUID(int=902),
            UUID(int=903),
            time.monotonic() + 60,
        )
        await c.processor.process_frame(SPEECH, SILENCE)
        assert c.session._reply_context_reservation_id is not None
        await c.processor.close()
        assert c.session._reply_context_reservation_id is None
        assert c.processor._frame_frozen_turn is None
        assert c.api.voice_commit_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("upload_accepted", [False, True])
async def test_shared_asr_endpoint_keeps_revision_while_draft_upload_is_pending(
    upload_accepted: bool,
) -> None:
    class BlockingTransport(FakeDraftTransport):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self, key: object, *, revision: int, content: str, **_metadata: object
        ) -> bool:
            self.upserts.append((key, revision, content))
            self.started.set()
            await self.release.wait()
            return upload_accepted

    decoder = PausingCandidateDecoder()
    transport = BlockingTransport()
    async with case(
        [True, True, False],
        [True, False, False],
        candidate_decoder=decoder,
        draft_transport=transport,
    ) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(decoder.started.wait, 1)
        finishing = asyncio.create_task(c.processor.process_frame(SILENCE, SILENCE))
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            assert not finishing.done()
            decoder.release.set()
            await asyncio.wait_for(transport.started.wait(), 1)
            for _ in range(20):
                await asyncio.sleep(0)
            assert not finishing.done()
            assert c.api.voice_commit_calls == []
            assert not transport.release.is_set()
            transport.release.set()
            await asyncio.wait_for(finishing, 1)
            await wait_for_commit(c)

            assert decoder.calls == 1
            assert c.asr.finish_calls == c.asr.snapshot_calls == 0
            key, revision, promote_revision, content = c.api.voice_commit_calls[0]
            assert (key, 1, content) == transport.upserts[0]
            assert revision == 2
            assert promote_revision == int(upload_accepted)

            # A late acknowledgement/failure cannot restore retired ownership.
            await c.controller.wait_idle()
            assert c.controller._promote_revision == 0
            assert len(c.api.voice_commit_calls) == 1
        finally:
            decoder.release.set()
            transport.release.set()
            await finishing


@pytest.mark.asyncio
async def test_resumed_speech_before_end_requires_fresh_recognition() -> None:
    async with case([True, True, True, False], [True, False, True, False]) as c:
        for pcm in [SPEECH, SILENCE]:
            await c.processor.process_frame(pcm, SILENCE)
        await c.controller.wait_idle()
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)
        assert c.api.voice_commit_calls[0][3] == "正式识别"
        assert c.asr.snapshot_calls == 1
        assert c.asr.finish_calls == 0
        resumed = [
            kwargs["metadata"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_resumed"
        ]
        assert len(resumed) == 1
        assert resumed[0]["reason"] == "fast_vad"
        assert resumed[0]["attemptRevision"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("short_voice_on_end", [False, True])
@pytest.mark.parametrize("shared_candidate", [False, True])
async def test_fast_vad_observes_end_and_wait_before_formal_start(
    short_voice_on_end: bool,
    shared_candidate: bool,
) -> None:
    smart = PausingSmartTurn()
    fast = [True, False, short_voice_on_end, not short_voice_on_end]
    async with case(
        [True, True, False, False],
        [False] * 4 if shared_candidate else fast,
        smart_turn=smart,
        candidate=fast if shared_candidate else None,
    ) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await c.controller.wait_idle()
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(smart.started.wait, 1)
        await c.processor.process_frame(SPEECH, SILENCE)
        smart.release.set()
        await asyncio.wait_for(c.processor._frame_finish_task, 1)
        assert c.api.voice_commit_calls[0][3] == "正式识别"
        assert c.asr.snapshot_calls == 1
        assert "asr_draft_reused" not in [e[0] for e in c.diagnostics.events]
        reason = "primary_inference_tail" if shared_candidate else "fast_vad_tail"
        assert reason in [
            kwargs["metadata"]["reason"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_asr_reuse_missed"
        ]


@pytest.mark.asyncio
async def test_different_asr_tail_track_is_not_assumed_silent() -> None:
    async with case([True, True, False], [True, False, False, False]) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await c.controller.wait_idle()
        await c.processor._process_clean_frame(
            ProcessedAudioFrame(SILENCE, True),
            capture_pcm=SPEECH,
            frame_index=2,
        )
        await c.processor.process_frame(SILENCE, SILENCE)
        await wait_for_commit(c)
        assert c.api.voice_commit_calls[0][3] == "正式识别"
        assert c.asr.snapshot_calls == 1
        assert c.asr.finish_calls == 0
        assert "pcm_track_changed" in [
            kwargs["metadata"]["reason"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_asr_reuse_missed"
        ]


@pytest.mark.asyncio
async def test_reused_non_speech_event_still_rejects_the_turn() -> None:
    async with case([True, True, False], [True, False, False], event="Cough") as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await c.controller.wait_idle()
        await c.processor.process_frame(SILENCE, SILENCE)
        assert c.asr.finish_calls == c.asr.snapshot_calls == 0
        assert c.api.voice_commit_calls == []
        assert "asr_final_rejected" in [e[0] for e in c.diagnostics.events]


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_voice", [False, True])
@pytest.mark.parametrize("shared_candidate", [False, True])
async def test_owner_capture_is_observed_before_zero_substitution(
    capture_voice: bool,
    shared_candidate: bool,
) -> None:
    smart = PausingSmartTurn()
    async with case(
        [True, True, False, False],
        [True] * 4 if shared_candidate else [True, False, False, False],
        smart_turn=smart,
        owner=True,
        capture=[True] * 4 if shared_candidate else [True, False, False, capture_voice],
        candidate=[True, False, False, capture_voice] if shared_candidate else None,
    ) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await c.controller.wait_idle()
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(smart.started.wait, 1)
        await c.controller.wait_idle()
        assert c.processor._voice_draft_transcript_reusable
        await c.processor.process_frame(SPEECH, SILENCE)
        smart.release.set()
        await asyncio.wait_for(c.processor._frame_finish_task, 1)
        expected_text = "正式识别" if capture_voice else "预识别结果"
        assert c.api.voice_commit_calls[0][3] == expected_text
        assert c.asr.snapshot_calls == int(capture_voice)
        if shared_candidate:
            assert c.processor._voice_draft_vad.accepted == []
            assert c.processor._voice_draft_capture_vad.accepted == []


@pytest.mark.asyncio
async def test_frame_still_being_classified_prevents_early_cache_commit() -> None:
    smart = PausingSmartTurn()
    started = threading.Event()
    release = threading.Event()
    async with case(
        [True, True, False, False], [True, False, False, True], smart_turn=smart
    ) as c:
        delegate = c.processor._frame_processor

        class BlockingFrameProcessor:
            def process(self, *, capture_pcm: bytes, render_pcm: bytes) -> object:
                if capture_pcm == b"b" * 320:
                    started.set()
                    assert release.wait(1)
                return delegate.process(capture_pcm=capture_pcm, render_pcm=render_pcm)

        c.processor._frame_processor = BlockingFrameProcessor()
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        await c.controller.wait_idle()
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(smart.started.wait, 1)
        pending = asyncio.create_task(c.processor.process_frame(b"b" * 320, SILENCE))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            smart.release.set()
            await asyncio.wait_for(c.processor._frame_finish_task, 1)
            assert c.api.voice_commit_calls[0][3] == "正式识别"
            assert c.asr.snapshot_calls == 1
            assert "asr_draft_reused" not in [e[0] for e in c.diagnostics.events]
            assert "frame_in_flight" in [
                kwargs["metadata"]["reason"]
                for name, kwargs in c.diagnostics.events
                if name == "voice_draft_asr_reuse_missed"
            ]
        finally:
            release.set()
            await pending


@pytest.mark.asyncio
async def test_short_voice_wakes_wait_before_old_decode_finishes() -> None:
    decoder = PausingCandidateDecoder()
    smart = PausingSmartTurn()
    smart.release.set()
    async with case(
        [True, True, False, False],
        [True, False, False, True],
        smart_turn=smart,
        candidate_decoder=decoder,
    ) as c:
        await c.processor.process_frame(SPEECH, SILENCE)
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(decoder.started.wait, 1)
        await c.processor.process_frame(SILENCE, SILENCE)
        assert await asyncio.to_thread(smart.started.wait, 1)
        for _ in range(20):
            await asyncio.sleep(0)
        assert c.asr.snapshot_calls == 0
        await c.processor.process_frame(SPEECH, SILENCE)
        await asyncio.wait_for(c.processor._frame_finish_task, 0.2)
        assert not decoder.release.is_set()
        assert c.asr.snapshot_calls == 1
        assert c.api.voice_commit_calls[0][3] == "正式识别"
        decoder.release.set()
        await c.controller.wait_idle()
        assert len(c.api.voice_commit_calls) == 1
