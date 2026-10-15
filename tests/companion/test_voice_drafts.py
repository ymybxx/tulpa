import asyncio
import gc
import threading
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import pytest

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.client import CompanionReplyContext
from fish_memory.companion.voice_drafts import (
    CandidateEvent,
    SpeculativeSilenceGate,
    VoiceDraftOwner,
    VoiceDraftSubmission,
    VoiceDraftTurnController,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_candidate_gate_emits_only_after_exactly_200_ms_of_turn_silence() -> None:
    gate = SpeculativeSilenceGate(silence_frames=20)

    assert gate.accept(False) is CandidateEvent.NONE
    gate.open_turn()
    assert [gate.accept(False) for _ in range(19)] == [CandidateEvent.NONE] * 19
    assert gate.accept(False) is CandidateEvent.CANDIDATE
    assert gate.accept(False) is CandidateEvent.NONE


def test_candidate_gate_resets_partial_silence_and_reports_resumed_speech() -> None:
    gate = SpeculativeSilenceGate(silence_frames=20)
    gate.open_turn()

    for _ in range(12):
        assert gate.accept(False) is CandidateEvent.NONE
    assert gate.accept(True) is CandidateEvent.NONE
    assert [gate.accept(False) for _ in range(19)] == [CandidateEvent.NONE] * 19
    assert gate.accept(False) is CandidateEvent.CANDIDATE
    assert gate.accept(True) is CandidateEvent.RESUMED
    assert gate.accept(True) is CandidateEvent.NONE

    gate.close_turn()
    assert gate.accept(False) is CandidateEvent.NONE


@dataclass(frozen=True, slots=True)
class DraftCall:
    key: VoiceDraftKey
    revision: int
    content: str
    promote_revision: int | None = None


class FakeVoiceDraftClient:
    def __init__(self) -> None:
        self.upserts: list[DraftCall] = []
        self.cancels: list[tuple[VoiceDraftKey, int, str]] = []
        self.commits: list[DraftCall] = []

    async def upsert_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        **_metadata: object,
    ) -> bool:
        self.upserts.append(DraftCall(key, revision, content))
        return True

    async def cancel_voice_draft(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        reason: str,
    ) -> bool:
        self.cancels.append((key, revision, reason))
        return True


class FakeDraftTts:
    def __init__(self) -> None:
        self.starts: list[tuple[VoiceDraftKey, int]] = []
        self.cancels: list[tuple[VoiceDraftKey, int]] = []

    async def start(self, key: VoiceDraftKey, *, revision: int) -> bool:
        self.starts.append((key, revision))
        return True

    async def cancel(
        self,
        key: VoiceDraftKey,
        *,
        through_revision: int,
    ) -> None:
        self.cancels.append((key, through_revision))


class ImmediateDecoder:
    def __init__(self, text: str = "候选问题") -> None:
        self.text = text

    def recognize(self, pcm: bytes) -> str:
        assert type(pcm) is bytes
        return self.text


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [b"", bytes(320)])
async def test_cached_transcript_covers_candidate_pcm_prefix(tail: bytes) -> None:
    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(),
        decoder=ImmediateDecoder(" 完整问题 "),
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = b"a" * 320

    assert controller.cached_transcript(owner, pcm) is None
    candidate = await controller.candidate(owner, pcm=pcm)
    assert candidate is not None
    expected = AsrUpdate(text="完整问题", is_final=True)
    assert controller.cached_transcript(owner, pcm + tail) == expected
    assert controller.cached_transcript(candidate, pcm + tail) == expected
    assert controller.cached_transcript(owner, pcm[:-2]) is None
    assert controller.cached_transcript(owner, b"b" + pcm[1:]) is None
    await controller.close()


@pytest.mark.asyncio
async def test_cached_transcript_preserves_decoder_event_metadata() -> None:
    class MetadataDecoder(ImmediateDecoder):
        def recognize_update(self, pcm: bytes) -> AsrUpdate:
            assert pcm == bytes(320)
            return AsrUpdate(text=" 识别文本 ", is_final=True, event="Cough")

        def recognize(self, pcm: bytes) -> str:
            raise AssertionError("metadata decoder must be preferred")

    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(),
        decoder=MetadataDecoder(),
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    await controller.candidate(owner, pcm=bytes(320))

    assert controller.cached_transcript(owner, bytes(320)) == AsrUpdate(
        text="识别文本", is_final=True, event="Cough"
    )
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transcript", ["嗯。", "哼！", "咳。"])
@pytest.mark.parametrize(
    "reply_context", [None, CompanionReplyContext("run_result", UUID(int=4))]
)
async def test_low_evidence_candidate_keeps_transcript_without_put_or_tts(
    transcript: str,
    reply_context: CompanionReplyContext | None,
) -> None:
    class LowEvidenceDecoder(ImmediateDecoder):
        def recognize_update(self, pcm: bytes) -> AsrUpdate:
            assert pcm == bytes(320)
            return AsrUpdate(text=transcript, is_final=True, event="Speech")

        def recognize(self, pcm: bytes) -> str:
            raise AssertionError("recognize_update must preserve event metadata")

    client = FakeVoiceDraftClient()
    draft_tts = FakeDraftTts()
    diagnostics = RecordingDiagnostics()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=LowEvidenceDecoder(),
        draft_tts=draft_tts,
        diagnostics=diagnostics,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
        reply_context=reply_context,
    )
    pcm = bytes(320)

    candidate = controller.candidate(owner, pcm=pcm)
    assert candidate is not None
    assert await candidate is None

    expected = AsrUpdate(text=transcript, is_final=True, event="Speech")
    assert controller.cached_transcript(owner, pcm) == expected
    assert await controller.wait_for_transcript(owner, pcm) == expected
    assert not await controller.wait_for_registration(owner)
    assert client.upserts == []
    assert draft_tts.starts == []
    assert [
        kwargs["metadata"]["reason"]
        for name, kwargs in diagnostics.events
        if name == "voice_draft_fallback"
    ] == ["nonlexical_vocalization"]

    submission = controller.commit(owner, content=transcript)
    assert submission.promote_revision == 0
    assert submission.draft_request_started is False
    await controller.cancel_submission(submission, reason="rejected")
    assert client.cancels == []
    assert draft_tts.cancels == []
    await controller.close()


@pytest.mark.asyncio
async def test_interaction_context_allows_nonlexical_candidate_registration() -> None:
    class ReplyDecoder(ImmediateDecoder):
        def recognize_update(self, pcm: bytes) -> AsrUpdate:
            assert pcm == bytes(320)
            return AsrUpdate(text="嗯。", is_final=True, event="Speech")

    client = FakeVoiceDraftClient()
    draft_tts = FakeDraftTts()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=ReplyDecoder(),
        draft_tts=draft_tts,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
        reply_context=CompanionReplyContext("interaction", UUID(int=3)),
    )

    accepted = await controller.candidate(owner, pcm=bytes(320))

    assert accepted is not None
    assert client.upserts == [DraftCall(owner.key, 1, "嗯。")]
    assert draft_tts.starts == [(owner.key, 1)]
    await controller.close()


@pytest.mark.asyncio
async def test_explicit_cough_event_is_blocked_even_with_reply_context() -> None:
    class CoughDecoder(ImmediateDecoder):
        def recognize_update(self, pcm: bytes) -> AsrUpdate:
            assert pcm == bytes(320)
            return AsrUpdate(text="好的。", is_final=True, event="Cough")

    client = FakeVoiceDraftClient()
    draft_tts = FakeDraftTts()
    diagnostics = RecordingDiagnostics()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=CoughDecoder(),
        draft_tts=draft_tts,
        diagnostics=diagnostics,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
        reply_context=CompanionReplyContext("run_result", UUID(int=4)),
    )

    candidate = controller.candidate(owner, pcm=bytes(320))
    assert candidate is not None
    assert await candidate is None

    assert client.upserts == []
    assert draft_tts.starts == []
    assert [
        kwargs["metadata"]["reason"]
        for name, kwargs in diagnostics.events
        if name == "voice_draft_fallback"
    ] == ["non_speech_event"]
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["resume", "abandon", "close"])
async def test_never_registered_low_evidence_revision_never_sends_cancel(
    action: str,
) -> None:
    class LowEvidenceDecoder(ImmediateDecoder):
        def recognize_update(self, pcm: bytes) -> AsrUpdate:
            return AsrUpdate(text="嗯。", is_final=True, event="Speech")

    client = FakeVoiceDraftClient()
    draft_tts = FakeDraftTts()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=LowEvidenceDecoder(),
        draft_tts=draft_tts,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    assert await controller.candidate(owner, pcm=bytes(320)) is None

    if action == "resume":
        controller.resume(owner)
    elif action == "abandon":
        controller.abandon(owner, reason="rejected")
    else:
        await controller.close()
    await controller.wait_idle()

    assert client.upserts == []
    assert client.cancels == []
    assert draft_tts.starts == []
    assert draft_tts.cancels == []
    await controller.close()


@pytest.mark.asyncio
async def test_substantive_candidate_after_filtered_pause_uses_new_revision() -> None:
    class SequenceDecoder(ImmediateDecoder):
        def __init__(self) -> None:
            super().__init__()
            self.updates = iter(
                [
                    AsrUpdate(text="嗯。", is_final=True, event="Speech"),
                    AsrUpdate(text="停止。", is_final=True, event="Speech"),
                ]
            )

        def recognize_update(self, pcm: bytes) -> AsrUpdate:
            return next(self.updates)

    client = FakeVoiceDraftClient()
    controller = VoiceDraftTurnController(client=client, decoder=SequenceDecoder())
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    assert await controller.candidate(owner, pcm=b"a" * 320) is None
    resumed = controller.resume(owner)
    accepted = await controller.candidate(resumed, pcm=b"b" * 640)

    assert accepted is not None
    assert accepted.revision == 3
    assert client.upserts == [DraftCall(owner.key, 3, "停止。")]
    assert client.cancels == []
    assert controller.commit(resumed, content="停止。").promote_revision == 3
    await controller.close()


@pytest.mark.asyncio
async def test_resume_and_reopen_schedule_only_one_cancel_for_same_revision() -> None:
    class BlockingCancelClient(FakeVoiceDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_started = asyncio.Event()
            self.cancel_release = asyncio.Event()

        async def cancel_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            reason: str,
        ) -> bool:
            self.cancels.append((key, revision, reason))
            self.cancel_started.set()
            await self.cancel_release.wait()
            return True

    client = BlockingCancelClient()
    controller = VoiceDraftTurnController(client=client, decoder=ImmediateDecoder())
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    assert await controller.candidate(owner, pcm=bytes(320)) is not None

    resumed = controller.resume(owner)
    reopened = controller.reopen(speech_generation=2)
    await asyncio.wait_for(client.cancel_started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert resumed.revision == reopened.revision == 2
    assert [call[1:] for call in client.cancels] == [(2, "speech_resumed")]
    client.cancel_release.set()
    await controller.wait_idle()
    controller.commit(reopened, content="最终问题")
    await controller.close()


@pytest.mark.asyncio
async def test_finalized_cancel_survives_opening_the_next_turn() -> None:
    client = FakeVoiceDraftClient()
    controller = VoiceDraftTurnController(client=client, decoder=ImmediateDecoder())
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    assert await controller.candidate(owner, pcm=bytes(320)) is not None
    submission = controller.commit(owner, content="候选问题")
    assert submission.draft_request_started is True

    cancel = controller.cancel_submission(submission, reason="binding_changed")
    controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=2,
    )
    await cancel

    assert [call[1:] for call in client.cancels] == [(3, "binding_changed")]
    await controller.close()


@pytest.mark.asyncio
async def test_resume_fences_an_upsert_that_has_already_started() -> None:
    class BlockingUpsertClient(FakeVoiceDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.upsert_started = asyncio.Event()
            self.upsert_release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            self.upsert_started.set()
            await self.upsert_release.wait()
            return True

    client = BlockingUpsertClient()
    controller = VoiceDraftTurnController(client=client, decoder=ImmediateDecoder())
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    candidate = controller.candidate(owner, pcm=bytes(320))
    assert candidate is not None
    await asyncio.wait_for(client.upsert_started.wait(), timeout=1)

    resumed = controller.resume(owner)
    for _ in range(10):
        if client.cancels:
            break
        await asyncio.sleep(0)
    assert [call[1:] for call in client.cancels] == [(2, "speech_resumed")]
    client.upsert_release.set()
    assert await candidate is None
    await controller.wait_idle()

    controller.commit(resumed, content="最终问题")
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["resume", "reopen", "abandon", "open", "commit", "close"]
)
async def test_cached_transcript_is_invalidated_with_turn_ownership(
    action: str,
) -> None:
    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(), decoder=ImmediateDecoder()
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = bytes(320)
    await controller.candidate(owner, pcm=pcm)
    assert controller.cached_transcript(owner, pcm) is not None

    current = owner
    if action == "resume":
        current = controller.resume(owner)
    elif action == "reopen":
        current = controller.reopen(speech_generation=2)
    elif action == "abandon":
        current = controller.abandon(owner, reason="rejected")
    elif action == "open":
        current = controller.open_turn(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            binding_generation=2,
            speech_generation=2,
        )
    elif action == "commit":
        controller.commit(owner, content="候选问题")
    else:
        await controller.close()

    assert controller.cached_transcript(owner, pcm) is None
    assert controller.cached_transcript(current, pcm) is None
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["accept", "reject", "error", "cancel"])
async def test_cached_transcript_is_ready_before_upload_and_cleared_on_failure(
    outcome: str,
) -> None:
    class BlockingClient(FakeVoiceDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            self.started.set()
            await self.release.wait()
            if outcome == "error":
                raise RuntimeError("draft unavailable")
            return outcome == "accept"

    client = BlockingClient()
    controller = VoiceDraftTurnController(client=client, decoder=ImmediateDecoder())
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = bytes(320)
    task = controller.candidate(owner, pcm=pcm)
    assert task is not None
    try:
        await asyncio.wait_for(client.started.wait(), timeout=1)
        assert not task.done()
        assert controller.cached_transcript(owner, pcm) == AsrUpdate(
            text="候选问题", is_final=True
        )
        registration = asyncio.create_task(controller.wait_for_registration(owner))
        await asyncio.sleep(0)
        assert not registration.done()
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            client.release.set()
            await task
        assert (controller.cached_transcript(owner, pcm) is not None) is (
            outcome == "accept"
        )
        assert await registration is (outcome == "accept")
        submission = controller.commit(owner, content="候选问题")
        assert submission.promote_revision == (1 if outcome == "accept" else 0)
    finally:
        client.release.set()
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_accepted", [False, True])
async def test_late_upload_completion_preserves_newer_candidate_transcript(
    first_accepted: bool,
) -> None:
    class FirstUploadBlockedClient(FakeVoiceDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            if revision == 1:
                self.started.set()
                await self.release.wait()
                return first_accepted
            return True

    client = FirstUploadBlockedClient()
    decoder = ImmediateDecoder("第一份识别")
    controller = VoiceDraftTurnController(client=client, decoder=decoder)
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    first = controller.candidate(owner, pcm=b"a" * 320)
    assert first is not None
    try:
        await asyncio.wait_for(client.started.wait(), timeout=1)
        registration = asyncio.create_task(controller.wait_for_registration(owner))
        await asyncio.sleep(0)
        assert not registration.done()
        decoder.text = "第二份识别"
        # A nonmatching snapshot supersedes the first registration. A matching
        # prefix now intentionally joins the same authorized preparation.
        second_pcm = b"b" * 640
        second = controller.candidate(owner, pcm=second_pcm)
        assert second is not None
        assert await registration is False
        assert await second is not None

        client.release.set()
        assert await first is None
        assert controller.cached_transcript(owner, second_pcm) == AsrUpdate(
            text="第二份识别", is_final=True
        )
        assert controller.commit(owner, content="第二份识别").promote_revision == 2
    finally:
        client.release.set()
        await controller.close()


class RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_frame(self, *_args: object, **_kwargs: object) -> int:
        return -1

    def record_event(self, name: str, **kwargs: object) -> None:
        self.events.append((name, kwargs))


@pytest.mark.asyncio
async def test_controller_records_privacy_safe_real_lifecycle_events() -> None:
    diagnostics = RecordingDiagnostics()
    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(),
        decoder=ImmediateDecoder("不能进入指标的候选问题"),
        diagnostics=diagnostics,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    candidate_task = controller.candidate(owner, pcm=b"a" * 320)
    assert candidate_task is not None
    accepted = await candidate_task
    assert accepted is not None
    controller.resume(owner)
    await controller.wait_idle()

    assert [name for name, _ in diagnostics.events] == [
        "voice_draft_started",
        "voice_draft_transcript_ready",
        "voice_draft_request_started",
        "voice_draft_resumed",
        "voice_draft_canceled",
        "voice_draft_cleanup",
    ]
    assert diagnostics.events[1][1]["metadata"] == {
        "turnId": str(owner.turn_id),
        "revision": 1,
        "charCount": 11,
        "utf8Bytes": 33,
    }
    assert all("text" not in kwargs for _, kwargs in diagnostics.events)
    assert all(
        cast(dict[str, object], kwargs["metadata"])["turnId"] == str(owner.turn_id)
        and isinstance(
            cast(dict[str, object], kwargs["metadata"])["revision"],
            int,
        )
        for _, kwargs in diagnostics.events
    )
    await controller.close()


@pytest.mark.asyncio
async def test_owner_revisions_survive_resume_and_smart_turn_reopen() -> None:
    client = FakeVoiceDraftClient()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=ImmediateDecoder(),
    )
    first_owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=7,
        speech_generation=1,
    )

    first_candidate = await controller.candidate(first_owner, pcm=b"a" * 320)
    resumed = controller.resume(first_owner)
    reopened = controller.reopen(speech_generation=2)
    second_candidate = await controller.candidate(reopened, pcm=b"b" * 640)
    committed = controller.commit(
        reopened,
        content="候选问题",
    )
    await controller.wait_idle()

    assert isinstance(first_candidate, VoiceDraftOwner)
    assert isinstance(second_candidate, VoiceDraftOwner)
    assert (
        len(
            {
                first_owner.turn_id,
                first_candidate.turn_id,
                resumed.turn_id,
                reopened.turn_id,
                second_candidate.turn_id,
                committed.key.turn_id,
            }
        )
        == 1
    )
    assert [
        first_owner.revision,
        first_candidate.revision,
        resumed.revision,
        reopened.revision,
        second_candidate.revision,
        committed.revision,
    ] == [0, 1, 2, 2, 3, 4]
    assert [
        first_owner.candidate_epoch,
        first_candidate.candidate_epoch,
        resumed.candidate_epoch,
        reopened.candidate_epoch,
        second_candidate.candidate_epoch,
        reopened.candidate_epoch,
    ] == [0, 0, 1, 1, 1, 1]
    assert reopened.speech_generation == 2
    assert [call.revision for call in client.upserts] == [1, 3]
    assert committed == VoiceDraftSubmission(
        VoiceDraftKey(FISH_ID, SESSION_ID, first_owner.turn_id),
        4,
        "候选问题",
        3,
        draft_request_started=True,
    )
    assert client.commits == []


@pytest.mark.asyncio
async def test_controller_starts_tts_only_after_accepted_exact_revision() -> None:
    client = FakeVoiceDraftClient()
    draft_tts = FakeDraftTts()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=ImmediateDecoder(),
        draft_tts=draft_tts,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    accepted = await controller.candidate(owner, pcm=b"a" * 320)

    assert isinstance(accepted, VoiceDraftOwner)
    assert draft_tts.starts == [(owner.key, 1)]
    controller.resume(owner)
    await controller.wait_idle()
    assert draft_tts.cancels == [(owner.key, 2)]
    await controller.close()


@pytest.mark.asyncio
async def test_controller_never_starts_tts_for_best_effort_upsert_miss() -> None:
    class MissingClient(FakeVoiceDraftClient):
        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            self.upserts.append(DraftCall(key, revision, content))
            return False

    draft_tts = FakeDraftTts()
    controller = VoiceDraftTurnController(
        client=MissingClient(),
        decoder=ImmediateDecoder(),
        draft_tts=draft_tts,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    assert await controller.candidate(owner, pcm=b"a" * 320) is None
    assert draft_tts.starts == []
    assert controller.commit(owner, content="最终问题").promote_revision == 0
    await controller.close()


@pytest.mark.asyncio
async def test_shadow_controller_runs_draft_but_never_requests_promotion() -> None:
    client = FakeVoiceDraftClient()
    draft_tts = FakeDraftTts()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=ImmediateDecoder("最终问题"),
        draft_tts=draft_tts,
        allow_promotion=False,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    candidate_task = controller.candidate(owner, pcm=b"a" * 320)
    assert candidate_task is not None
    assert await candidate_task is not None
    assert controller.cached_transcript(owner, b"a" * 320) is None
    submission = controller.commit(owner, content="最终问题")
    await controller.wait_idle()

    assert client.upserts == [DraftCall(owner.key, 1, "最终问题")]
    assert submission.promote_revision == 0
    assert draft_tts.cancels == [(owner.key, 2)]
    await controller.close()


@pytest.mark.asyncio
async def test_draft_tts_start_failure_never_changes_authoritative_handoff() -> None:
    class FailingDraftTts(FakeDraftTts):
        async def start(self, key: VoiceDraftKey, *, revision: int) -> bool:
            del key, revision
            raise RuntimeError("provider unavailable")

    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(),
        decoder=ImmediateDecoder("最终问题"),
        draft_tts=FailingDraftTts(),
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    accepted = await controller.candidate(owner, pcm=b"a" * 320)
    submission = controller.commit(owner, content="最终问题")

    assert isinstance(accepted, VoiceDraftOwner)
    assert submission.promote_revision == 1
    assert submission.content == "最终问题"
    await controller.close()


class BlockingDecoder:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def recognize(self, pcm: bytes) -> str:
        assert type(pcm) is bytes
        self.started.set()
        self.release.wait(timeout=1)
        return "过期草稿"


class SerializedBlockingDecoder:
    def __init__(self) -> None:
        self.attempts = 0
        self.first_started = threading.Event()
        self.release = threading.Event()
        self._counter_lock = threading.Lock()
        self._decode_lock = threading.Lock()

    def recognize(self, pcm: bytes) -> str:
        assert type(pcm) is bytes
        with self._counter_lock:
            self.attempts += 1
        with self._decode_lock:
            self.first_started.set()
            self.release.wait(timeout=1)
            return "候选问题"


@pytest.mark.asyncio
async def test_transcript_waiters_share_decode_without_waiting_for_upload() -> None:
    class BlockingClient(FakeVoiceDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            self.started.set()
            await self.release.wait()
            return True

    client = BlockingClient()
    decoder = SerializedBlockingDecoder()
    controller = VoiceDraftTurnController(client=client, decoder=decoder)
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = b"a" * 320
    candidate = controller.candidate(owner, pcm=pcm)
    assert candidate is not None
    try:
        assert await asyncio.to_thread(decoder.first_started.wait, 1)
        waiters = [
            asyncio.create_task(controller.wait_for_transcript(owner, pcm + tail))
            for tail in (b"", bytes(320))
        ]
        await asyncio.sleep(0)
        assert all(not waiter.done() for waiter in waiters)
        decoder.release.set()
        await asyncio.wait_for(client.started.wait(), timeout=1)

        expected = AsrUpdate(text="候选问题", is_final=True)
        assert await asyncio.wait_for(asyncio.gather(*waiters), timeout=0.2) == [
            expected,
            expected,
        ]
        assert await controller.wait_for_transcript(owner, pcm) == expected
        assert not candidate.done()
        assert decoder.attempts == 1
    finally:
        decoder.release.set()
        client.release.set()
        await controller.close()


@pytest.mark.asyncio
async def test_transcript_invalidation_before_put_releases_registration() -> None:
    client = FakeVoiceDraftClient()
    decoder = SerializedBlockingDecoder()
    controller = VoiceDraftTurnController(client=client, decoder=decoder)
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = bytes(320)
    candidate = controller.candidate(owner, pcm=pcm)
    assert candidate is not None
    try:
        assert await asyncio.to_thread(decoder.first_started.wait, 1)
        waiter = asyncio.create_task(controller.wait_for_transcript(owner, pcm))
        registration = asyncio.create_task(controller.wait_for_registration(owner))
        await asyncio.sleep(0)
        controller.invalidate_transcript(owner)

        assert await asyncio.wait_for(waiter, timeout=0.2) is None
        assert await asyncio.wait_for(registration, timeout=0.2) is False
        assert not decoder.release.is_set()
        assert await asyncio.wait_for(candidate, timeout=0.2) is None
        decoder.release.set()
        await controller.wait_idle()
        assert client.upserts == []
        assert client.cancels == []
        assert controller.cached_transcript(owner, pcm) is None
        assert await controller.wait_for_transcript(owner, pcm) is None
        submission = controller.commit(owner, content="候选问题")
        assert submission.revision == 2
        assert submission.promote_revision == 0
    finally:
        decoder.release.set()
        await controller.close()


@pytest.mark.asyncio
async def test_transcript_invalidation_also_revokes_completed_cache() -> None:
    client = FakeVoiceDraftClient()
    controller = VoiceDraftTurnController(client=client, decoder=ImmediateDecoder())
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = bytes(320)
    accepted = await controller.candidate(owner, pcm=pcm)
    assert accepted is not None
    assert controller.cached_transcript(owner, pcm) is not None

    controller.invalidate_transcript(owner)

    assert controller.is_current(accepted)
    assert controller.cached_transcript(owner, pcm) is None
    assert await controller.wait_for_transcript(owner, pcm) is None
    assert client.cancels == []
    assert controller.commit(owner, content="候选问题").promote_revision == 1
    await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["resume", "reopen", "abandon", "open", "commit", "close"]
)
async def test_turn_invalidation_wakes_waiter_before_native_decode_finishes(
    action: str,
) -> None:
    decoder = SerializedBlockingDecoder()
    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(), decoder=decoder
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = bytes(320)
    candidate = controller.candidate(owner, pcm=pcm)
    assert candidate is not None
    try:
        assert await asyncio.to_thread(decoder.first_started.wait, 1)
        waiter = asyncio.create_task(controller.wait_for_transcript(owner, pcm))
        await asyncio.sleep(0)
        assert not waiter.done()
        if action == "resume":
            controller.resume(owner)
        elif action == "reopen":
            controller.reopen(speech_generation=2)
        elif action == "abandon":
            controller.abandon(owner, reason="rejected")
        elif action == "open":
            controller.open_turn(
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                binding_generation=2,
                speech_generation=2,
            )
        elif action == "commit":
            controller.commit(owner, content="权威问题")
        else:
            await controller.close()

        assert await asyncio.wait_for(waiter, timeout=0.2) is None
        assert not decoder.release.is_set()
    finally:
        decoder.release.set()
        await asyncio.gather(candidate, return_exceptions=True)
        await controller.close()


@pytest.mark.asyncio
async def test_cancelling_transcript_waiter_preserves_shared_decode() -> None:
    decoder = SerializedBlockingDecoder()
    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(), decoder=decoder
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = bytes(320)
    candidate = controller.candidate(owner, pcm=pcm)
    assert candidate is not None
    try:
        assert await asyncio.to_thread(decoder.first_started.wait, 1)
        cancelled_waiter = asyncio.create_task(
            controller.wait_for_transcript(owner, pcm)
        )
        surviving_waiter = asyncio.create_task(
            controller.wait_for_transcript(owner, pcm)
        )
        await asyncio.sleep(0)
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        assert not candidate.done()
        assert not surviving_waiter.done()

        decoder.release.set()
        assert await asyncio.wait_for(surviving_waiter, timeout=0.2) == AsrUpdate(
            text="候选问题", is_final=True
        )
        assert await candidate is not None
        assert decoder.attempts == 1
    finally:
        decoder.release.set()
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["empty", "error", "cancel"])
async def test_decode_failure_wakes_transcript_waiter(outcome: str) -> None:
    class FailingDecoder(SerializedBlockingDecoder):
        def recognize(self, pcm: bytes) -> str:
            super().recognize(pcm)
            if outcome == "error":
                raise RuntimeError("decode failed")
            return ""

    decoder = FailingDecoder()
    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(), decoder=decoder
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = bytes(320)
    candidate = controller.candidate(owner, pcm=pcm)
    assert candidate is not None
    try:
        assert await asyncio.to_thread(decoder.first_started.wait, 1)
        waiter = asyncio.create_task(controller.wait_for_transcript(owner, pcm))
        await asyncio.sleep(0)
        assert not waiter.done()
        if outcome == "cancel":
            candidate.cancel()
        else:
            decoder.release.set()
        assert await asyncio.wait_for(waiter, timeout=0.2) is None
        assert controller.cached_transcript(owner, pcm) is None
    finally:
        decoder.release.set()
        await asyncio.gather(candidate, return_exceptions=True)
        await controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["absent", "shadow", "mismatch"])
async def test_wait_for_transcript_requires_registered_matching_candidate(
    state: str,
) -> None:
    decoder = SerializedBlockingDecoder()
    diagnostics = RecordingDiagnostics()
    controller = VoiceDraftTurnController(
        client=FakeVoiceDraftClient(),
        decoder=decoder,
        allow_promotion=state != "shadow",
        diagnostics=diagnostics,
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    pcm = b"a" * 320
    candidate = None if state == "absent" else controller.candidate(owner, pcm=pcm)
    try:
        if state in {"shadow", "mismatch"}:
            assert await asyncio.to_thread(decoder.first_started.wait, 1)
        final_pcm = b"b" * 320 if state == "mismatch" else pcm
        assert await controller.wait_for_transcript(owner, final_pcm) is None
        assert await controller.wait_for_transcript(owner, final_pcm) is None
        misses = [
            kwargs["metadata"]
            for name, kwargs in diagnostics.events
            if name == "voice_draft_asr_reuse_missed"
        ]
        assert len(misses) == 1
        assert (
            misses[0]["reason"]
            == {
                "absent": "no_matching_decode",
                "shadow": "promotion_disabled",
                "mismatch": "pcm_prefix_mismatch",
            }[state]
        )
    finally:
        decoder.release.set()
        if candidate is not None:
            await candidate
        await controller.close()


@pytest.mark.asyncio
async def test_resumed_candidate_drops_while_native_draft_decode_is_still_running() -> (
    None
):
    client = FakeVoiceDraftClient()
    decoder = SerializedBlockingDecoder()
    diagnostics = RecordingDiagnostics()
    controller = VoiceDraftTurnController(
        client=client, decoder=decoder, diagnostics=diagnostics
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    first = controller.prepare(owner, pcm=b"a" * 320)
    assert first is not None
    try:
        await asyncio.to_thread(decoder.first_started.wait, 1)
        resumed = controller.resume(owner)
        second = controller.candidate(resumed, pcm=b"a" * 320 + b"b" * 320)
        repeated = controller.candidate(resumed, pcm=b"a" * 320 + b"b" * 320)

        assert second is None
        assert repeated is None
        assert decoder.attempts == 1
        fallbacks = [
            kwargs["metadata"]
            for name, kwargs in diagnostics.events
            if name == "voice_draft_fallback"
            and kwargs["metadata"]["reason"] == "decode_busy"
        ]
        assert len(fallbacks) == 1
        assert fallbacks[0]["revision"] == resumed.revision
        decoder.release.set()
        assert await first is None
        await controller.wait_idle()
        assert decoder.attempts == 1
        assert client.upserts == []
        submission = controller.commit(resumed, content="权威问题")
        assert submission.promote_revision == 0
    finally:
        decoder.release.set()
        await first
        await controller.close()


@pytest.mark.asyncio
async def test_resumed_speech_fences_late_decoder_before_client_mutation() -> None:
    client = FakeVoiceDraftClient()
    decoder = BlockingDecoder()
    controller = VoiceDraftTurnController(client=client, decoder=decoder)
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    candidate_task = controller.candidate(owner, pcm=b"a" * 320)
    await asyncio.to_thread(decoder.started.wait, 1)
    assert controller.cached_transcript(owner, b"a" * 320) is None
    resumed = controller.resume(owner)
    decoder.release.set()
    assert await candidate_task is None
    await controller.wait_idle()

    assert controller.cached_transcript(resumed, b"a" * 320) is None
    assert controller.is_current(owner) is False
    assert client.upserts == []
    assert client.cancels == []
    assert resumed.revision == 2
    assert resumed.candidate_epoch == 1


@pytest.mark.asyncio
async def test_abandon_revokes_owner_before_late_decode_and_never_commits() -> None:
    client = FakeVoiceDraftClient()
    decoder = BlockingDecoder()
    controller = VoiceDraftTurnController(client=client, decoder=decoder)
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    candidate_task = controller.candidate(owner, pcm=b"a" * 320)
    await asyncio.to_thread(decoder.started.wait, 1)
    abandoned = controller.abandon(owner, reason="rejected")
    decoder.release.set()
    assert await candidate_task is None
    await controller.wait_idle()

    assert abandoned.revision == 2
    assert abandoned.candidate_epoch == 1
    assert client.upserts == []
    assert client.commits == []
    assert client.cancels == []


@pytest.mark.asyncio
async def test_close_invalidates_owner_before_blocked_decoder_can_finish() -> None:
    client = FakeVoiceDraftClient()
    decoder = BlockingDecoder()
    controller = VoiceDraftTurnController(client=client, decoder=decoder)
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    candidate_task = controller.candidate(owner, pcm=b"a" * 320)
    await asyncio.to_thread(decoder.started.wait, 1)

    await controller.close()
    decoder.release.set()
    await asyncio.sleep(0)

    assert controller.is_current(owner) is False
    assert candidate_task.cancelled()
    assert client.upserts == []
    assert client.cancels == []
    with pytest.raises(RuntimeError, match="closed"):
        controller.open_turn(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            binding_generation=2,
            speech_generation=2,
        )


@pytest.mark.asyncio
async def test_close_cancels_blocked_client_before_upsert_mutation() -> None:
    class BlockingClient(FakeVoiceDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            self.started.set()
            await self.release.wait()
            self.upserts.append(DraftCall(key, revision, content))
            return True

    client = BlockingClient()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=ImmediateDecoder(),
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    candidate_task = controller.candidate(owner, pcm=b"a" * 320)
    await client.started.wait()

    await controller.close()
    client.release.set()
    await asyncio.sleep(0)

    assert candidate_task.cancelled()
    assert client.upserts == []
    assert [call[1:] for call in client.cancels] == [(2, "shutdown")]


@pytest.mark.asyncio
async def test_completed_voice_draft_tasks_do_not_accumulate() -> None:
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        controller = VoiceDraftTurnController(
            client=FakeVoiceDraftClient(),
            decoder=ImmediateDecoder(),
        )
        owner = controller.open_turn(
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            binding_generation=1,
            speech_generation=1,
        )
        for index in range(100):
            task = controller.candidate(owner, pcm=bytes([index]) * 320)
            assert task is not None
            candidate_owner = await task
            assert isinstance(candidate_owner, VoiceDraftOwner)
            await asyncio.sleep(0)

        assert controller._tasks == set()

        async def fail() -> None:
            raise RuntimeError("observed background failure")

        failed = controller._track(
            asyncio.create_task(fail(), name="voice-draft-observed-failure")
        )
        for _ in range(100):
            if failed.done() and not controller._tasks:
                break
            await asyncio.sleep(0.001)
        assert failed.done()
        assert controller._tasks == set()
        del failed
        gc.collect()
        await asyncio.sleep(0)

        assert unhandled == []
        assert controller.is_current(candidate_owner) is True
        await controller.close()
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_wait_idle_deterministically_drains_pending_task() -> None:
    class BlockingClient(FakeVoiceDraftClient):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            self.started.set()
            await self.release.wait()
            return await super().upsert_voice_draft(
                key,
                revision=revision,
                content=content,
            )

    client = BlockingClient()
    controller = VoiceDraftTurnController(client=client, decoder=ImmediateDecoder())
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    task = controller.candidate(owner, pcm=b"a" * 320)
    assert task is not None
    await client.started.wait()

    drain = asyncio.create_task(controller.wait_idle())
    await asyncio.sleep(0)
    assert drain.done() is False
    client.release.set()
    await asyncio.wait_for(drain, timeout=0.2)

    assert task.done()
    assert controller._tasks == set()
    await controller.close()


@pytest.mark.asyncio
async def test_fast_candidate_failure_does_not_advance_promotable_owner_state() -> None:
    class FailingUpsertClient(FakeVoiceDraftClient):
        async def upsert_voice_draft(
            self,
            key: VoiceDraftKey,
            *,
            revision: int,
            content: str,
        ) -> bool:
            del key, revision, content
            return False

    client = FailingUpsertClient()
    controller = VoiceDraftTurnController(
        client=client,
        decoder=ImmediateDecoder(),
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )

    assert await controller.candidate(owner, pcm=b"a" * 320) is None
    assert controller._promote_revision == 0
    resumed = controller.resume(owner)
    await controller.wait_idle()

    assert resumed.revision == 2
    assert resumed.candidate_epoch == 1
    await controller.close()
