"""Early ASR is shared with the later, separately authorized draft request."""

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.voice_drafts import (
    CandidateEvent,
    SpeculativeSilenceGate,
    VoiceDraftTurnController,
)

from .test_voice_drafts import (
    FISH_ID,
    SESSION_ID,
    DraftCall,
    FakeDraftTts,
    FakeVoiceDraftClient,
)


class ControlledDecoder:
    def __init__(self, *, blocked: bool = False) -> None:
        self.pcm: list[bytes] = []
        self.entered = [threading.Event() for _ in range(4)]
        self.release = [threading.Event() for _ in range(4)]
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        if not blocked:
            self.release_all()

    def recognize_update(self, pcm: bytes) -> AsrUpdate:
        with self.lock:
            index = len(self.pcm)
            self.pcm.append(pcm)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered[index].set()
        try:
            assert self.release[index].wait(timeout=2)
            return AsrUpdate(text=f"识别{index + 1}", is_final=True, event="Speech")
        finally:
            with self.lock:
                self.active -= 1

    def recognize(self, pcm: bytes) -> str:
        raise AssertionError("recognize_update must preserve event metadata")

    def release_all(self) -> None:
        for event in self.release:
            event.set()


@asynccontextmanager
async def preparation_case(
    *, blocked: bool = False, client: FakeVoiceDraftClient | None = None
) -> AsyncIterator[SimpleNamespace]:
    decoder = ControlledDecoder(blocked=blocked)
    client = client or FakeVoiceDraftClient()
    tts = FakeDraftTts()
    controller = VoiceDraftTurnController(client=client, decoder=decoder, draft_tts=tts)
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
    )
    try:
        yield SimpleNamespace(
            decoder=decoder, client=client, tts=tts, controller=controller, owner=owner
        )
    finally:
        decoder.release_all()
        await controller.close()


def test_gate_prepares_on_first_silence_and_authorizes_at_100_ms() -> None:
    gate = SpeculativeSilenceGate(silence_frames=10, prepare_silence_frames=1)
    gate.open_turn()
    assert gate.accept(True) is CandidateEvent.NONE
    assert gate.accept(False) is CandidateEvent.PREPARE
    assert [gate.accept(False) for _ in range(8)] == [CandidateEvent.NONE] * 8
    assert gate.accept(False) is CandidateEvent.CANDIDATE
    assert gate.accept(False) is CandidateEvent.NONE
    assert gate.accept(True) is CandidateEvent.RESUMED
    assert gate.accept(True) is CandidateEvent.NONE
    assert gate.accept(False) is CandidateEvent.PREPARE


def test_speech_after_prepare_before_candidate_invalidates_that_pause() -> None:
    gate = SpeculativeSilenceGate(silence_frames=10, prepare_silence_frames=1)
    gate.open_turn()
    assert gate.accept(False) is CandidateEvent.PREPARE
    assert gate.accept(True) is CandidateEvent.RESUMED
    assert gate.accept(False) is CandidateEvent.PREPARE
    assert [gate.accept(False) for _ in range(8)] == [CandidateEvent.NONE] * 8
    assert gate.accept(False) is CandidateEvent.CANDIDATE
    summary = gate.take_summary()
    assert summary is not None
    assert summary["gateCandidateCount"] == 1
    assert summary["gateResumedCount"] == 1


def test_zero_wait_configuration_can_use_one_direct_candidate_frame() -> None:
    # Runner maps zero milliseconds to the first observed non-speech frame.
    gate = SpeculativeSilenceGate(silence_frames=1, prepare_silence_frames=1)
    gate.open_turn()
    assert gate.accept(False) is CandidateEvent.CANDIDATE
    assert gate.accept(False) is CandidateEvent.NONE
    assert gate.accept(True) is CandidateEvent.RESUMED


@pytest.mark.parametrize("prepare_frames", [-1, 0, 11])
def test_prepare_threshold_must_fit_the_candidate_window(prepare_frames: int) -> None:
    with pytest.raises(ValueError, match="prepare_silence_frames"):
        SpeculativeSilenceGate(silence_frames=10, prepare_silence_frames=prepare_frames)


@pytest.mark.asyncio
async def test_ready_preparation_waits_for_authorization_and_registers_once() -> None:
    async with preparation_case() as case:
        pcm = b"a" * 320
        prepared = await case.controller.prepare(case.owner, pcm=pcm)
        assert prepared is not None and prepared.revision == 1
        assert case.client.upserts == []
        assert case.tts.starts == []
        assert case.controller._promote_revision == 0
        assert not await case.controller.wait_for_registration(case.owner)
        assert case.controller.has_preparation(case.owner, pcm=pcm + bytes(320))
        assert not case.controller.has_preparation(case.owner, pcm=b"b" * 320)

        candidate = case.controller.candidate(case.owner, pcm=pcm + bytes(320))
        duplicate = case.controller.candidate(case.owner, pcm=pcm + bytes(640))
        assert duplicate is candidate
        assert await candidate == prepared
        assert case.decoder.pcm == [pcm]
        assert case.client.upserts == [DraftCall(case.owner.key, 1, "识别1")]
        assert case.tts.starts == [(case.owner.key, 1)]
        assert case.controller.cached_transcript(
            case.owner, pcm + bytes(640)
        ) == AsrUpdate(text="识别1", is_final=True, event="Speech")


@pytest.mark.asyncio
async def test_pending_preparation_and_candidate_share_one_decode_and_ack() -> None:
    async with preparation_case(blocked=True) as case:
        pcm = b"a" * 320
        preparation = case.controller.prepare(case.owner, pcm=pcm)
        assert await asyncio.to_thread(case.decoder.entered[0].wait, 1)
        candidate = case.controller.candidate(case.owner, pcm=pcm + bytes(320))
        registration = asyncio.create_task(
            case.controller.wait_for_registration(case.owner)
        )
        transcript = asyncio.create_task(
            case.controller.wait_for_transcript(case.owner, pcm)
        )
        await asyncio.sleep(0)
        assert not registration.done()
        assert not transcript.done()
        assert case.client.upserts == []
        case.decoder.release[0].set()

        assert await asyncio.wait_for(preparation, 1) == await asyncio.wait_for(
            candidate, 1
        )
        assert await registration
        assert (await transcript).text == "识别1"
        assert case.decoder.pcm == [pcm]
        assert len(case.client.upserts) == 1


@pytest.mark.asyncio
async def test_native_busy_drops_new_preparations_until_current_decode_returns() -> (
    None
):
    async with preparation_case(blocked=True) as case:
        first_pcm = b"a" * 320
        first = case.controller.prepare(case.owner, pcm=first_pcm)
        assert await asyncio.to_thread(case.decoder.entered[0].wait, 1)
        second_owner = case.controller.resume(case.owner)
        second_pcm = first_pcm + b"b" * 320
        second = case.controller.prepare(second_owner, pcm=second_pcm)
        second_candidate = case.controller.candidate(
            second_owner, pcm=second_pcm + bytes(320)
        )
        third_owner = case.controller.resume(second_owner)
        third_pcm = second_pcm + b"c" * 320
        third = case.controller.prepare(third_owner, pcm=third_pcm)
        third_candidate = case.controller.candidate(
            third_owner, pcm=third_pcm + bytes(320)
        )
        assert not case.controller.has_preparation(third_owner, pcm=third_pcm)
        assert await asyncio.wait_for(first, 0.2) is None
        assert second is None
        assert second_candidate is None
        assert third is None
        assert third_candidate is None
        assert case.decoder.pcm == [first_pcm]

        case.decoder.release[0].set()
        await case.controller.wait_idle()
        assert await case.controller.wait_for_transcript(third_owner, third_pcm) is None
        assert case.decoder.pcm == [first_pcm]
        assert case.decoder.max_active == 1
        assert case.client.upserts == []
        submission = case.controller.commit(third_owner, content="权威问题")
        assert submission.promote_revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("decoded", [False, True])
async def test_invalidation_restarts_full_snapshot_only_after_native_slot_is_free(
    decoded: bool,
) -> None:
    async with preparation_case(blocked=True) as case:
        old_pcm = b"a" * 320
        old = case.controller.prepare(case.owner, pcm=old_pcm)
        assert await asyncio.to_thread(case.decoder.entered[0].wait, 1)
        if decoded:
            case.decoder.release[0].set()
            assert await old is not None
        case.controller.invalidate_transcript(case.owner)
        assert not case.controller.has_preparation(case.owner, pcm=old_pcm)
        new_pcm = old_pcm + b"b" * 320
        candidate = case.controller.candidate(case.owner, pcm=new_pcm)
        case.decoder.release[0].set()
        if decoded:
            assert candidate is not None
            assert await asyncio.to_thread(case.decoder.entered[1].wait, 1)
            case.decoder.release[1].set()
            assert await asyncio.wait_for(candidate, 1) is not None
            assert case.decoder.pcm == [old_pcm, new_pcm]
            assert case.client.upserts == [DraftCall(case.owner.key, 2, "识别2")]
            assert case.controller.cached_transcript(case.owner, new_pcm).text == (
                "识别2"
            )
        else:
            assert candidate is None
            assert await old is None
            await case.controller.wait_idle()
            assert case.decoder.pcm == [old_pcm]
            assert case.client.upserts == []
            assert case.controller.cached_transcript(case.owner, new_pcm) is None


@pytest.mark.asyncio
async def test_ready_result_invalidated_before_register_task_runs_cannot_upload() -> (
    None
):
    async with preparation_case() as case:
        pcm = b"a" * 320
        assert await case.controller.prepare(case.owner, pcm=pcm) is not None
        candidate = case.controller.candidate(case.owner, pcm=pcm)
        # No event-loop yield: decoded is ready but PUT has not started.
        case.controller.invalidate_transcript(case.owner)
        assert not await case.controller.wait_for_registration(case.owner)
        assert await candidate is None
        assert case.client.upserts == []
        assert case.tts.starts == []
        assert (
            case.controller.commit(case.owner, content="正式问题").promote_revision == 0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["resume", "reopen", "open", "abandon", "commit", "close"]
)
async def test_busy_dropped_preparation_never_starts_native_or_uploads(
    action: str,
) -> None:
    async with preparation_case(blocked=True) as case:
        first = case.controller.prepare(case.owner, pcm=b"a" * 320)
        assert await asyncio.to_thread(case.decoder.entered[0].wait, 1)
        owner = case.controller.resume(case.owner)
        dropped = case.controller.prepare(owner, pcm=b"b" * 320)
        candidate = case.controller.candidate(owner, pcm=b"b" * 640)
        assert dropped is None
        assert candidate is None
        if action == "resume":
            case.controller.resume(owner)
        elif action == "reopen":
            case.controller.reopen(speech_generation=2)
        elif action == "open":
            case.controller.open_turn(
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                binding_generation=2,
                speech_generation=2,
            )
        elif action == "abandon":
            case.controller.abandon(owner, reason="rejected")
        elif action == "commit":
            case.controller.commit(owner, content="正式问题")
        else:
            await case.controller.close()
        await asyncio.gather(first, return_exceptions=True)
        case.decoder.release_all()
        await case.controller.wait_idle()
        assert case.decoder.pcm == [b"a" * 320]
        assert case.client.upserts == []
