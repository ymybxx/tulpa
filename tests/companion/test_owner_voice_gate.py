import asyncio
import struct
import threading
from collections import deque

import pytest

from fish_memory.companion.audio_processing import ProcessedAudioFrame
from fish_memory.companion.owner_voice_gate import OwnerVoiceFrame, OwnerVoiceGate

PCM = struct.pack("<h", 1000) * 160


class Verifier:
    def __init__(self, *scores: float):
        self.scores = deque(scores or [0.9])
        self.inputs: list[bytes] = []

    def score_pcm(self, pcm: bytes) -> float:
        self.inputs.append(pcm)
        return self.scores.popleft() if len(self.scores) > 1 else self.scores[0]


def frame(index=0, *, speech=True, pcm=PCM):
    return OwnerVoiceFrame(ProcessedAudioFrame(pcm, speech), index)


def gate(verifier, **kwargs):
    return OwnerVoiceGate(
        verifier,
        check_speech_ms=40,
        minimum_speech_ms=20,
        max_buffer_ms=100,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_buffers_and_releases_checked_pcm_in_original_order():
    verifier = Verifier()
    admission = gate(verifier)
    for i in range(3):
        assert not (await admission.accept(frame(i))).frames
    result = await admission.accept(frame(3))
    assert result.reason == "matched"
    assert [item.frame_index for item in result.frames] == [0, 1, 2, 3]
    assert verifier.inputs == [PCM * 4]
    assert admission.buffered_frames == 0


@pytest.mark.asyncio
async def test_match_never_opens_gate_for_later_unverified_audio():
    admission = gate(Verifier(0.9, 0.1))
    for i in range(4):
        result = await admission.accept(frame(i))
    assert len(result.frames) == 4
    for i in range(4, 7):
        assert not (await admission.accept(frame(i))).frames
    result = await admission.accept(frame(7))
    assert result.reason == "not_owner"
    assert result.revoked and not result.frames


@pytest.mark.asyncio
async def test_final_short_speech_is_verified_not_stuck_forever():
    verifier = Verifier()
    admission = gate(verifier)
    await admission.accept(frame(0))
    await admission.accept(frame(1))
    result = await admission.accept(frame(2, speech=False, pcm=bytes(320)))
    assert result.reason == "matched"
    assert len(result.frames) == 3
    assert len(verifier.inputs) == 1
    assert admission.buffered_frames == 0


@pytest.mark.asyncio
async def test_insufficient_initial_speech_does_not_call_model():
    verifier = Verifier()
    admission = gate(verifier)
    await admission.accept(frame())
    result = await admission.accept(frame(speech=False))
    assert result.reason == "insufficient_speech"
    assert not result.frames and not verifier.inputs


@pytest.mark.asyncio
async def test_short_tail_uses_only_bounded_recent_checked_context():
    verifier = Verifier()
    admission = gate(verifier, context_speech_ms=60)
    for _ in range(8):
        await admission.accept(frame())
    await admission.accept(frame(8))
    result = await admission.accept(frame(9, speech=False, pcm=bytes(320)))
    assert result.reason == "matched"
    assert len(result.frames) == 2
    assert len(verifier.inputs[-1]) == 320 * 8


@pytest.mark.asyncio
async def test_each_block_is_checked_with_this_utterance_so_far():
    verifier = Verifier()
    admission = gate(verifier)
    for _ in range(8):
        result = await admission.accept(frame())
    assert result.reason == "matched"
    assert [len(pcm) // 320 for pcm in verifier.inputs] == [4, 8]


@pytest.mark.asyncio
async def test_a_miss_before_any_match_waits_to_check_a_longer_stretch():
    verifier = Verifier(0.3, 0.9)
    admission = gate(verifier)
    for i in range(4):
        result = await admission.accept(frame(i))
    assert result.reason == "not_owner"
    assert not result.frames and not result.revoked
    for i in range(4, 8):
        result = await admission.accept(frame(i))
    assert result.reason == "matched"
    assert [item.frame_index for item in result.frames] == list(range(8))
    assert [len(pcm) // 320 for pcm in verifier.inputs] == [4, 8]


@pytest.mark.asyncio
async def test_an_utterance_that_ends_unmatched_is_still_refused():
    verifier = Verifier(0.3)
    admission = gate(verifier)
    for _ in range(4):
        await admission.accept(frame())
    result = await admission.accept(frame(speech=False, pcm=bytes(320)))
    assert result.reason == "not_owner"
    assert not result.frames
    assert admission.buffered_frames == 0


@pytest.mark.asyncio
async def test_vad_end_clears_identity_before_next_speaker():
    admission = gate(Verifier(0.9, 0.1))
    for _ in range(4):
        await admission.accept(frame())
    await admission.accept(frame(speech=False, pcm=bytes(320)))
    for _ in range(4):
        result = await admission.accept(frame())
    assert not result.frames and not result.revoked
    assert result.reason == "not_owner"


@pytest.mark.asyncio
async def test_vad_trailing_silence_flags_preserve_existing_duration_accounting():
    admission = gate(Verifier())
    for _ in range(4):
        await admission.accept(frame())
    await admission.accept(frame(4, pcm=bytes(320)))
    result = await admission.accept(frame(5, speech=False, pcm=bytes(320)))
    assert [item.processed.voice_detected for item in result.frames] == [True, False]
    assert all(item.processed.cleaned_pcm == bytes(320) for item in result.frames)


@pytest.mark.asyncio
async def test_idle_pcm_is_silenced_not_forwarded_to_asr_or_kws():
    result = await gate(Verifier()).accept(frame(speech=False))
    assert result.frames[0].processed.cleaned_pcm == bytes(320)


@pytest.mark.asyncio
async def test_buffers_stay_bounded_even_when_vad_labels_digital_silence_speech():
    verifier = Verifier()
    admission = gate(verifier)
    reasons = []
    for _ in range(1000):
        result = await admission.accept(frame(pcm=bytes(320)))
        assert admission.buffered_frames <= 10
        if result.reason:
            reasons.append(result.reason)
    assert "buffer_limit" in reasons
    assert not verifier.inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [float("nan"), float("inf"), 2, -2, 0.1])
async def test_bad_or_non_owner_scores_fail_closed(score):
    admission = gate(Verifier(score))
    for _ in range(4):
        result = await admission.accept(frame())
    assert not result.frames
    assert result.reason in {"invalid_score", "not_owner"}


@pytest.mark.asyncio
async def test_model_exception_revokes_previous_admission():
    class Broken(Verifier):
        def score_pcm(self, pcm):
            if self.inputs:
                raise RuntimeError("internal")
            return super().score_pcm(pcm)

    admission = gate(Broken())
    for _ in range(8):
        result = await admission.accept(frame())
    assert result.reason == "inference_failed"
    assert result.revoked and not result.frames


@pytest.mark.asyncio
async def test_reset_discards_late_inference_result():
    entered, release = threading.Event(), threading.Event()

    class Blocking(Verifier):
        def score_pcm(self, pcm):
            entered.set()
            release.wait(2)
            return 0.9

    admission = gate(Blocking())
    for _ in range(3):
        await admission.accept(frame())
    task = asyncio.create_task(admission.accept(frame()))
    try:
        await asyncio.to_thread(entered.wait, 1)
        admission.reset()
        release.set()
        result = await task
        assert result.reason == "stale_result" and not result.frames
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_timeout_never_launches_parallel_unbounded_model_workers():
    release = threading.Event()
    calls = []

    class Blocking(Verifier):
        def score_pcm(self, pcm):
            calls.append(pcm)
            release.wait(2)
            return 0.9

    admission = gate(Blocking(), inference_timeout=0.01)
    try:
        for _ in range(4):
            result = await admission.accept(frame())
        assert result.reason == "inference_timeout"
        for _ in range(12):
            result = await admission.accept(frame())
        assert result.reason == "inference_busy"
        assert len(calls) == 1
    finally:
        release.set()
        await admission._inference


@pytest.mark.parametrize("threshold", [float("nan"), -0.1, 1.1])
def test_configuration_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        gate(Verifier(), threshold=threshold)


async def miss_once(admission, start=0):
    """Four frames of speech then silence: one check, one final miss."""
    for i in range(start, start + 4):
        await admission.accept(frame(i))
    return await admission.accept(frame(start + 4, speech=False, pcm=bytes(320)))


@pytest.mark.asyncio
async def test_near_misses_are_kept_in_memory_for_the_owner_to_add():
    seen = []
    admission = gate(Verifier(0.45), on_unrecognized=seen.append)
    result = await miss_once(admission)
    assert result.reason == "not_owner"
    # Only the unreleased speech of that utterance, 40 ms of it.
    assert admission.unrecognized_seconds() == pytest.approx(0.04)
    assert seen[-1] == pytest.approx(0.04)
    speech, heard = admission.take_unrecognized()
    # The voiceprint gets the speech; listening gets the utterance as heard.
    assert speech == PCM * 4
    assert heard == [PCM * 4 + bytes(320)]
    assert admission.unrecognized_seconds() == 0
    assert seen[-1] == 0


@pytest.mark.asyncio
async def test_clear_misses_are_not_kept():
    seen = []
    admission = gate(Verifier(0.1), on_unrecognized=seen.append)
    assert (await miss_once(admission)).reason == "not_owner"
    assert admission.unrecognized_seconds() == 0
    assert seen == []


@pytest.mark.asyncio
async def test_kept_near_misses_are_forgotten_after_a_few_minutes():
    now = [0.0]
    admission = gate(Verifier(0.45), clock=lambda: now[0])
    await miss_once(admission)
    now[0] = 299
    assert admission.unrecognized_seconds() == pytest.approx(0.04)
    now[0] = 301
    assert admission.unrecognized_seconds() == 0
    assert admission.take_unrecognized() == (b"", [])


@pytest.mark.asyncio
async def test_an_extended_voiceprint_decides_the_next_check():
    admission = gate(Verifier(0.45))
    assert (await miss_once(admission)).reason == "not_owner"
    admission.replace_verifier(Verifier(0.9))
    for i in range(4):
        result = await admission.accept(frame(10 + i))
    assert result.reason == "matched"
