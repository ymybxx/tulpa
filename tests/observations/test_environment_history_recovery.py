from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from fish_memory.observations.environment_history_audio import (
    LocalSystemAudioTranscriber,
    SystemAudioTranscript,
)
from fish_memory.observations.environment_history_recovery import (
    RecoveredTranscript,
    sparse_speech_region,
    splice_recovered_speech,
)
from fish_memory.observations.environment_history_speakers import SpeakerSpan


def source() -> RecoveredTranscript:
    return RecoveredTranscript(
        "订单已。", ("订", "单", "已", "。"), (3.2, 3.4, 3.65, 9.9)
    )


def activity() -> tuple[SpeakerSpan, ...]:
    return (
        SpeakerSpan(52800, 84000, None, local_speaker_id="local:0"),
        SpeakerSpan(89280, 97440, None, local_speaker_id="local:1"),
    )


def region_for(value: RecoveredTranscript):
    return sparse_speech_region(
        value.text, value.tokens, value.timestamps, activity(), 10
    )


def test_sparse_whole_passage_includes_adjacent_speakers_without_word_crops():
    region = region_for(source())
    assert region is not None
    assert region.start == pytest.approx(3.15)
    assert region.end == pytest.approx(6.24)
    assert region.anchors == "订单已"
    assert region.first_token == 0 and region.last_token == 3


def test_recovered_question_retains_anchor_and_shifts_its_actual_token_times():
    original = source()
    region = region_for(original)
    text = "订单已接收，什么时候送到？"
    candidate = RecoveredTranscript(
        text, tuple(text), tuple(i * 0.2 for i in range(len(text)))
    )
    result = splice_recovered_speech(original, candidate, region)
    assert result is not None
    assert result.text == text
    assert result.timestamps[0] == pytest.approx(3.15)
    assert result.timestamps[-1] <= 6.24


@pytest.mark.parametrize(
    "text", ["订单已。", "啊啊啊啊啊啊。", "对方说明天再来。", "咻咻咻咻咻。"]
)
def test_short_repetition_or_rewritten_original_words_cannot_replace_primary(text):
    candidate = RecoveredTranscript(
        text, tuple(text), tuple(i * 0.1 for i in range(len(text)))
    )
    assert splice_recovered_speech(source(), candidate, region_for(source())) is None


def test_missing_or_changed_token_alignment_disables_recovery():
    assert region_for(replace(source(), tokens=(), timestamps=())) is None
    assert region_for(replace(source(), text="完全不同的正文")) is None
    assert region_for(replace(source(), timestamps=(3.2, 3.4, 3.1, 9.9))) is None


def test_one_sound_or_dense_dialogue_does_not_start_a_recovery_request():
    one = RecoveredTranscript("咻。", ("咻", "。"), (3.5, 9.9))
    assert region_for(one) is None
    text = "完整的话已经都听到了。"
    dense = RecoveredTranscript(
        text, tuple(text), tuple(3.3 + i * 0.2 for i in range(len(text)))
    )
    assert region_for(dense) is None


def test_missing_beginning_of_a_later_speech_passage_can_be_recovered():
    text = "之前说完了。继续。"
    original = RecoveredTranscript(
        text, tuple(text), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 8.8, 9.0, 9.2)
    )
    spans = (
        SpeakerSpan(0, 19000, None, local_speaker_id="first"),
        SpeakerSpan(80000, 150000, None, local_speaker_id="later"),
    )
    region = sparse_speech_region(text, original.tokens, original.timestamps, spans, 10)
    assert region is not None and region.anchors == "继续"
    assert region.start == pytest.approx(4.85)
    candidate_text = "事情还没结束，我们继续。"
    candidate = RecoveredTranscript(
        candidate_text,
        tuple(candidate_text),
        tuple(i * 0.2 for i in range(len(candidate_text))),
    )
    recovered = splice_recovered_speech(original, candidate, region)
    assert recovered is not None and recovered.text == "之前说完了。" + candidate_text
    assert recovered.timestamps[:6] == original.timestamps[:6]


def test_short_words_with_no_unexplained_speech_gap_do_not_retry():
    text = "过来。"
    value = RecoveredTranscript(text, tuple(text), (3.4, 4.0, 4.2))
    spans = (SpeakerSpan(51200, 70400, None, local_speaker_id="voice"),)
    assert sparse_speech_region(text, value.tokens, value.timestamps, spans, 10) is None


def test_locally_finished_short_sentences_do_not_retry_inside_broad_vad():
    text = "第一句。第二句。"
    value = RecoveredTranscript(
        text, tuple(text), (0.5, 0.6, 0.7, 0.8, 2.5, 2.6, 2.7, 2.8)
    )
    spans = (SpeakerSpan(0, 80000, None, local_speaker_id="voice"),)
    assert sparse_speech_region(text, value.tokens, value.timestamps, spans, 10) is None


@pytest.mark.asyncio
async def test_foreground_asr_busy_preserves_primary_text_and_records_diagnostic(
    monkeypatch,
):
    transcriber = LocalSystemAudioTranscriber(SimpleNamespace())
    calls = []

    async def busy(pcm):
        calls.append(len(pcm))
        return SystemAudioTranscript("", "unavailable")

    monkeypatch.setattr(transcriber, "transcribe", busy)
    original = SystemAudioTranscript(
        source().text, "Speech", tokens=source().tokens, timestamps=source().timestamps
    )
    try:
        result = await transcriber._recover_sparse_speech(
            b"\x01\x01" * 160000, original, activity()
        )
        assert result.text == original.text
        assert result.tokens == original.tokens
        assert result.recovery_status == "unavailable"
        assert len(calls) == 1 and 32000 < calls[0] < 7 * 32000
    finally:
        await transcriber.close()
