from __future__ import annotations

import pytest

from fish_memory.observations.environment_history_alignment import align_transcript
from fish_memory.observations.environment_history_speakers import (
    SAMPLE_RATE,
    SpeakerSpan,
)


def span(start: float, end: float, speaker: str | None, **kwargs) -> SpeakerSpan:
    return SpeakerSpan(
        round(start * SAMPLE_RATE), round(end * SAMPLE_RATE), speaker, **kwargs
    )


def test_complete_chinese_text_keeps_context_and_does_not_cross_voice_change():
    text = "今天想吃什么？我想吃面。"
    times = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.4, 1.5, 1.6, 1.7, 3.9)
    result = align_transcript(
        text, tuple(text), times, (span(0, 1, "A"), span(1.2, 2, "B")), 4
    )
    assert result.status == "aligned"
    assert [(part.text, part.span_index) for part in result.parts] == [
        ("今天想吃什么？", 0),
        ("我想吃面。", 1),
    ]
    assert result.parts[-1].end_seconds < 2
    assert "".join(part.text for part in result.parts) == text


def test_point_belongs_to_its_actual_interval_not_presentation_padding():
    result = align_transcript(
        "好吧我来。",
        ("好", "吧", "我", "来", "。"),
        (0.5, 0.95, 1.0, 1.4, 1.5),
        (span(0, 1, "A"), span(1, 2, "B")),
        2,
    )
    assert "".join(part.text for part in result.parts) == "好吧我来。"
    assert [(part.text, part.span_index, part.reason) for part in result.parts] == [
        ("好吧", 0, "token_time"),
        ("我来。", 1, "token_time"),
    ]
    assert not any(part.overlap for part in result.parts)


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("5500。", ("5", "500", "。")),
        ("19.80。", ("19.", "80", "。")),
        ("1,980。", ("1,", "980", "。")),
        ("Marvell。", ("Mar", "vell", "。")),
    ],
)
def test_an_acoustic_boundary_cannot_split_one_number_or_latin_word(text, tokens):
    result = align_transcript(
        text, tokens, (0.3, 0.6, 0.8), (span(0, 0.5, "A"), span(0.5, 1, "B")), 1
    )
    assert [(part.text, part.span_index, part.reason) for part in result.parts] == [
        (text, None, "ambiguous_word_boundary")
    ]
    assert result.parts[0].start_seconds == pytest.approx(0.18)
    assert result.parts[0].end_seconds == pytest.approx(0.72)


def test_unknown_islands_cannot_turn_one_amount_into_two_amounts():
    result = align_transcript(
        "5500。",
        ("5", "500", "。"),
        (0.3, 0.6, 0.8),
        (span(0.5, 1, None),),
        1,
    )
    assert len(result.parts) == 1
    assert result.parts[0].text == "5500。" and result.parts[0].span_index is None


def test_ambiguous_word_does_not_erase_neighbouring_attribution_or_spacing():
    result = align_transcript(
        "报价5500。 Next",
        ("报", "价", "5", "500", "。", " Next"),
        (0.1, 0.2, 0.3, 0.6, 0.7, 0.8),
        (span(0, 0.5, "A"), span(0.5, 1, "B")),
        1,
    )
    assert [(part.text, part.span_index) for part in result.parts] == [
        ("报价", 0),
        ("5500。 ", None),
        ("Next", 1),
    ]
    assert "".join(part.text for part in result.parts) == "报价5500。 Next"


def test_punctuation_between_words_still_allows_a_real_speaker_change():
    result = align_transcript(
        "Hi,No",
        ("Hi", ",", "No"),
        (0.2, 0.3, 0.7),
        (span(0, 0.5, "A"), span(0.5, 1, "B")),
        1,
    )
    assert [(part.text, part.span_index) for part in result.parts] == [
        ("Hi,", 0),
        ("No", 1),
    ]


def test_overlap_and_low_confidence_keep_words_but_no_identity():
    result = align_transcript(
        "别说了好。",
        ("别", "说", "了", "好", "。"),
        (0.3, 0.5, 0.7, 1.6, 1.8),
        (
            span(0, 1, "must-not-leak", overlap=True),
            span(1.2, 2, "must-not-leak", reason="low_segmentation_confidence"),
        ),
        2,
    )
    assert "".join(part.text for part in result.parts) == "别说了好。"
    assert [
        (part.text, part.span_index, part.reason, part.overlap) for part in result.parts
    ] == [
        ("别说了", None, "overlap", True),
        ("好。", None, "low_segmentation_confidence", False),
    ]


def test_unknown_speaker_islands_and_silence_are_not_one_person():
    result = align_transcript(
        "好。可以。来吧。",
        tuple("好。可以。来吧。"),
        (0.2, 0.3, 1.3, 1.5, 1.6, 3.1, 3.2, 3.3),
        (
            span(0, 0.7, None, local_speaker_id="island1"),
            span(1, 2, None, local_speaker_id="island2"),
            span(3, 4, None, local_speaker_id="island3"),
        ),
        4,
    )
    assert [(part.text, part.span_index) for part in result.parts] == [
        ("好。", 0),
        ("可以。", 1),
        ("来吧。", 2),
    ]


def test_same_span_still_keeps_a_long_silence_boundary():
    result = align_transcript(
        "好。再见。",
        tuple("好。再见。"),
        (0.2, 0.3, 2.1, 2.3, 2.5),
        (span(0, 3, "A"),),
        3,
    )
    assert [part.text for part in result.parts] == ["好。", "再见。"]
    assert all(part.span_index == 0 for part in result.parts)


@pytest.mark.parametrize(
    "tokens",
    [
        ("▁I", "▁like", "▁", "苹", "果", "😀", "."),
        ("I", " like", " ", "苹", "果", "😀", "."),
        ("I", " li", "##ke", "苹", "果", "😀", "."),
    ],
)
def test_spaces_subwords_chinese_and_emoji_preserve_the_original_text(tokens):
    text = " I like 苹果😀. "
    result = align_transcript(
        text, tokens, (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (span(0, 2, "A"),), 2
    )
    assert result.status == "aligned"
    assert len(result.parts) == 1
    assert result.parts[0].text == text and result.parts[0].span_index == 0


def test_identical_repeated_sentences_use_exact_sequential_token_positions():
    text = "我给你加钱。我给你加钱。"
    result = align_transcript(
        text,
        tuple(text),
        (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7),
        (span(0, 1, "A"), span(2, 3, "B")),
        3,
    )
    assert [(part.text, part.span_index) for part in result.parts] == [
        ("我给你加钱。", 0),
        ("我给你加钱。", 1),
    ]


def test_itn_number_replacement_retains_unique_anchors_and_unknown_number():
    spoken = "这个包要八千元，你知道吗？"
    text = "这个包要8000元，你知道吗？"
    result = align_transcript(
        text,
        tuple(spoken),
        tuple(index * 0.1 + 0.2 for index in range(len(spoken))),
        (span(0, 3, "A"),),
        3,
    )
    assert result.status == "partial"
    assert "".join(part.text for part in result.parts) == text
    assert [(part.text, part.span_index, part.reason) for part in result.parts] == [
        ("这个包要", 0, "token_time"),
        ("8000", None, "text_unaligned"),
        ("元，你知道吗？", 0, "token_time"),
    ]


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("你好你好", "你好"),
        ("你好", "你好你好"),
        ("10", "十"),
        ("cat", "category"),
        ("谢谢。谢谢。", "谢谢。"),
    ],
)
def test_ambiguous_repeated_or_short_itn_alignment_uses_whole_text_fallback(
    text, spoken
):
    result = align_transcript(
        text,
        tuple(spoken),
        tuple(index * 0.1 for index in range(len(spoken))),
        (span(0, 3, "A"),),
        3,
    )
    assert result.parts == () and result.status == "unalignable_text"


def test_missing_span_keeps_audio_the_segmenter_did_not_find():
    result = align_transcript("您好。", tuple("您好。"), (0.2, 0.4, 0.6), (), 1)
    assert len(result.parts) == 1
    assert result.parts[0].text == "您好。"
    assert result.parts[0].span_index is None
    assert result.parts[0].reason == "outside_speech"


def test_equal_points_and_partial_last_window_remain_bounded():
    result = align_transcript(
        "您好。",
        tuple("您好。"),
        (0.0, 0.0, 0.0),
        (span(0, 0.1, "A"),),
        0.1,
    )
    assert len(result.parts) == 1
    assert result.parts[0].start_seconds == 0
    assert result.parts[0].end_seconds == 0.1
    assert result.parts[0].span_index == 0
    edge = align_transcript("好", ("好",), (0.11,), (span(0, 0.1, "A"),), 0.1)
    assert edge.parts[0].span_index == 0
    assert 0 <= edge.parts[0].start_seconds < edge.parts[0].end_seconds <= 0.1


@pytest.mark.parametrize(
    ("tokens", "times", "status"),
    [
        ((), (), "missing_alignment"),
        (("好",), (), "invalid_alignment"),
        (("好",), (float("nan"),), "invalid_alignment"),
        (("好",), (float("inf"),), "invalid_alignment"),
        (("好",), (-0.1,), "invalid_alignment"),
        (("好",), (1.1,), "invalid_alignment"),
        (("好", "吧"), (0.7, 0.2), "invalid_alignment"),
    ],
)
def test_missing_or_invalid_native_points_never_produce_identity(tokens, times, status):
    result = align_transcript("好吧", tokens, times, (span(0, 1, "A"),), 1)
    assert result.parts == () and result.status == status


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), True])
def test_invalid_duration_is_an_explicit_fallback(duration):
    assert (
        align_transcript("好", ("好",), (0.2,), (), duration).status
        == "invalid_duration"
    )


def test_invalid_segmentation_is_an_explicit_fallback():
    result = align_transcript("好", ("好",), (0.2,), (span(2, 3, "A"),), 1)
    assert result.parts == () and result.status == "invalid_spans"


def test_decoder_metadata_tokens_do_not_shift_the_remaining_text():
    result = align_transcript(
        "你好。",
        ("<|Speech|>", "<Music>", "你", "好", "。", "</Music>"),
        (0, 0.1, 0.2, 0.4, 0.6, 0.7),
        (span(0, 1, "A"),),
        1,
    )
    assert result.status == "aligned"
    assert result.parts[0].text == "你好。" and result.parts[0].span_index == 0
