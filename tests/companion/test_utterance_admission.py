import pytest

from fish_memory.companion.utterance_admission import (
    classify_utterance,
    compact_utterance_text,
    low_evidence_utterance_reason,
)


def test_compact_utterance_normalizes_nfkc_and_removes_unicode_separators() -> None:
    assert compact_utterance_text(" \u3000Ａ，\tＢ。\n") == "ab"


def test_compact_utterance_removes_unicode_non_breaking_spaces() -> None:
    assert compact_utterance_text("嗯\u00a0\u2007\u202f哼") == "嗯哼"


@pytest.mark.parametrize("text", ["。！？", "…… ——", "\u3000\t\n"])
def test_compact_utterance_can_be_empty(text: str) -> None:
    assert compact_utterance_text(text) == ""


@pytest.mark.parametrize(
    "event",
    [
        "<|BGM|>",
        "<|Breath|>",
        "<|Cough|>",
        "<|Sneeze|>",
        "cOuGh",
    ],
)
def test_explicit_sensevoice_non_speech_event_is_rejected(event: str) -> None:
    assert classify_utterance("停止", event) == "non_speech_event"


@pytest.mark.parametrize("event", ["", "<|Speech|>", "<|Unknown|>"])
def test_other_sensevoice_events_do_not_reject_lexical_speech(event: str) -> None:
    assert classify_utterance("停止", event) is None


@pytest.mark.parametrize(
    "event",
    [
        "<|Speech|><|Cough|>",
        "<|BGM|><|Speech|>",
        "speech,breath",
    ],
)
def test_speech_label_prevents_mixed_event_from_being_explicit_non_speech(
    event: str,
) -> None:
    assert classify_utterance("停止", event) is None


@pytest.mark.parametrize(
    "text",
    [
        "嗯。",
        "哼！",
        "咳",
        "嗯哼",
        "嗯，哼……咳。",
        "(嗯), [哼]... 咳?!",
    ],
)
def test_standalone_nonlexical_vocalizations_are_rejected(text: str) -> None:
    assert classify_utterance(text, "<|Speech|>") == "nonlexical_vocalization"


@pytest.mark.parametrize(
    "text",
    [
        "好",
        "停",
        "对",
        "不",
        "开灯",
        "你好",
        "嗯好",
        "哼歌",
        "咳嗽",
        "嗯，我想看下一集",
        "哼，我不看了",
        "咳嗽得厉害",
    ],
)
def test_short_commands_and_lexical_sentences_are_admitted(text: str) -> None:
    assert classify_utterance(text, "<|Speech|>") is None


def test_allow_nonlexical_only_exempts_ambiguous_vocalization() -> None:
    assert classify_utterance("嗯。", "<|Speech|>", allow_nonlexical=True) is None
    assert (
        classify_utterance("嗯。", "<|Cough|>", allow_nonlexical=True)
        == "non_speech_event"
    )


def test_pipeline_compatibility_entry_point_uses_the_same_rules() -> None:
    assert low_evidence_utterance_reason("哼。", "<|Speech|>") == (
        "nonlexical_vocalization"
    )
    assert (
        low_evidence_utterance_reason(
            "哼。",
            "<|Speech|>",
            allow_nonlexical=True,
        )
        is None
    )


@pytest.mark.parametrize("text", ["", "。", " \u3000！\t"])
def test_empty_compact_text_is_left_to_other_admission_rules(text: str) -> None:
    assert classify_utterance(text) is None
