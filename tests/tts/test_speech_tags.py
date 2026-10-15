import pytest

from fish_memory.tts.speech_tags import (
    CONTROL_TAGS,
    SpeechTagFilter,
    SpeechTagRepeater,
    SpeechTextBuffer,
    speech_prefix,
    strip_speech_tags,
    supports_speech_tags,
)


@pytest.mark.parametrize("split", range(1, 10))
def test_split_tag_never_leaks_and_unknown_brackets_remain(split):
    value = "[excited]你好😀，[版本二]已经准备好了。[laughing]"
    parser = SpeechTagFilter()
    output = parser.feed(value[:split]) + parser.feed(value[split:]) + parser.finish()
    assert output == "你好😀，[版本二]已经准备好了。"


@pytest.mark.parametrize("tag", ["disappointed", "pouting", "pause"])
def test_unsupported_voice_label_is_hidden_from_display_and_audio(tag):
    raw = f"[{tag}]啊？为什么我得吃肥的呀？"
    for split in range(1, len(tag) + 3):
        parser = SpeechTagFilter()
        visible = parser.feed(raw[:split]) + parser.feed(raw[split:])
        visible += parser.finish()
        assert visible == "啊？为什么我得吃肥的呀？"
    expression, provider = SpeechTagRepeater().prepare(raw)
    assert expression == provider == "啊？为什么我得吃肥的呀？"
    assert strip_speech_tags(raw) == "啊？为什么我得吃肥的呀？"


def test_unsupported_voice_label_does_not_erase_literal_brackets():
    raw = "[sad]你说的[not a tag]和[版本二]，我记下了。[pause]继续。"
    parser = SpeechTagFilter()
    visible = parser.feed(raw) + parser.finish()
    expression, provider = SpeechTagRepeater().prepare(raw)
    assert expression == "[sad]你说的[not a tag]和[版本二]，我记下了。继续。"
    assert strip_speech_tags(provider) == visible
    assert "[pause]" not in provider
    assert visible == "你说的[not a tag]和[版本二]，我记下了。继续。"


def test_capability_checks_provider_and_exact_model():
    assert supports_speech_tags("dashscope", "qwen-audio-3.0-tts-flash")
    assert not supports_speech_tags("local", "qwen-audio-3.0-tts-flash")
    assert not supports_speech_tags("dashscope", "qwen3-tts-flash-realtime")
    assert not supports_speech_tags("dashscope", "cosyvoice-v3-flash")


def test_split_speech_keeps_style_within_sentence_and_effect_once():
    buffer = SpeechTextBuffer()
    raw = (
        "[excited]这次终于把问题修好了，[laughing]"
        "现在可以继续测试了。[serious]先检查声音。"
    )
    chunks = []
    for char in raw:
        chunks.extend(buffer.append(char))
    chunks.extend(buffer.finish())
    assert chunks == [
        "[excited]这次终于把问题修好了，[laughing]",
        "[excited]现在可以继续测试了。",
        "[serious]先检查声音。",
    ]
    assert "".join(map(strip_speech_tags, chunks)) == strip_speech_tags(raw)


def test_does_not_cut_tiny_interjection_and_preserves_complete_final_tag():
    buffer = SpeechTextBuffer()
    assert buffer.append("[excited]太好了，接着检查") == []
    chunks = buffer.append("声音。[laugh")
    chunks.extend(buffer.append("ing]"))
    chunks.extend(buffer.finish())
    assert chunks == ["[excited]太好了，接着检查声音。[laughing]"]


def test_prefix_adoption_skips_spoken_words_without_replaying_sound():
    raw = "[excited]这次终于把问题修好了，[laughing]现在继续。"
    prefix = "这次终于把问题修好了，"
    assert strip_speech_tags(speech_prefix(raw, len(prefix))) == prefix
    buffer = SpeechTextBuffer()
    chunks = (
        buffer.adopt_sent_prefix(
            raw, len(prefix), speech_offset=len("[excited]" + prefix)
        )
        + buffer.finish()
    )
    assert chunks == ["[excited][laughing]现在继续。"]


@pytest.mark.parametrize("speech_offset", [5, 100])
def test_rejected_prefix_adoption_does_not_duplicate_fallback_speech(speech_offset):
    raw = "[trembling]好嘛我走就是了……"
    buffer = SpeechTextBuffer()
    with pytest.raises(ValueError, match="marked prefix"):
        buffer.adopt_sent_prefix(raw, 0, speech_offset=speech_offset)
    chunks = buffer.append(raw) + buffer.finish()
    assert "".join(map(strip_speech_tags, chunks)) == "好嘛我走就是了……"


def test_rejected_prefix_adoption_preserves_existing_pending_speech():
    buffer = SpeechTextBuffer()
    buffer.append("[whispers]原来的半句")
    with pytest.raises(ValueError, match="marked prefix"):
        buffer.adopt_sent_prefix("[sad]另一句。", 0, speech_offset=100)
    assert buffer.finish() == ["[whispers]原来的半句"]


def test_semantic_flush_includes_sound_without_speaking_control_only():
    buffer = SpeechTextBuffer()
    buffer.append("你好。[laughing]")
    assert buffer.flush_through(3) == ["你好。[laughing]"]
    assert buffer.finish() == []
    buffer.append("[excited]")
    assert buffer.finish() == []
    buffer.append("[sad][sighing]")
    assert buffer.flush_through(3) == ["[sad][sighing]"]


@pytest.mark.parametrize("control", CONTROL_TAGS)
def test_provider_reasserts_control_at_each_clause_without_changing_source(control):
    tag = f"[{control}]"
    raw = tag + "伊利丹，我现在饿得直抽抽，连口热饭都没有，好惨啊呜呜"
    expression, provider = SpeechTagRepeater().prepare(raw)
    assert expression == raw
    assert provider == (
        f"{tag}伊利丹，{tag}我现在饿得直抽抽，{tag}连口热饭都没有，{tag}好惨啊呜呜"
    )
    assert strip_speech_tags(provider) == strip_speech_tags(raw)


def test_provider_continuity_survives_sentences_but_not_new_utterance():
    tags = SpeechTagRepeater()
    tags.prepare("[whispers]先轻声说。")
    expression, provider = tags.prepare("接下来也轻声说，别吵醒它。")
    assert expression == "[whispers]接下来也轻声说，别吵醒它。"
    assert provider == "[whispers]接下来也轻声说，[whispers]别吵醒它。"
    assert SpeechTagRepeater().prepare("正常说话。") == ("正常说话。", "正常说话。")


def test_provider_explicit_change_overrides_style_and_effects_stay_once():
    tags = SpeechTagRepeater()
    raw = "[whispers]小声说，[sighing][serious]现在认真讲，别走神。[laughing]"
    expression, provider = tags.prepare(raw)
    assert expression == raw
    assert provider == (
        "[whispers]小声说，[sighing][serious]现在认真讲，[serious]别走神。[laughing]"
    )
    assert tags.prepare("[excited]完成了，真好！")[1] == (
        "[excited]完成了，[excited]真好！"
    )
    assert provider.count("[sighing]") == provider.count("[laughing]") == 1


def test_provider_preserves_numeric_punctuation_and_unknown_brackets():
    raw = "[serious]金额1,000.25，时间12:30；地址docs.example.com。\n[备注]到时见。"
    expression, provider = SpeechTagRepeater().prepare(raw)
    assert expression == raw
    assert provider == (
        "[serious]金额1,000.25，[serious]时间12:30；"
        "[serious]地址docs.example.com。\n[serious][备注]到时见。"
    )
    assert strip_speech_tags(provider) == strip_speech_tags(raw)


def test_provider_reassertion_leaves_precise_draft_cursor_unchanged():
    raw = "[whispers]伊利丹，我现在饿得直抽抽，连口热饭都没有。"
    draft = SpeechTextBuffer()
    first = draft.append(raw)[0]
    _, provider = SpeechTagRepeater().prepare(first)
    assert provider == "[whispers]伊利丹，[whispers]我现在饿得直抽抽，"
    assert len(provider) > len(first)
    adopted = SpeechTextBuffer()
    remaining = (
        adopted.adopt_sent_prefix(
            raw, len(strip_speech_tags(first)), speech_offset=len(first)
        )
        + adopted.finish()
    )
    assert remaining == ["[whispers]连口热饭都没有。"]
