import pytest

from fish_memory.companion.text_buffer import TtsTextBuffer


def speech_buffer() -> TtsTextBuffer:
    return TtsTextBuffer(first_chunk_at_pause=True)


@pytest.mark.parametrize(
    "prefix",
    [
        "好，",
        "嗯。",
        "哈？",
        "可以！",
        "先说这句；",
        "接下来：",
        "Hi,",
        "Yes!",
        "😊好，",
    ],
)
def test_first_spoken_pause_does_not_wait_for_fifteen_characters(prefix: str) -> None:
    buffer = speech_buffer()
    for char in prefix:
        assert buffer.append(char) == []
    assert buffer.append("后") == [prefix]
    assert buffer.take_first_pause_flush() is True
    assert buffer.take_first_pause_flush() is False
    assert buffer.append("续短句，") == []
    assert buffer.finish() == ["后续短句，"]
    assert buffer.take_first_pause_flush() is False


@pytest.mark.parametrize(
    "text", ["价格是3.14元，", "余额1,000元，", "时间是12:30，", "地址 https://a.cn，"]
)
def test_numbers_and_addresses_are_not_spoken_as_partial_fragments(text: str) -> None:
    buffer = speech_buffer()
    for char in text[:-1]:
        assert buffer.append(char) == []
    assert buffer.append(text[-1]) == []
    assert buffer.append("后") == [text]


def test_leading_punctuation_does_not_create_an_empty_speech_segment() -> None:
    buffer = speech_buffer()
    assert buffer.append("……， ") == []
    assert buffer.append("好！") == []
    assert buffer.append("接") == ["……， 好！"]


@pytest.mark.parametrize("text", ["“好的！”", "好！！", "好……", "好。😊"])
def test_complete_short_reply_does_not_leave_a_punctuation_only_task(text: str) -> None:
    buffer = speech_buffer()
    for char in text:
        assert buffer.append(char) == []
    assert buffer.finish() == [text]
    assert buffer.take_first_pause_flush() is False


def test_first_pause_keeps_closing_quotes_and_repeated_punctuation() -> None:
    buffer = speech_buffer()
    assert buffer.append("“好的！！”") == []
    assert buffer.append("接下来还有内容。") == ["“好的！！”"]
    assert buffer.take_first_pause_flush() is True
    assert buffer.finish() == ["接下来还有内容。"]


def test_explicit_short_segment_consumes_the_first_chunk_allowance() -> None:
    buffer = speech_buffer()
    assert buffer.append("明白了") == []
    assert buffer.flush_through(3) == ["明白了"]
    assert buffer.flush_through(3) == []
    assert buffer.append("接下来，") == []
    assert buffer.finish() == ["接下来，"]


def test_promotion_preserves_the_sent_prefix_and_subsequent_buffering() -> None:
    buffer = speech_buffer()
    prefix = "😊好，"
    tail = "后续短句，"
    assert buffer.adopt_sent_prefix(prefix + tail, len(prefix)) == []
    assert buffer.take_first_pause_flush() is False
    assert buffer.flush_through(len(prefix + tail)) == [tail]
    assert buffer.finish() == []


def test_an_unsent_promoted_prefix_can_still_start_at_its_first_pause() -> None:
    buffer = speech_buffer()
    assert buffer.adopt_sent_prefix("好，后续短句，", 0) == ["好，"]
    assert buffer.take_first_pause_flush() is True
    assert buffer.finish() == ["后续短句，"]


def test_reset_restores_first_pause_behavior_without_replaying_old_text() -> None:
    buffer = speech_buffer()
    assert buffer.append("好，旧尾巴") == ["好，"]
    buffer.reset()
    assert buffer.take_first_pause_flush() is False
    assert buffer.append("明白，继续") == ["明白，"]
    assert buffer.finish() == ["继续"]


def test_non_speech_readiness_callers_keep_the_existing_minimum() -> None:
    buffer = TtsTextBuffer()
    assert buffer.append("好，") == []
    assert buffer.append("后续短句，") == []
    assert buffer.finish() == ["好，后续短句，"]
