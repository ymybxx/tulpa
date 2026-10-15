import pytest

from fish_memory.answer.structured_stream import StructuredAnswerStreamParser
from fish_memory.pet.types import PetEmotion


def parse_chunks(chunks: list[str]) -> tuple[list[int], list[bool], str]:
    parser = StructuredAnswerStreamParser(enable_web_search=True)
    depths: list[int] = []
    searches: list[bool] = []
    parts: list[str] = []
    for chunk in chunks:
        delta = parser.feed(chunk)
        if delta.recall_depth is not None:
            depths.append(delta.recall_depth)
            assert parts == []
        if delta.need_web_search is not None:
            searches.append(delta.need_web_search)
            assert parts == []
        if delta.answer:
            parts.append(delta.answer)
    parser.finish()
    return depths, searches, "".join(parts)


def test_decodes_compact_json_when_every_character_is_a_chunk() -> None:
    payload = '{"r":1,"w":0,"a":"你好\\n\\"鱼\\"🐟","e":1}'

    depths, searches, answer = parse_chunks(list(payload))

    assert depths == [1]
    assert searches == [False]
    assert answer == '你好\n"鱼"🐟'


def test_rejects_recall_depth_two_before_later_fields_arrive() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    with pytest.raises(ValueError, match="r"):
        parser.feed('{"r":2')


def test_emits_web_search_before_answer_arrives() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    recall = parser.feed('{"r":0')
    search = parser.feed(',"w":1')
    answer = parser.feed(',"a":"我看看最新消息。","e":0}')
    parser.finish()

    assert recall.recall_depth == 0
    assert search.need_web_search is True
    assert search.answer == ""
    assert answer.answer == "我看看最新消息。"


def test_emits_end_only_after_answer() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    first = parser.feed('{"r":0,"w":0,"a":"回答","e":')
    second = parser.feed("1}")
    parser.finish()

    assert first.answer == "回答"
    assert first.end is None
    assert second.end is True


def test_missing_optional_end_field_defaults_to_continuing() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    delta = parser.feed('{"r":0,"w":0,"m":"neutral","a":"讲完了。"}')
    parser.finish()

    assert delta.answer == "讲完了。"
    assert delta.end is False


def test_decodes_unicode_escape_and_surrogate_pair_across_chunks() -> None:
    payload = '{"r":1,"w":0,"a":"\\u4f60\\u597d\\ud83d\\udc1f","e":0}'

    depths, searches, answer = parse_chunks(
        [payload[:24], payload[24:31], payload[31:39], payload[39:]]
    )

    assert depths == [1]
    assert searches == [False]
    assert answer == "你好🐟"


@pytest.mark.parametrize(
    "payload",
    [
        '{"w":0,"r":1,"a":"hello","e":0}',
        '{"r":3,"w":0,"a":"hello","e":0}',
        '{"r":1,"w":2,"a":"hello","e":0}',
        '{"r":1,"w":0,"a":" ","e":0}',
        '{"r":1,"w":0,"a":"hello","e":2}',
        '{"r":1,"w":0,"a":"hello","extra":true,"e":0}',
        '{"r":1,"w":0,"a":"hello","e":0} trailing',
        '{"r":1,"w":0,"a":"\\udc1f","e":0}',
        '{"r":1,"w":0,"a":"\\ud83dX","e":0}',
        (
            '{"contextComplete":false,"recallDepth":1,'
            '"recallQuery":"query","answer":"hello","end":false}'
        ),
    ],
)
def test_rejects_invalid_or_legacy_structured_answer(payload: str) -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    with pytest.raises(ValueError):
        parser.feed(payload)
        parser.finish()


def test_finish_rejects_incomplete_json() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)
    parser.feed('{"r":1,"w":0,"a":"partial')

    with pytest.raises(ValueError, match="incomplete"):
        parser.finish()


def test_exposes_raw_response_for_performance_logging() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)
    payload = '{"r":0,"w":0,"a":"hello","e":0}'

    delta = parser.feed(payload)
    parser.finish()

    assert delta.recall_depth == 0
    assert delta.need_web_search is False
    assert delta.answer == "hello"
    assert delta.end is False
    assert parser.raw_response == payload


def test_emits_emotion_before_answer_text() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    metadata = parser.feed('{"r":0,"w":0,"m":"shy"')
    answer = parser.feed(',"a":"欢迎回来。","e":0}')
    parser.finish()

    assert metadata.emotion is PetEmotion.SHY
    assert metadata.answer == ""
    assert answer.answer == "欢迎回来。"


def test_rejects_removed_emotion_values() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    with pytest.raises(ValueError, match="m"):
        parser.feed('{"r":0,"w":0,"m":"affectionate"')


def test_new_protocol_parses_when_every_character_is_a_chunk() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)
    payload = '{"r":0,"w":0,"m":"happy","a":"你好。","e":0}'
    emotions: list[PetEmotion] = []
    answer: list[str] = []

    for character in payload:
        delta = parser.feed(character)
        if delta.emotion is not None:
            emotions.append(delta.emotion)
        answer.append(delta.answer)
    parser.finish()

    assert emotions == [PetEmotion.HAPPY]
    assert "".join(answer) == "你好。"


def test_disabled_search_protocol_omits_web_search_decision() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=False)
    payload = '{"r":0,"m":"happy","a":"你好。","e":0}'
    depths: list[int] = []
    searches: list[bool] = []
    answer: list[str] = []

    for character in payload:
        delta = parser.feed(character)
        if delta.recall_depth is not None:
            depths.append(delta.recall_depth)
        if delta.need_web_search is not None:
            searches.append(delta.need_web_search)
        answer.append(delta.answer)
    parser.finish()

    assert depths == [0]
    assert searches == []
    assert "".join(answer) == "你好。"


def test_disabled_search_parser_rejects_web_search_field() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=False)

    with pytest.raises(ValueError):
        parser.feed('{"r":0,"w":0,"m":"neutral","a":"你好","e":0}')


def test_enabled_search_parser_requires_web_search_field() -> None:
    parser = StructuredAnswerStreamParser(enable_web_search=True)

    with pytest.raises(ValueError):
        parser.feed('{"r":0,"m":"neutral","a":"你好","e":0}')
