import pytest

from fish_memory.agent.routing import AgentTaskDecisionKind
from fish_memory.answer.lane_stream import (
    HomeLaneStreamParser,
    RecallLaneStreamParser,
    TaskLaneStreamParser,
    WebLaneStreamParser,
)
from fish_memory.pet.types import PetEmotion
from tests.answer.task_lane_wire_samples import INVALID_TASK_LANE_WIRE_SAMPLES


def test_recall_lane_streams_emotion_then_answer_before_recall_decision() -> None:
    parser = RecallLaneStreamParser()
    payload = '{"m":"shy","a":"我想一下。","r":1,"q":"用户最近使用的音乐服务","e":0}'
    emotions = []
    answer = []
    depths = []
    queries = []
    answer_completed_at = []

    for character in payload:
        delta = parser.feed(character)
        if delta.emotion is not None:
            assert "".join(answer) == ""
            emotions.append(delta.emotion)
        answer.append(delta.answer)
        if delta.answer_complete:
            answer_completed_at.append("".join(answer))
        if delta.recall_depth is not None:
            assert "".join(answer) == "我想一下。"
            depths.append(delta.recall_depth)
        if delta.query is not None:
            queries.append(delta.query)
    parser.finish()

    assert emotions == [PetEmotion.SHY]
    assert "".join(answer) == "我想一下。"
    assert depths == [1]
    assert queries == ["用户最近使用的音乐服务"]
    assert answer_completed_at == ["我想一下。"]


def test_recall_lane_direct_answer_omits_query() -> None:
    parser = RecallLaneStreamParser()
    delta = parser.feed('{"m":"neutral","a":"答案是四十二。","r":0,"e":0}')
    parser.finish()

    assert delta.emotion is PetEmotion.NEUTRAL
    assert delta.answer == "答案是四十二。"
    assert delta.answer_complete is True
    assert delta.recall_depth == 0
    assert delta.query is None
    assert delta.end is False


@pytest.mark.parametrize(
    "chunks",
    [
        ['{"m":"neutral","a":"我这就去查看您的当前位置。","r":0}'],
        ['{"m":"neutral","a":"我这就去查看您的当前位置。","r":0', "}"],
        list('{"m":"neutral","a":"我这就去查看您的当前位置。","r":0}'),
    ],
    ids=["same-chunk", "separate-close", "character-chunks"],
)
def test_recall_lane_missing_optional_end_defaults_to_false(
    chunks: list[str],
) -> None:
    parser = RecallLaneStreamParser()
    ends: list[bool] = []

    for chunk in chunks:
        delta = parser.feed(chunk)
        if delta.end is not None:
            ends.append(delta.end)
    parser.finish()

    assert ends == [False]


def test_recall_lane_depth_one_missing_optional_end_defaults_to_false() -> None:
    parser = RecallLaneStreamParser()

    delta = parser.feed('{"m":"neutral","a":"让我想想。","r":1,"q":"用户当前位置"}')
    parser.finish()

    assert delta.recall_depth == 1
    assert delta.query == "用户当前位置"
    assert delta.end is False


@pytest.mark.parametrize(
    "payload",
    [
        '{"m":"neutral","a":"缺少记忆判定。"}',
        '{"m":"neutral","a":"缺少查询。","r":1}',
    ],
    ids=["missing-r", "depth-one-missing-q"],
)
def test_recall_lane_optional_end_does_not_relax_required_fields(payload: str) -> None:
    parser = RecallLaneStreamParser()

    with pytest.raises(ValueError):
        parser.feed(payload)
        parser.finish()


def test_recall_lane_rejects_depth_two() -> None:
    parser = RecallLaneStreamParser()

    with pytest.raises(ValueError, match="r"):
        parser.feed('{"m":"neutral","a":"我想想。","r":2')


def test_web_lane_rejection_is_one_field_object() -> None:
    parser = WebLaneStreamParser()

    delta = parser.feed('{"w":0}')
    parser.finish()

    assert delta.accepted is False
    assert delta.emotion is None
    assert delta.answer == ""
    assert delta.answer_complete is False
    assert delta.query is None


def test_web_lane_acceptance_orders_decision_emotion_answer_query() -> None:
    parser = WebLaneStreamParser()
    payload = '{"w":1,"m":"happy","a":"我去了解一下。","q":"杭州下周活动"}'
    accepted = []
    emotions = []
    answer = []
    queries = []
    answer_completed_at = []

    for character in payload:
        delta = parser.feed(character)
        if delta.accepted is not None:
            accepted.append(delta.accepted)
        if delta.emotion is not None:
            assert accepted == [True]
            assert "".join(answer) == ""
            emotions.append(delta.emotion)
        answer.append(delta.answer)
        if delta.answer_complete:
            answer_completed_at.append("".join(answer))
        if delta.query is not None:
            assert "".join(answer) == "我去了解一下。"
            queries.append(delta.query)
    parser.finish()

    assert accepted == [True]
    assert emotions == [PetEmotion.HAPPY]
    assert "".join(answer) == "我去了解一下。"
    assert queries == ["杭州下周活动"]
    assert answer_completed_at == ["我去了解一下。"]


def test_home_lane_rejection_is_one_field_object() -> None:
    parser = HomeLaneStreamParser()

    delta = parser.feed('{"h":0}')
    parser.finish()

    assert delta.accepted is False
    assert delta.emotion is None
    assert delta.answer == ""
    assert delta.answer_complete is False


def test_home_lane_acceptance_orders_decision_type_emotion_answer() -> None:
    parser = HomeLaneStreamParser()
    payload = '{"h":1,"t":"control","m":"happy","a":"我来处理一下。"}'
    accepted = []
    intents = []
    emotions = []
    answer = []
    answer_completed_at = []

    for character in payload:
        delta = parser.feed(character)
        if delta.accepted is not None:
            accepted.append(delta.accepted)
        if delta.home_intent is not None:
            assert accepted == [True]
            intents.append(delta.home_intent)
        if delta.emotion is not None:
            assert accepted == [True]
            assert intents == ["control"]
            assert "".join(answer) == ""
            emotions.append(delta.emotion)
        answer.append(delta.answer)
        if delta.answer_complete:
            answer_completed_at.append("".join(answer))
    parser.finish()

    assert accepted == [True]
    assert intents == ["control"]
    assert emotions == [PetEmotion.HAPPY]
    assert "".join(answer) == "我来处理一下。"
    assert answer_completed_at == ["我来处理一下。"]


def test_home_lane_streams_query_type_before_emotion_and_answer() -> None:
    parser = HomeLaneStreamParser()

    first = parser.feed('{"h":1,"t":"query",')
    second = parser.feed('"m":"neutral","a":"我看看。"}')
    parser.finish()

    assert first.accepted is True
    assert first.home_intent == "query"
    assert second.emotion is PetEmotion.NEUTRAL
    assert second.answer == "我看看。"


def test_home_lane_rejects_unknown_intent_type() -> None:
    parser = HomeLaneStreamParser()

    with pytest.raises(ValueError, match="t must be one of"):
        parser.feed('{"h":1,"t":"other"')


@pytest.mark.parametrize(
    ("parser", "payload"),
    [
        (RecallLaneStreamParser(), '{"r":0,"m":"neutral","a":"错序"}'),
        (WebLaneStreamParser(), '{"m":"neutral","w":0}'),
        (HomeLaneStreamParser(), '{"m":"neutral","h":0}'),
    ],
)
def test_lane_parsers_reject_wrong_field_order(parser: object, payload: str) -> None:
    with pytest.raises(ValueError):
        parser.feed(payload)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("payload", "accepted", "kind", "indexes", "emotion", "answer"),
    [
        ('{"t":0,"m":"neutral","a":""}', False, None, (), None, ""),
        (
            '{"t":1,"m":"neutral","a":"好，我来处理。"}',
            True,
            AgentTaskDecisionKind.NEW_TASK,
            (),
            PetEmotion.NEUTRAL,
            "好，我来处理。",
        ),
        (
            '{"t":1,"m":"happy","a":"我继续处理。","c":0}',
            True,
            AgentTaskDecisionKind.CONTINUE_TASK,
            (0,),
            PetEmotion.HAPPY,
            "我继续处理。",
        ),
    ],
)
def test_task_lane_accepts_compact_wire_shapes(
    payload: str,
    accepted: bool,
    kind: AgentTaskDecisionKind | None,
    indexes: tuple[int, ...],
    emotion: PetEmotion | None,
    answer: str,
) -> None:
    parser = TaskLaneStreamParser(candidate_count=2)

    delta = parser.feed(payload)
    parser.finish()

    assert delta.accepted is accepted
    assert delta.task_kind is kind
    assert delta.candidate_indexes == indexes
    assert delta.emotion is emotion
    assert delta.answer == answer
    assert delta.answer_complete is (accepted is True)


@pytest.mark.parametrize(
    "payload",
    [
        *INVALID_TASK_LANE_WIRE_SAMPLES,
        '{"t":0,"m":"neutral","a":"","c":0}',
        '{"t":3,"m":"neutral","a":"好。"}',
        '{"t":1,"m":"neutral","a":""}',
    ],
)
def test_task_lane_rejects_wrong_order_or_invalid_compact_shape(
    payload: str,
) -> None:
    parser = TaskLaneStreamParser(candidate_count=2)

    with pytest.raises(ValueError):
        parser.feed(payload)
        parser.finish()


def test_not_task_rejects_a_nonempty_model_answer() -> None:
    parser = TaskLaneStreamParser(candidate_count=0)

    with pytest.raises(ValueError, match="empty"):
        parser.feed('{"t":0,"m":"angry","a":"不要公开这段"}')
        parser.finish()


def test_task_lane_t_one_omitted_candidate_is_new_task() -> None:
    parser = TaskLaneStreamParser(candidate_count=0)

    delta = parser.feed('{"t":1,"m":"happy","a":"我来处理。"}')
    parser.finish()

    assert delta.task_kind is AgentTaskDecisionKind.NEW_TASK
    assert delta.candidate_indexes == ()


@pytest.mark.parametrize(
    "answer",
    [
        "请告诉我您希望怎么处理？",
        "Which target should I use?",
    ],
)
def test_task_lane_rejects_confirmation_questions(answer: str) -> None:
    parser = TaskLaneStreamParser(candidate_count=0)
    payload = f'{{"t":1,"m":"neutral","a":"{answer}"}}'

    with pytest.raises(ValueError, match="must not ask"):
        parser.feed(payload)


def test_task_lane_ignores_hallucinated_candidate_when_none_were_supplied() -> None:
    parser = TaskLaneStreamParser(candidate_count=0)

    delta = parser.feed('{"t":1,"m":"neutral","a":"我来处理。","c":0}')
    parser.finish()

    assert delta.accepted is True
    assert delta.task_kind is AgentTaskDecisionKind.NEW_TASK
    assert delta.candidate_indexes == ()


def test_task_lane_acceptance_waits_for_complete_route() -> None:
    parser = TaskLaneStreamParser(candidate_count=2)

    assert parser.feed('{"t":1,').accepted is None


def test_task_lane_new_task_waits_for_the_optional_c_or_object_close() -> None:
    parser = TaskLaneStreamParser(candidate_count=0)

    assert parser.feed('{"t":1,').accepted is None
    partial = parser.feed('"m":"happy","a":"我来处理。"')

    assert partial.answer_complete is True
    assert partial.task_kind is None

    complete = parser.feed("}")
    parser.finish()

    assert complete.task_kind is AgentTaskDecisionKind.NEW_TASK
    assert complete.accepted is True
    assert complete.candidate_indexes == ()


def test_task_lane_missing_c_with_candidates_remains_a_new_task() -> None:
    parser = TaskLaneStreamParser(candidate_count=1)

    delta = parser.feed('{"t":1,"m":"happy","a":"我来处理。"}')
    parser.finish()

    assert delta.task_kind is AgentTaskDecisionKind.NEW_TASK
    assert delta.candidate_indexes == ()


def test_task_lane_keeps_candidate_when_closing_brace_is_a_separate_chunk() -> None:
    parser = TaskLaneStreamParser(candidate_count=1)

    selected = parser.feed('{"t":1,"m":"neutral","a":"我来查。","c":0')
    closed = parser.feed("}")
    parser.finish()

    assert selected.task_kind is AgentTaskDecisionKind.CONTINUE_TASK
    assert selected.candidate_indexes == (0,)
    assert closed.task_kind is AgentTaskDecisionKind.CONTINUE_TASK
    assert closed.candidate_indexes == (0,)


def test_task_lane_keeps_candidate_across_single_character_chunks() -> None:
    parser = TaskLaneStreamParser(candidate_count=1)
    payload = '{"t":1,"m":"neutral","a":"我来查。","c":0}'
    latest_kind: AgentTaskDecisionKind | None = None
    latest_indexes: tuple[int, ...] = ()

    for character in payload:
        delta = parser.feed(character)
        if delta.task_kind is not None:
            latest_kind = delta.task_kind
            latest_indexes = delta.candidate_indexes
    parser.finish()

    assert latest_kind is AgentTaskDecisionKind.CONTINUE_TASK
    assert latest_indexes == (0,)


@pytest.mark.parametrize("candidate_count", [True, "2", -1])
def test_task_lane_candidate_count_requires_a_non_negative_integer(
    candidate_count: object,
) -> None:
    with pytest.raises(ValueError, match="candidate_count"):
        TaskLaneStreamParser(candidate_count=candidate_count)  # type: ignore[arg-type]


def test_an_emotion_outside_the_list_falls_back_to_neutral_and_the_answer_survives():
    parser = RecallLaneStreamParser()
    answer = []
    emotions = []
    for character in '{"m":"sarcastic","a":"晚上看星星挺浪漫的。","r":0,"e":0}':
        delta = parser.feed(character)
        answer.append(delta.answer)
        if delta.emotion is not None:
            emotions.append(delta.emotion)
    parser.finish()

    assert emotions == [PetEmotion.NEUTRAL]
    assert "".join(answer) == "晚上看星星挺浪漫的。"


def test_a_runaway_emotion_is_still_a_broken_stream():
    parser = RecallLaneStreamParser()
    with pytest.raises(ValueError):
        parser.feed('{"m":"' + "x" * 40)
