import random
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.experimental.config import ExperimentalConfiguration
from fish_memory.experimental.context import recent_speech, timeline
from fish_memory.experimental.jev import Judge
from fish_memory.experimental.materials import (
    AWAY_SECONDS,
    MATERIAL_OBSERVATION,
    Boredom,
    MaterialScout,
    OpenQuestion,
    cue_text,
    last_question,
)
from fish_memory.experimental.thoughts import fresh_thoughts
from fish_memory.history.models import HistoryUnit
from fish_memory.recall.models import RecallCandidate, RecallFactHead
from tests.test_experimental_companion import NOW, setup

HISTORY = {
    "role": "user",
    "context_reference": True,
    "content": (
        "History:\n"
        "2026-09-19 06:50:00-05:00 - 2026-09-19 07:00:00-05:00\n"
        "u: 我周末想去爬山。\n"
        "a: 好呀，你想去哪座山？\n"
        "as: 屏幕上在放一段登山视频。\n"
        "  画面里有人在攀岩。\n"
        "event: 宠物的内心想法（尚未表达）：不知道山顶的风大不大？"
    ),
}
MESSAGES = [
    {"role": "system", "content": "你喜欢音乐。"},
    {"role": "user", "content": "# 记忆摘要\n用户喜欢爬山", "context_reference": True},
    HISTORY,
    {"role": "assistant", "content": "那我陪你看完这段视频吧(被打断)"},
    {"role": "user", "content": "<message_time>12:00</message_time>\n好的"},
]


def test_timeline_reads_history_lines_and_native_turns_in_order():
    assert timeline(MESSAGES) == [
        ("u", "我周末想去爬山。"),
        ("a", "好呀，你想去哪座山？"),
        ("as", "屏幕上在放一段登山视频。\n画面里有人在攀岩。"),
        ("event", "宠物的内心想法（尚未表达）：不知道山顶的风大不大？"),
        ("a", "那我陪你看完这段视频吧(被打断)"),
        ("u", "好的"),
    ]
    assert recent_speech(MESSAGES, 1) == ["那我陪你看完这段视频吧"]
    assert recent_speech(MESSAGES, 0) == []


def test_thoughts_that_restate_a_spoken_line_are_not_fresh():
    spoken = recent_speech(MESSAGES, 6)
    candidates = ("好呀，你到底想去哪座山呢？", "山上的日落应该很好看。")
    assert fresh_thoughts(candidates, spoken) == (candidates[1],)


def test_cue_is_the_latest_lines_and_thought():
    assert cue_text(MESSAGES).splitlines()[-1] == "不知道山顶的风大不大？"
    assert "我周末想去爬山。" in cue_text(MESSAGES)


def test_last_question_takes_the_final_question_sentence():
    text = "你看，大海还在拍岸呢，哗啦……哗啦……是不是很像在伴奏呀？"
    assert last_question(text) == "是不是很像在伴奏呀？"
    assert last_question("嗨～我在这里呢！今天过得怎么样呀？") == "今天过得怎么样呀？"
    assert last_question("好的。") is None


def test_boredom_grows_with_stale_time_and_news_lowers_it():
    boredom = Boredom()
    boredom.grow(600)
    assert boredom.level == 0 and boredom.chance(60) == 0
    for _ in range(20):
        boredom.note("duplicate")
    # How many stale rounds there were does not matter, only for how long.
    assert boredom.level == 0
    boredom.grow(300)
    assert 0.49 < boredom.level < 0.5
    boredom.grow(600)
    assert boredom.level == 1
    # About two minutes on average at full boredom, never certain in one check.
    assert 0.39 < boredom.chance(60) < 0.4
    boredom.news("as")
    assert boredom.level == 0.5
    assert boredom.chance(60) < 0.12
    boredom.news("a")
    assert boredom.level == 0.5
    boredom.news("u")
    assert boredom.level == 0
    for _ in range(20):
        boredom.note("fresh")
    boredom.grow(600)
    # Thoughts that keep finding something new keep TA from getting bored.
    assert boredom.level < 0.01


class Store:
    def __init__(self):
        self.surfaced_keys: set[str] = set()
        self.interrupted_rows: list[dict] = []
        self.asked_rows: list[dict] = []
        self.plans: list[dict] = []
        self.facts: list[dict] = []

    async def surfaced(self, fish, since):
        return set(self.surfaced_keys)

    async def interrupted(self, fish, session, since, until):
        return self.interrupted_rows

    async def asked(self, fish, session, since, until):
        return self.asked_rows

    async def due_plans(self, fish, since, until):
        return self.plans

    async def old_facts(self, fish, before):
        return self.facts


class Embedder:
    space = SimpleNamespace(dimensions=3)

    async def embed_query(self, texts):
        self.cue = texts[0]
        return SimpleNamespace(vectors=((0.1, 0.2, 0.3),))


class Recall:
    def __init__(self, *facts):
        self.heads = [
            RecallFactHead(
                RecallCandidate(
                    uuid4(),
                    uuid4(),
                    "user",
                    text,
                    None,
                    None,
                    NOW - timedelta(days=2),
                    (),
                    semantic_score=score,
                ),
                (),
                None,
                None,
            )
            for text, score in facts
        ]

    async def search_fact_head(self, fish, dimensions, vector, **kwargs):
        return self.heads


class Observations:
    def __init__(self):
        self.events = []

    async def record_event(self, fish, event, **kwargs):
        self.events.append(event)
        return event


def scout(*, idle=5.0, recall=None, draw=0.0):
    f = setup()
    f.service.configure(
        f.service.configuration.model_copy(update={"thoughts_enabled": True})
    )
    f.service.observations = Observations()
    f.store = Store()
    f.contexts = SimpleNamespace()
    f.target = (uuid4(), uuid4())
    f.idle = idle

    async def load(*args, **kwargs):
        return LoadedAnswerContext(MESSAGES, ())

    f.contexts.load_for_proactive = load
    f.searches = []

    async def search(context, question):
        f.searches.append(question)
        return "山顶常有大风，建议带防风外套。"

    f.service.search = search
    rng = random.Random(7)
    rng.random = lambda: draw
    f.scout = MaterialScout(
        f.service,
        f.contexts,
        SimpleNamespace(),
        f.store,
        recall=recall,
        embedder=Embedder(),
        binding=lambda: f.target,
        idle=lambda: f.idle,
        rng=rng,
        start=False,
    )
    return f


@pytest.mark.asyncio
async def test_nothing_turns_up_without_boredom():
    f = scout(recall=Recall(("用户去年爬过华山", 0.9)), draw=0.001)
    assert await f.scout.check() is None
    assert f.service.observations.events == []


@pytest.mark.asyncio
async def test_a_bored_scout_recalls_a_memory_tied_to_the_moment():
    f = scout(recall=Recall(("该用户去年和该助手聊过华山", 0.9), ("用户喜欢吃辣", 0.7)))
    f.scout.boredom.level = 1
    f.clock.now = NOW + timedelta(seconds=10)
    material = await f.scout.check()
    assert material is not None and material.kind == "recall"
    assert material.cause == "cue"
    assert material.text == ("想起：用户去年和我聊过华山（2天前记下的）")
    [event] = f.service.observations.events
    assert event.observation_type == MATERIAL_OBSERVATION
    assert event.payload["context_text"] == material.text
    assert event.payload["cause"] == "cue" and event.payload["boredom"] == 1


@pytest.mark.asyncio
async def test_memories_already_surfaced_or_on_screen_are_not_recalled_again():
    f = scout(recall=Recall(("用户去年爬过华山", 0.9), ("用户周末想去爬山", 0.95)))
    f.store.surfaced_keys.add(f"recall:{f.scout.recall.heads[0].fact.id}")
    f.scout.boredom.level = 1
    f.clock.now = NOW + timedelta(seconds=10)
    # Only the curious thought is left: TA goes to look it up instead.
    material = await f.scout.check()
    assert material is not None and material.kind == "browse"
    assert material.cause == "curiosity"
    assert f.searches == ["不知道山顶的风大不大？"]
    assert material.text == (
        "刚看到：想知道“不知道山顶的风大不大？”，查到：山顶常有大风，建议带防风外套。"
    )


@pytest.mark.asyncio
async def test_browsing_needs_the_user_at_the_computer_and_stays_rare():
    f = scout(idle=200.0)
    f.scout.boredom.level = 1
    f.clock.now = NOW + timedelta(seconds=10)
    assert await f.scout.check() is None
    f.idle = 5.0
    for _ in range(5):
        f.store.surfaced_keys.clear()
        f.clock.now += timedelta(seconds=10)
        await f.scout.check()
    assert len(f.searches) == 3


@pytest.mark.asyncio
async def test_nothing_turns_up_while_the_user_is_away():
    f = scout(idle=AWAY_SECONDS + 1, recall=Recall(("用户去年爬过华山", 0.9)))
    f.scout.boredom.level = 1
    f.clock.now = NOW + timedelta(seconds=10)
    assert await f.scout.check() is None


@pytest.mark.asyncio
async def test_coming_back_picks_up_a_thread_left_open():
    f = scout(idle=AWAY_SECONDS + 1)
    message = uuid4()
    f.store.interrupted_rows = [
        {
            "id": message,
            "content": "我给你讲讲那个故事",
            "unsaid": "，后来小猫终于找到了回家的路。",
            "created_at": NOW - timedelta(minutes=20),
        }
    ]
    f.clock.now = NOW + timedelta(seconds=10)
    assert await f.scout.check() is None
    f.idle = 1.0
    f.clock.now += timedelta(seconds=5)
    assert await f.scout.check() is None  # a moment after coming back
    f.clock.now += timedelta(seconds=45)
    material = await f.scout.check()
    assert material is not None and material.kind == "pending"
    assert material.cause == "interrupted"
    assert material.key == f"pending:interrupted:{message}"
    assert material.text == (
        "还没聊完：21分钟前说到“我给你讲讲那个故事”时被打断了，"
        "没说完的是：后来小猫终于找到了回家的路。"
    )


@pytest.mark.asyncio
async def test_open_questions_ride_along_once_and_wait_until_the_user_speaks():
    f = scout()
    asked = uuid4()
    f.store.asked_rows = [
        {
            "id": asked,
            "content": "你周末想去哪座山呀？",
            "consumed_offset": None,
            "created_at": NOW - timedelta(minutes=5),
            "next_at": None,
        }
    ]
    f.clock.now = NOW + timedelta(seconds=10)
    await f.scout.check()
    [question] = f.scout.open_questions()
    assert question.text == "你周末想去哪座山呀？"
    f.scout.judged((question,), 0)
    assert f.scout.open_questions() == ()
    f.scout.boredom.level = 1
    f.clock.now += timedelta(seconds=10)
    material = await f.scout.check()
    assert material is not None and material.cause == "question"
    assert material.text == (
        "还没聊完：5分钟前问过“你周末想去哪座山呀？”，还没等到回答。"
    )
    f.scout.judged((question,), 0)
    f.scout.observe(HistoryUnit(uuid4(), "u", "去泰山", NOW, NOW))
    assert f.scout._open == {}


class Answers(Judge):
    def __init__(self, choice):
        self.choice = choice
        self.questions = None

    async def evaluate(self, configuration, state, questions, *, kind):
        self.questions = questions
        answers = {"next_action": {"type": "choice", "choice": "wait"}}
        if self.choice is not None:
            answers["open_question"] = {"type": "choice", "choice": self.choice}
        return {"answers": answers}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("choice", "expected"), [("2", [1]), ("0", [None]), ("9", []), (None, [])]
)
async def test_the_decision_also_says_which_question_still_waits(choice, expected):
    judge = Answers(choice)
    seen = []
    action = await judge.decide(
        ExperimentalConfiguration(),
        "state",
        ("s", "wait"),
        open_questions=("你想去哪座山呀？", "是不是很像伴奏呀？"),
        on_open_question=seen.append,
    )
    assert action == "wait"
    assert seen == expected
    assert judge.questions["open_question"]["criteria"] == {
        "1": "“你想去哪座山呀？”",
        "2": "“是不是很像伴奏呀？”",
        "0": "都不是",
    }


@pytest.mark.asyncio
async def test_decisions_without_open_questions_ask_nothing_extra():
    judge = Answers(None)
    await judge.decide(ExperimentalConfiguration(), "state", ("s", "wait"))
    assert set(judge.questions) == {"next_action"}


@pytest.mark.asyncio
async def test_service_decisions_carry_the_scout_questions():
    f = scout()
    judge = Answers("1")
    f.service.judge = judge
    f.service.runtime = SimpleNamespace(materials=f.scout)
    question = OpenQuestion(uuid4(), "你想去哪座山呀？", NOW)
    f.scout._asked = (question,)
    await f.service.decide("state", ("s", "wait"))
    assert f.scout._open == {question.message_id: question}
    assert f.scout.open_questions() == ()


def test_a_memory_mostly_worded_in_the_context_is_already_known():
    from fish_memory.experimental.materials import _known, _shown

    shown = _shown(
        [
            {
                "role": "assistant",
                "content": "比如背景里的大海其实一直在轻轻拍打岸边呢，哗啦……哗啦……"
                "是不是很像在给我们伴奏呀？",
            }
        ]
    )
    assert _known(
        "该助手描述背景里的大海一直在轻轻拍打岸边，发出哗啦……哗啦……的声音", shown
    )
    assert not _known("该助手观察到当前桌面背景是大海，并认为其感觉挺舒服", shown)


@pytest.mark.asyncio
async def test_a_line_cut_off_while_being_written_is_unfinished_without_a_rest():
    f = scout()
    f.store.interrupted_rows = [
        {
            "id": uuid4(),
            "content": "我跟你说，昨天那只小猫",
            "unsaid": None,
            "created_at": NOW - timedelta(minutes=20),
        },
        {
            "id": uuid4(),
            "content": "好呀，你想去",
            "unsaid": "哪座山？",
            "created_at": NOW - timedelta(minutes=30),
        },
    ]
    f.scout.boredom.level = 1
    f.clock.now = NOW + timedelta(seconds=10)
    material = await f.scout.check()
    # The second only repeats a line TA said in full, so only the first is open.
    assert material is not None
    assert material.text == (
        "还没聊完：20分钟前说到“我跟你说，昨天那只小猫”时被打断了。"
    )


def test_only_curiosity_about_the_world_is_looked_up():
    from fish_memory.experimental.materials import _curious

    assert _curious(["不知道山顶的风大不大？"]) == "不知道山顶的风大不大？"
    assert _curious(["大海的背景一直在动，像不像我们在呼吸？"]) is None
    assert _curious(["你说‘社学会的要’，是不是在感叹生活节奏太快？"]) is None
    assert _curious(["不知道你现在忙不忙？"]) is None


@pytest.mark.asyncio
async def test_a_search_that_found_nothing_is_not_material():
    f = scout()

    async def search(context, question):
        f.searches.append(question)
        return "无相关外部事实"

    f.service.search = search
    f.scout.boredom.level = 1
    f.clock.now = NOW + timedelta(seconds=10)
    assert await f.scout.check() is None
    assert f.searches == ["不知道山顶的风大不大？"]
    assert f.service.observations.events == []


@pytest.mark.asyncio
async def test_a_question_ta_talked_past_is_not_waiting_for_an_answer():
    f = scout()
    waiting, talked_past = uuid4(), uuid4()
    f.store.asked_rows = [
        {
            "id": waiting,
            "content": "你周末想去哪座山呀？",
            "consumed_offset": None,
            "created_at": NOW - timedelta(minutes=3),
            "next_at": NOW - timedelta(minutes=1),
        },
        {
            "id": talked_past,
            "content": "是不是很像在伴奏呀？",
            "consumed_offset": None,
            "created_at": NOW - timedelta(minutes=5),
            "next_at": NOW - timedelta(minutes=5) + timedelta(seconds=15),
        },
    ]
    f.clock.now = NOW + timedelta(seconds=10)
    await f.scout.check()
    assert [q.message_id for q in f.scout.open_questions()] == [waiting]


def test_only_the_latest_open_question_is_kept():
    f = scout()
    older = OpenQuestion(uuid4(), "你周末想去哪座山呀？", NOW - timedelta(minutes=9))
    newer = OpenQuestion(uuid4(), "盖好被子了吗？", NOW - timedelta(minutes=3))
    f.scout.judged((older,), 0)
    f.scout.judged((newer,), 0)
    assert list(f.scout._open) == [newer.message_id]
    f.scout.judged((older,), 0)
    assert list(f.scout._open) == [newer.message_id]


@pytest.mark.asyncio
async def test_a_line_said_to_the_end_before_the_cut_is_not_unfinished():
    f = scout()
    f.store.interrupted_rows = [
        {
            "id": uuid4(),
            "content": "知道啦，就像小星星一样亮亮地陪着你就好。",
            "unsaid": None,
            "created_at": NOW - timedelta(minutes=20),
        }
    ]
    f.scout.boredom.level = 1
    f.clock.now = NOW + timedelta(seconds=10)
    material = await f.scout.check()
    assert material is None or material.kind != "pending"


def test_ta_sees_each_of_its_own_lines_once():
    from fish_memory.experimental.context import without_repeated_speech

    history = {
        "role": "user",
        "context_reference": True,
        "content": (
            "History:\n"
            "2026-09-19 06:50:00-05:00 - 2026-09-19 07:00:00-05:00\n"
            "a: 夜深了，我就在这里安安静静陪着你。\n"
            "u: 嗯\n"
            "a: 夜深了，我就在这里安安静静陪着你呀。\n"
            "  (被打断)\n"
            "event: 宠物的内心想法（尚未表达）：好安静。"
        ),
    }
    container = {**history, "history_messages": (history,)}
    folded = without_repeated_speech(
        [
            {"role": "system", "content": "你喜欢音乐。"},
            container,
            {"role": "assistant", "content": "夜深了，我就在这里安安静静陪着你。"},
            {"role": "assistant", "content": "要不要听首歌？"},
        ]
    )
    [item] = folded[1]["history_messages"]
    assert item["content"].splitlines()[2:] == [
        "a: 夜深了，我就在这里安安静静陪着你。",
        "u: 嗯",
        "event: 宠物的内心想法（尚未表达）：好安静。",
    ]
    # The joined text models do not read stays as it was.
    assert folded[1]["content"] == history["content"]
    assert [m["content"] for m in folded[2:]] == ["要不要听首歌？"]


def test_recalling_reaches_back_past_what_was_just_said():
    from fish_memory.experimental.materials import _recallable

    def fact(role, age):
        return SimpleNamespace(source_role=role, mentioned_at=NOW - age)

    # Live, cue recall kept surfacing TA's own lines from two hours before.
    assert not _recallable(fact("user", timedelta(hours=2)), NOW)
    assert _recallable(fact("user", timedelta(hours=4)), NOW)
    assert not _recallable(fact("assistant", timedelta(hours=20)), NOW)
    assert _recallable(fact("assistant", timedelta(days=2)), NOW)


@pytest.mark.asyncio
async def test_something_browsed_on_a_whim_reads_as_ta_own_find():
    f = scout()
    f.clock.now = NOW + timedelta(seconds=10)
    material = await f.scout._browse(MESSAGES, None, f.clock.now)
    assert material is not None and material.cause == "boredom"
    # Not "看到", which read as if the user had seen it too.
    assert material.text == "刚看到：自己随手刷到一条：山顶常有大风，建议带防风外套。"
