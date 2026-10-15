"""New material for idle moments: what TA recalls, left unfinished, or looks up.

Nothing here runs on a schedule. Boredom rises while thoughts find nothing new
and timed wakes keep choosing to wait; every check draws against a chance that
grows with it, so material turns up at irregular, state-driven moments. Each
piece becomes an ordinary History event, and TA still decides whether to
mention it.
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import logging
import math
import random
import re
import sys
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fish_memory.clock import format_prompt_timestamp
from fish_memory.experimental.context import (
    THOUGHT_PREFIX,
    recent_speech,
    recorded_thoughts,
    timeline,
)
from fish_memory.experimental.thoughts import restates, trigrams
from fish_memory.history.models import HistoryUnit
from fish_memory.llm.logs import call_scope
from fish_memory.observations.models import ImportantObservationEvent
from fish_memory.performance.tracker import MessagePerformanceTracker

if TYPE_CHECKING:
    import asyncpg

    from fish_memory.answer.context import ChatMessage
    from fish_memory.answer.context_loader import AnswerContextLoader
    from fish_memory.db.recall import RecallStore
    from fish_memory.embeddings.client import EmbeddingClient
    from fish_memory.experimental.service import ExperimentalService
    from fish_memory.history.service import SessionHistoryService

logger = logging.getLogger(__name__)

MATERIAL_OBSERVATION = "experimental.material"

# Ten minutes in which TA's thoughts find nothing new make TA fully bored.
BOREDOM_RAMP_SECONDS = 600.0
# At full boredom material turns up after about two minutes on average. The
# chance grows with the square of boredom, so a little rarely produces any.
# Replaying the trial's real signals, a lull brings material every 4-8 minutes.
FULL_BOREDOM_MEAN_SECONDS = 120.0
# Without keyboard or mouse input for this long the user is away: no material.
AWAY_SECONDS = 300.0
# Looking things up on the web needs the user at the computer right now.
PRESENT_SECONDS = 120.0
# Coming back after being away is a natural moment to pick up an open thread.
RETURN_CHANCE = 0.7
RETURN_DELAY_SECONDS = (5.0, 40.0)
BROWSES_PER_HOUR = 3
# Cosine similarity of the local e5 embeddings above which an old memory is
# tied to what is going on now. On the trial pet's 167 memories, cues from an
# unrelated topic topped out at 0.87-0.88 and related ones scored 0.89-0.96.
CUE_SCORE = 0.89
# Memories are third-person rewrites of what was said; one whose wording is
# mostly in the context already is not something TA would newly recall.
KNOWN_SHARE = 0.5
# Recalling means reaching back: what the user said at least this long ago,
# and TA's own words only once they are a day old. Live, cue recall kept
# surfacing TA's own lines from two hours before.
RECALL_AFTER = timedelta(hours=3)
OWN_WORDS_AFTER = timedelta(days=1)
# Only a very bored TA picks an unrelated old memory or browses on a whim.
IDLE_WHIM = 0.8
# How long each kind of open thread stays worth picking up.
INTERRUPTED_FOR = timedelta(hours=2)
QUESTION_FOR = timedelta(hours=2)
PLAN_FOR = timedelta(days=1)
# A question gets time to be answered before it counts as left open.
QUESTION_AFTER = timedelta(minutes=2)
# TA speaking again this soon after asking means TA moved on: live, TA asked
# every fifteen seconds while talking to itself, and most such questions were
# judged still open.
TALKED_ON = timedelta(minutes=1)
QUESTION_WITHIN = timedelta(minutes=40)
SURFACED_FOR = timedelta(days=7)


def _hid_idle() -> Callable[[], float | None]:
    """Seconds since the last keyboard or mouse input, where the OS says."""
    if sys.platform != "darwin":
        return lambda: None
    try:
        graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        since = graphics.CGEventSourceSecondsSinceLastEventType
        since.restype = ctypes.c_double
        since.argtypes = (ctypes.c_int32, ctypes.c_uint32)
    except (OSError, AttributeError):
        return lambda: None
    # kCGEventSourceStateHIDSystemState and kCGAnyInputEventType; needs no
    # permission.
    return lambda: float(since(1, 0xFFFFFFFF))


user_idle_seconds = _hid_idle()


class Boredom:
    """How long nothing new has happened, as a level from 0 to 1.

    It grows with time, not with the number of signals: thought rounds come
    every few seconds, and most of them restate something, so counting them
    made boredom depend on how fast thoughts run.
    """

    # How stale each signal says TA's inner life is: a thought round that only
    # restated old thoughts or lines, a timed wake that chose to wait, a
    # thought formed with nothing new since the last, a thought about news.
    STALENESS = {"duplicate": 1.0, "wait": 1.0, "quiet": 0.6, "fresh": 0.0}
    # The share of the level that news of each History label takes away. TA's
    # own lines and thoughts are not news.
    NEWS = {"u": 1.0, "as": 0.5, "event": 0.5, "material": 0.8}

    def __init__(self) -> None:
        self.level = 0.0
        self.staleness = 0.0

    def note(self, signal: str) -> None:
        self.staleness = 0.7 * self.staleness + 0.3 * self.STALENESS[signal]

    def grow(self, seconds: float) -> None:
        step = self.staleness * max(0.0, seconds) / BOREDOM_RAMP_SECONDS
        self.level = min(1.0, self.level + step)

    def news(self, kind: str) -> None:
        self.level *= 1 - self.NEWS.get(kind, 0.0)

    def chance(self, seconds: float) -> float:
        """Probability that material turns up within the next `seconds`."""
        rate = self.level**2 / FULL_BOREDOM_MEAN_SECONDS
        return 1 - math.exp(-rate * max(0.0, seconds))


@dataclass(frozen=True, slots=True)
class Material:
    kind: str  # recall, pending or browse
    key: str  # each key is surfaced at most once
    text: str
    cause: str


@dataclass(frozen=True, slots=True)
class OpenQuestion:
    message_id: UUID
    text: str
    asked_at: datetime


_QUESTION = re.compile(r"[^。！？!?…～~\n]*[？?]")
# Wanting to know something about the world. A bare question mark is not
# enough: on the trial every such thought was rhetorical ("像不像…？") or a
# guess about the user, which a web search cannot answer.
_CURIOUS = re.compile(r"好奇|不知道|到底|为什么|怎么会|是什么|有没有|是真的吗")
# Search models sometimes report finding nothing instead of returning nothing.
_NOTHING_FOUND = re.compile(r"^(暂)?(无|没有|未)(找到|查到|检索到|相关|可靠)")


def last_question(text: str) -> str | None:
    found = [part.strip(" ，,、") for part in _QUESTION.findall(text)]
    found = [part for part in found if len(part) >= 4]
    return found[-1] if found else None


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class MaterialStore:
    """The few reads material needs beyond History and Recall."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def surfaced(self, fish: UUID, since: datetime) -> set[str]:
        rows = await self.pool.fetch(
            """SELECT semantic_key FROM observations
               WHERE fish_id=$1 AND observation_type=$2 AND started_at>=$3""",
            fish,
            MATERIAL_OBSERVATION,
            since,
        )
        return {row["semantic_key"] for row in rows}

    async def interrupted(
        self, fish: UUID, session: UUID, since: datetime, until: datetime
    ) -> list[asyncpg.Record]:
        """TA's lines cut off after TA started saying them.

        The message keeps what was said; the unsaid rest is kept only for
        replies that were fully written before the cut. Speaking up stops
        writing when cut off, so its rest never existed.
        """
        return await self.pool.fetch(
            """SELECT message.id, message.content, unsaid.content AS unsaid,
                      message.created_at
               FROM messages AS message
               LEFT JOIN message_unsaid AS unsaid ON unsaid.message_id=message.id
               WHERE message.fish_id=$1 AND message.session_id=$2
                 AND message.role='assistant' AND message.status='interrupted'
                 AND message.created_at BETWEEN $3 AND $4
                 AND char_length(message.content) >= 4
               ORDER BY message.created_at DESC LIMIT 3""",
            fish,
            session,
            since,
            until,
        )

    async def asked(
        self, fish: UUID, session: UUID, since: datetime, until: datetime
    ) -> list[asyncpg.Record]:
        """TA's lines with a question, each with when TA spoke next."""
        return await self.pool.fetch(
            """SELECT id, content, consumed_offset, created_at, next_at FROM (
                   SELECT id, content, consumed_offset, created_at,
                          lead(created_at) OVER (ORDER BY created_at, sequence)
                              AS next_at
                   FROM messages
                   WHERE fish_id=$1 AND session_id=$2 AND role='assistant'
                     AND status IN ('completed','interrupted')
                     AND created_at >= $3
               ) AS said
               WHERE created_at <= $4 AND content ~ '[？?]'
               ORDER BY created_at DESC LIMIT 6""",
            fish,
            session,
            since,
            until,
        )

    async def due_plans(
        self, fish: UUID, since: datetime, until: datetime
    ) -> list[asyncpg.Record]:
        """Things said to happen later whose time has now come."""
        return await self.pool.fetch(
            """SELECT id, text, occurred_start, mentioned_at FROM facts
               WHERE fish_id=$1 AND invalidated_at IS NULL
                 AND occurred_start > mentioned_at + interval '30 minutes'
                 AND occurred_start BETWEEN $2 AND $3
               ORDER BY occurred_start DESC LIMIT 3""",
            fish,
            since,
            until,
        )

    async def old_facts(self, fish: UUID, before: datetime) -> list[asyncpg.Record]:
        return await self.pool.fetch(
            """SELECT id, text, mentioned_at FROM facts
               WHERE fish_id=$1 AND invalidated_at IS NULL AND mentioned_at<$2
               ORDER BY random() LIMIT 5""",
            fish,
            before,
        )


class MaterialScout:
    def __init__(
        self,
        service: ExperimentalService,
        contexts: AnswerContextLoader,
        history: SessionHistoryService,
        store: MaterialStore,
        *,
        recall: RecallStore | None,
        embedder: EmbeddingClient | None,
        binding: Callable[[], tuple[UUID, UUID] | None],
        idle: Callable[[], float | None] = user_idle_seconds,
        rng: random.Random | None = None,
        start: bool = True,
    ) -> None:
        self.service, self.contexts, self.history = service, contexts, history
        self.store, self.recall, self.embedder = store, recall, embedder
        self.binding, self._idle = binding, idle
        self._rng = rng or random.Random()
        self.boredom = Boredom()
        self._closed = False
        self._last_check = service.now()
        self._last_user: datetime | None = None
        self._away_since: datetime | None = None
        self._return_at: datetime | None = None
        self._browses: deque[datetime] = deque()
        self._asked: tuple[OpenQuestion, ...] = ()
        self._asked_at: datetime | None = None
        self._judged: OrderedDict[UUID, None] = OrderedDict()
        self._open: dict[UUID, OpenQuestion] = {}
        self._loop = (
            asyncio.create_task(self._run(), name="experimental-idle-material")
            if start
            else None
        )

    @property
    def enabled(self) -> bool:
        return (
            not self._closed
            and not self.service.closed
            and self.service.configuration.thoughts_enabled
        )

    def note(self, signal: str) -> None:
        self.boredom.note(signal)

    def observe(self, unit: HistoryUnit) -> None:
        if unit.label == "u":
            self._last_user = unit.ended_at
            self._open.clear()
        if unit.label == "event" and unit.content.startswith(THOUGHT_PREFIX):
            return
        self.boredom.news(
            "material" if unit.content.startswith(_MATERIAL_HEADS) else unit.label
        )

    def open_questions(self) -> tuple[OpenQuestion, ...]:
        """TA's recent questions nobody has judged, for the next decision."""
        return tuple(q for q in self._asked if q.message_id not in self._judged)

    def judged(self, asked: Sequence[OpenQuestion], index: int | None) -> None:
        for question in asked:
            self._judged[question.message_id] = None
        while len(self._judged) > 256:
            self._judged.popitem(last=False)
        if index is not None:
            question = asked[index]
            # Only the latest question left open still matters.
            if all(
                question.asked_at >= other.asked_at for other in self._open.values()
            ):
                self._open = {question.message_id: question}
                logger.info("TA's question is still open: %s", question.text)

    async def _run(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._rng.uniform(3, 8))
            try:
                await self.check()
            except Exception:
                logger.exception("Idle material check failed")

    async def check(self) -> Material | None:
        now = self.service.now()
        elapsed = (now - self._last_check).total_seconds()
        self._last_check = now
        target = self.binding()
        if not self.enabled or target is None:
            return None
        presence = self._presence(now)
        if presence == "away":
            return None
        self.boredom.grow(elapsed)
        await self._refresh_asked(*target, now)
        if self._return_at is not None and now >= self._return_at:
            self._return_at = None
            if self._rng.random() < RETURN_CHANCE:
                return await self._produce(*target, now, presence, returned=True)
        if self._rng.random() >= self.boredom.chance(elapsed):
            return None
        return await self._produce(*target, now, presence, returned=False)

    def _presence(self, now: datetime) -> str:
        idle = self._idle()
        if self._last_user is not None:
            spoke = (now - self._last_user).total_seconds()
            idle = spoke if idle is None else min(idle, spoke)
        if idle is None:
            return "present"
        if idle >= AWAY_SECONDS:
            if self._away_since is None:
                self._away_since = now - timedelta(seconds=idle)
            return "away"
        if self._away_since is not None:
            self._away_since = None
            self._return_at = now + timedelta(
                seconds=self._rng.uniform(*RETURN_DELAY_SECONDS)
            )
        return "present" if idle < PRESENT_SECONDS else "nearby"

    async def _refresh_asked(self, fish: UUID, session: UUID, now: datetime) -> None:
        if self._asked_at is not None and now - self._asked_at < timedelta(seconds=30):
            return
        self._asked_at = now
        rows = await self.store.asked(
            fish, session, now - QUESTION_WITHIN, now - QUESTION_AFTER
        )
        asked = []
        for row in rows:
            if (
                row["next_at"] is not None
                and row["next_at"] - row["created_at"] < TALKED_ON
            ):
                continue
            question = last_question(_spoken(row))
            if question is not None:
                asked.append(OpenQuestion(row["id"], question, row["created_at"]))
        self._asked = tuple(reversed(asked[:3]))

    async def _produce(
        self, fish: UUID, session: UUID, now: datetime, presence: str, *, returned: bool
    ) -> Material | None:
        level = self.boredom.level
        context = await self.contexts.load_for_proactive(
            fish, session, include_image=False, tracker=MessagePerformanceTracker()
        )
        surfaced = await self.store.surfaced(fish, now - SURFACED_FOR)
        pending = await self._pending(fish, session, now, context.messages, surfaced)
        options: list[tuple[float, Callable[[], Awaitable[Material | None]]]] = []
        if pending is not None:
            chosen = pending
            options.append((2.0, lambda: _ready(chosen)))
        if not returned:
            recalled = await self._cue_recall(fish, now, context.messages, surfaced)
            if recalled is not None:
                options.append((2.0, lambda: _ready(recalled)))
            elif level >= IDLE_WHIM:
                options.append(
                    (
                        0.5,
                        lambda: self._old_memory(fish, now, context.messages, surfaced),
                    )
                )
            if presence == "present" and self._may_browse(now):
                question = _curious(recorded_thoughts(context.messages)[-3:])
                if (
                    question is not None
                    and f"browse:{_digest(question)}" not in surfaced
                ):
                    options.append(
                        (1.5, lambda: self._browse(context.messages, question, now))
                    )
                elif level >= IDLE_WHIM:
                    options.append(
                        (0.5, lambda: self._browse(context.messages, None, now))
                    )
        while options:
            index = self._pick([weight for weight, _ in options])
            _, make = options.pop(index)
            material = await make()
            if material is not None:
                break
        else:
            return None
        if not self.enabled or self.binding() != (fish, session):
            return None
        await self._record(fish, session, material, now, level)
        return material

    def _pick(self, weights: Sequence[float]) -> int:
        point = self._rng.uniform(0, sum(weights))
        for index, weight in enumerate(weights):
            point -= weight
            if point <= 0:
                return index
        return len(weights) - 1

    async def _pending(
        self,
        fish: UUID,
        session: UUID,
        now: datetime,
        context: Sequence[ChatMessage],
        surfaced: set[str],
    ) -> Material | None:
        found: list[Material] = []
        spoken = recent_speech(context, 12)
        for row in await self.store.interrupted(
            fish, session, now - INTERRUPTED_FOR, now - timedelta(minutes=1)
        ):
            said, rest = row["content"], (row["unsaid"] or "").lstrip(_PAUSES)
            # Marked cut off only after its last sentence was already said.
            if not rest and said.rstrip().endswith(_ENDS):
                continue
            key = f"pending:interrupted:{row['id']}"
            # A cut-off line TA had already said before is not unfinished.
            others = [line for line in spoken if line.strip() != said.strip()]
            if key in surfaced or restates(said + rest, others):
                continue
            text = (
                f"还没聊完：{_ago(row['created_at'], now)}"
                f"说到“{said[-24:].strip(_PAUSES)}”时被打断了"
            )
            found.append(
                Material(
                    "pending",
                    key,
                    f"{text}，没说完的是：{rest}" if len(rest) >= 8 else f"{text}。",
                    "interrupted",
                )
            )
        for question in list(self._open.values()):
            key = f"pending:question:{question.message_id}"
            if now - question.asked_at > QUESTION_FOR or key in surfaced:
                self._open.pop(question.message_id, None)
                continue
            found.append(
                Material(
                    "pending",
                    key,
                    f"还没聊完：{_ago(question.asked_at, now)}"
                    f"问过“{question.text}”，还没等到回答。",
                    "question",
                )
            )
        for row in await self.store.due_plans(fish, now - PLAN_FOR, now):
            key = f"pending:plan:{row['id']}"
            if key in surfaced:
                continue
            found.append(
                Material(
                    "pending",
                    key,
                    f"还没聊完：{row['text']}（{format_prompt_timestamp(row['mentioned_at'])}提到，"
                    f"原定{format_prompt_timestamp(row['occurred_start'])}，现在已经过了）",
                    "plan",
                )
            )
        return self._rng.choice(found) if found else None

    async def _cue_recall(
        self,
        fish: UUID,
        now: datetime,
        context: Sequence[ChatMessage],
        surfaced: set[str],
    ) -> Material | None:
        if self.recall is None or self.embedder is None:
            return None
        cue = cue_text(context)
        if not cue:
            return None
        vector = (await self.embedder.embed_query([cue])).vectors[0]
        heads = await self.recall.search_fact_head(
            fish,
            self.embedder.space.dimensions,
            vector,
            time_range=None,
            limit=12,
            use_hnsw=True,
        )
        shown = _shown(context)
        candidates = [
            head.fact
            for head in heads
            if (head.fact.semantic_score or 0) >= CUE_SCORE
            and f"recall:{head.fact.id}" not in surfaced
            and _recallable(head.fact, now)
            and not _known(head.fact.text, shown)
        ]
        if not candidates:
            return None
        fact = self._rng.choice(candidates[:3])
        return Material(
            "recall",
            f"recall:{fact.id}",
            _recalled(fact.text, fact.mentioned_at, now),
            "cue",
        )

    async def _old_memory(
        self,
        fish: UUID,
        now: datetime,
        context: Sequence[ChatMessage],
        surfaced: set[str],
    ) -> Material | None:
        shown = _shown(context)
        for row in await self.store.old_facts(fish, now - timedelta(days=1)):
            key = f"recall:{row['id']}"
            if key not in surfaced and not _known(row["text"], shown):
                return Material(
                    "recall",
                    key,
                    _recalled(row["text"], row["mentioned_at"], now),
                    "boredom",
                )
        return None

    def _may_browse(self, now: datetime) -> bool:
        while self._browses and now - self._browses[0] > timedelta(hours=1):
            self._browses.popleft()
        return self.service.search_enabled and len(self._browses) < BROWSES_PER_HOUR

    async def _browse(
        self, context: Sequence[ChatMessage], question: str | None, now: datetime
    ) -> Material | None:
        self._browses.append(now)
        topic = question or "按你的人格和兴趣，随便逛逛网上最近的一件新鲜事"
        try:
            async with asyncio.timeout(90):
                with call_scope("experimental.material.browse"):
                    found = await self.service.search(list(context), topic)
        except Exception:
            logger.warning("Idle browsing failed", exc_info=True)
            return None
        found = found.strip()
        if not found or _NOTHING_FOUND.match(found):
            return None
        if question is None:
            return Material(
                "browse",
                f"browse:{_digest(found)}",
                # "自己随手刷到" says TA found it, not that the user saw it:
                # live, TA asked "你还记得那条新闻吗？" about its own find.
                f"刚看到：自己随手刷到一条：{found}",
                "boredom",
            )
        return Material(
            "browse",
            f"browse:{_digest(question)}",
            f"刚看到：想知道“{question}”，查到：{found}",
            "curiosity",
        )

    async def _record(
        self, fish: UUID, session: UUID, material: Material, now: datetime, level: float
    ) -> None:
        logger.info(
            "Idle material %s from %s at boredom %.2f: %s",
            material.kind,
            material.cause,
            level,
            material.key,
        )
        await self.service.observations.record_event(
            fish,
            ImportantObservationEvent(
                source="experimental",
                observation_type=MATERIAL_OBSERVATION,
                semantic_key=material.key,
                summary=material.text[:2000],
                occurred_at=now,
                payload={
                    "session_id": str(session),
                    "context_text": material.text,
                    "material": material.kind,
                    "cause": material.cause,
                    "boredom": round(level, 2),
                },
            ),
        )

    async def close(self) -> None:
        self._closed = True
        if self._loop is not None:
            self._loop.cancel()
            await asyncio.gather(self._loop, return_exceptions=True)


_MATERIAL_HEADS = ("想起：", "还没聊完：", "刚看到：")
_PAUSES = " ，,。.！!？?、…～~"
_ENDS = ("。", "！", "？", "!", "?", "～", "~", "…")
_THIRD_PERSON = re.compile(r"该助手|该用户")


async def _ready(material: Material) -> Material:
    return material


def _spoken(row: Any) -> str:
    offset = row["consumed_offset"]
    content = row["content"]
    return content[:offset] if offset is not None and offset >= 0 else content


def _ago(then: datetime, now: datetime) -> str:
    # Relative: replaying the trial, TA read an absolute time from an hour and
    # a half before as "just now" in half of its lines.
    minutes = (now - then).total_seconds() / 60
    if minutes < 90:
        return f"{max(1, round(minutes))}分钟前"
    if minutes < 36 * 60:
        return f"约{round(minutes / 60)}小时前"
    return f"{round(minutes / 60 / 24)}天前"


def _recalled(memory: str, mentioned_at: datetime, now: datetime) -> str:
    # Memories are stored in the third person; TA recalls them as its own.
    memory = memory.replace("该助手", "我").replace("该用户", "用户")
    return f"想起：{memory}（{_ago(mentioned_at, now)}记下的）"


def _recallable(fact: Any, now: datetime) -> bool:
    age = now - fact.mentioned_at
    role = getattr(fact.source_role, "value", fact.source_role)
    return age >= (OWN_WORDS_AFTER if role == "assistant" else RECALL_AFTER)


def _shown(context: Sequence[ChatMessage]) -> frozenset[str]:
    return frozenset().union(*(trigrams(text) for _, text in timeline(context)))


def _known(memory: str, shown: frozenset[str]) -> bool:
    grams = trigrams(_THIRD_PERSON.sub("", memory))
    return bool(grams) and len(grams & shown) >= KNOWN_SHARE * len(grams)


def _curious(thoughts: Sequence[str]) -> str | None:
    for thought in reversed(thoughts):
        # What the user meant is for asking the user, not the web.
        if _CURIOUS.search(thought) and "你" not in thought:
            return thought
    return None


def cue_text(context: Sequence[ChatMessage]) -> str:
    """What is on TA's mind now: the latest lines and the latest thought."""
    lines = timeline(context)
    thought = next(
        (
            text.removeprefix(THOUGHT_PREFIX)
            for label, text in reversed(lines)
            if label == "event" and text.startswith(THOUGHT_PREFIX)
        ),
        None,
    )
    recent = [text[:200] for label, text in lines if label in {"u", "a", "as"}][-5:]
    return "\n".join([*recent, *([thought] if thought else [])])
