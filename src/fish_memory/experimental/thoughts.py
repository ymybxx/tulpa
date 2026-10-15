"""Independent, non-overlapping thought batches with an atomic context fence."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.experimental.channels import render_state
from fish_memory.experimental.context import (
    THOUGHT_PREFIX,
    expression_context,
    model_messages,
    new_since_last_thought,
    recent_speech,
    recorded_thoughts,
    trigrams,
    with_recent_thoughts,
    without_repeated_speech,
)
from fish_memory.experimental.perception import PerceptionInput
from fish_memory.experimental.service import ExperimentalService
from fish_memory.history.service import SessionHistoryService
from fish_memory.llm.logs import call_scope
from fish_memory.llm.streaming import closing_iterator
from fish_memory.observations.models import ImportantObservationEvent
from fish_memory.observations.store import ObservationContextGuard
from fish_memory.performance.tracker import MessagePerformanceTracker

logger = logging.getLogger(__name__)

THOUGHT_RULES = """以给定人格的第一人称，写下此刻可能浮现、尚未说出口的内心想法。

【个人视角与联想】
从当前触动、仍在意的经历或自己的兴趣出发，让偏好和性格体现在你注意什么、如何理解。
可以联系输入中明确提供的记忆、发现意外关联、换个角度、想象一种可能，或修正先前的判断。
联想允许走远，但保留能理解的来由；五条各自独立，探索不同的关注点，不按固定类别凑数。
结合History向前多想一点：同一个比喻、疑问或评价只换措辞，仍是重复。

【自然与洞见】
用平常、简短的话保留一个具体念头；可以好奇、幽默、自嘲、有矛盾的感受或暂时没答案。
有值得深想的地方，再看见另一种解释、被忽略的细节、选择的代价或自己的判断局限。
比喻和深刻结论都不是必需，不靠散文式铺陈、金句、泛泛感悟或反复解释人格来制造趣味。

【事实与归属】
事实限于已提供的内容；不补造见闻、身体感受、画面细节、个人经历或已完成的行动。
常识、类比和想象写作当下的联想，不包装成曾经亲历；回忆必须有输入记录支持。
推测和想象明确保留其性质；先前的想法不因再次出现就成为事实。
保留话语与判断的归属，反省只针对自己在历史中确实表达过的判断，不把旁人的想法安在自己身上。

【输出】
只返回一个JSON对象，仅含thoughts字段，其值为5个不同的非空字符串。
每条只写一句第一人称念头，通常15至50字，最多80字，单行；不续写解释，不附分类或评分。"""

Thought = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=300),
]
Outcome = Literal["written", "stale", "duplicate"]


# A line of thought needs its recent steps; speech only needs the latest few.
THOUGHT_CONTEXT = 8
# A thought that only restates something TA said adds nothing new either. Live,
# a thought restated a line from twenty lines before and TA then said it again
# word for word, so every line in the context counts, not just the last few.
SPOKEN_CONTEXT = 40


# Character-trigram overlap above which a candidate is judged a rewording of a
# recent thought. On 853 logged selections every sampled candidate above it was
# a restatement; Jev picked one in 64% of rounds although a new one existed.
RESTATEMENT_OVERLAP = 0.3


def restates(text: str, others: Sequence[str]) -> bool:
    """Whether `text` rewords any of `others`."""
    grams = trigrams(text)
    return any(
        len(grams & other) > RESTATEMENT_OVERLAP * len(grams | other)
        for other in map(trigrams, others)
    )


def fresh_thoughts(candidates: Sequence[str], recent: Sequence[str]) -> tuple[str, ...]:
    """Drop candidates that reword a recent thought or line before Jev chooses."""
    return tuple(
        candidate for candidate in candidates if not restates(candidate, recent)
    )


class ThoughtBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    thoughts: list[Thought] = Field(min_length=5, max_length=5)

    @field_validator("thoughts")
    @classmethod
    def require_distinct_lines(cls, thoughts: list[str]) -> list[str]:
        if len(set(thoughts)) != 5 or any("\n" in t or "\r" in t for t in thoughts):
            raise ValueError("Thought candidates must be five distinct single lines")
        return thoughts


_THOUGHT_KEY = re.compile(r"thoughts\d*")


def _one_key_per_thought(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    if keys != ["thoughts"] and all(
        _THOUGHT_KEY.fullmatch(key) and isinstance(value, str) for key, value in pairs
    ):
        return {"thoughts": [value for _, value in pairs]}
    return dict(pairs)


# Holds the model to the batch's shape; plain JSON mode let qwen answer with a
# bare array or a string for "thoughts". Distinct lines are still checked on
# parse, which a schema cannot express.
THOUGHT_BATCH_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "thoughts": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
            "minItems": 5,
            "maxItems": 5,
        }
    },
    "required": ["thoughts"],
    "additionalProperties": False,
}


def parse_thought_batch(raw: str) -> ThoughtBatch:
    """Accept the array contract and qwen's occasional key-per-thought layout.

    qwen sometimes writes each candidate under a repeated "thoughts" key or
    under "thoughts2".."thoughts5"; the candidates are fine, only the layout
    differs. Every other rule of ThoughtBatch still applies.
    """
    try:
        value = json.loads(raw, object_pairs_hook=_one_key_per_thought)
    except json.JSONDecodeError:
        return ThoughtBatch.model_validate_json(raw)
    return ThoughtBatch.model_validate(value)


class ThoughtWorker:
    def __init__(
        self,
        service: ExperimentalService,
        contexts: AnswerContextLoader,
        history: SessionHistoryService,
        *,
        binding: Callable[[], tuple[UUID, UUID] | None],
        context_state: Callable[[UUID, UUID], object] = lambda fish, session: None,
    ) -> None:
        self.service, self.contexts, self.history = service, contexts, history
        self.binding, self.context_state = binding, context_state
        # Told "duplicate" when a round found only restatements, "quiet" when
        # it thought with nothing new since the last thought, else "fresh".
        self.on_outcome: Callable[[str], None] = lambda outcome: None
        self._quiet = False
        self._generation = 0
        self._closed = False
        self._changed = asyncio.Event()
        self._round: asyncio.Task[Outcome] | None = None
        self.last_outcome: str | None = None
        self._last_sight: str | None = None
        self._timer = asyncio.create_task(
            self._ticks(), name="experimental-thought-timer"
        )

    @property
    def enabled(self) -> bool:
        return (
            not self._closed
            and not self.service.closed
            and self.service.configuration.thoughts_enabled
        )

    def reset(self) -> None:
        self._generation += 1
        if self._round is not None:
            self._round.cancel()
        self._changed.set()

    def tick(self) -> None:
        target = self.binding()
        if not self.enabled or target is None:
            return
        if self._round is not None and not self._round.done():
            return
        self._round = asyncio.create_task(
            self._run(*target), name="experimental-thought-batch"
        )
        self._round.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[Outcome]) -> None:
        if task.cancelled():
            self.last_outcome = "cancelled"
        elif (error := task.exception()) is not None:
            self.last_outcome = "failed"
            logger.warning("Background thought batch failed", exc_info=error)
        else:
            self.last_outcome = task.result()
            logger.debug("Background thought batch: %s", self.last_outcome)
            if self.last_outcome == "duplicate":
                self.on_outcome("duplicate")
            elif self.last_outcome == "written":
                self.on_outcome("quiet" if self._quiet else "fresh")

    async def _ticks(self) -> None:
        while not self._closed:
            self._changed.clear()
            try:
                async with asyncio.timeout(self.service.configuration.thought_seconds):
                    await self._changed.wait()
            except TimeoutError:
                self.tick()

    async def _run(self, fish: UUID, session: UUID) -> Outcome:
        generation, configuration_revision = self._generation, self.service.revision
        configuration = self.service.configuration
        state_version = self.context_state(fish, session)

        def current() -> bool:
            return (
                self.enabled
                and generation == self._generation
                and configuration_revision == self.service.revision
                and self.binding() == (fish, session)
                and state_version == self.context_state(fish, session)
            )

        # Read before loading all sources and check again afterwards: the loader
        # uses several reads and may materialize pending changes or compact history.
        revision = await self.history.store.context_revision(fish, session)
        context = await self.contexts.load_for_proactive(
            fish, session, include_image=False, tracker=MessagePerformanceTracker()
        )
        if (
            revision is None
            or not current()
            or revision != await self.history.store.context_revision(fish, session)
        ):
            return "stale"
        # Anchor the thought to the frozen context it read, not the later model
        # completion. Loading may take time, so capture this after its fence.
        started_at = self.service.now()
        trigger = "后台生成内心想法，不是用户的新话语，也不要求开口或执行行动。"
        self._quiet = not new_since_last_thought(context.messages)
        if self._quiet:
            trigger += "上次想完之后还没有新发生的事。"
        # The stored label stays: relabelling here made qwen emit the five
        # candidates as repeated "thoughts" keys instead of one array.
        recent = without_repeated_speech(
            with_recent_thoughts(context.messages, THOUGHT_CONTEXT, relabel=False)
        )
        state = render_state(recent, started_at, trigger)
        messages = model_messages(
            THOUGHT_RULES, expression_context(recent, started_at, trigger)
        )
        recorded = recorded_thoughts(context.messages)
        prior_thoughts = frozenset(recorded)
        async with asyncio.timeout(30):
            if self.service.perception is not None:
                perception = await self.service.perception.capture(
                    fish, started_at, model=self.service.reasoning
                )
                if (
                    not current()
                    or revision
                    != await self.history.store.context_revision(fish, session)
                ):
                    return "stale"
                perception = self._only_new_sight(perception)
                perception.attach(messages[-1])
            with call_scope("experimental.thought.generate"):
                parts: list[str] = []
                async with closing_iterator(
                    self.service.reasoning.stream_json(
                        messages,
                        response_schema=THOUGHT_BATCH_SCHEMA,
                        max_output_tokens=1000,
                    )
                ) as stream:
                    async for chunk in stream:
                        if chunk.finish_reason == "length":
                            raise ValueError("Thought generation was truncated")
                        parts.append(chunk.content or "")
                thoughts = tuple(parse_thought_batch("".join(parts)).thoughts)
            if not current() or revision != await self.history.store.context_revision(
                fish, session
            ):
                return "stale"
            fresh = fresh_thoughts(
                thoughts,
                [
                    *recorded[-THOUGHT_CONTEXT:],
                    *recent_speech(context.messages, SPOKEN_CONTEXT),
                ],
            )
            if not fresh:
                return "duplicate"
            selected = await self.service.judge.select_thought(
                configuration, state, fresh
            )
        if not current():
            return "stale"
        if fresh[selected] in prior_thoughts:
            return "duplicate"
        text = THOUGHT_PREFIX + fresh[selected]
        event = ImportantObservationEvent(
            source="experimental",
            observation_type="experimental.thought",
            semantic_key=f"thought:{uuid4()}",
            summary=text,
            occurred_at=started_at,
            payload={
                "session_id": str(session),
                "context_text": text,
                "thought_started_at": started_at.isoformat(),
                "thought_completed_at": self.service.now().isoformat(),
            },
        )
        recorded = await self.service.observations.record_event(
            fish, event, guard=ObservationContextGuard(session, revision, current)
        )
        return "written" if recorded is not None else "stale"

    def _only_new_sight(self, perception: PerceptionInput) -> PerceptionInput:
        """Look at the screen when it changed; otherwise keep following the thought.

        Static screens produce the same unlabeled sheet, so an unchanged digest
        means nothing new was seen. Sound is always passed on.
        """
        digest = perception.images[-1]["digest"] if perception.images else None
        if digest is not None and digest == self._last_sight:
            return PerceptionInput(heard=perception.heard)
        self._last_sight = digest
        return perception

    async def close(self) -> None:
        self._closed = True
        self.reset()
        self._timer.cancel()
        await asyncio.gather(
            self._timer,
            *([self._round] if self._round is not None else []),
            return_exceptions=True,
        )
