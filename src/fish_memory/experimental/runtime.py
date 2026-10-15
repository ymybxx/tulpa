"""Event opportunities and idle ticks; events that arrive while busy wait and merge."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from datetime import datetime
from uuid import UUID, uuid4

from fish_memory.answer.lane_stream import AnswerLaneDelta, BlankFieldError
from fish_memory.clock import format_prompt_timestamp
from fish_memory.experimental.channels import generate_channel, render_state
from fish_memory.experimental.context import (
    SPEECH_THOUGHT_CONTEXT,
    expression_context,
    with_recent_thoughts,
    without_repeated_speech,
)
from fish_memory.experimental.materials import MaterialScout, MaterialStore
from fish_memory.experimental.perception import PerceptionInput
from fish_memory.experimental.service import ExperimentalService
from fish_memory.experimental.thoughts import ThoughtWorker
from fish_memory.history.models import HistoryUnit
from fish_memory.llm.logs import purpose_scope
from fish_memory.observations.environment_history_models import SemanticDocument
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.proactive.expression import GeneratedExpressionInput
from fish_memory.proactive.models import ExpressionIntent

logger = logging.getLogger(__name__)

MAX_WAITING_EVENTS = 16
TIMED_WAKE = (
    "定时唤醒：结合当前时间和完整上下文，"
    "决定现在是否适合自然延续、处理已有结果或安静陪伴。"
)


class _SpeechSink:
    def __init__(self, source: GeneratedExpressionInput):
        self.source = source

    async def set_emotion(self, emotion):
        await self.source.feed(AnswerLaneDelta(emotion=emotion))

    async def append(self, text, *, continuation=False):
        await self.source.feed(AnswerLaneDelta(answer=text))

    async def flush(self):
        await self.source.feed(AnswerLaneDelta(answer_complete=True))

    async def flush_through(self, end_offset):
        await self.flush()

    async def set_end(self, end):
        pass


class ExperimentalRuntime:
    def __init__(
        self,
        service: ExperimentalService,
        *,
        contexts,
        expression,
        streams,
        foreground_generation,
        cancel_delivery,
        history=None,
        thought_context_state=lambda fish, session: None,
        material_store: MaterialStore | None = None,
        recall_store=None,
        embedder=None,
    ):
        self.service = service
        service.runtime = self
        self.contexts, self.expression, self.streams = contexts, expression, streams
        self.history = history
        self.foreground_generation, self.cancel_delivery = (
            foreground_generation,
            cancel_delivery,
        )
        self.binding: tuple[UUID, UUID] | None = None
        self._consumers: set[UUID] = set()
        self._foreground_busy = False
        self.suspended = False
        self._decision: asyncio.Task | None = None
        self._speaking: UUID | None = None
        self._epoch = 0
        self._started = service.now()
        self._seen: OrderedDict[UUID, None] = OrderedDict()
        self._closed = False
        self._busy_started: datetime | None = None
        self._busy_intervals: deque[tuple[datetime, datetime]] = deque(maxlen=256)
        # Events that arrived while TA was busy, handed over together once idle.
        self._waiting: deque[tuple[str, UUID, bool]] = deque(maxlen=MAX_WAITING_EVENTS)
        self._timer = asyncio.create_task(self._ticks(), name="experimental-idle-wake")
        self.thoughts = (
            ThoughtWorker(
                service,
                contexts,
                history,
                binding=lambda: self.binding,
                context_state=thought_context_state,
            )
            if history is not None
            else None
        )
        self.materials = (
            MaterialScout(
                service,
                contexts,
                history,
                material_store,
                recall=recall_store,
                embedder=embedder,
                binding=lambda: self.binding,
            )
            if history is not None and material_store is not None
            else None
        )
        if self.thoughts is not None and self.materials is not None:
            self.thoughts.on_outcome = self.materials.note

    @property
    def foreground_busy(self):
        return self._foreground_busy

    @foreground_busy.setter
    def foreground_busy(self, value):
        self._foreground_busy = value
        self.refresh_busy()

    def refresh_busy(self):
        if self.busy and self._busy_started is None:
            self._busy_started = self.service.now()
        elif not self.busy and self._busy_started is not None:
            self._busy_intervals.append((self._busy_started, self.service.now()))
            self._busy_started = None

    @property
    def active(self):
        """主动陪伴: TA speaks up on its own."""
        return self.service.enabled and self._attached

    @property
    def _attached(self):
        return self.binding is not None and not self.suspended and not self._closed

    def _responsive(self, reports_work: bool) -> bool:
        # What TA went to find out answers the user's own request, so TA
        # reports it back even while 主动陪伴 is off.
        return self.active or (
            reports_work and self._attached and not self.service.closed
        )

    @property
    def busy(self):
        return (
            self.foreground_busy
            or any(item.pending_input for item in self.service.preparations)
            or self.foreground_generation()
            or self._speaking is not None
            or (self._decision is not None and not self._decision.done())
        )

    def bind(self, fish: UUID, session: UUID):
        if self.binding != (fish, session):
            self.reset()
            self.binding = fish, session

    def subscribe(self, token: UUID, fish: UUID, session: UUID):
        self._consumers.add(token)
        self.bind(fish, session)

    def unsubscribe(self, token: UUID):
        self._consumers.discard(token)
        if not self._consumers:
            self.reset()
            self.binding = None

    def reset(self):
        self._epoch += 1
        if self.thoughts is not None:
            self.thoughts.reset()
        self._started = self.service.now()
        self._seen.clear()
        self._waiting.clear()
        self.interrupt()

    def interrupt(self):
        if self._decision is not None:
            self._decision.cancel()
        if self._speaking is not None:
            stream = self.streams.get(self._speaking)
            if stream is not None:
                self.cancel_delivery(stream.intent.fish_id, self._speaking)
        self._speaking = None
        self.refresh_busy()

    def consumed(self, message_id: UUID, offset: int, interrupted: bool) -> bool:
        stream = self.streams.get(message_id)
        if stream is None or not stream.intent.opportunity_id.startswith(
            "experimental:"
        ):
            return False
        stream.intent.consumed_offset = min(max(0, offset), len(stream.text))
        stream.intent.state = "interrupted" if interrupted else "completed"
        if self._speaking == message_id:
            self._speaking = None
        self.refresh_busy()
        # Completion updates History through normal playback receipts. No wake.
        return True

    def observe(
        self,
        fish: UUID,
        session: UUID,
        unit: HistoryUnit,
        received_at: datetime | None = None,
    ):
        if self.materials is not None and self.binding == (fish, session):
            self.materials.observe(unit)
        # With 主动陪伴 on, what wakes TA wakes it now and the rest waits for
        # the timed wake; with it off, TA answers only what is for the user.
        wakes = unit.triggers_decision if self.active else unit.reports_work
        if (
            not self._responsive(unit.reports_work)
            or self.binding != (fish, session)
            or not wakes
            or unit.label not in {"as", "event"}
            or unit.ended_at < self._started
            or unit.source_id in self._seen
        ):
            return
        self._seen[unit.source_id] = None
        while len(self._seen) > 4096:
            self._seen.popitem(last=False)
        event = f"新事件（{format_prompt_timestamp(unit.ended_at)}）：{unit.content}"
        # History projection is asynchronous: an event received during speech
        # but projected afterwards still belongs to the busy period.
        received_while_busy = received_at is not None and any(
            start <= received_at < end for start, end in self._busy_intervals
        )
        if self.busy or received_while_busy or self._waiting:
            self._waiting.append((event, unit.source_id, unit.reports_work))
            return
        self.trigger(event, source=unit.source_id, reports_work=unit.reports_work)

    def trigger(
        self, trigger: str, *, source: UUID | None = None, reports_work: bool = False
    ):
        # `source` is the Observation behind the trigger; the chat page shows why.
        if not self._responsive(reports_work) or self.busy:
            return
        # Everything this decision calls counts as TA speaking up on its own.
        with purpose_scope("proactive"):
            self._decision = asyncio.create_task(
                self._act(trigger, source, reports_work), name="experimental-event"
            )
        self.refresh_busy()
        self._decision.add_done_callback(self._finished)

    def _finished(self, task):
        self.refresh_busy()
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "Experimental event processing failed", exc_info=task.exception()
            )

    async def tick(self):
        if self.busy:
            return
        if self._waiting:
            events = [event for event, _, _ in self._waiting]
            source = self._waiting[-1][1]
            reports_work = all(reports for _, _, reports in self._waiting)
            self._waiting.clear()
            self.trigger("\n".join(events), source=source, reports_work=reports_work)
            return
        if not self.active:
            return
        last = self.service.last_decision_at or self._started
        if (
            self.service.now() - last
        ).total_seconds() < self.service.configuration.wake_seconds:
            return
        if self.service.configuration.thoughts_enabled:
            if self.history is None or self.binding is None:
                return
            binding, epoch = self.binding, self._epoch
            try:
                ready = await self.history.store.has_thought_since_last_speech(
                    *binding, self.service.now()
                )
            except Exception:
                logger.exception("Experimental timed wake thought check failed")
                return
            if (
                self.binding != binding
                or self._epoch != epoch
                or (self.service.last_decision_at or self._started) != last
            ):
                return
            if self.service.configuration.thoughts_enabled and not ready:
                return
        # The check owns no busy slot: events and user input remain independent.
        self.trigger(TIMED_WAKE)

    async def _ticks(self):
        while not self._closed:
            await asyncio.sleep(1)
            await self.tick()

    async def _act(
        self, trigger: str, source: UUID | None = None, reports_work: bool = False
    ):
        assert self.binding is not None
        fish, session = self.binding
        epoch, revision = self._epoch, self.service.revision

        def valid():
            return (
                self._responsive(reports_work)
                and self.binding == (fish, session)
                and self._epoch == epoch
                and self.service.revision == revision
                and not self.foreground_busy
                and not self.foreground_generation()
            )

        context = await self.contexts.load_for_proactive(
            fish, session, include_image=False, tracker=MessagePerformanceTracker()
        )
        if not valid():
            return
        captured_at = self.service.now()
        recent = without_repeated_speech(
            with_recent_thoughts(context.messages, SPEECH_THOUGHT_CONTEXT)
        )
        state = render_state(recent, captured_at, trigger)
        available = await self.service.available(context)
        if not valid():
            return
        action = await self.service.decide(state, available)
        if action == "wait" and trigger == TIMED_WAKE and self.materials is not None:
            self.materials.note("wait")
        if not valid() or action == "wait":
            return
        perception = (
            await self.service.perception.capture(
                fish, self.service.now(), model=self.service.chat
            )
            if self.service.perception is not None
            else PerceptionInput()
        )
        if not valid():
            return
        identity = uuid4()
        intent = ExpressionIntent(
            identity,
            fish,
            session,
            revision,
            f"experimental:{identity}",
            str(identity),
            1,
            "speak",
            trigger,
            self.service.now(),
            source_observation_id=source,
        )
        stream = self.streams.create(intent)
        stream.task = asyncio.current_task()
        speech = GeneratedExpressionInput(speech_tags=context.speech_tags)
        self._speaking = identity
        delivery = asyncio.create_task(
            self.expression.compose_generated(stream, speech, valid=valid)
        )
        try:
            # The trigger reaches the speaker as a user message, so it is told
            # plainly that nobody said it: replayed, TA opened 2 of 15 lines
            # with "好的/好嘞" as if answering an order, and none once told.
            # Whether to speak on a timed wake is Jev's to weigh; handed that
            # question as well, the speaker spoke of going quiet instead, 24
            # of 60 times after being asked to keep talking, and 11 of 60
            # once told only that it wants to speak.
            said = "你现在想说话。" if trigger == TIMED_WAKE else trigger
            channel = await generate_channel(
                self.service.chat,
                action,
                expression_context(recent, captured_at, "这不是用户说的话。" + said),
                _SpeechSink(speech),
                perception=perception,
            )
            await speech.end()
            await delivery
            if valid() and action in {"r", "t", "w"}:
                self.service.launch(fish, session, identity, channel, context, state)
        except BlankFieldError:
            # qwen sometimes writes an empty line under the strict schema:
            # nothing to say this time, so TA stays quiet.
            logger.info("Experimental %s line came back empty; TA stays quiet", action)
        finally:
            await speech.end(failed=True)
            if not delivery.done():
                delivery.cancel()
            await asyncio.gather(delivery, return_exceptions=True)
            if not stream.published and self._speaking == identity:
                self._speaking = None

    async def admit_window(self, document: SemanticDocument) -> bool:
        # Part of 内心, judged whether or not 主动陪伴 is on.
        if self.service.closed:
            return True
        summaries = {
            "audiovisualSummary": document.audiovisual_summary,
            "activitySummary": document.activity_summary,
        }
        if not any(value.strip() for value in summaries.values()):
            return False
        revision = self.service.revision
        configuration = self.service.configuration
        # Perception filtering runs independently of the behavior/speech owner.
        accepted = await self.service.judge.admit(configuration, summaries)
        return accepted if revision == self.service.revision else True

    async def close(self):
        self._closed = True
        if self.thoughts is not None:
            await self.thoughts.close()
        if self.materials is not None:
            await self.materials.close()
        self.reset()
        self._timer.cancel()
        await asyncio.gather(
            self._timer,
            *([self._decision] if self._decision else []),
            return_exceptions=True,
        )
