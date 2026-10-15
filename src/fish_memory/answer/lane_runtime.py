from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from fish_memory.agent.routing import AgentTaskDecisionKind
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.lane_stream import AnswerLaneDelta
from fish_memory.home.models import HomeIntentType
from fish_memory.llm.client import ChatClient
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class LaneStreamError(Exception):
    """A lane provider or its compact streaming protocol failed."""


class AnswerLaneParser(Protocol):
    def feed(self, chunk: str) -> AnswerLaneDelta: ...

    def finish(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AnswerLaneResult:
    name: str
    accepted: bool
    query: str | None = None
    recall_depth: int = 0
    end_conversation: bool = False
    home_intent: HomeIntentType | None = None
    task_kind: AgentTaskDecisionKind | None = None
    candidate_indexes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerLaneOutput:
    text: str = ""
    flush: bool = False


LaneResultBuilder = Callable[[Mapping[str, object]], AnswerLaneResult]


@dataclass(frozen=True, slots=True)
class AnswerLaneDefinition:
    """Static behavior for one independently cancellable answer lane.

    A numeric priority makes the lane a decision gate. ``None`` marks the
    single fallback lane, which is selected only after every gate rejects.
    """

    name: str
    priority: int | None
    parser_factory: Callable[[], AnswerLaneParser]
    result_builder: LaneResultBuilder
    performance_step: str
    max_attempts: int = 1
    record_initial_timing: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("answer lane name must not be empty")
        if self.max_attempts < 1:
            raise ValueError("answer lane max attempts must be positive")


@dataclass(frozen=True, slots=True)
class AnswerLaneInvocation:
    definition: AnswerLaneDefinition
    messages: list[ChatMessage]


@dataclass(slots=True)
class AnswerLaneHandle:
    invocation: AnswerLaneInvocation
    emotion: asyncio.Future[PetEmotion]
    answers: asyncio.Queue[AnswerLaneOutput | None]
    decision: asyncio.Future[bool] | None
    task: asyncio.Task[AnswerLaneResult] | None = None
    values: dict[str, object] = field(default_factory=dict)
    answer_emitted: bool = False
    llm_started_ns: int = 0

    @property
    def name(self) -> str:
        return self.invocation.definition.name

    @property
    def priority(self) -> int | None:
        return self.invocation.definition.priority

    def request_cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()

    def consume_future_error(self) -> None:
        # Decision/emotion futures intentionally expose data before the full
        # lane task ends. Retrieve terminal exceptions during cleanup so a
        # failed, unselected lane never leaks "exception was never retrieved".
        for future in (self.decision, self.emotion):
            if future is not None and future.done() and not future.cancelled():
                future.exception()


class AnswerLaneRunner:
    """Runs lane LLM streams without touching shared answer state."""

    def __init__(self, llm: ChatClient) -> None:
        self._llm = llm

    def start(
        self,
        invocation: AnswerLaneInvocation,
        tracker: MessagePerformanceTracker,
    ) -> AnswerLaneHandle:
        loop = asyncio.get_running_loop()
        handle = AnswerLaneHandle(
            invocation=invocation,
            emotion=loop.create_future(),
            answers=asyncio.Queue(),
            decision=(
                loop.create_future()
                if invocation.definition.priority is not None
                else None
            ),
            llm_started_ns=tracker.now_ns(),
        )
        handle.task = asyncio.create_task(
            self._consume(handle, tracker),
            name=f"answer-lane:{handle.name}",
        )
        return handle

    async def settle(self, handles: Sequence[AnswerLaneHandle]) -> None:
        cancel_handles(handles)
        tasks: list[asyncio.Task[AnswerLaneResult]] = []
        for handle in handles:
            if handle.task is not None:
                tasks.append(handle.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for handle in handles:
            handle.consume_future_error()

    async def _consume(
        self,
        handle: AnswerLaneHandle,
        tracker: MessagePerformanceTracker,
    ) -> AnswerLaneResult:
        definition = handle.invocation.definition
        try:
            for attempt in range(definition.max_attempts):
                parser = definition.parser_factory()
                try:
                    await self._consume_attempt(handle, parser, tracker)
                    try:
                        return definition.result_builder(handle.values)
                    except (TypeError, ValueError) as exception:
                        raise LaneStreamError from exception
                except asyncio.CancelledError:
                    raise
                except LaneStreamError as exception:
                    if handle.answer_emitted or attempt + 1 == definition.max_attempts:
                        self._fail_early_values(handle, exception)
                        raise
                    handle.values.clear()
                    logger.warning(
                        "Retrying failed %s answer lane before visible content",
                        handle.name,
                    )
            raise AssertionError("lane attempt loop did not return")
        finally:
            handle.answers.put_nowait(None)

    async def _consume_attempt(
        self,
        handle: AnswerLaneHandle,
        parser: AnswerLaneParser,
        tracker: MessagePerformanceTracker,
    ) -> None:
        definition = handle.invocation.definition
        messages = handle.invocation.messages
        raw: list[str] = []
        finish_reason: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        error: str | None = None
        first_packet = True
        attempt_started_ns = tracker.now_ns()
        try:
            async for chunk in self._llm.stream_json(messages, force_search=False):
                if first_packet:
                    if definition.record_initial_timing:
                        tracker.record(
                            "llm.first_packet",
                            started_ns=handle.llm_started_ns,
                            finished_ns=tracker.now_ns(),
                        )
                    first_packet = False
                finish_reason = chunk.finish_reason or finish_reason
                if chunk.input_tokens is not None:
                    input_tokens = chunk.input_tokens
                if chunk.output_tokens is not None:
                    output_tokens = chunk.output_tokens
                if not chunk.content:
                    continue
                raw.append(chunk.content)
                delta = parser.feed(chunk.content)
                self._apply_delta(handle, delta, tracker)
            parser.finish()
        except asyncio.CancelledError as exception:
            error = type(exception).__name__
            raise
        except Exception as exception:
            error = type(exception).__name__
            raise LaneStreamError from exception
        finally:
            tracker.record(
                definition.performance_step,
                started_ns=attempt_started_ns,
                finished_ns=tracker.now_ns(),
                input_payload=self._llm.json_stream_request_payload(messages),
                output_payload={
                    "content": "".join(raw),
                    "finish_reason": finish_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                error=error,
            )

    @staticmethod
    def _apply_delta(
        handle: AnswerLaneHandle,
        delta: AnswerLaneDelta,
        tracker: MessagePerformanceTracker,
    ) -> None:
        if delta.accepted is not None:
            handle.values["accepted"] = delta.accepted
            if handle.decision is None:
                raise ValueError("fallback lane emitted an acceptance decision")
            _resolve(handle.decision, delta.accepted)
        if delta.emotion is not None:
            handle.values["emotion"] = delta.emotion
            _resolve(handle.emotion, delta.emotion)
        if delta.home_intent is not None:
            handle.values["home_intent"] = delta.home_intent
        if delta.recall_depth is not None:
            handle.values["recall_depth"] = delta.recall_depth
        if delta.query is not None:
            handle.values["query"] = delta.query
        if delta.end is not None:
            handle.values["end"] = delta.end
        if delta.task_kind is not None:
            handle.values["task_kind"] = delta.task_kind
            handle.values["candidate_indexes"] = delta.candidate_indexes
        elif delta.candidate_indexes:
            handle.values["candidate_indexes"] = delta.candidate_indexes
        if delta.answer:
            if not handle.answer_emitted:
                if handle.invocation.definition.record_initial_timing:
                    tracker.record(
                        "llm.first_content",
                        started_ns=handle.llm_started_ns,
                        finished_ns=tracker.now_ns(),
                    )
                handle.answer_emitted = True
            handle.answers.put_nowait(AnswerLaneOutput(text=delta.answer))
        if delta.answer_complete:
            handle.answers.put_nowait(AnswerLaneOutput(flush=True))

    @staticmethod
    def _fail_early_values(
        handle: AnswerLaneHandle,
        error: BaseException,
    ) -> None:
        for future in (handle.decision, handle.emotion):
            if future is not None and not future.done():
                future.set_exception(error)


class AnswerLaneArbitrator:
    """Selects one lane by priority without granting it shared-state access."""

    async def select(
        self,
        handles: Sequence[AnswerLaneHandle],
    ) -> AnswerLaneHandle:
        fallbacks = [handle for handle in handles if handle.priority is None]
        if len(fallbacks) != 1:
            raise ValueError("exactly one fallback answer lane is required")
        gates = sorted(
            (handle for handle in handles if handle.priority is not None),
            key=lambda handle: handle.priority or 0,
            reverse=True,
        )
        for handle in gates:
            if await self._accepted_or_false(handle):
                return handle
        return fallbacks[0]

    @staticmethod
    async def _accepted_or_false(handle: AnswerLaneHandle) -> bool:
        if handle.decision is None:
            raise ValueError("decision lane is missing its decision future")
        try:
            return await handle.decision
        except Exception:
            logger.exception(
                "%s answer lane failed before its decision; treating it as rejected",
                handle.name,
            )
            return False


def cancel_unselected(
    handles: Sequence[AnswerLaneHandle],
    selected: AnswerLaneHandle,
) -> None:
    """Request loser cancellation without delaying selected-lane publication."""

    cancel_handles(handle for handle in handles if handle is not selected)


def cancel_handles(handles: Iterable[AnswerLaneHandle]) -> None:
    """Synchronously revoke lane execution before asynchronous settling begins."""

    for handle in handles:
        handle.request_cancel()


def _resolve(future: asyncio.Future[_T], value: _T) -> None:
    if not future.done():
        future.set_result(value)
