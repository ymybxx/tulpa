"""One cancellable answer task, from private computation to public delivery."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.generation import AnswerGenerationError
from fish_memory.answer.lane_runtime import (
    AnswerLaneHandle,
    AnswerLaneResult,
    LaneStreamError,
    cancel_handles,
    cancel_unselected,
)
from fish_memory.answer.output_sink import AnswerOutputSink, PromotableAnswerSink
from fish_memory.answer.runtime import FishRuntime
from fish_memory.home.models import HomeIntentType, HomeRunOutcome
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker

if TYPE_CHECKING:
    from fish_memory.agent.replies import PreparedAgentReply
    from fish_memory.answer.parallel_generation import (
        ParallelAnswerGenerationPipeline,
        _RouteContext,
    )


class HomeRunResultHandle(Protocol):
    async def result(self) -> HomeRunOutcome: ...


class PreparedAnswer(Protocol):
    """Owned preparation shared by normal answers and exact Agent replies."""

    sink: PromotableAnswerSink
    input_context: LoadedAnswerContext
    prepared_reply: PreparedAgentReply | None

    @property
    def input_hash(self) -> str: ...

    async def input_ready(self) -> str: ...

    async def initial_ready(self) -> AnswerLaneResult: ...

    def initial_done(self) -> bool: ...

    async def preparation_ready(self) -> AnswerLaneResult: ...

    def preparation_done(self) -> bool: ...

    def output_token_usage(self) -> int | None: ...

    def route_evidence(self) -> dict[str, JsonValue] | None: ...

    def record_preparation_performance(
        self, tracker: MessagePerformanceTracker
    ) -> None: ...

    def cancel(self) -> asyncio.Task[None]: ...

    async def release_to_public(
        self,
        public_sink: AnswerOutputSink,
        *,
        source_message_id: UUID | None,
        answer_message_id: UUID,
        runtime: FishRuntime,
        session_id: UUID | None = None,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ) = None,
    ) -> None: ...


class PreparedInitialAnswer:
    """Owns frozen input, private computation, and exactly one public release."""

    def __init__(
        self,
        *,
        pipeline: ParallelAnswerGenerationPipeline,
        input_hash_builder: Callable[[], str],
        sink: PromotableAnswerSink,
        input_context: LoadedAnswerContext,
        prepared_reply: PreparedAgentReply | None = None,
        route_context: _RouteContext,
        handles: Sequence[AnswerLaneHandle],
    ) -> None:
        self._input_hash: str | None = None
        self._input_hash_builder = input_hash_builder
        self._input_ready = asyncio.Event()
        self.input_context = input_context
        self.prepared_reply = prepared_reply
        self.sink = sink
        self.result: AnswerLaneResult | None = None
        self._pipeline = pipeline
        self._route_context = route_context
        self._handles = tuple(handles)
        self._selected: AnswerLaneHandle | None = None
        self._selection_ready = asyncio.Event()
        self._publisher_task: asyncio.Task[None] | None = None
        self._commit_task: asyncio.Task[None] | None = None
        self._release_task: asyncio.Task[None] | None = None
        self._state_lock = asyncio.Lock()
        self._cancelled = False
        self._continued = False
        self._cancel_cleanup_task: asyncio.Task[None] | None = None
        self._preparation_performance_recorded = False
        self._initial_task = asyncio.create_task(
            self._prepare(),
            name="answer-initial-preparation",
        )
        self._preparation_task = asyncio.create_task(
            self._prepare_route(), name="answer-route-preparation"
        )
        self._preparation_task.add_done_callback(_observe_task_exception)

    async def input_ready(self) -> str:
        """Wait for the once-only input audit, without waiting for model output."""
        await self._input_ready.wait()
        if self._input_hash is None:
            await self.initial_ready()
        return self.input_hash

    async def preparation_ready(self) -> AnswerLaneResult:
        """Wait for all computation that can finish without a durable commit."""
        return await asyncio.shield(self._preparation_task)

    def preparation_done(self) -> bool:
        return self._preparation_task.done()

    async def _prepare_route(self) -> AnswerLaneResult:
        result = await self.initial_ready()
        self._require_owned()
        try:
            await self._pipeline._prepare_route(self._route_context, result)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise AnswerGenerationError from error
        self._require_owned()
        return result

    async def initial_ready(self) -> AnswerLaneResult:
        """Wait until the selected lane's complete initial output is buffered."""
        return await asyncio.shield(self._initial_task)

    def initial_done(self) -> bool:
        """Return whether initial preparation reached a terminal result."""
        return self._initial_task.done()

    def output_token_usage(self) -> int | None:
        """Return only provider-supplied aggregate output usage for draft lanes."""
        total = 0
        found = False
        for step in self._route_context.tracker.snapshot():
            payload = step.output_payload
            if not isinstance(payload, Mapping):
                continue
            value = payload.get("output_tokens")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                total += value
                found = True
        return total if found else None

    def route_evidence(self) -> dict[str, JsonValue] | None:
        """Return the selected route without prompt or generated content."""
        result = self.result
        if result is None:
            return None
        return {
            "route": result.name,
            "recall_depth": result.recall_depth,
            "context_complete": result.recall_depth == 0,
        }

    def record_preparation_performance(
        self, tracker: MessagePerformanceTracker
    ) -> None:
        """Preserve private timings without changing the public latency origin."""
        source = self._route_context.tracker
        if self._preparation_performance_recorded or source is tracker:
            return
        if not self._preparation_task.done() or not self._initial_task.done():
            raise RuntimeError("answer preparation must settle before auditing")
        steps: list[JsonValue] = [
            {
                "stepName": step.name,
                "stepIndex": step.index,
                "startedOffsetUs": step.started_offset_us,
                "durationUs": step.duration_us,
                "inputPayload": copy.deepcopy(step.input_payload),
                "outputPayload": copy.deepcopy(step.output_payload),
                "error": step.error,
            }
            for step in source.snapshot()
        ]
        now_ns = tracker.now_ns()
        tracker.record(
            "voice_draft.preparation",
            started_ns=now_ns,
            finished_ns=now_ns,
            output_payload={
                "originRelativeToMessageUs": (source.origin_ns - tracker.origin_ns)
                // 1_000,
                "steps": steps,
            },
        )
        self._preparation_performance_recorded = True

    @property
    def input_hash(self) -> str:
        if self._input_hash is None:
            raise RuntimeError("initial input hash is not ready")
        return self._input_hash

    async def release_to_public(
        self,
        public_sink: AnswerOutputSink,
        *,
        source_message_id: UUID | None,
        answer_message_id: UUID,
        runtime: FishRuntime,
        session_id: UUID | None = None,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ) = None,
    ) -> None:
        """Adopt existing computation; commit mutations once and publish its output."""
        self._require_owned()
        if self._release_task is not None:
            raise RuntimeError("prepared initial answer was already continued")
        self._release_task = asyncio.create_task(
            self._release_to_public(
                public_sink,
                source_message_id=source_message_id,
                answer_message_id=answer_message_id,
                runtime=runtime,
                session_id=session_id,
                start_home_action=start_home_action,
            ),
            name="answer-public-release",
        )
        await self._release_task

    async def _release_to_public(
        self,
        public_sink: AnswerOutputSink,
        *,
        source_message_id: UUID | None,
        answer_message_id: UUID,
        runtime: FishRuntime,
        session_id: UUID | None,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ),
    ) -> None:
        selected = await self._wait_for_selected_lane()
        async with self._state_lock:
            self._require_owned()
            if self._continued:
                raise RuntimeError("prepared initial answer was already continued")
            self._continued = True
            self._route_context.source_message_id = source_message_id
            self._route_context.answer_message_id = answer_message_id
            self._route_context.runtime = runtime
            self._route_context.session_id = session_id
        # Task acceptance must be durable before any acknowledgement is exposed.
        # Other lanes may publish their selected prefix while computation finishes.
        if selected.name == "task":
            await self._commit(start_home_action)
        self._require_owned()
        await self.sink.attach(public_sink)
        if selected.name != "task":
            await self._commit(start_home_action)

    async def continue_after_commit(
        self,
        *,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ) = None,
        answer_message_id: UUID | None = None,
        runtime: FishRuntime | None = None,
        session_id: UUID | None = None,
    ) -> None:
        """Compatibility entry point for callers that already own an output sink."""
        self._require_owned()
        result = await self.initial_ready()
        if result.name == "task":
            raise RuntimeError("Task lane requires release_to_public")
        async with self._state_lock:
            self._require_owned()
            if self._continued:
                raise RuntimeError("prepared initial answer was already continued")
            self._continued = True
            self._route_context.answer_message_id = answer_message_id
            self._route_context.runtime = runtime
            self._route_context.session_id = session_id
        await self._commit(start_home_action)

    async def _commit(
        self,
        start_home_action: (
            Callable[[HomeIntentType], HomeRunResultHandle | None] | None
        ),
    ) -> None:
        result = await self.preparation_ready()
        self._require_owned()
        self._commit_task = asyncio.create_task(
            self._pipeline._commit_route(
                self._route_context, result, start_home_action=start_home_action
            ),
            name="answer-route-commit",
        )
        try:
            await self._commit_task
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise AnswerGenerationError from error

    def cancel(self) -> asyncio.Task[None]:
        """Revoke the prepared owner and asynchronously consume all workers."""
        if self._cancelled:
            if self._cancel_cleanup_task is None:
                raise RuntimeError("cancelled prepared answer has no cleanup task")
            return self._cancel_cleanup_task
        self._cancelled = True
        self.sink.revoke()
        self._input_ready.set()
        self._selection_ready.set()
        cancel_handles(self._handles)
        for task in (
            self._publisher_task,
            self._commit_task,
            self._release_task,
            self._initial_task,
            self._preparation_task,
        ):
            if task is not None and not task.done():
                task.cancel()
        self._cancel_cleanup_task = asyncio.create_task(
            self._settle_cancelled(),
            name="answer-preparation-cancel-cleanup",
        )
        return self._cancel_cleanup_task

    async def _prepare(self) -> AnswerLaneResult:
        publisher_task: asyncio.Task[None] | None = None
        try:
            # The caller has started every configured lane before this task gets
            # a scheduling turn. Keep that observable all-lanes-started contract.
            await asyncio.sleep(0)
            self._require_owned()
            self._input_hash = self._input_hash_builder()
            self._input_ready.set()
            selected = await self._pipeline._arbitrator.select(self._handles)
            self._require_owned()
            self._selected = selected
            cancel_unselected(self._handles, selected)
            self._pipeline._record_route_decision(self._route_context.tracker)
            if selected.name != "recall":
                self._pipeline._record_recall_depth(0, self._route_context.tracker)
            publisher_task = asyncio.create_task(
                self._pipeline._publish_selected_lane(
                    selected,
                    self.sink,
                    owns=lambda: not self._cancelled,
                ),
                name=f"answer-publisher:{selected.name}",
            )
            self._publisher_task = publisher_task
            self._selection_ready.set()
            result = await _finish_selected_lane(selected, publisher_task)
            self._require_owned()
            self.result = result
            return result
        except asyncio.CancelledError:
            raise
        except LaneStreamError as error:
            raise AnswerGenerationError from error
        finally:
            self._input_ready.set()
            self._selection_ready.set()
            if publisher_task is not None and not publisher_task.done():
                publisher_task.cancel()
            await self._pipeline._lane_runner.settle(self._handles)
            if publisher_task is not None:
                await asyncio.gather(publisher_task, return_exceptions=True)

    async def _wait_for_selected_lane(self) -> AnswerLaneHandle:
        await self._selection_ready.wait()
        if self._selected is None:
            await self.initial_ready()
            raise RuntimeError(
                "prepared initial answer completed without a selected lane"
            )
        return self._selected

    async def _settle_cancelled(self) -> None:
        tasks: list[asyncio.Task[object]] = [self._initial_task, self._preparation_task]
        if self._publisher_task is not None:
            tasks.append(self._publisher_task)
        if self._commit_task is not None:
            tasks.append(self._commit_task)
        if self._release_task is not None:
            tasks.append(self._release_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        await self._pipeline._lane_runner.settle(self._handles)

    def _require_owned(self) -> None:
        if self._cancelled:
            raise RuntimeError("prepared initial answer was cancelled")


def _observe_task_exception(task: asyncio.Task[object]) -> None:
    # A revoked or expired private task can fail before a caller awaits it.
    # Retrieving the exception prevents warnings without changing await semantics.
    if not task.cancelled():
        task.exception()


async def _finish_selected_lane(
    lane: AnswerLaneHandle,
    publish_task: asyncio.Task[None],
) -> AnswerLaneResult:
    if lane.task is None:
        raise RuntimeError("selected answer lane has no task")
    try:
        result = await lane.task
    except asyncio.CancelledError:
        raise
    except BaseException:
        # The lane runner always closes its private answer queue on terminal
        # failure. Drain any already-visible prefix, then preserve the error.
        await asyncio.gather(publish_task, return_exceptions=True)
        raise
    await publish_task
    return result
