import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from fish_memory.answer.runtime import FishRuntime
from fish_memory.models import MessageStatus
from fish_memory.performance.models import PerformanceStep
from fish_memory.performance.tracker import MessagePerformanceTracker

logger = logging.getLogger(__name__)


class TerminalMessageStore(Protocol):
    async def finish_answer(
        self,
        message_id: UUID,
        fish_id: UUID,
        *,
        content: str,
        requested_status: MessageStatus,
        failure: str | None = None,
    ) -> MessageStatus: ...


class PerformanceStepStore(Protocol):
    async def insert_steps(
        self,
        message_id: UUID,
        steps: Sequence[PerformanceStep],
    ) -> None: ...


class AnswerLifecycle:
    def __init__(
        self,
        store: TerminalMessageStore,
        *,
        performance_store: PerformanceStepStore | None = None,
    ) -> None:
        self._store = store
        self._performance_store = performance_store
        self._performance_tasks: set[asyncio.Task[None]] = set()

    async def complete(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        source_message_id: UUID,
        runtime: FishRuntime,
        tracker: MessagePerformanceTracker,
    ) -> None:
        await self._finish(
            fish_id,
            answer_message_id,
            source_message_id,
            runtime,
            tracker,
            expected_status=MessageStatus.GENERATING,
            requested_status=MessageStatus.COMPLETED,
            fallback_status=MessageStatus.FAILED,
        )

    async def fail(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        source_message_id: UUID,
        runtime: FishRuntime,
        tracker: MessagePerformanceTracker,
        failure: str | None = None,
    ) -> None:
        await self._finish(
            fish_id,
            answer_message_id,
            source_message_id,
            runtime,
            tracker,
            expected_status=MessageStatus.GENERATING,
            requested_status=MessageStatus.FAILED,
            fallback_status=MessageStatus.FAILED,
            failure=failure,
        )

    async def interrupt(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        source_message_id: UUID,
        runtime: FishRuntime,
        tracker: MessagePerformanceTracker,
    ) -> None:
        await self._finish(
            fish_id,
            answer_message_id,
            source_message_id,
            runtime,
            tracker,
            expected_status=MessageStatus.INTERRUPTED,
            requested_status=MessageStatus.INTERRUPTED,
            fallback_status=MessageStatus.INTERRUPTED,
        )

    async def _finish(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        source_message_id: UUID,
        runtime: FishRuntime,
        tracker: MessagePerformanceTracker,
        *,
        expected_status: MessageStatus,
        requested_status: MessageStatus,
        fallback_status: MessageStatus,
        failure: str | None = None,
    ) -> None:
        async with runtime.condition:
            if (
                runtime.answer_message_id != answer_message_id
                or runtime.status is not expected_status
            ):
                return
            content = runtime.content

        try:
            with tracker.measure("answer.persist"):
                status = await self._store.finish_answer(
                    answer_message_id,
                    fish_id,
                    content=content,
                    requested_status=requested_status,
                    failure=failure,
                )
        except asyncio.CancelledError:
            # The coordinator cancels generation before persisting its playback
            # cutoff. Do not overwrite INTERRUPTED or finish its tracker here:
            # the cancelled transaction can still leave the row generating.
            raise
        except BaseException:
            await self._publish_terminal_status(
                answer_message_id,
                runtime,
                fallback_status,
                failure,
            )
            self.submit_performance(source_message_id, tracker)
            raise

        await self._publish_terminal_status(answer_message_id, runtime, status, failure)
        self.submit_performance(source_message_id, tracker)

    async def _publish_terminal_status(
        self,
        answer_message_id: UUID,
        runtime: FishRuntime,
        status: MessageStatus,
        failure: str | None = None,
    ) -> None:
        async with runtime.condition:
            if runtime.answer_message_id != answer_message_id:
                return
            runtime.failure = failure
            runtime.status = status
            runtime.source_message_id = None
            runtime.performance_tracker = None
            runtime.condition.notify_all()

    def submit_performance(
        self,
        message_id: UUID,
        tracker: MessagePerformanceTracker,
    ) -> None:
        steps = tracker.finish()
        if self._performance_store is None:
            return
        task = asyncio.create_task(self._persist_performance(message_id, steps))
        self._performance_tasks.add(task)
        task.add_done_callback(self._performance_tasks.discard)

    async def _persist_performance(
        self,
        message_id: UUID,
        steps: tuple[PerformanceStep, ...],
    ) -> None:
        assert self._performance_store is not None
        try:
            await self._performance_store.insert_steps(message_id, steps)
        except Exception:
            logger.exception(
                "Failed to persist performance steps for message %s",
                message_id,
            )

    async def close(self) -> None:
        if self._performance_tasks:
            await asyncio.gather(*self._performance_tasks, return_exceptions=True)
        self._performance_tasks.clear()
