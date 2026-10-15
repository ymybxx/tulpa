import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest

from fish_memory.answer.lifecycle import AnswerLifecycle
from fish_memory.answer.runtime import FishRuntime
from fish_memory.models import MessageStatus
from fish_memory.performance.models import PerformanceStep
from fish_memory.performance.tracker import MessagePerformanceTracker

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
ANSWER_ID = UUID("00000000-0000-0000-0000-000000000002")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000000003")


def runtime_with(
    status: MessageStatus,
    *,
    answer_id: UUID = ANSWER_ID,
    content: str = "hello",
) -> FishRuntime:
    runtime = FishRuntime()
    runtime.answer_message_id = answer_id
    runtime.status = status
    runtime.content = content
    runtime.source_message_id = SOURCE_ID
    runtime.performance_tracker = MessagePerformanceTracker()
    return runtime


class RecordingTerminalStore:
    def __init__(
        self,
        *,
        returned_status: MessageStatus | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.returned_status = returned_status
        self.failure = failure
        self.finish_calls: list[tuple[UUID, UUID, str, MessageStatus]] = []

    async def finish_answer(
        self,
        message_id: UUID,
        fish_id: UUID,
        *,
        content: str,
        requested_status: MessageStatus,
        failure: str | None = None,
    ) -> MessageStatus:
        self.finish_calls.append((message_id, fish_id, content, requested_status))
        if self.failure is not None:
            raise self.failure
        return self.returned_status or requested_status


class RecordingPerformanceStore:
    def __init__(self, *, blocked: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.writes: list[tuple[UUID, tuple[PerformanceStep, ...]]] = []
        if not blocked:
            self.release.set()

    async def insert_steps(
        self,
        message_id: UUID,
        steps: Sequence[PerformanceStep],
    ) -> None:
        self.started.set()
        await self.release.wait()
        self.writes.append((message_id, tuple(steps)))


@pytest.mark.asyncio
async def test_complete_persists_and_publishes_returned_status() -> None:
    store = RecordingTerminalStore(returned_status=MessageStatus.COMPLETED)
    performance = RecordingPerformanceStore()
    lifecycle = AnswerLifecycle(store, performance_store=performance)
    runtime = runtime_with(MessageStatus.GENERATING)
    tracker = MessagePerformanceTracker()

    await lifecycle.complete(FISH_ID, ANSWER_ID, SOURCE_ID, runtime, tracker)
    await lifecycle.close()

    assert store.finish_calls == [
        (ANSWER_ID, FISH_ID, "hello", MessageStatus.COMPLETED)
    ]
    assert runtime.status is MessageStatus.COMPLETED
    assert runtime.source_message_id is None
    assert runtime.performance_tracker is None
    assert performance.writes[0][0] == SOURCE_ID
    assert [step.name for step in performance.writes[0][1]] == [
        "answer.persist",
        "total",
    ]


@pytest.mark.asyncio
async def test_fail_persists_failed_status() -> None:
    store = RecordingTerminalStore()
    lifecycle = AnswerLifecycle(store)
    runtime = runtime_with(MessageStatus.GENERATING, content="partial")
    tracker = MessagePerformanceTracker()

    await lifecycle.fail(FISH_ID, ANSWER_ID, SOURCE_ID, runtime, tracker)

    assert store.finish_calls == [(ANSWER_ID, FISH_ID, "partial", MessageStatus.FAILED)]
    assert runtime.status is MessageStatus.FAILED
    with pytest.raises(RuntimeError, match="already finished"):
        tracker.finish()


@pytest.mark.asyncio
async def test_interrupt_persists_interrupted_status() -> None:
    store = RecordingTerminalStore()
    lifecycle = AnswerLifecycle(store)
    runtime = runtime_with(MessageStatus.INTERRUPTED, content="visible")
    tracker = MessagePerformanceTracker()

    await lifecycle.interrupt(FISH_ID, ANSWER_ID, SOURCE_ID, runtime, tracker)

    assert store.finish_calls == [
        (ANSWER_ID, FISH_ID, "visible", MessageStatus.INTERRUPTED)
    ]
    assert runtime.status is MessageStatus.INTERRUPTED


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_method", ["complete", "fail"])
async def test_cancelled_persistence_keeps_interruption_owner(
    terminal_method: str,
) -> None:
    class BlockedTerminalStore(RecordingTerminalStore):
        started = asyncio.Event()

        async def finish_answer(self, *args, **kwargs):
            if kwargs["requested_status"] is not MessageStatus.INTERRUPTED:
                self.started.set()
                await asyncio.Event().wait()
            return await super().finish_answer(*args, **kwargs)

    store = BlockedTerminalStore()
    performance = RecordingPerformanceStore()
    lifecycle = AnswerLifecycle(store, performance_store=performance)
    runtime = runtime_with(MessageStatus.GENERATING, content="我得好好捋捋哦，后半句")
    tracker = runtime.performance_tracker
    assert tracker is not None
    task = asyncio.create_task(
        getattr(lifecycle, terminal_method)(
            FISH_ID, ANSWER_ID, SOURCE_ID, runtime, tracker
        )
    )
    await asyncio.wait_for(store.started.wait(), timeout=1)

    # The coordinator owns the interruption and the device-confirmed prefix.
    runtime.status = MessageStatus.INTERRUPTED
    runtime.content = runtime.content[:7]
    runtime.consumed_offset = 7
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runtime.status is MessageStatus.INTERRUPTED
    assert runtime.source_message_id == SOURCE_ID
    assert runtime.performance_tracker is tracker
    await lifecycle.interrupt(FISH_ID, ANSWER_ID, SOURCE_ID, runtime, tracker)
    await lifecycle.close()

    assert store.finish_calls == [
        (ANSWER_ID, FISH_ID, "我得好好捋捋哦", MessageStatus.INTERRUPTED)
    ]
    assert len(performance.writes) == 1
    persist_steps = [
        step for step in performance.writes[0][1] if step.name == "answer.persist"
    ]
    assert [step.error for step in persist_steps] == ["CancelledError", None]


@pytest.mark.asyncio
async def test_terminal_store_failure_uses_path_specific_fallback() -> None:
    performance = RecordingPerformanceStore()
    store = RecordingTerminalStore(failure=RuntimeError("database unavailable"))
    lifecycle = AnswerLifecycle(store, performance_store=performance)
    completing = runtime_with(MessageStatus.GENERATING)
    interrupted = runtime_with(MessageStatus.INTERRUPTED)

    with pytest.raises(RuntimeError, match="database unavailable"):
        await lifecycle.complete(
            FISH_ID,
            ANSWER_ID,
            SOURCE_ID,
            completing,
            MessagePerformanceTracker(),
        )
    with pytest.raises(RuntimeError, match="database unavailable"):
        await lifecycle.interrupt(
            FISH_ID,
            ANSWER_ID,
            UUID(int=4),
            interrupted,
            MessagePerformanceTracker(),
        )
    await lifecycle.close()

    assert completing.status is MessageStatus.FAILED
    assert interrupted.status is MessageStatus.INTERRUPTED
    assert len(performance.writes) == 2
    assert all(
        next(step for step in steps if step.name == "answer.persist").error
        == "RuntimeError"
        for _message_id, steps in performance.writes
    )


@pytest.mark.asyncio
async def test_non_owned_answer_is_a_no_op() -> None:
    store = RecordingTerminalStore()
    lifecycle = AnswerLifecycle(store)
    runtime = runtime_with(MessageStatus.GENERATING, answer_id=UUID(int=99))
    tracker = MessagePerformanceTracker()

    await lifecycle.complete(FISH_ID, ANSWER_ID, SOURCE_ID, runtime, tracker)

    assert store.finish_calls == []
    assert tracker.snapshot() == ()
    assert tracker.finish()[-1].name == "total"


@pytest.mark.asyncio
async def test_close_waits_for_pending_performance_write() -> None:
    performance = RecordingPerformanceStore(blocked=True)
    lifecycle = AnswerLifecycle(
        RecordingTerminalStore(),
        performance_store=performance,
    )
    tracker = MessagePerformanceTracker()
    lifecycle.submit_performance(SOURCE_ID, tracker)
    await asyncio.wait_for(performance.started.wait(), timeout=1)

    close_task = asyncio.create_task(lifecycle.close())
    await asyncio.sleep(0)
    assert close_task.done() is False

    performance.release.set()
    await asyncio.wait_for(close_task, timeout=1)
    assert performance.writes[0][0] == SOURCE_ID
