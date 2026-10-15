import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

import pytest

import fish_memory.background.supervisor as supervisor_module
from fish_memory.background.supervisor import BackgroundTaskSupervisor


class SequenceProcessor:
    def __init__(self, outcomes: list[bool | BaseException]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def process_once(self) -> bool:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class BlockingProcessor:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def process_once(self) -> bool:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()
        return False


class ProcessorWithOwnedResources(BlockingProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.resources_closed = asyncio.Event()

    async def close(self) -> None:
        self.resources_closed.set()


@pytest.mark.asyncio
async def test_start_creates_exactly_one_loop_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = BlockingProcessor()
    supervisor = BackgroundTaskSupervisor(
        processor,
        name="test-processor",
        poll_interval_seconds=0.25,
    )
    created_tasks: list[asyncio.Task[None]] = []
    create_task = asyncio.create_task

    def track_task(
        coroutine: Coroutine[Any, Any, None],
        *,
        name: str | None = None,
    ) -> asyncio.Task[None]:
        task = create_task(coroutine, name=name)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(supervisor_module.asyncio, "create_task", track_task)

    supervisor.start()
    supervisor.start()

    assert len(created_tasks) == 1
    assert created_tasks[0].get_name() == "fish-memory-test-processor"
    await processor.entered.wait()
    await supervisor.close()
    assert created_tasks[0].done()


@pytest.mark.asyncio
async def test_no_work_sleeps_for_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SequenceProcessor([False])
    supervisor = BackgroundTaskSupervisor(
        processor,
        name="test-processor",
        poll_interval_seconds=0.25,
    )
    delays: list[float] = []

    async def stop_after_sleep(delay: float) -> None:
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(supervisor_module.asyncio, "sleep", stop_after_sleep)

    with pytest.raises(asyncio.CancelledError):
        await supervisor._run()

    assert processor.calls == 1
    assert delays == [0.25]


@pytest.mark.asyncio
async def test_work_yields_without_poll_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SequenceProcessor([True])
    supervisor = BackgroundTaskSupervisor(
        processor,
        name="test-processor",
        poll_interval_seconds=0.25,
    )
    delays: list[float] = []

    async def stop_after_sleep(delay: float) -> None:
        delays.append(delay)
        raise asyncio.CancelledError

    monkeypatch.setattr(supervisor_module.asyncio, "sleep", stop_after_sleep)

    with pytest.raises(asyncio.CancelledError):
        await supervisor._run()

    assert processor.calls == 1
    assert delays == [0]


@pytest.mark.asyncio
async def test_processor_exception_is_logged_and_loop_continues(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    processor = SequenceProcessor(
        [RuntimeError("processor failed"), asyncio.CancelledError()]
    )
    supervisor = BackgroundTaskSupervisor(
        processor,
        name="test-processor",
        poll_interval_seconds=0.25,
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(supervisor_module.asyncio, "sleep", record_sleep)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(asyncio.CancelledError),
    ):
        await supervisor._run()

    assert processor.calls == 2
    assert delays == [0.25]
    assert "background test-processor failed" in caplog.text


@pytest.mark.asyncio
async def test_processor_cancellation_propagates_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SequenceProcessor([asyncio.CancelledError()])
    supervisor = BackgroundTaskSupervisor(
        processor,
        name="test-processor",
        poll_interval_seconds=0.25,
    )

    async def unexpected_sleep(delay: float) -> None:
        pytest.fail(f"unexpected sleep({delay})")

    monkeypatch.setattr(supervisor_module.asyncio, "sleep", unexpected_sleep)

    with pytest.raises(asyncio.CancelledError):
        await supervisor._run()


@pytest.mark.asyncio
async def test_close_cancels_and_awaits_loop_and_is_idempotent() -> None:
    processor = BlockingProcessor()
    supervisor = BackgroundTaskSupervisor(
        processor,
        name="test-processor",
        poll_interval_seconds=0.25,
    )

    await supervisor.close()
    supervisor.start()
    await processor.entered.wait()

    await supervisor.close()
    await supervisor.close()

    assert processor.cancelled.is_set()


@pytest.mark.asyncio
async def test_supervisor_stops_loop_before_processor_owned_resources_close() -> None:
    processor = ProcessorWithOwnedResources()
    supervisor = BackgroundTaskSupervisor(
        processor,
        name="agent-run-processor",
        poll_interval_seconds=0.25,
    )
    supervisor.start()
    await processor.entered.wait()

    await supervisor.close()

    assert processor.cancelled.is_set()
    assert not processor.resources_closed.is_set()
    await processor.close()
    assert processor.resources_closed.is_set()
