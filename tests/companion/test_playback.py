import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

from fish_memory.companion.playback import (
    PlaybackArbiter,
    PlaybackArbiterClosed,
)

FIRST_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000101")
SECOND_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000102")
THIRD_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000103")


@pytest.mark.asyncio
async def test_playback_arbiter_grants_waiters_in_fifo_order() -> None:
    arbiter = PlaybackArbiter()
    first = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    second_task = asyncio.create_task(arbiter.acquire("home", SECOND_MESSAGE_ID))
    third_task = asyncio.create_task(arbiter.acquire("agent", THIRD_MESSAGE_ID))
    await asyncio.sleep(0)

    assert not second_task.done()
    assert not third_task.done()
    assert await arbiter.release(first) is True

    second = await asyncio.wait_for(second_task, timeout=0.1)
    assert second.kind == "home"
    assert second.message_id == SECOND_MESSAGE_ID
    assert not third_task.done()

    assert await arbiter.release(second) is True
    third = await asyncio.wait_for(third_task, timeout=0.1)
    assert third.kind == "agent"


@pytest.mark.asyncio
async def test_playback_arbiter_supports_proactive_owner() -> None:
    arbiter = PlaybackArbiter()

    owner = await arbiter.acquire("proactive", FIRST_MESSAGE_ID)

    assert owner.kind == "proactive"
    assert owner.message_id == FIRST_MESSAGE_ID
    assert await arbiter.release(owner)


@pytest.mark.asyncio
async def test_foreground_reservation_blocks_new_grant_without_preempting_owner() -> (
    None
):
    arbiter = PlaybackArbiter()
    current = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    reservation = arbiter.reserve_foreground()
    waiter = asyncio.create_task(arbiter.acquire("agent", SECOND_MESSAGE_ID))
    await asyncio.sleep(0)

    assert arbiter.is_current(current)
    assert not waiter.done()
    assert await arbiter.release(current) is True
    await asyncio.sleep(0)
    assert not waiter.done()

    assert arbiter.release_foreground(reservation) is True
    granted = await asyncio.wait_for(waiter, timeout=0.1)
    assert granted.kind == "agent"


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed_without_breaking_fifo() -> None:
    arbiter = PlaybackArbiter()
    first = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    cancelled = asyncio.create_task(arbiter.acquire("home", SECOND_MESSAGE_ID))
    surviving = asyncio.create_task(arbiter.acquire("wake", THIRD_MESSAGE_ID))
    await asyncio.sleep(0)

    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert await arbiter.release(first) is True

    owner = await asyncio.wait_for(surviving, timeout=0.1)
    assert owner.kind == "wake"


@pytest.mark.asyncio
async def test_cancel_after_grant_reclaims_owner_and_continues_fifo() -> None:
    arbiter = PlaybackArbiter()
    first = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    granted_then_cancelled = asyncio.create_task(
        arbiter.acquire("home", SECOND_MESSAGE_ID)
    )
    surviving = asyncio.create_task(arbiter.acquire("agent", THIRD_MESSAGE_ID))
    await asyncio.sleep(0)

    assert await arbiter.release(first) is True
    assert arbiter.current_owner is not None
    assert arbiter.current_owner.message_id == SECOND_MESSAGE_ID
    granted_then_cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await granted_then_cancelled

    owner = await asyncio.wait_for(surviving, timeout=0.1)
    assert owner.kind == "agent"
    assert owner.message_id == THIRD_MESSAGE_ID


@pytest.mark.asyncio
async def test_stale_release_is_idempotent_and_cannot_release_new_owner() -> None:
    arbiter = PlaybackArbiter()
    first = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    assert await arbiter.release(first) is True
    second = await arbiter.acquire("agent", SECOND_MESSAGE_ID)

    assert await arbiter.release(first) is False
    assert arbiter.is_current(second)
    assert await arbiter.release(second) is True
    assert await arbiter.release(second) is False


@pytest.mark.asyncio
async def test_only_current_owner_can_update_spoken_text() -> None:
    arbiter = PlaybackArbiter()
    first = await arbiter.acquire("answer", FIRST_MESSAGE_ID)

    assert arbiter.set_spoken_text(first, "正在回答") is True
    assert arbiter.current_spoken_text == "正在回答"
    assert await arbiter.release(first) is True
    assert arbiter.current_spoken_text == ""

    second = await arbiter.acquire("agent", SECOND_MESSAGE_ID)
    assert arbiter.set_spoken_text(first, "过期文本") is False
    assert arbiter.current_spoken_text == ""
    assert arbiter.set_spoken_text(second, "任务结果") is True
    assert arbiter.current_spoken_text == "任务结果"


@pytest.mark.asyncio
async def test_close_revokes_current_owner_and_cancels_waiters() -> None:
    arbiter = PlaybackArbiter()
    current = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    waiter = asyncio.create_task(arbiter.acquire("agent", SECOND_MESSAGE_ID))
    await asyncio.sleep(0)

    await arbiter.close()

    assert not arbiter.is_current(current)
    with pytest.raises(PlaybackArbiterClosed):
        await waiter
    with pytest.raises(PlaybackArbiterClosed):
        await arbiter.acquire("wake", None)


class _RecordingTts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.played_offset = 0

    async def start(self, *, text_offset: int = 0) -> None:
        self.calls.append(("start", text_offset))
        self.played_offset = text_offset

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))

    async def finish(self) -> None:
        self.calls.append(("finish", None))

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def resume(self) -> None:
        self.calls.append(("resume", None))

    async def duck(self) -> None:
        self.calls.append(("duck", None))

    async def unduck(self) -> None:
        self.calls.append(("unduck", None))


class _BlockingControlTts(_RecordingTts):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation
        self.started = asyncio.Event()
        self.release_operation = asyncio.Event()

    async def _block(self, name: str, value: object) -> None:
        self.calls.append((f"{name}_started", value))
        if self.operation == name:
            self.started.set()
            await self.release_operation.wait()
        self.calls.append((f"{name}_finished", value))

    async def start(self, *, text_offset: int = 0) -> None:
        await self._block("start", text_offset)

    async def append(self, text: str) -> None:
        await self._block("append", text)

    async def finish(self) -> None:
        await self._block("finish", None)

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))
        self.release_operation.set()


@pytest.mark.asyncio
async def test_owner_bound_tts_rejects_every_stale_live_operation() -> None:
    arbiter = PlaybackArbiter()
    output = _RecordingTts()
    first = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    bound = arbiter.bind_tts(first, output)
    assert arbiter.current_tts is bound

    await bound.start(text_offset=7)
    await bound.append("有效")
    assert bound.played_offset == 7
    assert await arbiter.release(first) is True
    assert arbiter.current_tts is None
    second = await arbiter.acquire("agent", SECOND_MESSAGE_ID)

    await bound.start(text_offset=99)
    await bound.append("过期")
    await bound.finish()
    await bound.stop()
    await bound.interrupt()
    await bound.pause()
    await bound.resume()
    await bound.duck()
    await bound.unduck()

    assert output.calls == [("start", 7), ("append", "有效")]
    assert arbiter.is_current(second)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["start", "append", "finish"])
async def test_release_waits_for_inflight_tts_control_operation(
    operation: str,
) -> None:
    arbiter = PlaybackArbiter()
    output = _BlockingControlTts(operation)
    owner = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    bound = arbiter.bind_tts(owner, output)
    if operation == "start":
        control_task = asyncio.create_task(bound.start())
    elif operation == "append":
        control_task = asyncio.create_task(bound.append("旧回答"))
    else:
        control_task = asyncio.create_task(bound.finish())
    await asyncio.wait_for(output.started.wait(), timeout=0.1)
    release_task = asyncio.create_task(arbiter.release(owner))
    next_owner_task = asyncio.create_task(arbiter.acquire("agent", SECOND_MESSAGE_ID))
    await asyncio.sleep(0)

    assert release_task.done() is False
    assert next_owner_task.done() is False

    output.release_operation.set()
    await control_task
    assert await release_task is True
    next_owner = await asyncio.wait_for(next_owner_task, timeout=0.1)
    assert next_owner.kind == "agent"


@pytest.mark.asyncio
async def test_close_waits_for_inflight_tts_control_operation() -> None:
    arbiter = PlaybackArbiter()
    output = _BlockingControlTts("append")
    owner = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    bound = arbiter.bind_tts(owner, output)
    control_task = asyncio.create_task(bound.append("旧回答"))
    await asyncio.wait_for(output.started.wait(), timeout=0.1)
    close_task = asyncio.create_task(arbiter.close())
    await asyncio.sleep(0)

    assert close_task.done() is False
    output.release_operation.set()
    await control_task
    await close_task
    assert arbiter.is_current(owner) is False


@pytest.mark.asyncio
async def test_interrupt_preempts_blocked_finish_before_release() -> None:
    arbiter = PlaybackArbiter()
    output = _BlockingControlTts("finish")
    owner = await arbiter.acquire("agent", FIRST_MESSAGE_ID)
    bound = arbiter.bind_tts(owner, output)
    finish_task = asyncio.create_task(bound.finish())
    await asyncio.wait_for(output.started.wait(), timeout=0.1)

    await asyncio.wait_for(bound.interrupt(), timeout=0.1)
    await asyncio.wait_for(finish_task, timeout=0.1)

    assert ("interrupt", None) in output.calls
    assert await arbiter.release(owner) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["release", "close"])
async def test_terminal_waiter_does_not_block_finish_interrupt(
    terminal: str,
) -> None:
    arbiter = PlaybackArbiter()
    output = _BlockingControlTts("finish")
    owner = await arbiter.acquire("agent", FIRST_MESSAGE_ID)
    bound = arbiter.bind_tts(owner, output)
    finish_task = asyncio.create_task(bound.finish())
    await asyncio.wait_for(output.started.wait(), timeout=0.1)
    terminal_task = asyncio.create_task(
        arbiter.release(owner) if terminal == "release" else arbiter.close()
    )
    await asyncio.sleep(0)
    interrupt_task = asyncio.create_task(bound.interrupt())
    done, _ = await asyncio.wait({interrupt_task}, timeout=0.05)
    if interrupt_task not in done:
        output.release_operation.set()
    await asyncio.gather(finish_task, interrupt_task, terminal_task)

    assert interrupt_task in done
    assert ("interrupt", None) in output.calls


class _RecordingPcm:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.played_offset = 0

    async def start(self) -> None:
        self.calls.append(("start", None))

    def reset(self, text_offset: int = 0) -> None:
        self.calls.append(("reset", text_offset))
        self.played_offset = text_offset

    async def append_owned(
        self,
        owner: object,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        self.calls.append(("append", (owner, pcm, start_offset, end_offset)))

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def resume(self) -> None:
        self.calls.append(("resume", None))

    async def duck(self) -> None:
        self.calls.append(("duck", None))

    async def unduck(self) -> None:
        self.calls.append(("unduck", None))

    async def wait_until_drained(self) -> None:
        self.calls.append(("drain", None))


class _BlockingDrainPcm(_RecordingPcm):
    def __init__(self) -> None:
        super().__init__()
        self.drain_started = asyncio.Event()
        self.release_drain = asyncio.Event()

    async def wait_until_drained(self) -> None:
        self.calls.append(("drain_started", None))
        self.drain_started.set()
        await self.release_drain.wait()
        self.calls.append(("drain_finished", None))


@pytest.mark.asyncio
async def test_owner_bound_pcm_pause_preempts_passive_drain_wait() -> None:
    arbiter = PlaybackArbiter()
    sink = _BlockingDrainPcm()
    owner = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    bound = arbiter.bind_pcm(owner, sink)

    drain_task = asyncio.create_task(bound.wait_until_drained())
    await asyncio.wait_for(sink.drain_started.wait(), timeout=0.1)

    await asyncio.wait_for(bound.pause(), timeout=0.1)

    assert drain_task.done() is False
    assert sink.calls == [("drain_started", None), ("pause", None)]
    sink.release_drain.set()
    await drain_task


@pytest.mark.asyncio
async def test_owner_bound_pcm_carries_one_token_and_stale_reset_is_ignored() -> None:
    arbiter = PlaybackArbiter()
    sink = _RecordingPcm()
    owner = await arbiter.acquire("wake", None)
    bound = arbiter.bind_pcm(owner, sink)

    await bound.start()
    bound.reset(5)
    await bound.append(b"\x01\x02", start_offset=5, end_offset=6)
    assert sink.calls[-1] == (
        "append",
        (owner, b"\x01\x02", 5, 6),
    )
    assert await arbiter.release(owner) is True
    await arbiter.acquire("home", THIRD_MESSAGE_ID)

    bound.reset(0)
    await bound.append(b"\x03\x04", start_offset=6, end_offset=7)
    await bound.stop()

    assert sink.calls == [
        ("start", None),
        ("reset", 5),
        ("append", (owner, b"\x01\x02", 5, 6)),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    ["start", "stop", "pause", "resume", "duck", "unduck"],
)
async def test_nested_tts_pcm_controls_do_not_reacquire_the_tts_lock(
    operation: str,
) -> None:
    arbiter = PlaybackArbiter()
    sink = _RecordingPcm()
    owner = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    pcm = arbiter.bind_pcm(owner, sink)
    output = SimpleNamespace(
        start=pcm.start,
        stop=pcm.stop,
        pause=pcm.pause,
        resume=pcm.resume,
        duck=pcm.duck,
        unduck=pcm.unduck,
    )
    tts = arbiter.bind_tts(owner, output)

    await asyncio.wait_for(getattr(tts, operation)(), timeout=0.1)

    assert sink.calls == [(operation, None)]
    assert await arbiter.release(owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["release", "close"])
async def test_owner_retirement_waits_for_inflight_pcm_control(
    terminal: str,
) -> None:
    started = asyncio.Event()
    release_control = asyncio.Event()

    class BlockingPausePcm(_RecordingPcm):
        async def pause(self) -> None:
            started.set()
            await release_control.wait()
            await super().pause()

    arbiter = PlaybackArbiter()
    sink = BlockingPausePcm()
    owner = await arbiter.acquire("answer", FIRST_MESSAGE_ID)
    pcm = arbiter.bind_pcm(owner, sink)
    pause = asyncio.create_task(pcm.pause())
    await asyncio.wait_for(started.wait(), timeout=0.1)
    retire = asyncio.create_task(
        arbiter.release(owner) if terminal == "release" else arbiter.close()
    )
    await asyncio.sleep(0)
    assert not retire.done()

    release_control.set()
    await asyncio.wait_for(asyncio.gather(pause, retire), timeout=0.1)
    assert not arbiter.is_current(owner)
    assert sink.calls == [("pause", None)]
