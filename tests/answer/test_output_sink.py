import asyncio
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from fish_memory.answer.output_sink import (
    DraftOutputLimitError,
    PromotableAnswerSink,
    RuntimeAnswerSink,
)
from fish_memory.answer.runtime import FishRuntime
from fish_memory.models import MessageStatus
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion

ANSWER_ID = UUID(int=2)


class RecordingAnswerSink:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def set_emotion(self, emotion: PetEmotion) -> None:
        del emotion
        self.events.append("emotion")

    async def append(self, text: str, *, continuation: bool = False) -> None:
        del text, continuation
        self.events.append("append")

    async def flush(self) -> None:
        self.events.append("flush")

    async def flush_through(self, end_offset: int) -> None:
        del end_offset
        self.events.append("flush")

    async def set_end(self, end: bool) -> None:
        del end
        self.events.append("end")


def generating_runtime() -> FishRuntime:
    runtime = FishRuntime()
    runtime.answer_message_id = ANSWER_ID
    runtime.status = MessageStatus.GENERATING
    return runtime


@pytest.mark.asyncio
async def test_empty_promotable_sink_attach_emits_no_public_metadata_events() -> None:
    """Replaying unset defaults would create a false public pet update."""
    sink = PromotableAnswerSink()
    target = RecordingAnswerSink()

    await sink.attach(target)

    assert target.events == []


@pytest.mark.asyncio
async def test_promotable_sink_attaches_flushed_prefix_without_flushing_suffix() -> (
    None
):
    sink = PromotableAnswerSink(max_chars=100)
    await sink.set_emotion(PetEmotion.HAPPY)
    await sink.append("已确认")
    await sink.flush()
    await sink.append("未确认")
    await sink.set_end(True)
    snapshot = sink.snapshot()

    assert snapshot.content == "已确认未确认"
    assert snapshot.flush_offset == len("已确认")
    assert snapshot.emotion is PetEmotion.HAPPY
    assert snapshot.end is True
    with pytest.raises(FrozenInstanceError):
        snapshot.content = "changed"  # type: ignore[misc]

    runtime = generating_runtime()
    target = RuntimeAnswerSink(runtime, ANSWER_ID, MessagePerformanceTracker())
    await sink.attach(target)

    assert runtime.content == "已确认未确认"
    assert runtime.flush_offset == len("已确认")
    assert runtime.pet_emotion is PetEmotion.HAPPY
    assert runtime.end_conversation is True


@pytest.mark.asyncio
async def test_promotable_sink_replays_once_then_writes_live() -> None:
    sink = PromotableAnswerSink(max_chars=100)
    await sink.set_emotion(PetEmotion.HAPPY)
    await sink.append("前缀")
    await sink.flush()
    runtime = generating_runtime()

    target = RuntimeAnswerSink(runtime, ANSWER_ID, MessagePerformanceTracker())
    await sink.attach(target)
    await sink.append("后缀")
    await sink.attach(target)

    assert runtime.content == "前缀后缀"
    assert runtime.flush_offset == len("前缀")
    assert runtime.pet_emotion is PetEmotion.HAPPY


@pytest.mark.asyncio
async def test_promotable_sink_rejects_output_beyond_its_limit() -> None:
    sink = PromotableAnswerSink(max_chars=3)

    await sink.append("abc")

    with pytest.raises(DraftOutputLimitError):
        await sink.append("d")

    assert sink.snapshot().content == "abc"


@pytest.mark.asyncio
async def test_committed_continuation_can_exceed_the_private_output_limit() -> None:
    sink = PromotableAnswerSink(max_chars=3)
    await sink.append("abc")
    runtime = generating_runtime()
    await sink.attach(
        RuntimeAnswerSink(runtime, ANSWER_ID, MessagePerformanceTracker())
    )

    await sink.append("defgh", continuation=True)

    assert sink.snapshot().content == runtime.content == "abcdefgh"


@pytest.mark.asyncio
async def test_continuation_timing_records_successful_public_write_after_lock_wait():
    now_ns = 1_000_000
    tracker = MessagePerformanceTracker(clock=lambda: now_ns)
    runtime = generating_runtime()
    target = RuntimeAnswerSink(runtime, ANSWER_ID, tracker)
    await runtime.condition.acquire()
    publishing = asyncio.create_task(target.append("补充", continuation=True))
    await asyncio.sleep(0)
    assert not publishing.done()
    assert tracker.snapshot() == ()
    now_ns = 5_000_000
    runtime.condition.release()
    await publishing
    await target.append("更多补充", continuation=True)

    steps = {step.name: step for step in tracker.snapshot()}
    assert runtime.content == "补充更多补充"
    assert len(tracker.snapshot()) == 2
    assert steps["answer.first_content"].duration_us == 4_000
    assert steps["answer.first_append"].duration_us == 4_000


@pytest.mark.asyncio
async def test_revocation_stops_remaining_attach_replay_after_await():
    entered, resume = asyncio.Event(), asyncio.Event()

    class GatedSink(RecordingAnswerSink):
        async def set_emotion(self, emotion: PetEmotion) -> None:
            await super().set_emotion(emotion)
            entered.set()
            await resume.wait()

    sink, target = PromotableAnswerSink(), GatedSink()
    await sink.set_emotion(PetEmotion.HAPPY)
    await sink.append("预生成内容", continuation=True)
    await sink.flush()
    await sink.set_end(True)
    attaching = asyncio.create_task(sink.attach(target))
    await entered.wait()
    sink.revoke()
    resume.set()
    await attaching

    assert target.events == ["emotion"]
