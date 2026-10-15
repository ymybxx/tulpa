import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.sse import AnswerEvent, format_sse_event, iter_answer_events
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.pet.types import PetEmotion

FIRST_ID = UUID("00000000-0000-0000-0000-000000000001")
SECOND_ID = UUID("00000000-0000-0000-0000-000000000002")
USER_ID = UUID("00000000-0000-0000-0000-000000000003")
FISH_ID = UUID("00000000-0000-0000-0000-000000000004")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000005")


def user_message(*, sequence: int = 9, content: str = "语音提问") -> Message:
    return Message(
        id=USER_ID,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=sequence,
        role=Role.USER,
        content=content,
        status=MessageStatus.INITIALIZED,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
    )


async def set_runtime(
    runtime: FishRuntime,
    *,
    message_id: UUID,
    sequence: int,
    content: str,
    status: MessageStatus,
    flush_offset: int = 0,
) -> None:
    async with runtime.condition:
        runtime.answer_message_id = message_id
        runtime.answer_sequence = sequence
        runtime.content = content
        runtime.flush_offset = flush_offset
        runtime.status = status
        runtime.condition.notify_all()


@pytest.mark.asyncio
async def test_subscribers_keep_independent_snapshot_and_delta_cursors() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="我",
        status=MessageStatus.GENERATING,
    )
    first = iter_answer_events(runtime)

    first_snapshot = await anext(first)
    assert first_snapshot.event == "answer.snapshot"
    assert first_snapshot.data["content"] == "我"
    assert first_snapshot.data["startOffset"] == 0
    assert first_snapshot.data["endOffset"] == 1

    async with runtime.condition:
        runtime.content += "记得🐟"
        runtime.condition.notify_all()

    first_delta = await anext(first)
    second = iter_answer_events(runtime)
    second_snapshot = await anext(second)

    assert first_delta.event == "answer.delta"
    assert first_delta.data["content"] == "记得🐟"
    assert first_delta.data["startOffset"] == 1
    assert first_delta.data["endOffset"] == 4
    assert second_snapshot.event == "answer.snapshot"
    assert second_snapshot.data["content"] == "我记得🐟"
    assert second_snapshot.data["startOffset"] == 0
    assert second_snapshot.data["endOffset"] == 4
    await first.aclose()
    await second.aclose()


@pytest.mark.asyncio
async def test_flush_event_follows_text_delta_and_is_emitted_once() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="",
        status=MessageStatus.GENERATING,
    )
    events = iter_answer_events(runtime)
    assert (await anext(events)).event == "answer.snapshot"

    async with runtime.condition:
        runtime.content = "我去看看最新情况。"
        runtime.flush_offset = len(runtime.content)
        runtime.condition.notify_all()

    delta = await anext(events)
    segment = await anext(events)
    assert delta.event == "answer.delta"
    assert segment.event == "answer.segment"
    assert segment.data == {
        "messageId": str(FIRST_ID),
        "flushOffset": len("我去看看最新情况。"),
        "reason": "lane_answer_complete",
    }

    pending = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    assert not pending.done()
    async with runtime.condition:
        runtime.status = MessageStatus.COMPLETED
        runtime.condition.notify_all()
    assert (await pending).event == "answer.completed"


@pytest.mark.asyncio
async def test_reconnected_snapshot_carries_latest_flush_offset() -> None:
    content = "完整前置回答。后续仍在生成"
    flush_offset = len("完整前置回答。")
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content=content,
        status=MessageStatus.GENERATING,
        flush_offset=flush_offset,
    )

    events = iter_answer_events(runtime)
    snapshot = await anext(events)

    assert snapshot.event == "answer.snapshot"
    assert snapshot.data["content"] == content
    assert snapshot.data["flushOffset"] == flush_offset
    await events.aclose()


@pytest.mark.asyncio
async def test_user_message_is_emitted_before_its_answer_snapshot() -> None:
    runtime = FishRuntime(latest_user_message=user_message())
    events = iter_answer_events(runtime)

    user = await anext(events)
    assert user.event == "answer.user"
    assert user.data == {
        "userMessageId": str(USER_ID),
        "sequence": 9,
        "role": "user",
        "content": "语音提问",
        "status": "initialized",
    }

    pending_snapshot = asyncio.create_task(anext(events))
    await asyncio.sleep(0)
    assert not pending_snapshot.done()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="回答",
        status=MessageStatus.GENERATING,
    )
    snapshot = await pending_snapshot

    assert snapshot.event == "answer.snapshot"
    assert snapshot.data["messageId"] == str(FIRST_ID)
    await events.aclose()


@pytest.mark.asyncio
async def test_user_message_at_or_before_stream_cursor_is_not_replayed() -> None:
    runtime = FishRuntime(latest_user_message=user_message(sequence=9))
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="回答",
        status=MessageStatus.GENERATING,
    )
    events = iter_answer_events(runtime, after_sequence=9)

    event = await anext(events)

    assert event.event == "answer.snapshot"
    await events.aclose()


@pytest.mark.asyncio
async def test_interrupted_is_emitted_once_then_stream_follows_new_answer() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="旧回答",
        status=MessageStatus.GENERATING,
    )
    events = iter_answer_events(runtime)
    await anext(events)

    async with runtime.condition:
        runtime.status = MessageStatus.INTERRUPTED
        runtime.condition.notify_all()
    interrupted = await anext(events)
    assert interrupted.event == "answer.interrupted"
    assert interrupted.data["messageId"] == str(FIRST_ID)

    pending_event = asyncio.ensure_future(anext(events))
    done, _ = await asyncio.wait({pending_event}, timeout=0.01)
    assert not done
    await set_runtime(
        runtime,
        message_id=SECOND_ID,
        sequence=11,
        content="新",
        status=MessageStatus.GENERATING,
    )

    snapshot = await pending_event
    assert snapshot.event == "answer.snapshot"
    assert snapshot.data["messageId"] == str(SECOND_ID)
    assert snapshot.data["content"] == "新"
    await events.aclose()


@pytest.mark.asyncio
async def test_interrupted_event_does_not_leak_unconsumed_content() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="一二三四五六",
        status=MessageStatus.GENERATING,
    )
    events = iter_answer_events(runtime)
    assert (await anext(events)).data["content"] == "一二三四五六"

    async with runtime.condition:
        runtime.content = runtime.content[:5]
        runtime.consumed_offset = 5
        runtime.status = MessageStatus.INTERRUPTED
        runtime.condition.notify_all()

    interrupted = await anext(events)
    assert interrupted.event == "answer.interrupted"
    assert interrupted.data["content"] == "一二三四五"
    await events.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [MessageStatus.COMPLETED, MessageStatus.FAILED])
async def test_completed_or_failed_terminal_event_closes_stream(
    status: MessageStatus,
) -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="final",
        status=status,
    )
    events = iter_answer_events(runtime)

    assert (await anext(events)).event == "answer.snapshot"
    terminal = await anext(events)
    assert terminal.event == f"answer.{status.value}"
    with pytest.raises(StopAsyncIteration):
        await anext(events)


@pytest.mark.asyncio
async def test_completed_event_exposes_end_of_conversation_flag() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="结束",
        status=MessageStatus.COMPLETED,
    )
    runtime.end_conversation = True
    events = iter_answer_events(runtime)

    await anext(events)
    terminal = await anext(events)

    assert terminal.event == "answer.completed"
    assert terminal.data["end"] is True


@pytest.mark.asyncio
async def test_answer_stream_can_wait_for_an_answer_after_a_user_sequence() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="旧回答",
        status=MessageStatus.COMPLETED,
    )
    events = iter_answer_events(runtime, after_sequence=10)
    pending = asyncio.ensure_future(anext(events))
    done, _ = await asyncio.wait({pending}, timeout=0.01)
    assert not done

    await set_runtime(
        runtime,
        message_id=SECOND_ID,
        sequence=11,
        content="新回答",
        status=MessageStatus.GENERATING,
    )

    snapshot = await pending
    assert snapshot.data["messageId"] == str(SECOND_ID)
    await events.aclose()


def test_formats_utf8_sse_without_ascii_escaping() -> None:
    event = AnswerEvent(
        event="answer.delta",
        event_id=f"{FIRST_ID}:2",
        data={"messageId": str(FIRST_ID), "content": "鱼🐟"},
    )

    encoded = format_sse_event(event)

    assert encoded.startswith(f"id: {FIRST_ID}:2\nevent: answer.delta\ndata: ")
    assert "鱼🐟" in encoded
    payload = encoded.split("data: ", maxsplit=1)[1].strip()
    assert json.loads(payload)["content"] == "鱼🐟"
    assert encoded.endswith("\n\n")


@pytest.mark.asyncio
async def test_metadata_event_precedes_content_delta() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="",
        status=MessageStatus.GENERATING,
    )
    events = iter_answer_events(runtime)
    snapshot = await anext(events)
    assert snapshot.data["emotion"] == "neutral"

    async with runtime.condition:
        runtime.pet_emotion = PetEmotion.HAPPY
        runtime.pet_metadata_revision += 1
        runtime.content = "你好"
        runtime.condition.notify_all()

    metadata = await anext(events)
    delta = await anext(events)

    assert metadata.event == "answer.metadata"
    assert metadata.data["emotion"] == "happy"
    assert "gesture" not in metadata.data
    assert delta.event == "answer.delta"
    assert delta.data["content"] == "你好"
    await events.aclose()


@pytest.mark.asyncio
async def test_failed_answer_says_which_service_let_it_down() -> None:
    runtime = FishRuntime()
    await set_runtime(
        runtime,
        message_id=FIRST_ID,
        sequence=10,
        content="",
        status=MessageStatus.GENERATING,
    )
    events = iter_answer_events(runtime)
    assert (await anext(events)).event == "answer.snapshot"
    async with runtime.condition:
        runtime.failure = "jev:timeout"
        runtime.status = MessageStatus.FAILED
        runtime.condition.notify_all()
    failed = await anext(events)
    assert failed.event == "answer.failed"
    assert failed.data["failure"] == "jev:timeout"
