from datetime import UTC, datetime
from uuid import uuid4

import pytest

from fish_memory.answer.lane_stream import AnswerLaneDelta
from fish_memory.proactive.expression import ExpressionStream, GeneratedExpressionInput
from fish_memory.proactive.models import ExpressionIntent


@pytest.mark.asyncio
async def test_inline_source_keeps_tail_completed_while_first_phrase_is_delivered():
    source = GeneratedExpressionInput()
    await source.feed(AnswerLaneDelta(answer="第一句。"))
    events = source.events()
    try:
        assert (await anext(events)).answer == "第一句。"
        # The consumer is busy delivering that prefix; the model completes now.
        await source.feed(AnswerLaneDelta(answer="第二句。", answer_complete=True))
        tail = await anext(events)
        assert tail.answer == "第二句。" and tail.answer_complete
        with pytest.raises(StopAsyncIteration):
            await anext(events)
    finally:
        await events.aclose()


@pytest.mark.asyncio
async def test_native_stream_drains_tail_before_end_after_consumer_yields():
    stream = ExpressionStream(
        ExpressionIntent(
            id=uuid4(),
            fish_id=uuid4(),
            session_id=uuid4(),
            generation=1,
            opportunity_id="test",
            focus_id="f1",
            focus_revision=1,
            kind="speak",
            purpose="分享线索",
            created_at=datetime.now(UTC),
        )
    )
    stream.append("第一句。")
    events = stream.events(0)
    try:
        assert await anext(events) == {"type": "text", "offset": 0, "text": "第一句。"}
        # The native consumer was processing the first event as generation ended.
        stream.append("第二句。")
        stream.end()
        assert await anext(events) == {"type": "text", "offset": 4, "text": "第二句。"}
        assert await anext(events) == {"type": "end", "offset": 8, "failed": False}
    finally:
        await events.aclose()
