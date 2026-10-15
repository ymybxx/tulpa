import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from fish_memory.companion.session import CompanionSession
from tests.companion.test_session import (
    FISH_ID,
    SESSION_ID,
    AgentFakeApi,
    CallbackTts,
    proactive_message,
    wait_until,
)


class StreamingApi(AgentFakeApi):
    def __init__(self):
        super().__init__()
        self.more = asyncio.Event()
        self.tail = "我想再继续看看他接下来会怎么解释这件事情。"
        self.stream_closed = False
        self.hold_end = asyncio.Event()

    async def iter_proactive_expression(
        self, fish_id, message_id, *, session_id, after_offset
    ):
        assert fish_id == FISH_ID and session_id == SESSION_ID
        try:
            await self.more.wait()
            yield self.tail
            await self.hold_end.wait()
        finally:
            self.stream_closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_expiry", [False, True])
async def test_late_first_phrase_and_tail_share_one_tts_without_expiry(legacy_expiry):
    api, tts = StreamingApi(), CallbackTts()
    clock = SimpleNamespace(now=datetime.now(UTC))
    session = CompanionSession(api, tts, now=lambda: clock.now)
    tts.on_first_frame = lambda: session.on_playback_started(
        session._playback.current_owner
    )
    message = replace(
        proactive_message(
            1,
            content="这个解释和刚才那句话对不上。",
        ),
        streaming=True,
        expires_at=clock.now + timedelta(seconds=2) if legacy_expiry else None,
    )
    try:
        await session.set_binding(FISH_ID)
        clock.now += timedelta(seconds=5)
        await api.agent_events.put(message)
        await wait_until(lambda: any(op == "append" for op, _ in tts.calls))
        assert not api.more.is_set()
        clock.now += timedelta(seconds=5)
        assert not api.agent_consumption_calls
        assert not session._proactive_expiry_tasks
        api.more.set()
        api.hold_end.set()
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert sum(op == "start" for op, _ in tts.calls) == 1
        assert (
            "".join(text for op, text in tts.calls if op == "append")
            == message.content + api.tail
        )
        assert api.agent_consumption_calls == [
            (message.message_id, len(message.content + api.tail), False)
        ]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_revoke_after_appending_stops_live_owner_and_preserves_played_prefix():
    api, tts = StreamingApi(), CallbackTts()
    session = CompanionSession(api, tts)
    message = replace(
        proactive_message(2, content="这个解释和刚才那句话对不上。"), streaming=True
    )
    try:
        await session.set_binding(FISH_ID)
        await api.agent_events.put(message)
        api.more.set()
        await wait_until(
            lambda: any(text == api.tail for op, text in tts.calls if op == "append")
        )
        tts.played_offset = 5
        await session._handle_answer_interrupt_request(FISH_ID, message.message_id)
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert api.stream_closed
        assert api.agent_consumption_calls == [(message.message_id, 5, True)]
        assert sum(op == "start" for op, _ in tts.calls) == 1
    finally:
        await session.close()
