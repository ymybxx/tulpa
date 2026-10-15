"""The coordinator only forwards app events to Jev; it never judges on its own."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from fish_memory.environment_frames import EnvironmentPerceptionPolicy
from fish_memory.proactive.coordinator import ProactiveCoordinator
from fish_memory.proactive.expression import ExpressionRegistry


class FakeJev:
    def __init__(self, enabled=True):
        self.service = SimpleNamespace(enabled=enabled)
        self.binding = None
        self.busy = False
        self.suspended = False
        self.foreground_busy = False
        self.calls = []

    def subscribe(self, token, fish, session):
        self.binding = (fish, session)
        self.calls.append("subscribe")

    def unsubscribe(self, token):
        self.calls.append("unsubscribe")

    def reset(self):
        self.calls.append("reset")

    def interrupt(self):
        self.calls.append("interrupt")

    def consumed(self, message_id, offset, interrupted):
        self.calls.append(("consumed", offset, interrupted))
        return True

    async def admit_window(self, document):
        self.calls.append("admit")
        return False

    async def close(self):
        self.calls.append("close")


def coordinator(jev):
    value = ProactiveCoordinator(
        streams=ExpressionRegistry(),
        policy=lambda fish: EnvironmentPerceptionPolicy(screen_enabled=True),
    )
    value.experimental = jev
    return value


@pytest.mark.asyncio
async def test_session_events_are_forwarded_to_jev():
    jev = FakeJev()
    c = coordinator(jev)
    fish, session = uuid4(), uuid4()
    token = c.bind(fish, session)
    c.foreground(fish, session, True)
    c.conversation_started(fish, session)
    c.consumed(uuid4(), 3, False)
    assert jev.foreground_busy is True
    assert jev.calls[:4] == ["reset", "subscribe", "interrupt", "interrupt"]
    assert ("consumed", 3, False) in jev.calls
    assert await c.admit_window_summary(fish, session, object()) is False
    c.unbind(token)
    assert jev.calls[-2:] == ["unsubscribe", "reset"]
    assert c.status()["bound"] is False


@pytest.mark.asyncio
async def test_external_owner_suspends_jev_until_every_owner_resumes():
    jev = FakeJev()
    c = coordinator(jev)
    fish, first, second = uuid4(), uuid4(), uuid4()
    c.suspend_external(fish, first)
    c.suspend_external(fish, second)
    c.resume_external(fish, first)
    assert jev.suspended is True
    c.resume_external(fish, second)
    assert jev.suspended is False


@pytest.mark.asyncio
async def test_speaking_up_off_still_stops_a_report_and_judges_summaries():
    jev = FakeJev(enabled=False)
    c = coordinator(jev)
    fish, session = uuid4(), uuid4()
    c.bind(fish, session)
    c.conversation_started(fish, session)
    # TA may be reporting what it looked up; the user speaking stops that.
    assert "interrupt" in jev.calls
    # What is worth remembering belongs to 内心, not to 主动陪伴.
    assert await c.admit_window_summary(fish, session, object()) is False
    assert "admit" in jev.calls
    assert c.responding_for(fish) is False
    await c.close()
    assert "close" in jev.calls
