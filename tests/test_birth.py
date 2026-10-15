from types import SimpleNamespace
from uuid import uuid4

import pytest

from fish_memory.birth import BIRTH_EVENT, announce_birth, birth_event_id


class Events:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def push_host_event(
        self, event, definition, *, target, supersession_key, fallback_text
    ):
        self.calls.append(
            {
                "event": event,
                "definition": definition,
                "target": await target(),
                "key": supersession_key,
            }
        )
        return self.calls[-1]["target"] is not None


class Fishes:
    def __init__(self, assistant) -> None:
        self.assistant = assistant

    async def get_assistant(self):
        return self.assistant


class Sessions:
    def __init__(self) -> None:
        self.created: list = []

    async def get_latest(self, fish_id):
        return None

    async def create(self, fish_id):
        session = SimpleNamespace(id=uuid4())
        self.created.append(session)
        return session


@pytest.mark.asyncio
async def test_birth_is_one_event_per_character_for_the_current_assistant():
    fish = uuid4()
    events, sessions = Events(), Sessions()
    assert await announce_birth(events, Fishes(fish), sessions, fish)
    call = events.calls[0]
    assert call["event"].event_id == birth_event_id(fish) == birth_event_id(fish)
    assert call["event"].type == BIRTH_EVENT.type == call["definition"].type
    assert call["target"] == (fish, sessions.created[0].id)
    assert list(call["event"].data) == ["经过"]

    # Another character is current: nothing is said on the wrong one's behalf.
    assert not await announce_birth(events, Fishes(uuid4()), sessions, fish)
