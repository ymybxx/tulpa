import asyncio
import json
from uuid import UUID

import pytest

from fish_memory.pet.protocol import unpack_pcm_frame
from fish_memory.pet.publisher import PetEventPublisher

GENERATION = UUID("00000000-0000-0000-0000-000000000123")


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[str | bytes] = []

    async def send(self, value: str | bytes) -> None:
        self.sent.append(value)

    async def close(self) -> None:
        return None


async def test_publisher_sends_control_and_positioned_pcm_in_order() -> None:
    socket = FakeSocket()

    async def connect(_url: str) -> FakeSocket:
        return socket

    publisher = PetEventPublisher(
        "http://127.0.0.1:8000",
        api_key="secret",
        websocket_connector=connect,
    )
    await publisher.start()
    publisher.audio_start(GENERATION, 16_000)
    publisher.pcm(b"\x01\x00\x02\x00")
    publisher.pcm(b"\x03\x00")
    publisher.audio_stop(GENERATION)

    for _ in range(100):
        if len(socket.sent) == 3:
            break
        await asyncio.sleep(0.001)
    await publisher.close()

    assert isinstance(socket.sent[0], str)
    assert isinstance(socket.sent[1], bytes)
    generation, position, pcm = unpack_pcm_frame(socket.sent[1])
    assert (generation, position, pcm) == (
        GENERATION,
        0,
        b"\x01\x00\x02\x00\x03\x00",
    )
    assert isinstance(socket.sent[2], str)


async def test_publisher_dispatches_plugin_playback_bindings_without_audio():
    class CommandSocket(FakeSocket):
        def __init__(self):
            super().__init__()
            self.commands = asyncio.Queue()

        async def recv(self):
            return await self.commands.get()

    socket = CommandSocket()
    seen = []
    received = asyncio.Event()

    async def connect(url):
        return socket

    async def handler(bindings):
        seen.append(bindings)
        received.set()

    publisher = PetEventPublisher(
        "http://127.0.0.1:8000", api_key="test", websocket_connector=connect
    )
    publisher.set_plugin_playback_handler(handler)
    await publisher.start()
    try:
        await socket.commands.put(
            json.dumps(
                {
                    "type": "pet.plugin.sessions",
                    "payload": {
                        "sessions": [
                            {"fishId": str(GENERATION), "sessionId": str(GENERATION)}
                        ]
                    },
                }
            )
        )
        await asyncio.wait_for(received.wait(), timeout=1)
        assert seen == [frozenset({(GENERATION, GENERATION)})]
        assert socket.sent == []
    finally:
        await publisher.close()


async def test_publisher_starts_listening_on_the_listen_command():
    class CommandSocket(FakeSocket):
        def __init__(self):
            super().__init__()
            self.commands = asyncio.Queue()

        async def recv(self):
            return await self.commands.get()

    socket = CommandSocket()
    heard = asyncio.Event()

    async def connect(url):
        return socket

    async def listen() -> bool:
        heard.set()
        return True

    publisher = PetEventPublisher(
        "http://127.0.0.1:8000", api_key="test", websocket_connector=connect
    )
    stopped = asyncio.Event()

    async def stop() -> bool:
        stopped.set()
        return True

    publisher.set_listen_handler(listen)
    publisher.set_stop_listening_handler(stop)
    await publisher.start()
    try:
        await socket.commands.put(json.dumps({"type": "pet.listen", "payload": {}}))
        await asyncio.wait_for(heard.wait(), timeout=1)
        await socket.commands.put(
            json.dumps({"type": "pet.listen.stop", "payload": {}})
        )
        await asyncio.wait_for(stopped.wait(), timeout=1)
        assert socket.sent == []
    finally:
        await publisher.close()


async def test_publisher_hands_a_new_conversation_to_the_companion():
    from uuid import UUID, uuid4

    class CommandSocket(FakeSocket):
        def __init__(self):
            super().__init__()
            self.commands = asyncio.Queue()

        async def recv(self):
            return await self.commands.get()

    socket = CommandSocket()
    started: asyncio.Queue[tuple[UUID, UUID]] = asyncio.Queue()

    async def connect(url):
        return socket

    async def new_session(fish_id: UUID, session_id: UUID) -> bool:
        await started.put((fish_id, session_id))
        return True

    publisher = PetEventPublisher(
        "http://127.0.0.1:8000", api_key="test", websocket_connector=connect
    )
    publisher.set_new_session_handler(new_session)
    fish, session = uuid4(), uuid4()
    await publisher.start()
    try:
        await socket.commands.put(
            json.dumps(
                {
                    "type": "pet.session.new",
                    "payload": {"fishId": str(fish), "sessionId": str(session)},
                }
            )
        )
        assert await asyncio.wait_for(started.get(), timeout=1) == (fish, session)
    finally:
        await publisher.close()


async def test_publisher_hands_the_owner_voice_add_command_to_the_companion():
    class CommandSocket(FakeSocket):
        def __init__(self):
            super().__init__()
            self.commands = asyncio.Queue()

        async def recv(self):
            return await self.commands.get()

    socket = CommandSocket()
    added = asyncio.Event()

    async def connect(url):
        return socket

    async def add() -> None:
        added.set()

    publisher = PetEventPublisher(
        "http://127.0.0.1:8000", api_key="test", websocket_connector=connect
    )
    publisher.set_owner_voice_add_handler(add)
    await publisher.start()
    try:
        await socket.commands.put(
            json.dumps({"type": "pet.owner_voice.add", "payload": {}})
        )
        await asyncio.wait_for(added.wait(), timeout=1)
    finally:
        await publisher.close()


def test_unrecognized_owner_voice_carries_only_bounded_numbers():
    from fish_memory.pet.protocol import parse_published_control

    ok = parse_published_control(
        {"type": "pet.owner_voice.unrecognized", "payload": {"seconds": 3.4}}
    )
    assert ok.payload == {"seconds": 3.4}
    parse_published_control(
        {
            "type": "pet.owner_voice.unrecognized",
            "payload": {"seconds": 0, "enrollmentCount": 2},
        }
    )
    for bad in (
        {"seconds": -1},
        {"seconds": 99},
        {"seconds": True},
        {"seconds": 1, "pcm": "AAAA"},
        {"seconds": 0, "enrollmentCount": 0},
    ):
        with pytest.raises(ValueError):
            parse_published_control(
                {"type": "pet.owner_voice.unrecognized", "payload": bad}
            )
