"""Socket-lifetime facial delivery with synthetic TTS and no model requests."""

import asyncio
from types import SimpleNamespace

import pytest
from starlette.websockets import WebSocketDisconnect

from fish_memory.tts import routes
from fish_memory.tts.facial import FacialProfile, SpeechFacialPlan

PROFILE = FacialProfile(
    revision="model-v1",
    modelId="model",
    modelName="猫",
    parameters=[
        {"id": "ParamAngleX", "minimum": -30, "maximum": 30, "defaultValue": 0}
    ],
)


class LocalService:
    is_ready = True
    model_name = "local-test"

    async def prepare(self):
        pass

    async def stream(self, text, **kwargs):
        yield b"\x00\x00" * 160


class Socket:
    def __init__(self):
        settings = SimpleNamespace(
            tts_provider="local", speech_facial_wait_for_ready=False
        )
        settings.model_copy = lambda: settings
        self.app = SimpleNamespace(
            state=SimpleNamespace(settings=settings, local_tts_service=LocalService())
        )
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.sent = []

    async def accept(self):
        pass

    async def receive_json(self):
        message = await self.incoming.get()
        if message is None:
            raise WebSocketDisconnect()
        return message

    async def send_json(self, message):
        self.sent.append(message)
        await self.outgoing.put(message)

    async def send_bytes(self, pcm):
        self.sent.append(pcm)

    async def close(self, **kwargs):
        pass

    async def start(self):
        await self.incoming.put(
            {
                "type": "start",
                "model": "local-test",
                "voice": "default",
                "sample_rate": 16000,
                "facial": True,
            }
        )
        await self.incoming.put({"type": "append", "text": "你好。"})
        await self.incoming.put({"type": "finish"})

    async def until(self, kind):
        async with asyncio.timeout(0.5):
            while True:
                message = await self.outgoing.get()
                if message["type"] == kind:
                    return message


def factory(monkeypatch):
    plans, releases = [], []

    def create(connection, enabled, *, wait_for_ready):
        release = asyncio.Event()
        releases.append(release)

        async def predict(text, previous):
            await release.wait()
            return {"ParamAngleX": 12}

        plan = SpeechFacialPlan(PROFILE, predict, wait_for_ready=wait_for_ready)
        plans.append(plan)
        return plan

    monkeypatch.setattr(routes, "create_speech_facial_plan", create)
    return plans, releases


async def test_finished_does_not_wait_or_cancel_late_face_and_socket_stays_available(
    monkeypatch,
):
    plans, releases = factory(monkeypatch)
    socket = Socket()
    runner = asyncio.create_task(routes.tts_stream(socket))
    try:
        await socket.start()
        ready = await socket.until("ready")
        audio = await socket.until("audio")
        await socket.until("finished")
        assert audio["facialWaitMs"] == 0
        assert not releases[0].is_set()
        assert not plans[0]._closed
        assert not any(
            isinstance(item, dict) and item["type"] == "facial" for item in socket.sent
        )
        releases[0].set()
        face = await socket.until("facial")
        assert face["streamId"] == ready["facialStreamId"]
        assert face["startOffset"] == 0 and face["endOffset"] == 3
        assert face["waitMs"] == 0
    finally:
        await socket.incoming.put(None)
        await asyncio.wait_for(runner, 0.5)
    assert plans[0]._closed and not plans[0]._deliveries


@pytest.mark.parametrize("terminal", ["stop", "disconnect", "new_start"])
async def test_finished_background_face_is_cancelled_at_actual_utterance_retirement(
    monkeypatch, terminal
):
    plans, releases = factory(monkeypatch)
    socket = Socket()
    runner = asyncio.create_task(routes.tts_stream(socket))
    await socket.start()
    await socket.until("finished")
    if terminal == "new_start":
        # The next request snapshots the newly saved setting, not a mutable
        # reference changing the old utterance's synchronization contract.
        socket.app.state.settings.speech_facial_wait_for_ready = True
        await socket.start()
        await socket.until("ready")
        assert plans[0].wait_for_ready is False
        assert plans[1].wait_for_ready is True
    else:
        await socket.incoming.put(
            None if terminal == "disconnect" else {"type": "stop"}
        )
    for _ in range(10):
        await asyncio.sleep(0)
    assert plans[0]._closed
    releases[0].set()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not any(
        isinstance(item, dict)
        and item["type"] == "facial"
        and item["streamId"] == plans[0].stream_id
        for item in socket.sent
    )
    if terminal == "new_start":
        releases[1].set()
        await socket.until("finished")
    if terminal != "disconnect":
        await socket.incoming.put(None)
    await asyncio.wait_for(runner, 0.5)
    assert all(not plan._deliveries for plan in plans)


async def test_implicit_append_recovery_keeps_identity_offsets_and_policy(monkeypatch):
    plans, releases = factory(monkeypatch)
    socket = Socket()
    runner = asyncio.create_task(routes.tts_stream(socket))
    try:
        await socket.start()
        first_ready = await socket.until("ready")
        await socket.until("finished")
        socket.app.state.settings.speech_facial_wait_for_ready = True
        await socket.incoming.put({"type": "append", "text": "你好。"})
        await socket.incoming.put({"type": "finish"})
        second_ready = await socket.until("ready")
        await socket.until("finished")
        assert first_ready["facialStreamId"] == second_ready["facialStreamId"]
        assert len(plans) == 1 and plans[0].wait_for_ready is False
        releases[0].set()
        faces = [await socket.until("facial"), await socket.until("facial")]
        assert sorted((face["startOffset"], face["endOffset"]) for face in faces) == [
            (0, 3),
            (3, 6),
        ]
    finally:
        await socket.incoming.put(None)
        await asyncio.wait_for(runner, 0.5)
