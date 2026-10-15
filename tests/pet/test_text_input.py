import asyncio
from uuid import uuid4

import pytest

from fish_memory.pet.hub import PetEventHub
from fish_memory.pet.protocol import parse_published_control


@pytest.mark.asyncio
async def test_typing_is_acknowledged_once_and_never_replayed_in_snapshot() -> None:
    hub = PetEventHub()
    request_id, fish_id, message_id = uuid4(), uuid4(), uuid4()
    commands = []
    first = asyncio.create_task(
        hub.text_input.submit(request_id, fish_id, "hello", commands.append)
    )
    await asyncio.sleep(0)
    assert len(commands) == 1
    assert not first.done()
    result = {"requestId": str(request_id), "ok": True, "messageId": str(message_id)}
    parse_published_control({"type": "pet.chat.result", "payload": result})
    hub.publish_control("pet.chat.result", None, result)
    assert await first == result
    assert (
        await hub.text_input.submit(request_id, fish_id, "hello", commands.append)
        == result
    )
    assert len(commands) == 1
    assert hub.snapshot()["payload"]["events"] == []
    with pytest.raises(ValueError, match="different message"):
        await hub.text_input.submit(request_id, fish_id, "different", commands.append)
    await hub.close()


@pytest.mark.asyncio
async def test_cancelled_request_does_not_cancel_or_resubmit_an_accepted_turn() -> None:
    hub = PetEventHub()
    request_id, fish_id = uuid4(), uuid4()
    commands = []
    task = asyncio.create_task(
        hub.text_input.submit(request_id, fish_id, "hi", commands.append)
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    result = {"requestId": str(request_id), "ok": True, "messageId": str(uuid4())}
    hub.publish_control("pet.chat.result", None, result)
    assert (
        await hub.text_input.submit(request_id, fish_id, "hi", commands.append)
        == result
    )
    assert len(commands) == 1
    await hub.close()
