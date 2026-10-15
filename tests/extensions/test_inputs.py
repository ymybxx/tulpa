import asyncio
from unittest.mock import Mock
from uuid import uuid4

import pytest

from fish_memory.extensions.inputs import PluginInputs
from fish_memory.extensions.models import ExtensionError, InputDefinition, Instance


def instance():
    item = Instance(uuid4(), "example", uuid4(), uuid4(), False, "test-token")
    item.connected = True
    item.inputs["example.message"] = InputDefinition(
        type="example.message",
        title="测试输入",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string", "minLength": 1}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )
    return item


def request(text="hello"):
    return {"event_id": str(uuid4()), "type": "example.message", "data": {"text": text}}


@pytest.mark.asyncio
async def test_input_waits_for_plugin_receipt_and_duplicate_does_not_deliver_twice():
    inputs, item, emit, body = PluginInputs(), instance(), Mock(), request()
    first = asyncio.create_task(inputs.send(item, body, emit))
    second = asyncio.create_task(inputs.send(item, body, emit))
    await asyncio.sleep(0)
    emit.assert_called_once_with(item, "input", body)
    assert not first.done() and not second.done()
    receipt = {"disposition": "stored_only", "reason": "pet_busy"}
    assert inputs.complete(item, {"request_id": body["event_id"], "result": receipt})
    assert await first == await second == receipt
    assert await inputs.send(item, body, emit) == receipt
    assert emit.call_count == 1
    with pytest.raises(ExtensionError, match="不同内容"):
        await inputs.send(item, {**body, "data": {"text": "changed"}}, emit)


@pytest.mark.asyncio
async def test_cancelled_http_waiter_does_not_cancel_other_waiters():
    inputs, item, body = PluginInputs(), instance(), request()
    first = asyncio.create_task(inputs.send(item, body, Mock()))
    await asyncio.sleep(0)
    second = asyncio.create_task(inputs.send(item, body, Mock()))
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    inputs.complete(
        item, {"request_id": body["event_id"], "result": {"accepted": True}}
    )
    assert await second == {"accepted": True}


@pytest.mark.asyncio
async def test_stop_and_new_run_cannot_receive_old_input_receipt():
    inputs, old, body = PluginInputs(), instance(), request()
    task = asyncio.create_task(inputs.send(old, body, Mock()))
    await asyncio.sleep(0)
    new = instance()
    new.id = old.id
    assert inputs.complete(new, {"request_id": body["event_id"], "result": {}}) == {
        "received": False
    }
    inputs.disconnect(old)
    assert await task == {"disposition": "unknown", "reason": "disconnected"}
    old.stopped = True
    with pytest.raises(ExtensionError, match="启动插件"):
        await inputs.send(old, request(), Mock())


@pytest.mark.asyncio
async def test_invalid_input_never_reaches_plugin():
    inputs, item, emit = PluginInputs(), instance(), Mock()
    with pytest.raises(ExtensionError):
        await inputs.send(item, request(""), emit)
    with pytest.raises(ExtensionError):
        await inputs.send(item, {**request(), "type": "other"}, emit)
    emit.assert_not_called()


@pytest.mark.asyncio
async def test_slow_plugin_has_bounded_inputs_and_no_automatic_replay(monkeypatch):
    monkeypatch.setattr("fish_memory.extensions.inputs.INPUT_TIMEOUT_SECONDS", 0.02)
    inputs, item, emit = PluginInputs(), instance(), Mock()
    bodies = [request() for _ in range(4)]
    tasks = [asyncio.create_task(inputs.send(item, body, emit)) for body in bodies]
    await asyncio.sleep(0)
    with pytest.raises(ExtensionError, match="稍后"):
        await inputs.send(item, request(), emit)
    results = await asyncio.gather(*tasks)
    assert all(
        r == {"disposition": "unknown", "reason": "input_timeout"} for r in results
    )
    assert await inputs.send(item, bodies[0], emit) == results[0]
    assert emit.call_count == 4
