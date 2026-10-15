import asyncio
from typing import Any

import pytest

from fish_memory.home.client import HomeAssistantClient, HomeAssistantError


class FakeSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, value: str) -> None:
        import json

        message = json.loads(value)
        self.sent.append(message)
        if message["type"] == "auth":
            await self.incoming.put({"type": "auth_ok"})
        elif message["type"] == "get_states":
            await self.incoming.put(
                {
                    "id": message["id"],
                    "type": "result",
                    "success": True,
                    "result": [
                        {
                            "entity_id": "light.entry",
                            "state": "off",
                            "attributes": {"friendly_name": "玄关灯"},
                        }
                    ],
                }
            )
        elif message["type"].startswith("config/"):
            await self.incoming.put(
                {
                    "id": message["id"],
                    "type": "result",
                    "success": True,
                    "result": [],
                }
            )
        elif message["type"] == "get_services":
            await self.incoming.put(
                {
                    "id": message["id"],
                    "type": "result",
                    "success": True,
                    "result": {"light": {"turn_on": {"fields": {}}}},
                }
            )
        elif message["type"] == "subscribe_events":
            await self.incoming.put(
                {
                    "id": message["id"],
                    "type": "result",
                    "success": True,
                    "result": None,
                }
            )
        elif message["type"] == "call_service":
            await self.incoming.put(
                {
                    "id": message["id"],
                    "type": "result",
                    "success": True,
                    "result": {"context": {"id": "ctx"}},
                }
            )

    async def recv(self) -> str:
        import json

        return json.dumps(await self.incoming.get())

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_client_authenticates_loads_catalog_and_updates_state() -> None:
    socket = FakeSocket()
    await socket.incoming.put({"type": "auth_required"})

    async def connect(_url: str) -> FakeSocket:
        return socket

    client = HomeAssistantClient(
        base_url="http://ha.local:8123",
        token="secret",
        connect=connect,
        reconnect_delays=(0.01,),
    )
    await client.start()
    await asyncio.wait_for(client.wait_ready(), timeout=1)

    assert socket.sent[0] == {"type": "auth", "access_token": "secret"}
    assert client.catalog.get("light.entry").state == "off"

    await socket.incoming.put(
        {
            "id": 6,
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "light.entry",
                    "new_state": {
                        "entity_id": "light.entry",
                        "state": "on",
                        "attributes": {"friendly_name": "玄关灯"},
                    },
                },
            },
        }
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert client.catalog.get("light.entry").state == "on"
    await client.close()
    assert socket.closed is True


@pytest.mark.asyncio
async def test_call_service_uses_home_assistant_protocol() -> None:
    socket = FakeSocket()
    await socket.incoming.put({"type": "auth_required"})

    async def connect(_url: str) -> FakeSocket:
        return socket

    client = HomeAssistantClient(
        base_url="http://ha.local:8123",
        token="secret",
        connect=connect,
    )
    await client.start()
    await asyncio.wait_for(client.wait_ready(), timeout=1)

    result = await client.call_service(
        "light",
        "turn_on",
        service_data={"brightness_pct": 40},
        target={"entity_id": "light.entry"},
    )

    call = next(item for item in socket.sent if item["type"] == "call_service")
    assert call["domain"] == "light"
    assert call["service"] == "turn_on"
    assert call["target"] == {"entity_id": "light.entry"}
    assert "return_response" not in call
    assert result == {"context": {"id": "ctx"}}
    await client.close()


@pytest.mark.asyncio
async def test_call_service_preserves_provider_validation_error() -> None:
    socket = FakeSocket()
    await socket.incoming.put({"type": "auth_required"})

    original_send = socket.send

    async def reject_call(value: str) -> None:
        import json

        message = json.loads(value)
        if message["type"] != "call_service":
            await original_send(value)
            return
        socket.sent.append(message)
        await socket.incoming.put(
            {
                "id": message["id"],
                "type": "result",
                "success": False,
                "error": {"code": "service_validation_error"},
            }
        )

    socket.send = reject_call  # type: ignore[method-assign]

    async def connect(_url: str) -> FakeSocket:
        return socket

    client = HomeAssistantClient(
        base_url="http://ha.local:8123",
        token="secret",
        connect=connect,
    )
    await client.start()
    await asyncio.wait_for(client.wait_ready(), timeout=1)

    with pytest.raises(HomeAssistantError, match="service_validation_error"):
        await client.call_service("light", "turn_on")

    await client.close()


@pytest.mark.asyncio
async def test_wait_for_state_observes_state_changed_event() -> None:
    socket = FakeSocket()
    await socket.incoming.put({"type": "auth_required"})

    async def connect(_url: str) -> FakeSocket:
        return socket

    client = HomeAssistantClient(
        base_url="http://ha.local:8123",
        token="secret",
        connect=connect,
    )
    await client.start()
    await asyncio.wait_for(client.wait_ready(), timeout=1)
    waiting = asyncio.create_task(
        client.wait_for_state("light.entry", frozenset({"on"}), timeout_seconds=1)
    )
    await socket.incoming.put(
        {
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "light.entry",
                    "new_state": {
                        "entity_id": "light.entry",
                        "state": "on",
                        "attributes": {"friendly_name": "玄关灯"},
                    },
                },
            },
        }
    )

    assert await waiting == "on"
    await client.close()


@pytest.mark.asyncio
async def test_wait_for_cover_movement_ignores_unchanged_generic_open_state() -> None:
    socket = FakeSocket()
    await socket.incoming.put({"type": "auth_required"})

    async def connect(_url: str) -> FakeSocket:
        return socket

    client = HomeAssistantClient(
        base_url="http://ha.local:8123",
        token="secret",
        connect=connect,
    )
    await client.start()
    await asyncio.wait_for(client.wait_ready(), timeout=1)
    client._states.append(  # noqa: SLF001 - exercise state-event waiting
        {
            "entity_id": "cover.bedroom",
            "state": "open",
            "attributes": {"current_position": 1},
        }
    )
    client._rebuild_catalog()  # noqa: SLF001
    waiting = asyncio.create_task(
        client.wait_for_cover_movement(
            "cover.bedroom",
            1,
            100,
            timeout_seconds=1,
        )
    )
    await asyncio.sleep(0)
    assert waiting.done() is False

    await socket.incoming.put(
        {
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "cover.bedroom",
                    "new_state": {
                        "entity_id": "cover.bedroom",
                        "state": "open",
                        "attributes": {"current_position": 2},
                    },
                },
            },
        }
    )

    assert await waiting == "open"
    await client.close()
