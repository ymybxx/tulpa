import asyncio
from collections.abc import Sequence
from uuid import UUID

import httpx
import pytest

from fish_memory.companion.binding import (
    AssistantBindingClient,
    AssistantBindingWatcher,
)

FISH_ID = UUID(int=1)
SECOND_FISH_ID = UUID(int=2)


@pytest.mark.asyncio
async def test_binding_client_sends_api_key_as_bearer() -> None:
    authorization: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization.append(request.headers.get("Authorization"))
        return httpx.Response(200, json={"fishId": None})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AssistantBindingClient(
        "http://test",
        http_client=http_client,
        api_key="fm_companion_key",
    )

    assert await client.get_current() is None
    await http_client.aclose()
    assert authorization == ["Bearer fm_companion_key"]


class FakeBindingClient:
    def __init__(self, values: Sequence[UUID | None | Exception]) -> None:
        self.values = list(values)
        self.calls = 0

    async def get_current(self) -> UUID | None:
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.asyncio
async def test_poll_once_returns_current_binding_without_creating_fish() -> None:
    client = FakeBindingClient([None, FISH_ID, FISH_ID, SECOND_FISH_ID])
    watcher = AssistantBindingWatcher(client, poll_interval_seconds=0)

    assert await watcher.poll_once() is None
    assert await watcher.poll_once() == FISH_ID
    assert await watcher.poll_once() == FISH_ID
    assert await watcher.poll_once() == SECOND_FISH_ID
    assert client.calls == 4


@pytest.mark.asyncio
async def test_run_emits_only_binding_changes_and_handles_unbinding() -> None:
    client = FakeBindingClient([None, FISH_ID, FISH_ID, SECOND_FISH_ID, None])
    watcher = AssistantBindingWatcher(client, poll_interval_seconds=0)
    changes: list[UUID | None] = []

    async def on_change(fish_id: UUID | None) -> None:
        changes.append(fish_id)
        if len(changes) == 4:
            watcher.close()

    await watcher.run(on_change)

    assert changes == [None, FISH_ID, SECOND_FISH_ID, None]


@pytest.mark.asyncio
async def test_run_continues_polling_after_initial_binding() -> None:
    client = FakeBindingClient([FISH_ID, SECOND_FISH_ID])
    watcher = AssistantBindingWatcher(client, poll_interval_seconds=0)
    changes: list[UUID | None] = []

    async def on_change(fish_id: UUID | None) -> None:
        changes.append(fish_id)
        if len(changes) == 2:
            watcher.close()

    await watcher.run(on_change)

    assert changes == [FISH_ID, SECOND_FISH_ID]
    assert client.calls == 2


@pytest.mark.asyncio
async def test_run_keeps_last_binding_when_poll_fails() -> None:
    client = FakeBindingClient([FISH_ID, RuntimeError("temporary"), SECOND_FISH_ID])
    watcher = AssistantBindingWatcher(client, poll_interval_seconds=0)
    changes: list[UUID | None] = []

    async def on_change(fish_id: UUID | None) -> None:
        changes.append(fish_id)
        if len(changes) == 2:
            watcher.close()

    await watcher.run(on_change)

    assert changes == [FISH_ID, SECOND_FISH_ID]


@pytest.mark.asyncio
async def test_run_retries_binding_when_change_callback_fails_once() -> None:
    client = FakeBindingClient([FISH_ID, FISH_ID])
    watcher = AssistantBindingWatcher(client, poll_interval_seconds=0)
    attempts: list[UUID | None] = []

    async def on_change(fish_id: UUID | None) -> None:
        attempts.append(fish_id)
        if len(attempts) == 1:
            raise RuntimeError("temporary audio transition failure")
        watcher.close()

    await watcher.run(on_change)

    assert attempts == [FISH_ID, FISH_ID]
    assert client.calls == 2


@pytest.mark.asyncio
async def test_run_observes_later_switch_after_callback_retry_succeeds() -> None:
    client = FakeBindingClient([FISH_ID, FISH_ID, SECOND_FISH_ID])
    watcher = AssistantBindingWatcher(client, poll_interval_seconds=0)
    attempts: list[UUID | None] = []

    async def on_change(fish_id: UUID | None) -> None:
        attempts.append(fish_id)
        if len(attempts) == 1:
            raise RuntimeError("temporary audio transition failure")
        if fish_id == SECOND_FISH_ID:
            watcher.close()

    await watcher.run(on_change)

    assert attempts == [FISH_ID, FISH_ID, SECOND_FISH_ID]
    assert client.calls == 3


@pytest.mark.asyncio
async def test_close_stops_a_running_watcher() -> None:
    client = FakeBindingClient([FISH_ID])
    watcher = AssistantBindingWatcher(client, poll_interval_seconds=60)
    task = asyncio.create_task(watcher.run(lambda _: _completed()))

    await asyncio.sleep(0)
    watcher.close()
    await asyncio.wait_for(task, timeout=0.2)


async def _completed() -> None:
    return None
