import asyncio

import pytest

from fish_memory.llm.configuration import DesktopModelRoutes, LanguageModelEndpoint
from fish_memory.llm.runtime import ModelRuntime


def routes(chat="old", reasoning="reason", key="test-key"):
    def endpoint(model):
        return LanguageModelEndpoint(
            baseUrl="https://example.invalid/v1", model=model, apiKey=key
        )

    return DesktopModelRoutes(chat=endpoint(chat), reasoning=endpoint(reasoning))


class FakeClient:
    def __init__(self, *, model, api_key, base_url):
        self.model = model
        self.key = api_key
        self.closed = 0
        self.block = asyncio.Event()
        self.entered = asyncio.Event()
        self.stream_closed = False

    async def stream(self, *_):
        try:
            yield self.model
            self.entered.set()
            await self.block.wait()
            yield self.model
        finally:
            self.stream_closed = True

    async def summarize_window(self):
        self.entered.set()
        await self.block.wait()
        return self.model

    async def complete_json(self):
        return self.model

    def request_payload(self):
        return {"model": self.model}

    async def close(self):
        self.closed += 1


@pytest.mark.asyncio
async def test_model_revision_detects_switching_back_to_previous_configuration():
    runtime = ModelRuntime(
        routes(), text_factory=FakeClient, environment_factory=FakeClient
    )
    initial = runtime.revision
    assert not runtime.configure(routes())
    assert runtime.revision == initial
    runtime.configure(routes("new"))
    runtime.configure(routes())
    assert runtime.revision == initial + 2
    await runtime.close()


@pytest.mark.asyncio
async def test_switch_keeps_old_stream_and_reuses_unchanged_reasoning_connection():
    runtime = ModelRuntime(
        routes(), text_factory=FakeClient, environment_factory=FakeClient
    )
    old, reason = runtime.client("chat"), runtime.client("reasoning")
    stream = runtime.chat.stream([])
    assert await anext(stream) == "old"
    runtime.configure(routes("new"))
    assert await runtime.chat.complete_json() == "new"
    assert runtime.client("reasoning") is reason
    assert not old.closed
    old.block.set()
    assert await anext(stream) == "old"
    await stream.aclose()
    await asyncio.sleep(0)
    assert old.closed == 1
    assert old.stream_closed
    assert not reason.closed
    await runtime.close()
    assert reason.closed == 1


@pytest.mark.asyncio
async def test_environment_request_keeps_its_borrowed_chat_client_during_switch():
    runtime = ModelRuntime(
        routes(), text_factory=FakeClient, environment_factory=FakeClient
    )
    old = runtime.client("chat")
    env = runtime.client("environment")
    task = asyncio.create_task(runtime.environment.summarize_window())
    await env.entered.wait()
    runtime.configure(routes("new", "new-reason"))
    assert not old.closed and not env.closed
    env.block.set()
    assert await task == "reason"
    await asyncio.sleep(0)
    assert old.closed == env.closed == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_cancel_releases_retired_stream_without_closing_current_client():
    runtime = ModelRuntime(
        routes(), text_factory=FakeClient, environment_factory=FakeClient
    )
    old = runtime.client("chat")
    stream = runtime.chat.stream([])
    assert await anext(stream) == "old"
    task = asyncio.create_task(anext(stream))
    await old.entered.wait()
    runtime.configure(routes("new"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert old.closed == 1
    assert not runtime.client("chat").closed
    await runtime.close()


@pytest.mark.asyncio
async def test_equal_endpoints_share_client_and_unchanged_save_does_no_work():
    created = []

    def factory(**kwargs):
        client = FakeClient(**kwargs)
        created.append(client)
        return client

    runtime = ModelRuntime(
        routes("same", "same"), text_factory=factory, environment_factory=FakeClient
    )
    assert len(created) == 1
    assert runtime.client("chat") is runtime.client("reasoning")
    assert runtime.configure(routes("same", "same")) is False
    assert len(created) == 1
    runtime.configure(routes("same", "same", "new-key"))
    assert len(created) == 2
    assert runtime.client("chat").key == "new-key"
    await runtime.close()
    assert [client.closed for client in created] == [1, 1]


@pytest.mark.asyncio
async def test_failed_replacement_keeps_routes_and_retires_partial_clients():
    created = []

    def factory(**kwargs):
        if kwargs["model"] == "invalid":
            raise ValueError("invalid model")
        client = FakeClient(**kwargs)
        created.append(client)
        return client

    runtime = ModelRuntime(
        routes(), text_factory=factory, environment_factory=FakeClient
    )
    with pytest.raises(ValueError):
        runtime.configure(routes("new", "invalid"))
    assert runtime.routes == routes()
    assert runtime.chat.request_payload() == {"model": "old"}
    await asyncio.sleep(0)
    assert created[-1].closed == 1
    assert created[0].closed == 0
    await runtime.close()


class FailingClient(FakeClient):
    error: Exception = RuntimeError("provider down")

    async def stream(self, *_):
        yield "partial"
        raise self.error

    async def complete_json(self):
        raise self.error


@pytest.mark.asyncio
async def test_each_route_marks_its_own_failures_and_clears_them_on_success(
    monkeypatch,
):
    from fish_memory import failures

    health = failures.ServiceHealth()
    monkeypatch.setattr(failures, "health", health)
    runtime = ModelRuntime(
        routes(), text_factory=FailingClient, environment_factory=FailingClient
    )
    FailingClient.error = PermissionError("Arrearage: account overdue")
    with pytest.raises(PermissionError) as raised:
        async for _ in runtime.chat.stream([]):
            pass
    assert failures.failure_of(raised.value) == failures.ServiceFailure("chat", "quota")
    FailingClient.error = TimeoutError()
    with pytest.raises(TimeoutError) as raised:
        await runtime.environment.complete_json()
    assert failures.failure_of(raised.value) == failures.ServiceFailure(
        "reasoning", "timeout"
    )
    assert health.problems() == (failures.ServiceFailure("chat", "quota"),)

    runtime.configure(routes("working"))
    monkeypatch.setattr(FailingClient, "complete_json", FakeClient.complete_json)
    assert await runtime.chat.complete_json() == "working"
    assert health.problems() == ()
    await runtime.close()
