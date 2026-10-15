from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI

from fish_memory.llm.client import BailianChatClient, ChatStreamChunk
from fish_memory.llm.log_transport import LoggedHttpClient
from fish_memory.llm.logs import CallJournal


def sse(content: str | None, *, usage: bool = False) -> bytes:
    value: dict[str, Any] = {
        "id": "test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test",
        "choices": (
            []
            if usage
            else [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": "stop" if content is None else None,
                }
            ]
        ),
    }
    if usage:
        value["usage"] = {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "total_tokens": 15,
        }
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


DONE = b"data: [DONE]\n\n"


class Body(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes],
        *,
        tail: Literal["eof", "block", "error"] = "eof",
    ) -> None:
        self.chunks = chunks
        self.tail = tail
        self.tail_started = asyncio.Event()
        self.closed = False
        self.eof = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for data in self.chunks:
            yield data
        self.tail_started.set()
        if self.tail == "block":
            await asyncio.Event().wait()
        if self.tail == "error":
            raise httpx.ReadError("broken transport")
        self.eof = True

    async def aclose(self) -> None:
        self.closed = True


@asynccontextmanager
async def mock_client(body: Body) -> AsyncIterator[BailianChatClient]:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=body,
        )
    )
    sdk = AsyncOpenAI(
        api_key="local-test-only",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport, trust_env=False),
    )
    client = BailianChatClient(
        api_key="local-test-only",
        base_url="https://example.invalid/v1",
        model="test",
        client=sdk,
    )
    try:
        yield client
    finally:
        await client.close()


async def collect(client: BailianChatClient) -> list[ChatStreamChunk]:
    return [chunk async for chunk in client.stream_json([])]


@pytest.mark.asyncio
@pytest.mark.parametrize("record_calls", [False, True])
async def test_reuses_actual_http_connection_after_sse_done(record_calls: bool) -> None:
    connections = 0
    bodies: list[dict[str, Any]] = []
    handlers: set[asyncio.Task[Any]] = set()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        task = asyncio.current_task()
        assert task is not None
        handlers.add(task)
        try:
            while True:
                headers = await reader.readuntil(b"\r\n\r\n")
                fields = dict(
                    line.lower().split(b":", 1)
                    for line in headers.split(b"\r\n")[1:]
                    if b":" in line
                )
                body = await reader.readexactly(int(fields[b"content-length"]))
                bodies.append(json.loads(body))
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                for part in [sse("你好🐟"), sse(None), sse(None, usage=True), DONE]:
                    writer.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
                await writer.drain()
                # DONE and HTTP EOF deliberately arrive in separate reads.
                await asyncio.sleep(0.005)
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            handlers.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    store = SimpleNamespace(save=AsyncMock())
    journal = CallJournal(store) if record_calls else None
    sdk = AsyncOpenAI(
        api_key="local-test-only",
        base_url=f"http://127.0.0.1:{port}/v1",
        max_retries=0,
        http_client=(
            LoggedHttpClient(journal, trust_env=False, timeout=2)
            if journal
            else httpx.AsyncClient(trust_env=False, timeout=2)
        ),
    )
    client = BailianChatClient(
        api_key="local-test-only", base_url=str(sdk.base_url), model="test", client=sdk
    )
    try:
        async with server:
            for method in [client.stream, client.stream_json, client.stream_json]:
                chunks = [part async for part in method([])]
                assert chunks == [
                    ChatStreamChunk(content="你好🐟"),
                    ChatStreamChunk(finish_reason="stop"),
                    ChatStreamChunk(input_tokens=12, output_tokens=3),
                ]
            assert connections == 1
            assert len(bodies) == 3
            assert all(body["enable_thinking"] is False for body in bodies)
            assert "response_format" not in bodies[0]
            assert bodies[1]["response_format"] == {"type": "json_object"}
            if journal:
                await journal.queue.join()
                completed = [
                    c.args[0]
                    for c in store.save.await_args_list
                    if c.args[0]["status"] == "completed"
                ]
                assert len(completed) == 3
                assert all(c["response"]["text"] == "你好🐟" for c in completed)
    finally:
        await client.close()
        if journal:
            await journal.close()
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 2)


@pytest.mark.asyncio
async def test_drains_after_done_without_publishing_trailing_events() -> None:
    body = Body([sse("你好🐟"), sse(None, usage=True), DONE, b"data: invalid-json\n\n"])
    async with mock_client(body) as client:
        chunks = await collect(client)
    assert chunks == [
        ChatStreamChunk(content="你好🐟"),
        ChatStreamChunk(input_tokens=12, output_tokens=3),
    ]
    assert body.eof and body.closed


@pytest.mark.asyncio
async def test_eof_without_done_still_completes() -> None:
    body = Body([sse("answer"), sse(None, usage=True)])
    async with mock_client(body) as client:
        chunks = await collect(client)
    assert chunks[-1].input_tokens == 12
    assert body.eof and body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["block", "error"])
async def test_failed_tail_drain_keeps_completed_answer(
    tail: Literal["block", "error"], monkeypatch: pytest.MonkeyPatch
) -> None:
    import fish_memory.llm.streaming as streaming

    monkeypatch.setattr(streaming, "_HTTP_EOF_DRAIN_TIMEOUT_SECONDS", 0.01)
    body = Body([sse("answer"), sse(None, usage=True), DONE], tail=tail)
    async with mock_client(body) as client:
        chunks = await asyncio.wait_for(collect(client), 0.5)
    assert chunks[0].content == "answer"
    assert chunks[-1].input_tokens == 12
    assert body.closed and not body.eof


@pytest.mark.asyncio
async def test_transport_error_before_done_propagates() -> None:
    body = Body([sse("partial")], tail="error")
    async with mock_client(body) as client:
        with pytest.raises(httpx.ReadError):
            await collect(client)
    assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("after_done", [False, True])
async def test_cancellation_closes_immediately_including_during_drain(
    after_done: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fish_memory.llm.streaming as streaming

    monkeypatch.setattr(streaming, "_HTTP_EOF_DRAIN_TIMEOUT_SECONDS", 60)
    body = Body([sse("partial"), *([DONE] if after_done else [])], tail="block")
    async with mock_client(body) as client:
        task = asyncio.create_task(collect(client))
        try:
            await asyncio.wait_for(body.tail_started.wait(), 0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 0.5)
            assert body.closed
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["stream", "stream_json"])
async def test_first_content_streams_immediately_and_early_close_releases_response(
    method: str,
) -> None:
    body = Body([sse("partial"), DONE], tail="block")
    async with mock_client(body) as client:
        stream = getattr(client, method)([])
        try:
            first = await asyncio.wait_for(anext(stream), 0.5)
            assert first.content == "partial"
            assert not body.tail_started.is_set()
            await asyncio.wait_for(stream.aclose(), 0.5)
            assert body.closed
            assert not body.tail_started.is_set()
        finally:
            await stream.aclose()
