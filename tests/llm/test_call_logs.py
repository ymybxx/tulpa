import asyncio
import gzip
import json
from copy import deepcopy

import httpx
import pytest
from openai import AsyncOpenAI

from fish_memory.llm.client import BailianChatClient
from fish_memory.llm.configuration import DesktopModelRoutes, LanguageModelEndpoint
from fish_memory.llm.log_transport import LoggedHttpClient
from fish_memory.llm.logs import CallJournal, call_scope, response_view
from fish_memory.llm.runtime import ModelRuntime


class Store:
    def __init__(self):
        self.records = []

    async def save(self, value):
        self.records.append(deepcopy(value))


class Parts(httpx.AsyncByteStream):
    def __init__(self, *parts):
        self.parts = parts
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            yield part

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_actual_stream_body_is_preserved_without_credentials_or_added_requests():
    store = Store()
    journal = CallJournal(store)
    raw = (
        'data: {"choices":[{"delta":{"content":"第一行\\n第二行"}}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":5}}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    source = Parts(raw[:37], raw[37:])
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=source
        )

    transport = httpx.MockTransport(respond)
    http = LoggedHttpClient(journal, transport=transport)
    sdk = AsyncOpenAI(
        api_key="fixture-private-key",
        base_url="https://model.invalid/v1",
        http_client=http,
    )
    client = BailianChatClient(
        api_key="fixture-private-key",
        base_url="https://model.invalid/v1",
        model="fixture",
        client=sdk,
    )
    try:
        with call_scope("plugin.speech"):
            received = [
                chunk
                async for chunk in client.stream_json(
                    [{"role": "user", "content": "完整请求\n保留换行"}]
                )
            ]
        await journal.queue.join()
        record = store.records[-1]
        assert record["status"] == "completed"
        assert record["call_type"] == "plugin.speech"
        assert record["request"] == requests[0] and len(requests) == 1
        assert record["response"]["raw"] == raw.decode()
        assert record["response"]["text"] == "第一行\n第二行"
        assert record["response"]["usage"]["completion_tokens"] == 5
        assert "fixture-private-key" not in str(store.records)
        assert "".join(c.content or "" for c in received) == "第一行\n第二行"
        assert source.closed
    finally:
        await client.close()
        await journal.close()


@pytest.mark.asyncio
async def test_error_retry_and_json_response_are_separate_actual_calls():
    store = Store()
    journal = CallJournal(store)
    attempts = []

    def respond(request):
        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            return httpx.Response(
                500, json={"error": {"message": "temporary fixture error"}}
            )
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                    }
                ],
            },
        )

    http = LoggedHttpClient(journal, transport=httpx.MockTransport(respond))
    sdk = AsyncOpenAI(
        api_key="fixture",
        base_url="https://model.invalid/v1",
        http_client=http,
        max_retries=1,
    )
    client = BailianChatClient(
        api_key="fixture",
        base_url="https://model.invalid/v1",
        model="fixture",
        client=sdk,
    )
    try:
        result = await client.complete_json([{"role": "user", "content": "fixture"}])
        await journal.queue.join()
        finished = [r for r in store.records if r["status"] != "running"]
        assert [r["status"] for r in finished] == ["failed", "completed"]
        assert len({r["id"] for r in finished}) == 2
        assert json.loads(result.content) == {"ok": True}
        assert (
            finished[1]["response"]["body"]["choices"][0]["message"]["content"]
            == result.content
        )
    finally:
        await client.close()
        await journal.close()


@pytest.mark.asyncio
async def test_cancelled_partial_stream_is_recorded_and_closed_once():
    store = Store()
    journal = CallJournal(store)
    source = Parts(b"first", b"second")
    http = LoggedHttpClient(
        journal,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=source)),
    )
    try:
        async with http.stream(
            "POST",
            "https://model.invalid/v1/chat/completions?secret=hidden",
            json={"model": "fixture"},
        ) as response:
            async for chunk in response.aiter_bytes():
                assert chunk == b"first"
                break
        await journal.queue.join()
        assert source.closed
        assert store.records[-1]["status"] == "cancelled"
        assert store.records[-1]["response"]["raw"] == "first"
        assert "secret=" not in store.records[-1]["request_url"]
        assert len(store.records) == 2
    finally:
        await http.aclose()
        await journal.close()


@pytest.mark.asyncio
async def test_compressed_response_and_concurrent_scope_are_not_changed():
    store = Store()
    journal = CallJournal(store)
    body = json.dumps({"line": "a\nb"}).encode()
    http = LoggedHttpClient(
        journal,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "content-encoding": "gzip",
                },
                stream=Parts(gzip.compress(body)),
            )
        ),
    )

    async def invoke(name):
        with call_scope(name):
            await asyncio.sleep(0)
            return (
                await http.post(
                    "https://model.invalid/v1/responses", json={"model": "fixture"}
                )
            ).json()

    try:
        assert (
            await asyncio.gather(invoke("memory.static"), invoke("memory.dynamic"))
            == [{"line": "a\nb"}] * 2
        )
        await journal.queue.join()
        finished = [r for r in store.records if r["status"] == "completed"]
        assert {r["call_type"] for r in finished} == {"memory.static", "memory.dynamic"}
        assert all(r["response"]["body"] == {"line": "a\nb"} for r in finished)
    finally:
        await http.aclose()
        await journal.close()


@pytest.mark.asyncio
async def test_logging_failure_never_breaks_model_return():
    class FailedStore:
        async def save(self, value):
            raise OSError("fixture failure")

    journal = CallJournal(FailedStore())
    async with LoggedHttpClient(
        journal,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True})),
    ) as client:
        response = await client.post(
            "https://model.invalid/v1/chat/completions", json={"model": "fixture"}
        )
        assert response.json() == {"ok": True}
    await journal.close()
    assert journal.dropped == 2


def test_responses_api_stream_and_malformed_json_remain_readable():
    data = 'data: {"type":"response.output_text.delta","delta":"hello\\nworld"}\n\n'
    assert response_view(data, "text/event-stream")["text"] == "hello\nworld"
    assert response_view("{broken", "application/json")["body"] == "{broken"
    malformed = 'data: {"type":"response.output_text.delta","delta":{"bad":true}}\n\n'
    assert response_view(malformed, "text/event-stream")["text"] == ""


@pytest.mark.asyncio
async def test_route_scope_does_not_leak_to_consumer_and_preserves_explicit_type():
    store = Store()
    journal = CallJournal(store)
    raw = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'

    def factory(**kwargs):
        kwargs.pop("proactive_llm", None)
        http = LoggedHttpClient(
            journal,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=Parts(raw),
                )
            ),
        )
        return BailianChatClient(
            **kwargs,
            client=AsyncOpenAI(
                api_key="fixture", base_url="https://model.invalid/v1", http_client=http
            ),
        )

    endpoint = LanguageModelEndpoint(
        apiKey="fixture", baseUrl="https://model.invalid/v1", model="fixture"
    )
    runtime = ModelRuntime(
        DesktopModelRoutes(chat=endpoint, reasoning=endpoint),
        text_factory=factory,
        environment_factory=factory,
    )
    try:
        stream = runtime.chat.stream_json([{"role": "user", "content": "fixture"}])
        assert (await anext(stream)).content == "ok"
        # Interleaved consumer work is outside the route's logging scope.
        journal.start(b"{}", "https://fixture.invalid").finish([], "", 200)
        await asyncio.create_task(stream.aclose())
        with call_scope("plugin.speech"):
            assert [chunk async for chunk in runtime.chat.stream_json([])]
        await journal.queue.join()
        assert [r["call_type"] for r in store.records if r["status"] == "running"] == [
            "chat",
            "other",
            "plugin.speech",
        ]
    finally:
        await runtime.close()
        await journal.close()


@pytest.mark.asyncio
async def test_capture_limit_never_truncates_actual_response():
    store = Store()
    journal = CallJournal(store)
    part = b"a" * (1024 * 1024)
    async with LoggedHttpClient(
        journal,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, stream=Parts(*([part] * 17)))
        ),
    ) as client:
        response = await client.post(
            "https://model.invalid/v1/chat/completions", json={"model": "fixture"}
        )
        assert len(response.content) == 17 * len(part)
    await journal.close()
    assert len(store.records[-1]["response"]["raw"]) == 16 * len(part)
    assert store.records[-1]["response"]["truncated"] is True


@pytest.mark.asyncio
async def test_done_followed_by_slow_http_tail_is_completed():
    class SlowTail(Parts):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
            await asyncio.Event().wait()

    store = Store()
    journal = CallJournal(store)
    source = SlowTail()
    http = LoggedHttpClient(
        journal,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=source
            )
        ),
    )
    client = BailianChatClient(
        api_key="fixture",
        base_url="https://model.invalid/v1",
        model="fixture",
        client=AsyncOpenAI(api_key="fixture", http_client=http),
    )
    try:
        assert [chunk.content async for chunk in client.stream_json([])] == ["ok"]
        await journal.queue.join()
        assert store.records[-1]["status"] == "completed"
        assert source.closed
    finally:
        await client.close()
        await journal.close()


@pytest.mark.asyncio
async def test_invalid_compressed_reply_still_has_a_diagnostic_record():
    store = Store()
    journal = CallJournal(store)
    call = journal.start(b"{}", "https://fixture.invalid")
    call.finish(
        [b"bad gzip"], "application/json", 200, "DecodingError", encoding="gzip"
    )
    await journal.close()
    assert journal.dropped == 0
    assert store.records[-1]["status"] == "failed"
    assert store.records[-1]["response"]["decode_error"] == "DecodingError"


@pytest.mark.asyncio
async def test_provider_sse_error_with_http_200_is_failed():
    store = Store()
    journal = CallJournal(store)
    raw = (
        b'data: {"type":"response.failed",'
        b'"response":{"error":{"code":"ToolNotOpen"}}}\n\n'
    )
    journal.start(b"{}", "https://fixture.invalid").finish(
        [raw], "text/event-stream", 200
    )
    await journal.close()
    assert store.records[-1]["status"] == "failed"
    assert "ToolNotOpen" in store.records[-1]["error"]
    assert store.records[-1]["response"]["raw"] == raw.decode()


def test_token_counts_read_every_usage_shape():
    from fish_memory.llm.log_store import token_counts

    assert token_counts({"usage": {"prompt_tokens": 30, "completion_tokens": 4}}) == (
        30,
        4,
    )
    assert token_counts(
        {"body": {"usage": {"input_tokens": 7, "output_tokens": 2}}}
    ) == (
        7,
        2,
    )
    assert token_counts({"body": {"answers": {}}}) == (None, None)
    assert token_counts({"usage": {"prompt_tokens": True}}) == (None, None)
    assert token_counts(None) == (None, None)


@pytest.mark.asyncio
async def test_calls_made_while_speaking_up_carry_that_purpose():
    from fish_memory.llm.logs import purpose_scope

    store = Store()
    journal = CallJournal(store)
    try:
        with call_scope("chat"):
            journal.start(b'{"model":"m"}', "https://model.invalid")
            with purpose_scope("proactive"):

                async def later():
                    journal.start(b'{"model":"m"}', "https://model.invalid")

                task = asyncio.create_task(later())
            await task
        await journal.queue.join()
        assert [record["purpose"] for record in store.records] == [None, "proactive"]
    finally:
        await journal.close()
