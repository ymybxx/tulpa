import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from fish_memory.answer.context import ChatMessage
from fish_memory.answer.generation import AnswerGenerationPipeline
from fish_memory.llm.ark_search import ArkSearchError, is_ark_endpoint
from fish_memory.llm.client import BailianChatClient
from fish_memory.performance.tracker import MessagePerformanceTracker

ARK_URL = "https://ark.cn-beijing.volces.com/api/v3"
MESSAGES: list[ChatMessage] = [{"role": "user", "content": "杭州天气，输出 JSON"}]


def completed(*, calls: bool = True) -> dict[str, Any]:
    citation = {
        "type": "url_citation",
        "url": "https://weather.example/",
        "title": "天气",
    }
    return {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [
                *(
                    [
                        {
                            "type": "web_search_call",
                            "status": "completed",
                            "action": {"type": "search", "query": "杭州天气"},
                        }
                    ]
                    if calls
                    else []
                ),
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [citation, citation],
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "tool_usage": {"web_search": int(calls)},
                "tool_usage_details": {"web_search": {"search_engine": int(calls)}},
            },
        },
    }


class EventStream(httpx.AsyncByteStream):
    def __init__(self, events: list[dict[str, Any]], *, block: bool = False) -> None:
        self.events = events
        self.block = block
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
        if self.block:
            self.waiting.set()
            await asyncio.Future()
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


def make_client(
    events: list[dict[str, Any]],
    *,
    base_url: str = ARK_URL,
    block: bool = False,
    http_error: int | None = None,
) -> tuple[BailianChatClient, list[httpx.Request], EventStream]:
    requests: list[httpx.Request] = []
    wire = EventStream(events, block=block)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if http_error:
            return httpx.Response(
                http_error,
                json={
                    "error": {
                        "code": "ToolNotOpen",
                        "message": "private request echoed here",
                    }
                },
            )
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    'data: {"choices":[{"delta":{"content":"hello"},'
                    '"finish_reason":"stop","index":0}]}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    sdk = AsyncOpenAI(
        api_key="secret",
        base_url=base_url,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    return (
        BailianChatClient(
            api_key="secret",
            base_url=base_url,
            model="doubao-seed-character-260628",
            client=sdk,
        ),
        requests,
        wire,
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (ARK_URL, True),
        ("https://ark.cn-shanghai.volces.com/api/v3/", True),
        ("https://ARK.CN-BEIJING.VOLCES.COM/api/v3", True),
        ("https://proxy.example/v1", False),
        ("https://ark.cn-beijing.volces.com.evil.example/v1", False),
        ("https://proxy.example/v1?provider=ark.cn-beijing.volces.com", False),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", False),
    ],
)
def test_provider_detection_uses_hostname(url, expected):
    assert is_ark_endpoint(url) is expected


@pytest.mark.asyncio
async def test_native_search_wire_json_usage_and_diagnostics():
    client, requests, wire = make_client(
        [
            {"type": "response.web_search_call.searching"},
            {"type": "response.output_text.delta", "delta": '{"searchContext":'},
            {"type": "response.output_text.delta", "delta": '"晴天"}'},
            completed(),
        ]
    )
    try:
        chunks = [c async for c in client.stream_json(MESSAGES, force_search=True)]
        assert "".join(c.content or "" for c in chunks) == '{"searchContext":"晴天"}'
        assert chunks[-1].finish_reason == "stop"
        assert (chunks[-1].input_tokens, chunks[-1].output_tokens) == (100, 20)
        assert chunks[-1].search_diagnostics == {
            "provider": "ark",
            "status": "completed",
            "toolStatus": "searching",
            "resultFormat": "json",
            "completedSearchCalls": 1,
            "calls": [{"status": "completed", "query": "杭州天气"}],
            "citations": [{"url": "https://weather.example/", "title": "天气"}],
            "toolUsage": {"web_search": 1},
            "toolUsageDetails": {"web_search": {"search_engine": 1}},
        }
        assert len(requests) == 1
        assert str(requests[0].url) == ARK_URL + "/responses"
        body = json.loads(requests[0].content)
        assert body == {
            "model": "doubao-seed-character-260628",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": MESSAGES[0]["content"]}],
                }
            ],
            "tools": [{"type": "web_search", "max_keyword": 1, "limit": 5}],
            "max_tool_calls": 1,
            "tool_choice": "required",
            "store": False,
            "stream": True,
            "thinking": {"type": "disabled"},
            "text": {"format": {"type": "json_object"}},
        }
        diagnostic = client.json_stream_request_payload(MESSAGES, force_search=True)
        assert diagnostic["protocol"] == "ark_responses"
        assert diagnostic["parameters"] == {
            k: v for k, v in body.items() if k not in {"input", "model"}
        }
        assert "secret" not in json.dumps(diagnostic)
        assert wire.closed
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("base_url", "force_search"), [(ARK_URL, False), ("https://proxy.example/v1", True)]
)
async def test_ordinary_chat_and_doubao_proxy_keep_chat_completions(
    base_url, force_search
):
    client, requests, _ = make_client([], base_url=base_url)
    try:
        chunks = [
            c async for c in client.stream_json(MESSAGES, force_search=force_search)
        ]
        assert chunks[0].content == "hello"
        assert requests[0].url.path.endswith("/chat/completions")
        assert "tools" not in json.loads(requests[0].content)
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("description", [None, "最近5秒的合成画面，最后一格最近。"])
async def test_optional_plain_search_and_images_preserve_input_order(description):
    image = {
        "data_url": "data:image/png;base64,aW1hZ2U=",
        "observed_at": "2026-09-08T22:00:00",
        "digest": "digest",
        "byte_count": 5,
        "width": 10,
        "height": 10,
    }
    if description:
        image["description"] = description
    messages = [
        {"role": "system", "content": "test"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "查图中的天气", "environment_images": [image]},
    ]
    client, requests, _ = make_client([completed(calls=False)])
    try:
        _ = [c async for c in client.stream(messages, enable_search=True)]
        body = json.loads(requests[0].content)
        assert body["tool_choice"] == "auto"
        assert "text" not in body
        assert [m["role"] for m in body["input"]] == ["system", "assistant", "user"]
        assert body["input"][-1]["content"][:2] == [
            {"type": "input_text", "text": "查图中的天气"},
            {
                "type": "input_text",
                "text": "你看到的屏幕画面：\n"
                + (description or "你在 2026-09-08T22:00:00 看到的画面。"),
            },
        ]
        assert body["input"][-1]["content"][-1] == {
            "type": "input_image",
            "image_url": image["data_url"],
            "detail": "low",
        }
        diagnostic = client.request_payload(messages, enable_search=True)
        assert "base64" not in json.dumps(diagnostic)
        assert diagnostic["messages"][-1]["environmentImages"][0]["digest"] == "digest"
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (
            {
                "type": "error",
                "code": "ToolNotOpen",
                "message": "private request echoed here",
            },
            "ToolNotOpen",
        ),
        (
            {
                "type": "error",
                "error": {
                    "code": "ToolNotOpen",
                    "message": "private request echoed here",
                },
            },
            "ToolNotOpen",
        ),
        (
            {
                "type": "response.failed",
                "response": {"status": "failed", "error": {"code": "ToolNotOpen"}},
            },
            "ToolNotOpen",
        ),
        (
            {
                "type": "response.incomplete",
                "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            },
            "max_output_tokens",
        ),
        (completed(calls=False), "SearchNotExecuted"),
        (None, "IncompleteStream"),
    ],
)
async def test_tool_completed_is_not_response_success(failure, code):
    events = [{"type": "response.web_search_call.completed"}]
    if failure:
        events.append(failure)
    client, requests, wire = make_client(events)
    tracker = MessagePerformanceTracker()
    pipeline = AnswerGenerationPipeline(
        client, recall_service=None, recall_context_token_budget=100, enable_search=True
    )
    try:
        with pytest.raises(ArkSearchError) as caught:
            await pipeline._collect_web_search(
                MESSAGES, tracker, step_name="llm.web_search.context"
            )
        assert caught.value.code == code
        assert "private" not in str(caught.value)
        step = tracker.snapshot()[0]
        assert step.error == f"ArkSearchError:{code}"
        assert step.output_payload["search"]["status"] == "failed"
        assert step.output_payload["search"]["errorCode"] == code
        assert wire.closed
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_error_does_not_retry_or_fall_back_to_paid_search():
    client, requests, _ = make_client([], http_error=500)
    try:
        with pytest.raises(ArkSearchError, match="ToolNotOpen"):
            _ = [c async for c in client.stream_json(MESSAGES, force_search=True)]
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "error_code"),
    [
        ("杭州明天多云转晴，21℃~28℃。", None),
        ('{"searchContext":"晴天"}', None),
        ('{"searchContext":"晴天"', "InvalidSearchContext"),
        ('{"unexpected":"晴天"}', "InvalidSearchContext"),
        ("长" * 11_000, "SearchContextTooLarge"),
    ],
    ids=["plain-text", "json", "invalid-json", "wrong-schema", "too-large"],
)
async def test_completed_native_text_is_encoded_without_another_llm(text, error_code):
    client, requests, _ = make_client(
        [
            {"type": "response.output_text.delta", "delta": text},
            completed(),
        ]
    )
    try:
        if error_code:
            with pytest.raises(ArkSearchError, match=error_code):
                _ = [c async for c in client.stream_json(MESSAGES, force_search=True)]
        else:
            chunks = [c async for c in client.stream_json(MESSAGES, force_search=True)]
            result = json.loads("".join(c.content or "" for c in chunks))
            assert result == {"searchContext": "晴天" if text.startswith("{") else text}
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cancel_buffered_json_search_before_completion():
    client, requests, wire = make_client(
        [
            {"type": "response.output_text.delta", "delta": "尚未完成的联网结果"},
        ],
        block=True,
    )
    stream = client.stream_json(MESSAGES, force_search=True)
    try:
        task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(wire.waiting.wait(), timeout=1)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wire.closed
        assert len(requests) == 1
    finally:
        await stream.aclose()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_cancel_and_consumer_close_release_underlying_response(cancel):
    client, requests, wire = make_client(
        [
            {"type": "response.output_text.delta", "delta": "partial"},
        ],
        block=True,
    )
    stream = client.stream(MESSAGES, force_search=True)
    try:
        assert (await anext(stream)).content == "partial"
        if cancel:
            task = asyncio.create_task(anext(stream))
            await asyncio.wait_for(wire.waiting.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await stream.aclose()
        assert wire.closed
        assert len(requests) == 1
    finally:
        await stream.aclose()
        await client.close()
