from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest

import fish_memory.llm.client as llm_client_module
from fish_memory.answer.context import ChatMessage
from fish_memory.context_descriptions import SCREEN_INPUT_PRESENT
from fish_memory.llm.client import BailianChatClient


class FakeStream:
    def __init__(self, chunks: list[Any], *, cancel_after: int | None = None) -> None:
        self.chunks = chunks
        self.cancel_after = cancel_after
        self.index = 0

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> Any:
        if self.cancel_after is not None and self.index == self.cancel_after:
            raise asyncio.CancelledError
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


class FakeStreamingResponse:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream

    async def parse(self, *, to: Any) -> FakeStream:
        return self.stream


class StreamingCompletions:
    @property
    def with_streaming_response(self) -> Any:
        return SimpleNamespace(create=self.create_stream)

    @asynccontextmanager
    async def create_stream(
        self, **kwargs: Any
    ) -> AsyncIterator[FakeStreamingResponse]:
        yield FakeStreamingResponse(await self.create(**kwargs))

    async def create(self, **kwargs: Any) -> Any:
        raise NotImplementedError


class FakeCompletions(StreamingCompletions):
    def __init__(self, stream: FakeStream, completion: Any | None = None) -> None:
        self.stream = stream
        self.completion = completion
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs["stream"] is False:
            return self.completion
        return self.stream


class FakeOpenAI:
    def __init__(self, stream: FakeStream, completion: Any | None = None) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(stream, completion))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_json_sampling_is_per_request_and_matches_diagnostic_payload():
    sdk = FakeOpenAI(FakeStream([]))
    client = BailianChatClient(
        api_key="unused",
        base_url="https://example.invalid/v1",
        model="test",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]
    _ = [
        part async for part in client.stream_json(messages, temperature=1.0, top_p=0.9)
    ]
    _ = [part async for part in client.stream_json(messages)]
    actual, ordinary = sdk.chat.completions.calls
    assert actual["temperature"] == 1.0 and actual["top_p"] == 0.9
    assert "temperature" not in ordinary and "top_p" not in ordinary
    payload = client.json_stream_request_payload(messages, temperature=1.0, top_p=0.9)
    assert payload["parameters"]["temperature"] == actual["temperature"]
    assert payload["parameters"]["top_p"] == actual["top_p"]
    await client.close()


@pytest.mark.asyncio
async def test_explicit_summary_budget_reaches_provider_without_capping_other_calls():
    sdk = FakeOpenAI(FakeStream([]))
    client = BailianChatClient(
        api_key="unused",
        base_url="https://example.invalid/v1",
        model="test",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "history"}]
    _ = [part async for part in client.stream(messages, max_output_tokens=2_000)]
    _ = [part async for part in client.stream(messages)]
    assert sdk.chat.completions.calls[0]["max_tokens"] == 2_000
    assert "max_tokens" not in sdk.chat.completions.calls[1]
    await client.close()


@pytest.mark.asyncio
async def test_json_output_budget_is_local_and_visible_in_request_diagnostics():
    sdk = FakeOpenAI(FakeStream([]))
    client = BailianChatClient(
        api_key="unused",
        base_url="https://example.invalid/v1",
        model="test",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "five thoughts"}]
    _ = [part async for part in client.stream_json(messages, max_output_tokens=1000)]
    _ = [part async for part in client.stream_json(messages)]
    assert sdk.chat.completions.calls[0]["max_tokens"] == 1000
    assert "max_tokens" not in sdk.chat.completions.calls[1]
    assert (
        client.json_stream_request_payload(messages, max_output_tokens=1000)[
            "parameters"
        ]["max_tokens"]
        == 1000
    )
    await client.close()


def chunk(
    content: str | None,
    *,
    with_choice: bool = True,
    finish_reason: str | None = None,
    usage: Any = None,
) -> Any:
    choices = [
        SimpleNamespace(
            delta=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )
    ]
    if not with_choice:
        choices = []
    return SimpleNamespace(choices=choices, usage=usage)


def completion(content: str | None, *, with_choice: bool = True) -> Any:
    choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
    if not with_choice:
        choices = []
    return SimpleNamespace(
        choices=choices,
        usage=SimpleNamespace(prompt_tokens=21, completion_tokens=8),
    )


@pytest.mark.asyncio
async def test_streams_content_finish_reason_and_usage() -> None:
    sdk = FakeOpenAI(
        FakeStream(
            [
                chunk(None),
                chunk(""),
                chunk("你"),
                chunk("好🐟", finish_reason="stop"),
                chunk(
                    None,
                    with_choice=False,
                    usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
                ),
            ]
        )
    )
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    result = [event async for event in client.stream(messages)]

    assert [event.content for event in result] == ["你", "好🐟", None]
    assert result[1].finish_reason == "stop"
    assert result[2].input_tokens == 12
    assert result[2].output_tokens == 3
    assert sdk.chat.completions.calls == [
        {
            "model": "qwen-plus",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"enable_thinking": False},
        }
    ]


@pytest.mark.asyncio
async def test_streams_structured_json_without_thinking() -> None:
    sdk = FakeOpenAI(FakeStream([chunk('{"recallDepth":0,"answer":"你好"}')]))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    result = [event async for event in client.stream_json(messages)]

    assert [event.content for event in result] == ['{"recallDepth":0,"answer":"你好"}']
    assert sdk.chat.completions.calls == [
        {
            "model": "qwen-plus",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False},
        }
    ]


@pytest.mark.asyncio
async def test_schema_response_is_opt_in_and_reported_in_request() -> None:
    sdk = FakeOpenAI(FakeStream([chunk('{"s":1,"m":"neutral","a":"你好"}')]))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="test",
        client=cast(Any, sdk),
    )
    schema = {
        "type": "object",
        "properties": {"s": {"type": "integer"}},
        "required": ["s"],
    }
    messages: list[ChatMessage] = [{"role": "user", "content": "synthetic event"}]
    result = [
        item async for item in client.stream_json(messages, response_schema=schema)
    ]
    assert result[0].content
    sent = sdk.chat.completions.calls[0]["response_format"]
    assert sent == {
        "type": "json_schema",
        "json_schema": {"name": "fish_response", "strict": True, "schema": schema},
    }
    assert (
        client.json_stream_request_payload(messages, response_schema=schema)[
            "parameters"
        ]["response_format"]
        == sent
    )
    assert client.json_stream_request_payload(messages)["parameters"][
        "response_format"
    ] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_explicit_unsupported_schema_falls_back_once_per_client() -> None:
    class Unsupported(RuntimeError):
        status_code = 400
        body = {"message": "response_format json_schema is not supported by this model"}

    class Completions(StreamingCompletions):
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["response_format"]["type"] == "json_schema":
                raise Unsupported()
            return FakeStream([chunk('{"s":0}')])

    completions = Completions()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="test",
        client=cast(Any, sdk),
    )
    for _ in range(2):
        result = [
            item
            async for item in client.stream_json([], response_schema={"type": "object"})
        ]
        assert result[0].content == '{"s":0}'
    assert [call["response_format"]["type"] for call in completions.calls] == [
        "json_schema",
        "json_object",
        "json_object",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401])
async def test_schema_fallback_never_retries_auth_or_partial_delivery(status) -> None:
    class ProviderError(RuntimeError):
        status_code = status
        body = {"message": "json_schema is not supported"}

    class FailingStream(FakeStream):
        async def __anext__(self):
            if self.index:
                raise ProviderError()
            return await super().__anext__()

    class Completions(StreamingCompletions):
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if status == 401:
                raise ProviderError()
            return FailingStream([chunk('{"s":1,')])

    completions = Completions()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="test",
        client=cast(Any, sdk),
    )
    with pytest.raises(ProviderError):
        _ = [
            item
            async for item in client.stream_json([], response_schema={"type": "object"})
        ]
    assert completions.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("description", [None, "最近5秒的合成画面，最后一格最近。"])
async def test_environment_image_reaches_provider_but_not_diagnostic_payload(
    description: str | None,
) -> None:
    sdk = FakeOpenAI(FakeStream([chunk('{"m":"neutral","a":"ok","r":0,"e":0}')]))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="vision-model",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": "what is visible?",
            "environment_image": {
                "data_url": "data:image/jpeg;base64,secret-frame",
                "digest": "a" * 64,
                "byte_count": 9,
                "width": 640,
                "height": 360,
                "observed_at": "2026-09-02T07:20:00Z",
            },
        }
    ]
    if description:
        messages[0]["environment_image"]["description"] = description

    _ = [event async for event in client.stream_json(messages)]

    provider_message = sdk.chat.completions.calls[0]["messages"][0]
    assert provider_message["content"][0] == {
        "type": "text",
        "text": "what is visible?",
    }
    assert provider_message["content"][1] == {
        "type": "text",
        "text": "你看到的屏幕画面：\n"
        + (description or "你在 2026-09-02T07:20:00Z 看到的画面。"),
    }
    assert provider_message["content"][2] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,secret-frame",
            "detail": "low",
        },
    }
    provider_payload = json.dumps(sdk.chat.completions.calls[0])
    assert all(
        field not in provider_payload
        for field in ("environmentImage", "digest", "byteCount", "width", "height")
    )
    diagnostic = client.json_stream_request_payload(messages)
    serialized = json.dumps(diagnostic)
    assert "secret-frame" not in serialized
    assert "a" * 64 in serialized
    assert diagnostic["messages"][0]["environmentImage"]["byteCount"] == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [True, False])
async def test_image_rejection_only_disables_support_when_explicit(explicit) -> None:
    class ImageUnsupportedError(RuntimeError):
        status_code = 400
        body = {
            "message": "image_url is unsupported by this model"
            if explicit
            else "invalid image payload"
        }

    class RejectingCompletions(StreamingCompletions):
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise ImageUnsupportedError(str(ImageUnsupportedError.body))
            return FakeStream([chunk('{"m":"neutral","a":"ok","r":0,"e":0}')])

    completions = RejectingCompletions()
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="misreported-vision-model",
        client=cast(Any, sdk),
    )
    client._image_input_capability = "supported"
    messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": "what is visible?",
            "environment_image": {
                "data_url": "data:image/jpeg;base64,secret-frame",
                "digest": "a" * 64,
                "byte_count": 9,
                "width": 640,
                "height": 360,
                "observed_at": "2026-09-02T07:20:00Z",
            },
        }
    ]

    result = [event async for event in client.stream_json(messages)]

    assert [event.content for event in result] == [
        '{"m":"neutral","a":"ok","r":0,"e":0}'
    ]
    assert len(completions.calls) == 2
    assert isinstance(completions.calls[0]["messages"][0]["content"], list)
    assert completions.calls[1]["messages"] == [
        {"role": "user", "content": "what is visible?"}
    ]
    assert client.image_input_capability == ("unsupported" if explicit else "unknown")
    assert client._image_capability_checked is explicit


def test_screen_input_state_describes_actual_attachments_not_historical_claims():
    messages: list[ChatMessage] = [
        {"role": "system", "content": "pet"},
        {"role": "assistant", "content": "以前的外部任务说无法看见屏幕。"},
        {
            "role": "user",
            "content": "现在呢？",
            "response_reminder": "最后再检查回答格式。",
            "request_time": "当前时间：2026-09-13T07:25:22Z",
            "environment_image": {
                "data_url": "data:image/jpeg;base64,private-frame",
                "digest": "a" * 64,
                "byte_count": 9,
                "width": 640,
                "height": 360,
                "observed_at": "2026-09-13T07:25:21Z",
            },
        },
    ]
    wire = llm_client_module._provider_messages(messages)
    diagnostic = llm_client_module._diagnostic_messages(messages)
    screen_context = wire[-1]["content"][-3]["text"]
    assert screen_context == SCREEN_INPUT_PRESENT
    assert "2026-09-13T07:25:21Z" not in screen_context
    assert wire[0]["content"] == "pet"
    assert wire[-1]["content"][-2]["text"] == messages[-1]["response_reminder"]
    assert wire[-1]["content"][-1]["text"] == messages[-1]["request_time"]
    suffix = messages[-1]["response_reminder"] + "\n\n" + messages[-1]["request_time"]
    assert diagnostic[-1]["content"].endswith(suffix)
    assert diagnostic[0] == wire[0]
    assert "private-frame" not in str(diagnostic)
    assert messages[0]["content"] == "pet"
    # A compatibility retry that actually removes the image must also remove
    # the derived presence claim, without mutating the original request.
    retry = llm_client_module._provider_messages(
        llm_client_module._without_environment_images(messages)
    )
    assert retry[0]["content"] == "pet"
    assert SCREEN_INPUT_PRESENT not in str(retry)
    assert retry[-1]["content"].endswith(suffix)
    assert "response_reminder" not in str(wire) and "response_reminder" not in str(
        retry
    )


@pytest.mark.asyncio
async def test_nonretryable_image_error_never_exposes_provider_body() -> None:
    class EchoingProviderError(RuntimeError):
        status_code = 500
        body = {"echo": "data:image/jpeg;base64,secret-frame"}

    class FailingCompletions(StreamingCompletions):
        async def create(self, **_kwargs: Any) -> Any:
            raise EchoingProviderError("data:image/jpeg;base64,secret-frame")

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="vision-model",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [
        {
            "role": "user",
            "content": "what is visible?",
            "environment_image": {
                "data_url": "data:image/jpeg;base64,secret-frame",
                "digest": "a" * 64,
                "byte_count": 9,
                "width": 640,
                "height": 360,
                "observed_at": "2026-09-02T07:20:00Z",
            },
        }
    ]

    with pytest.raises(llm_client_module.EnvironmentImageRequestError) as raised:
        _ = [event async for event in client.stream_json(messages)]

    assert "secret-frame" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_image_capability_probe_is_cached_only_after_visual_challenge_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_client_module,
        "_image_capability_probe",
        lambda: ("data:image/png;base64,visual-challenge", "RED,BLUE"),
    )
    sdk = FakeOpenAI(FakeStream([]), completion(" RED, BLUE\n"))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="vision-model",
        client=cast(Any, sdk),
    )

    assert await client.probe_image_input_capability() == "supported"
    assert await client.probe_image_input_capability() == "supported"
    assert len(sdk.chat.completions.calls) == 1
    call = sdk.chat.completions.calls[0]
    assert call["stream"] is False
    assert call["max_tokens"] == 32
    assert call["messages"][0]["content"][1]["type"] == "image_url"
    assert call["messages"][0]["content"][1]["image_url"]["url"] == (
        "data:image/png;base64,visual-challenge"
    )


@pytest.mark.asyncio
async def test_image_capability_probe_does_not_trust_a_gateway_that_ignores_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_client_module,
        "_image_capability_probe",
        lambda: ("data:image/png;base64,visual-challenge", "RED,BLUE"),
    )
    sdk = FakeOpenAI(FakeStream([]), completion("OK"))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="text-only-gateway",
        client=cast(Any, sdk),
    )

    assert await client.probe_image_input_capability() == "unknown"
    assert await client.probe_image_input_capability() == "unknown"
    assert len(sdk.chat.completions.calls) == 1
    assert client._image_capability_last_probe_at is not None
    client._image_capability_last_probe_at -= 61
    assert await client.probe_image_input_capability() == "unknown"
    assert len(sdk.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_ark_accepting_images_is_not_disabled_by_color_recognition_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_client_module,
        "_image_capability_probe",
        lambda: ("data:image/png;base64,visual-challenge", "RED,BLUE"),
    )
    sdk = FakeOpenAI(FakeStream([]), completion("BLUE, RED"))
    client = BailianChatClient(
        api_key="key",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        model="doubao-seed-character-260628",
        client=cast(Any, sdk),
    )
    assert await client.probe_image_input_capability() == "supported"
    assert await client.probe_image_input_capability() == "supported"
    assert len(sdk.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_structured_answer_stream_can_enable_web_search() -> None:
    sdk = FakeOpenAI(FakeStream([chunk('{"recallDepth":0,"answer":"你好"}')]))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    result = [event async for event in client.stream_json(messages, enable_search=True)]

    assert [event.content for event in result] == ['{"recallDepth":0,"answer":"你好"}']
    assert sdk.chat.completions.calls == [
        {
            "model": "qwen-plus",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"},
            "extra_body": {
                "enable_thinking": False,
                "enable_search": True,
            },
        }
    ]


def test_json_stream_payload_contains_contract_without_credentials() -> None:
    client = BailianChatClient(
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, FakeOpenAI(FakeStream([]))),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    payload = client.json_stream_request_payload(messages)

    assert payload == {
        "model": "qwen-plus",
        "messages": messages,
        "parameters": {
            "enable_thinking": False,
            "include_usage": True,
            "response_format": {"type": "json_object"},
            "stream": True,
        },
    }
    assert "secret" not in json.dumps(payload)


def test_json_stream_payload_can_enable_web_search() -> None:
    client = BailianChatClient(
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, FakeOpenAI(FakeStream([]))),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    payload = client.json_stream_request_payload(
        messages,
        enable_search=True,
    )

    assert payload["parameters"]["enable_search"] is True
    assert "secret" not in json.dumps(payload)


def test_json_stream_payload_can_force_web_search() -> None:
    client = BailianChatClient(
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, FakeOpenAI(FakeStream([]))),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    payload = client.json_stream_request_payload(
        messages,
        force_search=True,
    )

    assert payload["parameters"]["enable_search"] is True
    assert payload["parameters"]["search_options"] == {"forced_search": True}
    assert "secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_completes_structured_json_without_thinking() -> None:
    sdk = FakeOpenAI(FakeStream([]), completion('{"facts":[]}'))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, sdk),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    result = await client.complete_json(messages)

    assert result.content == '{"facts":[]}'
    assert result.input_tokens == 21
    assert result.output_tokens == 8
    assert sdk.chat.completions.calls[-1] == {
        "model": "qwen-plus",
        "messages": messages,
        "stream": False,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "extra_body": {"enable_thinking": False},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [completion(None), completion("   "), completion(None, with_choice=False)],
)
async def test_rejects_structured_completion_without_content(response: Any) -> None:
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, FakeOpenAI(FakeStream([]), response)),
    )

    with pytest.raises(ValueError, match="content"):
        await client.complete_json([])


def test_json_request_payload_describes_non_streaming_request() -> None:
    client = BailianChatClient(
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, FakeOpenAI(FakeStream([]))),
    )
    messages: list[ChatMessage] = [{"role": "user", "content": "hello"}]

    payload = client.json_request_payload(messages)

    assert payload == {
        "model": "qwen-plus",
        "messages": messages,
        "parameters": {
            "enable_thinking": False,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "stream": False,
        },
    }
    assert "secret" not in json.dumps(payload)


def test_request_payload_contains_full_prompt_without_credentials() -> None:
    client = BailianChatClient(
        api_key="secret",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, FakeOpenAI(FakeStream([]))),
    )
    messages: list[ChatMessage] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]

    payload = client.request_payload(messages)

    assert payload == {
        "model": "qwen-plus",
        "messages": messages,
        "parameters": {
            "enable_thinking": False,
            "include_usage": True,
            "stream": True,
        },
    }
    assert "secret" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_propagates_stream_cancellation() -> None:
    sdk = FakeOpenAI(FakeStream([chunk("partial")], cancel_after=0))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, sdk),
    )

    with pytest.raises(asyncio.CancelledError):
        _ = [text async for text in client.stream([])]


@pytest.mark.asyncio
async def test_closes_underlying_openai_client() -> None:
    sdk = FakeOpenAI(FakeStream([]))
    client = BailianChatClient(
        api_key="key",
        base_url="https://example.invalid/v1",
        model="qwen-plus",
        client=cast(Any, sdk),
    )

    await client.close()

    assert sdk.closed is True


@pytest.mark.parametrize(
    ("base_url", "sent"),
    [
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", True),
        ("https://ark.cn-beijing.volces.com/api/v3", False),
        ("https://api.openai.com/v1", False),
        ("http://127.0.0.1:8080/v1", False),
    ],
)
def test_repetition_penalty_is_only_sent_where_it_is_accepted(base_url, sent) -> None:
    # Not an OpenAI parameter; strict OpenAI-compatible servers reject it, and
    # Ark accepts it without a measurable effect.
    client = BailianChatClient(api_key="sk-test", base_url=base_url, model="m")
    request = client._request_kwargs(
        [{"role": "user", "content": "你好"}], json_mode=True, repetition_penalty=1.1
    )
    assert (request["extra_body"].get("repetition_penalty") == 1.1) is sent
    plain = client._request_kwargs(
        [{"role": "user", "content": "你好"}], json_mode=True
    )
    assert "repetition_penalty" not in plain["extra_body"]
