import asyncio
import base64
import secrets
import struct
import zlib
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from fish_memory.answer.context import (
    ChatMessage,
    environment_image_caption,
    get_environment_images,
    user_attachment_parts,
    with_request_time_last,
)
from fish_memory.context_descriptions import (
    SCREEN_INPUT_PRESENT,
)
from fish_memory.llm.ark_search import (
    ArkSearchError,
    is_ark_endpoint,
    search_parameters,
    stream_search,
)
from fish_memory.llm.streaming import ConnectionReusingAsyncStream
from fish_memory.llm.types import ChatStreamChunk as ChatStreamChunk
from fish_memory.performance.models import JsonValue

_IMAGE_PROBE_COLORS = (
    ("RED", (255, 0, 0)),
    ("GREEN", (0, 192, 0)),
    ("BLUE", (0, 96, 255)),
    ("YELLOW", (255, 224, 0)),
    ("MAGENTA", (224, 0, 224)),
    ("CYAN", (0, 208, 208)),
)
_IMAGE_CAPABILITY_RETRY_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class JsonCompletion:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class EnvironmentImageRequestError(RuntimeError):
    """Provider image failure sanitized before it reaches application logs."""


class JsonCompletionClient(Protocol):
    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion: ...

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue: ...


class ChatClient(Protocol):
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[ChatStreamChunk]: ...

    def request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue: ...

    def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        repetition_penalty: float | None = None,
    ) -> AsyncIterator[ChatStreamChunk]: ...

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> JsonValue: ...

    async def close(self) -> None: ...


class BailianChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        client: AsyncOpenAI | None = None,
        call_journal: Any = None,
    ) -> None:
        self._model = model
        self._ark_endpoint = is_ark_endpoint(base_url)
        # Not an OpenAI parameter, and strict OpenAI-compatible servers reject
        # it. Only 阿里云百炼 was measured to apply it (Ark accepts it without
        # any measurable effect), so it is sent there alone.
        self._repetition_penalty_supported = _is_bailian(base_url)
        if call_journal is not None and client is None:
            from fish_memory.llm.log_transport import LoggedHttpClient

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=LoggedHttpClient(call_journal),
            )
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._image_input_capability: Literal["supported", "unsupported", "unknown"] = (
            "unknown"
        )
        self._image_capability_checked = False
        self._image_capability_lock = asyncio.Lock()
        self._image_capability_last_probe_at: float | None = None
        self._json_schema_supported = True

    @property
    def image_input_capability(self) -> Literal["supported", "unsupported", "unknown"]:
        return self._image_input_capability

    async def probe_image_input_capability(
        self,
    ) -> Literal["supported", "unsupported", "unknown"]:
        if self._image_capability_checked:
            return self._image_input_capability
        async with self._image_capability_lock:
            if self._image_capability_checked:
                return self._image_input_capability
            now = asyncio.get_running_loop().time()
            if (
                self._image_capability_last_probe_at is not None
                and now - self._image_capability_last_probe_at
                < _IMAGE_CAPABILITY_RETRY_SECONDS
            ):
                return self._image_input_capability
            self._image_capability_last_probe_at = now
            capability: Literal["supported", "unsupported", "unknown"] = "unknown"
            data_url, expected = _image_capability_probe()
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=cast(
                        Any,
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "Read the six solid-color vertical stripes "
                                            "from left to right. Reply only with six "
                                            "uppercase color names, separated by "
                                            "commas. "
                                            "Possible names: RED, GREEN, BLUE, YELLOW, "
                                            "MAGENTA, CYAN."
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": data_url,
                                            "detail": "low",
                                        },
                                    },
                                ],
                            }
                        ],
                    ),
                    stream=False,
                    max_tokens=32,
                    extra_body={"enable_thinking": False},
                )
                choices = getattr(response, "choices", None)
                content = (
                    getattr(getattr(choices[0], "message", None), "content", None)
                    if choices
                    else None
                )
                if (
                    isinstance(content, str)
                    and content.strip()
                    and (
                        self._ark_endpoint
                        or _normalized_image_probe_answer(content) == expected
                    )
                ):
                    # For Ark's direct API, a valid image request establishes
                    # input compatibility. A wrong colour order is model quality,
                    # not a reason to withhold the screen on future turns.
                    # Unknown gateways still need the existing visual challenge.
                    capability = "supported"
            except Exception as error:
                if _is_explicit_image_unsupported_error(error):
                    capability = "unsupported"
            self._image_input_capability = capability
            self._image_capability_checked = capability != "unknown"
            return capability

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        async with aclosing(
            self._stream(
                messages,
                json_mode=False,
                enable_search=enable_search,
                force_search=force_search,
                max_output_tokens=max_output_tokens,
            )
        ) as stream:
            async for chunk in stream:
                yield chunk

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        repetition_penalty: float | None = None,
    ) -> AsyncIterator[ChatStreamChunk]:
        async with aclosing(
            self._stream(
                messages,
                json_mode=True,
                enable_search=enable_search,
                force_search=force_search,
                response_schema=response_schema,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                repetition_penalty=repetition_penalty,
            )
        ) as stream:
            async for chunk in stream:
                yield chunk

    async def _stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        enable_search: bool = False,
        force_search: bool = False,
        max_output_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> AsyncGenerator[ChatStreamChunk, None]:
        request_messages: Sequence[ChatMessage] = messages
        may_retry_without_image = _has_environment_image(messages)
        while True:
            emitted = False
            try:
                if self._ark_endpoint and (enable_search or force_search):
                    async with aclosing(
                        stream_search(
                            self._client,
                            model=self._model,
                            messages=request_messages,
                            json_mode=json_mode,
                            force_search=force_search,
                        )
                    ) as search:
                        async for chunk in search:
                            emitted = True
                            yield chunk
                    return
                async with self._client.chat.completions.with_streaming_response.create(
                    **self._request_kwargs(
                        request_messages,
                        json_mode=json_mode,
                        enable_search=enable_search,
                        force_search=force_search,
                        response_schema=response_schema,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                ) as response:
                    stream = await response.parse(
                        to=ConnectionReusingAsyncStream[ChatCompletionChunk]
                    )
                    async for chunk in stream:
                        content: str | None = None
                        finish_reason: str | None = None
                        if chunk.choices:
                            content = chunk.choices[0].delta.content
                            finish_reason = chunk.choices[0].finish_reason

                        usage = getattr(chunk, "usage", None)
                        input_tokens = getattr(usage, "prompt_tokens", None)
                        output_tokens = getattr(usage, "completion_tokens", None)
                        if content or finish_reason is not None or usage is not None:
                            emitted = True
                            yield ChatStreamChunk(
                                content=content or None,
                                finish_reason=finish_reason,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                            )
                return
            except Exception as error:
                if isinstance(error, ArkSearchError):
                    raise
                if (
                    response_schema is not None
                    and self._json_schema_supported
                    and not emitted
                    and _is_json_schema_unsupported(error)
                ):
                    # Only opt-in callers probe this capability. Cache an explicit
                    # rejection; never retry a partially delivered response.
                    self._json_schema_supported = False
                    continue
                if (
                    may_retry_without_image
                    and not emitted
                    and _is_image_request_compatibility_error(error)
                ):
                    explicit = _is_explicit_image_unsupported_error(error)
                    self._image_input_capability = (
                        "unsupported" if explicit else "unknown"
                    )
                    self._image_capability_checked = explicit
                    request_messages = _without_environment_images(messages)
                    may_retry_without_image = False
                    continue
                if _has_environment_image(request_messages):
                    if _is_explicit_image_unsupported_error(error):
                        self._image_input_capability = "unsupported"
                        self._image_capability_checked = True
                    status = getattr(error, "status_code", None)
                    detail = (
                        f" (status={status})"
                        if isinstance(status, int) and not isinstance(status, bool)
                        else ""
                    )
                    raise EnvironmentImageRequestError(
                        f"Environment image request failed{detail}"
                    ) from None
                raise

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        response = await self._client.chat.completions.create(
            **self._json_request_kwargs(messages)
        )
        if not response.choices:
            raise ValueError("JSON completion response has no content")

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("JSON completion response has no content")

        usage = getattr(response, "usage", None)
        return JsonCompletion(
            content=content,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
        )

    def request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        if self._ark_endpoint and (enable_search or force_search):
            return self._search_request_payload(
                messages, json_mode=False, force_search=force_search
            )
        request = self._request_kwargs(
            messages,
            json_mode=False,
            enable_search=enable_search,
            force_search=force_search,
        )
        stream_options = cast(dict[str, bool], request["stream_options"])
        extra_body = cast(dict[str, Any], request["extra_body"])
        parameters: dict[str, JsonValue] = {
            "enable_thinking": extra_body["enable_thinking"],
            "include_usage": stream_options["include_usage"],
            "stream": cast(bool, request["stream"]),
        }
        if extra_body.get("enable_search") is True:
            parameters["enable_search"] = True
        if "search_options" in extra_body:
            parameters["search_options"] = cast(JsonValue, extra_body["search_options"])
        return {
            "model": cast(str, request["model"]),
            "messages": _diagnostic_messages(messages),
            "parameters": parameters,
        }

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> JsonValue:
        if self._ark_endpoint and (enable_search or force_search):
            return self._search_request_payload(
                messages, json_mode=True, force_search=force_search
            )
        request = self._request_kwargs(
            messages,
            json_mode=True,
            enable_search=enable_search,
            force_search=force_search,
            response_schema=response_schema,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        stream_options = cast(dict[str, bool], request["stream_options"])
        extra_body = cast(dict[str, Any], request["extra_body"])
        parameters: dict[str, JsonValue] = {
            "enable_thinking": extra_body["enable_thinking"],
            "include_usage": stream_options["include_usage"],
            "response_format": cast(
                JsonValue,
                request["response_format"],
            ),
            "stream": cast(bool, request["stream"]),
        }
        if extra_body.get("enable_search") is True:
            parameters["enable_search"] = True
        if "search_options" in extra_body:
            parameters["search_options"] = cast(JsonValue, extra_body["search_options"])
        for key in ("temperature", "top_p", "max_tokens"):
            if key in request:
                parameters[key] = request[key]
        return {
            "model": cast(str, request["model"]),
            "messages": _diagnostic_messages(messages),
            "parameters": parameters,
        }

    def _search_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        force_search: bool,
    ) -> JsonValue:
        return {
            "model": self._model,
            "protocol": "ark_responses",
            "messages": _diagnostic_messages(messages),
            "parameters": search_parameters(
                json_mode=json_mode, force_search=force_search
            ),
        }

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        request = self._json_request_kwargs(messages)
        extra_body = cast(dict[str, bool], request["extra_body"])
        return {
            "model": cast(str, request["model"]),
            "messages": _diagnostic_messages(messages),
            "parameters": {
                "enable_thinking": extra_body["enable_thinking"],
                "temperature": cast(float, request["temperature"]),
                "response_format": cast(JsonValue, request["response_format"]),
                "stream": cast(bool, request["stream"]),
            },
        }

    def _request_kwargs(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_mode: bool,
        enable_search: bool = False,
        force_search: bool = False,
        max_output_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> dict[str, Any]:
        extra_body: dict[str, Any] = {"enable_thinking": False}
        if repetition_penalty is not None and self._repetition_penalty_supported:
            extra_body["repetition_penalty"] = repetition_penalty
        if enable_search or force_search:
            extra_body["enable_search"] = True
        if force_search:
            extra_body["search_options"] = {"forced_search": True}

        request: dict[str, Any] = {
            "model": self._model,
            "messages": cast(Any, _provider_messages(messages)),
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": extra_body,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p
        if max_output_tokens is not None:
            if max_output_tokens < 1:
                raise ValueError("max_output_tokens must be positive")
            request["max_tokens"] = max_output_tokens
        if json_mode:
            request["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fish_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
                if response_schema is not None and self._json_schema_supported
                else {"type": "json_object"}
            )
        return request

    def _json_request_kwargs(
        self,
        messages: Sequence[ChatMessage],
    ) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": cast(Any, _provider_messages(messages)),
            "stream": False,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "extra_body": {"enable_thinking": False},
        }

    async def close(self) -> None:
        await self._client.close()


def _is_json_schema_unsupported(error: Exception) -> bool:
    if getattr(error, "status_code", None) not in {400, 422}:
        return False
    detail = str(getattr(error, "body", error)).lower()
    return "json_schema" in detail and any(
        marker in detail
        for marker in (
            "not supported",
            "does not support",
            "not support",
            "unsupported",
        )
    )


def _screen_input_context(messages: Sequence[ChatMessage]) -> str:
    if not any(get_environment_images(message) for message in messages):
        return ""
    return "\n\n" + SCREEN_INPUT_PRESENT


def _provider_messages(messages: Sequence[ChatMessage]) -> list[dict[str, object]]:
    messages = with_request_time_last(messages)
    result: list[dict[str, object]] = []
    screen_context = _screen_input_context(messages)
    last_image = max(
        (
            index
            for index, message in enumerate(messages)
            if get_environment_images(message)
        ),
        default=-1,
    )
    for index, message in enumerate(messages):
        images = get_environment_images(message)
        attached = user_attachment_parts(message)
        if not images and not attached:
            text_content = message["content"]
            if reminder := message.get("response_reminder"):
                text_content += "\n\n" + reminder
            if timestamp := message.get("request_time"):
                text_content += "\n\n" + timestamp
            result.append({"role": message["role"], "content": text_content})
            continue
        content: list[dict[str, object]] = [
            {"type": "text", "text": message["content"]}
        ]
        for kind, value in attached:
            content.append(
                {"type": "text", "text": value}
                if kind == "text"
                else {"type": "image_url", "image_url": {"url": value}}
            )
        for image in images:
            content.append(
                {
                    "type": "text",
                    "text": environment_image_caption(image),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image["data_url"],
                        "detail": image.get("detail", "low"),
                    },
                }
            )
        if index == last_image:
            content.append({"type": "text", "text": screen_context.strip()})
        if reminder := message.get("response_reminder"):
            content.append({"type": "text", "text": reminder})
        if timestamp := message.get("request_time"):
            content.append({"type": "text", "text": timestamp})
        result.append({"role": message["role"], "content": content})
    return result


def _diagnostic_messages(messages: Sequence[ChatMessage]) -> list[dict[str, object]]:
    messages = with_request_time_last(messages)
    result: list[dict[str, object]] = []
    screen_context = _screen_input_context(messages)
    last_image = max(
        (
            index
            for index, message in enumerate(messages)
            if get_environment_images(message)
        ),
        default=-1,
    )
    for index, message in enumerate(messages):
        content = message["content"]
        for kind, value in user_attachment_parts(message):
            # The log keeps file text but not picture bytes.
            content += "\n\n" + (value if kind == "text" else "[图片]")
        if index == last_image:
            content += screen_context
        if reminder := message.get("response_reminder"):
            content += "\n\n" + reminder
        if timestamp := message.get("request_time"):
            content += "\n\n" + timestamp
        value: dict[str, object] = {
            "role": message["role"],
            "content": content,
        }
        image = message.get("environment_image")
        if image is not None:
            value["environmentImage"] = {
                "digest": image["digest"],
                "byteCount": image["byte_count"],
                "width": image["width"],
                "height": image["height"],
                "observedAt": image["observed_at"],
            }
        images = message.get("environment_images")
        if images:
            value["environmentImages"] = [
                {
                    "digest": item["digest"],
                    "byteCount": item["byte_count"],
                    "width": item["width"],
                    "height": item["height"],
                    "observedAt": item["observed_at"],
                }
                for item in images
            ]
        result.append(value)
    return result


def _has_environment_image(messages: Sequence[ChatMessage]) -> bool:
    return any(get_environment_images(message) for message in messages)


def _without_environment_images(
    messages: Sequence[ChatMessage],
) -> list[ChatMessage]:
    result: list[ChatMessage] = []
    for message in messages:
        item: ChatMessage = {
            "role": message["role"],
            "content": message["content"],
        }
        if reminder := message.get("response_reminder"):
            item["response_reminder"] = reminder
        if timestamp := message.get("request_time"):
            item["request_time"] = timestamp
        result.append(item)
    return result


def _is_explicit_image_unsupported_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status not in {400, 404, 415, 422}:
        return False
    body = getattr(error, "body", None)
    text = f"{error} {body}".casefold()
    mentions_image = any(
        marker in text
        for marker in ("image", "vision", "multimodal", "image_url", "图片", "视觉")
    )
    explicitly_unsupported = any(
        marker in text
        for marker in (
            "not support",
            "unsupported",
            "does not accept",
            "text-only",
            "不支持",
            "仅支持文本",
        )
    )
    return mentions_image and explicitly_unsupported


def _is_image_request_compatibility_error(error: Exception) -> bool:
    if _is_explicit_image_unsupported_error(error):
        return True
    status = getattr(error, "status_code", None)
    return status in {400, 413, 415, 422}


def _image_capability_probe() -> tuple[str, str]:
    """Build a nonce-like visual challenge a text-only gateway cannot infer."""

    colors = secrets.SystemRandom().sample(
        list(_IMAGE_PROBE_COLORS),
        k=len(_IMAGE_PROBE_COLORS),
    )
    stripe_width = 20
    width = stripe_width * len(colors)
    height = 40
    row = b"".join(bytes(rgb) * stripe_width for _name, rgb in colors)
    raw = b"".join(b"\x00" + row for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    expected = ",".join(name for name, _rgb in colors)
    return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}", expected


def _normalized_image_probe_answer(value: str) -> str:
    return ",".join(part.strip().upper() for part in value.strip().split(","))


def _is_bailian(base_url: str) -> bool:
    """阿里云百炼 (DashScope) OpenAI-compatible endpoints."""
    hostname = urlsplit(base_url).hostname or ""
    return hostname == "dashscope.aliyuncs.com" or hostname.endswith(
        ".dashscope.aliyuncs.com"
    )
