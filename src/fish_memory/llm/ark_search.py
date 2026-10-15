"""Ark's native search, behind the same cancellable stream as ordinary chat.

Only the ordinary search engine is enabled. Omitted ``sources`` is Ark's
search_engine default; adding moji/toutiao/douyin opts into separately billed
resources. Keyword/tool limits bound a request, not the account's monthly usage.
"""

import json
import re
from collections.abc import AsyncGenerator, Sequence
from typing import Any, cast
from urllib.parse import urlsplit

from openai import APIError, AsyncOpenAI

from fish_memory.answer.context import (
    ChatMessage,
    environment_image_caption,
    get_environment_images,
    user_attachment_parts,
    with_request_time_last,
)
from fish_memory.llm.streaming import ConnectionReusingAsyncStream
from fish_memory.llm.types import ChatStreamChunk
from fish_memory.performance.models import JsonValue


def is_ark_endpoint(base_url: str) -> bool:
    hostname = urlsplit(base_url).hostname or ""
    return re.fullmatch(r"ark\.[a-z0-9-]+\.volces\.com", hostname) is not None


def search_parameters(*, json_mode: bool, force_search: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tools": [{"type": "web_search", "max_keyword": 1, "limit": 5}],
        "max_tool_calls": 1,
        "tool_choice": "required" if force_search else "auto",
        "thinking": {"type": "disabled"},
        "store": False,
        "stream": True,
    }
    if json_mode:
        result["text"] = {"format": {"type": "json_object"}}
    return result


def _input_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in with_request_time_last(messages):
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": message["content"]}
        ]
        for kind, value in user_attachment_parts(message):
            content.append(
                {"type": "input_text", "text": value}
                if kind == "text"
                else {"type": "input_image", "image_url": value}
            )
        for image in get_environment_images(message):
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": environment_image_caption(image),
                    },
                    {
                        "type": "input_image",
                        "image_url": image["data_url"],
                        "detail": image.get("detail", "low"),
                    },
                ]
            )
        if reminder := message.get("response_reminder"):
            content.append({"type": "input_text", "text": reminder})
        if timestamp := message.get("request_time"):
            content.append({"type": "input_text", "text": timestamp})
        result.append({"role": message["role"], "content": content})
    return result


class ArkSearchError(RuntimeError):
    """Retain a provider code and compact evidence without logging request data."""

    def __init__(self, code: object, diagnostics: dict[str, JsonValue]) -> None:
        self.code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", code)
            else "SearchFailed"
        )
        self.diagnostics = {**diagnostics, "status": "failed", "errorCode": self.code}
        super().__init__(f"Ark web search failed ({self.code})")


async def stream_search(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: Sequence[ChatMessage],
    json_mode: bool,
    force_search: bool,
) -> AsyncGenerator[ChatStreamChunk, None]:
    parameters = search_parameters(json_mode=json_mode, force_search=force_search)
    # with_options shares the existing HTTP pool. Do not replay a billed search
    # automatically after an ambiguous network/provider failure.
    sdk = client.with_options(max_retries=0)
    parameters["extra_body"] = {"thinking": parameters.pop("thinking")}
    diagnostics: dict[str, JsonValue] = {"provider": "ark", "status": "pending"}
    completed = False
    text_parts: list[str] = []
    text_bytes = 0
    try:
        async with sdk.responses.with_streaming_response.create(
            model=model,
            input=cast(Any, _input_messages(messages)),
            **parameters,
        ) as response:
            events = await response.parse(
                to=ConnectionReusingAsyncStream[dict[str, Any]]
            )
            async for event in events:
                kind = event.get("type")
                if kind == "response.output_text.delta":
                    if text := event.get("delta"):
                        if json_mode:
                            text_bytes += len(text.encode())
                            if text_bytes > 32_000:
                                raise ArkSearchError(
                                    "SearchContextTooLarge", diagnostics
                                )
                            text_parts.append(text)
                        else:
                            yield ChatStreamChunk(content=text)
                elif kind in {
                    "response.web_search_call.in_progress",
                    "response.web_search_call.searching",
                    "response.web_search_call.completed",
                }:
                    # A tool-completed event alone is not a successful response:
                    # Ark can subsequently emit ToolNotOpen and response.failed.
                    diagnostics["toolStatus"] = kind.rsplit(".", 1)[-1]
                elif kind == "error":
                    raise ArkSearchError(event.get("code"), diagnostics)
                elif kind in {"response.failed", "response.incomplete"}:
                    result = event.get("response") or {}
                    detail = (
                        result.get("error") or result.get("incomplete_details") or {}
                    )
                    raise ArkSearchError(
                        detail.get("code") or detail.get("reason") or kind,
                        diagnostics,
                    )
                elif kind == "response.completed":
                    result = event.get("response") or {}
                    diagnostics.update(_completed_diagnostics(result))
                    if result.get("status") != "completed" or result.get("error"):
                        raise ArkSearchError("ResponseNotCompleted", diagnostics)
                    if force_search and not diagnostics["completedSearchCalls"]:
                        raise ArkSearchError("SearchNotExecuted", diagnostics)
                    usage = result.get("usage") or {}
                    content = None
                    if json_mode:
                        content, result_format = _search_context_json(
                            "".join(text_parts), diagnostics
                        )
                        diagnostics["resultFormat"] = result_format
                    completed = True
                    yield ChatStreamChunk(
                        content=content,
                        finish_reason="stop",
                        input_tokens=usage.get("input_tokens"),
                        output_tokens=usage.get("output_tokens"),
                        search_diagnostics=diagnostics,
                    )
            if not completed:
                raise ArkSearchError("IncompleteStream", diagnostics)
    except APIError as error:
        # Includes HTTP failures and SDK-raised SSE error objects. Do not copy
        # their free-form messages (which can echo input) into application logs.
        raise ArkSearchError(
            getattr(error, "code", None) or type(error).__name__, diagnostics
        ) from None


def _search_context_json(
    text: str, diagnostics: dict[str, JsonValue]
) -> tuple[str, str]:
    """Bridge native search text to the application's searchContext contract.

    Ark can return plain output_text despite text.format=json_object when tools
    are used. Its completed search result is already the required context; encode
    it locally instead of discarding it or asking another LLM to format it.
    This runs only after the tool and response have both completed successfully.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if text.lstrip().startswith(("{", "[")):
            raise ArkSearchError("InvalidSearchContext", diagnostics) from None
        return json.dumps({"searchContext": text}, ensure_ascii=False), "text"
    if (
        not isinstance(parsed, dict)
        or set(parsed) != {"searchContext"}
        or not isinstance(parsed["searchContext"], str)
    ):
        raise ArkSearchError("InvalidSearchContext", diagnostics)
    return text, "json"


def _completed_diagnostics(result: dict[str, Any]) -> dict[str, JsonValue]:
    calls: list[JsonValue] = []
    citations: list[JsonValue] = []
    seen_urls: set[str] = set()
    completed_calls = 0
    for item in result.get("output") or []:
        if item.get("type") == "web_search_call":
            status = item.get("status")
            if status == "completed":
                completed_calls += 1
            calls.append(
                {"status": status, "query": (item.get("action") or {}).get("query")}
            )
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                for annotation in part.get("annotations") or []:
                    url = annotation.get("url")
                    if (
                        annotation.get("type") != "url_citation"
                        or not isinstance(url, str)
                        or url in seen_urls
                    ):
                        continue
                    seen_urls.add(url)
                    citations.append({"url": url, "title": annotation.get("title")})
    usage = result.get("usage") or {}
    return {
        "status": result.get("status"),
        "completedSearchCalls": completed_calls,
        "calls": calls,
        "citations": citations,
        "toolUsage": usage.get("tool_usage"),
        "toolUsageDetails": usage.get("tool_usage_details"),
    }
