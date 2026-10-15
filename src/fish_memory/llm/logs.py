"""Local, bounded asynchronous recording of actual provider requests."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from typing import Any
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)
_call_type: ContextVar[str] = ContextVar("llm_call_type", default="")
# Why a call was made when its type alone cannot tell, e.g. TA speaking up
# on its own rather than replying; shown as token usage by purpose.
_purpose: ContextVar[str] = ContextVar("llm_call_purpose", default="")


@contextmanager
def purpose_scope(name: str) -> Iterator[None]:
    token = _purpose.set(name)
    try:
        yield
    finally:
        _purpose.reset(token)


@contextmanager
def call_scope(name: str, *, fallback: bool = False) -> Iterator[None]:
    token = _call_type.set(_call_type.get() if fallback and _call_type.get() else name)
    try:
        yield
    finally:
        _call_type.reset(token)


@asynccontextmanager
async def async_call_scope(name: str):
    with call_scope(name):
        yield


def logged_call(name: str):
    def decorate(function):
        @wraps(function)
        async def invoke(*args, **kwargs):
            with call_scope(name):
                return await function(*args, **kwargs)

        return invoke

    return decorate


def response_view(raw: str, content_type: str) -> dict[str, Any]:
    """Keep the wire body as well as readable, unescaped output text."""
    if "text/event-stream" not in content_type:
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        return {
            "body": value,
            "raw": raw,
            "error": value.get("error") if isinstance(value, dict) else None,
        }
    text, reasoning, events = [], [], []
    usage = error = None
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")
        )
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except ValueError:
            events.append(data)
            continue
        events.append(event)
        if not isinstance(event, dict):
            continue
        choices = event.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                text.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning.append(delta["reasoning_content"])
        if event.get("type") == "response.output_text.delta" and isinstance(
            event.get("delta"), str
        ):
            text.append(event["delta"])
        response = event.get("response")
        if event.get("error"):
            error = event["error"]
        elif event.get("type") in {"error", "response.failed"}:
            error = (
                (response.get("error") if isinstance(response, dict) else None)
                or event.get("message")
                or event["type"]
            )
        usage = (
            event.get("usage")
            or (response.get("usage") if isinstance(response, dict) else None)
            or usage
        )
    return {
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "events": events,
        "raw": raw,
        "error": error,
    }


class CallLog:
    def __init__(self, journal: CallJournal, body: bytes, url: str):
        self.journal = journal
        self.started = time.monotonic()
        self.finished = False
        try:
            request = json.loads(body)
        except ValueError:
            request = body.decode("utf-8", errors="replace")
        self.value = {
            "id": str(uuid4()),
            "call_type": _call_type.get() or "other",
            "model": str(request.get("model") or "")
            if isinstance(request, dict)
            else "",
            "request_url": url,
            "request": request,
            "response": None,
            "started_at": datetime.now(UTC).replace(microsecond=0),
            "ended_at": None,
            "duration_ms": None,
            "status": "running",
            "http_status": None,
            "error": None,
            "source": "host",
            "purpose": _purpose.get() or None,
        }
        journal.submit(dict(self.value))

    def finish(
        self,
        parts: list[bytes],
        content_type: str,
        status: int | None,
        error: str | None = None,
        cancelled: bool = False,
        encoding: str = "",
        truncated: bool = False,
    ) -> None:
        if self.finished:
            return
        self.finished = True
        self.value.update(
            ended_at=datetime.now(UTC).replace(microsecond=0),
            duration_ms=round((time.monotonic() - self.started) * 1000),
            status="cancelled"
            if cancelled
            else "failed"
            if error or (status is not None and status >= 400)
            else "completed",
            http_status=status,
            error=error,
        )
        # Decode and persist in the writer, after releasing the generation path.
        self.journal.submit(
            {
                **self.value,
                "wire_parts": parts,
                "content_type": content_type,
                "content_encoding": encoding,
                "capture_truncated": truncated,
            }
        )


class CallJournal:
    def __init__(self, store: Any):
        self.store = store
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=512)
        self.dropped = 0
        self._worker = asyncio.create_task(self._write())

    def start(self, body: bytes, url: str) -> CallLog:
        return CallLog(self, body, url)

    def submit(self, value: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(value)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning("LLM log writer is full; %s records missed", self.dropped)

    async def _write(self) -> None:
        while True:
            value = await self.queue.get()
            try:
                if value is None:
                    return
                if "wire_parts" in value:
                    body = b"".join(value.pop("wire_parts"))
                    encoding = value.pop("content_encoding")
                    decode_error = None
                    try:
                        raw = httpx.Response(
                            200,
                            headers={"content-encoding": encoding} if encoding else {},
                            content=body,
                        ).text
                    except httpx.DecodingError as error:
                        raw = body.decode("utf-8", errors="replace")
                        decode_error = type(error).__name__
                    content_type = value.pop("content_type")
                    value["response"] = response_view(raw, content_type)
                    value["response"]["truncated"] = value.pop("capture_truncated")
                    if decode_error:
                        value["response"]["decode_error"] = decode_error
                    # The SDK can abandon a slow HTTP tail after [DONE]. The
                    # provider has already completed its answer in that case.
                    if (
                        value["status"] == "cancelled"
                        and "text/event-stream" in content_type
                        and any(
                            line.strip() == "data: [DONE]" for line in raw.splitlines()
                        )
                    ):
                        value["status"] = "completed"
                        value["error"] = None
                    if provider_error := value["response"].get("error"):
                        value["status"] = "failed"
                        value["error"] = (
                            json.dumps(provider_error, ensure_ascii=False)
                            if isinstance(provider_error, (dict, list))
                            else str(provider_error)
                        )[:2000]
                await self.store.save(value)
            except Exception:
                self.dropped += 1
                logger.exception("Could not save LLM call log")
            finally:
                self.queue.task_done()

    async def close(self) -> None:
        await self.queue.put(None)
        await self._worker
