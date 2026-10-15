"""Preserve HTTP keep-alive after a normally completed OpenAI-compatible SSE.

The SDK stops at [DONE], which can precede the HTTP chunk terminator. Reuse
requires consuming that terminator. Keep the SDK's event decoding and validation;
only its post-DONE behavior differs. The isolated SDK hook is covered with real
HTTP connection reuse and cancellation tests for dependency upgrades.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Generic, TypeVar, cast

from openai import AsyncStream
from openai._streaming import ServerSentEvent

logger = logging.getLogger(__name__)

_HTTP_EOF_DRAIN_TIMEOUT_SECONDS = 0.1
_T = TypeVar("_T")


@asynccontextmanager
async def closing_iterator(
    stream: AsyncIterator[_T],
) -> AsyncIterator[AsyncIterator[_T]]:
    """Close cancellable provider iterators without requiring it of test fakes."""
    try:
        yield stream
    finally:
        close = getattr(stream, "aclose", None)
        if callable(close):
            await close()


class ConnectionReusingAsyncStream(AsyncStream[_T], Generic[_T]):
    async def _iter_events(self) -> AsyncIterator[ServerSentEvent]:
        sdk_events = cast(AsyncGenerator[ServerSentEvent, None], super()._iter_events())
        async with aclosing(sdk_events) as events:
            async for event in events:
                if event.data.startswith("[DONE]"):
                    await _drain_http_tail(events)
                    return
                yield event


async def _drain_http_tail(events: AsyncIterator[ServerSentEvent]) -> None:
    try:
        async with asyncio.timeout(_HTTP_EOF_DRAIN_TIMEOUT_SECONDS):
            async for _ in events:
                pass
    except Exception as error:
        # DONE already completed the answer. A failed/slow optional drain only
        # forfeits this connection; it must not retry or fail that answer.
        # CancelledError/GeneratorExit are deliberately not caught or shielded.
        logger.debug("LLM HTTP tail drain abandoned: %s", type(error).__name__)
