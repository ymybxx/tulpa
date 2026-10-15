"""Tap HTTP bodies without consuming, replaying, or changing provider streams."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from openai import DefaultAsyncHttpxClient

from fish_memory.llm.logs import CallJournal, CallLog


class LoggedHttpClient(DefaultAsyncHttpxClient):
    def __init__(self, journal: CallJournal, **kwargs: Any):
        super().__init__(**kwargs)
        self.journal = journal

    async def send(
        self, request: httpx.Request, *, stream: bool = False, **kwargs: Any
    ) -> httpx.Response:
        if request.method != "POST" or not request.url.path.rstrip("/").endswith(
            ("/chat/completions", "/responses")
        ):
            return await super().send(request, stream=stream, **kwargs)
        # Credentials and headers never enter the log. Request content is the
        # already-serialized JSON body that the SDK is actually sending.
        url = str(request.url.copy_with(query=None, username=None, password=None))
        call = self.journal.start(request.content, url)
        try:
            response = await super().send(request, stream=True, **kwargs)
            if response.is_stream_consumed:
                call.finish(
                    [response.content],
                    response.headers.get("content-type", ""),
                    response.status_code,
                )
                return response
            assert isinstance(response.stream, httpx.AsyncByteStream)
            response.stream = CapturedStream(response.stream, call, response)
            if not stream:
                try:
                    await response.aread()
                except BaseException:
                    await response.aclose()
                    raise
            return response
        except BaseException as error:
            call.finish(
                [],
                "",
                None,
                type(error).__name__,
                isinstance(error, asyncio.CancelledError),
            )
            raise


class CapturedStream(httpx.AsyncByteStream):
    def __init__(
        self, source: httpx.AsyncByteStream, call: CallLog, response: httpx.Response
    ):
        self.source, self.call, self.response = source, call, response
        self.parts: list[bytes] = []
        self.captured_bytes = 0
        self.truncated = False

    def finish(self, error: str | None = None, cancelled: bool = False) -> None:
        self.call.finish(
            self.parts,
            self.response.headers.get("content-type", ""),
            self.response.status_code,
            error,
            cancelled,
            self.response.headers.get("content-encoding", ""),
            self.truncated,
        )

    async def __aiter__(self):
        try:
            async for part in self.source:
                remaining = 16 * 1024 * 1024 - self.captured_bytes
                if remaining > 0:
                    self.parts.append(part[:remaining])
                    self.captured_bytes += min(remaining, len(part))
                self.truncated = self.truncated or len(part) > remaining
                yield part
        except BaseException as error:
            self.finish(
                type(error).__name__,
                isinstance(error, (asyncio.CancelledError, GeneratorExit)),
            )
            raise
        else:
            self.finish()

    async def aclose(self) -> None:
        try:
            await self.source.aclose()
        finally:
            self.finish(cancelled=True)
