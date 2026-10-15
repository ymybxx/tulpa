"""Bounded live text transport shared by background response producers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Generic, Protocol, TypeVar
from uuid import UUID

from fish_memory.pet.types import PetEmotion
from fish_memory.tts.speech_tags import SpeechTagFilter


class ExpressionIdentity(Protocol):
    id: UUID
    fish_id: UUID
    session_id: UUID
    text: str


IntentT = TypeVar("IntentT", bound=ExpressionIdentity)


class ExpressionStream(Generic[IntentT]):
    def __init__(self, intent: IntentT) -> None:
        self.intent = intent
        self.emotion = PetEmotion.NEUTRAL
        self.text = ""
        self.speech_content = ""
        self.speech_tags = False
        self._speech_filter: SpeechTagFilter | None = None
        self.delivery_prefix = ""
        self.published = False
        self.finished = False
        self.failed = False
        self.suppression: str | None = None
        self.task: asyncio.Task[None] | None = None
        self.changed = asyncio.Event()

    def configure_speech_tags(self, enabled: bool) -> None:
        if enabled == self.speech_tags:
            return
        if self.text or self.speech_content:
            raise RuntimeError("speech protocol must be frozen before generation")
        self.speech_tags = enabled
        self._speech_filter = SpeechTagFilter() if enabled else None

    def append(self, text: str) -> str:
        if self.finished:
            raise RuntimeError("expression is already finished")
        if len(self.text) + len(text) > 2000:
            raise ValueError("expression exceeds its text budget")
        if self._speech_filter is not None:
            self.speech_content += text
            text = self._speech_filter.feed(text)
        self.text += text
        self.intent.text = self.text
        self.changed.set()
        return text

    def finish_text(self) -> str:
        tail = self._speech_filter.finish() if self._speech_filter else ""
        self.text += tail
        self.intent.text = self.text
        return tail

    def end(self, *, failed: bool = False) -> None:
        self.finished = True
        self.failed = failed
        self.changed.set()

    async def events(self, offset: int) -> AsyncIterator[dict[str, object]]:
        if not 0 <= offset <= len(self.text):
            raise ValueError("expression offset exceeds available text")
        speech_cursor = -1
        while True:
            self.changed.clear()
            if offset < len(self.text) or (
                self.speech_tags and speech_cursor != len(self.speech_content)
            ):
                text = self.text[offset:]
                speech_cursor = len(self.speech_content)
                yield {
                    "type": "text",
                    "offset": offset,
                    "text": text,
                    **self._speech_data(),
                }
                offset += len(text)
                # The producer may append and finish while the consumer handles
                # this event. Drain that new suffix before sending end.
                continue
            if self.finished:
                yield {
                    "type": "end",
                    "offset": offset,
                    "failed": self.failed,
                    **self._speech_data(),
                }
                return
            try:
                async with asyncio.timeout(10):
                    await self.changed.wait()
            except TimeoutError:
                yield {"type": "heartbeat"}

    def _speech_data(self) -> dict[str, object]:
        return (
            {"speechContent": self.speech_content, "speechTags": True}
            if self.speech_tags
            else {}
        )


class ExpressionRegistry(Generic[IntentT]):
    def __init__(
        self,
        *,
        capacity: int = 32,
        reclaimable: Callable[[ExpressionStream[IntentT]], bool] | None = None,
    ) -> None:
        self._capacity = capacity
        self._reclaimable = reclaimable or (lambda stream: stream.finished)
        self._streams: dict[UUID, ExpressionStream[IntentT]] = {}
        self.wake = asyncio.Event()

    def notify_delivery(self) -> None:
        wake, self.wake = self.wake, asyncio.Event()
        wake.set()

    def create(self, intent: IntentT) -> ExpressionStream[IntentT]:
        for key, stream in tuple(self._streams.items()):
            if len(self._streams) < self._capacity:
                break
            if self._reclaimable(stream):
                del self._streams[key]
        if len(self._streams) >= self._capacity:
            raise RuntimeError("expression capacity exhausted")
        if intent.id in self._streams:
            raise ValueError("expression already exists")
        stream = ExpressionStream(intent)
        self._streams[intent.id] = stream
        return stream

    def get(self, message_id: UUID) -> ExpressionStream[IntentT] | None:
        return self._streams.get(message_id)

    async def settle(self, message_id: UUID, *, interrupt: bool) -> None:
        stream = self.get(message_id)
        if stream is None or stream.task is None:
            return
        if interrupt and not stream.task.done():
            stream.task.cancel()
        await asyncio.gather(stream.task, return_exceptions=True)
