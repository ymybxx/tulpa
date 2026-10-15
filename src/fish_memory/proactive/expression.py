"""One expression stream, shared by the producer, delivery and native consumer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Protocol

from fish_memory.answer.lane_stream import AnswerLaneDelta
from fish_memory.background_messages.streams import (
    ExpressionRegistry as ExpressionRegistry,
)
from fish_memory.background_messages.streams import ExpressionStream as ExpressionStream
from fish_memory.companion.text_buffer import TtsTextBuffer
from fish_memory.llm.streaming import closing_iterator
from fish_memory.pet.types import PetEmotion
from fish_memory.proactive.models import ExpressionIntent


class ExpressionDelivery(Protocol):
    async def begin(self, stream: ExpressionStream[ExpressionIntent]) -> None: ...
    async def finish(self, stream: ExpressionStream[ExpressionIntent]) -> None: ...


class GeneratedExpressionInput:
    """Bounded text streamed in by Jev; cancellation leaves history alone."""

    def __init__(self, *, speech_tags: bool = False) -> None:
        self.text = ""
        self.emotion: PetEmotion | None = None
        self.finished = False
        self.failed = False
        self.changed = asyncio.Event()
        self.speech_tags = speech_tags

    async def feed(self, delta: AnswerLaneDelta) -> None:
        if self.finished:
            return
        if len(self.text) + len(delta.answer) > 2000:
            raise ValueError("expression exceeds its text budget")
        self.text += delta.answer
        if delta.emotion is not None:
            self.emotion = delta.emotion
        if delta.answer_complete:
            self.finished = True
        self.changed.set()

    async def end(self, failed: bool = False) -> None:
        if not self.finished:
            self.finished = True
            self.failed = failed
            self.changed.set()

    async def events(self) -> AsyncIterator[AnswerLaneDelta]:
        offset = 0
        while True:
            self.changed.clear()
            if self.failed:
                raise ValueError("joint response ended before speech was complete")
            finished = self.finished
            if offset < len(self.text) or finished:
                text, offset = self.text[offset:], len(self.text)
                yield AnswerLaneDelta(
                    emotion=self.emotion,
                    answer=text,
                    answer_complete=finished,
                )
            if finished:
                return
            await self.changed.wait()


class ExpressionComposer:
    def __init__(self, delivery: ExpressionDelivery) -> None:
        self._delivery = delivery

    async def compose_generated(
        self,
        stream: ExpressionStream[ExpressionIntent],
        source: GeneratedExpressionInput,
        *,
        valid: Callable[[], bool],
        on_first_text: Callable[[], None] = lambda: None,
    ) -> None:
        stream.configure_speech_tags(source.speech_tags)
        buffer = TtsTextBuffer()
        accepted = True
        failed = False
        ready_characters = 0
        try:
            async with closing_iterator(source.events()) as response:
                async for delta in response:
                    if not valid():
                        raise asyncio.CancelledError
                    if delta.accepted is not None:
                        accepted = delta.accepted
                    if delta.emotion is not None:
                        stream.emotion = delta.emotion
                    if delta.answer:
                        if not stream.text:
                            on_first_text()
                        clean = stream.append(delta.answer)
                    else:
                        clean = ""
                    if delta.answer_complete:
                        clean += stream.finish_text()
                    chunks = buffer.append(clean)
                    if delta.answer_complete:
                        chunks.extend(buffer.finish())
                    ready_characters += sum(map(len, chunks))
                    if accepted and ready_characters and not stream.published:
                        stream.delivery_prefix = stream.text[:ready_characters]
                        if not valid():
                            raise asyncio.CancelledError
                        await self._delivery.begin(stream)
                        stream.published = True
        except BaseException:
            failed = True
            raise
        finally:
            # Persist the complete produced prefix before acknowledging stream end.
            # Consumption can then freeze a played prefix without racing generation.
            try:
                if stream.published:
                    await self._delivery.finish(stream)
            finally:
                stream.end(failed=failed)
