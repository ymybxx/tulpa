from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from fish_memory.answer.runtime import FishRuntime
from fish_memory.models import MessageStatus
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion
from fish_memory.tts.speech_tags import SpeechTagFilter


class DraftOutputLimitError(ValueError):
    """Raised when speculative answer output exceeds its bounded buffer."""


class AnswerOutputSink(Protocol):
    async def set_emotion(self, emotion: PetEmotion) -> None: ...

    async def append(self, text: str, *, continuation: bool = False) -> None: ...

    async def flush(self) -> None: ...

    async def flush_through(self, end_offset: int) -> None: ...

    async def set_end(self, end: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class AnswerOutputSnapshot:
    emotion: PetEmotion
    end: bool
    content: str
    flush_offset: int
    speech_content: str = ""
    speech_tags: bool = False


class RuntimeAnswerSink:
    """Publishes selected answer output to its owning runtime."""

    def __init__(
        self,
        runtime: FishRuntime,
        answer_message_id: UUID,
        tracker: MessagePerformanceTracker,
    ) -> None:
        self._runtime = runtime
        self._answer_message_id = answer_message_id
        self._tracker = tracker
        self._first_content = True
        self._first_append = True
        self._speech_filter = SpeechTagFilter() if runtime.speech_tags else None

    def configure_speech_tags(self, enabled: bool) -> None:
        if enabled == self._runtime.speech_tags:
            return
        if self._runtime.content or self._runtime.speech_content:
            raise RuntimeError("speech protocol must be frozen before generation")
        self._runtime.speech_tags = enabled
        self._speech_filter = SpeechTagFilter() if enabled else None

    async def set_emotion(self, emotion: PetEmotion) -> None:
        async with self._runtime.condition:
            if not _owns_generating_answer(self._runtime, self._answer_message_id):
                return
            self._runtime.pet_emotion = emotion
            self._runtime.pet_metadata_revision += 1
            self._runtime.condition.notify_all()

    async def append(self, text: str, *, continuation: bool = False) -> None:
        if not text:
            return
        async with self._runtime.condition:
            if not _owns_generating_answer(self._runtime, self._answer_message_id):
                return
            if self._speech_filter is not None:
                self._runtime.speech_content += text
                text = self._speech_filter.feed(text)
                if not text:
                    self._runtime.condition.notify_all()
                    return
            now_ns = self._tracker.now_ns()
            if self._first_content:
                self._tracker.record(
                    "answer.first_content",
                    started_ns=self._tracker.origin_ns,
                    finished_ns=now_ns,
                )
                self._first_content = False
            if continuation and self._first_append:
                self._tracker.record(
                    "answer.first_append",
                    started_ns=self._tracker.origin_ns,
                    finished_ns=now_ns,
                )
                self._first_append = False
            self._runtime.content += text
            self._runtime.condition.notify_all()

    async def flush(self) -> None:
        async with self._runtime.condition:
            if not _owns_generating_answer(self._runtime, self._answer_message_id):
                return
            if self._speech_filter is not None:
                self._runtime.content += self._speech_filter.finish()
            self._flush_through_locked(len(self._runtime.content))

    async def flush_through(self, end_offset: int) -> None:
        if isinstance(end_offset, bool) or not isinstance(end_offset, int):
            raise TypeError("flush offset must be an integer")
        async with self._runtime.condition:
            if not _owns_generating_answer(self._runtime, self._answer_message_id):
                return
            if end_offset < 0 or end_offset > len(self._runtime.content):
                raise ValueError("flush offset must be within published content")
            self._flush_through_locked(end_offset)

    async def set_end(self, end: bool) -> None:
        async with self._runtime.condition:
            if not _owns_generating_answer(self._runtime, self._answer_message_id):
                return
            self._runtime.end_conversation = end
            self._runtime.condition.notify_all()

    def _flush_through_locked(self, end_offset: int) -> None:
        if end_offset <= self._runtime.flush_offset:
            return
        self._runtime.flush_offset = end_offset
        self._runtime.condition.notify_all()


class PromotableAnswerSink:
    """Buffers speculative answer output until it is promoted to a target sink."""

    def __init__(self, *, max_chars: int = 8192, speech_tags: bool = False) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self._max_chars = max_chars
        self._lock = asyncio.Lock()
        self._emotion = PetEmotion.NEUTRAL
        self._has_emotion = False
        self._end = False
        self._has_end = False
        self._content = ""
        self._has_continuation = False
        self._flush_offset = 0
        self._target: AnswerOutputSink | None = None
        self._revoked = False
        self._speech_content = ""
        self._speech_filter = SpeechTagFilter() if speech_tags else None
        self._speech_flush_offsets: list[int] = []

    def configure_speech_tags(self, enabled: bool) -> None:
        if enabled == (self._speech_filter is not None):
            return
        if self._content or self._speech_content or self._target is not None:
            raise RuntimeError("speech protocol must be frozen before generation")
        self._speech_filter = SpeechTagFilter() if enabled else None

    def revoke(self) -> None:
        """Reject late provider output even when cancellation is not cooperative."""
        self._revoked = True

    def snapshot(self) -> AnswerOutputSnapshot:
        return AnswerOutputSnapshot(
            emotion=self._emotion,
            end=self._end,
            content=self._content,
            flush_offset=self._flush_offset,
            speech_content=self._speech_content,
            speech_tags=self._speech_filter is not None,
        )

    async def set_emotion(self, emotion: PetEmotion) -> None:
        async with self._lock:
            if self._revoked:
                return
            self._emotion = emotion
            self._has_emotion = True
            if self._target is not None:
                await self._target.set_emotion(emotion)

    async def append(self, text: str, *, continuation: bool = False) -> None:
        if not text:
            return
        async with self._lock:
            if self._revoked:
                return
            raw_text = text
            if self._target is None and (
                len(self._speech_content if self._speech_filter else self._content)
                + len(text)
                > self._max_chars
            ):
                raise DraftOutputLimitError(
                    f"draft output exceeds {self._max_chars} character limit"
                )
            if self._speech_filter is not None:
                self._speech_content += text
                text = self._speech_filter.feed(text)
            self._content += text
            self._has_continuation = self._has_continuation or continuation
            if self._target is not None:
                await self._target.append(raw_text, continuation=continuation)

    async def flush(self) -> None:
        async with self._lock:
            if self._revoked:
                return
            if self._speech_filter is not None:
                self._content += self._speech_filter.finish()
                speech_offset = len(self._speech_content)
                if (
                    not self._speech_flush_offsets
                    or self._speech_flush_offsets[-1] != speech_offset
                ):
                    self._speech_flush_offsets.append(speech_offset)
            self._flush_offset = len(self._content)
            if self._target is not None:
                await self._target.flush()

    async def flush_through(self, end_offset: int) -> None:
        if isinstance(end_offset, bool) or not isinstance(end_offset, int):
            raise TypeError("flush offset must be an integer")
        async with self._lock:
            if self._revoked:
                return
            if end_offset < self._flush_offset or end_offset > len(self._content):
                raise ValueError("flush offset must advance within buffered content")
            self._flush_offset = end_offset
            if self._target is not None:
                await self._target.flush_through(end_offset)

    async def set_end(self, end: bool) -> None:
        async with self._lock:
            if self._revoked:
                return
            self._end = end
            self._has_end = True
            if self._target is not None:
                await self._target.set_end(end)

    async def attach(self, target: AnswerOutputSink) -> None:
        async with self._lock:
            if self._revoked or self._target is not None:
                return
            configure = getattr(target, "configure_speech_tags", None)
            if callable(configure):
                configure(self._speech_filter is not None)
            self._target = target
            if self._has_emotion:
                await target.set_emotion(self._emotion)
            if self._revoked:
                return
            if self._content or self._speech_content:
                if self._speech_filter is not None:
                    start = 0
                    for end in self._speech_flush_offsets:
                        await target.append(
                            self._speech_content[start:end],
                            continuation=self._has_continuation,
                        )
                        if self._revoked:
                            return
                        await target.flush()
                        if self._revoked:
                            return
                        start = end
                    await target.append(
                        self._speech_content[start:],
                        continuation=self._has_continuation,
                    )
                else:
                    await target.append(
                        self._content, continuation=self._has_continuation
                    )
            if self._revoked:
                return
            if self._flush_offset:
                await target.flush_through(self._flush_offset)
            if self._revoked:
                return
            if self._has_end:
                await target.set_end(self._end)


def _owns_generating_answer(runtime: FishRuntime, answer_message_id: UUID) -> bool:
    return (
        runtime.answer_message_id == answer_message_id
        and runtime.status is MessageStatus.GENERATING
    )
