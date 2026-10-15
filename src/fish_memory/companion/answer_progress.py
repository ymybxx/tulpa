"""One immutable answer's rendered progress, publication and retrying persistence."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Protocol
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class ProgressApi(Protocol):
    async def update_progress(
        self, fish_id: UUID, message_id: UUID, consumed_offset: int
    ) -> None: ...
    async def interrupt_answer(
        self, fish_id: UUID, message_id: UUID, consumed_offset: int
    ) -> None: ...


class AnswerPlaybackProgress:
    def __init__(
        self,
        api: ProgressApi,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        sample: Callable[[], tuple[str, int, int, str]],
        publish: Callable[[dict[str, object]], None],
        *,
        message_sequence: int | None = None,
    ) -> None:
        self.api, self.fish_id, self.session_id, self.message_id = (
            api,
            fish_id,
            session_id,
            message_id,
        )
        self.playback_id = uuid4()
        self.message_sequence = message_sequence
        self._sample, self._publish = sample, publish
        self.consumed = self.display = self.revision = 0
        self.content = ""
        self.status = "buffering"
        self.failure: str | None = None
        self.terminal = False
        self._wake = asyncio.Event()
        self._last_snapshot: tuple[object, ...] | None = None
        self._reported = -1
        self.refresh()
        self.task = asyncio.create_task(
            self._run(), name=f"answer-progress:{message_id}"
        )

    def refresh(self) -> None:
        if self.terminal:
            return
        content, display, consumed, status = self._sample()
        self.content = content
        self.consumed = max(self.consumed, min(len(content), max(0, consumed)))
        self.display = max(
            self.display, self.consumed, min(len(content), max(0, display))
        )
        self.status = status
        self._emit()

    def finish(self, status: str, failure: str | None = None) -> None:
        if self.terminal:
            return
        self.refresh()
        self.terminal = True
        self.failure = failure
        if status == "completed" and self.consumed < len(self.content):
            status = "failed"
        self.status = status
        if status in {"interrupted", "failed"}:
            self.display = self.consumed
        self._emit()
        self._wake.set()

    def _emit(self) -> None:
        visible = self.content[: self.display]
        signature = (
            self.message_sequence,
            self.display,
            self.consumed,
            self.status,
            visible,
            self.failure,
        )
        if signature == self._last_snapshot:
            return
        self._last_snapshot = signature
        self.revision += 1
        self._publish(
            {
                "fishId": str(self.fish_id),
                "sessionId": str(self.session_id),
                "messageId": str(self.message_id),
                **(
                    {"messageSequence": self.message_sequence}
                    if self.message_sequence is not None
                    else {}
                ),
                "playbackId": str(self.playback_id),
                "revision": self.revision,
                "status": self.status,
                "displayOffset": self.display,
                "consumedOffset": self.consumed,
                "visibleContent": visible,
                **({"failure": self.failure} if self.failure else {}),
            }
        )

    async def _run(self) -> None:
        delay = 0.1
        while True:
            self._wake.clear()
            self.refresh()
            offset, terminal, status = self.consumed, self.terminal, self.status
            try:
                if offset > self._reported or (terminal and status == "interrupted"):
                    async with asyncio.timeout(0.5):
                        if terminal and status == "interrupted":
                            await self.api.interrupt_answer(
                                self.fish_id, self.message_id, offset
                            )
                        else:
                            await self.api.update_progress(
                                self.fish_id, self.message_id, offset
                            )
                    self._reported = offset
                if terminal:
                    return
                delay = 0.1
            except Exception:
                logger.warning(
                    "Answer progress pending: message_id=%s offset=%s",
                    self.message_id,
                    offset,
                    exc_info=True,
                )
                delay = min(2.0, delay * 2)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def settle(self, max_wait_seconds: float = 0.5) -> bool:
        self._wake.set()
        try:
            await asyncio.wait_for(asyncio.shield(self.task), max_wait_seconds)
            return True
        except TimeoutError:
            return False

    async def close(self) -> None:
        if not await self.settle(1.0):
            logger.error(
                "Unpersisted answer progress on shutdown: message_id=%s offset=%s",
                self.message_id,
                self.consumed,
            )
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
