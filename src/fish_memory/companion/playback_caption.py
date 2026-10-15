"""Display-only progress for background speech; persistence stays with its owner."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from uuid import UUID, uuid4


class PlaybackCaption:
    def __init__(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        sample: Callable[[], tuple[str, int, int, str]],
        publish: Callable[[dict[str, object]], None],
        *,
        message_sequence: int | None = None,
    ) -> None:
        self._identity = {
            "fishId": str(fish_id),
            "sessionId": str(session_id),
            "messageId": str(message_id),
            "playbackId": str(uuid4()),
            **(
                {"messageSequence": message_sequence}
                if message_sequence is not None
                else {}
            ),
        }
        self._sample, self._publish = sample, publish
        self._loop = asyncio.get_running_loop()
        self._timer: asyncio.TimerHandle | None = None
        self._terminal = False
        self._consumed = self._display = self._revision = 0
        self._previous: tuple[object, ...] | None = None
        self._tick()

    def _tick(self) -> None:
        self._emit()
        self._timer = self._loop.call_later(0.1, self._tick)

    def finish(self, status: str) -> None:
        if self._terminal:
            return
        self._terminal = True
        if self._timer:
            self._timer.cancel()
        self._emit(status)

    def _emit(self, terminal: str | None = None) -> None:
        content, display, consumed, status = self._sample()
        self._consumed = max(self._consumed, min(len(content), max(0, consumed)))
        self._display = max(
            self._display, self._consumed, min(len(content), max(0, display))
        )
        if terminal:
            status = terminal
            if status == "completed" and self._consumed < len(content):
                status = "failed"
            self._display = self._consumed
        signature = (content[: self._display], self._display, self._consumed, status)
        if signature == self._previous:
            return
        self._previous = signature
        self._revision += 1
        self._publish(
            {
                **self._identity,
                "revision": self._revision,
                "status": status,
                "displayOffset": self._display,
                "consumedOffset": self._consumed,
                "visibleContent": content[: self._display],
            }
        )
