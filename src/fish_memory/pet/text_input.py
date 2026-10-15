"""Acknowledged, non-replayed quick-chat commands for the native voice owner."""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Callable
from uuid import UUID


class PetTextInput:
    def __init__(self) -> None:
        self._requests: OrderedDict[
            str, tuple[str, asyncio.Future[dict[str, object]]]
        ] = OrderedDict()

    async def submit(
        self,
        request_id: UUID,
        fish_id: UUID,
        content: str,
        publish: Callable[[dict[str, object]], object],
        *,
        attachment_ids: tuple[UUID, ...] = (),
    ) -> dict[str, object]:
        key = str(request_id)
        attached = [str(item) for item in attachment_ids]
        digest = hashlib.sha256(
            f"{fish_id}:{content}:{','.join(attached)}".encode()
        ).hexdigest()
        previous = self._requests.get(key)
        if previous is not None:
            if previous[0] != digest:
                raise ValueError("requestId already belongs to a different message")
            future = previous[1]
        else:
            if len(self._requests) >= 64:
                # Never evict an uncertain submission and accidentally replay it.
                settled = next(
                    (k for k, v in self._requests.items() if v[1].done()), None
                )
                if settled is None:
                    raise ValueError("too many pending quick-chat requests")
                del self._requests[settled]
            future = asyncio.get_running_loop().create_future()
            self._requests[key] = (digest, future)
            publish(
                {
                    "requestId": key,
                    "fishId": str(fish_id),
                    "content": content,
                    "attachmentIds": attached,
                }
            )
        # Client timeout does not cancel a message already being persisted.
        async with asyncio.timeout(15):
            return await asyncio.shield(future)

    def complete(self, payload: dict[str, object]) -> None:
        entry = self._requests.get(str(payload.get("requestId")))
        if entry is not None and not entry[1].done():
            entry[1].set_result(dict(payload))

    def close(self) -> None:
        for _, future in self._requests.values():
            if not future.done():
                future.cancel()
        self._requests.clear()
