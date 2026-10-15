from __future__ import annotations

from fish_memory.background_messages.store import BackgroundMessageStore


class BackgroundMessageExpiryProcessor:
    """Releases expired proactive messages even when no SSE client is connected."""

    def __init__(self, store: BackgroundMessageStore) -> None:
        self._store = store
        self._closed = False

    async def process_once(self) -> bool:
        if self._closed:
            return False
        return await self._store.expire_proactive_messages() > 0

    async def close(self) -> None:
        self._closed = True


__all__ = ["BackgroundMessageExpiryProcessor"]
