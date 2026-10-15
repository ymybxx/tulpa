"""Deliver a generated prefix through the existing consumption barrier."""

from __future__ import annotations

import asyncio

from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.proactive.expression import ExpressionRegistry, ExpressionStream


class ProactiveDelivery:
    def __init__(self, store: BackgroundMessageStore, streams: ExpressionRegistry):
        self._store = store
        self._streams = streams

    async def begin(self, stream: ExpressionStream) -> None:
        intent = stream.intent
        write = asyncio.create_task(
            self._store.materialize_proactive_observation(
                intent.fish_id,
                intent.session_id,
                stream.delivery_prefix,
                emotion=stream.emotion,
                expires_at=None,
                supersession_key=f"attention:{intent.generation}:{intent.focus_id}",
                message_id=intent.id,
                source_observation_id=intent.source_observation_id,
            )
        )
        try:
            await asyncio.shield(write)
        except asyncio.CancelledError:
            # Own an in-flight transaction until we know whether it committed.
            await write
            stream.published = True
            raise
        stream.published = True
        self._streams.notify_delivery()

    async def finish(self, stream: ExpressionStream) -> None:
        intent = stream.intent
        await self._store.update_proactive_content(
            intent.id, intent.fish_id, intent.session_id, stream.text
        )
