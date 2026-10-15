"""Event bridge from voice activity to the companion coordinator."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from uuid import UUID

from fish_memory.pet.hub import PetEventHub
from fish_memory.proactive.coordinator import ProactiveCoordinator


class ProactiveRuntimeBridge:
    def __init__(self, coordinator: ProactiveCoordinator, hub: PetEventHub):
        self._coordinator = coordinator
        self._hub = hub
        self._subscription, self._queue = hub.subscribe(control_only=True)
        self._task = asyncio.create_task(self._run(), name="proactive-companion-events")

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            if event.kind != "control":
                continue
            value = json.loads(event.value)
            if value.get("type") != "pet.companion.activity":
                continue
            payload = value.get("payload", {})
            try:
                self._coordinator.foreground(
                    UUID(payload["fishId"]),
                    UUID(payload["sessionId"]),
                    bool(payload["busy"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

    async def close(self) -> None:
        self._hub.unsubscribe(self._subscription)
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
