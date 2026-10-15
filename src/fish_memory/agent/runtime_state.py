"""Live Task admission shared by chat, voice drafts and plugins."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fish_memory.background.supervisor import ProcessorProtocol


class AgentRuntimeState:
    def __init__(self, *, enabled: bool, default_target_id: UUID | None):
        self.default_target_id = default_target_id
        self.revision = 0
        self._ready = asyncio.Event()
        if enabled:
            self._ready.set()

    @property
    def enabled(self) -> bool:
        return self._ready.is_set()

    def configure(self, *, enabled: bool, default_target_id: UUID | None) -> None:
        self.default_target_id = default_target_id
        self.revision += 1
        if enabled:
            self._ready.set()
        else:
            self._ready.clear()

    async def wait_enabled(self) -> None:
        while not self.enabled:
            await self._ready.wait()


class EnabledAgentProcessor:
    """Disabled Task workers sleep without polling the database."""

    def __init__(self, processor: ProcessorProtocol, state: AgentRuntimeState):
        self.processor, self.state = processor, state

    async def process_once(self) -> bool:
        await self.state.wait_enabled()
        return await self.processor.process_once()
