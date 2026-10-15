"""Autonomous work Jev may start on the pet's behalf."""

from __future__ import annotations

from typing import Protocol

from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.proactive.models import ExpressionIntent, Supplement


class AutonomousTasks(Protocol):
    async def available(self, context: LoadedAnswerContext) -> bool: ...
    async def execute(
        self, intent: ExpressionIntent, context: LoadedAnswerContext
    ) -> Supplement | None: ...
    async def close(self) -> None: ...
