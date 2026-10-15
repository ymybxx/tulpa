"""Bounded FishSession context for external Agent Run submission."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, cast
from uuid import UUID

from fish_memory.agent.connector import AgentHistoryMessage
from fish_memory.agent.models import AgentRunWorkItem
from fish_memory.answer.history_tokens import estimate_history_message_tokens
from fish_memory.models import Message, MessageStatus, Role, visible_message_content

_MAX_SOURCE_SESSION_MESSAGES = 12
_MAX_SOURCE_SESSION_TOKENS = 8_000
_MAX_SOURCE_SESSION_MESSAGE_CHARS = 4_000
_MAX_SOURCE_SESSION_CHARS = 48_000


class SourceSessionHistoryStore(Protocol):
    async def load_history_before_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        *,
        limit: int,
    ) -> list[Message] | None: ...

    async def load_history_before_task_root_source(
        self,
        fish_id: UUID,
        session_id: UUID,
        task_id: UUID,
        *,
        limit: int,
    ) -> list[Message] | None: ...


class AgentSessionContextLoader:
    """Projects only consumable context preceding a Run's source Message."""

    def __init__(self, store: SourceSessionHistoryStore) -> None:
        self._store = store

    async def load_for_run(
        self, work: AgentRunWorkItem
    ) -> tuple[AgentHistoryMessage, ...] | None:
        if work.run.origin == "plugin":
            # Plugin Task input is already frozen and selected by the SDK.
            # Do not attach whatever conversation happens at dispatch time.
            return ()
        if work.run.origin == "proactive":
            context = work.run.origin_context or {}
            values = context.get("history", [])
            if not isinstance(values, list):
                return None
            history = []
            remaining = 16_000
            for value in values[-12:]:
                if not isinstance(value, dict):
                    return None
                role, content = value.get("role"), value.get("content")
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    return None
                if len(content) > remaining:
                    break
                remaining -= len(content)
                history.append(
                    AgentHistoryMessage(
                        role=cast(Literal["user", "assistant"], role), content=content
                    )
                )
            return tuple(history)
        if work.run.source_message_id is None:
            return None
        if work.run.previous_run_id is None:
            messages = await self._store.load_history_before_message(
                work.run.fish_id,
                work.run.session_id,
                work.run.source_message_id,
                limit=_MAX_SOURCE_SESSION_MESSAGES,
            )
        else:
            messages = await self._store.load_history_before_task_root_source(
                work.run.fish_id,
                work.run.session_id,
                work.run.task_id,
                limit=_MAX_SOURCE_SESSION_MESSAGES,
            )
        if messages is None:
            return None
        return _bound_source_session_history(messages)


def _bound_source_session_history(
    messages: Sequence[Message],
) -> tuple[AgentHistoryMessage, ...]:
    """Keep recent complete rounds without exposing unconsumed assistant tails."""

    selected: list[AgentHistoryMessage] = []
    remaining_tokens = _MAX_SOURCE_SESSION_TOKENS
    remaining_chars = _MAX_SOURCE_SESSION_CHARS
    for message in reversed(messages[-_MAX_SOURCE_SESSION_MESSAGES:]):
        if message.role not in {Role.USER, Role.ASSISTANT}:
            continue
        if message.role is Role.ASSISTANT and message.status not in {
            MessageStatus.COMPLETED,
            MessageStatus.INTERRUPTED,
        }:
            continue
        content = visible_message_content(message)[:_MAX_SOURCE_SESSION_MESSAGE_CHARS]
        if not content:
            continue
        token_count = estimate_history_message_tokens(content)
        if token_count > remaining_tokens or len(content) > remaining_chars:
            break
        selected.append(AgentHistoryMessage(role=message.role.value, content=content))
        remaining_tokens -= token_count
        remaining_chars -= len(content)
    selected.reverse()
    return tuple(selected)


__all__ = ["AgentSessionContextLoader", "SourceSessionHistoryStore"]
