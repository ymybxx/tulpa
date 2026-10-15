"""Compose and atomically materialize durable Agent Assistant messages."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from fish_memory.agent.models import (
    AgentInteractionKind,
    AgentInteractionWorkItem,
    AgentRunWorkItem,
)
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.speech_style import SPOKEN_TEXT_RULES
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.soul import SoulProfile

SAFE_AGENT_MESSAGE_FALLBACK = (
    "任务状态已更新，但我暂时无法安全说明详情。请不要据此授权操作。"
)

_MAX_COMPLETION_BYTES = 16_384
_MAX_ANSWER_CHARS = 4_000
_MAX_PROMPT_TEXT_CHARS = 8_000
_MAX_DECISION_CHARS = 64
_EMOTIONS = frozenset({"neutral", "angry", "happy", "shy", "sad"})
_APPROVAL_TEXT_FIELDS = ("command", "tool", "description", "reason")
_APPROVAL_BOOL_FIELDS = ("smart_denied", "recoveredWithoutDetails")

_MESSAGE_SYSTEM_PROMPT = (
    """# Agent 状态消息协议
你负责把一个已经持久化的后台任务状态通顺、自然地转达给用户。
只根据给出的角色配置、Task 标题和规范化状态数据表达，不补充工具调用、外部标识、
端点、凭证、内部过程或未提供的操作细节。

kind=run_result 时，output 是外部 Agent 已经执行任务后交付的完整结果，不是供你概括的
素材。可以调整语序、指代和衔接使表达通顺，但不得省略、合并或模糊任何关键信息，
包括列表项、名称、版本、数量、大小、路径、链接、比较关系、重复或替代关系、结论、
限制、风险、未执行事项和下一步。完整性优先于简短；不得用“已整理”“已标出”等概述
代替 output 中的具体内容，也不得把具体结果改写成让用户再次索取详情的建议。

kind=interaction 时，审批问题必须准确复述给出的操作与允许选择，不得扩大授权范围；
详情不足时只建议拒绝或稍后重试。

只返回一个 JSON object，字段必须恰好为 m、a，各出现一次：
- m 只能是 neutral、angry、happy、shy、sad 之一；
- a 必须是非空自然语言，不得包含 Markdown 或内部数据。
不要输出 JSON 之外的内容。"""
    + "\n\n"
    + SPOKEN_TEXT_RULES
)


class FishSoulLoader(Protocol):
    async def get_profile(self, fish_id: UUID) -> SoulProfile | None: ...


class ForegroundActivityGate(Protocol):
    """Read-only adapter supplied by application assembly in Task 10."""

    async def is_active(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ComposedAgentMessage:
    emotion: str
    content: str
    used_fallback: bool


class AgentMessageComposer:
    """A strict JSON composer over bounded, provider-safe inputs only."""

    def __init__(
        self,
        client: JsonCompletionClient,
        soul_loader: FishSoulLoader,
    ) -> None:
        self._client = client
        self._soul_loader = soul_loader

    async def compose_run(self, work: AgentRunWorkItem) -> ComposedAgentMessage:
        try:
            payload: dict[str, object] = {
                "kind": "run_result",
                "title": _bounded_prompt_text("title", work.task.title),
                "status": work.run.status.value,
            }
            if work.run.output is not None:
                payload["output"] = _bounded_prompt_text("output", work.run.output)
        except Exception:
            return _safe_fallback()
        return await self._compose(work.task.fish_id, payload)

    async def compose_interaction(
        self,
        work: AgentInteractionWorkItem,
    ) -> ComposedAgentMessage:
        try:
            payload: dict[str, object] = {
                "kind": "interaction",
                "interactionKind": work.interaction.kind.value,
                "title": _bounded_prompt_text("title", work.task.title),
            }
            if work.interaction.kind is AgentInteractionKind.INPUT:
                if work.run.output is not None:
                    payload["request"] = _bounded_prompt_text(
                        "request", work.run.output
                    )
            else:
                payload["approval"] = _safe_approval_data(
                    work.interaction.provider_data
                )
        except Exception:
            return _safe_fallback()
        return await self._compose(work.task.fish_id, payload)

    async def _compose(
        self,
        fish_id: UUID,
        payload: Mapping[str, object],
    ) -> ComposedAgentMessage:
        try:
            profile = await self._soul_loader.get_profile(fish_id)
            if profile is None:
                raise ValueError("Fish soul profile is unavailable")
            profile = SoulProfile.from_values(
                profile.role,
                profile.setting,
                profile.name,
            )
            messages: list[ChatMessage] = [
                {
                    "role": "system",
                    "content": profile.render(_MESSAGE_SYSTEM_PROMPT),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ]
            completion = await self._client.complete_json(messages)
            return _parse_completion(completion.content)
        except Exception:
            return _safe_fallback()


class AgentMessageStore(Protocol):
    async def peek_unmaterialized_interaction(
        self,
    ) -> AgentInteractionWorkItem | None: ...

    async def peek_unmaterialized_terminal_run(self) -> AgentRunWorkItem | None: ...

    async def materialize_interaction_request(
        self,
        interaction_id: UUID,
        content: str,
        *,
        emotion: str = "neutral",
    ) -> object | None: ...

    async def materialize_run_result(
        self,
        run_id: UUID,
        content: str,
        *,
        emotion: str = "neutral",
    ) -> object | None: ...


class AgentMessageContentComposer(Protocol):
    async def compose_interaction(
        self,
        work: AgentInteractionWorkItem,
    ) -> ComposedAgentMessage: ...

    async def compose_run(
        self,
        work: AgentRunWorkItem,
    ) -> ComposedAgentMessage: ...


class AgentMessageProcessor:
    """Select read-only work, compose outside a transaction, then enter CAS."""

    def __init__(
        self,
        store: AgentMessageStore,
        composer: AgentMessageContentComposer,
        *,
        background_semaphore: asyncio.Semaphore,
        foreground_gate: ForegroundActivityGate,
    ) -> None:
        self._store = store
        self._composer = composer
        self._background_semaphore = background_semaphore
        self._foreground_gate = foreground_gate
        self._closed = False

    async def process_once(self) -> bool:
        if self._closed or await self._foreground_active():
            return False

        interaction = await self._store.peek_unmaterialized_interaction()
        run = (
            None
            if interaction is not None
            else await self._store.peek_unmaterialized_terminal_run()
        )
        if interaction is None and run is None:
            return False

        async with self._background_semaphore:
            if self._closed or await self._foreground_active():
                return False
            if interaction is not None:
                draft = await self._composer.compose_interaction(interaction)
            else:
                assert run is not None
                draft = await self._composer.compose_run(run)

        if interaction is not None:
            await self._store.materialize_interaction_request(
                interaction.interaction.id,
                draft.content,
                emotion=draft.emotion,
            )
        else:
            assert run is not None
            await self._store.materialize_run_result(
                run.run.id,
                draft.content,
                emotion=draft.emotion,
            )
        return True

    async def close(self) -> None:
        self._closed = True

    async def _foreground_active(self) -> bool:
        try:
            return await self._foreground_gate.is_active()
        except asyncio.CancelledError:
            raise
        except Exception:
            return True


def _safe_approval_data(
    provider_data: Mapping[str, object] | None,
) -> dict[str, object]:
    source = provider_data or {}
    safe: dict[str, object] = {}
    for key in _APPROVAL_TEXT_FIELDS:
        value = source.get(key)
        if isinstance(value, str) and value:
            safe[key] = _bounded_prompt_text(key, value)
    for key in _APPROVAL_BOOL_FIELDS:
        value = source.get(key)
        if isinstance(value, bool):
            safe[key] = value
    raw_decisions = source.get("allowed_decisions")
    if isinstance(raw_decisions, list):
        decisions = [
            item
            for item in raw_decisions
            if isinstance(item, str) and item and len(item) <= _MAX_DECISION_CHARS
        ]
        if decisions:
            safe["allowedDecisions"] = decisions[:16]
    return safe


def _bounded_prompt_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be non-empty")
    if len(value) > _MAX_PROMPT_TEXT_CHARS:
        return value[:_MAX_PROMPT_TEXT_CHARS]
    return value


def _parse_completion(content: str) -> ComposedAgentMessage:
    if not isinstance(content, str):
        raise ValueError("Agent message completion must be text")
    if len(content.encode("utf-8")) > _MAX_COMPLETION_BYTES:
        raise ValueError("Agent message completion is too large")
    value = json.loads(content, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(value, dict) or set(value) != {"m", "a"}:
        raise ValueError("Agent message completion must contain only m and a")
    emotion = value["m"]
    answer = value["a"]
    if not isinstance(emotion, str) or emotion not in _EMOTIONS:
        raise ValueError("Agent message completion has an invalid emotion")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Agent message completion answer must be non-empty")
    normalized = answer.strip()
    if "\x00" in normalized or len(normalized) > _MAX_ANSWER_CHARS:
        raise ValueError("Agent message completion answer is too large")
    return ComposedAgentMessage(emotion, normalized, False)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Agent message completion contains duplicate keys")
        value[key] = item
    return value


def _safe_fallback() -> ComposedAgentMessage:
    return ComposedAgentMessage(
        emotion="neutral",
        content=SAFE_AGENT_MESSAGE_FALLBACK,
        used_fallback=True,
    )


__all__ = [
    "SAFE_AGENT_MESSAGE_FALLBACK",
    "AgentMessageComposer",
    "AgentMessageProcessor",
    "ComposedAgentMessage",
    "FishSoulLoader",
    "ForegroundActivityGate",
]
