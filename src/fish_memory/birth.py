"""A new character's first moment: it says hello on its own, right away."""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from fish_memory.answer.speech_style import SPOKEN_TEXT_RULES
from fish_memory.db.fishes import FishStore
from fish_memory.db.sessions import SessionStore
from fish_memory.extensions.models import EventDefinition, PushEvent
from fish_memory.extensions.service import ExtensionService

BIRTH_EVENT = EventDefinition(
    type="character.born",
    schema={
        "type": "object",
        "description": """你刚刚诞生，这是你和用户的第一次见面。
按你的性格和你们的关系，说出你的第一句话：打个招呼，可以带一点刚来到这里的新鲜感，
或好奇地问对方一个简单的问题。一两句就好，像真的第一次开口。
不要介绍功能、不要教程式说明，不要输出动作旁白、Markdown 或内部信息。"""
        + SPOKEN_TEXT_RULES,
        "properties": {"经过": {"type": "string", "maxLength": 200}},
        "required": ["经过"],
        "additionalProperties": False,
    },
)
BIRTH_FACT = "用户刚刚为你选好了样子、名字和性格，你第一次睁开眼睛"


def birth_event_id(fish_id: UUID) -> UUID:
    """One birth per character, however often the desktop asks."""
    return uuid5(NAMESPACE_URL, f"fish-birth:{fish_id}")


async def announce_birth(
    events: ExtensionService,
    fishes: FishStore,
    sessions: SessionStore,
    fish_id: UUID,
    *,
    fallback: str = "你好呀，我来了！",
) -> bool:
    """Hands the birth to the character as an event; False if it cannot speak now."""

    async def target() -> tuple[UUID, UUID] | None:
        if await fishes.get_assistant() != fish_id:
            return None
        session = await sessions.get_latest(fish_id)
        if session is None:
            session = await sessions.create(fish_id)
        return fish_id, session.id

    data: dict[str, Any] = {"经过": BIRTH_FACT}
    return await events.push_host_event(
        PushEvent(event_id=birth_event_id(fish_id), type=BIRTH_EVENT.type, data=data),
        BIRTH_EVENT,
        target=target,
        supersession_key=f"birth:{fish_id}",
        fallback_text=fallback,
    )
