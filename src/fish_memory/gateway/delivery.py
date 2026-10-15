"""Keep an event another program sent in, like any other event TA receives."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field

from fish_memory.extensions.service import ExtensionService

EVENT_TYPE = "external.message"


class GatewayEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The sender's own ID for this event: sending it again is not a new event.
    id: str | None = Field(None, min_length=1, max_length=200)
    source: str | None = Field(None, max_length=60)
    title: str | None = Field(None, max_length=200)
    text: str = Field(min_length=1, max_length=8000)
    # For the user: TA answers it as it answers the result of its own work.
    # Otherwise it is only kept in TA's context.
    speak: bool = True


class GatewayDelivery:
    def __init__(
        self,
        events: ExtensionService,
        target: Callable[[], Awaitable[tuple[UUID, UUID] | None]],
    ) -> None:
        self.events, self.target = events, target

    @staticmethod
    def event_id(event: GatewayEvent) -> UUID:
        if event.id is None:
            return uuid4()
        return uuid5(NAMESPACE_URL, f"fish-gateway:{event.source or ''}:{event.id}")

    async def __call__(self, event: GatewayEvent) -> UUID | None:
        """The accepted event's ID, or None when there is no character to tell."""
        recipient = await self.target()
        if recipient is None:
            return None
        data = {"source": (event.source or "").strip() or "外部程序"}
        if event.title and event.title.strip():
            data["title"] = event.title.strip()
        data["text"] = event.text.strip()
        identifier = self.event_id(event)
        # It never wakes TA on its own: with 主动陪伴 on, TA reads it at its
        # usual wake; with it off, a message for the user is answered like
        # the result of TA's own work.
        await self.events.store.receive_host_event(
            *recipient,
            identifier,
            EVENT_TYPE,
            data,
            wakes=False,
            reports_work=event.speak,
        )
        self.events.changed(*recipient)
        return identifier
