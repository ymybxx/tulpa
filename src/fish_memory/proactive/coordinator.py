"""Companion ownership of the pet's autonomous speech.

主动陪伴 is driven by Jev (``ExperimentalRuntime``). This coordinator keeps the
app-facing surface — binding sessions, user activity, playback receipts and
external suspension — and forwards it to Jev. It does not judge on its own.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from uuid import UUID, uuid4

from fish_memory.environment_frames import EnvironmentPerceptionPolicy
from fish_memory.experimental.runtime import ExperimentalRuntime
from fish_memory.proactive.executors import AutonomousTasks
from fish_memory.proactive.expression import ExpressionRegistry
from fish_memory.proactive.models import ExpressionIntent


class ProactiveCoordinator:
    def __init__(
        self,
        *,
        streams: ExpressionRegistry[ExpressionIntent],
        policy: Callable[[UUID], EnvironmentPerceptionPolicy],
        tasks: AutonomousTasks | None = None,
    ) -> None:
        self.streams = streams
        self.tasks = tasks
        self._policy = policy
        self.experimental: ExperimentalRuntime | None = None
        self._binding: tuple[UUID, UUID] | None = None
        self._consumers: set[UUID] = set()
        self._current_policy: EnvironmentPerceptionPolicy | None = None
        self._foreground_states: OrderedDict[tuple[UUID, UUID], bool] = OrderedDict()
        self._external_sources: dict[UUID, set[UUID]] = {}

    @property
    def enabled(self) -> bool:
        return self.experimental is not None and self.experimental.service.enabled

    def bind(self, fish_id: UUID, session_id: UUID) -> UUID:
        token = uuid4()
        if self._binding != (fish_id, session_id):
            self._reset()
            self._binding = (fish_id, session_id)
            self._current_policy = self._policy(fish_id)
        self._consumers.add(token)
        if self.experimental is not None:
            self.experimental.subscribe(token, fish_id, session_id)
            self.experimental.foreground_busy = self._foreground_states.get(
                (fish_id, session_id), False
            )
        return token

    def unbind(self, token: UUID) -> None:
        self._consumers.discard(token)
        if self.experimental is not None:
            self.experimental.unsubscribe(token)
        if not self._consumers:
            self._reset()
            self._binding = None

    def suspend_external(self, fish_id: UUID, owner: UUID) -> None:
        """Lease autonomous speech without changing the user's saved setting."""
        self._external_sources.setdefault(fish_id, set()).add(owner)
        if self.experimental is not None:
            self.experimental.suspended = True
            self.experimental.interrupt()

    def resume_external(self, fish_id: UUID, owner: UUID) -> None:
        owners = self._external_sources.get(fish_id)
        if owners is not None:
            owners.discard(owner)
            if not owners:
                self._external_sources.pop(fish_id, None)
        if self.experimental is not None:
            self.experimental.suspended = bool(self._external_sources.get(fish_id))

    def responding_for(self, fish_id: UUID) -> bool:
        return bool(
            self.enabled
            and self.experimental is not None
            and self.experimental.binding
            and self.experimental.binding[0] == fish_id
            and self.experimental.busy
        )

    def _reset(self) -> None:
        if self.experimental is not None:
            self.experimental.reset()
        self._current_policy = None

    def policy_changed(self, fish_id: UUID) -> None:
        if self._binding is None or self._binding[0] != fish_id:
            return
        policy = self._policy(fish_id)
        if policy != self._current_policy:
            self._reset()
            self._current_policy = policy

    def profile_changed(self, fish_id: UUID) -> None:
        if self._binding is not None and self._binding[0] == fish_id:
            self._reset()
            self._current_policy = self._policy(fish_id)

    def foreground(self, fish_id: UUID, session_id: UUID, busy: bool) -> None:
        self._foreground_states[fish_id, session_id] = busy
        self._foreground_states.move_to_end((fish_id, session_id))
        while len(self._foreground_states) > 4:
            self._foreground_states.popitem(last=False)
        if self.experimental is not None and self.experimental.binding == (
            fish_id,
            session_id,
        ):
            self.experimental.foreground_busy = busy
            # TA may be reporting a result while 主动陪伴 is off.
            if busy:
                self.experimental.interrupt()

    def conversation_started(self, fish_id: UUID, session_id: UUID) -> None:
        if self.experimental is not None and self.experimental.binding == (
            fish_id,
            session_id,
        ):
            self.experimental.interrupt()

    def consumed(self, message_id: UUID, offset: int, interrupted: bool) -> None:
        if self.experimental is not None:
            self.experimental.consumed(message_id, offset, interrupted)

    async def admit_window_summary(self, fish_id, session_id, document) -> bool:
        if self.experimental is None:
            return True
        return await self.experimental.admit_window(document)

    def experimental_configuration_changed(self) -> None:
        if self._binding is not None:
            self.profile_changed(self._binding[0])

    def status(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "bound": self._binding is not None,
            "fishId": str(self._binding[0]) if self._binding else None,
        }

    async def close(self) -> None:
        if self.experimental is not None:
            await self.experimental.close()
        self._reset()
        if self.tasks is not None:
            await self.tasks.close()
