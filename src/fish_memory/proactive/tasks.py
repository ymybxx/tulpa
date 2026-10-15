"""Selected T execution through durable external Agent tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from fish_memory.agent.autonomous import AutonomousRunPolicy
from fish_memory.agent.connector import AgentConnector
from fish_memory.agent.models import AgentRunStatus
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.db.autonomous_tasks import AutonomousTaskStore
from fish_memory.proactive.models import ExpressionIntent, Supplement


def task_reference(intent: ExpressionIntent) -> dict[str, object]:
    """Carry the triggering facts with the goal, within the Agent input budget."""
    basis = intent.basis
    reference: dict[str, object] = {
        "topic": basis.topic if basis else "",
        "progress": basis.progress if basis else "",
    }
    selected = []
    remaining = 5500
    for evidence in reversed(intent.evidence):
        # This connector accepts text. A pointer saying 'see the image' is not
        # an image attachment; the understood scene remains in topic/progress.
        if evidence.kind == "screen" or evidence.id == "context":
            continue
        if len(evidence.text) > remaining:
            continue
        selected.insert(0, {"kind": evidence.kind, "text": evidence.text})
        remaining -= len(evidence.text)
    reference["evidence"] = selected
    return reference


class AutonomousTaskExecutor:
    def __init__(
        self,
        policy: AutonomousRunPolicy[AutonomousTaskStore],
        connectors: Mapping[str, AgentConnector],
        *,
        cooldown_seconds: float = 600,
        starts_per_hour: int = 3,
        runtime_state: AgentRuntimeState | None = None,
    ):
        self._policy = policy
        self._connectors = connectors
        self._cooldown = cooldown_seconds
        self._limit = starts_per_hour
        self._admitting = False
        self._closed = False
        self._runtime_state = runtime_state

    async def available(self, context: LoadedAnswerContext) -> bool:
        target = context.agent_target
        if (
            self._closed
            or (self._runtime_state is not None and not self._runtime_state.enabled)
            or self._admitting
            or target is None
            or not target.target.enabled
            or str(target.endpoint.connector_type) not in self._connectors
        ):
            return False
        return await self._policy.store.available(self._cooldown, self._limit)

    async def execute(
        self, intent: ExpressionIntent, context: LoadedAnswerContext
    ) -> Supplement | None:
        if not await self.available(context):
            return None
        target = context.agent_target
        assert target is not None
        self._admitting = True
        work = None
        try:
            connector = self._connectors[str(target.endpoint.connector_type)]
            async with asyncio.timeout(10):
                capabilities = await connector.probe(target)
            if not (
                capabilities.run_submission
                and capabilities.run_status
                and capabilities.stop
            ):
                return None
            if self._runtime_state is not None and not self._runtime_state.enabled:
                return None
            # Preserve actual roles; attention evidence is metadata, never a
            # fabricated user Message or a remembered autonomous conversation.
            history = []
            remaining = 16_000
            for message in reversed(context.recall_conversation[-12:]):
                content = message.get("content", "")
                if not isinstance(content, str) or len(content) > remaining:
                    continue
                history.insert(0, {"role": message["role"], "content": content})
                remaining -= len(content)
            admission = asyncio.create_task(
                self._policy.store.create(
                    fish_id=intent.fish_id,
                    session_id=intent.session_id,
                    target_id=target.target.id,
                    purpose=intent.purpose,
                    origin_context={
                        "intentId": str(intent.id),
                        "focusId": intent.focus_id,
                        "history": history,
                        "reference": task_reference(intent),
                    },
                    cooldown_seconds=self._cooldown,
                    starts_per_hour=self._limit,
                )
            )
            try:
                work = await asyncio.shield(admission)
            except asyncio.CancelledError:
                work = await admission
                if work is not None:
                    await self._policy.stop(work)
                raise
        finally:
            self._admitting = False
        if work is None:
            return None
        intent.timings["taskCreatedAt"] = work.run.created_at.timestamp()
        try:
            async with asyncio.timeout(self._policy.timeout_seconds):
                while True:
                    current = await self._policy.store.get(work.run.id)
                    if current is None:
                        return None
                    work = current
                    if work.run.status.is_terminal:
                        if (
                            work.run.status == AgentRunStatus.COMPLETED
                            and not (work.run.origin_context or {}).get("stopRequested")
                            and work.run.output
                        ):
                            return Supplement(
                                "task",
                                intent.purpose,
                                work.run.output,
                                (str(work.task.id), str(work.run.id)),
                            )
                        return None
                    if work.run.status in {
                        AgentRunStatus.WAITING_USER,
                        AgentRunStatus.UNKNOWN,
                    }:
                        return None
                    await asyncio.sleep(1)
        except TimeoutError:
            return None
        finally:
            if not work.run.status.is_terminal:
                await self._policy.stop(work)

    async def close(self) -> None:
        self._closed = True
        await self._policy.close()
