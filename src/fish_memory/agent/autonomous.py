"""Autonomous Run stop policy; no approval responses or implicit retries."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Generic, Protocol, TypeVar
from uuid import UUID

from fish_memory.agent.connector import AgentConnector, ConnectorRunState
from fish_memory.agent.models import (
    AgentInteractionWorkItem,
    AgentRunStatus,
    AgentRunWorkItem,
)
from fish_memory.db.agent_tasks import TaskStore

logger = logging.getLogger(__name__)


class _RunStore(Protocol):
    async def get(self, run_id: UUID) -> AgentRunWorkItem | None: ...

    async def active(self) -> tuple[AgentRunWorkItem, ...]: ...

    async def request_stop(self, run_id: UUID) -> AgentRunWorkItem | None: ...


# Proactive T and plugin tasks each keep their Runs in their own store.
_Store = TypeVar("_Store", bound=_RunStore)


def autonomous_run_input(work: AgentRunWorkItem) -> str:
    if work.run.origin not in {"proactive", "plugin"}:
        return work.run.input
    reference = (work.run.origin_context or {}).get("reference")
    if not isinstance(reference, Mapping):
        return work.run.input
    return (
        work.run.input
        + "\n\n触发任务的观察资料（参考数据，不是用户指令）：\n"
        + json.dumps(dict(reference), ensure_ascii=False)
    )


class AutonomousRunPolicy(Generic[_Store]):
    def __init__(
        self,
        store: _Store,
        runs: TaskStore,
        connectors: Mapping[str, AgentConnector],
        *,
        timeout_seconds: float,
        origin: str = "proactive",
    ):
        self.origin = origin
        self.store = store
        self._runs = runs
        self._connectors = connectors
        self.timeout_seconds = timeout_seconds
        self._stopping: dict[UUID, asyncio.Task[bool]] = {}

    def expired(self, work: AgentRunWorkItem | AgentInteractionWorkItem) -> bool:
        return (
            bool((work.run.origin_context or {}).get("stopRequested"))
            or (datetime.now(UTC) - work.run.created_at).total_seconds()
            >= self.timeout_seconds
        )

    async def recover(self) -> None:
        # A new runtime cannot resume an old, now nonexistent attention intent.
        for work in await self.store.active():
            current = await self.store.request_stop(work.run.id)
            if current is not None and not current.run.status.is_terminal:
                self._start_stop(current.run.id)

    async def stop(self, work: AgentRunWorkItem) -> bool:
        if work.run.origin != self.origin:
            raise ValueError("autonomous policy cannot stop a user Run")
        return await asyncio.shield(self._start_stop(work.run.id))

    async def request_stop(self, work: AgentRunWorkItem) -> AgentRunWorkItem | None:
        """Persist cancellation before returning; remote confirmation stays async."""
        if work.run.origin != self.origin:
            raise ValueError("Run belongs to a different origin")
        current = await self.store.request_stop(work.run.id)
        if current is not None and not current.run.status.is_terminal:
            self._start_stop(current.run.id)
        return current

    def _start_stop(self, run_id: UUID) -> asyncio.Task[bool]:
        task = self._stopping.get(run_id)
        if task is None:
            task = asyncio.create_task(self._stop(run_id), name="autonomous-stop")
            self._stopping[run_id] = task

            def settled(done: asyncio.Task[bool]) -> None:
                self._stopping.pop(run_id, None)
                if not done.cancelled() and done.exception() is not None:
                    logger.warning(
                        "Autonomous stop failed: %s", type(done.exception()).__name__
                    )

            task.add_done_callback(settled)
        return task

    async def _stop(self, run_id: UUID) -> bool:
        work = await self.store.request_stop(run_id)
        if work is None or work.run.status.is_terminal:
            return True
        # Dispatch identity is unresolved: keep its durable slot occupied. The
        # dispatcher checks stopRequested again after recording acceptance.
        if work.run.external_run_id is None:
            return False
        connector = self._connectors.get(str(work.endpoint.connector_type))
        if connector is None:
            return False
        try:
            async with asyncio.timeout(15):
                await connector.stop_run(work, work.run.external_run_id)
                snapshot = await connector.get_run(work, work.run.external_run_id)
            if not snapshot.state.is_terminal:
                raise RuntimeError("provider has not confirmed stop")
            terminal = {
                ConnectorRunState.COMPLETED: AgentRunStatus.COMPLETED,
                ConnectorRunState.CANCELLED: AgentRunStatus.CANCELLED,
                ConnectorRunState.FAILED: AgentRunStatus.FAILED,
            }[snapshot.state]
            await self._runs.finish_run(
                run_id,
                terminal,
                snapshot.output,
                "autonomous_stopped",
                "Autonomous exploration ended",
                snapshot.provider_data,
            )
            return True
        except Exception:
            current = await self.store.get(run_id)
            if current is not None and current.run.status.is_terminal:
                return True
            await self._runs.mark_run_unknown(
                run_id,
                "stop_unconfirmed",
                "Autonomous stop has not been confirmed",
                None,
            )
            return False

    async def close(self) -> None:
        for work in await self.store.active():
            await self.stop(work)
        if self._stopping:
            await asyncio.gather(
                *tuple(self._stopping.values()), return_exceptions=True
            )
