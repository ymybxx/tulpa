"""SDK Task RPCs. The host owns identity and execution; plugins own material."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fish_memory.agent.autonomous import AutonomousRunPolicy
from fish_memory.agent.models import AgentRunWorkItem
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.db.plugin_tasks import PluginTaskConflict, PluginTaskStore
from fish_memory.extensions.models import ExtensionError, Instance

logger = logging.getLogger(__name__)
_CHANNEL = "fish_plugin_tasks"


class StartTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000, strict=True)
    request_id: UUID
    context_key: str = Field(min_length=1, max_length=256, strict=True)
    reference: dict[str, Any] = Field(default_factory=dict)
    context_revision: str | None = Field(default=None, max_length=128)


class TaskQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: UUID | None = None
    request_id: UUID | None = None
    cursor: int = Field(default=0, ge=0, strict=True)

    @model_validator(mode="after")
    def identity(self):
        if (self.run_id is None) == (self.request_id is None):
            raise ValueError("Provide run_id or request_id")
        return self


def task_snapshot(work: AgentRunWorkItem) -> dict[str, Any]:
    run, scope = work.run, work.run.origin_context or {}
    stopped = bool(scope.get("stopRequested"))
    return {
        "task_id": str(run.task_id),
        "run_id": str(run.id),
        "request_id": scope.get("requestId"),
        "context_key": scope.get("contextKey"),
        "status": run.status.value,
        "stop_requested": stopped,
        "result_available": run.status.value == "completed" and not stopped,
        "created_at": run.created_at.isoformat(timespec="seconds"),
        "finished_at": run.finished_at.isoformat(timespec="seconds")
        if run.finished_at
        else None,
        "error_code": run.error_code,
    }


class PluginTasks:
    def __init__(
        self,
        store: PluginTaskStore,
        targets: Any,
        policy: AutonomousRunPolicy[PluginTaskStore],
        *,
        default_target_id: UUID | None,
        instances: dict[UUID, Instance],
        emit: Callable[[Instance, str, dict[str, Any]], None],
        runtime_state: AgentRuntimeState | None = None,
    ):
        self.store, self.targets, self.policy = store, targets, policy
        self.default_target_id, self.instances, self.emit = (
            default_target_id,
            instances,
            emit,
        )
        self._pending: set[UUID] = set()
        self._runtime_state = runtime_state
        self._wake = asyncio.Event()
        self._listener: asyncio.Task[None] | None = None
        self._worker: asyncio.Task[None] | None = None

    @property
    def available(self) -> bool:
        return self._runtime_state is None or self._runtime_state.enabled

    @property
    def revision(self) -> int:
        return self._runtime_state.revision if self._runtime_state is not None else 0

    def start(self) -> None:
        if self._listener is None:
            self._listener = asyncio.create_task(
                self._listen(), name="plugin-task-listener"
            )
            self._worker = asyncio.create_task(
                self._changes(), name="plugin-task-notifications"
            )

    def changed(self, run_id: UUID) -> None:
        # Hints are coalesced and bounded. Authoritative state is always queryable.
        if len(self._pending) < 128:
            self._pending.add(run_id)
            self._wake.set()

    async def _listen(self) -> None:
        def notified(_connection: Any, _pid: int, _channel: str, payload: str):
            try:
                self.changed(UUID(payload))
            except ValueError:
                pass

        while True:
            try:
                if self._runtime_state is not None:
                    await self._runtime_state.wait_enabled()
                async with self.store.pool.acquire() as connection:
                    lost = asyncio.Event()

                    def terminated(_c: Any, event: asyncio.Event = lost):
                        event.set()

                    connection.add_termination_listener(terminated)
                    await connection.add_listener(_CHANNEL, notified)
                    try:
                        await lost.wait()
                    finally:
                        connection.remove_termination_listener(terminated)
                        if not connection.is_closed():
                            await connection.remove_listener(_CHANNEL, notified)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Plugin Task notifications disconnected; queries remain available"
                )
            await asyncio.sleep(2)

    async def _changes(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            pending, self._pending = self._pending, set()
            for run_id in pending:
                try:
                    work = await self.store.get(run_id)
                    if work is None:
                        continue
                    scope = work.run.origin_context or {}
                    item = self.instances.get(UUID(str(scope.get("instanceId"))))
                    if item and not item.stopped and self._owns(item, work):
                        self.emit(item, "task_changed", task_snapshot(work))
                except Exception:
                    logger.warning("Plugin Task notification failed", exc_info=True)

    @staticmethod
    def _owns(item: Instance, work: AgentRunWorkItem) -> bool:
        run, scope = work.run, work.run.origin_context or {}
        return (
            run.origin == "plugin"
            and run.fish_id == item.fish_id
            and run.session_id == item.session_id
            and scope.get("pluginId") == item.plugin_id
            and scope.get("instanceId") == str(item.id)
            and scope.get("generation") == str(item.generation)
        )

    async def request(
        self, item: Instance, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        if method == "start":
            request = StartTask.model_validate(params)
            payload = request.model_dump(mode="json")
            encoded = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            if len(encoded.encode()) > 20_000:
                raise ExtensionError(
                    "task_input_too_large", "Select at most 20 KB of input"
                )
            digest = hashlib.sha256(encoded.encode()).hexdigest()
            # Resolve lost ACK before consulting today's target configuration.
            existing = await self.store.by_request(
                item.fish_id,
                item.session_id,
                item.id,
                item.generation,
                request.request_id,
            )
            if existing:
                if (existing.run.origin_context or {}).get("requestDigest") != digest:
                    raise ExtensionError("task_conflict", "request_id input changed")
                return task_snapshot(existing)
            if not self.available:
                raise ExtensionError("task_unavailable", "Agent Tasks are disabled")
            revision = self.revision
            target = await self.targets.resolve_for_fish(
                item.fish_id,
                self._runtime_state.default_target_id
                if self._runtime_state is not None
                else self.default_target_id,
            )
            if not self.available or self.revision != revision:
                raise ExtensionError("task_unavailable", "Agent configuration changed")
            if target is None:
                raise ExtensionError(
                    "task_unavailable", "Configure an Agent target for this pet"
                )
            scope = {
                "pluginId": item.plugin_id,
                "instanceId": str(item.id),
                "generation": str(item.generation),
                "requestId": str(request.request_id),
                "requestDigest": digest,
                "contextKey": request.context_key,
                "reference": request.reference,
                "contextRevision": request.context_revision,
            }
            try:
                work = await self.store.create(
                    fish_id=item.fish_id,
                    session_id=item.session_id,
                    target_id=target.target.id,
                    purpose=request.prompt,
                    origin_context=scope,
                )
            except PluginTaskConflict as error:
                raise ExtensionError("task_conflict", str(error)) from error
            self.changed(work.run.id)
            return task_snapshot(work)
        if method not in {"get", "result", "cancel"}:
            raise ExtensionError("unknown_method", "Unknown Task method")
        query = TaskQuery.model_validate(params)
        work = (
            await self.store.get(query.run_id)
            if query.run_id is not None
            else await self.store.by_request(
                item.fish_id,
                item.session_id,
                item.id,
                item.generation,
                cast(UUID, query.request_id),
            )
        )
        if work is None or not self._owns(item, work):
            raise ExtensionError(
                "task_not_found", "Task is unavailable for this instance"
            )
        if method == "cancel":
            current = await self.policy.request_stop(work)
            self.changed(work.run.id)
            return task_snapshot(current or work)
        result = task_snapshot(work)
        if method == "result":
            if not result["result_available"]:
                raise ExtensionError(
                    "task_not_ready", "Task has no usable completed result"
                )
            raw = (work.run.output or "").encode()
            if query.cursor > len(raw):
                raise ValueError("Cursor exceeds Task result")
            try:
                raw[: query.cursor].decode()
            except UnicodeDecodeError as error:
                raise ValueError("Cursor is not a text boundary") from error
            text = raw[query.cursor : query.cursor + 16_000].decode(errors="ignore")
            end = query.cursor + len(text.encode())
            result.update(
                text=text,
                cursor=query.cursor,
                next_cursor=end if end < len(raw) else None,
                total_bytes=len(raw),
                truncated=end < len(raw),
            )
        return result

    async def stop_instance(self, item: Instance) -> None:
        for work in await self.store.active():
            if self._owns(item, work):
                await self.policy.request_stop(work)

    async def close(self) -> None:
        tasks = [task for task in (self._listener, self._worker) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.policy.close()
