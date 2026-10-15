"""Background research through the pet's configured Agent, without speech."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .client import Pet


class TaskSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    task_id: UUID
    run_id: UUID
    request_id: UUID
    context_key: str
    status: str
    stop_requested: bool = False
    result_available: bool = False
    created_at: str
    finished_at: str | None = None
    error_code: str | None = None

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "failed", "cancelled"}


class TaskResult(TaskSnapshot):
    text: str
    cursor: int
    next_cursor: int | None
    total_bytes: int
    truncated: bool


class Tasks:
    def __init__(self, pet: Pet):
        self._pet = pet

    @property
    def available(self) -> bool:
        return bool(self._pet.capabilities.get("tasks"))

    async def start(
        self,
        prompt: str,
        *,
        request_id: UUID | str,
        context_key: str,
        reference: dict[str, Any] | None = None,
        context_revision: str | None = None,
    ) -> TaskSnapshot:
        """Return a persisted queue receipt, never wait for research or auto-retry.

        Keep request_id: after a lost receipt use get(request_id=...) or resubmit
        exactly the same input and ID. Materials are frozen at this call.
        """
        response = await self._pet._request(
            "tasks.start",
            {
                "prompt": prompt,
                "request_id": str(UUID(str(request_id))),
                "context_key": context_key,
                "reference": reference or {},
                "context_revision": context_revision,
            },
        )
        return TaskSnapshot.model_validate(response)

    async def get(
        self,
        run_id: UUID | str | None = None,
        *,
        request_id: UUID | str | None = None,
    ) -> TaskSnapshot:
        if (run_id is None) == (request_id is None):
            raise ValueError("Provide run_id or request_id")
        params = {"run_id": str(run_id)} if run_id else {"request_id": str(request_id)}
        return TaskSnapshot.model_validate(
            await self._pet._request("tasks.get", params)
        )

    async def result(self, run_id: UUID | str, *, cursor: int = 0) -> TaskResult:
        return TaskResult.model_validate(
            await self._pet._request(
                "tasks.result",
                {
                    "run_id": str(run_id),
                    "cursor": cursor,
                },
            )
        )

    async def cancel(self, run_id: UUID | str) -> TaskSnapshot:
        """Record cancellation; stop_requested is not remote stop confirmation."""
        return TaskSnapshot.model_validate(
            await self._pet._request(
                "tasks.cancel",
                {
                    "run_id": str(run_id),
                },
            )
        )
