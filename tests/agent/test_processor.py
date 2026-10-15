from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest

from fish_memory.agent.connector import (
    AgentCapabilities,
    AgentConnectorError,
    AgentHistoryMessage,
    AgentProviderEvent,
    AgentProviderEventKind,
    AgentRunAccepted,
    AgentRunRequest,
    AgentRunSnapshot,
    ConnectorRunState,
)
from fish_memory.agent.models import (
    AgentEndpoint,
    AgentHistorySource,
    AgentInteraction,
    AgentInteractionKind,
    AgentInteractionWorkItem,
    AgentResolvedTarget,
    AgentRun,
    AgentRunStatus,
    AgentRunWorkItem,
    AgentTarget,
    AgentTask,
    ConnectorTypeSlug,
)
from fish_memory.agent.processor import AgentRunProcessor
from fish_memory.answer.history_tokens import estimate_history_message_tokens

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 8, 30, tzinfo=UTC)


def work(
    number: int = 1,
    *,
    status: AgentRunStatus = AgentRunStatus.QUEUED,
    previous: bool = False,
    external_run_id: str | None = None,
    connector_type: str = "hermes",
    provider_data: Mapping[str, object] | None = None,
    enabled: bool = True,
) -> AgentRunWorkItem:
    base = number * 100
    endpoint = AgentEndpoint(
        id=UUID(int=base + 1),
        connector_type=ConnectorTypeSlug(connector_type),
        base_url="https://agent.example.test",
        created_at=_NOW,
        updated_at=_NOW,
    )
    target = AgentTarget(
        id=UUID(int=base + 2),
        endpoint_id=endpoint.id,
        external_key=f"target-{number}",
        route_prefix="",
        credential_ref="agent:test",
        connector_config={},
        display_name="Agent",
        enabled=enabled,
        created_at=_NOW,
        updated_at=_NOW,
    )
    task = AgentTask(
        id=UUID(int=base + 3),
        fish_id=UUID(int=base + 4),
        session_id=UUID(int=base + 40),
        target_id=target.id,
        external_session_id=f"session-{number}",
        title="Task",
        created_at=_NOW,
        updated_at=_NOW,
    )
    run = AgentRun(
        id=UUID(int=base + 5),
        task_id=task.id,
        target_id=target.id,
        fish_id=task.fish_id,
        session_id=task.session_id,
        previous_run_id=UUID(int=base + 6) if previous else None,
        source_message_id=UUID(int=base + 7),
        confirmation_message_id=None,
        input=f"input-{number}",
        external_run_id=external_run_id,
        history_source=AgentHistorySource.NONE,
        status=status,
        output=None,
        result_message_id=None,
        error_code=None,
        error_message=None,
        provider_data=provider_data,
        created_at=_NOW,
        started_at=None,
        updated_at=_NOW,
        finished_at=None,
    )
    return AgentRunWorkItem(run=run, task=task, target=target, endpoint=endpoint)


def approval_work(
    run_work: AgentRunWorkItem,
    *,
    response_state: str = "queued",
    choice: str = "deny",
) -> AgentInteractionWorkItem:
    interaction = AgentInteraction(
        id=UUID(int=run_work.run.id.int + 20),
        run_id=run_work.run.id,
        fish_id=run_work.run.fish_id,
        session_id=run_work.run.session_id,
        kind=AgentInteractionKind.APPROVAL,
        request_message_id=UUID(int=run_work.run.id.int + 21),
        response_message_id=UUID(int=run_work.run.id.int + 22),
        next_run_id=None,
        provider_data={
            "allowed_decisions": ["deny"],
            "approval_response": {
                "choice": choice,
                "resolve_all": False,
                "state": response_state,
            },
        },
        created_at=_NOW,
        resolved_at=None,
        closed_at=None,
    )
    return AgentInteractionWorkItem(
        interaction=interaction,
        run=run_work.run,
        task=run_work.task,
        target=run_work.target,
        endpoint=run_work.endpoint,
    )


class FakeStore:
    def __init__(self) -> None:
        self.audit: list[tuple[str, object]] = []
        self.queued: AgentRunWorkItem | None = None
        self.stale: list[AgentRunWorkItem] = []
        self.reconcilable: list[AgentRunWorkItem] = []
        self.reconcilable_approvals: list[AgentInteractionWorkItem] = []
        self.pending_approval: AgentInteractionWorkItem | None = None
        self.pending_steer: AgentRunWorkItem | None = None
        self.projection: Sequence[tuple[str, str]] = ()
        self.message_content: str | None = None
        self.claim_wins = True
        self.unidentified_dispatches = 0
        self.unidentified_approvals = 0
        self.unidentified_steers = 0
        self.approval_is_open = False

    async def mark_unidentified_dispatches_unknown(self) -> int:
        self.audit.append(("recover_dispatches", None))
        return self.unidentified_dispatches

    async def mark_unidentified_approval_dispatches_unknown(self) -> int:
        self.audit.append(("recover_approvals", None))
        return self.unidentified_approvals

    async def mark_unidentified_steer_dispatches_unknown(self) -> int:
        self.audit.append(("recover_steers", None))
        return self.unidentified_steers

    async def list_reconcilable_run_work(
        self,
        limit: int,
        after: tuple[datetime, UUID] | None = None,
    ) -> Sequence[AgentRunWorkItem]:
        self.audit.append(("list_reconcilable", (limit, after)))
        ordered = sorted(
            self.reconcilable, key=lambda item: (item.run.created_at, item.run.id)
        )
        if after is not None:
            ordered = [
                item for item in ordered if (item.run.created_at, item.run.id) > after
            ]
        return tuple(ordered[:limit])

    async def list_stale_run_work(
        self,
        cutoff: datetime,
        limit: int = 1_000,
    ) -> Sequence[AgentRunWorkItem]:
        self.audit.append(("list_stale", (cutoff, limit)))
        return tuple(self.stale[:limit])

    async def fail_stale_run(
        self, run_id: UUID, error_code: str, error_message: str
    ) -> AgentRun | None:
        self.audit.append(("fail_stale", (run_id, error_code, error_message)))
        source = next((item for item in self.stale if item.run.id == run_id), None)
        self.stale = [item for item in self.stale if item.run.id != run_id]
        return (
            None
            if source is None
            else replace(
                source.run,
                status=AgentRunStatus.FAILED,
                error_code=error_code,
                error_message=error_message,
            )
        )

    async def list_reconcilable_approval_responses(
        self,
        limit: int = 1_000,
        after: tuple[datetime, UUID] | None = None,
    ) -> Sequence[AgentInteractionWorkItem]:
        self.audit.append(("list_reconcilable_approvals", (limit, after)))
        ordered = sorted(
            self.reconcilable_approvals,
            key=lambda item: (item.interaction.created_at, item.interaction.id),
        )
        if after is not None:
            ordered = [
                item
                for item in ordered
                if (item.interaction.created_at, item.interaction.id) > after
            ]
        return tuple(ordered[:limit])

    async def claim_pending_approval_response(
        self,
    ) -> AgentInteractionWorkItem | None:
        self.audit.append(("claim_approval", None))
        item, self.pending_approval = self.pending_approval, None
        return item

    async def claim_pending_steer(self) -> AgentRunWorkItem | None:
        self.audit.append(("claim_steer", None))
        item, self.pending_steer = self.pending_steer, None
        return item

    async def peek_queued_run(self) -> AgentRunWorkItem | None:
        self.audit.append(("peek", None))
        return self.queued

    async def load_local_projection(
        self, run_id: UUID, limit: int = 64
    ) -> Sequence[tuple[str, str]]:
        self.audit.append(("projection", run_id))
        return self.projection

    async def load_user_message_content(self, message_id: UUID) -> str | None:
        self.audit.append(("message", message_id))
        return self.message_content

    async def fail_queued_run(
        self, run_id: UUID, error_code: str, error_message: str
    ) -> AgentRun | None:
        self.audit.append(("fail_queued", (error_code, error_message)))
        return self.queued.run if self.queued is not None else None

    async def claim_prepared_run(
        self,
        run_id: UUID,
        history_source: AgentHistorySource | str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRunWorkItem | None:
        self.audit.append(("claim_run", AgentHistorySource(history_source)))
        if not self.claim_wins or self.queued is None:
            return None
        self.claim_wins = False
        return replace(
            self.queued,
            run=replace(
                self.queued.run,
                status=AgentRunStatus.DISPATCHING,
                history_source=AgentHistorySource(history_source),
            ),
        )

    async def record_dispatch_accepted(
        self,
        run_id: UUID,
        external_run_id: str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun:
        self.audit.append(("accepted", external_run_id))
        assert self.queued is not None
        return replace(
            self.queued.run,
            status=AgentRunStatus.RUNNING,
            external_run_id=external_run_id,
            provider_data=provider_data,
        )

    async def update_active_run(
        self,
        run_id: UUID,
        status: AgentRunStatus | str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun | None:
        self.audit.append(("update", (AgentRunStatus(status), provider_data)))
        self.stale = [item for item in self.stale if item.run.id != run_id]
        return None

    async def mark_run_unknown(
        self,
        run_id: UUID,
        error_code: str,
        error_message: str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun | None:
        self.audit.append(("unknown", (error_code, error_message, provider_data)))
        return None

    async def finish_run(
        self,
        run_id: UUID,
        status: AgentRunStatus | str,
        output: str | None,
        error_code: str | None,
        error_message: str | None,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun:
        self.audit.append(
            (
                "finish",
                (
                    AgentRunStatus(status),
                    output,
                    error_code,
                    error_message,
                    provider_data,
                ),
            )
        )
        source = self.queued or next(
            item for item in (*self.reconcilable, *self.stale) if item.run.id == run_id
        )
        self.stale = [item for item in self.stale if item.run.id != run_id]
        return replace(source.run, status=AgentRunStatus(status), output=output)

    async def complete_run_requiring_input(
        self,
        run_id: UUID,
        output: str,
        provider_data: Mapping[str, object] | None,
    ) -> tuple[AgentRun, AgentInteraction]:
        self.audit.append(("input_required", (output, provider_data)))
        source = self.queued or self.reconcilable[0]
        return source.run, approval_work(source).interaction

    async def open_approval_interaction(
        self, run_id: UUID, provider_data: Mapping[str, object] | None
    ) -> AgentInteraction:
        if not self.approval_is_open:
            self.audit.append(("open_approval", provider_data))
            self.approval_is_open = True
        source = self.queued or self.reconcilable[0]
        return approval_work(source).interaction

    async def mark_approval_dispatch_unknown(
        self, interaction_id: UUID, error_code: str, error_message: str
    ) -> AgentInteraction | None:
        self.audit.append(("approval_unknown", (error_code, error_message)))
        return None

    async def resolve_approval(self, interaction_id: UUID) -> AgentInteraction | None:
        self.audit.append(("resolve_approval", interaction_id))
        return None

    async def record_steer_accepted(self, run_id: UUID) -> AgentRun | None:
        self.audit.append(("steer_accepted", run_id))
        return None

    async def mark_steer_dispatch_unknown(
        self, run_id: UUID, error_code: str, error_message: str
    ) -> AgentRun | None:
        self.audit.append(("steer_unknown", (error_code, error_message)))
        return None


class FakeConnector:
    def __init__(self, audit: list[tuple[str, object]] | None = None) -> None:
        self.audit = audit if audit is not None else []
        self.accepted = AgentRunAccepted("remote-1", ConnectorRunState.RUNNING)
        self.start_error: AgentConnectorError | None = None
        self.history: Sequence[AgentHistoryMessage] = ()
        self.history_error: AgentConnectorError | None = None
        self.events: list[AgentProviderEvent | BaseException | asyncio.Event] = []
        self.snapshots: list[AgentRunSnapshot | BaseException] = []
        self.block_watch = False
        self.watch_entered = asyncio.Event()
        self.watch_cancelled = asyncio.Event()
        self.requests: list[AgentRunRequest] = []
        self.get_calls = 0
        self.approval_error: AgentConnectorError | None = None
        self.steer_error: AgentConnectorError | None = None
        self.preflight_error: AgentConnectorError | None = None

    async def probe(self, target: AgentResolvedTarget) -> AgentCapabilities:
        return AgentCapabilities(True, True, True, True, True, True, True)

    def preflight_run(self, work: AgentRunWorkItem) -> AgentRunWorkItem:
        self.audit.append(("preflight", work.run.id))
        if self.preflight_error is not None:
            raise self.preflight_error
        return work

    async def get_session_history(
        self, work: AgentRunWorkItem
    ) -> Sequence[AgentHistoryMessage]:
        self.audit.append(("connector_history", work.run.id))
        if self.history_error is not None:
            raise self.history_error
        return self.history

    async def start_run(
        self, work: AgentRunWorkItem, request: AgentRunRequest
    ) -> AgentRunAccepted:
        self.audit.append(("start", work.run.id))
        self.requests.append(request)
        if self.start_error is not None:
            raise self.start_error
        return self.accepted

    async def watch_run(
        self, work: AgentRunWorkItem, external_run_id: str
    ) -> AsyncIterator[AgentProviderEvent]:
        self.audit.append(("watch", external_run_id))
        self.watch_entered.set()
        if self.block_watch:
            try:
                await asyncio.Event().wait()
            finally:
                self.watch_cancelled.set()
        for item in self.events:
            if isinstance(item, asyncio.Event):
                await item.wait()
                continue
            if isinstance(item, BaseException):
                raise item
            yield item

    async def get_run(
        self, work: AgentRunWorkItem, external_run_id: str
    ) -> AgentRunSnapshot:
        self.get_calls += 1
        self.audit.append(("get", external_run_id))
        if not self.snapshots:
            return AgentRunSnapshot(ConnectorRunState.UNKNOWN)
        item = self.snapshots.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def respond_approval(
        self,
        work: AgentRunWorkItem,
        external_run_id: str,
        choice: str,
        resolve_all: bool,
    ) -> None:
        self.audit.append(("respond", (choice, resolve_all)))
        if self.approval_error is not None:
            raise self.approval_error

    async def steer_run(
        self, work: AgentRunWorkItem, external_run_id: str, input: str
    ) -> None:
        self.audit.append(("steer", input))
        if self.steer_error is not None:
            raise self.steer_error

    async def stop_run(self, work: AgentRunWorkItem, external_run_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeSessionContextLoader:
    def __init__(self, history: Sequence[AgentHistoryMessage]) -> None:
        self.history = history
        self.calls: list[AgentRunWorkItem] = []

    async def load_for_run(
        self, work: AgentRunWorkItem
    ) -> Sequence[AgentHistoryMessage] | None:
        self.calls.append(work)
        return self.history


async def wait_until(predicate: Any, *, turns: int = 50) -> None:
    for _ in range(turns):
        if predicate():
            return
        await asyncio.sleep(0.001)
    pytest.fail("condition was not reached")


async def test_continuation_prepares_local_history_before_cas_and_post() -> None:
    store = FakeStore()
    store.queued = work(previous=True)
    store.projection = (("user", "earlier"), ("assistant", "result"))
    connector = FakeConnector(store.audit)
    connector.events = [
        AgentProviderEvent(
            AgentProviderEventKind.TERMINAL,
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"done"}',
        )
    ]
    processor = AgentRunProcessor(
        store, {"hermes": connector}, concurrency=2, poll_interval_seconds=0.001
    )

    assert await processor.process_once() is True
    await wait_until(lambda: any(name == "finish" for name, _ in store.audit))
    names = [name for name, _ in store.audit]
    assert names.index("connector_history") < names.index("projection")
    assert names.index("projection") < names.index("claim_run")
    assert names.index("claim_run") < names.index("start")
    assert names.index("accepted") < names.index("watch")
    assert connector.requests[0].conversation_history == (
        AgentHistoryMessage("user", "earlier"),
        AgentHistoryMessage("assistant", "result"),
    )
    assert ("claim_run", AgentHistorySource.LOCAL_PROJECTION) in store.audit
    await processor.close()


async def test_first_run_uses_none_history_and_fixed_outcome_instructions() -> None:
    store = FakeStore()
    store.queued = work()
    connector = FakeConnector(store.audit)
    connector.block_watch = True
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    await connector.watch_entered.wait()
    assert not any(
        name in {"connector_history", "projection"} for name, _ in store.audit
    )
    assert ("claim_run", AgentHistorySource.NONE) in store.audit
    assert connector.requests[0].instructions
    assert "input_required" in (connector.requests[0].instructions or "")
    await processor.close()


async def test_first_run_sends_source_session_context_before_current_input() -> None:
    store = FakeStore()
    store.queued = work()
    connector = FakeConnector(store.audit)
    connector.block_watch = True
    session_context = FakeSessionContextLoader(
        (
            AgentHistoryMessage("user", "Earlier request"),
            AgentHistoryMessage("assistant", "Earlier consumed answer"),
        )
    )
    processor = AgentRunProcessor(
        store,
        {"hermes": connector},
        session_context_loader=session_context,
    )

    assert await processor.process_once() is True
    await connector.watch_entered.wait()
    assert session_context.calls == [store.queued]
    assert connector.requests[0].input == "input-1"
    assert connector.requests[0].conversation_history == session_context.history
    await processor.close()


async def test_continuation_prefixes_agent_history_with_source_session_context() -> (
    None
):
    store = FakeStore()
    store.queued = work(previous=True)
    connector = FakeConnector(store.audit)
    connector.block_watch = True
    connector.history = (AgentHistoryMessage("assistant", "Hermes task result"),)
    session_context = FakeSessionContextLoader(
        (AgentHistoryMessage("user", "Relevant FishSession request"),)
    )
    processor = AgentRunProcessor(
        store,
        {"hermes": connector},
        session_context_loader=session_context,
    )

    assert await processor.process_once() is True
    await connector.watch_entered.wait()
    assert connector.requests[0].conversation_history == (
        *session_context.history,
        *connector.history,
    )
    assert ("claim_run", AgentHistorySource.FISH_SESSION_HERMES_SESSION) in store.audit
    await processor.close()


async def test_unknown_or_disabled_connector_fails_before_claim_and_post() -> None:
    for candidate in (
        work(connector_type="missing"),
        work(enabled=False),
    ):
        store = FakeStore()
        store.queued = candidate
        processor = AgentRunProcessor(store, {})

        assert await processor.process_once() is True
        assert any(name == "fail_queued" for name, _ in store.audit)
        assert not any(name == "claim_run" for name, _ in store.audit)
        await processor.close()


async def test_first_run_missing_credential_fails_while_still_queued() -> None:
    store = FakeStore()
    store.queued = work()
    connector = FakeConnector(store.audit)
    connector.preflight_error = AgentConnectorError(
        "credential unavailable",
        code="authentication_required",
    )
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    assert any(name == "preflight" for name, _ in store.audit)
    assert any(name == "fail_queued" for name, _ in store.audit)
    assert not any(name in {"claim_run", "start"} for name, _ in store.audit)
    await processor.close()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            AgentConnectorError("bad request", code="invalid_request"),
            "finish",
        ),
        (
            AgentConnectorError(
                "lost response",
                code="endpoint_unavailable",
                dispatch_uncertain=True,
            ),
            "unknown",
        ),
    ],
)
async def test_dispatch_error_is_failed_or_unknown_without_retry(
    error: AgentConnectorError, expected: str
) -> None:
    store = FakeStore()
    store.queued = work()
    connector = FakeConnector(store.audit)
    connector.start_error = error
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    assert sum(name == "start" for name, _ in store.audit) == 1
    assert any(name == expected for name, _ in store.audit)
    await processor.process_once()
    assert sum(name == "start" for name, _ in store.audit) == 1
    await processor.close()


async def test_accepted_unknown_persists_external_id_before_local_unknown() -> None:
    store = FakeStore()
    store.queued = work()
    connector = FakeConnector(store.audit)
    connector.accepted = AgentRunAccepted("remote-unknown", ConnectorRunState.UNKNOWN)
    connector.snapshots = [AgentRunSnapshot(ConnectorRunState.UNKNOWN)]
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    names = [name for name, _ in store.audit]
    assert names.index("accepted") < names.index("unknown")
    assert ("accepted", "remote-unknown") in store.audit
    await wait_until(lambda: connector.get_calls == 1)
    assert not any(name == "watch" for name, _ in store.audit)
    await processor.close()


async def test_stale_run_recovers_remote_terminal_result_before_failing() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.stale = [remote]
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"recovered"}',
        )
    ]
    processor = AgentRunProcessor(
        store,
        {"hermes": connector},
        stale_timeout_seconds=300,
    )

    assert await processor.process_once() is True
    assert connector.get_calls == 1
    assert not any(name == "fail_stale" for name, _ in store.audit)
    finish = cast(
        tuple[AgentRunStatus, str | None, object, object, object],
        next(value for name, value in store.audit if name == "finish"),
    )
    assert finish[:2] == (AgentRunStatus.COMPLETED, "recovered")
    await processor.close()


@pytest.mark.parametrize(
    "final_result",
    [
        AgentRunSnapshot(ConnectorRunState.UNKNOWN),
        AgentConnectorError(
            "still unavailable",
            code="endpoint_unavailable",
            retryable=True,
        ),
    ],
)
async def test_stale_run_fails_when_final_status_remains_unavailable(
    final_result: AgentRunSnapshot | AgentConnectorError,
) -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.stale = [remote]
    connector = FakeConnector(store.audit)
    connector.snapshots = [final_result]
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    assert connector.get_calls == 1
    failure = cast(
        tuple[UUID, str, str],
        next(value for name, value in store.audit if name == "fail_stale"),
    )
    assert failure == (
        remote.run.id,
        "timeout",
        "Agent run status remained unavailable beyond the configured timeout",
    )
    assert remote.run.id not in processor._watchers
    await processor.close()


@pytest.mark.parametrize(
    "remote",
    [
        work(status=AgentRunStatus.UNKNOWN),
        work(
            status=AgentRunStatus.UNKNOWN,
            external_run_id="remote-1",
            connector_type="missing",
        ),
    ],
)
async def test_stale_run_without_a_reconcilable_remote_identity_fails(
    remote: AgentRunWorkItem,
) -> None:
    store = FakeStore()
    store.stale = [remote]
    connector = FakeConnector(store.audit)
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True

    assert any(name == "fail_stale" for name, _ in store.audit)
    assert connector.get_calls == 0
    await processor.close()


async def test_stale_run_still_running_is_refreshed_and_watched() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.UNKNOWN, external_run_id="remote-1")
    store.stale = [remote]
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(ConnectorRunState.RUNNING, provider_data={"fresh": True})
    ]
    connector.block_watch = True
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    await connector.watch_entered.wait()
    assert ("update", (AgentRunStatus.RUNNING, {"fresh": True})) in store.audit
    assert not any(name == "fail_stale" for name, _ in store.audit)
    assert remote.run.id in processor._watchers
    await processor.close()


async def test_stale_run_waiting_for_user_is_recovered_without_failure() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.stale = [remote]
    store.reconcilable = [remote]
    connector = FakeConnector(store.audit)
    connector.snapshots = [AgentRunSnapshot(ConnectorRunState.WAITING_USER)]
    connector.block_watch = True
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    await connector.watch_entered.wait()

    assert any(name == "open_approval" for name, _ in store.audit)
    assert not any(name == "fail_stale" for name, _ in store.audit)
    assert remote.run.id in processor._watchers
    await processor.close()


async def test_failed_stale_watcher_is_cancelled_before_capacity_is_reused() -> None:
    store = FakeStore()
    remote = work(1, status=AgentRunStatus.RUNNING, external_run_id="remote-old")
    store.stale = [remote]
    queued = work(2)
    store.queued = queued
    connector = FakeConnector(store.audit)
    connector.block_watch = True
    connector.snapshots = [AgentRunSnapshot(ConnectorRunState.UNKNOWN)]
    processor = AgentRunProcessor(store, {"hermes": connector}, concurrency=1)
    processor._start_watcher(remote, connector)
    await connector.watch_entered.wait()

    assert await processor.process_once() is True
    await wait_until(lambda: queued.run.id in processor._watchers)
    names = [name for name, _ in store.audit]
    assert connector.watch_cancelled.is_set()
    assert names.index("fail_stale") < names.index("peek") < names.index("start")
    assert remote.run.id not in processor._watchers
    await processor.close()


async def test_existing_remote_watchers_start_above_new_dispatch_capacity() -> None:
    store = FakeStore()
    store.unidentified_dispatches = 1
    store.reconcilable = [
        work(1, status=AgentRunStatus.RUNNING, external_run_id="remote-1"),
        work(2, status=AgentRunStatus.WAITING_USER, external_run_id="remote-2"),
    ]
    store.queued = work(3)
    connector = FakeConnector(store.audit)
    connector.block_watch = True
    processor = AgentRunProcessor(store, {"hermes": connector}, concurrency=1)

    assert await processor.process_once() is True
    await wait_until(lambda: sum(name == "watch" for name, _ in store.audit) == 2)
    assert not any(name == "peek" for name, _ in store.audit)
    assert [name for name, _ in store.audit[:5]] == [
        "recover_dispatches",
        "recover_approvals",
        "recover_steers",
        "list_stale",
        "list_reconcilable",
    ]
    await processor.close()
    assert connector.watch_cancelled.is_set()


async def test_reconcilable_run_pages_are_fully_drained_and_ids_pruned() -> None:
    store = FakeStore()
    store.reconcilable = [
        work(1, status=AgentRunStatus.RUNNING, external_run_id="remote-1"),
        work(2, status=AgentRunStatus.RUNNING, external_run_id="remote-2"),
    ]
    connector = FakeConnector(store.audit)
    connector.block_watch = True
    processor = AgentRunProcessor(
        store,
        {"hermes": connector},
        concurrency=1,
        reconcile_limit=1,
    )
    stale = UUID(int=999_999)
    processor._reconciled_unknown_ids.add(stale)

    await processor.process_once()
    await wait_until(lambda: sum(name == "watch" for name, _ in store.audit) == 2)

    assert stale not in processor._reconciled_unknown_ids
    pages = [value for name, value in store.audit if name == "list_reconcilable"]
    assert len(pages) == 3
    assert pages[0] == (1, None)
    second_page = cast(tuple[int, tuple[datetime, UUID] | None], pages[1])
    assert second_page[1] is not None
    await processor.close()


async def test_approval_event_and_recovered_waiting_snapshot_are_guarded() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.reconcilable = [remote]
    connector = FakeConnector(store.audit)
    connector.events = [
        AgentProviderEvent(
            AgentProviderEventKind.APPROVAL,
            ConnectorRunState.WAITING_USER,
            approval_choices=("once", "deny"),
            provider_data={"request_id": "safe-id"},
        ),
        AgentProviderEvent(
            AgentProviderEventKind.APPROVAL,
            ConnectorRunState.WAITING_USER,
            approval_choices=("once", "deny"),
            provider_data={"request_id": "safe-id"},
        ),
    ]
    processor = AgentRunProcessor(
        store, {"hermes": connector}, poll_interval_seconds=0.001
    )

    await processor.process_once()
    await wait_until(lambda: any(name == "open_approval" for name, _ in store.audit))
    approval_payloads = [
        value for name, value in store.audit if name == "open_approval"
    ]
    assert len(approval_payloads) == 1
    approval_payload = cast(Mapping[str, object], approval_payloads[0])
    assert approval_payload["allowed_decisions"] == ["once", "deny"]
    await processor.close()

    recovered_store = FakeStore()
    recovered_store.reconcilable = [remote]
    recovered_connector = FakeConnector(recovered_store.audit)
    recovered_connector.snapshots = [AgentRunSnapshot(ConnectorRunState.WAITING_USER)]
    recovered = AgentRunProcessor(
        recovered_store,
        {"hermes": recovered_connector},
        poll_interval_seconds=0.001,
    )
    await recovered.process_once()
    await wait_until(
        lambda: any(name == "open_approval" for name, _ in recovered_store.audit)
    )
    payload = next(
        value for name, value in recovered_store.audit if name == "open_approval"
    )
    assert payload == {
        "allowed_decisions": ["deny"],
        "recoveredWithoutDetails": True,
    }
    await recovered.close()


async def test_later_approval_event_opens_after_prior_interaction_resolves() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.reconcilable = [remote]
    release_second = asyncio.Event()
    connector = FakeConnector(store.audit)
    connector.events = [
        AgentProviderEvent(
            AgentProviderEventKind.APPROVAL,
            ConnectorRunState.WAITING_USER,
            approval_choices=("once", "deny"),
            provider_data={"request_id": "first"},
        ),
        release_second,
        AgentProviderEvent(
            AgentProviderEventKind.APPROVAL,
            ConnectorRunState.WAITING_USER,
            approval_choices=("session", "deny"),
            provider_data={"request_id": "second"},
        ),
    ]
    processor = AgentRunProcessor(store, {"hermes": connector})

    await processor.process_once()
    await wait_until(
        lambda: sum(name == "open_approval" for name, _ in store.audit) == 1
    )
    store.approval_is_open = False
    release_second.set()
    await wait_until(
        lambda: sum(name == "open_approval" for name, _ in store.audit) == 2
    )

    payloads = [value for name, value in store.audit if name == "open_approval"]
    assert payloads[0] != payloads[1]
    await processor.close()


async def test_terminal_event_is_authoritative_without_second_get() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.reconcilable = [remote]
    connector = FakeConnector(store.audit)
    connector.events = [
        AgentProviderEvent(
            AgentProviderEventKind.TERMINAL,
            ConnectorRunState.COMPLETED,
            output="Bearer provider-secret malformed",
            provider_data={"pending_steer": "keep this"},
        )
    ]
    processor = AgentRunProcessor(store, {"hermes": connector})

    await processor.process_once()
    await wait_until(lambda: any(name == "finish" for name, _ in store.audit))
    assert connector.get_calls == 0
    finish = cast(
        tuple[
            AgentRunStatus,
            str | None,
            str | None,
            str | None,
            Mapping[str, object] | None,
        ],
        next(value for name, value in store.audit if name == "finish"),
    )
    assert finish[0] is AgentRunStatus.COMPLETED
    assert finish[1] is not None
    assert "provider-secret" not in finish[1]
    assert finish[4] == {"pending_steer": "keep this"}
    await processor.close()


async def test_stale_watcher_applies_only_normalized_incoming_provider_metadata() -> (
    None
):
    store = FakeStore()
    remote = work(
        status=AgentRunStatus.RUNNING,
        external_run_id="remote-1",
        provider_data=None,
    )
    store.reconcilable = [remote]
    connector = FakeConnector(store.audit)
    connector.events = [
        AgentProviderEvent(
            AgentProviderEventKind.STATE,
            ConnectorRunState.RUNNING,
            provider_data=None,
        ),
        AgentProviderEvent(
            AgentProviderEventKind.TERMINAL,
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"done"}',
            provider_data={"pending_steer": "provider-next"},
        ),
    ]
    processor = AgentRunProcessor(store, {"hermes": connector})

    await processor.process_once()
    await wait_until(lambda: any(name == "finish" for name, _ in store.audit))

    update = next(value for name, value in store.audit if name == "update")
    finish = cast(
        tuple[AgentRunStatus, str | None, object, object, object],
        next(value for name, value in store.audit if name == "finish"),
    )
    assert update == (AgentRunStatus.RUNNING, None)
    assert finish[-1] == {"pending_steer": "provider-next"}
    await processor.close()


async def test_terminal_state_event_waits_for_authoritative_poll_snapshot() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.reconcilable = [remote]
    connector = FakeConnector(store.audit)
    connector.events = [
        AgentProviderEvent(
            AgentProviderEventKind.STATE,
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"stale"}',
        )
    ]
    connector.snapshots = [
        AgentRunSnapshot(
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"fresh"}',
        )
    ]
    processor = AgentRunProcessor(store, {"hermes": connector})

    await processor.process_once()
    await wait_until(lambda: any(name == "finish" for name, _ in store.audit))
    finish = cast(
        tuple[AgentRunStatus, str | None, object, object, object],
        next(value for name, value in store.audit if name == "finish"),
    )
    assert connector.get_calls == 1
    assert finish[1] == "fresh"
    await processor.close()


async def test_disconnect_polls_through_transient_get_to_terminal_once() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.RUNNING, external_run_id="remote-1")
    store.reconcilable = [remote]
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentConnectorError("temporary", code="endpoint_unavailable", retryable=True),
        AgentRunSnapshot(
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"done"}',
        ),
    ]
    processor = AgentRunProcessor(
        store, {"hermes": connector}, poll_interval_seconds=0.001
    )

    await processor.process_once()
    await wait_until(lambda: any(name == "finish" for name, _ in store.audit))
    assert connector.get_calls == 2
    assert sum(name == "finish" for name, _ in store.audit) == 1
    await processor.close()


async def test_unknown_external_id_reconciles_once_per_processor() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.UNKNOWN, external_run_id="remote-1")
    store.reconcilable = [remote]
    connector = FakeConnector(store.audit)
    connector.block_watch = True
    connector.snapshots = [AgentRunSnapshot(ConnectorRunState.UNKNOWN)]
    processor = AgentRunProcessor(
        store, {"hermes": connector}, poll_interval_seconds=0.001
    )

    await processor.process_once()
    await wait_until(lambda: connector.get_calls == 1)
    await processor.process_once()
    await asyncio.sleep(0.01)
    assert connector.get_calls == 1
    assert not any(name == "watch" for name, _ in store.audit)
    assert remote.run.id in processor._reconciled_unknown_ids
    await processor.close()


async def test_approval_response_gets_before_and_after_single_post() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    store.pending_approval = approval_work(remote)
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(ConnectorRunState.WAITING_USER),
        AgentRunSnapshot(ConnectorRunState.RUNNING),
    ]
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    names = [name for name, _ in store.audit]
    first_get = names.index("get")
    assert (
        first_get < names.index("respond") < len(names) - 1 - names[::-1].index("get")
    )
    assert sum(name == "respond" for name, _ in store.audit) == 1
    assert any(name == "resolve_approval" for name, _ in store.audit)
    await processor.close()


async def test_approval_response_resolves_before_terminal_snapshot_finishes_run() -> (
    None
):
    store = FakeStore()
    remote = work(status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    store.pending_approval = approval_work(remote)
    store.queued = remote
    store.claim_wins = False
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(ConnectorRunState.WAITING_USER),
        AgentRunSnapshot(
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"done"}',
        ),
    ]
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True

    names = [name for name, _ in store.audit]
    assert names.index("resolve_approval") < names.index("finish")
    assert sum(name == "respond" for name, _ in store.audit) == 1
    await processor.close()


async def test_recovered_approval_that_left_waiting_resolves_without_post() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    store.reconcilable_approvals = [approval_work(remote, response_state="unknown")]
    connector = FakeConnector(store.audit)
    connector.snapshots = [AgentRunSnapshot(ConnectorRunState.RUNNING)]
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True
    assert any(name == "resolve_approval" for name, _ in store.audit)
    assert not any(name == "respond" for name, _ in store.audit)
    names = [name for name, _ in store.audit]
    assert names.index("resolve_approval") < names.index("update")
    await processor.close()


async def test_terminal_pending_approval_resolves_before_finish_without_post() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    store.pending_approval = approval_work(remote)
    store.queued = remote
    store.claim_wins = False
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"done"}',
        )
    ]
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True

    names = [name for name, _ in store.audit]
    assert names.index("resolve_approval") < names.index("finish")
    assert "respond" not in names
    await processor.close()


async def test_unknown_approval_snapshot_stays_open_before_or_after_post() -> None:
    for snapshots, expected_responds in (
        ([AgentRunSnapshot(ConnectorRunState.UNKNOWN)], 0),
        (
            [
                AgentRunSnapshot(ConnectorRunState.WAITING_USER),
                AgentRunSnapshot(ConnectorRunState.UNKNOWN),
            ],
            1,
        ),
    ):
        store = FakeStore()
        remote = work(
            status=AgentRunStatus.WAITING_USER,
            external_run_id="remote-1",
        )
        store.pending_approval = approval_work(remote)
        connector = FakeConnector(store.audit)
        connector.snapshots = list(snapshots)
        processor = AgentRunProcessor(store, {"hermes": connector})

        assert await processor.process_once() is True

        names = [name for name, _ in store.audit]
        assert "resolve_approval" not in names
        assert "unknown" in names
        assert "approval_unknown" in names
        assert names.count("respond") == expected_responds
        await processor.close()


async def test_reconcilable_approval_retries_until_provider_leaves_waiting(
    monkeypatch,
) -> None:
    loop = asyncio.get_running_loop()
    now = loop.time()
    monkeypatch.setattr(loop, "time", lambda: now)
    store = FakeStore()
    remote = work(status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    store.reconcilable_approvals = [approval_work(remote, response_state="unknown")]
    store.queued = remote
    store.claim_wins = False
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(ConnectorRunState.WAITING_USER),
        AgentRunSnapshot(
            ConnectorRunState.COMPLETED,
            output='{"disposition":"completed","message":"done"}',
        ),
    ]
    processor = AgentRunProcessor(
        store,
        {"hermes": connector},
        poll_interval_seconds=0.001,
    )

    assert await processor.process_once() is True
    assert "resolve_approval" not in [name for name, _ in store.audit]
    now += 0.002
    assert await processor.process_once() is True

    names = [name for name, _ in store.audit]
    assert names.count("get") == 2
    assert names.index("resolve_approval") < names.index("finish")
    await processor.close()


async def test_waiting_approval_reconcile_is_poll_rate_limited() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    store.reconcilable_approvals = [approval_work(remote, response_state="unknown")]
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(ConnectorRunState.WAITING_USER),
        AgentRunSnapshot(ConnectorRunState.RUNNING),
    ]
    processor = AgentRunProcessor(
        store,
        {"hermes": connector},
        poll_interval_seconds=0.01,
    )

    assert await processor.process_once() is False
    assert await processor.process_once() is False
    assert connector.get_calls == 1
    await asyncio.sleep(0.012)
    assert await processor.process_once() is True
    assert connector.get_calls == 2
    assert "resolve_approval" in [name for name, _ in store.audit]
    await processor.close()


async def test_local_provider_terminal_resolves_approval_without_remote_get() -> None:
    store = FakeStore()
    remote = work(status=AgentRunStatus.COMPLETED, external_run_id="expired-run")
    store.reconcilable_approvals = [approval_work(remote, response_state="unknown")]
    connector = FakeConnector(store.audit)
    processor = AgentRunProcessor(store, {"hermes": connector})

    assert await processor.process_once() is True

    names = [name for name, _ in store.audit]
    assert "resolve_approval" in names
    assert "get" not in names
    await processor.close()


async def test_reconcilable_approval_pages_are_fully_drained_and_ids_pruned() -> None:
    store = FakeStore()
    first = work(1, status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    second = work(2, status=AgentRunStatus.WAITING_USER, external_run_id="remote-2")
    store.reconcilable_approvals = [
        approval_work(first, response_state="unknown"),
        approval_work(second, response_state="unknown"),
    ]
    connector = FakeConnector(store.audit)
    connector.snapshots = [
        AgentRunSnapshot(ConnectorRunState.RUNNING),
        AgentRunSnapshot(ConnectorRunState.RUNNING),
    ]
    processor = AgentRunProcessor(store, {"hermes": connector}, reconcile_limit=1)
    stale = UUID(int=999_998)
    processor._approval_reconcile_not_before[stale] = float("inf")

    await processor.process_once()

    assert sum(name == "resolve_approval" for name, _ in store.audit) == 2
    assert stale not in processor._approval_reconcile_not_before
    pages = [
        value for name, value in store.audit if name == "list_reconcilable_approvals"
    ]
    assert len(pages) == 3
    await processor.close()


async def test_ambiguous_approval_and_steer_are_not_requeued() -> None:
    store = FakeStore()
    approval_run = work(status=AgentRunStatus.WAITING_USER, external_run_id="remote-1")
    store.pending_approval = approval_work(approval_run)
    connector = FakeConnector(store.audit)
    connector.snapshots = [AgentRunSnapshot(ConnectorRunState.WAITING_USER)]
    connector.approval_error = AgentConnectorError(
        "uncertain", code="timeout", dispatch_uncertain=True
    )
    processor = AgentRunProcessor(store, {"hermes": connector})

    await processor.process_once()
    assert any(name == "approval_unknown" for name, _ in store.audit)
    await processor.process_once()
    assert sum(name == "respond" for name, _ in store.audit) == 1
    await processor.close()

    steer_store = FakeStore()
    steer_run = work(
        status=AgentRunStatus.RUNNING,
        external_run_id="remote-2",
        provider_data={
            "steer": {
                "state": "dispatching",
                "source_message_id": str(UUID(int=999)),
            }
        },
    )
    steer_store.pending_steer = steer_run
    steer_store.message_content = "change direction"
    steer_connector = FakeConnector(steer_store.audit)
    steer_connector.steer_error = AgentConnectorError(
        "uncertain", code="timeout", dispatch_uncertain=True
    )
    steer_processor = AgentRunProcessor(steer_store, {"hermes": steer_connector})
    await steer_processor.process_once()
    assert ("steer", "change direction") in steer_store.audit
    assert any(name == "steer_unknown" for name, _ in steer_store.audit)
    await steer_processor.process_once()
    assert sum(name == "steer" for name, _ in steer_store.audit) == 1
    await steer_processor.close()


async def test_connector_history_is_bounded_before_dispatch() -> None:
    store = FakeStore()
    store.queued = work(previous=True)
    connector = FakeConnector(store.audit)
    connector.history = tuple(
        AgentHistoryMessage("user", f"{index}:" + "界" * 9_000) for index in range(70)
    )
    connector.block_watch = True
    processor = AgentRunProcessor(store, {"hermes": connector})

    await processor.process_once()
    history = connector.requests[0].conversation_history
    assert 1 <= len(history) <= 16
    assert all(len(item.content) <= 4_000 for item in history)
    assert sum(len(item.content.encode()) for item in history) <= 64_000
    assert (
        sum(estimate_history_message_tokens(item.content) for item in history) <= 8_000
    )
    assert history[-1].content.startswith("69:")
    await processor.close()


async def test_continuation_without_safe_history_fails_before_claim() -> None:
    store = FakeStore()
    store.queued = work(previous=True)
    connector = FakeConnector(store.audit)
    connector.history_error = AgentConnectorError(
        "missing", code="endpoint_unavailable", retryable=True
    )
    processor = AgentRunProcessor(store, {"hermes": connector})

    await processor.process_once()
    assert any(name == "fail_queued" for name, _ in store.audit)
    assert not any(name in {"claim_run", "start"} for name, _ in store.audit)
    await processor.close()
