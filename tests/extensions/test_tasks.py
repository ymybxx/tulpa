import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from fish_memory.agent.autonomous import AutonomousRunPolicy, autonomous_run_input
from fish_memory.agent.models import AgentRunStatus
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.agent.session_context import AgentSessionContextLoader
from fish_memory.extensions.models import ExtensionError, Instance
from fish_memory.extensions.tasks import PluginTasks
from tests.agent.test_processor import work as example_work


class Store:
    def __init__(self):
        self.items = {}

    async def by_request(self, fish, session, instance, generation, request):
        return next(
            (
                w
                for w in self.items.values()
                if w.run.fish_id == fish
                and w.run.session_id == session
                and w.run.origin_context["instanceId"] == str(instance)
                and w.run.origin_context["generation"] == str(generation)
                and w.run.origin_context["requestId"] == str(request)
            ),
            None,
        )

    async def create(self, **kwargs):
        base = example_work()
        current = replace(
            base,
            run=replace(
                base.run,
                id=uuid4(),
                fish_id=kwargs["fish_id"],
                session_id=kwargs["session_id"],
                source_message_id=None,
                previous_run_id=None,
                origin="plugin",
                origin_context=kwargs["origin_context"],
                input=kwargs["purpose"],
                status=AgentRunStatus.QUEUED,
            ),
        )
        self.items[current.run.id] = current
        return current

    async def get(self, identifier):
        return self.items.get(identifier)

    async def active(self):
        return tuple(w for w in self.items.values() if not w.run.status.is_terminal)

    async def request_stop(self, identifier):
        w = self.items[identifier]
        w = replace(
            w,
            run=replace(
                w.run,
                status=AgentRunStatus.CANCELLED,
                origin_context={**w.run.origin_context, "stopRequested": True},
            ),
        )
        self.items[identifier] = w
        return w


def setup():
    item = Instance(
        uuid4(), "example", uuid4(), uuid4(), True, "scoped", connected=True
    )
    store = Store()
    targets = SimpleNamespace(resolve_for_fish=AsyncMock(return_value=example_work()))
    policy = AutonomousRunPolicy(store, store, {}, timeout_seconds=900, origin="plugin")
    emit = Mock()
    service = PluginTasks(
        store,
        targets,
        policy,
        default_target_id=None,
        instances={item.id: item},
        emit=emit,
    )
    return service, item, emit


def request():
    return {
        "prompt": "研究对象 A",
        "request_id": str(uuid4()),
        "context_key": "A:1",
        "reference": {"topic": "A", "audio": "冻结的原话"},
        "context_revision": "captured",
    }


@pytest.mark.asyncio
async def test_live_toggle_preserves_plugin_identity_and_allows_read_and_cancel():
    service, item, _ = setup()
    generation = item.generation
    runtime = AgentRuntimeState(enabled=False, default_target_id=None)
    service._runtime_state = runtime
    params = request()
    with pytest.raises(ExtensionError, match="disabled"):
        await service.request(item, "start", params)
    service.targets.resolve_for_fish.assert_not_called()
    target_id = uuid4()
    runtime.configure(enabled=True, default_target_id=target_id)
    started = await service.request(item, "start", params)
    service.targets.resolve_for_fish.assert_awaited_with(item.fish_id, target_id)
    runtime.configure(enabled=False, default_target_id=target_id)
    # An ACK retry remains idempotent even after disabling new work.
    assert await service.request(item, "start", params) == started
    assert (
        await service.request(item, "get", {"run_id": started["run_id"]})
    ) == started
    assert (await service.request(item, "cancel", {"run_id": started["run_id"]}))[
        "stop_requested"
    ]
    assert item.generation == generation and item.connected and not item.stopped


@pytest.mark.asyncio
async def test_submission_freezes_sdk_material_and_lost_ack_recovers_identity():
    service, item, _ = setup()
    params = request()
    receipt = await asyncio.wait_for(service.request(item, "start", params), 0.5)
    work = service.store.items[UUID(receipt["run_id"])]
    assert receipt["status"] == "queued" and receipt["context_key"] == "A:1"
    assert "冻结的原话" in autonomous_run_input(work)
    assert await AgentSessionContextLoader(Mock()).load_for_run(work) == ()
    assert await service.request(item, "start", params) == receipt
    assert (
        await service.request(item, "get", {"request_id": params["request_id"]})
        == receipt
    )
    assert len(service.store.items) == 1
    with pytest.raises(ExtensionError, match="request_id input changed"):
        await service.request(item, "start", {**params, "prompt": "另一个问题"})
    with pytest.raises(ValueError):
        await service.request(item, "start", {**params, "fish_id": str(uuid4())})


@pytest.mark.asyncio
async def test_cross_instance_generation_and_session_cannot_read_or_cancel():
    service, item, _ = setup()
    receipt = await service.request(item, "start", request())
    query = {"run_id": receipt["run_id"]}
    for changed in (
        replace(item, id=uuid4()),
        replace(item, generation=uuid4()),
        replace(item, session_id=uuid4()),
        replace(item, fish_id=uuid4()),
    ):
        for operation in ("get", "result", "cancel"):
            with pytest.raises(ExtensionError, match="unavailable for this instance"):
                await service.request(changed, operation, query)
    await service.stop_instance(replace(item, id=uuid4()))
    assert (await service.request(item, "get", query))["status"] == "queued"
    await service.stop_instance(item)
    assert (await service.request(item, "get", query))["stop_requested"]


@pytest.mark.asyncio
async def test_bounded_unicode_result_is_not_speech_and_cancelled_result_unusable():
    service, item, emit = setup()
    receipt = await service.request(item, "start", request())
    identifier = UUID(receipt["run_id"])
    current = service.store.items[identifier]
    service.store.items[identifier] = replace(
        current,
        run=replace(
            current.run, status=AgentRunStatus.COMPLETED, output="中文🙂" * 8000
        ),
    )
    text, cursor = "", 0
    while True:
        part = await service.request(
            item, "result", {"run_id": str(identifier), "cursor": cursor}
        )
        assert len(part["text"].encode()) <= 16000
        text += part["text"]
        cursor = part["next_cursor"]
        if cursor is None:
            break
    assert text == "中文🙂" * 8000
    assert not emit.called  # Completion cannot directly create speech.
    await service.request(item, "cancel", {"run_id": str(identifier)})
    with pytest.raises(ExtensionError, match="no usable completed result"):
        await service.request(item, "result", {"run_id": str(identifier)})


@pytest.mark.asyncio
async def test_plugin_approval_uses_its_policy_and_never_opens_user_interaction():
    from fish_memory.agent.connector import (
        AgentProviderEvent,
        AgentProviderEventKind,
        ConnectorRunState,
    )
    from fish_memory.agent.processor import AgentRunProcessor
    from tests.agent.test_processor import FakeStore
    from tests.proactive.test_autonomous import Records, Remote

    records = Records()
    records.work = replace(records.work, run=replace(records.work.run, origin="plugin"))
    remote = Remote(ConnectorRunState.CANCELLED)
    policy = AutonomousRunPolicy(
        records, records, {"hermes": remote}, timeout_seconds=900, origin="plugin"
    )
    ordinary = FakeStore()
    processor = AgentRunProcessor(ordinary, {"hermes": remote}, plugin_policy=policy)
    try:
        done = await processor._apply_event(
            records.work,
            AgentProviderEvent(
                kind=AgentProviderEventKind.APPROVAL,
                state=ConnectorRunState.WAITING_USER,
                approval_choices=("allow", "deny"),
            ),
        )
        assert done and remote.stops == 1
        assert not ordinary.approval_is_open
    finally:
        await processor.close()
