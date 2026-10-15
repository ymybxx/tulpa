import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fish_plugin_sdk import Activity

from fish_memory.answer.environment_context import render_perception_policy
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.background_messages.streams import ExpressionRegistry
from fish_memory.environment_frames import EnvironmentPerceptionPolicy
from fish_memory.extensions.models import (
    CreateInstance,
    EventDefinition,
    ExtensionError,
)
from fish_memory.extensions.routes import extension_router
from fish_memory.extensions.service import ExtensionService
from fish_memory.pet.hub import PetEventHub


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.before_commit = None

    @staticmethod
    def observation_id(item, event_id):
        return event_id

    async def definition(self, item, definition):
        return uuid4()

    async def receive(self, item, event_id, definition, data, disposition, reason):
        if self.before_commit:
            await self.before_commit.wait()
        if event_id in self.values:
            old = self.values[event_id]
            if old["data"] != data:
                raise ExtensionError("id_conflict", "changed data")
            return dict(old), False
        value = dict(
            observation_id=str(event_id),
            event_id=str(event_id),
            type=definition.type,
            data=data,
            disposition=disposition,
            reason=reason,
            status="queued"
            if disposition == "queued"
            else "pending"
            if disposition == "responding"
            else "stored_only",
        )
        self.values[event_id] = value
        return dict(value), True

    async def update(self, identifier, **changes):
        self.values[identifier].update(changes)

    async def result(self, item, identifier):
        if identifier not in self.values:
            raise ExtensionError("not_found", "Event not found")
        return self.values[identifier]

    async def update_many(self, identifiers, **changes):
        for identifier in identifiers:
            await self.update(identifier, **changes)


class Model:
    def __init__(self, text='{"m":"happy","a":"这倒让我想起一件有趣的事情。"}'):
        self.calls = []
        self.text = text
        self.release = None
        self.schemas = []

    async def stream_json(self, messages, *, response_schema=None):
        self.calls.append(messages)
        self.schemas.append(response_schema)
        if self.release:
            await self.release.wait()
        for part in [self.text[:20], self.text[20:]]:
            yield SimpleNamespace(content=part)


def make_service(model=None, *, on_response_state_changed=lambda: None):
    context = SimpleNamespace(
        read=AsyncMock(
            return_value={
                "history": [],
                "personality": {"name": "鱼", "role": "宠物", "setting": "好奇"},
            }
        ),
        environment=SimpleNamespace(
            load_cached=AsyncMock(return_value=SimpleNamespace(image=None))
        ),
        invalidate=Mock(),
        close=AsyncMock(),
    )
    context.read_for_speech = AsyncMock(
        side_effect=lambda *_: (context.read.return_value, [])
    )
    proactive = SimpleNamespace(
        streams=ExpressionRegistry(),
        responding_for=Mock(return_value=False),
        suspend_external=Mock(),
        resume_external=Mock(),
    )
    service = ExtensionService(
        store=MemoryStore(),
        delivery=SimpleNamespace(
            materialize_proactive_observation=AsyncMock(),
            update_proactive_content=AsyncMock(),
        ),
        contexts=context,
        llm=model or Model(),
        hub=PetEventHub(),
        proactive=proactive,
        foreground=lambda: False,
        on_response_state_changed=on_response_state_changed,
    )
    return service


async def connected(service, speak=True, *, autonomous=True):
    item, token = await service.create(
        CreateInstance(
            plugin_id="example",
            fish_id=uuid4(),
            session_id=uuid4(),
            autonomous=autonomous,
        )
    )
    service.attach(item)
    service.native_consumer(item.fish_id, item.session_id, True)
    await service.request(
        item,
        "register",
        {
            "events": [
                {
                    "type": "example.notice",
                    "schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    "speak": speak,
                }
            ]
        },
    )
    return item, token


@pytest.mark.asyncio
async def test_host_expression_reads_live_flags_even_without_an_image():
    service = make_service()
    policies = (
        EnvironmentPerceptionPolicy(True, False, True),
        EnvironmentPerceptionPolicy(False, True, False),
    )
    service.contexts.environment.perception_context = lambda _: (
        render_perception_policy(policy)
    )
    definition = EventDefinition(type="example.notice", schema={"type": "object"})
    try:
        for policy in policies:
            _, messages = await service._messages_for_event(
                uuid4(), uuid4(), definition, {}, include_image=False
            )
            assert render_perception_policy(policy) in messages[0]["content"]
            assert "environment_image" not in messages[-1]
        service.contexts.environment.load_cached.assert_not_awaited()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_app_input_reaches_plugin_without_directly_starting_expression():
    service = make_service()
    item, _ = await connected(service)
    try:
        from fish_memory.extensions.models import InputDefinition

        item.inputs["example.manual"] = InputDefinition(
            type="example.manual", title="测试", schema={"type": "object"}
        )
        body = {"event_id": str(uuid4()), "type": "example.manual", "data": {}}
        task = asyncio.create_task(service.inputs.send(item, body, service.emit))
        await asyncio.sleep(0)
        assert not service.store.values
        assert not service.llm.calls
        assert service.diagnostics(item.id)["events"][-1]["method"] == "input"
        await service.request(
            item,
            "input_result",
            {
                "request_id": body["event_id"],
                "result": {"disposition": "filtered"},
            },
        )
        assert await task == {"disposition": "filtered"}
    finally:
        await service.close()


def event(text="新事件", identifier=None):
    return {
        "event_id": str(identifier or uuid4()),
        "type": "example.notice",
        "data": {"text": text},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("autonomous", [False, True])
async def test_activity_is_scoped_to_connected_plugin_and_current_session(autonomous):
    service = make_service()
    item, _ = await connected(service, autonomous=autonomous)
    if autonomous:
        service.proactive.suspend_external.assert_called_once_with(
            item.fish_id, item.id
        )
    else:
        service.proactive.suspend_external.assert_not_called()
    activity = Activity(name="直播", role="主播", audience="观众", objective="推进内容")
    try:
        assert service.current_activity(item.fish_id, item.session_id) == ""
        await service.request(
            item,
            "register",
            {
                "events": [
                    d.model_dump(by_alias=True) for d in item.definitions.values()
                ],
                "activity": activity.model_dump(),
            },
        )
        assert "主播" in service.current_activity(item.fish_id, item.session_id)
        assert service.current_activity(uuid4(), item.session_id) == ""
        assert service.current_activity(item.fish_id, uuid4()) == ""
        queue = next(iter(service._subscribers[item.id]))
        await service.detach(item, queue)
        assert service.current_activity(item.fish_id, item.session_id) == ""
        service.attach(item)
        assert "主播" in service.current_activity(item.fish_id, item.session_id)
        await service.stop(item.id)
        assert service.current_activity(item.fish_id, item.session_id) == ""
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("speak", [False, True])
async def test_playback_session_lease_tracks_registered_speech_and_stop(speak):
    service = make_service()
    item, _ = await connected(service, speak)

    def bindings():
        return next(
            e["payload"]["sessions"]
            for e in service.hub.snapshot()["payload"]["events"]
            if e["type"] == "pet.plugin.sessions"
        )

    assert bindings() == (
        []
        if speak is False
        else [{"fishId": str(item.fish_id), "sessionId": str(item.session_id)}]
    )
    await service.stop(item.id)
    assert bindings() == []
    assert not service.llm.calls
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("speak,busy", [(False, False), (False, True)])
async def test_silent_or_busy_records_without_model(speak, busy):
    service = make_service()
    item, _ = await connected(service, speak)
    service.foreground = lambda: busy
    receipt = await service.request(item, "push", event())
    await asyncio.sleep(0)
    assert receipt["disposition"] == "stored_only"
    assert not service.llm.calls
    service.foreground = lambda: False
    await asyncio.sleep(0)
    assert not service.llm.calls  # becoming idle never replays this event
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("speak", [False, True])
@pytest.mark.parametrize("busy", [False, True])
async def test_experimental_plugin_events_never_enter_direct_speech_queue(speak, busy):
    service = make_service()
    service.proactive.experimental = SimpleNamespace(
        service=SimpleNamespace(enabled=True)
    )
    item, _ = await connected(service, speak, autonomous=False)
    service.foreground = lambda: busy
    service.native_consumer(item.fish_id, item.session_id, False)
    service.contexts.invalidate.reset_mock()
    payload = event("学脱口秀")
    try:
        receipt = await service.request(item, "push", payload)
        assert receipt["disposition"] == "stored_only"
        assert receipt["reason"] == "experimental_event"
        assert receipt["data"]["text"] == "学脱口秀"
        assert "message_id" not in receipt
        assert (await service.request(item, "push", payload))["duplicate"]
        service.contexts.invalidate.assert_called_once_with(
            item.fish_id, item.session_id
        )
        service.foreground = lambda: False
        service.native_consumer(item.fish_id, item.session_id, True)
        await service._dispatch_events(item.fish_id)
        assert not service._pending_events and not service.has_active_response()
        assert not service.llm.calls
        service.delivery.materialize_proactive_observation.assert_not_awaited()
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("experimental_enabled", [None, False, True])
async def test_duplicate_and_playback_hold_one_response(experimental_enabled):
    # Autonomous plugins retain their exclusive speech lease; live is event-only.
    service = make_service()
    if experimental_enabled is not None:
        service.proactive.experimental = SimpleNamespace(
            service=SimpleNamespace(enabled=experimental_enabled)
        )
    item, _ = await connected(service)
    payload = event()
    receipt = await service.request(item, "push", payload)
    identifier = UUID(receipt["message_id"])
    await service.streams.settle(identifier, interrupt=False)
    assert service.streams.get(identifier).published
    assert (await service.request(item, "push", payload))["duplicate"] is True
    queued = await service.request(item, "push", event())
    assert queued["disposition"] == "queued" and queued["reason"] == "pet_busy"
    assert len(service.llm.calls) == 1
    await service.consumed(identifier, len(service.streams.get(identifier).text), False)
    assert service.state(item)["can_respond"] is False
    await service._dispatch_events(item.fish_id)
    following = service._owners[item.fish_id]
    assert following != identifier
    await service.streams.settle(following, interrupt=False)
    assert len(service.llm.calls) == 2
    assert service.store.values[UUID(payload["event_id"])]["status"] == "completed"
    await service.close()


@pytest.mark.asyncio
async def test_user_arrives_while_event_commits_no_model_call():
    service = make_service()
    item, _ = await connected(service)
    service.store.before_commit = asyncio.Event()
    push = asyncio.create_task(service.request(item, "push", event()))
    await asyncio.sleep(0)
    service.foreground = lambda: True
    service.store.before_commit.set()
    receipt = await push
    assert receipt["disposition"] == "queued" and "message_id" not in receipt
    assert service.llm.calls == []
    service.foreground = lambda: False
    await service._dispatch_events(item.fish_id)
    await service.streams.settle(service._owners[item.fish_id], interrupt=False)
    assert len(service.llm.calls) == 1
    await service.close()


@pytest.mark.asyncio
async def test_auto_is_rejected_at_registration():
    service = make_service()
    try:
        with pytest.raises(ValueError, match="speak must be false or true"):
            await connected(service, "auto")
        assert not service.llm.calls
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_empty_model_object_is_visible_failure_not_silence():
    service = make_service(Model("{}"))
    item, _ = await connected(service, True)
    payload = event()
    receipt = await service.request(item, "push", payload)
    await service.streams.settle(UUID(receipt["message_id"]), interrupt=False)
    assert not service.delivery.materialize_proactive_observation.called
    assert service.state(item)["can_respond"] is True
    assert service.state(item)["last_response"]["reason"] == "invalid_response"
    saved = service.store.values[UUID(payload["event_id"])]
    assert saved["status"] == "failed"
    assert saved["error_code"] == "invalid_response"
    assert len(service.llm.calls) == 1  # no unbounded retry or forced speech
    await service.close()


@pytest.mark.asyncio
async def test_required_event_schema_cannot_select_silence():
    service = make_service()
    item, _ = await connected(service, True)
    receipt = await service.request(item, "push", event())
    await service.streams.settle(UUID(receipt["message_id"]), interrupt=False)
    assert service.llm.schemas[0]["required"] == ["m", "a"]
    assert "speak" not in str(service.llm.calls[0])
    assert service.streams.get(UUID(receipt["message_id"])).published
    await service.close()


@pytest.mark.asyncio
async def test_schema_failure_has_no_record_and_no_external_refs():
    service = make_service()
    item, _ = await connected(service)
    with pytest.raises(ExtensionError) as error:
        await service.request(item, "push", {**event(), "data": {}})
    assert error.value.code == "invalid_data"
    assert service.store.values == {}
    with pytest.raises(ValueError, match="external schema"):
        EventDefinition(
            type="x",
            schema={"type": "object", "$ref": "https://example.com/schema"},
            speak=False,
        )
    await service.close()


def test_scoped_token_cannot_use_management_or_other_pet_context():
    service = make_service()
    app = FastAPI()
    app.state.extension_service = service
    app.state.session_store = SimpleNamespace(require_owned=AsyncMock())
    app.include_router(extension_router)
    app.add_middleware(
        ApiAuthMiddleware,
        auth_service=SimpleNamespace(
            authenticate_headers=lambda headers: (
                True if dict(headers).get(b"authorization") == b"Bearer owner" else None
            )
        ),
        scoped_auth=lambda scope: (
            scope.get("type") == "websocket"
            and scope["path"] == "/v1/extensions/ws"
            and service.authenticate(list(scope["headers"])) is not None
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            "/internal/extensions/instances",
            headers={"Authorization": "Bearer owner"},
            json={
                "plugin_id": "test",
                "fish_id": str(uuid4()),
                "session_id": str(uuid4()),
            },
        )
        assert response.status_code == 200
        token = response.json()["token"]
        assert (
            client.post(
                "/internal/extensions/instances",
                headers={"Authorization": "Bearer " + token},
                json={},
            ).status_code
            == 401
        )
        with client.websocket_connect(
            "/v1/extensions/ws?protocol=1", headers={"Authorization": "Bearer " + token}
        ) as socket:
            assert socket.receive_json()["method"] == "state"
            socket.send_json(
                {
                    "id": "1",
                    "method": "context",
                    "params": {"fields": ["personality"], "fish_id": str(uuid4())},
                }
            )
            result = socket.receive_json()
            assert "result" in result
            assert (
                service.contexts.read.call_args.args[0]
                == next(iter(service.instances.values())).fish_id
            )
        client.delete(
            "/internal/extensions/instances/" + response.json()["instance_id"],
            headers={"Authorization": "Bearer owner"},
        )


@pytest.mark.asyncio
async def test_concurrent_stop_does_not_reopen_a_completed_response():
    service = make_service()
    item, _ = await connected(service)
    receipt = await service.request(item, "push", event())
    identifier = UUID(receipt["message_id"])
    await service.streams.settle(identifier, interrupt=False)

    async def acknowledge(_fish, message):
        await service.consumed(message, 3, True)

    service.hub.request_answer_interrupt = AsyncMock(side_effect=acknowledge)
    await asyncio.gather(
        service._cancel_owned(item, "user_busy"), service._cancel_owned(item, "stopped")
    )
    assert service.hub.request_answer_interrupt.call_count == 1
    assert service.streams.get(identifier).intent.state == "interrupted"
    assert service.store.values[UUID(receipt["event_id"])]["status"] == "interrupted"
    await service.close()


@pytest.mark.asyncio
async def test_consumption_write_failure_can_be_retried():
    service = make_service()
    item, _ = await connected(service)
    receipt = await service.request(item, "push", event())
    identifier = UUID(receipt["message_id"])
    await service.streams.settle(identifier, interrupt=False)
    update = service.store.update
    service.store.update = AsyncMock(side_effect=OSError("temporary database failure"))
    with pytest.raises(OSError):
        await service.consumed(identifier, 3, True)
    assert service.state(item)["reason"] == "pet_busy"
    service.store.update = update
    await service.consumed(identifier, 3, True)
    assert service.state(item)["can_respond"] is True
    await service.close()


def test_reasoning_does_not_block_state_or_cancel_messages_on_the_socket():
    service = make_service()
    model = Model()
    model.release = asyncio.Event()
    service.reasoning.llm = model
    app = FastAPI()
    app.state.extension_service = service
    app.state.session_store = SimpleNamespace(require_owned=AsyncMock())
    app.include_router(extension_router)
    with TestClient(app) as client:
        instance = client.post(
            "/internal/extensions/instances",
            json={
                "plugin_id": "test",
                "fish_id": str(uuid4()),
                "session_id": str(uuid4()),
            },
        ).json()
        with client.websocket_connect(
            "/v1/extensions/ws?protocol=1",
            headers={"Authorization": "Bearer " + instance["token"]},
        ) as ws:
            assert ws.receive_json()["method"] == "state"
            request_id = str(uuid4())
            ws.send_json(
                {
                    "id": "slow",
                    "method": "reason",
                    "params": {
                        "prompt": "返回数据",
                        "schema": {"type": "object"},
                        "request_id": request_id,
                    },
                }
            )
            ws.send_json({"id": "quick", "method": "state", "params": {}})
            assert ws.receive_json()["id"] == "quick"
            ws.send_json(
                {
                    "id": "cancel",
                    "method": "cancel_reason",
                    "params": {"request_id": request_id},
                }
            )
            replies = {r["id"]: r for r in (ws.receive_json(), ws.receive_json())}
            assert replies["cancel"]["result"]["cancelled"]
            assert replies["slow"]["error"]["code"] == "reasoning_cancelled"
        client.delete("/internal/extensions/instances/" + instance["instance_id"])
