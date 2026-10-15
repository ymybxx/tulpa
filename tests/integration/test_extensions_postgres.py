import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest

from fish_memory.background_messages.models import BackgroundMessageSource
from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.background_messages.streams import ExpressionRegistry
from fish_memory.db.facts import FactStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import run_migrations
from fish_memory.extensions.models import CreateInstance, EventDefinition, Instance
from fish_memory.extensions.service import ExtensionService
from fish_memory.extensions.store import ExtensionStore
from fish_memory.pet.hub import PetEventHub


@pytest.mark.asyncio
async def test_queued_merge_keeps_individual_receipts_and_complete_fact_context(
    postgres_pool,
):
    from tests.extensions.test_service import make_service

    pool = postgres_pool
    await run_migrations(pool)
    messages = MessagesStore(pool)
    user = await messages.insert_user_message(uuid4(), "虚构合并测试")
    store = ExtensionStore(pool)
    delivery = BackgroundMessageStore(pool, messages_store=messages)
    service = make_service()
    service.store, service.delivery = store, delivery
    item, _ = await service.create(
        CreateInstance(
            plugin_id="test.merge", fish_id=user.fish_id, session_id=user.session_id
        )
    )
    service.attach(item)
    service.native_consumer(item.fish_id, item.session_id, True)
    definition = EventDefinition(
        type="example.enter", schema={"type": "object"}, mergeable=True
    )
    await service.request(
        item, "register", {"events": [definition.model_dump(by_alias=True)]}
    )
    service.foreground = lambda: True
    event_ids = [uuid4(), uuid4()]
    try:
        for identifier, name in zip(
            event_ids, ("虚构观众甲", "虚构观众乙"), strict=True
        ):
            receipt = await service.request(
                item,
                "push",
                {
                    "event_id": str(identifier),
                    "type": definition.type,
                    "data": {"name": name},
                },
            )
            assert receipt["disposition"] == "queued"
            assert (await store.result(item, identifier))["status"] == "queued"
        service.foreground = lambda: False
        await service._dispatch_events(item.fish_id)
        message_id = service._owners[item.fish_id]
        await service.streams.settle(message_id, interrupt=False)
        stream = service.streams.get(message_id)
        assert stream.published and len(service.llm.calls) == 1
        await delivery.finalize_consumption(
            message_id, item.fish_id, item.session_id, len(stream.text), False
        )
        await service.consumed(message_id, len(stream.text), False)
        for identifier in event_ids:
            saved = await store.result(item, identifier)
            assert saved["status"] == "completed"
            assert saved["message_id"] == str(message_id)
            assert saved["response_event_ids"] == list(map(str, event_ids))
        await messages.insert_user_message(
            item.fish_id, "下一条虚构输入", session_id=item.session_id
        )
        pending = await FactStore(pool).list_pending_messages(limit=5, batch_size=10)
        context = next(p.external_event for p in pending if p.message.id == message_id)
        assert "虚构观众甲" in context and "虚构观众乙" in context
        assert "response_events" in context
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM background_assistant_deliveries "
                "WHERE message_id=$1",
                message_id,
            )
            == 1
        )
        service.foreground = lambda: True
        queued_id = uuid4()
        await service.request(
            item,
            "push",
            {
                "event_id": str(queued_id),
                "type": definition.type,
                "data": {"name": "重启前等待者"},
            },
        )
        await store.recover()
        assert (await store.result(item, queued_id))["reason"] == "host_restarted"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_plugin_event_message_progress_fact_and_restart(postgres_pool):
    pool = postgres_pool
    await run_migrations(pool)
    messages = MessagesStore(pool)
    user = await messages.insert_user_message(
        uuid4(), "测试独立插件", created_at=datetime.now(UTC)
    )
    store = ExtensionStore(pool)
    delivery = BackgroundMessageStore(pool, messages_store=messages)

    class Model:
        async def stream_json(self, messages, *, response_schema=None):
            yield SimpleNamespace(
                content='{"m":"happy","a":"你好小明，听起来你也喜欢爵士乐。"}'
            )

    contexts = SimpleNamespace(
        read=AsyncMock(return_value={"history": [], "personality": {}}),
        read_for_speech=AsyncMock(
            return_value=({"history": [], "personality": {}}, [])
        ),
        invalidate=Mock(),
        close=AsyncMock(),
        environment=SimpleNamespace(
            load_cached=AsyncMock(return_value=SimpleNamespace(image=None))
        ),
    )
    proactive = SimpleNamespace(
        streams=ExpressionRegistry(),
        responding_for=Mock(return_value=False),
        suspend_external=Mock(),
        resume_external=Mock(),
    )
    service = ExtensionService(
        store=store,
        delivery=delivery,
        contexts=contexts,
        llm=Model(),
        hub=PetEventHub(),
        proactive=proactive,
        foreground=lambda: False,
    )
    try:
        item, _ = await service.create(
            CreateInstance(
                plugin_id="live", fish_id=user.fish_id, session_id=user.session_id
            )
        )
        service.attach(item)
        service.native_consumer(item.fish_id, item.session_id, True)
        definition = EventDefinition(
            type="live.reply", schema={"type": "object"}, speak=True
        )
        await service.request(
            item, "register", {"events": [definition.model_dump(by_alias=True)]}
        )
        event_id = uuid4()
        payload = {
            "event_id": str(event_id),
            "type": "live.reply",
            "data": {
                "viewer_id": "external-42",
                "viewer_name": "小明",
                "text": "我喜欢爵士乐",
            },
        }
        receipt = await service.request(item, "push", payload)
        message_id = UUID(receipt["message_id"])
        await service.streams.settle(message_id, interrupt=False)
        stream = service.streams.get(message_id)
        assert stream.published and not stream.failed
        assert (await service.request(item, "push", payload))["duplicate"]
        pending = await delivery.list_incomplete_messages_for_delivery(
            item.fish_id,
            item.session_id,
            after_sequence=0,
            through_sequence=2**63 - 1,
            limit=10,
        )
        assert len(pending) == 1 and pending[0].source is BackgroundMessageSource.PLUGIN
        row = await pool.fetchrow(
            "SELECT content,consumed_offset,fact_extraction_status "
            "FROM messages WHERE id=$1",
            message_id,
        )
        assert row["fact_extraction_status"] == "waiting_consumption"
        assert row["consumed_offset"] == 0
        # Keep only the device-confirmed segment when stopping halfway.
        offset = len("你好小明，")
        await delivery.finalize_consumption(
            message_id, item.fish_id, item.session_id, offset, True
        )
        await service.consumed(message_id, offset, True)
        result = await store.result(item, event_id)
        assert result["text"] == "你好小明，" and result["status"] == "interrupted"
        await messages.insert_user_message(
            item.fish_id,
            "下一条",
            session_id=item.session_id,
            created_at=datetime.now(UTC),
        )
        facts = await FactStore(pool).list_pending_messages(limit=5, batch_size=10)
        external = next(p for p in facts if p.message.id == message_id)
        assert "external-42" in external.external_event
        assert "小明" in external.external_event
        # Restart cannot recreate / replay a received event.
        await store.recover()
        assert (await store.result(item, event_id))["status"] == "interrupted"
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM background_assistant_deliveries "
                "WHERE source_observation_id=$1",
                UUID(receipt["observation_id"]),
            )
            == 1
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_review_window_separates_runs_and_reads_actual_prefix(postgres_pool):
    pool = postgres_pool
    await run_migrations(pool)
    messages = MessagesStore(pool)
    user = await messages.insert_user_message(
        uuid4(),
        "只评价这次插件运行",
        created_at=datetime.now(UTC) - timedelta(seconds=2),
    )
    store = ExtensionStore(pool)
    instance = Instance(
        uuid4(), "live", user.fish_id, user.session_id, True, "test-token"
    )
    definition = EventDefinition(
        type="live.idle",
        schema={"type": "object", "description": "你正在主持直播。"},
        speak=True,
    )
    await store.definition(instance, definition)
    event_id = uuid4()
    receipt, _ = await store.receive(
        instance, event_id, definition, {}, "responding", None
    )
    answer = await messages.insert_generating_answer(
        user.fish_id, user.session_id, user.sequence
    )
    await pool.execute(
        "UPDATE messages SET content='已播。未播尾巴',consumed_offset=3,"
        "status='interrupted' WHERE id=$1",
        answer.id,
    )
    await store.update(
        UUID(receipt["observation_id"]),
        status="interrupted",
        text="已播。",
        consumed_offset=3,
        message_id=str(answer.id),
        finished_at=datetime.now(UTC).isoformat(),
    )
    await pool.execute(
        """INSERT INTO background_assistant_deliveries
        (message_id,source,emotion,interrupt_policy,supersession_key,source_observation_id)
        VALUES ($1,'plugin','neutral','drop','test-review',$2)""",
        answer.id,
        UUID(receipt["observation_id"]),
    )
    chat = await messages.insert_generating_answer(
        user.fish_id, user.session_id, user.sequence
    )
    await pool.execute(
        """UPDATE messages SET content='知道了。未播尾巴',consumed_offset=4,
        status='completed',created_at=$2,finished_at=$2 WHERE id=$1""",
        chat.id,
        datetime.now(UTC) - timedelta(seconds=1),
    )
    await pool.execute(
        """INSERT INTO message_performance_steps
        (message_id,step_name,started_offset_us,duration_us,input_payload)
        VALUES ($1,'llm.answer.plugin',0,120000,$2::jsonb)""",
        answer.id,
        json.dumps(
            {
                "model": "captured-model",
                "extensionContext": {
                    "activity": {"name": "直播", "role": "主播"},
                    "latest_user_input": "你来主持",
                    "history_format": "quoted_records",
                    "history_messages": 12,
                },
                "messages": [
                    {
                        "role": "user",
                        "environmentImage": {
                            "observedAt": "2026-09-11T06:09:12+00:00",
                            "width": 960,
                            "height": 623,
                        },
                        "content": json.dumps(
                            {
                                "current_context": {
                                    "perception": {
                                        "available": True,
                                        "text": "当时正在看视频",
                                    }
                                }
                            }
                        ),
                    }
                ],
            }
        ),
    )
    old_run = instance.generation
    instance.generation = uuid4()
    await store.receive(instance, uuid4(), definition, {}, "stored_only", "pet_busy")
    now = datetime.now(UTC)
    evidence = await store.review_window(
        instance.id,
        user.fish_id,
        user.session_id,
        old_run,
        now - timedelta(minutes=1),
        now,
    )
    assert evidence["total_events"] == 1
    event = evidence["events"][0]
    assert event["event_id"] == str(event_id) and event["text"] == "已播。"
    assert event["perception"]["text"] == "当时正在看视频"
    assert event["model"] == "captured-model"
    assert event["screen_frame"]["width"] == 960
    assert event["context_audit"]["activity"]["role"] == "主播"
    assert event["context_audit"]["latest_user_input"] == "你来主持"
    assert evidence["definitions"][0]["description"] == "你正在主持直播。"
    conversation = evidence["conversation_messages"]
    assert conversation[0]["text"] == "只评价这次插件运行"
    assert len(conversation) == 2 and conversation[1]["message_id"] == str(chat.id)
    assert conversation[1]["reply_to_message_id"] == str(user.id)
    assert conversation[1]["consumed_offset"] == 4
    empty = await store.review_window(
        instance.id, uuid4(), user.session_id, old_run, now - timedelta(minutes=1), now
    )
    assert not empty["events"] and not empty["conversation_messages"]


@pytest.mark.asyncio
async def test_review_includes_failure_that_finishes_after_previous_window(
    postgres_pool,
):
    pool = postgres_pool
    await run_migrations(pool)
    user = await MessagesStore(pool).insert_user_message(uuid4(), "late result")
    store = ExtensionStore(pool)
    instance = Instance(uuid4(), "live", user.fish_id, user.session_id, True, "token")
    definition = EventDefinition(
        type="live.idle", schema={"type": "object"}, speak=True
    )
    receipt, _ = await store.receive(
        instance, uuid4(), definition, {}, "responding", None
    )
    identifier = UUID(receipt["observation_id"])
    now = datetime.now(UTC)
    await pool.execute(
        "UPDATE observations SET started_at=$2,ended_at=$2 WHERE id=$1",
        identifier,
        now - timedelta(seconds=30),
    )
    await store.update(
        identifier,
        status="failed",
        generated_at=now.isoformat(),
        error_code="invalid_response",
    )
    evidence = await store.review_window(
        instance.id,
        user.fish_id,
        user.session_id,
        instance.generation,
        now - timedelta(seconds=5),
        now + timedelta(seconds=1),
    )
    assert evidence["total_events"] == 1
    assert evidence["events"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_trigger_history_and_reasoning_trace_use_existing_observations(
    postgres_pool,
):
    pool = postgres_pool
    await run_migrations(pool)
    store = ExtensionStore(pool)
    messages = MessagesStore(pool)
    user = await messages.insert_user_message(
        uuid4(), "测试 SDK 推理记录", created_at=datetime.now(UTC)
    )
    item = Instance(
        uuid4(), "test.reason", user.fish_id, user.session_id, False, "scoped"
    )
    external = EventDefinition(type="sample.input", schema={"type": "object"})
    trigger = EventDefinition(
        type="sample.idle", schema={"type": "object"}, include_in_context=False
    )
    await store.definition(item, external)
    await store.definition(item, trigger)
    await store.receive(
        item, uuid4(), external, {"text": "真实外部信息"}, "stored_only", "silent_event"
    )
    for _ in range(15):
        await store.receive(
            item,
            uuid4(),
            trigger,
            {"prior": "不需要递归重放"},
            "stored_only",
            "pet_busy",
        )
    recent = await store.recent(user.fish_id, user.session_id, 12)
    assert len(recent) == 1 and recent[0]["data"]["text"] == "真实外部信息"
    rid = uuid4()
    await store.record_reasoning(
        item,
        rid,
        input_payload={"model": "test-reasoning"},
        result={"topic": "下一步"},
        status="completed",
        duration_ms=800,
    )
    row = await pool.fetchrow("SELECT payload FROM observations WHERE id=$1", rid)
    value = (
        json.loads(row["payload"])
        if isinstance(row["payload"], str)
        else row["payload"]
    )
    assert value["session_id"] == str(user.session_id) and value["run_id"] == str(
        item.generation
    )
    assert value["result"] == {"topic": "下一步"}
