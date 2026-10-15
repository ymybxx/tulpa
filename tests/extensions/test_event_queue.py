import asyncio
import json
from uuid import UUID

import pytest

from fish_memory.extensions.models import (
    CreateInstance,
    EventDefinition,
    ExtensionError,
)
from tests.extensions.test_service import Model, connected, event, make_service


async def active(service, item):
    await service._dispatch_events(item.fish_id)
    identifier = service._owners[item.fish_id]
    await service.streams.settle(identifier, interrupt=False)
    return service.streams.get(identifier)


async def finish(service, stream):
    await service.consumed(stream.intent.id, len(stream.text), False)


def current_request(service):
    return json.loads(service.llm.calls[-1][-1]["content"])["event"]


@pytest.mark.asyncio
async def test_busy_speaking_events_wait_for_actual_playback_and_keep_fifo():
    service = make_service()
    item, _ = await connected(service)
    service.foreground = lambda: True
    try:
        payloads = [event(word) for word in ("第一条", "第二条", "第三条")]
        for payload in payloads:
            assert (await service.request(item, "push", payload))[
                "disposition"
            ] == "queued"
        assert not service.llm.calls
        assert service.state(item)["queued_events"] == 3
        service.foreground = lambda: False
        for index, word in enumerate(("第一条", "第二条", "第三条"), 1):
            stream = await active(service, item)
            assert current_request(service)["data"]["text"] == word
            assert len(service.llm.calls) == index
            # Generation completing does not release the audio owner.
            await service._dispatch_events(item.fish_id)
            assert len(service.llm.calls) == index
            await finish(service, stream)
        assert service.state(item)["can_respond"]
        assert all(
            service.store.values[UUID(p["event_id"])]["status"] == "completed"
            for p in payloads
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_merge_only_adjacent_equal_definitions_and_ack_every_original_event():
    service = make_service()
    item, _ = await connected(service)
    merge = EventDefinition(
        type="example.enter", schema={"type": "object"}, mergeable=True
    )
    await service.request(
        item, "register", {"events": [merge.model_dump(by_alias=True)]}
    )
    service.foreground = lambda: True
    inputs = [
        {**event("A"), "type": merge.type},
        {**event("B"), "type": merge.type},
        event("聊天 C"),
        {**event("D"), "type": merge.type},
    ]
    try:
        for payload in inputs:
            await service.request(item, "push", payload)
        service.foreground = lambda: False
        first = await active(service, item)
        assert current_request(service) == {
            "type": merge.type,
            "items": [{"text": "A"}, {"text": "B"}],
        }
        await finish(service, first)
        for payload in inputs[:2]:
            saved = service.store.values[UUID(payload["event_id"])]
            assert saved["status"] == "completed"
            assert saved["message_id"] == str(first.intent.id)
            assert saved["response_event_ids"] == [p["event_id"] for p in inputs[:2]]
        notifications = service.diagnostics(item.id)["events"]
        assert {
            n["params"]["event_id"]
            for n in notifications
            if n["method"] == "response" and n["params"]["status"] == "completed"
        } == {p["event_id"] for p in inputs[:2]}
        second = await active(service, item)
        assert current_request(service)["data"]["text"] == "聊天 C"
        await finish(service, second)
        third = await active(service, item)
        assert current_request(service)["data"]["text"] == "D"
        await finish(service, third)
        assert len(service.llm.calls) == 3
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_queued_retry_is_idempotent_and_conflicting_retry_is_rejected():
    service = make_service()
    item, _ = await connected(service)
    service.foreground = lambda: True
    payload = event("一条弹幕")
    try:
        await service.request(item, "push", payload)
        assert (await service.request(item, "push", payload))["duplicate"]
        with pytest.raises(ExtensionError, match="changed data"):
            await service.request(
                item, "push", {**payload, "data": {"text": "别的内容"}}
            )
        assert service.state(item)["queued_events"] == 1
        service.foreground = lambda: False
        await finish(service, await active(service, item))
        assert len(service.llm.calls) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_no_playback_waits_and_consumer_connection_wakes_queue_without_polling():
    service = make_service()
    item, _ = await connected(service)
    service.native_consumer(item.fish_id, item.session_id, False)
    try:
        receipt = await service.request(item, "push", event())
        assert receipt["disposition"] == "queued"
        assert receipt["reason"] == "no_playback_consumer"
        service.native_consumer(item.fish_id, item.session_id, True)
        await asyncio.wait_for(service._dispatch_tasks[item.fish_id], 1)
        await service.streams.settle(service._owners[item.fish_id], interrupt=False)
        assert len(service.llm.calls) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stopping_cancels_pending_requests_and_cannot_speak_later():
    service = make_service()
    item, _ = await connected(service)
    service.foreground = lambda: True
    payload = event()
    await service.request(item, "push", payload)
    await service.stop(item.id)
    service.foreground = lambda: False
    await service._dispatch_events(item.fish_id)
    assert not service.llm.calls
    saved = service.store.values[UUID(payload["event_id"])]
    assert saved["status"] == "interrupted" and saved["reason"] == "stopped"
    await service.close()


@pytest.mark.asyncio
async def test_full_queue_rejects_new_input_explicitly_but_accepts_same_id_retry(
    monkeypatch,
):
    monkeypatch.setattr("fish_memory.extensions.service.MAX_PENDING_EVENTS", 2)
    service = make_service()
    item, _ = await connected(service)
    service.foreground = lambda: True
    first = event()
    try:
        await service.request(item, "push", first)
        await service.request(item, "push", event())
        assert (await service.request(item, "push", first))["duplicate"]
        with pytest.raises(ExtensionError) as caught:
            await service.request(item, "push", event())
        assert caught.value.code == "capacity"
        assert len(service.store.values) == service.state(item)["queued_events"] == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_equal_mergeable_types_from_two_instances_remain_separate():
    service = make_service()
    first, _ = await connected(service)
    second, _ = await service.create(
        CreateInstance(
            plugin_id="another", fish_id=first.fish_id, session_id=first.session_id
        )
    )
    service.attach(second)
    definition = EventDefinition(
        type="example.enter", schema={"type": "object"}, mergeable=True
    )
    service.foreground = lambda: True
    try:
        for item, text in ((first, "来源一"), (second, "来源二")):
            await service.request(
                item, "register", {"events": [definition.model_dump(by_alias=True)]}
            )
            await service.request(
                item, "push", {**event(text), "type": definition.type}
            )
        service.foreground = lambda: False
        for item, text in ((first, "来源一"), (second, "来源二")):
            stream = await active(service, item)
            assert stream.intent.instance_id == item.id
            assert current_request(service)["data"]["text"] == text
            await finish(service, stream)
        assert len(service.llm.calls) == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_failed_generation_releases_the_next_queued_event():
    class FailOnce(Model):
        async def stream_json(self, messages, *, response_schema=None):
            self.text = (
                "{}" if not self.calls else '{"m":"neutral","a":"第二条正常回应。"}'
            )
            async for chunk in super().stream_json(
                messages, response_schema=response_schema
            ):
                yield chunk

    service = make_service(FailOnce())
    item, _ = await connected(service)
    service.foreground = lambda: True
    first, second = event("失败一条"), event("下一条")
    try:
        await service.request(item, "push", first)
        await service.request(item, "push", second)
        service.foreground = lambda: False
        await active(service, item)
        assert service.store.values[UUID(first["event_id"])]["status"] == "failed"
        following = await active(service, item)
        assert following.intent.event_id == UUID(second["event_id"])
        assert (
            following.published and current_request(service)["data"]["text"] == "下一条"
        )
        await finish(service, following)
    finally:
        await service.close()
