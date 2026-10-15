import asyncio
import json
from copy import deepcopy
from uuid import uuid4

import pytest

from fish_memory.extensions.event_context import (
    event_data_view,
    event_history_text,
    event_history_view,
    event_schema_view,
)
from fish_memory.extensions.models import EventDefinition
from tests.extensions.test_prompt import expression, historical_payload
from tests.extensions.test_reasoning import SCHEMA, setup
from tests.extensions.test_service import connected, make_service


def definition():
    return EventDefinition(
        type="live.reply",
        schema={
            "type": "object",
            "properties": {
                "platform": {
                    "type": "string",
                    "enum": ["bilibili", "mock"],
                    "x-fish-context": "metadata",
                },
                "room_id": {"type": "string", "x-fish-context": "metadata"},
                "viewer_id": {"type": "string", "x-fish-context": "identity"},
                "viewer_name": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["platform", "room_id", "viewer_id", "viewer_name", "text"],
        },
    )


def data(platform="mock", viewer="same-viewer"):
    return {
        "platform": platform,
        "room_id": "local-test" if platform == "mock" else "123",
        "viewer_id": viewer,
        "viewer_name": "大葱",
        "text": "小美女",
    }


def test_mock_and_platform_have_the_same_model_visible_payload_and_contract():
    event = definition()
    before = event.model_dump()
    visible = event_data_view(data(), event)
    assert visible == event_data_view(data("bilibili"), event)
    assert set(visible) == {"viewer_id", "viewer_name", "text"}
    assert visible["viewer_id"].startswith("speaker-")
    assert visible != event_data_view(data(viewer="other-viewer"), event)
    schema = event_schema_view(event)
    assert schema["schema"]["required"] == ["viewer_id", "viewer_name", "text"]
    assert "mock" not in json.dumps(schema)
    assert event.model_dump() == before


def test_shared_viewer_identity_connects_chat_entry_and_gift_with_legacy_chat_id():
    chat = definition()
    identity = event_data_view(data(), chat)["viewer_id"]
    for event_type in ("live.reply", "live.enter", "live.gift"):
        event = definition().model_copy(update={"type": event_type}, deep=True)
        event.schema_["properties"]["viewer_id"]["x-fish-identity-scope"] = "live.reply"
        original = event.model_dump()
        assert event_data_view(data(), event)["viewer_id"] == identity
        assert (
            event_data_view(data(viewer="someone-else"), event)["viewer_id"] != identity
        )
        assert "x-fish-identity-scope" not in json.dumps(event_schema_view(event))
        assert event.model_dump() == original
    unrelated = definition().model_copy(update={"type": "other.notice"})
    assert event_data_view(data(), unrelated)["viewer_id"] != identity


def test_history_narrative_substitutes_once_and_does_not_change_source_or_schema():
    event = definition()
    event.schema_["x-fish-history-template"] = (
        "直播间观众「${viewer_name}」发来弹幕：${text}"
    )
    raw = {**data(), "text": '原话含有 ${viewer_id} 和 {"a": "文字"}'}
    before = deepcopy(raw)
    assert event_history_text(raw, event) == (
        '直播间观众「大葱」发来弹幕：原话含有 ${viewer_id} 和 {"a": "文字"}'
    )
    assert raw == before
    assert "x-fish-history-template" not in str(event_schema_view(event))
    assert "x-fish-history-template" in event.schema_
    assert event_history_text(data(), definition()) is None
    assert event_history_text({"viewer_name": "大葱"}, event) is None


@pytest.mark.parametrize(
    "template", [123, "", "${unknown}", "${room_id}", "${viewer_id}", "${text"]
)
def test_history_template_rejects_invalid_or_internal_fields(template):
    value = definition().model_dump(by_alias=True)
    value["schema"]["x-fish-history-template"] = template
    with pytest.raises(ValueError):
        EventDefinition.model_validate(value)


def test_speech_hides_sources_in_current_and_older_events_without_editing_history():
    event = definition()
    old = {
        "type": "live.reply",
        "data": data(viewer="mock:viewer"),
        "status": "completed",
    }
    context = {"events": [old]}
    original = deepcopy(context)
    messages = expression([], definition=event, data=old["data"], context=context)
    assert all(
        marker not in str(messages)
        for marker in ["mock", "local-test", "platform", "room_id"]
    )
    payload = json.loads(messages[-1]["content"])["event"]
    historical = historical_payload(messages[-2])
    assert payload["data"]["viewer_id"] == historical["data"]["viewer_id"]
    assert context == original
    assert event_history_view([old], {}) == [old]


@pytest.mark.asyncio
async def test_planner_model_receives_the_same_projected_events():
    reasoner, item, model, _, contexts = setup()
    item.definitions = {"live.reply": definition()}
    raw = {"type": "live.reply", "data": data(viewer="mock:viewer")}
    contexts.read.return_value["events"] = [raw]
    await reasoner.run(
        item, {"prompt": "接续观众交流", "schema": SCHEMA, "fields": ["events"]}
    )
    payload = json.loads(model.calls[0][-1]["content"])
    assert payload["context"]["events"][0]["data"] == event_data_view(
        raw["data"], definition()
    )
    assert "mock" not in str(model.calls)
    assert contexts.read.return_value["events"] == [raw]


@pytest.mark.asyncio
async def test_sdk_context_is_projected_but_cached_source_is_unchanged():
    service = make_service()
    item, _ = await connected(service)
    event = definition()
    item.definitions[event.type] = event
    raw = {"type": event.type, "data": data(viewer="mock:viewer")}
    service.contexts.read.return_value["events"] = [raw]
    try:
        context = await service.request(item, "context", {"fields": ["events"]})
        assert context["events"][0]["data"] == event_data_view(raw["data"], event)
        assert service.contexts.read.return_value["events"] == [raw]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_expression_generation_keeps_raw_event_only_in_diagnostics():
    service = make_service()
    item, _ = await connected(service)
    event = definition()
    raw = data(viewer="mock:viewer")
    identifier = uuid4()
    try:
        await service.request(
            item, "register", {"events": [event.model_dump(by_alias=True)]}
        )
        receipt = await service.request(
            item,
            "push",
            {
                "event_id": str(identifier),
                "type": event.type,
                "data": raw,
            },
        )
        assert receipt["disposition"] == "responding"
        await asyncio.gather(*tuple(service._background))
        assert service.store.values[identifier]["data"] == raw
        assert "mock" not in str(service.llm.calls)
        visible = json.loads(service.llm.calls[0][-1]["content"])["event"]["data"]
        assert visible == event_data_view(raw, event)
    finally:
        await service.close()
