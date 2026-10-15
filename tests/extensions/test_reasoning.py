import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from fish_memory.answer.environment_context import render_perception_policy
from fish_memory.context_descriptions import HISTORY_SOURCE_RULES
from fish_memory.environment_frames import EnvironmentPerceptionPolicy
from fish_memory.extensions.models import ExtensionError
from fish_memory.extensions.reasoning import (
    PluginReasoning,
    ReasonRequest,
    reason_messages,
)

SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("text_mode", [False, True])
def test_reasoning_explains_screen_history_in_both_request_formats(text_mode):
    request = ReasonRequest(
        prompt="概括刚才的屏幕活动",
        schema=SCHEMA,
        fields=["history"],
        input_text="只说大概" if text_mode else None,
    )
    messages = reason_messages(
        request, {"history_text": "as: 用户打开设置并查看日志。"}, None
    )
    assert HISTORY_SOURCE_RULES in messages[0]["content"]
    assert "as: 用户打开设置并查看日志。" in messages[-1]["content"]


class Model:
    image_input_capability = "unsupported"

    def __init__(self):
        self.calls = []
        self.release = None
        self.entered = asyncio.Event()

    async def stream_json(self, messages, *, response_schema):
        self.calls.append(messages)
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        yield SimpleNamespace(content='{"topic":"一个新观点"}')


def setup():
    contexts = SimpleNamespace(
        read=AsyncMock(
            return_value={
                "history": [{"id": "u1", "role": "user", "content": "先停一下"}],
                "perception": {"available": True, "text": "当前台词"},
                "revision": "r1",
            }
        )
    )
    store = SimpleNamespace(record_reasoning=AsyncMock())
    model = Model()
    reasoner = PluginReasoning(model, contexts, store)
    instance = SimpleNamespace(
        id=uuid4(),
        fish_id=uuid4(),
        session_id=uuid4(),
        activity=None,
        definitions={},
        connected=True,
        stopped=False,
    )
    return reasoner, instance, model, store, contexts


@pytest.mark.asyncio
async def test_reasoning_uses_scoped_context_and_returns_data_with_trace():
    reasoner, item, model, store, contexts = setup()
    value = await reasoner.run(
        item, {"prompt": "提出一个有依据的新观点", "schema": SCHEMA}
    )
    assert value["value"] == {"topic": "一个新观点"}
    assert contexts.read.call_args.args[:2] == (item.fish_id, item.session_id)
    payload = json.loads(model.calls[0][-1]["content"])
    assert payload["latest_user_message"] == "先停一下"
    assert "JSON" in model.calls[0][0]["content"]
    assert payload["context"]["perception"]["recent"] == "当前台词"
    assert store.record_reasoning.call_args.kwargs["status"] == "completed"
    assert not reasoner.tasks


@pytest.mark.asyncio
async def test_reasoning_reads_live_perception_flags_after_loading_context():
    reasoner, item, model, _, contexts = setup()
    policy = EnvironmentPerceptionPolicy(False, False, False)
    contexts.environment = SimpleNamespace(
        perception_context=lambda _: render_perception_policy(policy)
    )

    async def load_context(*args, **kwargs):
        nonlocal policy
        policy = EnvironmentPerceptionPolicy(True, False, True)
        return {"revision": "r1"}

    contexts.read.side_effect = load_context
    await reasoner.run(item, {"prompt": "理解当前情况", "schema": SCHEMA})
    assert render_perception_policy(policy) in model.calls[0][0]["content"]
    assert "environment_image" not in model.calls[0][-1]


@pytest.mark.asyncio
async def test_one_inflight_call_and_cancellation_cannot_cancel_a_different_request():
    reasoner, item, model, store, _ = setup()
    model.release = asyncio.Event()
    request_id = uuid4()
    task = asyncio.create_task(
        reasoner.run(
            item, {"prompt": "选择", "schema": SCHEMA, "request_id": str(request_id)}
        )
    )
    await model.entered.wait()
    with pytest.raises(ExtensionError, match="already running"):
        await reasoner.run(item, {"prompt": "重复", "schema": SCHEMA})
    reasoner.cancel(item.id, uuid4())
    await asyncio.sleep(0)
    assert not task.done()
    reasoner.cancel(item.id, request_id)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not reasoner.tasks and not reasoner.requests
    assert store.record_reasoning.call_args.kwargs["status"] == "cancelled"


def test_reason_request_cannot_select_another_pet_or_fetch_external_schema():
    with pytest.raises(ValidationError):
        ReasonRequest(prompt="p", schema=SCHEMA, fish_id=str(uuid4()))
    with pytest.raises(ValidationError):
        ReasonRequest(
            prompt="p", schema={"type": "object", "$ref": "https://example.com/schema"}
        )
    with pytest.raises(ValidationError):
        ReasonRequest(prompt="p", schema=SCHEMA, fields=["api_keys"])


@pytest.mark.asyncio
async def test_reasoning_probes_its_own_model_before_using_current_frame():
    reasoner, item, model, _, contexts = setup()
    model.image_input_capability = "unknown"

    async def probe():
        model.image_input_capability = "supported"

    model.probe_image_input_capability = AsyncMock(side_effect=probe)
    frame = object()
    contexts.environment = SimpleNamespace(
        load_cached=AsyncMock(return_value=SimpleNamespace(image=frame))
    )
    await reasoner.run(item, {"prompt": "观察当前画面", "schema": SCHEMA})
    assert model.calls[0][-1]["environment_image"] is frame
    assert model.probe_image_input_capability.call_count == 1


@pytest.mark.asyncio
async def test_topic_only_text_reasoning_has_no_implicit_context_or_schema():
    reasoner, item, model, store, contexts = setup()
    item.activity = SimpleNamespace(model_dump=lambda: {"name": "must-not-leak"})
    await reasoner.run(
        item,
        {
            "prompt": "判断是否查资料",
            "schema": SCHEMA,
            "fields": [],
            "input_text": "电影演员的公开经历",
        },
    )
    contexts.read.assert_not_awaited()
    assert model.calls[0][-1] == {"role": "user", "content": "电影演员的公开经历"}
    assert all(
        token not in str(model.calls)
        for token in (
            "must-not-leak",
            "先停一下",
            "output_schema",
            "conversation_progress",
            "$defs",
        )
    )
    assert store.record_reasoning.call_args.kwargs["status"] == "completed"


@pytest.mark.asyncio
async def test_text_reasoning_uses_media_and_memory_without_operation_context():
    reasoner, item, model, _, contexts = setup()
    model.image_input_capability = "supported"
    frame = object()
    contexts.environment = SimpleNamespace(
        load_cached=AsyncMock(return_value=SimpleNamespace(image=frame))
    )
    contexts.read.return_value = {
        "memory": {"static": "静态信息", "dynamic": "动态信息"},
        "audiovisual": {"available": True, "text": "剧情和对白"},
        "perception": {"text": "不应加载的ACS操作"},
        "revision": "invisible-revision",
        "history": [{"id": "invisible-id"}],
    }
    await reasoner.run(
        item,
        {
            "prompt": "选择可聊内容",
            "schema": SCHEMA,
            "fields": ["memory", "audiovisual"],
            "input_text": "当前话题：topic_1",
        },
    )
    text = model.calls[0][-1]["content"]
    assert all(word in text for word in ("静态信息", "剧情和对白", "topic_1"))
    assert all(
        word not in text for word in ("ACS", "invisible", "output_schema", "动态信息")
    )
    assert model.calls[0][-1]["environment_image"] is frame


def test_text_and_structured_plugin_inputs_cannot_be_combined():
    with pytest.raises(ValidationError):
        ReasonRequest(
            prompt="p", schema=SCHEMA, input_text="plain", data={"other": True}
        )
