from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from fish_memory.extensions.context import ExtensionContext


def source():
    fishes = SimpleNamespace(get_profile=AsyncMock())
    history = SimpleNamespace(load_history_readonly=AsyncMock(return_value=[]))
    snapshots = SimpleNamespace(
        load_normalized_static_profile=AsyncMock(return_value=None),
        load_normalized_dynamic_snapshot=AsyncMock(return_value=None),
    )
    environment = SimpleNamespace(load_cached=AsyncMock())
    events = SimpleNamespace(recent=AsyncMock(return_value=[]))
    media = SimpleNamespace(
        get_latest_media_context_summary=AsyncMock(
            return_value=SimpleNamespace(summary="媒体内容")
        )
    )
    return ExtensionContext(
        fishes, history, snapshots, environment, events, audiovisual=media
    )


@pytest.mark.asyncio
async def test_empty_context_is_an_actual_no_read_operation():
    context = source()
    assert await context.read(uuid4(), uuid4(), []) == {}
    context.fishes.get_profile.assert_not_awaited()
    context.history.load_history_readonly.assert_not_awaited()
    context.snapshots.load_normalized_static_profile.assert_not_awaited()
    context.environment.load_cached.assert_not_awaited()
    context.events.recent.assert_not_awaited()
    context.audiovisual.get_latest_media_context_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_media_request_does_not_load_combined_operation_history():
    context = source()
    value = await context.read(uuid4(), uuid4(), ["memory", "audiovisual"])
    assert value["audiovisual"]["text"] == "媒体内容"
    assert set(value) == {"memory", "audiovisual", "revision", "as_of"}
    context.environment.load_cached.assert_not_awaited()
    context.history.load_history_readonly.assert_not_awaited()
    context.events.recent.assert_not_awaited()
    context.fishes.get_profile.assert_not_awaited()
    assert context.audiovisual.get_latest_media_context_summary.await_count == 1


@pytest.mark.asyncio
async def test_auxiliary_reads_sixteen_events_and_speech_keeps_its_window():
    context = source()
    messages = [
        {"type": "live.reply", "data": {"text": f"弹幕 {i}"}} for i in range(20)
    ]
    context.events.recent.side_effect = lambda fish, session, limit: messages[-limit:]
    context.fishes.get_profile.return_value = None
    context.environment.load_cached.return_value = SimpleNamespace(
        enabled=False, summary_context=""
    )
    fish, session = uuid4(), uuid4()
    auxiliary = await context.read(fish, session, ["history", "events"])
    speech, _ = await context.read_for_speech(fish, session)
    assert [e["data"]["text"] for e in auxiliary["events"]] == [
        f"弹幕 {i}" for i in range(4, 20)
    ]
    assert [e["data"]["text"] for e in speech["events"]] == [
        f"弹幕 {i}" for i in range(8, 20)
    ]
