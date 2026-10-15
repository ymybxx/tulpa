import asyncio
from dataclasses import replace
from uuid import UUID

import pytest

from fish_memory.companion.session import CompanionSession, CompanionState
from fish_memory.extensions.models import CreateInstance
from tests.companion.test_proactive_streaming import StreamingApi
from tests.companion.test_session import (
    ANSWER_ID,
    FISH_ID,
    SECOND_FISH_ID,
    SESSION_ID,
    AgentFakeApi,
    BlockingAgentTts,
    CallbackTts,
    RecordingPetEvents,
    proactive_message,
    wait_until,
)
from tests.extensions.test_service import make_service


class PluginApi(StreamingApi):
    iter_plugin_expression = StreamingApi.iter_proactive_expression


@pytest.mark.asyncio
@pytest.mark.parametrize("start_after_end", [False, True])
async def test_plugin_can_speak_from_standby_after_user_conversation_ends(
    start_after_end,
):
    api, tts = PluginApi(), CallbackTts()
    api.resolved_session_ids = [SESSION_ID, SESSION_ID]
    session = CompanionSession(api, tts)
    try:
        await session.set_binding(FISH_ID)
        if not start_after_end:
            await session._handle_plugin_playback_sessions(
                frozenset({(FISH_ID, SESSION_ID)})
            )
        await session.on_wake_word(play_ack=False)
        await session.submit_transcript("再见")
        await api.answer_events.put(
            {
                "event": "answer.delta",
                "data": {"messageId": str(ANSWER_ID), "content": "再见。"},
            }
        )
        await api.answer_events.put(
            {
                "event": "answer.completed",
                "data": {"messageId": str(ANSWER_ID), "content": "再见。", "end": True},
            }
        )
        await asyncio.wait_for(session.wait_for_idle(), timeout=1)
        assert session.state is CompanionState.STANDBY
        assert session.voice_conversation_id is None
        if start_after_end:
            # The background listener has already rejoined the current Session.
            assert session.session_id == SESSION_ID
            await session._handle_plugin_playback_sessions(
                frozenset({(FISH_ID, SESSION_ID)})
            )
            await wait_until(lambda: session.session_id == SESSION_ID)
        assert session._agent_listener_task is not None
        assert not session._agent_listener_task.done()
        message = replace(
            proactive_message(103, content="我想到一个有趣的问题。"),
            source="plugin",
            streaming=True,
            expires_at=None,
        )
        api.more.set()
        api.hold_end.set()
        await api.agent_events.put(message)
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert api.agent_consumption_calls == [
            (message.message_id, len(message.content + api.tail), False)
        ]
    finally:
        await session.close()


NEW_SESSION = UUID(int=202)


@pytest.mark.asyncio
@pytest.mark.parametrize("before_binding", [False, True])
async def test_plugin_selected_session_receives_audio_after_restart(before_binding):
    api, tts = AgentFakeApi(), CallbackTts()
    session = CompanionSession(api, tts)
    sessions = frozenset({(FISH_ID, NEW_SESSION)})
    try:
        if before_binding:
            await session._handle_plugin_playback_sessions(sessions)
        await session.set_binding(FISH_ID)
        if not before_binding:
            await wait_until(lambda: bool(api.agent_connections))
            old_listener = session._agent_listener_task
            session._agent_after_cursor = 42
            await session._handle_plugin_playback_sessions(sessions)
        await wait_until(lambda: session._agent_listener_session_id == NEW_SESSION)
        await wait_until(
            lambda: (
                bool(api.agent_connections)
                and api.agent_connections[-1] == (FISH_ID, NEW_SESSION, 0)
            )
        )
        if not before_binding:
            assert old_listener.cancelled()
        assert session.session_id == NEW_SESSION
        assert session.state is CompanionState.STANDBY
        assert api.resolve_calls == ([] if before_binding else [FISH_ID])
        message = replace(
            proactive_message(104, content="新会话的第一段主持。"),
            session_id=NEW_SESSION,
            source="plugin",
            expires_at=None,
        )
        await api.agent_events.put(message)
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert api.agent_consumption_calls == [
            (message.message_id, len(message.content), False)
        ]
        assert ("append", message.content) in tts.calls
        listener = session._agent_listener_task
        generation = session.binding_generation
        await session._handle_plugin_playback_sessions(sessions)
        await session._handle_plugin_playback_sessions(frozenset())
        assert session._agent_listener_task is listener
        assert session.binding_generation == generation
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plugin_session_waits_for_existing_user_answer_without_interrupting():
    api, tts = AgentFakeApi(), CallbackTts()
    session = CompanionSession(api, tts)
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        assert await session.submit_transcript("这句话在旧会话回答")
        await wait_until(lambda: bool(api.ask_calls))
        await session._handle_plugin_playback_sessions(
            frozenset({(FISH_ID, NEW_SESSION)})
        )
        assert session.session_id == SESSION_ID
        assert session._plugin_binding_task is None
        await api.answer_events.put(
            {
                "event": "answer.delta",
                "data": {
                    "messageId": str(ANSWER_ID),
                    "content": "好的。",
                },
            }
        )
        await api.answer_events.put(
            {
                "event": "answer.completed",
                "data": {
                    "messageId": str(ANSWER_ID),
                    "content": "好的。",
                    "end": False,
                },
            }
        )
        await wait_until(lambda: session.session_id == NEW_SESSION)
        assert api.ask_session_ids == [SESSION_ID]
        assert ("append", "好的。") in tts.calls
        assert session.state is CompanionState.LISTENING
        assert session.freeze_turn_binding().session_id == NEW_SESSION
        assert not api.interrupt_calls
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_cancelled_or_other_fish_plugin_cannot_move_voice_session():
    api = AgentFakeApi()
    session = CompanionSession(api, CallbackTts())
    try:
        await session.set_binding(FISH_ID)
        for sessions in (
            frozenset({(SECOND_FISH_ID, NEW_SESSION)}),
            frozenset({(FISH_ID, NEW_SESSION), (FISH_ID, UUID(int=203))}),
        ):
            await session._handle_plugin_playback_sessions(sessions)
            assert session._plugin_binding_task is None
            assert session.session_id == SESSION_ID
        await session._handle_plugin_playback_sessions(
            frozenset({(FISH_ID, NEW_SESSION)})
        )
        await session._handle_plugin_playback_sessions(frozenset())
        await wait_until(lambda: session._plugin_binding_task is None)
        assert session.session_id == SESSION_ID
        assert not session._binding_quiesced
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_session_switch_preserves_completed_old_playback():
    api, tts = AgentFakeApi(), BlockingAgentTts()
    session = CompanionSession(api, tts)
    message = replace(proactive_message(105), source="plugin", expires_at=None)
    try:
        await session.set_binding(FISH_ID)
        await api.agent_events.put(message)
        await asyncio.wait_for(tts.finish_started.wait(), 1)
        await session._handle_plugin_playback_sessions(
            frozenset({(FISH_ID, NEW_SESSION)})
        )
        assert session.session_id == SESSION_ID
        assert not api.agent_consumption_calls
        tts.played_offset = len(message.content)
        tts.release_finish.set()
        await wait_until(lambda: session.session_id == NEW_SESSION)
        assert api.agent_consumption_calls == [
            (message.message_id, len(message.content), False)
        ]
        assert not any(op == "interrupt" for op, _ in tts.calls)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_speech_reservation_defers_plugin_session_and_fences_old_draft():
    api = AgentFakeApi()
    session = CompanionSession(api, CallbackTts())
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        frozen = session.freeze_turn_binding()
        # A microphone frame can arrive after the control event but before its
        # scheduled task runs. It must still finish against the original binding.
        await session._handle_plugin_playback_sessions(
            frozenset({(FISH_ID, NEW_SESSION)})
        )
        reservation = session.reserve_foreground_speech()
        await wait_until(lambda: session._plugin_binding_task is None)
        assert session.session_id == SESSION_ID
        session.release_foreground_speech(reservation)
        await wait_until(lambda: session.session_id == NEW_SESSION)
        assert not await session.submit_transcript("旧草稿", frozen_turn=frozen)
        assert not api.ask_calls
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["fish", "audio_recovery", "close"])
async def test_pending_plugin_session_rebind_is_cancelled_with_lifecycle(change):
    api = AgentFakeApi()
    api.resolved_session_ids.append(SESSION_ID)
    session = CompanionSession(api, CallbackTts())
    try:
        await session.set_binding(FISH_ID)
        await session._handle_plugin_playback_sessions(
            frozenset({(FISH_ID, NEW_SESSION)})
        )
        task = session._plugin_binding_task
        if change == "fish":
            await session.set_binding(SECOND_FISH_ID)
            assert session.fish_id == SECOND_FISH_ID
            assert session.session_id == SESSION_ID
        elif change == "audio_recovery":
            assert await session.reset_for_audio_recovery(FISH_ID)
            assert await session.resume_after_audio_recovery(FISH_ID)
            assert session.session_id == NEW_SESSION
        else:
            await session.close()
        assert task.done()
        assert session._plugin_binding_task is None
        if change != "close":
            assert not session._binding_quiesced
    finally:
        if change != "close":
            await session.close()


@pytest.mark.asyncio
async def test_restarted_voice_binding_replaces_stale_busy_before_wake():
    service = make_service()
    item, _ = await service.create(
        CreateInstance(plugin_id="live", fish_id=FISH_ID, session_id=SESSION_ID)
    )
    service.attach(item)
    service.native_consumer(FISH_ID, SESSION_ID, True)

    class ActivityEvents(RecordingPetEvents):
        def companion_activity(self, payload):
            service._observe_activity(
                {"type": "pet.companion.activity", "payload": payload}
            )

    events = ActivityEvents()
    # The previous audio process disconnected in the middle of a user turn.
    events.companion_activity(
        {
            "fishId": str(FISH_ID),
            "sessionId": str(SESSION_ID),
            "busy": True,
            "playbackBusy": True,
        }
    )
    assert service.state(item)["reason"] == "user_busy"
    session = CompanionSession(PluginApi(), CallbackTts(), pet_events=events)
    try:
        await session.set_binding(FISH_ID)
        assert session.state is CompanionState.STANDBY
        assert service.state(item)["can_respond"] is True
        assert not service.llm.calls
    finally:
        await session.close()
        await service.close()


@pytest.mark.asyncio
async def test_plugin_uses_existing_tts_and_authoritative_interrupt_offset():
    api, tts = PluginApi(), CallbackTts()
    session = CompanionSession(api, tts)
    message = replace(
        proactive_message(101, content="这是插件说出的第一句话。"),
        source="plugin",
        streaming=True,
        expires_at=None,
    )
    try:
        await session.set_binding(FISH_ID)
        await api.agent_events.put(message)
        api.more.set()
        await wait_until(
            lambda: any(
                text == api.tail
                for operation, text in tts.calls
                if operation == "append"
            )
        )
        assert sum(operation == "start" for operation, _ in tts.calls) == 1
        tts.played_offset = 5
        await session._handle_answer_interrupt_request(FISH_ID, message.message_id)
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert api.agent_consumption_calls == [(message.message_id, 5, True)]
        assert api.stream_closed
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_plugin_does_not_wait_behind_existing_audio():
    api, tts = PluginApi(), CallbackTts()
    session = CompanionSession(api, tts)
    message = replace(
        proactive_message(102, content="这句话不能补播。"),
        source="plugin",
        streaming=True,
        expires_at=None,
    )
    try:
        await session.set_binding(FISH_ID)
        owner = await session._playback.acquire("answer", None)
        await api.agent_events.put(message)
        await wait_until(lambda: bool(api.agent_consumption_calls))
        assert api.agent_consumption_calls == [(message.message_id, 0, True)]
        await session._playback.release(owner)
        await asyncio.sleep(0)
        assert not any(operation == "start" for operation, _ in tts.calls)
    finally:
        await session.close()
