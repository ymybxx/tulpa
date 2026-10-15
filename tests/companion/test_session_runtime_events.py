"""Content-free session diagnostics without changing voice behavior."""

from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import cast
from uuid import UUID

import pytest

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.client import CompanionAskResult, CompanionMessage
from fish_memory.companion.session import (
    CompanionSession,
    CompanionState,
    FrozenAgentReply,
    FrozenVoiceTurn,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000002")
CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000003")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000004")
TURN_ID = UUID("00000000-0000-0000-0000-000000000005")
OTHER_ID = UUID("00000000-0000-0000-0000-000000000006")
PRIVATE_TEXT = "这段话只能进入正常语音请求，不能出现在检测事件里"


class RecordingEvents:
    def __init__(self, order: list[str] | None = None) -> None:
        self.events: list[dict[str, object]] = []
        self.order = order if order is not None else []
        self.on_event: Callable[[str], None] | None = None

    def record_event(
        self,
        name: str,
        *,
        text: str | None = None,
        source: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.order.append(name)
        self.events.append(
            {"name": name, "text": text, "source": source, "metadata": metadata}
        )
        if self.on_event is not None:
            self.on_event(name)

    def named(self, name: str) -> list[dict[str, object]]:
        return [event for event in self.events if event["name"] == name]


class FakeApi:
    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order if order is not None else []
        self.calls: list[tuple[str, str]] = []
        self.answer_expected = False
        self.submit_error: Exception | None = None
        self.stream_error: Exception | None = None
        self.progress_calls = 0

    async def ask(
        self,
        fish_id: UUID,
        content: str,
        **_kwargs: object,
    ) -> CompanionAskResult:
        assert fish_id == FISH_ID
        return self._submit("ask", content)

    async def commit_voice_turn(
        self,
        key: VoiceDraftKey,
        *,
        content: str,
        **_kwargs: object,
    ) -> CompanionAskResult:
        assert key.turn_id == TURN_ID
        return self._submit("voice_commit", content)

    def _submit(self, route: str, content: str) -> CompanionAskResult:
        self.order.append(f"api.{route}")
        self.calls.append((route, content))
        if self.submit_error is not None:
            raise self.submit_error
        return CompanionAskResult(MESSAGE_ID, self.answer_expected)

    async def iter_answer(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> AsyncIterator[dict[str, object]]:
        if self.stream_error is not None:
            raise self.stream_error
        for event in ():
            yield event

    async def update_progress(self, *_args: object, **_kwargs: object) -> None:
        self.progress_calls += 1


class FakeTts:
    def __init__(self) -> None:
        self.played_offset = 0
        self.calls: list[str] = []

    async def start(self) -> None:
        self.calls.append("start")

    async def append(self, _text: str) -> None:
        self.calls.append("append")

    async def finish(self) -> None:
        self.calls.append("finish")

    async def stop(self) -> None:
        self.calls.append("stop")

    async def pause(self) -> None:
        self.calls.append("pause")

    async def resume(self) -> None:
        self.calls.append("resume")

    async def duck(self) -> None:
        self.calls.append("duck")

    async def unduck(self) -> None:
        self.calls.append("unduck")


def make_session() -> tuple[CompanionSession, FakeApi, FakeTts, RecordingEvents]:
    order: list[str] = []
    api = FakeApi(order)
    tts = FakeTts()
    events = RecordingEvents(order)
    session = CompanionSession(api, tts, diagnostics=events)
    session._fish_id = FISH_ID
    session._session_id = SESSION_ID
    session._voice_conversation_id = CONVERSATION_ID
    session._binding_generation = 1
    session._state = CompanionState.LISTENING
    return session, api, tts, events


def frozen_turn() -> FrozenVoiceTurn:
    return FrozenVoiceTurn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        voice_conversation_id=CONVERSATION_ID,
        binding_generation=1,
        reply_context=None,
        context_reservation_id=None,
        reply_blocked=False,
        agent_reply_finalize=None,
    )


def metadata(event: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], event["metadata"])


def test_session_state_events_only_describe_actual_transitions() -> None:
    session, _, _, events = make_session()
    session._set_state(CompanionState.LISTENING)
    session._set_state(CompanionState.SPEAKING)
    session._set_state(CompanionState.SPEAKING)
    session._set_state(CompanionState.LISTENING)

    changes = events.named("session_state_changed")
    assert len(changes) == 2
    assert metadata(changes[0]) == {
        "previousState": "listening",
        "state": "speaking",
        "fishId": str(FISH_ID),
        "sessionId": str(SESSION_ID),
    }
    assert metadata(changes[1])["previousState"] == "speaking"
    assert metadata(changes[1])["state"] == "listening"
    assert all(event["text"] is None for event in changes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scenario", "expected_reason"),
    [
        ("missing_binding", "missing_turn_binding"),
        ("empty", "empty_content"),
        ("reply_blocked", "reply_context_blocked"),
        ("fish_changed", "binding_changed"),
        ("generation_changed", "binding_changed"),
        ("conversation_changed", "voice_conversation_changed"),
        ("agent_progress", "agent_progress_unsettled"),
        ("agent_finalize", "agent_reply_finalize_failed"),
        ("wake_binding_changed", "binding_changed_after_wake_ack"),
        ("session_changed", "session_changed"),
        ("not_listening", "state_not_listening"),
        ("binding_changed_after_await", "binding_changed"),
    ],
)
async def test_submit_rejections_report_fixed_reasons_without_text(
    scenario: str,
    expected_reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, api, _, events = make_session()
    frozen: FrozenVoiceTurn | None = frozen_turn()
    content = PRIVATE_TEXT
    if scenario == "missing_binding":
        frozen = None
    elif scenario == "empty":
        content = "  "
    elif scenario == "reply_blocked":
        frozen = replace(frozen, reply_blocked=True)
    elif scenario == "fish_changed":
        frozen = replace(frozen, fish_id=OTHER_ID)
    elif scenario == "generation_changed":
        frozen = replace(frozen, binding_generation=2)
    elif scenario == "conversation_changed":
        frozen = replace(frozen, voice_conversation_id=OTHER_ID)
    elif scenario == "agent_progress":

        async def unsettled(_turn: FrozenVoiceTurn) -> bool:
            return False

        monkeypatch.setattr(session, "_settle_frozen_agent_progress", unsettled)
    elif scenario == "agent_finalize":
        message = CompanionMessage(
            MESSAGE_ID,
            1,
            SESSION_ID,
            PRIVATE_TEXT,
            0,
            subject_kind="run_result",
            subject_id=OTHER_ID,
            activation_policy="playback_complete",
        )
        frozen = replace(frozen, agent_reply_finalize=FrozenAgentReply(message, 0))

        async def failed_finalize(*_args: object, **_kwargs: object) -> bool:
            return False

        monkeypatch.setattr(session, "_persist_agent_consumption", failed_finalize)
    elif scenario == "wake_binding_changed":
        session._state = CompanionState.ACKNOWLEDGING

        async def stop_ack() -> None:
            session._session_id = OTHER_ID

        monkeypatch.setattr(session, "_stop_wake_ack", stop_ack)
    elif scenario == "session_changed":
        frozen = replace(frozen, session_id=OTHER_ID)
    elif scenario == "not_listening":
        session._state = CompanionState.SPEAKING
    elif scenario == "binding_changed_after_await":

        async def changed_binding(_turn: FrozenVoiceTurn) -> bool:
            session._binding_generation += 1
            return True

        monkeypatch.setattr(session, "_settle_frozen_agent_progress", changed_binding)

    accepted = await session._submit_turn(
        content,
        voice_turn=(TURN_ID, 1, 0),
        frozen_turn=frozen,
    )

    assert accepted is False
    assert api.calls == []
    rejected = events.named("submit_turn_rejected")
    assert len(rejected) == 1
    assert metadata(rejected[0])["reason"] == expected_reason
    assert metadata(rejected[0])["inputChars"] == len(content.strip())
    assert metadata(rejected[0])["turnId"] == str(TURN_ID)
    assert metadata(rejected[0])["state"] == session.state.value
    assert PRIVATE_TEXT not in repr(rejected)
    assert rejected[0]["text"] is None


@pytest.mark.asyncio
async def test_invalid_voice_revision_is_diagnosed_before_any_api_call() -> None:
    session, api, _, events = make_session()

    assert not await session.submit_voice_turn(TURN_ID, 0, 0, PRIVATE_TEXT)

    rejected = events.named("submit_turn_rejected")
    assert len(rejected) == 1
    assert metadata(rejected[0])["reason"] == "invalid_voice_revision"
    assert metadata(rejected[0])["inputChars"] == len(PRIVATE_TEXT)
    assert metadata(rejected[0])["turnId"] == str(TURN_ID)
    assert api.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ask", "voice_commit"])
@pytest.mark.parametrize("answer_expected", [False, True])
async def test_api_submission_events_preserve_one_call_and_existing_boundary(
    route: str,
    answer_expected: bool,
) -> None:
    session, api, _, events = make_session()
    api.answer_expected = answer_expected
    if route == "ask":
        accepted = await session.submit_transcript(PRIVATE_TEXT)
    else:
        accepted = await session.submit_voice_turn(TURN_ID, 1, 0, PRIVATE_TEXT)
    assert accepted
    task = session._turn_task
    assert task is not None
    await task

    assert api.calls == [(route, PRIVATE_TEXT)]
    started = events.named("ask_submit_started")
    submitted = events.named("ask_submitted")
    assert len(started) == len(submitted) == 1
    assert metadata(started[0])["route"] == route
    assert metadata(started[0])["inputChars"] == len(PRIVATE_TEXT)
    assert metadata(submitted[0])["route"] == route
    assert metadata(submitted[0])["answerExpected"] is answer_expected
    assert metadata(submitted[0])["messageId"] == str(MESSAGE_ID)
    if route == "voice_commit":
        assert metadata(started[0])["turnId"] == str(TURN_ID)
    else:
        assert "turnId" not in metadata(started[0])
    assert events.order.index("ask_submit_started") < events.order.index(f"api.{route}")
    assert events.order.index(f"api.{route}") < events.order.index("ask_submitted")
    assert events.named("ask_submit_failed") == []
    assert PRIVATE_TEXT not in repr(events.events)
    assert all(event["text"] is None for event in events.events)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["ask", "voice_commit"])
async def test_api_failure_event_records_only_exception_class(route: str) -> None:
    session, api, _, events = make_session()
    api.submit_error = ValueError("exception detail must not be included")
    if route == "ask":
        assert await session.submit_transcript(PRIVATE_TEXT)
    else:
        assert await session.submit_voice_turn(TURN_ID, 1, 0, PRIVATE_TEXT)
    task = session._turn_task
    assert task is not None
    await task

    failures = events.named("ask_submit_failed")
    assert len(failures) == 1
    assert metadata(failures[0])["errorType"] == "ValueError"
    assert metadata(failures[0])["route"] == route
    assert "exception detail" not in repr(events.events)
    assert PRIVATE_TEXT not in repr(events.events)
    assert events.named("ask_submitted") == []
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
async def test_answer_stream_failure_is_not_mislabeled_as_submit_failure() -> None:
    session, api, _, events = make_session()
    api.answer_expected = True
    api.stream_error = RuntimeError("later stream detail")
    assert await session.submit_transcript(PRIVATE_TEXT)
    task = session._turn_task
    assert task is not None
    await task

    assert len(events.named("ask_submitted")) == 1
    assert events.named("ask_submit_failed") == []
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
async def test_completed_playback_event_reads_offset_before_owner_release() -> None:
    session, api, tts, events = make_session()
    owner = await session._playback.acquire("answer", MESSAGE_ID)
    bound = session._playback.bind_tts(owner, tts)
    session._playback_owner = owner
    session._turn_tts = bound
    session._answer_message_id = MESSAGE_ID
    session._tts_started = True
    session._state = CompanionState.SPEAKING
    tts.played_offset = 7
    owner_was_current: list[bool] = []

    def observe(name: str) -> None:
        if name == "answer_playback_completed":
            owner_was_current.append(session._playback.is_current(owner))
            assert tts.calls == ["finish"]

    events.on_event = observe
    assert await session._consume_event(
        {"event": "answer.completed", "data": {"messageId": str(MESSAGE_ID)}}
    )

    completed = events.named("answer_playback_completed")
    assert len(completed) == 1
    assert metadata(completed[0])["playedOffset"] == 7
    assert metadata(completed[0])["messageId"] == str(MESSAGE_ID)
    assert metadata(completed[0])["state"] == "speaking"
    assert owner_was_current == [True]
    assert bound.played_offset == 0
    assert len(events.named("tts_playback_stopped")) == 1
    assert api.progress_calls == 1
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
async def test_completion_without_tts_does_not_claim_playback_occurred() -> None:
    session, api, tts, events = make_session()

    assert await session._consume_event(
        {"event": "answer.completed", "data": {"messageId": str(MESSAGE_ID)}}
    )

    assert events.named("answer_playback_completed") == []
    assert tts.calls == []
    assert api.progress_calls == 1


@pytest.mark.asyncio
async def test_runtime_observer_failure_cannot_block_normal_submission() -> None:
    session, api, _, events = make_session()

    def failed_observer(_name: str) -> None:
        raise RuntimeError("diagnostic observer failure")

    events.on_event = failed_observer
    assert await session.submit_transcript(PRIVATE_TEXT)
    task = session._turn_task
    assert task is not None
    await task

    assert api.calls == [("ask", PRIVATE_TEXT)]
    assert session.state is CompanionState.LISTENING
    assert api.progress_calls == 0
