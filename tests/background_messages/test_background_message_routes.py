from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from fish_memory.agent.models import AgentActivationPolicy, AgentMessageSubjectKind
from fish_memory.background_messages.models import (
    BackgroundAssistantMessage,
    BackgroundMessageInterruptPolicy,
    BackgroundMessageReplyContext,
    BackgroundMessageSource,
)
from fish_memory.background_messages.routes import (
    BackgroundMessageConsumptionRequest,
    consume_background_message,
    stream_background_messages,
)
from fish_memory.db.sessions import SessionNotFoundError

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000002")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000003")
SUBJECT_ID = UUID("00000000-0000-0000-0000-000000000004")
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def message(
    *,
    source: BackgroundMessageSource = BackgroundMessageSource.AGENT,
) -> BackgroundAssistantMessage:
    return BackgroundAssistantMessage(
        source=source,
        message_id=MESSAGE_ID,
        sequence=7,
        session_id=SESSION_ID,
        content="后台消息",
        consumed_offset=0,
        emotion="happy",
        interrupt_policy=(
            BackgroundMessageInterruptPolicy.RESUME
            if source is BackgroundMessageSource.AGENT
            else BackgroundMessageInterruptPolicy.DROP
        ),
        expires_at=(
            None
            if source is BackgroundMessageSource.AGENT
            else NOW + timedelta(seconds=30)
        ),
        supersession_key=(
            None if source is BackgroundMessageSource.AGENT else "desktop-attention"
        ),
        reply_context=(
            BackgroundMessageReplyContext(
                AgentMessageSubjectKind.RUN_RESULT,
                SUBJECT_ID,
                AgentActivationPolicy.PLAYBACK_START,
            )
            if source is BackgroundMessageSource.AGENT
            else None
        ),
    )


class FakeStore:
    def __init__(self, item: BackgroundAssistantMessage | None = None) -> None:
        self.item = item
        self.finalized: list[tuple[UUID, UUID, UUID, int, bool]] = []

    async def list_incomplete_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        through_sequence: int,
        limit: int,
    ) -> list[BackgroundAssistantMessage]:
        del fish_id, session_id, after_sequence, through_sequence, limit
        return []

    async def list_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        include_incomplete: bool,
        limit: int,
        *,
        incomplete_after_sequence: int = 0,
    ) -> list[BackgroundAssistantMessage]:
        del fish_id, session_id, include_incomplete, limit, incomplete_after_sequence
        if self.item is None or self.item.sequence <= after_sequence:
            return []
        item, self.item = self.item, None
        return [item]

    async def finalize_consumption(
        self,
        message_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> object | None:
        self.finalized.append(
            (message_id, fish_id, session_id, consumed_offset, interrupted)
        )
        return object()


class FakeSessions:
    async def require_owned(self, fish_id: UUID, session_id: UUID) -> object:
        if (fish_id, session_id) != (FISH_ID, SESSION_ID):
            raise SessionNotFoundError(str(session_id))
        return object()


class FakeRequest:
    def __init__(self, store: FakeStore) -> None:
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                background_message_store=store,
                session_store=FakeSessions(),
            )
        )
        self.headers: dict[str, str] = {}
        self.disconnect_calls = 0

    async def is_disconnected(self) -> bool:
        self.disconnect_calls += 1
        return self.disconnect_calls > 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source",
    [BackgroundMessageSource.AGENT, BackgroundMessageSource.PROACTIVE_OBSERVATION],
)
async def test_stream_emits_exact_companion_message_contract(
    source: BackgroundMessageSource,
) -> None:
    item = message(source=source)
    request = FakeRequest(FakeStore(item))

    response = await stream_background_messages(
        FISH_ID,
        cast(Request, request),
        SESSION_ID,
        after_sequence=0,
    )
    assert isinstance(response, StreamingResponse)
    body = cast(AsyncGenerator[str, None], response.body_iterator)
    frame = await anext(body)
    await body.aclose()
    payload = json.loads(frame.split("data: ", 1)[1])

    assert frame.startswith("id: 7\nevent: companion.message\n")
    assert set(payload) == {
        "source",
        "messageId",
        "sequence",
        "sessionId",
        "content",
        "consumedOffset",
        "emotion",
        "interruptPolicy",
        "expiresAt",
        "supersessionKey",
        "replyContext",
    }
    assert payload["source"] == source.value
    assert payload["emotion"] == "happy"
    if source is BackgroundMessageSource.AGENT:
        assert payload["replyContext"] == {
            "kind": "run_result",
            "id": str(SUBJECT_ID),
            "activationPolicy": "playback_start",
        }
        assert payload["interruptPolicy"] == "resume"
    else:
        assert payload["replyContext"] is None
        assert payload["interruptPolicy"] == "drop"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native", "state", "expected"),
    [
        (True, "pending", True),
        (True, "playing", True),
        (True, "expired", False),
        (True, None, False),
        (False, "pending", False),
    ],
)
async def test_live_expression_requires_its_active_native_owner(
    native, state, expected
):
    item = replace(
        message(source=BackgroundMessageSource.PROACTIVE_OBSERVATION), expires_at=None
    )
    request = FakeRequest(FakeStore(item))
    stream = SimpleNamespace(intent=SimpleNamespace(state=state)) if state else None
    request.app.state.proactive_coordinator = SimpleNamespace(
        streams=SimpleNamespace(get=lambda _: stream, wake=asyncio.Event()),
        bind=lambda *_: "token",
        unbind=lambda _: None,
    )
    response = await stream_background_messages(
        FISH_ID,
        cast(Request, request),
        SESSION_ID,
        after_sequence=0,
        native_consumer=native,
    )
    body = cast(AsyncGenerator[str, None], response.body_iterator)
    try:
        if expected:
            frame = await anext(body)
            payload = json.loads(frame.split("data: ", 1)[1])
            assert payload["streaming"] is True and payload["expiresAt"] is None
        else:
            assert not [frame async for frame in body]
    finally:
        await body.aclose()


@pytest.mark.asyncio
async def test_initial_live_speech_contains_only_the_delivered_clean_prefix():
    item = replace(
        message(source=BackgroundMessageSource.PROACTIVE_OBSERVATION),
        content="你好。",
        expires_at=None,
    )
    request = FakeRequest(FakeStore(item))
    stream = SimpleNamespace(
        intent=SimpleNamespace(state="pending"),
        text="你好。下一句。",
        speech_tags=True,
        speech_content="[excited]你好。[serious]下一句。",
    )
    request.app.state.proactive_coordinator = SimpleNamespace(
        streams=SimpleNamespace(get=lambda _: stream, wake=asyncio.Event()),
        bind=lambda *_: "token",
        unbind=lambda _: None,
    )
    response = await stream_background_messages(
        FISH_ID, cast(Request, request), SESSION_ID, native_consumer=True
    )
    body = cast(AsyncGenerator[str, None], response.body_iterator)
    try:
        payload = json.loads((await anext(body)).split("data: ", 1)[1])
        assert payload["content"] == "你好。"
        assert payload["speechContent"] == "[excited]你好。"
        assert payload["speechTags"] is True
    finally:
        await body.aclose()


@pytest.mark.asyncio
async def test_consumption_is_session_scoped_and_forwards_exact_progress() -> None:
    store = FakeStore()
    request = FakeRequest(store)

    response = await consume_background_message(
        FISH_ID,
        MESSAGE_ID,
        SESSION_ID,
        BackgroundMessageConsumptionRequest(
            consumedOffset=4,
            interrupted=True,
        ),
        cast(Request, request),
    )

    assert response.status_code == 204
    assert store.finalized == [(MESSAGE_ID, FISH_ID, SESSION_ID, 4, True)]


def test_consumption_request_forbids_unknown_fields_and_non_strict_values() -> None:
    with pytest.raises(ValueError):
        BackgroundMessageConsumptionRequest.model_validate(
            {"consumedOffset": 0, "interrupted": False, "unexpected": True}
        )
    with pytest.raises(ValueError):
        BackgroundMessageConsumptionRequest.model_validate(
            {"consumedOffset": "0", "interrupted": False}
        )


@pytest.mark.parametrize("playback_enabled", [False, True])
async def test_text_delivery_does_not_invent_audio_progress(playback_enabled):
    store = FakeStore()
    request = FakeRequest(store)
    request.app.state.settings = SimpleNamespace(
        desktop_mode=True, companion_playback_enabled=playback_enabled
    )
    body = BackgroundMessageConsumptionRequest(consumedOffset=-1, interrupted=False)
    if playback_enabled:
        with pytest.raises(HTTPException) as error:
            await consume_background_message(
                FISH_ID, MESSAGE_ID, SESSION_ID, body, cast(Request, request)
            )
        assert error.value.status_code == 409
        assert not store.finalized
    else:
        response = await consume_background_message(
            FISH_ID, MESSAGE_ID, SESSION_ID, body, cast(Request, request)
        )
        assert response.status_code == 204
        assert store.finalized == [(MESSAGE_ID, FISH_ID, SESSION_ID, -1, False)]
