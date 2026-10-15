import asyncio
import json
from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.client import (
    ApiCompanionClient,
    CompanionAgentMessage,
    CompanionAskResult,
    CompanionMessage,
    CompanionReplyContext,
    parse_sse_event,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000002")
TURN_ID = UUID("00000000-0000-0000-0000-000000000003")


def test_parse_sse_event_returns_event_name_and_json_data() -> None:
    event = parse_sse_event(
        "id: 1:4\n"
        "event: answer.delta\n"
        'data: {"messageId":"00000000-0000-0000-0000-000000000001",'
        '"content":"你好","end":false}\n\n'
    )

    assert event == {
        "event": "answer.delta",
        "data": {
            "messageId": str(UUID(int=1)),
            "content": "你好",
            "end": False,
        },
    }


def test_parse_sse_event_ignores_comments_and_blank_data_lines() -> None:
    event = parse_sse_event(
        ": keep-alive\n"
        "event: answer.snapshot\n"
        "data: {\n"
        'data: "messageId":"00000000-0000-0000-0000-000000000001",\n'
        'data: "content":"你好"\n'
        "data: }\n\n"
    )

    assert event["event"] == "answer.snapshot"
    assert event["data"]["content"] == "你好"


@pytest.mark.asyncio
async def test_owned_http_client_sends_api_key_as_bearer() -> None:
    authorization: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization.append(request.headers.get("Authorization"))
        return httpx.Response(
            200,
            json={"fishId": str(UUID(int=3))},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient(
        "http://test",
        http_client=http_client,
        api_key="fm_companion_key",
    )

    await client.create_fish(name="authenticated")
    await http_client.aclose()

    assert authorization == ["Bearer fm_companion_key"]


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self.payload


class _FakeHttpClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        del url, kwargs
        return _FakeResponse(self.payload)


class _FakeCrudHttpClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, object]] = []
        self.gets: list[tuple[str, object]] = []

    async def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.posts.append((url, kwargs.get("json")))
        return _FakeResponse({"fishId": str(UUID(int=3))})

    async def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.gets.append((url, kwargs.get("params")))
        return _FakeResponse({"items": [{"sequence": 1, "content": "你好"}]})


@pytest.mark.asyncio
async def test_ask_remembers_sequence_for_the_next_answer_subscription() -> None:
    client = ApiCompanionClient(
        "http://test",
        http_client=_FakeHttpClient(
            {
                "messageId": "00000000-0000-0000-0000-000000000001",
                "sequence": 42,
                "sessionId": "00000000-0000-0000-0000-000000000003",
                "answerExpected": True,
            }
        ),
    )

    assert await client.ask(
        UUID(int=2), "你好", session_id=UUID(int=3)
    ) == CompanionAskResult(UUID(int=1), True, 42)
    assert client.answer_start_sequence == 42


@pytest.mark.asyncio
async def test_ask_sends_exact_optional_reply_context_and_parses_no_answer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            json={
                "messageId": str(TURN_ID),
                "sessionId": str(SESSION_ID),
                "sequence": 43,
                "status": "initialized",
                "createdAt": "2026-08-31T00:00:00Z",
                "answerExpected": False,
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    context = CompanionReplyContext("interaction", UUID(int=901))
    try:
        result = await client.ask(
            FISH_ID,
            "同意",
            session_id=SESSION_ID,
            reply_context=context,
        )
    finally:
        await http_client.aclose()

    assert result == CompanionAskResult(TURN_ID, False, 43)
    assert json.loads(requests[0].content) == {
        "content": "同意",
        "playbackOwner": "companion",
        "sessionId": str(SESSION_ID),
        "replyContext": {
            "kind": "interaction",
            "id": str(UUID(int=901)),
        },
    }


@pytest.mark.asyncio
async def test_reads_effective_fish_voice_config() -> None:
    class VoiceHttpClient:
        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            del kwargs
            assert url.endswith("/v1/fishes/00000000-0000-0000-0000-000000000002/voice")
            return _FakeResponse(
                {
                    "provider": "dashscope",
                    "config": {
                        "model": "qwen-audio-3.0-tts-flash",
                        "voice": "qwen-audio-3.0-tts-flash-mine-1",
                        "kind": "custom",
                    },
                    "usedFallback": False,
                }
            )

    client = ApiCompanionClient("http://test", http_client=VoiceHttpClient())

    config = await client.get_voice_config(UUID(int=2))

    assert config.model == "qwen-audio-3.0-tts-flash"
    assert config.voice == "qwen-audio-3.0-tts-flash-mine-1"


@pytest.mark.asyncio
async def test_resolve_session_creates_one_after_one_hour_of_inactivity() -> None:
    now = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)

    class SessionHttpClient:
        def __init__(self) -> None:
            self.posts: list[str] = []

        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            del url, kwargs
            return _FakeResponse(
                {
                    "items": [
                        {
                            "sessionId": str(UUID(int=3)),
                            "createdAt": "2026-08-09T10:00:00+00:00",
                            "lastMessageAt": "2026-08-09T10:30:00+00:00",
                        }
                    ]
                }
            )

        async def post(self, url: str, **kwargs: object) -> _FakeResponse:
            del kwargs
            self.posts.append(url)
            return _FakeResponse(
                {
                    "sessionId": str(UUID(int=4)),
                    "createdAt": "2026-08-09T12:00:00+00:00",
                    "lastMessageAt": None,
                }
            )

    http_client = SessionHttpClient()
    client = ApiCompanionClient(
        "http://test",
        http_client=http_client,
        now=lambda: now,
    )

    assert await client.resolve_session(UUID(int=2)) == UUID(int=4)
    assert http_client.posts == [
        "http://test/v1/fishes/00000000-0000-0000-0000-000000000002/sessions"
    ]


@pytest.mark.asyncio
async def test_resolve_session_reuses_the_newest_recent_session() -> None:
    now = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)

    class SessionHttpClient:
        def __init__(self) -> None:
            self.posts: list[str] = []

        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            del url, kwargs
            return _FakeResponse(
                {
                    "items": [
                        {
                            "sessionId": str(UUID(int=4)),
                            "createdAt": "2026-08-09T11:50:00+00:00",
                            "lastMessageAt": "2026-08-09T11:55:00+00:00",
                        },
                        {
                            "sessionId": str(UUID(int=3)),
                            "createdAt": "2026-08-09T10:00:00+00:00",
                            "lastMessageAt": "2026-08-09T11:59:00+00:00",
                        },
                    ]
                }
            )

        async def post(self, url: str, **kwargs: object) -> _FakeResponse:
            del kwargs
            self.posts.append(url)
            raise AssertionError("a recent newest-created session must be reused")

    http_client = SessionHttpClient()
    client = ApiCompanionClient(
        "http://test",
        http_client=http_client,
        now=lambda: now,
    )

    assert await client.resolve_session(UUID(int=2)) == UUID(int=4)
    assert http_client.posts == []


@pytest.mark.asyncio
async def test_resolve_session_rejects_a_naive_api_timestamp() -> None:
    class SessionHttpClient:
        async def get(self, url: str, **kwargs: object) -> _FakeResponse:
            del url, kwargs
            return _FakeResponse(
                {
                    "items": [
                        {
                            "sessionId": str(UUID(int=3)),
                            "createdAt": "2026-08-09T11:50:00",
                            "lastMessageAt": None,
                        }
                    ]
                }
            )

    client = ApiCompanionClient(
        "http://test",
        http_client=SessionHttpClient(),
    )

    with pytest.raises(ValueError, match="timezone"):
        await client.resolve_session(UUID(int=2))


@pytest.mark.asyncio
async def test_create_fish_and_list_messages_use_the_normal_api() -> None:
    http_client = _FakeCrudHttpClient()
    client = ApiCompanionClient("http://test", http_client=http_client)

    fish_id = await client.create_fish(name="companion-test")
    messages = await client.list_messages(fish_id)

    assert fish_id == UUID(int=3)
    assert messages == {"items": [{"sequence": 1, "content": "你好"}]}
    assert http_client.posts == [
        (
            "http://test/v1/fishes",
            {"role": None, "setting": None, "name": "companion-test"},
        )
    ]
    assert http_client.gets == [
        (
            "http://test/v1/fishes/00000000-0000-0000-0000-000000000003/messages",
            {"limit": "500"},
        )
    ]


@pytest.mark.asyncio
async def test_persists_pet_voice_performance_through_the_internal_api() -> None:
    http_client = _FakeCrudHttpClient()
    client = ApiCompanionClient("http://test", http_client=http_client)
    metrics = {
        "status": "completed",
        "answerMessageId": str(UUID(int=5)),
        "nativeMs": {"endpointToPlayback": 950.0},
    }

    await client.persist_pet_voice_performance(UUID(int=2), UUID(int=5), metrics)

    assert http_client.posts == [
        (
            "http://test/internal/fishes/00000000-0000-0000-0000-000000000002/"
            "messages/00000000-0000-0000-0000-000000000005/"
            "pet-voice-performance",
            {"metrics": metrics},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_context", [False, True])
@pytest.mark.parametrize("with_frame_window", [False, True])
async def test_voice_draft_rest_contract_is_bounded_and_remembers_sequence(
    with_context: bool,
    with_frame_window: bool,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/draft"):
            return httpx.Response(
                202,
                json={
                    "turnId": str(TURN_ID),
                    "revision": 1,
                    "content": "候选问题",
                    "state": "active",
                },
            )
        if request.url.path.endswith("/cancel"):
            return httpx.Response(204)
        return httpx.Response(
            201,
            json={
                "messageId": str(TURN_ID),
                "sessionId": str(SESSION_ID),
                "sequence": 17,
                "status": "initialized",
                "createdAt": "2026-08-30T00:00:00Z",
                "answerExpected": True,
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    key = VoiceDraftKey(FISH_ID, SESSION_ID, TURN_ID)

    try:
        assert (
            await client.upsert_voice_draft(
                key,
                revision=1,
                content="候选问题",
                reply_context=(
                    CompanionReplyContext("run_result", UUID(int=902))
                    if with_context
                    else None
                ),
                speech_started_at=(
                    datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
                    if with_frame_window
                    else None
                ),
                frame_snapshot_at=(
                    datetime(2026, 9, 3, 12, 0, 5, tzinfo=UTC)
                    if with_frame_window
                    else None
                ),
            )
            is True
        )
        assert (
            await client.cancel_voice_draft(
                key,
                revision=2,
                reason="speech_resumed",
            )
            is True
        )
        assert await client.commit_voice_turn(
            key,
            revision=3,
            promote_revision=1,
            content="最终问题",
            reply_context=CompanionReplyContext("run_result", UUID(int=902)),
        ) == CompanionAskResult(TURN_ID, True, 17)
    finally:
        await http_client.aclose()

    assert [request.method for request in requests] == ["PUT", "POST", "POST"]
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "draft",
        "cancel",
        "commit",
    ]
    assert [request.extensions["timeout"]["connect"] for request in requests] == [
        0.5,
        0.5,
        10.0,
    ]
    expected_draft: dict[str, object] = {
        "revision": 1,
        "content": "候选问题",
    }
    if with_context:
        expected_draft["replyContext"] = {
            "kind": "run_result",
            "id": str(UUID(int=902)),
        }
    if with_frame_window:
        expected_draft |= {
            "speechStartedAt": "2026-09-03T12:00:00+00:00",
            "frameSnapshotAt": "2026-09-03T12:00:05+00:00",
        }
    assert json.loads(requests[0].content) == expected_draft
    assert requests[1].content == b'{"revision":2,"reason":"speech_resumed"}'
    assert json.loads(requests[2].content) == {
        "revision": 3,
        "promoteRevision": 1,
        "content": "最终问题",
        "replyContext": {
            "kind": "run_result",
            "id": str(UUID(int=902)),
        },
    }
    assert client.answer_start_sequence == 17


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"messageId": str(TURN_ID), "sequence": 1},
        {
            "messageId": str(TURN_ID),
            "sequence": 1,
            "answerExpected": 1,
        },
    ],
)
async def test_turn_response_requires_strict_answer_expected(
    payload: dict[str, object],
) -> None:
    client = ApiCompanionClient(
        "http://test",
        http_client=_FakeHttpClient(payload),
    )

    with pytest.raises(ValueError, match="answerExpected"):
        await client.ask(FISH_ID, "你好", session_id=SESSION_ID)


@pytest.mark.asyncio
async def test_agent_message_sse_uses_one_session_scoped_connection_and_cursor() -> (
    None
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                "event: agent.heartbeat\n"
                "data: {}\n\n"
                "id: 12\n"
                "event: agent.message\n"
                'data: {"messageId":"00000000-0000-0000-0000-000000000003",'
                '"sequence":12,'
                '"sessionId":"00000000-0000-0000-0000-000000000002",'
                '"content":"任务完成。","consumedOffset":2,'
                '"subject":{"kind":"run_result",'
                '"id":"00000000-0000-0000-0000-000000000901"},'
                '"activationPolicy":"playback_start"}\n\n'
            ).encode(),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        messages = [
            message
            async for message in client.iter_agent_messages(
                FISH_ID,
                session_id=SESSION_ID,
                after_sequence=11,
            )
        ]
    finally:
        await http_client.aclose()

    assert messages == [
        CompanionAgentMessage(
            message_id=TURN_ID,
            sequence=12,
            session_id=SESSION_ID,
            content="任务完成。",
            consumed_offset=2,
            subject_kind="run_result",
            subject_id=UUID("00000000-0000-0000-0000-000000000901"),
            activation_policy="playback_start",
        )
    ]
    assert len(requests) == 1
    assert requests[0].url.path == f"/v1/fishes/{FISH_ID}/agent-messages/events"
    assert dict(requests[0].url.params) == {
        "sessionId": str(SESSION_ID),
        "afterSequence": "11",
    }
    assert requests[0].headers["Last-Event-ID"] == "11"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {
            "messageId": str(TURN_ID),
            "sequence": True,
            "sessionId": str(SESSION_ID),
            "content": "任务完成。",
            "consumedOffset": 0,
            "subject": {"kind": "run_result", "id": str(UUID(int=901))},
            "activationPolicy": "playback_start",
        },
        {
            "messageId": str(TURN_ID),
            "sequence": 12,
            "sessionId": str(SESSION_ID),
            "content": "短",
            "consumedOffset": 2,
            "subject": {"kind": "run_result", "id": str(UUID(int=901))},
            "activationPolicy": "playback_start",
        },
        {
            "messageId": str(TURN_ID),
            "sequence": 12,
            "sessionId": str(SESSION_ID),
            "content": "任务完成。",
            "consumedOffset": 0,
            "subject": {
                "kind": "interaction",
                "id": str(UUID(int=901)),
                "extra": True,
            },
            "activationPolicy": "playback_complete",
        },
    ],
)
async def test_agent_message_sse_rejects_malformed_envelopes(
    data: dict[str, object],
) -> None:
    body = (f"event: agent.message\ndata: {json.dumps(data)}\n\n").encode()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        )
    )
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        with pytest.raises(ValueError):
            _ = [
                message
                async for message in client.iter_agent_messages(
                    FISH_ID,
                    session_id=SESSION_ID,
                    after_sequence=0,
                )
            ]
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_consume_agent_message_uses_exact_session_scoped_endpoint_and_body() -> (
    None
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        await client.consume_agent_message(
            FISH_ID,
            TURN_ID,
            session_id=SESSION_ID,
            consumed_offset=4,
            interrupted=False,
        )
    finally:
        await http_client.aclose()

    assert requests[0].url.path == (
        f"/v1/fishes/{FISH_ID}/agent-messages/{TURN_ID}/consumption"
    )
    assert dict(requests[0].url.params) == {"sessionId": str(SESSION_ID)}
    assert json.loads(requests[0].content) == {
        "consumedOffset": 4,
        "interrupted": False,
    }


@pytest.mark.asyncio
async def test_companion_message_stream_parses_proactive_delivery_contract() -> None:
    requests: list[httpx.Request] = []
    expires_at = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                "event: companion.heartbeat\n"
                "data: {}\n\n"
                "id: 14\n"
                "event: companion.message\n"
                'data: {"source":"proactive_observation",'
                '"messageId":"00000000-0000-0000-0000-000000000003",'
                '"sequence":14,'
                '"sessionId":"00000000-0000-0000-0000-000000000002",'
                '"content":"你已经专注很久了。","consumedOffset":0,'
                '"emotion":"happy","interruptPolicy":"drop",'
                '"expiresAt":"2026-09-01T12:30:00Z",'
                '"supersessionKey":"focus","replyContext":null}\n\n'
            ).encode(),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        messages = [
            message
            async for message in client.iter_companion_messages(
                FISH_ID,
                session_id=SESSION_ID,
                after_sequence=13,
            )
        ]
    finally:
        await http_client.aclose()

    assert messages == [
        CompanionMessage(
            message_id=TURN_ID,
            sequence=14,
            session_id=SESSION_ID,
            content="你已经专注很久了。",
            consumed_offset=0,
            source="proactive_observation",
            emotion="happy",
            interrupt_policy="drop",
            expires_at=expires_at,
            supersession_key="focus",
        )
    ]
    assert requests[0].url.path == (f"/v1/fishes/{FISH_ID}/companion-messages/events")
    assert requests[0].headers["Last-Event-ID"] == "13"


@pytest.mark.asyncio
async def test_consume_companion_message_uses_unified_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(204)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        await client.consume_companion_message(
            FISH_ID,
            TURN_ID,
            session_id=SESSION_ID,
            consumed_offset=3,
            interrupted=True,
            source="proactive_observation",
        )
    finally:
        await http_client.aclose()

    assert requests[0].url.path == (
        f"/v1/fishes/{FISH_ID}/companion-messages/{TURN_ID}/consumption"
    )
    assert dict(requests[0].url.params) == {"sessionId": str(SESSION_ID)}
    assert json.loads(requests[0].content) == {
        "consumedOffset": 3,
        "interrupted": True,
    }


@pytest.mark.asyncio
async def test_missing_proactive_consumption_is_already_terminal() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, json={"detail": "not found"})
        )
    )
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        await client.consume_companion_message(
            FISH_ID,
            TURN_ID,
            session_id=SESSION_ID,
            consumed_offset=0,
            interrupted=True,
            source="proactive_observation",
        )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_unified_stream_preserves_agent_reply_context() -> None:
    body = (
        "event: companion.message\n"
        'data: {"source":"agent",'
        f'"messageId":"{TURN_ID}","sequence":12,'
        f'"sessionId":"{SESSION_ID}",'
        '"content":"任务完成。","consumedOffset":2,'
        '"emotion":"neutral","interruptPolicy":"resume",'
        '"expiresAt":null,"supersessionKey":null,'
        '"replyContext":{"kind":"run_result",'
        f'"id":"{UUID(int=901)}",'
        '"activationPolicy":"playback_start"}}\n\n'
    ).encode()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        )
    )
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        messages = [
            message
            async for message in client.iter_companion_messages(
                FISH_ID,
                session_id=SESSION_ID,
                after_sequence=11,
            )
        ]
    finally:
        await http_client.aclose()

    assert messages == [
        CompanionMessage(
            message_id=TURN_ID,
            sequence=12,
            session_id=SESSION_ID,
            content="任务完成。",
            consumed_offset=2,
            subject_kind="run_result",
            subject_id=UUID(int=901),
            activation_policy="playback_start",
        )
    ]


@pytest.mark.asyncio
async def test_unified_stream_falls_back_to_legacy_agent_endpoint() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/companion-messages/events"):
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(
            200,
            content=(
                "event: agent.message\n"
                f'data: {{"messageId":"{TURN_ID}","sequence":12,'
                f'"sessionId":"{SESSION_ID}","content":"任务完成。",'
                '"consumedOffset":0,"subject":{"kind":"run_result",'
                f'"id":"{UUID(int=901)}"}},'
                '"activationPolicy":"playback_start"}\n\n'
            ).encode(),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        messages = [
            message
            async for message in client.iter_companion_messages(
                FISH_ID,
                session_id=SESSION_ID,
                after_sequence=11,
            )
        ]
    finally:
        await http_client.aclose()

    assert len(messages) == 1
    assert messages[0].source == "agent"
    assert paths == [
        f"/v1/fishes/{FISH_ID}/companion-messages/events",
        f"/v1/fishes/{FISH_ID}/agent-messages/events",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "interrupt_policy", "reply_context"),
    [
        ("agent", "resume", None),
        (
            "proactive_observation",
            "drop",
            {
                "kind": "run_result",
                "id": str(UUID(int=901)),
                "activationPolicy": "playback_start",
            },
        ),
        ("proactive_observation", "resume", None),
        ("proactive_observation", "drop", None),
    ],
)
async def test_unified_stream_rejects_source_policy_context_mismatch(
    source: str,
    interrupt_policy: str,
    reply_context: dict[str, object] | None,
) -> None:
    data = {
        "source": source,
        "messageId": str(TURN_ID),
        "sequence": 12,
        "sessionId": str(SESSION_ID),
        "content": "消息。",
        "consumedOffset": 0,
        "emotion": "neutral",
        "interruptPolicy": interrupt_policy,
        "expiresAt": None,
        "supersessionKey": None,
        "replyContext": reply_context,
    }
    body = (f"event: companion.message\ndata: {json.dumps(data)}\n\n").encode()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=body)
        )
    )
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        with pytest.raises(ValueError):
            _ = [
                message
                async for message in client.iter_companion_messages(
                    FISH_ID,
                    session_id=SESSION_ID,
                    after_sequence=0,
                )
            ]
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_unified_stream_accepts_live_expression_without_wall_clock_expiry():
    data = {
        "source": "proactive_observation",
        "streaming": True,
        "messageId": str(TURN_ID),
        "sequence": 12,
        "sessionId": str(SESSION_ID),
        "content": "消息。",
        "consumedOffset": 0,
        "emotion": "neutral",
        "interruptPolicy": "drop",
        "expiresAt": None,
        "supersessionKey": "attention:1:f1",
        "replyContext": None,
    }
    body = (f"event: companion.message\ndata: {json.dumps(data)}\n\n").encode()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    ) as http:
        client = ApiCompanionClient("http://test", http_client=http)
        received = [
            item
            async for item in client.iter_companion_messages(
                FISH_ID, session_id=SESSION_ID, after_sequence=0
            )
        ]
    assert len(received) == 1
    assert received[0].streaming and received[0].expires_at is None


@pytest.mark.asyncio
async def test_voice_draft_capability_uses_read_only_internal_config() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"voiceDrafts": True})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        assert await client.voice_draft_capability() is True
    finally:
        await http_client.aclose()

    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/internal/capabilities")
    ]
    assert requests[0].extensions["timeout"]["connect"] == 0.5


@pytest.mark.asyncio
async def test_missing_voice_draft_capability_fails_closed() -> None:
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(404))
    )
    client = ApiCompanionClient("http://old-api", http_client=http_client)
    try:
        assert await client.voice_draft_capability() is False
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_message_performance_evidence_uses_internal_read_only_route() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"items": [], "homeActions": []})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        payload = await client.get_message_performance(FISH_ID, TURN_ID)
    finally:
        await http_client.aclose()

    assert payload == {"items": [], "homeActions": []}
    assert requests[0].url.path == (
        f"/internal/fishes/{FISH_ID}/messages/{TURN_ID}/performance"
    )


@pytest.mark.asyncio
async def test_voice_draft_controls_miss_but_commit_propagates() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    key = VoiceDraftKey(FISH_ID, SESSION_ID, TURN_ID)

    try:
        assert (
            await client.upsert_voice_draft(
                key,
                revision=1,
                content="候选问题",
            )
            is False
        )
        assert (
            await client.cancel_voice_draft(
                key,
                revision=2,
                reason="rejected",
            )
            is False
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.commit_voice_turn(
                key,
                revision=3,
                promote_revision=1,
                content="最终问题",
            )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_voice_draft_private_sse_uses_existing_event_parser() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == {"revision": "4"}
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=(
                "event: draft.snapshot\n"
                'data: {"turnId":"00000000-0000-0000-0000-000000000003",'
                '"revision":4,"content":"私有回答","flushOffset":0}\n\n'
                "event: draft.ready\n"
                'data: {"turnId":"00000000-0000-0000-0000-000000000003",'
                '"revision":4,"content":"私有回答","flushOffset":4}\n\n'
            ).encode(),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    key = VoiceDraftKey(FISH_ID, SESSION_ID, TURN_ID)

    try:
        events = [
            event async for event in client.iter_voice_draft_events(key, revision=4)
        ]
    finally:
        await http_client.aclose()

    assert [event["event"] for event in events] == [
        "draft.snapshot",
        "draft.ready",
    ]
    assert events[1]["data"]["flushOffset"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["answer", "home"])
async def test_sse_monitor_open_callback_runs_after_authenticated_headers(
    stream_kind: str,
) -> None:
    opened = asyncio.Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        event = (
            "event: answer.completed\n"
            'data: {"messageId":"00000000-0000-0000-0000-000000000003"}\n\n'
            if stream_kind == "answer"
            else 'event: home.action.succeeded\ndata: {"status":"succeeded"}\n\n'
        )
        return httpx.Response(200, content=event.encode())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        if stream_kind == "answer":
            events = [
                event
                async for event in client.iter_answer(
                    FISH_ID,
                    session_id=SESSION_ID,
                    on_open=opened.set,
                )
            ]
        else:
            events = [
                event
                async for event in client.iter_home_actions(
                    FISH_ID,
                    session_id=SESSION_ID,
                    on_open=opened.set,
                )
            ]
    finally:
        await http_client.aclose()

    assert opened.is_set()
    assert len(events) == 1


@pytest.mark.asyncio
async def test_sse_monitor_open_callback_does_not_run_on_auth_failure() -> None:
    opened = asyncio.Event()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid API key"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApiCompanionClient("http://test", http_client=http_client)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            async for _event in client.iter_answer(
                FISH_ID,
                session_id=SESSION_ID,
                on_open=opened.set,
            ):
                pass
    finally:
        await http_client.aclose()

    assert not opened.is_set()
