import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr

from fish_memory.app import create_app
from fish_memory.config import Settings
from fish_memory.db.home_actions import HomeActionRun
from fish_memory.home.models import HomeActionStatus
from fish_memory.models import FishSummary, Message, MessageStatus, Role
from fish_memory.performance.models import PerformanceStep

from .fakes import (
    DEFAULT_SESSION_ID,
    FakeApiCoordinator,
    FakeApiStore,
    FakePerformanceStore,
    FakeSessionStore,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000010")
NOW = datetime(2026, 7, 20, 8, 30, tzinfo=UTC)
TEST_API_KEY = "fm_test_service_key"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_KEY}"}


class FakeHomeActionStore:
    def __init__(self) -> None:
        self.runs: list[HomeActionRun] = []
        self.calls: list[tuple[UUID, UUID]] = []

    async def list_message_runs(
        self,
        fish_id: UUID,
        message_id: UUID,
    ) -> list[HomeActionRun]:
        self.calls.append((fish_id, message_id))
        return self.runs


def build_test_app(
    store: FakeApiStore,
    coordinator: FakeApiCoordinator | None = None,
    performance_store: FakePerformanceStore | None = None,
    home_action_store: FakeHomeActionStore | None = None,
):
    settings = Settings.model_construct(
        database_url="postgresql://test",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
        api_key=SecretStr(TEST_API_KEY),
    )
    app = create_app(settings)
    app.state.message_store = store
    app.state.session_store = FakeSessionStore()
    app.state.answer_coordinator = coordinator or FakeApiCoordinator(store.events)
    app.state.performance_store = performance_store or FakePerformanceStore()
    app.state.home_action_store = home_action_store or FakeHomeActionStore()
    return app


@pytest.mark.asyncio
async def test_lists_fish_summaries() -> None:
    store = FakeApiStore()
    store.fishes = [
        FishSummary(
            fish_id=FISH_ID,
            last_message="最近一条",
            last_active_at=NOW,
            message_count=3,
        )
    ]
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get("/v1/fishes")

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "fishId": str(FISH_ID),
                "lastMessage": "最近一条",
                "lastActiveAt": "2026-07-20T08:30:00Z",
                "messageCount": 3,
                "isAssistant": False,
                "name": "",
                "role": "",
                "evaluation": False,
            }
        ]
    }


@pytest.mark.asyncio
async def test_lists_fish_messages_with_requested_limit() -> None:
    store = FakeApiStore()
    store.messages = [
        Message(
            id=MESSAGE_ID,
            fish_id=FISH_ID,
            session_id=DEFAULT_SESSION_ID,
            sequence=2,
            role=Role.ASSISTANT,
            content="说到一半",
            status=MessageStatus.INTERRUPTED,
            created_at=NOW,
            consumed_offset=2,
            finished_at=NOW,
        )
    ]
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(f"/v1/fishes/{FISH_ID}/messages?limit=2")

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "messageId": str(MESSAGE_ID),
                "sequence": 2,
                "role": "assistant",
                "content": "说到",
                "status": "interrupted",
                "consumedOffset": 2,
                "createdAt": "2026-07-20T08:30:00Z",
                "finishedAt": "2026-07-20T08:30:00Z",
            }
        ]
    }
    assert store.message_limits == [(FISH_ID, 2)]
    assert store.message_sessions == [DEFAULT_SESSION_ID]


@pytest.mark.asyncio
async def test_lists_messages_from_explicit_session() -> None:
    store = FakeApiStore()
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(
            f"/v1/fishes/{FISH_ID}/messages?sessionId={DEFAULT_SESSION_ID}"
        )

    assert response.status_code == 200
    assert store.message_sessions == [DEFAULT_SESSION_ID]


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, 501])
async def test_rejects_message_limit_outside_bounds(limit: int) -> None:
    transport = httpx.ASGITransport(app=build_test_app(FakeApiStore()))

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(f"/v1/fishes/{FISH_ID}/messages?limit={limit}")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_delete_fish_returns_no_content() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    app = build_test_app(store, coordinator)
    _, events = app.state.pet_event_hub.subscribe(control_only=True)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.delete(f"/v1/fishes/{FISH_ID}")

    assert response.status_code == 204
    assert response.content == b""
    assert coordinator.delete_calls == [FISH_ID]
    assert json.loads(events.get_nowait().value)["type"] == "pet.assistant.changed"


@pytest.mark.asyncio
async def test_delete_missing_fish_returns_not_found() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    coordinator.delete_result = False
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.delete(f"/v1/fishes/{FISH_ID}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "fish_not_found"


@pytest.mark.asyncio
async def test_lists_complete_message_performance_payloads() -> None:
    performance_store = FakePerformanceStore()
    performance_store.steps = [
        PerformanceStep(
            name="llm.answer",
            index=0,
            started_offset_us=12_000,
            duration_us=900_000,
            input_payload={
                "model": "qwen3.7-plus",
                "messages": [{"role": "system", "content": "完整 prompt"}],
            },
            output_payload={"content": "完整回答"},
        )
    ]
    transport = httpx.ASGITransport(
        app=build_test_app(
            FakeApiStore(),
            performance_store=performance_store,
        )
    )

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/performance"
        )

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "stepName": "llm.answer",
                "stepIndex": 0,
                "startedOffsetUs": 12_000,
                "durationUs": 900_000,
                "inputPayload": {
                    "model": "qwen3.7-plus",
                    "messages": [{"role": "system", "content": "完整 prompt"}],
                },
                "outputPayload": {"content": "完整回答"},
                "error": None,
            }
        ],
        "homeActions": [],
    }
    assert performance_store.calls == [(FISH_ID, MESSAGE_ID)]


@pytest.mark.asyncio
async def test_message_performance_includes_home_assistant_execution() -> None:
    home_action_store = FakeHomeActionStore()
    home_action_store.runs = [
        HomeActionRun(
            id=UUID("00000000-0000-0000-0000-000000000020"),
            sequence=3,
            event_sequence=5,
            fish_id=FISH_ID,
            session_id=DEFAULT_SESSION_ID,
            source_message_id=MESSAGE_ID,
            action={
                "entityId": "cover.living_room",
                "operation": "open",
                "value": None,
            },
            status=HomeActionStatus.SUCCEEDED,
            entity_id="cover.living_room",
            service_domain="cover",
            service_name="open_cover",
            accepted=True,
            confirmed=False,
            result={"state": "opening"},
            error=None,
            created_at=NOW,
            updated_at=NOW,
            finished_at=NOW,
        )
    ]
    transport = httpx.ASGITransport(
        app=build_test_app(
            FakeApiStore(),
            home_action_store=home_action_store,
        )
    )

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/performance"
        )

    assert response.status_code == 200
    assert response.json()["homeActions"] == [
        {
            "id": "00000000-0000-0000-0000-000000000020",
            "status": "succeeded",
            "action": {
                "entityId": "cover.living_room",
                "operation": "open",
                "value": None,
            },
            "entityId": "cover.living_room",
            "serviceDomain": "cover",
            "serviceName": "open_cover",
            "accepted": True,
            "confirmed": False,
            "result": {"state": "opening"},
            "error": None,
            "createdAt": "2026-07-20T08:30:00Z",
            "updatedAt": "2026-07-20T08:30:00Z",
            "finishedAt": "2026-07-20T08:30:00Z",
        }
    ]
    assert home_action_store.calls == [(FISH_ID, MESSAGE_ID)]


@pytest.mark.asyncio
async def test_persists_client_tts_metrics_for_message() -> None:
    performance_store = FakePerformanceStore()
    transport = httpx.ASGITransport(
        app=build_test_app(
            FakeApiStore(),
            performance_store=performance_store,
        )
    )
    metrics = {
        "messageId": str(MESSAGE_ID),
        "model": "qwen-audio-3.0-tts-flash",
        "voice": "longanhuan_v3.6",
        "status": "completed",
        "totalMs": 842.2,
        "playedOffset": 11,
    }

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/tts-performance",
            json={"metrics": metrics},
        )

    assert response.status_code == 204
    assert performance_store.tts_calls == [(FISH_ID, MESSAGE_ID, metrics)]


def pet_voice_metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "status": "completed",
        "userMessageId": "00000000-0000-0000-0000-000000000011",
        "answerMessageId": str(MESSAGE_ID),
        "provider": "dashscope",
        "model": "qwen-audio-3.0-tts-flash",
        "voice": "longanhuan_v3.6",
        "connectionReused": True,
        "nativeMs": {
            "endpointToAsrFinal": 80.1,
            "endpointToAsk": 85.2,
            "askToAnswerEvent": 400.0,
            "askToFirstContent": 431.9,
            "firstContentToTtsStart": 1.0,
            "ttsStartToReady": 52.5,
            "ttsReadyToTextSubmitted": 150.0,
            "ttsTextToFirstAudio": 442.7,
            "firstAudioToPcmQueued": 0.1,
            "pcmQueuedToPlayback": 2.0,
            "askToPlayback": 1080.0,
            "endpointToPlayback": 1165.2,
        },
    }
    metrics.update(overrides)
    return metrics


@pytest.mark.asyncio
async def test_persists_pet_voice_metrics_for_the_answer_message() -> None:
    performance_store = FakePerformanceStore()
    transport = httpx.ASGITransport(
        app=build_test_app(FakeApiStore(), performance_store=performance_store)
    )
    metrics = pet_voice_metrics()

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/pet-voice-performance",
            json={"metrics": metrics},
        )

    assert response.status_code == 204
    assert performance_store.pet_voice_calls == [(FISH_ID, MESSAGE_ID, metrics)]


@pytest.mark.asyncio
async def test_persists_draft_and_acoustic_pet_voice_metrics() -> None:
    performance_store = FakePerformanceStore()
    transport = httpx.ASGITransport(
        app=build_test_app(FakeApiStore(), performance_store=performance_store)
    )
    metrics = pet_voice_metrics()
    assert isinstance(metrics["nativeMs"], dict)
    native_ms = dict(metrics["nativeMs"])
    native_ms.update(
        {
            "acousticEndToFirstContent": 481.2,
            "acousticEndToPlayback": 604.8,
        }
    )
    metrics["nativeMs"] = native_ms
    metrics["voiceDraft"] = {
        "draftLeadMs": 280.0,
        "draftFirstContentLeadMs": 50.0,
        "draftUsefulHit": True,
        "draftPromoted": True,
        "draftDiscardedPcmBytes": 0,
        "selectedRevision": 1,
        "startedCount": 1,
        "transcriptReadyCount": 1,
        "requestStartedCount": 1,
        "firstContentCount": 1,
        "promotedCount": 1,
        "canceledCount": 0,
        "fallbackCount": 0,
        "lateEventDroppedCount": 0,
        "cleanupCount": 1,
        "ttsProviderReadyCount": 1,
        "ttsTextSubmittedCount": 1,
        "ttsFirstAudioBufferedCount": 1,
        "ttsPcmPromotedCount": 1,
        "ttsPcmDiscardedCount": 0,
        "attempts": [
            {
                "revision": 1,
                "startedCount": 1,
                "firstContentCount": 1,
                "promotedCount": 1,
                "canceledCount": 0,
                "fallbackCount": 0,
                "lateEventDroppedCount": 0,
                "draftFirstContentLeadMs": 50.0,
                "draftUsefulHit": True,
            }
        ],
    }

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/pet-voice-performance",
            json={"metrics": metrics},
        )

    assert response.status_code == 204
    assert performance_store.pet_voice_calls == [(FISH_ID, MESSAGE_ID, metrics)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metrics",
    [
        pet_voice_metrics(status="unknown"),
        pet_voice_metrics(
            nativeMs={
                "endpointToAsrFinal": -1,
                "endpointToAsk": None,
                "askToAnswerEvent": None,
                "askToFirstContent": None,
                "firstContentToTtsStart": None,
                "ttsStartToReady": None,
                "ttsReadyToTextSubmitted": None,
                "ttsTextToFirstAudio": None,
                "firstAudioToPcmQueued": None,
                "pcmQueuedToPlayback": None,
                "askToPlayback": None,
                "endpointToPlayback": None,
            }
        ),
    ],
)
async def test_rejects_invalid_pet_voice_metrics(metrics: dict[str, object]) -> None:
    performance_store = FakePerformanceStore()
    transport = httpx.ASGITransport(
        app=build_test_app(FakeApiStore(), performance_store=performance_store)
    )

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/pet-voice-performance",
            json={"metrics": metrics},
        )

    assert response.status_code == 400
    assert performance_store.pet_voice_calls == []


@pytest.mark.asyncio
async def test_rejects_nonfinite_voice_draft_metrics() -> None:
    performance_store = FakePerformanceStore()
    transport = httpx.ASGITransport(
        app=build_test_app(FakeApiStore(), performance_store=performance_store)
    )
    metrics = pet_voice_metrics(voiceDraft={"draftLeadMs": float("nan")})

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/pet-voice-performance",
            content=json.dumps({"metrics": metrics}),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert performance_store.pet_voice_calls == []


@pytest.mark.asyncio
async def test_rejects_pet_voice_metrics_for_a_different_answer() -> None:
    performance_store = FakePerformanceStore()
    transport = httpx.ASGITransport(
        app=build_test_app(FakeApiStore(), performance_store=performance_store)
    )
    metrics = pet_voice_metrics(answerMessageId="00000000-0000-0000-0000-000000000099")

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/pet-voice-performance",
            json={"metrics": metrics},
        )

    assert response.status_code == 400
    assert performance_store.pet_voice_calls == []


@pytest.mark.asyncio
async def test_missing_pet_voice_answer_returns_not_found() -> None:
    performance_store = FakePerformanceStore()
    performance_store.pet_voice_result = False
    transport = httpx.ASGITransport(
        app=build_test_app(FakeApiStore(), performance_store=performance_store)
    )

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.post(
            f"/internal/fishes/{FISH_ID}/messages/{MESSAGE_ID}/pet-voice-performance",
            json={"metrics": pet_voice_metrics()},
        )

    assert response.status_code == 404


def _timeline_message(sequence: int, content: str) -> Message:
    return Message(
        id=UUID(int=sequence),
        fish_id=FISH_ID,
        session_id=DEFAULT_SESSION_ID,
        sequence=sequence,
        role=Role.USER,
        content=content,
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_timeline_pages_the_whole_conversation() -> None:
    store = FakeApiStore()
    store.messages = [_timeline_message(4, "旧的"), _timeline_message(9, "新的")]
    transport = httpx.ASGITransport(app=build_test_app(store))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(f"/v1/fishes/{FISH_ID}/timeline?limit=1&before=20")
        both = await client.get(f"/v1/fishes/{FISH_ID}/timeline?before=3&after=4")

    assert response.status_code == 200
    body = response.json()
    assert [item["content"] for item in body["items"]] == ["新的"]
    assert body["hasBefore"] is True and body["maxSequence"] == 9
    assert store.timeline_calls == [(FISH_ID, 1, 20, None)]
    assert both.status_code == 422


@pytest.mark.asyncio
async def test_search_finds_messages_by_text() -> None:
    store = FakeApiStore()
    store.messages = [_timeline_message(4, "构建报错了"), _timeline_message(9, "好了")]
    transport = httpx.ASGITransport(app=build_test_app(store))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(
            f"/v1/fishes/{FISH_ID}/messages/search", params={"q": " 报错 "}
        )
        empty = await client.get(f"/v1/fishes/{FISH_ID}/messages/search?q=")

    assert response.status_code == 200
    assert [item["content"] for item in response.json()["items"]] == ["构建报错了"]
    assert store.searches == [(FISH_ID, "报错", 50)]
    assert empty.status_code == 400


@pytest.mark.asyncio
async def test_timeline_says_why_ta_spoke_up() -> None:
    from fish_memory.db.messages import MessageOrigin

    store = FakeApiStore()
    sensed, idle, task, plugin, asked = (
        _timeline_message(sequence, text)
        for sequence, text in [
            (1, "还在改代码呀"),
            (2, "想到一件事"),
            (3, "查好了"),
            (4, "提醒"),
            (5, "嗯"),
        ]
    )
    store.messages = [
        replace(message, role=Role.ASSISTANT)
        for message in (sensed, idle, task, plugin)
    ] + [asked]
    store.origins = {
        sensed.id: MessageOrigin(
            "proactive_observation", "environment_history", "window", "VS Code"
        ),
        idle.id: MessageOrigin("proactive_observation"),
        task.id: MessageOrigin("agent"),
        plugin.id: MessageOrigin("plugin", "plugin", "plugin.event"),
    }
    transport = httpx.ASGITransport(app=build_test_app(store))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        response = await client.get(f"/v1/fishes/{FISH_ID}/timeline")

    assert [item.get("origin") for item in response.json()["items"]] == [
        {"kind": "sense", "appName": "VS Code"},
        {"kind": "self"},
        {"kind": "task"},
        {"kind": "plugin"},
        None,
    ]


@pytest.mark.asyncio
async def test_timeline_shows_what_ta_never_said_after_an_interruption() -> None:
    store = FakeApiStore()
    cut = replace(
        _timeline_message(2, "今天天气不错，我们出去走走吧。"),
        role=Role.ASSISTANT,
        status=MessageStatus.INTERRUPTED,
        consumed_offset=7,
    )
    whole = replace(
        _timeline_message(3, "好的。"),
        role=Role.ASSISTANT,
        status=MessageStatus.COMPLETED,
    )
    store.messages = [cut, whole]
    store.unsaid_by_id = {cut.id: "我们出去走走吧。"}
    transport = httpx.ASGITransport(app=build_test_app(store))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        items = (await client.get(f"/v1/fishes/{FISH_ID}/timeline")).json()["items"]

    assert items[0]["content"] == "今天天气不错，"
    assert items[0]["unsaid"] == "我们出去走走吧。"
    assert "unsaid" not in items[1]


@pytest.mark.asyncio
async def test_timeline_says_whether_a_message_was_typed_or_spoken() -> None:
    store = FakeApiStore()
    typed, spoken, unknown = (
        _timeline_message(sequence, text)
        for sequence, text in [(1, "打的字"), (2, "说的话"), (3, "导入的")]
    )
    store.messages = [typed, spoken, unknown]
    store.input_modes_by_id = {typed.id: "typed", spoken.id: "spoken"}
    transport = httpx.ASGITransport(app=build_test_app(store))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", headers=AUTH_HEADERS
    ) as client:
        items = (await client.get(f"/v1/fishes/{FISH_ID}/timeline")).json()["items"]

    assert [item.get("inputMode") for item in items] == ["typed", "spoken", None]
