from datetime import datetime
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.agent.models import AgentMessageSubjectKind
from fish_memory.agent.replies import (
    AgentReplyClassificationError,
    AgentReplyConflictError,
    AgentReplyContext,
    AgentReplyResult,
)
from fish_memory.api.routes import router
from fish_memory.db.embedding_spaces import EmbeddingSpaceMismatchError
from fish_memory.db.messages import FishDeletedError, LegacyFishReadOnlyError
from fish_memory.embeddings.client import EmbeddingSpace

from .fakes import (
    DEFAULT_SESSION_ID,
    FakeApiCoordinator,
    FakeApiStore,
    FakeSessionStore,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")


def build_test_app(
    store: FakeApiStore,
    coordinator: FakeApiCoordinator,
    home_service: object | None = None,
    reply_service: object | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.message_store = store
    app.state.session_store = FakeSessionStore()
    app.state.answer_coordinator = coordinator
    app.state.home_automation_service = home_service
    if reply_service is not None:
        app.state.agent_reply_service = reply_service
    app.include_router(router)
    return app


class FakeAssistantStore:
    def __init__(self) -> None:
        self.assistant_id: UUID | None = None

    async def get_assistant(self) -> UUID | None:
        return self.assistant_id

    async def bind_assistant(self, fish_id: UUID) -> bool:
        self.assistant_id = fish_id
        return True


class FakeHomeService:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.notifications = []

    def notify_user_message(self, *args, **kwargs) -> None:
        self.events.append("home_notify")
        self.notifications.append((args, kwargs))


class FakeReplyService:
    def __init__(
        self,
        store: FakeApiStore,
        *,
        handled: bool,
        inserted: bool = True,
    ) -> None:
        self.store = store
        self.handled = handled
        self.inserted = inserted
        self.calls: list[tuple[object, ...]] = []
        self.error: Exception | None = None

    async def handle(
        self,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        context: AgentReplyContext,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyResult:
        assert self.store.contents == []
        self.calls.append(
            (fish_id, session_id, content, context, message_id, created_at)
        )
        if self.error is not None:
            raise self.error
        message = await self.store.insert_user_message(
            fish_id,
            content,
            session_id=session_id,
            message_id=message_id,
            created_at=created_at,
        )
        return AgentReplyResult(
            message=message,
            handled=self.handled,
            inserted=self.inserted,
            replayed=not self.inserted,
        )


@pytest.mark.asyncio
async def test_ask_persists_original_content_before_notifying() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "  原始内容  "},
        )

    assert response.status_code == 201
    assert response.json() == {
        "messageId": "00000000-0000-0000-0000-000000000010",
        "sessionId": str(DEFAULT_SESSION_ID),
        "sequence": 7,
        "status": "initialized",
        "createdAt": "2026-07-20T08:30:00Z",
        "answerExpected": True,
    }
    assert store.contents == [(FISH_ID, "  原始内容  ")]
    assert store.events == ["persist", "notify"]
    notification = coordinator.notifications[0]
    assert notification[:4] == (
        FISH_ID,
        DEFAULT_SESSION_ID,
        7,
        UUID("00000000-0000-0000-0000-000000000010"),
    )
    assert len(coordinator.source_messages) == 1
    assert coordinator.source_messages[0].id == notification[3]
    assert coordinator.source_messages[0].content == "  原始内容  "


@pytest.mark.asyncio
async def test_first_desktop_chat_binds_implicitly_created_fish_as_assistant() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    app = build_test_app(store, coordinator)
    assistant_store = FakeAssistantStore()
    app.state.fish_store = assistant_store
    app.state.settings = type("Settings", (), {"desktop_mode": True})()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "第一次聊天"},
        )

    assert response.status_code == 201
    assert assistant_store.assistant_id == FISH_ID


@pytest.mark.asyncio
async def test_ask_passes_message_tracker_with_persist_step() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "你好"},
        )

    assert response.status_code == 201
    _, _, _, message_id, tracker = coordinator.notifications[0]
    assert message_id == UUID(response.json()["messageId"])
    steps = tracker.snapshot()
    assert [step.name for step in steps] == ["ask.persist_message"]
    assert steps[0].duration_us >= 0


@pytest.mark.asyncio
async def test_ask_leaves_home_router_start_to_answer_coordinator() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    home = FakeHomeService(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator, home))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "打开玄关灯"},
        )

    assert response.status_code == 201
    assert store.events == ["persist", "notify"]
    assert home.notifications == []


@pytest.mark.asyncio
async def test_ask_uses_explicit_session_and_returns_it() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "旧会话继续", "sessionId": str(DEFAULT_SESSION_ID)},
        )

    assert response.status_code == 201
    assert response.json()["sessionId"] == str(DEFAULT_SESSION_ID)
    assert coordinator.notifications[0][1] == DEFAULT_SESSION_ID


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "  \n\t"])
async def test_ask_rejects_blank_content(content: str) -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": content},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_content"
    assert store.contents == []


@pytest.mark.asyncio
async def test_ask_rejects_content_over_unicode_limit() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "鱼" * 32_001},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "content_too_large"


@pytest.mark.asyncio
async def test_ask_does_not_notify_when_persistence_fails() -> None:
    store = FakeApiStore()
    store.error = RuntimeError("database unavailable")
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(
        app=build_test_app(store, coordinator),
        raise_app_exceptions=False,
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "hello"},
        )

    assert response.status_code == 500
    assert coordinator.notifications == []


@pytest.mark.asyncio
async def test_ask_deleted_fish_returns_gone() -> None:
    store = FakeApiStore()
    store.error = FishDeletedError(str(FISH_ID))
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "回来"},
        )

    assert response.status_code == 410
    assert response.json()["error"]["code"] == "fish_deleted"
    assert coordinator.notifications == []


@pytest.mark.asyncio
async def test_ask_legacy_fish_returns_conflict_without_notifying() -> None:
    store = FakeApiStore()
    store.error = LegacyFishReadOnlyError(str(FISH_ID))
    coordinator = FakeApiCoordinator(store.events)
    app = build_test_app(store, coordinator)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "回来"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "legacy_fish_read_only"
    assert coordinator.notifications == []
    assert not hasattr(app.state, "fish_store")


@pytest.mark.asyncio
async def test_ask_returns_conflict_when_fish_embedding_space_differs() -> None:
    store = FakeApiStore()
    store.error = EmbeddingSpaceMismatchError(
        FISH_ID,
        EmbeddingSpace("openai-compatible", "text-embedding-v4", 1024),
        EmbeddingSpace("local", "intfloat/multilingual-e5-small", 384),
    )
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "hello"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "embedding_space_mismatch"
    assert "1024" in response.json()["error"]["message"]
    assert "384" in response.json()["error"]["message"]
    assert coordinator.notifications == []


@pytest.mark.asyncio
async def test_ask_reply_context_requires_session_and_forbids_extra() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing_session = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "允许",
                "replyContext": {"kind": "interaction", "id": str(UUID(int=301))},
            },
        )
        extra = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "允许",
                "sessionId": str(DEFAULT_SESSION_ID),
                "replyContext": {
                    "kind": "interaction",
                    "id": str(UUID(int=301)),
                    "sourceMessageId": str(UUID(int=302)),
                },
            },
        )
        bad_kind = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "允许",
                "sessionId": str(DEFAULT_SESSION_ID),
                "replyContext": {"kind": "task", "id": str(UUID(int=301))},
            },
        )

    assert missing_session.status_code == 422
    assert extra.status_code == 422
    assert bad_kind.status_code == 422
    assert store.contents == []
    assert coordinator.notifications == []


@pytest.mark.asyncio
async def test_ask_reply_context_fails_closed_when_service_is_absent() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    transport = httpx.ASGITransport(app=build_test_app(store, coordinator))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "允许",
                "sessionId": str(DEFAULT_SESSION_ID),
                "replyContext": {"kind": "interaction", "id": str(UUID(int=301))},
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "agent_replies_unavailable"
    assert store.contents == []
    assert coordinator.notifications == []


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [True, False])
async def test_ask_routes_exact_reply_before_persist_and_returns_answer_semantics(
    handled: bool,
) -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    replies = FakeReplyService(store, handled=handled)
    transport = httpx.ASGITransport(
        app=build_test_app(store, coordinator, reply_service=replies)
    )
    context_id = UUID(int=301)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "reply",
                "sessionId": str(DEFAULT_SESSION_ID),
                "replyContext": {"kind": "run_result", "id": str(context_id)},
            },
        )

    assert response.status_code == 201
    assert response.json()["answerExpected"] is (not handled)
    assert replies.calls[0][3] == AgentReplyContext(
        AgentMessageSubjectKind.RUN_RESULT, context_id
    )
    assert store.contents == [(FISH_ID, "reply")]
    assert len(coordinator.notifications) == (0 if handled else 1)


@pytest.mark.asyncio
async def test_ask_ordinary_reply_replay_does_not_notify_twice() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    replies = FakeReplyService(store, handled=False, inserted=False)
    transport = httpx.ASGITransport(
        app=build_test_app(store, coordinator, reply_service=replies)
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "thanks",
                "sessionId": str(DEFAULT_SESSION_ID),
                "replyContext": {"kind": "run_result", "id": str(UUID(int=301))},
            },
        )

    assert response.status_code == 201
    assert response.json()["answerExpected"] is True
    assert coordinator.notifications == []


@pytest.mark.asyncio
async def test_ask_reply_conflict_is_409_without_ordinary_insert_or_notify() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    replies = FakeReplyService(store, handled=True)
    replies.error = AgentReplyConflictError("stale subject")
    transport = httpx.ASGITransport(
        app=build_test_app(store, coordinator, reply_service=replies)
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "reply",
                "sessionId": str(DEFAULT_SESSION_ID),
                "replyContext": {"kind": "interaction", "id": str(UUID(int=301))},
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "reply_context_conflict"
    assert store.contents == []
    assert coordinator.notifications == []


@pytest.mark.asyncio
async def test_ask_reply_classification_failure_is_distinct_and_fail_closed() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    replies = FakeReplyService(store, handled=True)
    replies.error = AgentReplyClassificationError("classifier unavailable")
    transport = httpx.ASGITransport(
        app=build_test_app(store, coordinator, reply_service=replies)
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "reply",
                "sessionId": str(DEFAULT_SESSION_ID),
                "replyContext": {"kind": "run_result", "id": str(UUID(int=301))},
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "reply_classification_failed"
    assert store.contents == []
    assert coordinator.notifications == []


class FakeAttachments:
    def __init__(self, events: list[str], *, waiting: bool = True) -> None:
        self.events = events
        self.free = waiting
        self.attached: list[tuple[UUID, UUID, list[UUID]]] = []

    async def waiting(self, fish_id, attachment_ids) -> bool:
        return self.free

    async def attach(self, fish_id, message_id, attachment_ids) -> None:
        self.events.append("attach")
        self.attached.append((fish_id, message_id, list(attachment_ids)))


ATTACHMENT = UUID("00000000-0000-0000-0000-0000000000aa")


@pytest.mark.asyncio
async def test_ask_gives_uploads_to_the_message_before_its_reply_starts() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    app = build_test_app(store, coordinator)
    app.state.message_attachments = attachments = FakeAttachments(store.events)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={
                "content": "看图 [发来图片：a.png]",
                "attachmentIds": [str(ATTACHMENT)],
            },
        )

    assert response.status_code == 201
    assert store.events == ["persist", "attach", "notify"]
    assert attachments.attached == [
        (FISH_ID, UUID("00000000-0000-0000-0000-000000000010"), [ATTACHMENT])
    ]


@pytest.mark.asyncio
async def test_ask_with_uploads_already_gone_saves_nothing() -> None:
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    app = build_test_app(store, coordinator)
    app.state.message_attachments = FakeAttachments(store.events, waiting=False)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/ask",
            json={"content": "看图", "attachmentIds": [str(ATTACHMENT)]},
        )

    assert response.status_code == 400
    assert store.events == []
