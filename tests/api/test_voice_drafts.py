import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.agent.models import (
    AgentEndpoint,
    AgentMessageSubjectKind,
    AgentResolvedTarget,
    AgentRun,
    AgentTarget,
    ConnectorTypeSlug,
)
from fish_memory.agent.replies import (
    AgentReplyClassificationError,
    AgentReplyContext,
    AgentReplyDisposition,
    AgentReplyResult,
    AgentReplySubjectSnapshot,
    PreparedAgentReply,
    RunResultReplyAction,
)
from fish_memory.agent.routing import (
    AgentTaskDecisionKind,
    PreparedAgentTaskRoute,
    TaskLaneContext,
)
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.output_sink import AnswerOutputSnapshot, PromotableAnswerSink
from fish_memory.answer.parallel_generation import (
    ParallelAnswerGenerationPipeline,
    PreparedInitialAnswer,
)
from fish_memory.answer.sse import iter_answer_events
from fish_memory.answer.voice_draft_types import (
    VoiceDraftKey,
    VoiceDraftState,
    exact_content_hash,
    voice_answer_message_id,
)
from fish_memory.answer.voice_drafts import (
    PrepareVoiceDraft,
    VoiceDraftCoordinator,
    VoiceDraftEvent,
    VoiceDraftLease,
    VoiceDraftRevisionConflict,
    VoiceDraftSnapshot,
)
from fish_memory.api.routes import internal_router
from fish_memory.app import _voice_draft_preparer
from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.service import ApiAuthService
from fish_memory.db.messages import LegacyFishReadOnlyError
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.models import Message
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion

from .fakes import (
    DEFAULT_FISH_ID,
    DEFAULT_SESSION_ID,
    FakeApiCoordinator,
    FakeApiStore,
    FakeSessionStore,
)

FISH_ID = DEFAULT_FISH_ID
TURN_ID = UUID("00000000-0000-0000-0000-000000000123")
DRAFT_CREATED_AT = datetime(2026, 7, 20, 8, 29, 59, tzinfo=UTC)
BASE_PATH = (
    f"/internal/fishes/{FISH_ID}/sessions/{DEFAULT_SESSION_ID}/voice-turns/{TURN_ID}"
)


class FakePrepared:
    def __init__(self) -> None:
        self.sink = PromotableAnswerSink()
        self.prepared_reply: PreparedAgentReply | None = None
        self.cancelled = False
        self.settled = asyncio.Event()

    def cancel(self) -> asyncio.Task[None]:
        self.cancelled = True

        async def settle() -> None:
            self.settled.set()

        return asyncio.create_task(settle())


class ReadyGuardPrepared(FakePrepared):
    @property
    def input_hash(self) -> str:
        return "guard-input-hash"

    def initial_done(self) -> bool:
        return True

    async def initial_ready(self) -> None:
        return None

    def output_token_usage(self) -> int | None:
        return None


def lease_for(prepared: FakePrepared | None = None) -> VoiceDraftLease:
    item = prepared or FakePrepared()
    return VoiceDraftLease(
        key=VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID),
        revision=1,
        candidate_hash=exact_content_hash("最终问题"),
        input_hash="committed-input-hash",
        created_at=DRAFT_CREATED_AT,
        prepared=cast(PreparedInitialAnswer, item),
    )


class ProductionPrecommitStore(FakeApiStore):
    def __init__(self) -> None:
        super().__init__()
        self.readonly_history_calls = 0
        self.writable_history_calls = 0

    async def load_history_readonly(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
        appended_history_token_count: int | None = None,
    ) -> list[Message]:
        del fish_id, session_id, before_sequence, limit, appended_history_token_count
        self.readonly_history_calls += 1
        return []

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        del fish_id, session_id, before_sequence, limit
        self.writable_history_calls += 1
        return []


class ProductionLaneClient:
    def __init__(self) -> None:
        self.initial_lane_calls: list[str] = []
        self.web_search_calls: list[str] = []
        self.continuation_calls: list[str] = []
        self.responses = {
            "recall": '{"m":"neutral","a":"R。","r":1,"q":"旧问题","e":0}',
            "web": '{"w":1,"m":"neutral","a":"W。","q":"联网问题"}',
            "home": '{"h":1,"t":"control","m":"happy","a":"H。"}',
            "task": ('{"t":1,"m":"happy","a":"私有 T 确认。"}'),
        }

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search
        lane = self._lane(messages, force_search=force_search)
        if lane == "search":
            self.web_search_calls.append(lane)
            yield ChatStreamChunk(content='{"searchContext":"不应执行"}')
            return
        if lane == "continuation":
            self.continuation_calls.append(lane)
            yield ChatStreamChunk(content='{"append":"不应执行"}')
            return
        self.initial_lane_calls.append(lane)
        await asyncio.sleep(0)
        yield ChatStreamChunk(content=self.responses[lane])

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        async for chunk in self.stream_json(
            messages,
            enable_search=enable_search,
            force_search=force_search,
        ):
            yield chunk

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        del enable_search
        return {
            "model": "production-boundary-fake",
            "lane": self._lane(messages, force_search=force_search),
            "messages": [dict(message) for message in messages],
        }

    def request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        return self.json_stream_request_payload(
            messages,
            enable_search=enable_search,
            force_search=force_search,
        )

    async def close(self) -> None:
        return None

    @staticmethod
    def _lane(messages: Sequence[ChatMessage], *, force_search: bool) -> str:
        if force_search:
            return "search"
        system = messages[0]["content"]
        if system.startswith("# R 通道"):
            return "recall"
        if system.startswith("# W 通道"):
            return "web"
        if system.startswith("# H 通道"):
            return "home"
        if system.startswith("# T 通道"):
            return "task"
        return "continuation"


class RecordingRecallService:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def recall(self, *args: object, **kwargs: object) -> object:
        self.calls.append((*args, kwargs))
        raise AssertionError("Recall must stay behind authoritative commit")


def draft_agent_target() -> AgentResolvedTarget:
    endpoint = AgentEndpoint(
        id=UUID(int=701),
        connector_type=ConnectorTypeSlug("hermes"),
        base_url="http://localhost:9000",
        created_at=DRAFT_CREATED_AT,
        updated_at=DRAFT_CREATED_AT,
    )
    target = AgentTarget(
        id=UUID(int=700),
        endpoint_id=endpoint.id,
        external_key="default",
        route_prefix="",
        credential_ref="hermes:test",
        connector_config={},
        display_name="Hermes",
        enabled=True,
        created_at=DRAFT_CREATED_AT,
        updated_at=DRAFT_CREATED_AT,
    )
    return AgentResolvedTarget(target=target, endpoint=endpoint)


class ForbiddenTaskRouteService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.preparations: list[PreparedAgentTaskRoute] = []

    async def prepare_selected_route(self, **kwargs: object) -> PreparedAgentTaskRoute:
        target = cast(AgentResolvedTarget, kwargs["resolved_target"])
        prepared = PreparedAgentTaskRoute(
            fish_id=cast(UUID, kwargs["fish_id"]),
            target_id=target.target.id,
            kind=cast(AgentTaskDecisionKind, kwargs["kind"]),
            content=cast(str, kwargs["content"]),
        )
        self.preparations.append(prepared)
        return prepared

    async def commit_prepared_route(
        self, prepared: PreparedAgentTaskRoute, **kwargs: object
    ) -> AgentRun:
        self.calls.append({"prepared": prepared, **kwargs})
        raise AssertionError("private draft must not commit a Task route")

    async def commit_selected_route(self, **kwargs: object) -> AgentRun:
        self.calls.append(dict(kwargs))
        raise AssertionError("private draft must not commit a Task route")


class FakeVoiceDraftCoordinator:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.upserts: list[tuple[VoiceDraftKey, int, str]] = []
        self.reply_contexts: list[AgentReplyContext | None] = []
        self.cancels: list[tuple[VoiceDraftKey, int]] = []
        self.claims: list[tuple[VoiceDraftKey, int, int | None, str]] = []
        self.context_fingerprints: list[str | None] = []
        self.committed_contexts: dict[VoiceDraftKey, tuple[str, str | None]] = {}
        self.lease: VoiceDraftLease | None = None
        self.conflict = False
        self.cancelled = asyncio.Event()
        self.stream_started = asyncio.Event()
        self.guard_keys: list[VoiceDraftKey] = []

    @asynccontextmanager
    async def commit_guard(self, key: VoiceDraftKey) -> AsyncIterator[None]:
        self.guard_keys.append(key)
        yield

    async def upsert(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        reply_context: AgentReplyContext | None = None,
    ) -> VoiceDraftSnapshot | None:
        if self.conflict:
            raise VoiceDraftRevisionConflict("different content")
        self.upserts.append((key, revision, content))
        self.reply_contexts.append(reply_context)
        return snapshot_for(key, revision, content)

    async def cancel(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        state: VoiceDraftState = VoiceDraftState.CANCELLED,
    ) -> VoiceDraftSnapshot | None:
        del state
        self.cancels.append((key, revision))
        self.cancelled.set()
        return snapshot_for(key, revision, "候选问题", VoiceDraftState.CANCELLED)

    async def finalize(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        promote_revision: int | None,
        content: str,
        context_fingerprint: str | None = None,
    ) -> VoiceDraftLease | None:
        committed = self.committed_contexts.get(key)
        if committed is not None and committed != (content, context_fingerprint):
            raise VoiceDraftRevisionConflict("different reply context")
        self.committed_contexts[key] = (content, context_fingerprint)
        self.events.append("claim")
        self.claims.append((key, revision, promote_revision, content))
        self.context_fingerprints.append(context_fingerprint)
        lease = self.lease
        self.lease = None
        return lease

    async def iter_events(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
    ) -> AsyncIterator[VoiceDraftEvent]:
        self.stream_started.set()
        yield VoiceDraftEvent(
            event="draft.snapshot",
            event_id=f"{key.turn_id}:{revision}:snapshot:0",
            data={
                "turnId": str(key.turn_id),
                "revision": revision,
                "state": "active",
                "content": "私有回答",
                "flushOffset": 0,
            },
        )
        await self.cancelled.wait()
        yield VoiceDraftEvent(
            event="draft.canceled",
            event_id=f"{key.turn_id}:{revision}:terminal:1",
            data={
                "turnId": str(key.turn_id),
                "revision": revision,
                "state": "cancelled",
                "content": "私有回答",
                "flushOffset": 0,
            },
        )


class FakeReplyService:
    def __init__(self, store: FakeApiStore, *, handled: bool) -> None:
        self.store = store
        self.handled = handled
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
        assert message_id is not None
        inserted = message_id not in self.store.user_messages_by_id
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
            inserted=inserted,
            replayed=not inserted,
        )


class BlockingLeaseReplyService(FakeReplyService):
    def __init__(self, store: FakeApiStore) -> None:
        super().__init__(store, handled=False)
        self.holder_entered = asyncio.Event()
        self.release_holder = asyncio.Event()
        self.duplicate_entered = asyncio.Event()
        self.insert_order: list[str] = []

    async def handle(
        self,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        context: AgentReplyContext,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyResult:
        label = "holder" if created_at == DRAFT_CREATED_AT else "duplicate"
        if label == "holder":
            self.holder_entered.set()
            await self.release_holder.wait()
        else:
            self.duplicate_entered.set()
        self.insert_order.append(label)
        return await super().handle(
            fish_id,
            session_id,
            content,
            context,
            message_id=message_id,
            created_at=created_at,
        )


def snapshot_for(
    key: VoiceDraftKey,
    revision: int,
    content: str,
    state: VoiceDraftState = VoiceDraftState.ACTIVE,
) -> VoiceDraftSnapshot:
    return VoiceDraftSnapshot(
        key=key,
        revision=revision,
        content=content,
        candidate_hash=exact_content_hash(content),
        created_at=DRAFT_CREATED_AT,
        state=state,
        output=AnswerOutputSnapshot(
            emotion=PetEmotion.NEUTRAL,
            end=False,
            content="",
            flush_offset=0,
        ),
        ready=False,
    )


def build_voice_draft_app(
    *,
    with_drafts: bool = True,
    reply_service: object | None = None,
) -> tuple[FastAPI, FakeApiStore, FakeApiCoordinator, FakeVoiceDraftCoordinator]:
    app = FastAPI()
    store = FakeApiStore()
    coordinator = FakeApiCoordinator(store.events)
    drafts = FakeVoiceDraftCoordinator(store.events)
    app.state.message_store = store
    app.state.session_store = FakeSessionStore()
    app.state.answer_coordinator = coordinator
    if with_drafts:
        app.state.voice_draft_coordinator = drafts
    if reply_service is not None:
        app.state.agent_reply_service = reply_service
    app.include_router(internal_router)
    return app, store, coordinator, drafts


@pytest.mark.asyncio
async def test_draft_is_private_until_commit_across_public_runtime_and_sse() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    public_events: list[str] = []

    async def consume_public_events() -> None:
        async for event in iter_answer_events(coordinator.runtime):
            public_events.append(event.event)

    public_consumer = asyncio.create_task(consume_public_events())
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.put(
                f"{BASE_PATH}/draft",
                json={"revision": 1, "content": "候选问题"},
            )
        await asyncio.sleep(0)
    finally:
        public_consumer.cancel()
        await asyncio.gather(public_consumer, return_exceptions=True)

    assert response.status_code == 202
    assert response.json() == {
        "turnId": str(TURN_ID),
        "revision": 1,
        "content": "候选问题",
        "state": "active",
    }
    assert store.contents == []
    assert coordinator.notifications == []
    assert coordinator.voice_notifications == []
    assert coordinator.runtime.answer_message_id is None
    assert public_events == []
    assert drafts.upserts == [
        (VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID), 1, "候选问题")
    ]


@pytest.mark.asyncio
async def test_internal_voice_routes_require_api_authentication(tmp_path: Path) -> None:
    app, _, _, _ = build_voice_draft_app()
    auth = ApiAuthService(
        ApiKeyCredentialStore(
            api_key="voice-route-secret",
            key_file=tmp_path / "api-key",
        )
    )
    app.add_middleware(ApiAuthMiddleware, auth_service=auth)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        unauthorized = await client.put(
            f"{BASE_PATH}/draft",
            json={"revision": 1, "content": "候选问题"},
        )
        authorized = await client.put(
            f"{BASE_PATH}/draft",
            json={"revision": 1, "content": "候选问题"},
            headers={"Authorization": "Bearer voice-route-secret"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 202


@pytest.mark.asyncio
async def test_precommit_computation_has_no_public_output_or_mutations() -> None:
    app = FastAPI()
    store = ProductionPrecommitStore()
    lane_client = ProductionLaneClient()
    recall = RecordingRecallService()
    context_loader = AnswerContextLoader(
        store,
        now=lambda: DRAFT_CREATED_AT,
        system_prompt="system",
    )
    generation = ParallelAnswerGenerationPipeline(
        lane_client,
        recall_service=recall,  # ty: ignore[invalid-argument-type]
        recall_context_token_budget=1000,
        enable_search=True,
    )
    deferred_route_calls: list[str] = []

    for route_name in ("recall", "web", "home"):

        async def record_deferred_route(
            context: object,
            result: object,
            *,
            name: str = route_name,
        ) -> None:
            del context, result
            deferred_route_calls.append(name)

        generation._route_handlers[route_name] = record_deferred_route  # type: ignore[assignment]

    answer_coordinator = AnswerCoordinator(
        store,  # ty: ignore[invalid-argument-type]
        lane_client,
        system_prompt="system",
    )
    tts_calls: list[str] = []
    drafts = VoiceDraftCoordinator(
        prepare=_voice_draft_preparer(
            context_loader,
            generation,
            enable_home=True,
        ),
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    app.state.message_store = store
    app.state.session_store = FakeSessionStore()
    app.state.answer_coordinator = answer_coordinator
    app.state.voice_draft_coordinator = drafts
    app.include_router(internal_router)

    async def public_sse_to_tts() -> None:
        runtime = answer_coordinator.get_runtime(FISH_ID, DEFAULT_SESSION_ID)
        async for event in iter_answer_events(runtime):
            tts_calls.append(event.event)

    consumer = asyncio.create_task(public_sse_to_tts())
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as http_client:
            response = await http_client.put(
                f"{BASE_PATH}/draft",
                json={"revision": 1, "content": "打开客厅灯"},
            )

        async def preparation_ready() -> None:
            key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
            while True:
                entry = drafts._entries.get(key)
                if entry is not None and entry.ready:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(preparation_ready(), timeout=0.2)

        assert response.status_code == 202
        assert store.contents == []
        assert store.events == []
        assert store.created_ats == []
        assert store.readonly_history_calls == 1
        assert store.writable_history_calls == 0
        assert set(lane_client.initial_lane_calls) == {"recall", "web", "home"}
        assert recall.calls == []
        assert lane_client.web_search_calls == []
        assert lane_client.continuation_calls == []
        assert deferred_route_calls == ["home"]
        runtime = answer_coordinator.peek_runtime(FISH_ID, DEFAULT_SESSION_ID)
        assert runtime is not None
        assert runtime.answer_message_id is None
        assert tts_calls == []
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await drafts.close()
        await answer_coordinator.close()


@pytest.mark.asyncio
async def test_task_draft_keeps_confirmation_private_until_shared_release() -> None:
    app = FastAPI()
    store = ProductionPrecommitStore()
    lane_client = ProductionLaneClient()
    lane_client.responses["web"] = '{"w":0}'
    routes = ForbiddenTaskRouteService()
    context_loader = AnswerContextLoader(
        store,
        now=lambda: DRAFT_CREATED_AT,
        system_prompt="system",
    )
    generation = ParallelAnswerGenerationPipeline(
        lane_client,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=True,
        task_route_service=routes,
    )

    async def prepare_task_draft(
        key: VoiceDraftKey,
        revision: int,
        content: str,
        *,
        created_at: datetime,
    ) -> PreparedInitialAnswer:
        del revision
        tracker = MessagePerformanceTracker()
        loaded = await context_loader.load_for_draft(
            key.fish_id,
            key.session_id,
            turn_id=key.turn_id,
            content=content,
            created_at=created_at,
            tracker=tracker,
        )
        return generation.prepare_initial(
            key.fish_id,
            loaded.messages,
            loaded.recall_conversation,
            PromotableAnswerSink(),
            tracker,
            enable_home=False,
            agent_target=draft_agent_target(),
            task_lane_context=TaskLaneContext.empty(),
        )

    answer_coordinator = AnswerCoordinator(
        store,  # ty: ignore[invalid-argument-type]
        lane_client,
        system_prompt="system",
    )
    drafts = VoiceDraftCoordinator(
        prepare=prepare_task_draft,  # ty: ignore[invalid-argument-type]
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    app.state.message_store = store
    app.state.session_store = FakeSessionStore()
    app.state.answer_coordinator = answer_coordinator
    app.state.voice_draft_coordinator = drafts
    app.include_router(internal_router)
    public_events: list[str] = []

    async def consume_public_events() -> None:
        runtime = answer_coordinator.get_runtime(FISH_ID, DEFAULT_SESSION_ID)
        async for event in iter_answer_events(runtime):
            public_events.append(event.event)

    consumer = asyncio.create_task(consume_public_events())
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.put(
                f"{BASE_PATH}/draft",
                json={"revision": 1, "content": "帮我调查这个故障"},
            )

        async def task_draft_ready() -> None:
            key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
            while True:
                entry = drafts._entries.get(key)
                if entry is not None and entry.ready:
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(task_draft_ready(), timeout=0.2)
        entry = drafts._entries[VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)]
        runtime = answer_coordinator.get_runtime(FISH_ID, DEFAULT_SESSION_ID)

        assert response.status_code == 202
        assert set(lane_client.initial_lane_calls) == {"recall", "web", "task"}
        assert entry.output.content == "私有 T 确认。"
        assert entry.output.emotion is PetEmotion.HAPPY
        assert routes.calls == []
        assert len(routes.preparations) == 1
        assert runtime.answer_message_id is None
        assert runtime.content == ""
        assert runtime.flush_offset == 0
        assert runtime.pet_emotion is PetEmotion.NEUTRAL
        assert public_events == []
        assert store.contents == []
        assert store.events == []
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await drafts.close()
        await answer_coordinator.close()


@pytest.mark.asyncio
async def test_draft_validates_fish_session_ownership_before_preparation() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            BASE_PATH.replace(str(FISH_ID), str(UUID(int=999)), 1) + "/draft",
            json={"revision": 1, "content": "候选问题"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "session_not_found"
    assert drafts.upserts == []
    assert store.contents == []
    assert coordinator.voice_notifications == []


@pytest.mark.asyncio
async def test_draft_revision_conflict_returns_409() -> None:
    app, _, _, drafts = build_voice_draft_app()
    drafts.conflict = True
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            f"{BASE_PATH}/draft",
            json={"revision": 1, "content": "候选问题"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "revision_conflict"


@pytest.mark.asyncio
async def test_api_higher_cancel_revision_stops_real_draft_preparation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    drafts = VoiceDraftCoordinator(
        prepare=prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    app, _, _, _ = build_voice_draft_app()
    app.state.voice_draft_coordinator = drafts
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            accepted = await client.put(
                f"{BASE_PATH}/draft",
                json={"revision": 1, "content": "候选问题"},
            )
            await asyncio.wait_for(started.wait(), timeout=0.2)
            cancel_response = await client.post(
                f"{BASE_PATH}/cancel",
                json={"revision": 2, "reason": "speech_resumed"},
            )
            await asyncio.wait_for(cancelled.wait(), timeout=0.2)
            stale = await client.put(
                f"{BASE_PATH}/draft",
                json={"revision": 1, "content": "候选问题"},
            )
    finally:
        await drafts.close()

    assert accepted.status_code == 202
    assert cancel_response.status_code == 204
    assert stale.status_code == 202
    assert stale.json() == {
        "turnId": str(TURN_ID),
        "revision": 2,
        "content": "候选问题",
        "state": "cancelled",
    }


@pytest.mark.asyncio
async def test_api_cancel_before_upsert_fences_late_candidate_revision() -> None:
    prepare_calls = 0

    async def prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        nonlocal prepare_calls
        del key, revision, content
        prepare_calls += 1
        raise AssertionError("fenced candidate must not prepare")

    drafts = VoiceDraftCoordinator(
        prepare=prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    app, _, _, _ = build_voice_draft_app()
    app.state.voice_draft_coordinator = drafts
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            cancel_response = await client.post(
                f"{BASE_PATH}/cancel",
                json={"revision": 2, "reason": "speech_resumed"},
            )
            stale = await client.put(
                f"{BASE_PATH}/draft",
                json={"revision": 1, "content": "late candidate"},
            )
    finally:
        await drafts.close()

    assert cancel_response.status_code == 204
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "revision_conflict"
    assert prepare_calls == 0


@pytest.mark.asyncio
async def test_feature_disabled_draft_is_explicit_but_commit_falls_back() -> None:
    app, store, coordinator, drafts = build_voice_draft_app(with_drafts=False)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        draft_response = await client.put(
            f"{BASE_PATH}/draft",
            json={"revision": 1, "content": "候选问题"},
        )
        cancel_response = await client.post(
            f"{BASE_PATH}/cancel",
            json={"revision": 1, "reason": "speech_resumed"},
        )
        events_response = await client.get(
            f"{BASE_PATH}/events",
            params={"revision": 1},
        )
        commit_response = await client.post(
            f"{BASE_PATH}/commit",
            json={"revision": 2, "content": "最终问题"},
        )

    assert draft_response.status_code == 503
    assert draft_response.json()["error"]["code"] == "voice_drafts_disabled"
    assert cancel_response.status_code == 503
    assert cancel_response.json()["error"]["code"] == "voice_drafts_disabled"
    assert events_response.status_code == 503
    assert events_response.json()["error"]["code"] == "voice_drafts_disabled"
    assert commit_response.status_code == 201
    assert commit_response.json()["messageId"] == str(TURN_ID)
    assert store.created_ats == [None]
    assert len(coordinator.voice_notifications) == 1
    assert coordinator.voice_notifications[0][-1] is None
    assert drafts.claims == []


@pytest.mark.asyncio
async def test_matching_commit_claims_before_persist_and_preserves_draft_time() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    lease = lease_for()
    drafts.lease = lease
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 1,
                "content": "最终问题",
            },
        )

    assert response.status_code == 201
    assert response.json() == {
        "messageId": str(TURN_ID),
        "sessionId": str(DEFAULT_SESSION_ID),
        "sequence": 7,
        "status": "initialized",
        "createdAt": "2026-07-20T08:29:59Z",
        "answerExpected": True,
    }
    assert store.events == ["claim", "persist", "notify_voice"]
    assert store.created_ats == [DRAFT_CREATED_AT]
    notification = coordinator.voice_notifications[0]
    assert notification[:5] == (
        FISH_ID,
        DEFAULT_SESSION_ID,
        7,
        TURN_ID,
        voice_answer_message_id(TURN_ID),
    )
    assert notification[-1] is lease
    assert drafts.claims == [
        (
            VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID),
            2,
            1,
            "最终问题",
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_speech", [False, True])
async def test_commit_takes_over_pending_preparation_without_a_second_prepare(
    resume_speech: bool,
) -> None:
    prepared = ReadyGuardPrepared()
    prepare_started = asyncio.Event()
    prepare_release = asyncio.Event()
    prepare_calls = 0

    async def prepare(
        key: VoiceDraftKey, revision: int, content: str
    ) -> PreparedInitialAnswer:
        nonlocal prepare_calls
        del key, revision, content
        prepare_calls += 1
        prepare_started.set()
        await prepare_release.wait()
        return cast(PreparedInitialAnswer, prepared)

    drafts = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare),
        ttl_seconds=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    app, store, coordinator, _fake_drafts = build_voice_draft_app()
    app.state.voice_draft_coordinator = drafts
    transport = httpx.ASGITransport(app=app)
    commit: asyncio.Task[httpx.Response] | None = None
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            await client.put(
                f"{BASE_PATH}/draft", json={"revision": 1, "content": "最终问题"}
            )
            await asyncio.wait_for(prepare_started.wait(), timeout=0.2)
            commit = asyncio.create_task(
                client.post(
                    f"{BASE_PATH}/commit",
                    json={
                        "revision": 2,
                        "promoteRevision": 1,
                        "content": "最终问题",
                    },
                )
            )
            await asyncio.sleep(0)
            assert not commit.done()
            assert store.contents == []
            assert coordinator.voice_notifications == []
            if resume_speech:
                canceled = await client.post(
                    f"{BASE_PATH}/cancel",
                    json={"revision": 2, "reason": "speech_resumed"},
                )
                assert canceled.status_code == 204
            else:
                prepare_release.set()
            response = await asyncio.wait_for(commit, timeout=0.5)

        assert prepare_calls == 1
        if resume_speech:
            assert response.status_code == 409
            assert store.contents == []
            assert coordinator.voice_notifications == []
        else:
            assert response.status_code == 201
            assert store.created_ats == [DRAFT_CREATED_AT]
            lease = coordinator.voice_notifications[0][-1]
            assert lease is not None
            assert lease.prepared is prepared
            assert not prepared.cancelled
    finally:
        if commit is not None and not commit.done():
            commit.cancel()
            await asyncio.gather(commit, return_exceptions=True)
        await drafts.close()


@pytest.mark.asyncio
async def test_draft_put_freezes_reply_context_in_each_revision() -> None:
    prepared = ReadyGuardPrepared()
    received: list[AgentReplyContext | None] = []

    async def prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
        *,
        reply_context: AgentReplyContext | None,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        received.append(reply_context)
        return cast(PreparedInitialAnswer, prepared)

    drafts = VoiceDraftCoordinator(
        prepare=cast(PrepareVoiceDraft, prepare),
        ttl_seconds=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    app, _store, _coordinator, _fake_drafts = build_voice_draft_app()
    app.state.voice_draft_coordinator = drafts
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            accepted = await client.put(
                f"{BASE_PATH}/draft",
                json={
                    "revision": 1,
                    "content": "继续",
                    "replyContext": {"kind": "run_result", "id": str(UUID(int=401))},
                },
            )
            await asyncio.sleep(0)
            conflict = await client.put(
                f"{BASE_PATH}/draft",
                json={
                    "revision": 1,
                    "content": "继续",
                    "replyContext": {"kind": "run_result", "id": str(UUID(int=402))},
                },
            )
        assert accepted.status_code == 202
        assert received == [
            AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, UUID(int=401))
        ]
        assert conflict.status_code == 409
    finally:
        await drafts.close()


@pytest.mark.asyncio
async def test_commit_rejects_nonpreceding_promote_revision() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        equal = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 2,
                "content": "最终问题",
            },
        )
        newer = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 3,
                "content": "最终问题",
            },
        )

    assert equal.status_code == 422
    assert newer.status_code == 422
    assert store.contents == []
    assert coordinator.voice_notifications == []
    assert drafts.claims == []


@pytest.mark.asyncio
async def test_commit_persistence_failure_settles_claimed_prepared_answer() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    prepared = FakePrepared()
    drafts.lease = lease_for(prepared)
    store.error = LegacyFishReadOnlyError(str(FISH_ID))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 1,
                "content": "最终问题",
            },
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "legacy_fish_read_only"
    assert prepared.cancelled is True
    assert prepared.settled.is_set()
    assert coordinator.voice_notifications == []


@pytest.mark.asyncio
async def test_duplicate_commit_is_stable_and_different_content_conflicts() -> None:
    app, store, coordinator, _ = build_voice_draft_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first, duplicate = await asyncio.gather(
            client.post(
                f"{BASE_PATH}/commit",
                json={"revision": 1, "content": "最终问题"},
            ),
            client.post(
                f"{BASE_PATH}/commit",
                json={"revision": 1, "content": "最终问题"},
            ),
        )
        conflict = await client.post(
            f"{BASE_PATH}/commit",
            json={"revision": 2, "content": "不同问题"},
        )

    assert first.status_code == duplicate.status_code == 201
    assert first.json()["messageId"] == duplicate.json()["messageId"] == str(TURN_ID)
    assert first.json()["sequence"] == duplicate.json()["sequence"] == 7
    assert store.contents == [(FISH_ID, "最终问题")]
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "voice_turn_conflict"
    assert len(coordinator.voice_notifications) == 2


@pytest.mark.asyncio
async def test_voice_commit_reply_context_shape_is_closed() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        extra = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 1,
                "content": "允许",
                "replyContext": {
                    "kind": "interaction",
                    "id": str(UUID(int=301)),
                    "sourceMessageId": str(UUID(int=302)),
                },
            },
        )
        bad_kind = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 1,
                "content": "允许",
                "replyContext": {"kind": "task", "id": str(UUID(int=301))},
            },
        )

    assert extra.status_code == 422
    assert bad_kind.status_code == 422
    assert drafts.claims == []
    assert store.contents == []
    assert coordinator.voice_notifications == []


@pytest.mark.asyncio
async def test_handled_voice_reply_settles_lease_and_skips_ordinary_answer() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    replies = FakeReplyService(store, handled=True)
    app.state.agent_reply_service = replies
    prepared = FakePrepared()
    drafts.lease = lease_for(prepared)
    context_id = UUID(int=301)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 1,
                "content": "最终问题",
                "replyContext": {"kind": "interaction", "id": str(context_id)},
            },
        )

    assert response.status_code == 201
    assert response.json()["answerExpected"] is False
    assert replies.calls == [
        (
            FISH_ID,
            DEFAULT_SESSION_ID,
            "最终问题",
            AgentReplyContext(AgentMessageSubjectKind.INTERACTION, context_id),
            TURN_ID,
            DRAFT_CREATED_AT,
        )
    ]
    assert prepared.cancelled is True
    assert prepared.settled.is_set()
    assert coordinator.voice_notifications == []
    assert coordinator.notifications == []
    assert drafts.context_fingerprints[0]


@pytest.mark.asyncio
async def test_unrelated_voice_reply_keeps_exact_promotion_and_notifies_once() -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    replies = FakeReplyService(store, handled=False)
    app.state.agent_reply_service = replies
    lease = lease_for()
    drafts.lease = lease
    context_id = UUID(int=301)
    body = {
        "revision": 2,
        "promoteRevision": 1,
        "content": "最终问题",
        "replyContext": {"kind": "run_result", "id": str(context_id)},
    }
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(f"{BASE_PATH}/commit", json=body)
        replay = await client.post(f"{BASE_PATH}/commit", json=body)

    assert first.status_code == replay.status_code == 201
    assert first.json()["answerExpected"] is True
    assert replay.json()["answerExpected"] is True
    assert store.contents == [(FISH_ID, "最终问题")]
    assert len(coordinator.voice_notifications) == 1
    assert coordinator.voice_notifications[0][-1] is lease
    assert len(replies.calls) == 2
    assert drafts.context_fingerprints[0] == drafts.context_fingerprints[1]


@pytest.mark.asyncio
async def test_real_draft_commit_serializes_lease_holder_before_duplicate() -> None:
    prepared = ReadyGuardPrepared()

    async def prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        return cast(PreparedInitialAnswer, prepared)

    drafts = VoiceDraftCoordinator(
        prepare=prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    app, store, coordinator, _fake_drafts = build_voice_draft_app()
    replies = BlockingLeaseReplyService(store)
    app.state.voice_draft_coordinator = drafts
    app.state.agent_reply_service = replies
    key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
    await drafts.upsert(
        key,
        revision=1,
        content="最终问题",
        reply_context=AgentReplyContext(
            AgentMessageSubjectKind.RUN_RESULT, UUID(int=301)
        ),
    )

    async def draft_ready() -> None:
        while True:
            entry = drafts._entries.get(key)
            if entry is not None and entry.ready:
                return
            await asyncio.sleep(0)

    await asyncio.wait_for(draft_ready(), timeout=0.5)
    body = {
        "revision": 2,
        "promoteRevision": 1,
        "content": "最终问题",
        "replyContext": {"kind": "run_result", "id": str(UUID(int=301))},
    }
    transport = httpx.ASGITransport(app=app)
    first_request: asyncio.Task[httpx.Response] | None = None
    duplicate_request: asyncio.Task[httpx.Response] | None = None
    duplicate_entered_before_release = False
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            first_request = asyncio.create_task(
                client.post(f"{BASE_PATH}/commit", json=body)
            )
            await asyncio.wait_for(replies.holder_entered.wait(), timeout=0.5)
            duplicate_request = asyncio.create_task(
                client.post(f"{BASE_PATH}/commit", json=body)
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(replies.duplicate_entered.wait()),
                    timeout=0.05,
                )
                duplicate_entered_before_release = True
            except TimeoutError:
                duplicate_entered_before_release = False
            replies.release_holder.set()
            first, duplicate = await asyncio.gather(
                first_request,
                duplicate_request,
            )
    finally:
        replies.release_holder.set()
        pending = [
            task
            for task in (first_request, duplicate_request)
            if task is not None and not task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await drafts.close()

    assert duplicate_entered_before_release is False
    assert replies.insert_order == ["holder", "duplicate"]
    assert first.status_code == duplicate.status_code == 201
    assert first.json() == duplicate.json()
    assert first.json()["createdAt"] == "2026-07-20T08:29:59Z"
    assert store.contents == [(FISH_ID, "最终问题")]
    assert store.created_ats == [DRAFT_CREATED_AT]
    assert len(coordinator.voice_notifications) == 1
    notified_lease = coordinator.voice_notifications[0][-1]
    assert notified_lease is not None
    assert notified_lease.created_at == DRAFT_CREATED_AT
    assert notified_lease.prepared is prepared


@pytest.mark.asyncio
async def test_same_voice_turn_content_with_different_reply_context_is_conflict() -> (
    None
):
    app, store, coordinator, drafts = build_voice_draft_app()
    replies = FakeReplyService(store, handled=False)
    app.state.agent_reply_service = replies
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 1,
                "content": "thanks",
                "replyContext": {"kind": "run_result", "id": str(UUID(int=301))},
            },
        )
        conflict = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 1,
                "content": "thanks",
                "replyContext": {"kind": "interaction", "id": str(UUID(int=302))},
            },
        )

    assert first.status_code == 201
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "voice_turn_conflict"
    assert len(replies.calls) == 1
    assert len(coordinator.voice_notifications) == 1


@pytest.mark.asyncio
async def test_voice_reply_context_without_service_fails_closed_and_settles_lease() -> (
    None
):
    app, store, coordinator, drafts = build_voice_draft_app()
    prepared = FakePrepared()
    drafts.lease = lease_for(prepared)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 1,
                "content": "最终问题",
                "replyContext": {"kind": "interaction", "id": str(UUID(int=301))},
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "agent_replies_unavailable"
    assert prepared.cancelled is True
    assert prepared.settled.is_set()
    assert store.contents == []
    assert coordinator.voice_notifications == []


@pytest.mark.asyncio
@pytest.mark.parametrize("handled", [False, True])
async def test_voice_commit_uses_prepared_reply_without_reclassification(
    handled: bool,
) -> None:
    app, store, coordinator, drafts = build_voice_draft_app()
    context = AgentReplyContext(AgentMessageSubjectKind.RUN_RESULT, UUID(int=301))
    reply = PreparedAgentReply(
        fish_id=FISH_ID,
        session_id=DEFAULT_SESSION_ID,
        content="最终问题",
        snapshot=AgentReplySubjectSnapshot(
            context=context,
            source_message_id=UUID(int=302),
            source_content="任务结果",
            interaction_kind=None,
            allowed_decisions=(),
            recovered_without_details=False,
            replay=None,
        ),
        disposition=(
            AgentReplyDisposition.RESULT if handled else AgentReplyDisposition.ORDINARY
        ),
        result_action=RunResultReplyAction.CONTINUE if handled else None,
        message_id=TURN_ID,
    )
    commit_calls: list[tuple[PreparedAgentReply, UUID, datetime]] = []

    class PreparedOnlyReplies:
        async def handle(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("commit must not classify a prepared reply again")

        async def commit_prepared_reply(
            self,
            prepared: PreparedAgentReply,
            *,
            message_id: UUID,
            created_at: datetime,
        ) -> AgentReplyResult:
            commit_calls.append((prepared, message_id, created_at))
            message = await store.insert_user_message(
                prepared.fish_id,
                prepared.content,
                session_id=prepared.session_id,
                message_id=message_id,
                created_at=created_at,
            )
            return AgentReplyResult(
                message=message, handled=prepared.handled, inserted=True, replayed=False
            )

    app.state.agent_reply_service = PreparedOnlyReplies()
    prepared = FakePrepared()
    prepared.prepared_reply = reply
    drafts.lease = lease_for(prepared)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 1,
                "content": "最终问题",
                "replyContext": {"kind": "run_result", "id": str(UUID(int=301))},
            },
        )

    assert response.status_code == 201
    assert response.json()["answerExpected"] is not handled
    assert commit_calls == [(reply, TURN_ID, DRAFT_CREATED_AT)]
    assert store.created_ats == [DRAFT_CREATED_AT]
    if handled:
        assert prepared.cancelled
        assert prepared.settled.is_set()
        assert coordinator.voice_notifications == []
    else:
        assert not prepared.cancelled
        assert coordinator.voice_notifications[0][-1].prepared is prepared


@pytest.mark.asyncio
async def test_voice_reply_classification_failure_is_distinct_and_settles_lease() -> (
    None
):
    app, store, coordinator, drafts = build_voice_draft_app()
    replies = FakeReplyService(store, handled=True)
    replies.error = AgentReplyClassificationError("classifier unavailable")
    app.state.agent_reply_service = replies
    prepared = FakePrepared()
    drafts.lease = lease_for(prepared)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{BASE_PATH}/commit",
            json={
                "revision": 2,
                "promoteRevision": 1,
                "content": "最终问题",
                "replyContext": {"kind": "run_result", "id": str(UUID(int=301))},
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "reply_classification_failed"
    assert prepared.cancelled is True
    assert prepared.settled.is_set()
    assert store.contents == []
    assert coordinator.voice_notifications == []


@pytest.mark.asyncio
async def test_real_committed_tombstone_rejects_different_reply_context() -> None:
    async def unused_prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        raise AssertionError("finalize without an active draft must not prepare")

    drafts = VoiceDraftCoordinator(
        prepare=unused_prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
    try:
        assert (
            await drafts.finalize(
                key,
                revision=1,
                promote_revision=None,
                content="same content",
                context_fingerprint="a" * 64,
            )
            is None
        )
        assert (
            await drafts.finalize(
                key,
                revision=1,
                promote_revision=None,
                content="same content",
                context_fingerprint="a" * 64,
            )
            is None
        )
        with pytest.raises(VoiceDraftRevisionConflict, match="context"):
            await drafts.finalize(
                key,
                revision=1,
                promote_revision=None,
                content="same content",
                context_fingerprint="b" * 64,
            )
    finally:
        await drafts.close()


@pytest.mark.asyncio
async def test_real_finalize_rejects_noncanonical_context_fingerprint() -> None:
    async def unused_prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        raise AssertionError("finalize without an active draft must not prepare")

    drafts = VoiceDraftCoordinator(
        prepare=unused_prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
    try:
        for fingerprint in ("x" * 10_000, "A" * 64, "f" * 63):
            with pytest.raises(ValueError, match="fingerprint"):
                await drafts.finalize(
                    key,
                    revision=1,
                    promote_revision=None,
                    content="same content",
                    context_fingerprint=fingerprint,
                )
        assert (
            await drafts.finalize(
                key,
                revision=1,
                promote_revision=None,
                content="same content",
                context_fingerprint="f" * 64,
            )
            is None
        )
    finally:
        await drafts.close()


@pytest.mark.asyncio
async def test_commit_guard_serializes_same_key_and_cleans_last_reference() -> None:
    async def unused_prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        raise AssertionError("commit guard must not prepare")

    drafts = VoiceDraftCoordinator(
        prepare=unused_prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with drafts.commit_guard(key):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        async with drafts.commit_guard(key):
            second_entered.set()

    try:
        first_task = asyncio.create_task(first())
        await asyncio.sleep(0)
        if first_task.done():
            await first_task
        await asyncio.wait_for(first_entered.wait(), timeout=0.2)
        second_task = asyncio.create_task(second())
        for _ in range(3):
            await asyncio.sleep(0)
        assert second_entered.is_set() is False
        release_first.set()
        await asyncio.gather(first_task, second_task)
        assert second_entered.is_set() is True
        assert drafts._commit_guards == {}
    finally:
        await drafts.close()


@pytest.mark.asyncio
async def test_commit_guard_does_not_block_a_different_turn_key() -> None:
    async def unused_prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        raise AssertionError("commit guard must not prepare")

    drafts = VoiceDraftCoordinator(
        prepare=unused_prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    first_key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
    second_key = VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, UUID(int=124))
    release_first = asyncio.Event()
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()

    async def hold_first() -> None:
        async with drafts.commit_guard(first_key):
            first_entered.set()
            await release_first.wait()

    try:
        first_task = asyncio.create_task(hold_first())
        await asyncio.sleep(0)
        if first_task.done():
            await first_task
        await asyncio.wait_for(first_entered.wait(), timeout=0.2)
        async with drafts.commit_guard(second_key):
            second_entered.set()
        assert second_entered.is_set() is True
        release_first.set()
        await first_task
        assert drafts._commit_guards == {}
    finally:
        await drafts.close()


@pytest.mark.asyncio
async def test_commit_guard_fails_closed_after_coordinator_close() -> None:
    async def unused_prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        raise AssertionError("commit guard must not prepare")

    drafts = VoiceDraftCoordinator(
        prepare=unused_prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    await drafts.close()

    with pytest.raises(RuntimeError, match="closed"):
        async with drafts.commit_guard(
            VoiceDraftKey(FISH_ID, DEFAULT_SESSION_ID, TURN_ID)
        ):
            raise AssertionError("closed guard must not enter")


@pytest.mark.asyncio
async def test_voice_commit_fails_closed_after_draft_coordinator_close() -> None:
    async def unused_prepare(
        key: VoiceDraftKey,
        revision: int,
        content: str,
    ) -> PreparedInitialAnswer:
        del key, revision, content
        raise AssertionError("closed coordinator must not prepare")

    drafts = VoiceDraftCoordinator(
        prepare=unused_prepare,
        ttl_seconds=2.0,
        max_concurrency=2,
        now=lambda: DRAFT_CREATED_AT,
    )
    await drafts.close()
    app, store, coordinator, _fake_drafts = build_voice_draft_app()
    app.state.voice_draft_coordinator = drafts
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"{BASE_PATH}/commit",
            json={"revision": 1, "content": "final"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "voice_draft_unavailable"
    assert store.contents == []
    assert coordinator.voice_notifications == []


@pytest.mark.asyncio
async def test_private_events_emit_only_draft_names_and_end_on_cancel() -> None:
    app, _, _, drafts = build_voice_draft_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events_request = asyncio.create_task(
            client.get(f"{BASE_PATH}/events", params={"revision": 1})
        )
        await asyncio.wait_for(drafts.stream_started.wait(), timeout=0.2)
        cancel_response = await client.post(
            f"{BASE_PATH}/cancel",
            json={"revision": 1, "reason": "speech_resumed"},
        )
        events_response = await asyncio.wait_for(events_request, timeout=0.2)

    assert cancel_response.status_code == 204
    assert events_response.status_code == 200
    event_names = [
        line.removeprefix("event: ")
        for line in events_response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert event_names == ["draft.snapshot", "draft.canceled"]
    assert all(not name.startswith("answer.") for name in event_names)
