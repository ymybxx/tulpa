import asyncio
import json
from collections.abc import AsyncIterator, Callable, Collection, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import pytest

from fish_memory.agent.models import (
    AgentEndpoint,
    AgentResolvedTarget,
    AgentTarget,
    AgentTaskCandidate,
    ConnectorTypeSlug,
)
from fish_memory.agent.routing import (
    AgentTaskDecisionKind,
    AgentTaskRouteConflictError,
    AgentTaskRouteService,
    PreparedAgentTaskRoute,
)
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.context_loader import LoadedAnswerContext
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.generation import AnswerGenerationError
from fish_memory.answer.lane_context import build_recall_lane_messages
from fish_memory.answer.output_sink import AnswerOutputSink, PromotableAnswerSink
from fish_memory.answer.parallel_generation import PreparedInitialAnswer
from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.voice_draft_types import VoiceDraftKey, voice_answer_message_id
from fish_memory.answer.voice_drafts import VoiceDraftLease
from fish_memory.home.models import HomeActionStatus, HomeIntentType, HomeRunOutcome
from fish_memory.home.preparation import PreparedHomeRequest
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion
from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    NormalizedStaticProfile,
    SnapshotItemState,
)
from fish_memory.soul import SoulProfile

from .fakes import (
    DEFAULT_SESSION_ID,
    BlockingLatestMessageStore,
    ControlledChatClient,
    FailingChatClient,
    FakeChatClient,
    FakeFishSoulStore,
    FakePerformanceStore,
    FakeStore,
    InvalidAfterVisibleJsonClient,
    LateChunkAfterCancelClient,
    MalformedThenValidJsonClient,
    MultiFishStore,
    MultiSessionStore,
    TimeoutThenValidJsonClient,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = DEFAULT_SESSION_ID
USER_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000201")
NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)


class HomeRequestStore(FakeStore):
    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        history = await super().load_history(
            fish_id,
            session_id,
            before_sequence=before_sequence,
            limit=limit,
        )
        return [replace(history[0], content="打开客厅灯")]


class HomeAcceptingChatClient(FakeChatClient):
    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        if messages[0]["content"].startswith("# H 通道"):
            yield ChatStreamChunk(
                content='{"h":1,"t":"control","m":"happy","a":"我来打开。"}'
            )
            return
        async for chunk in super().stream_json(
            messages,
            enable_search=enable_search,
            force_search=force_search,
        ):
            yield chunk


class TaskRequestStore(FakeStore):
    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        history = await super().load_history(
            fish_id,
            session_id,
            before_sequence=before_sequence,
            limit=limit,
        )
        return [replace(history[0], content="帮我调查这个故障")]


class TaskAcceptingChatClient(FakeChatClient):
    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        if messages[0]["content"].startswith("# T 通道"):
            yield ChatStreamChunk(content=('{"t":1,"m":"happy","a":"我来处理。"}'))
            return
        async for chunk in super().stream_json(
            messages,
            enable_search=enable_search,
            force_search=force_search,
        ):
            yield chunk


def fixed_agent_target() -> AgentResolvedTarget:
    endpoint = AgentEndpoint(
        id=UUID(int=901),
        connector_type=ConnectorTypeSlug("hermes"),
        base_url="http://localhost:9000",
        created_at=NOW,
        updated_at=NOW,
    )
    target = AgentTarget(
        id=UUID(int=900),
        endpoint_id=endpoint.id,
        external_key="default",
        route_prefix="",
        credential_ref="hermes:test",
        connector_config={},
        display_name="Hermes",
        enabled=True,
        created_at=NOW,
        updated_at=NOW,
    )
    return AgentResolvedTarget(target=target, endpoint=endpoint)


class FixedAgentTargetStore:
    async def resolve_for_fish(
        self,
        fish_id: UUID,
        default_target_id: UUID | None,
    ) -> AgentResolvedTarget:
        del fish_id, default_target_id
        return fixed_agent_target()


class EmptyAgentCandidateStore:
    async def list_task_candidates(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        recent_terminal_limit: int,
        limit: int,
        supported_connector_types: Collection[str] | None = None,
    ) -> tuple[AgentTaskCandidate, ...]:
        del (
            fish_id,
            session_id,
            recent_terminal_limit,
            limit,
            supported_connector_types,
        )
        return ()

    async def has_recent_agent_reply(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> bool:
        del fish_id, session_id, before_sequence, limit
        return False


class FailingTaskRouteService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def prepare_selected_route(self, **kwargs: object) -> PreparedAgentTaskRoute:
        target = cast(AgentResolvedTarget, kwargs["resolved_target"])
        return PreparedAgentTaskRoute(
            fish_id=cast(UUID, kwargs["fish_id"]),
            target_id=target.target.id,
            kind=cast(AgentTaskDecisionKind, kwargs["kind"]),
            content=cast(str, kwargs["content"]),
        )

    async def commit_prepared_route(
        self, prepared: PreparedAgentTaskRoute, **kwargs: object
    ) -> None:
        self.calls.append({"prepared": prepared, **kwargs})
        raise AgentTaskRouteConflictError("commit CAS lost")

    async def commit_selected_route(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))
        raise AgentTaskRouteConflictError("commit CAS lost")


class FakeHomeActionService:
    def __init__(self) -> None:
        self.notifications: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.preparations: list[tuple[str, HomeIntentType]] = []

    async def prepare(
        self,
        content: str,
        *,
        intent: HomeIntentType,
        history: Sequence[ChatMessage],
        tracker: MessagePerformanceTracker,
    ) -> PreparedHomeRequest:
        del history, tracker
        self.preparations.append((content, intent))
        return PreparedHomeRequest(intent=intent)

    def start_prepared(
        self,
        fish_id: UUID,
        session_id: UUID,
        message_id: UUID,
        prepared: PreparedHomeRequest,
        *,
        origin_ns: int | None = None,
    ) -> "FakeHomeRunHandle":
        self.notifications.append(
            (
                (fish_id, session_id, message_id, prepared),
                {"origin_ns": origin_ns},
            )
        )
        return FakeHomeRunHandle()

    def start(self, *args: object, **kwargs: object):
        self.notifications.append((args, kwargs))
        return FakeHomeRunHandle()


class FakeHomeRunHandle:
    async def result(self) -> HomeRunOutcome:
        return HomeRunOutcome(
            intent=HomeIntentType.CONTROL,
            status=HomeActionStatus.SUCCEEDED,
        )


@pytest.mark.asyncio
async def test_active_generation_signal_ignores_persisted_runtime_identity() -> None:
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(['{"m":"neutral","a":"你好。","r":0,"e":0}']),
        system_prompt="system",
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    runtime.latest_user_message = Message(
        id=USER_MESSAGE_ID,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=1,
        role=Role.USER,
        content="你好",
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )
    runtime.answer_message_id = UUID(int=302)

    assert coordinator.has_active_generation() is False

    release = asyncio.Event()
    runtime.llm_task = asyncio.create_task(release.wait())
    try:
        assert coordinator.has_active_generation() is True
        release.set()
        await runtime.llm_task
        assert coordinator.has_active_generation() is False
    finally:
        release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_ask_submission_callback_fires_after_context_for_text_and_voice() -> None:
    calls: list[tuple[UUID, UUID]] = []
    message = Message(
        id=UUID(int=1),
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=1,
        role=Role.USER,
        content="你好",
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )
    text = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(['{"m":"neutral","a":"你好。","r":0,"e":0}']),
        system_prompt="system",
        on_ask_submitted=lambda fish_id, session_id: calls.append(
            (fish_id, session_id)
        ),
    )
    try:
        text.notify_user_message(
            FISH_ID,
            SESSION_ID,
            1,
            message.id,
            MessagePerformanceTracker(),
            source_message=message,
        )
        assert calls == []
        await wait_until(lambda: len(calls) == 1)
    finally:
        await text.close()

    voice = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(['{"m":"neutral","a":"你好。","r":0,"e":0}']),
        system_prompt="system",
        on_ask_submitted=lambda fish_id, session_id: calls.append(
            (fish_id, session_id)
        ),
    )
    try:
        voice.notify_voice_turn(
            FISH_ID,
            SESSION_ID,
            1,
            message.id,
            voice_answer_message_id(message.id),
            MessagePerformanceTracker(),
            source_message=message,
            draft_lease=None,
        )
        await wait_until(lambda: len(calls) == 2)
        assert calls == [(FISH_ID, SESSION_ID), (FISH_ID, SESSION_ID)]
    finally:
        await voice.close()


@pytest.mark.asyncio
async def test_ask_submission_callback_failure_does_not_block_generation() -> None:
    store = FakeStore(FISH_ID)

    def fail(_fish_id: UUID, _session_id: UUID) -> None:
        raise RuntimeError("refresh signal unavailable")

    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(['{"m":"neutral","a":"你好。","r":0,"e":0}']),
        system_prompt="system",
        on_ask_submitted=fail,
    )
    message = Message(
        id=UUID(int=1),
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=1,
        role=Role.USER,
        content="你好",
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )
    try:
        coordinator.notify_user_message(
            FISH_ID,
            SESSION_ID,
            1,
            message.id,
            MessagePerformanceTracker(),
            source_message=message,
        )
        await wait_until(lambda: store.insert_count == 1)
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_home_router_starts_only_after_h_lane_accepts() -> None:
    store = HomeRequestStore(FISH_ID)
    home_service = FakeHomeActionService()
    coordinator = AnswerCoordinator(
        store,
        HomeAcceptingChatClient(["R 不应发布"]),
        home_action_service=home_service,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert runtime.content == "我来打开。"
    assert runtime.pet_emotion is PetEmotion.HAPPY
    assert len(home_service.notifications) == 1
    assert home_service.preparations == [("打开客厅灯", HomeIntentType.CONTROL)]
    args, kwargs = home_service.notifications[0]
    assert args == (
        FISH_ID,
        SESSION_ID,
        USER_MESSAGE_ID,
        PreparedHomeRequest(intent=HomeIntentType.CONTROL),
    )
    assert isinstance(kwargs["origin_ns"], int)


@pytest.mark.asyncio
async def test_renders_fish_profile_after_global_prompt() -> None:
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["好的"])
    soul_store = FakeFishSoulStore(
        SoulProfile.from_values("健身教练", "直接、简洁", "小麦")
    )
    coordinator = AnswerCoordinator(
        store,
        llm,
        fish_store=soul_store,
        system_prompt="global prompt",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert soul_store.calls == [FISH_ID]
    assert llm.calls[0][0]["role"] == "system"
    prompt = llm.calls[0][0]["content"]
    assert prompt.startswith("# R 通道输出协议")
    assert "global prompt" in prompt
    assert prompt.index("# R 通道输出协议") < prompt.index("global prompt")
    assert "身份与关系：健身教练" in prompt
    assert "性格与表达：直接、简洁" in prompt
    assert "名字：小麦" in prompt
    assert [message["role"] for message in llm.calls[0]].count("system") == 1
    assert llm.calls[0][-1]["role"] == "user"
    assert llm.calls[0][-1]["content"].startswith("# 当前用户消息")


@pytest.mark.asyncio
async def test_answer_does_not_enable_web_search_by_default() -> None:
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["好的"])
    coordinator = AnswerCoordinator(
        store,
        llm,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert llm.search_flags == [False]
    assert llm.force_search_flags == [False]
    assert {lane for lane, _messages in llm.lane_calls} == {"r"}
    assert '"w"' not in llm.calls[0][0]["content"]
    assert "联网判断" not in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_enabled_web_search_uses_new_contract_without_searching_initially() -> (
    None
):
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["好的"], search_aware=True)
    coordinator = AnswerCoordinator(
        store,
        llm,
        answer_enable_search=True,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert llm.search_flags == [False]
    assert llm.force_search_flags == [False]
    assert {lane for lane, _messages in llm.lane_calls} == {"r", "w"}
    web_messages = next(value for lane, value in llm.lane_calls if lane == "w")
    assert web_messages[0]["content"].startswith("# W 通道输出协议")
    assert llm.calls[0][-1]["role"] == "user"
    assert '"w"' not in llm.calls[0][-1]["content"]


@pytest.mark.asyncio
async def test_uses_configured_answer_history_message_limit() -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["好的"]),
        answer_history_message_limit=50,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert store.history_limits == [50]


class FakeSnapshotReader:
    def __init__(
        self,
        snapshot: NormalizedDynamicSnapshot | None,
        static_profile: NormalizedStaticProfile | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.static_profile = static_profile
        self.dynamic_calls: list[tuple[UUID, UUID]] = []
        self.static_calls: list[UUID] = []
        self.dynamic_failure: Exception | None = None
        self.static_failure: Exception | None = None

    async def load_normalized_dynamic_snapshot(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> NormalizedDynamicSnapshot | None:
        self.dynamic_calls.append((fish_id, session_id))
        if self.dynamic_failure is not None:
            raise self.dynamic_failure
        return self.snapshot

    async def load_normalized_static_profile(
        self,
        fish_id: UUID,
    ) -> NormalizedStaticProfile | None:
        self.static_calls.append(fish_id)
        if self.static_failure is not None:
            raise self.static_failure
        return self.static_profile


class BlockingContextStore(FakeStore):
    def __init__(self, release: asyncio.Event) -> None:
        super().__init__(FISH_ID)
        self.release = release
        self.history_started = asyncio.Event()

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        self.history_started.set()
        await self.release.wait()
        return await super().load_history(
            fish_id,
            session_id,
            before_sequence=before_sequence,
            limit=limit,
        )


class BlockingSnapshotReader(FakeSnapshotReader):
    def __init__(self, release: asyncio.Event) -> None:
        super().__init__(dynamic_snapshot(), static_profile())
        self.release = release
        self.static_started = asyncio.Event()
        self.dynamic_started = asyncio.Event()

    async def load_normalized_static_profile(
        self,
        fish_id: UUID,
    ) -> NormalizedStaticProfile | None:
        self.static_started.set()
        await self.release.wait()
        return await super().load_normalized_static_profile(fish_id)

    async def load_normalized_dynamic_snapshot(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> NormalizedDynamicSnapshot | None:
        self.dynamic_started.set()
        await self.release.wait()
        return await super().load_normalized_dynamic_snapshot(fish_id, session_id)


class BlockingSoulStore(FakeFishSoulStore):
    def __init__(self, release: asyncio.Event) -> None:
        super().__init__(SoulProfile.from_values("助手", "简洁", "小鱼"))
        self.release = release
        self.started = asyncio.Event()

    async def get_profile(self, fish_id: UUID) -> SoulProfile | None:
        self.started.set()
        await self.release.wait()
        return await super().get_profile(fish_id)


def dynamic_snapshot() -> NormalizedDynamicSnapshot:
    return NormalizedDynamicSnapshot(
        items=(
            SnapshotItemState(
                id=301,
                summary="该用户正在开发快照",
                support_fact_ids=(UUID(int=301),),
                lineage_fact_ids=(UUID(int=301), UUID(int=302)),
                last_supported_at=NOW,
                item_version=2,
            ),
        ),
        through_fact_sequence=3,
        version=1,
    )


def static_profile() -> NormalizedStaticProfile:
    return NormalizedStaticProfile(
        items=(
            SnapshotItemState(
                id=401,
                summary="用户喜欢篮球",
                support_fact_ids=(UUID(int=401),),
                lineage_fact_ids=(UUID(int=401), UUID(int=402)),
                last_supported_at=NOW,
                item_version=3,
            ),
        ),
        through_fact_sequence=3,
        version=1,
    )


@pytest.mark.asyncio
async def test_loads_independent_answer_context_concurrently() -> None:
    release = asyncio.Event()
    store = BlockingContextStore(release)
    snapshots = BlockingSnapshotReader(release)
    soul = BlockingSoulStore(release)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["完成"]),
        snapshot_store=snapshots,
        fish_store=soul,
        system_prompt="system",
    )
    handle_task = asyncio.create_task(
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
        )
    )
    all_started = False
    try:
        await asyncio.wait_for(
            asyncio.gather(
                store.history_started.wait(),
                snapshots.static_started.wait(),
                snapshots.dynamic_started.wait(),
                soul.started.wait(),
            ),
            timeout=0.2,
        )
        all_started = True
    except TimeoutError:
        pass
    finally:
        release.set()
        await handle_task
        runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
        assert runtime.llm_task is not None
        await runtime.llm_task
        await coordinator.close()

    assert all_started


async def wait_for_content(runtime: FishRuntime, expected: str) -> None:
    async with runtime.condition:
        await asyncio.wait_for(
            runtime.condition.wait_for(lambda: runtime.content == expected),
            timeout=1,
        )


def tracker_with_ask_step() -> MessagePerformanceTracker:
    tracker = MessagePerformanceTracker()
    now_ns = tracker.now_ns()
    tracker.record(
        "ask.persist_message",
        started_ns=now_ns,
        finished_ns=now_ns,
    )
    return tracker


@pytest.mark.asyncio
async def test_task_commit_failure_uses_failed_lifecycle_without_ack() -> None:
    store = TaskRequestStore(FISH_ID)
    route_service = FailingTaskRouteService()
    coordinator = AnswerCoordinator(
        store,
        TaskAcceptingChatClient(["普通回答不应回退"]),
        agent_target_store=FixedAgentTargetStore(),
        agent_task_store=EmptyAgentCandidateStore(),
        agent_task_route_service=cast(AgentTaskRouteService, route_service),
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert len(route_service.calls) == 1
    assert runtime.content == ""
    assert runtime.flush_offset == 0
    assert runtime.pet_emotion is PetEmotion.NEUTRAL
    assert runtime.status is MessageStatus.FAILED
    assert store.finish_calls[-1]["requested_status"] is MessageStatus.FAILED
    await coordinator.close()


@pytest.mark.asyncio
async def test_delete_fish_marks_store_and_removes_idle_runtime() -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient([]),
        system_prompt="system",
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)

    deleted = await coordinator.delete_fish(FISH_ID)

    assert deleted is True
    assert store.soft_delete_calls == [FISH_ID]
    assert coordinator.peek_runtime(FISH_ID, SESSION_ID) is None
    assert runtime.llm_task is None


@pytest.mark.asyncio
async def test_delete_fish_cancels_generation_without_persisting_answer() -> None:
    store = FakeStore(FISH_ID)
    llm = ControlledChatClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    await llm.wait_for_streams(1)
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    task = runtime.llm_task
    assert task is not None

    deleted = await coordinator.delete_fish(FISH_ID)

    assert deleted is True
    assert task.done()
    assert store.finish_calls == []
    assert store.soft_delete_calls == [FISH_ID]
    assert coordinator.peek_runtime(FISH_ID, SESSION_ID) is None


@pytest.mark.asyncio
async def test_delete_fish_cancels_queued_message_task() -> None:
    store = BlockingLatestMessageStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient([]),
        system_prompt="system",
    )
    coordinator.notify_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    await store.latest_lookup_started.wait()

    deleted = await coordinator.delete_fish(FISH_ID)

    assert deleted is True
    assert store.latest_lookup_cancelled is True
    assert store.soft_delete_calls == [FISH_ID]


@pytest.mark.asyncio
async def test_notified_user_message_is_published_before_answer_generation() -> None:
    store = BlockingLatestMessageStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient([]),
        system_prompt="system",
    )
    source = Message(
        id=USER_MESSAGE_ID,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=1,
        role=Role.USER,
        content="来自语音伴侣",
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )

    coordinator.notify_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
        source_message=source,
    )
    await store.latest_lookup_started.wait()

    assert coordinator.get_runtime(FISH_ID, SESSION_ID).latest_user_message == source
    await coordinator.delete_fish(FISH_ID)


@pytest.mark.asyncio
async def test_records_complete_pipeline_and_llm_payloads() -> None:
    store = FakeStore(FISH_ID)
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["你", "好🐟"]),
        performance_store=performance_store,
        system_prompt="system",
    )
    tracker = tracker_with_ask_step()

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker,
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert len(performance_store.writes) == 1
    message_id, steps = performance_store.writes[0]
    assert message_id == USER_MESSAGE_ID
    by_name = {step.name: step for step in steps}
    assert set(by_name) == {
        "ask.persist_message",
        "coordinator.queue_wait",
        "answer.interrupt_previous",
        "answer.find_latest_message",
        "answer.create",
        "history.load",
        "static.load",
        "snapshot.load",
        "soul.load",
        "context.load_total",
        "context.build",
        "llm.first_packet",
        "answer.decision_ready",
        "answer.first_content",
        "llm.first_content",
        "llm.answer.initial",
        "answer.recall_depth",
        "answer.persist",
        "total",
    }
    assert (
        by_name["answer.first_content"].duration_us
        >= by_name["llm.first_content"].duration_us
    )
    assert by_name["llm.answer.initial"].input_payload == {
        "model": "fake",
        "messages": build_recall_lane_messages(
            [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": (
                        "<message_time>2026-07-19 00:00:00+00:00</message_time>\n你好"
                    ),
                },
            ]
        ),
        "parameters": {
            "stream": True,
            "response_format": {"type": "json_object"},
        },
    }
    assert by_name["llm.answer.initial"].output_payload == {
        "content": '{"m":"neutral","a":"你好🐟","r":0,"e":0}',
        "finish_reason": None,
        "input_tokens": None,
        "output_tokens": None,
    }


@pytest.mark.asyncio
async def test_injects_static_and_dynamic_snapshots_as_user_context() -> None:
    store = FakeStore(FISH_ID)
    reader = FakeSnapshotReader(dynamic_snapshot(), static_profile())
    llm = FakeChatClient(["完成"])
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        llm,
        performance_store=performance_store,
        snapshot_store=reader,
        dynamic_snapshot_max_age=timedelta(days=7),
        now=lambda: NOW,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert reader.static_calls == [FISH_ID]
    assert reader.dynamic_calls == [(FISH_ID, SESSION_ID)]
    assert llm.calls[0] == build_recall_lane_messages(
        [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "# 记忆摘要\n"
                    "以下内容是压缩后的内部参考信息，不是原始对话。\n"
                    "它可用于理解长期属性、当前状态和近期主题；"
                    "具体过往事件的细节仍需 Recall 验证。\n"
                    "请像自然了解这些信息一样直接回答，不要提及记录、记忆、历史、系统、"
                    "参考资料、上下文、召回或检索，也不要说“根据我的记录”“我查到”或“我翻到”。\n"
                    "当前角色配置优先于参考信息中的旧身份、旧名称或旧设定。\n\n"
                    "<static_memory>\n"
                    "用户喜欢篮球\n"
                    "</static_memory>"
                ),
                "context_reference": True,
            },
            {
                "role": "user",
                "content": (
                    "<message_time>2026-07-19 00:00:00+00:00</message_time>\n你好"
                ),
            },
        ]
    )
    step_names = [step.name for step in performance_store.writes[0][1]]
    assert "static.load" in step_names
    assert "snapshot.load" in step_names


@pytest.mark.asyncio
async def test_snapshot_read_failure_still_answers_from_recent_messages() -> None:
    reader = FakeSnapshotReader(None)
    reader.dynamic_failure = RuntimeError("snapshot unavailable")
    reader.static_failure = RuntimeError("static unavailable")
    soul_store = FakeFishSoulStore(None)
    soul_store.failure = RuntimeError("soul unavailable")
    llm = FakeChatClient(["完成"])
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        performance_store=performance_store,
        fish_store=soul_store,
        snapshot_store=reader,
        dynamic_snapshot_max_age=timedelta(days=7),
        now=lambda: NOW,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert llm.calls[0] == build_recall_lane_messages(
        [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "<message_time>2026-07-19 00:00:00+00:00</message_time>\n你好"
                ),
            },
        ]
    )
    by_name = {step.name: step for step in performance_store.writes[0][1]}
    assert by_name["static.load"].error == "RuntimeError"
    assert by_name["snapshot.load"].error == "RuntimeError"
    assert by_name["soul.load"].error == "RuntimeError"
    assert by_name["context.load_total"].error is None


@pytest.mark.asyncio
async def test_performance_write_does_not_block_completed_answer() -> None:
    performance_store = FakePerformanceStore(block=True)
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(["完成"]),
        performance_store=performance_store,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert runtime.status is MessageStatus.COMPLETED
    await asyncio.wait_for(performance_store.started.wait(), timeout=1)
    assert performance_store.writes == []

    performance_store.release.set()
    await coordinator.close()
    assert len(performance_store.writes) == 1


@pytest.mark.asyncio
async def test_first_content_timing_excludes_runtime_condition_wait() -> None:
    performance_store = FakePerformanceStore()
    llm = ControlledChatClient()
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        performance_store=performance_store,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    await llm.wait_for_streams(1)
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    async with runtime.condition:
        await llm.emit(0, "首字")
        await asyncio.sleep(0.1)
    await llm.finish(0)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    _, steps = performance_store.writes[0]
    first_content = next(step for step in steps if step.name == "llm.first_content")
    assert first_content.duration_us < 50_000


@pytest.mark.asyncio
async def test_provider_failure_records_partial_llm_output_and_error() -> None:
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FailingChatClient(),
        performance_store=performance_store,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    _, steps = performance_store.writes[0]
    llm_step = next(step for step in steps if step.name == "llm.answer.initial")
    assert llm_step.error == "RuntimeError"
    assert llm_step.output_payload == {
        "content": '{"m":"neutral","a":"部分',
        "finish_reason": None,
        "input_tokens": None,
        "output_tokens": None,
    }


@pytest.mark.asyncio
async def test_generates_one_completed_answer() -> None:
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["你", "好🐟"])
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert store.insert_count == 1
    assert len(llm.calls) == 1
    assert llm.calls[0] == build_recall_lane_messages(
        [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "<message_time>2026-07-19 00:00:00+00:00</message_time>\n你好"
                ),
            },
        ]
    )
    assert runtime.content == "你好🐟"
    assert runtime.status is MessageStatus.COMPLETED
    assert runtime.source_message_id is None
    assert runtime.performance_tracker is None
    assert store.finish_calls == [
        {
            "message_id": UUID(int=101),
            "fish_id": FISH_ID,
            "content": "你好🐟",
            "requested_status": MessageStatus.COMPLETED,
        }
    ]


@pytest.mark.asyncio
async def test_concurrent_duplicate_notifications_start_one_answer() -> None:
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["你好"])
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        llm,
        performance_store=performance_store,
        system_prompt="system",
    )

    await asyncio.gather(
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
        ),
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
        ),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert store.insert_count == 1
    assert len(llm.calls) == 1
    await coordinator.close()
    assert len(performance_store.writes) == 2
    assert all(
        message_id == USER_MESSAGE_ID for message_id, _ in performance_store.writes
    )
    assert any(
        {step.name for step in steps}
        == {
            "ask.persist_message",
            "coordinator.queue_wait",
            "total",
        }
        for _, steps in performance_store.writes
    )


class FakeCommittedPrepared:
    def __init__(
        self,
        *,
        output_tokens: int | None = None,
        output_token_error: Exception | None = None,
    ) -> None:
        self.sink = PromotableAnswerSink()
        self.input_context = LoadedAnswerContext(
            messages=[
                {"role": "system", "content": "frozen draft system"},
                {"role": "user", "content": "你好"},
            ],
            recall_conversation=({"role": "user", "content": "你好"},),
        )
        self.continue_calls: list[dict[str, object]] = []
        self.release_calls: list[dict[str, object]] = []
        self.cancelled = False
        self.settled = asyncio.Event()
        self.output_tokens = output_tokens
        self.output_token_error = output_token_error

    async def continue_after_commit(
        self,
        *,
        start_home_action: object = None,
        answer_message_id: UUID | None = None,
        runtime: FishRuntime | None = None,
    ) -> None:
        self.continue_calls.append(
            {
                "start_home_action": start_home_action,
                "answer_message_id": answer_message_id,
                "runtime": runtime,
            }
        )

    async def release_to_public(
        self,
        public_sink: object,
        *,
        source_message_id: UUID | None,
        answer_message_id: UUID,
        runtime: FishRuntime,
        start_home_action: object = None,
        session_id: UUID | None = None,
    ) -> None:
        self.release_calls.append(
            {
                "public_sink": public_sink,
                "source_message_id": source_message_id,
                "answer_message_id": answer_message_id,
                "runtime": runtime,
                "start_home_action": start_home_action,
                "session_id": session_id,
            }
        )
        await self.sink.attach(cast(AnswerOutputSink, public_sink))

    def cancel(self) -> asyncio.Task[None]:
        self.cancelled = True

        async def settle() -> None:
            self.settled.set()

        return asyncio.create_task(settle())

    def output_token_usage(self) -> int | None:
        if self.output_token_error is not None:
            raise self.output_token_error
        return self.output_tokens


class BlockingCommittedPrepared(FakeCommittedPrepared):
    def __init__(self, *, output_tokens: int | None = None) -> None:
        super().__init__(output_tokens=output_tokens)
        self.cancel_started = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self._cleanup_task: asyncio.Task[None] | None = None

    def cancel(self) -> asyncio.Task[None]:
        self.cancelled = True
        self.cancel_started.set()
        if self._cleanup_task is None:

            async def settle() -> None:
                await self.cancel_release.wait()
                self.settled.set()

            self._cleanup_task = asyncio.create_task(settle())
        return self._cleanup_task


class BlockingReleasePrepared(FakeCommittedPrepared):
    def __init__(self) -> None:
        super().__init__()
        self.release_entered = asyncio.Event()
        self.release_gate = asyncio.Event()

    async def release_to_public(
        self,
        public_sink: object,
        *,
        source_message_id: UUID | None,
        answer_message_id: UUID,
        runtime: FishRuntime,
        start_home_action: object = None,
        session_id: UUID | None = None,
    ) -> None:
        self.release_calls.append(
            {
                "public_sink": public_sink,
                "source_message_id": source_message_id,
                "answer_message_id": answer_message_id,
                "runtime": runtime,
                "start_home_action": start_home_action,
                "session_id": session_id,
            }
        )
        self.release_entered.set()
        await self.release_gate.wait()
        await self.sink.attach(cast(AnswerOutputSink, public_sink))


def voice_source_message() -> Message:
    return Message(
        id=USER_MESSAGE_ID,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=1,
        role=Role.USER,
        content="你好",
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )


def draft_lease(prepared: FakeCommittedPrepared) -> VoiceDraftLease:
    return VoiceDraftLease(
        key=VoiceDraftKey(FISH_ID, SESSION_ID, USER_MESSAGE_ID),
        revision=1,
        candidate_hash="sha256:candidate",
        input_hash="matching-input",
        created_at=NOW,
        prepared=cast(PreparedInitialAnswer, prepared),
    )


async def wait_for_runtime_task(coordinator: AnswerCoordinator) -> asyncio.Task[None]:
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    for _ in range(100):
        if runtime.llm_task is not None:
            return runtime.llm_task
        await asyncio.sleep(0)
    raise AssertionError("voice notification did not schedule generation")


@pytest.mark.asyncio
@pytest.mark.parametrize("activity", ["", "当前活动：主持直播"])
async def test_voice_turn_promotes_frozen_context_without_reload_or_hashing(
    monkeypatch: pytest.MonkeyPatch,
    activity: str,
) -> None:
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["普通路径不应启动"])
    coordinator = AnswerCoordinator(
        store,
        llm,
        system_prompt="system",
        activity_context_provider=lambda fish, session: activity,
    )
    store.history_error = AssertionError("promotion must not reload history")
    prepared = FakeCommittedPrepared()
    prepared.input_context = replace(prepared.input_context, activity_context=activity)
    await prepared.sink.append("预生成回答。")
    await prepared.sink.flush()
    hash_calls: list[tuple[list[ChatMessage], bool]] = []

    def committed_hash(
        messages: list[ChatMessage],
        *,
        enable_home: bool,
        **kwargs: object,
    ) -> str:
        del kwargs
        hash_calls.append((messages, enable_home))
        return "matching-input"

    monkeypatch.setattr(
        coordinator._generation,
        "initial_input_hash",
        committed_hash,
        raising=False,
    )
    answer_id = voice_answer_message_id(USER_MESSAGE_ID)

    tracker = tracker_with_ask_step()
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        answer_id,
        tracker,
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    task = await wait_for_runtime_task(coordinator)
    await task
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)

    assert hash_calls == []
    assert store.answer_message_ids == [answer_id]
    assert llm.calls == []
    assert runtime.content == "预生成回答。"
    assert runtime.flush_offset == len("预生成回答。")
    assert runtime.status is MessageStatus.COMPLETED
    assert prepared.continue_calls == []
    assert len(prepared.release_calls) == 1
    assert prepared.release_calls[0]["source_message_id"] == USER_MESSAGE_ID
    assert prepared.release_calls[0]["answer_message_id"] == answer_id
    assert prepared.release_calls[0]["runtime"] is runtime
    assert prepared.release_calls[0]["session_id"] == SESSION_ID
    assert prepared.cancelled is False
    promoted = next(
        step for step in tracker.snapshot() if step.name == "voice_draft.promoted"
    )
    assert isinstance(promoted.output_payload, dict)
    assert promoted.output_payload == {"input_hash": "matching-input"}
    await coordinator.close()


@pytest.mark.asyncio
async def test_stopped_activity_does_not_promote_a_draft_from_that_activity():
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["新的回答"])
    coordinator = AnswerCoordinator(
        store,
        llm,
        system_prompt="system",
        activity_context_provider=lambda fish, session: "",
    )
    prepared = FakeCommittedPrepared()
    prepared.input_context = replace(
        prepared.input_context, activity_context="已经停止的直播活动"
    )
    try:
        coordinator.notify_voice_turn(
            FISH_ID,
            SESSION_ID,
            1,
            USER_MESSAGE_ID,
            voice_answer_message_id(USER_MESSAGE_ID),
            tracker_with_ask_step(),
            source_message=voice_source_message(),
            draft_lease=draft_lease(prepared),
        )
        task = await wait_for_runtime_task(coordinator)
        await task
        assert prepared.cancelled
        assert not prepared.release_calls
        assert len(llm.calls) == 1
        assert "已经停止的直播活动" not in json.dumps(llm.calls, ensure_ascii=False)
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_changed_tts_tag_capability_does_not_promote_old_voice_draft():
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["新的回答"])

    async def current_speech_tags(fish_id):
        return False

    coordinator = AnswerCoordinator(
        store, llm, system_prompt="system", speech_tags_provider=current_speech_tags
    )
    prepared = FakeCommittedPrepared()
    prepared.input_context = replace(prepared.input_context, speech_tags=True)
    try:
        coordinator.notify_voice_turn(
            FISH_ID,
            SESSION_ID,
            1,
            USER_MESSAGE_ID,
            voice_answer_message_id(USER_MESSAGE_ID),
            tracker_with_ask_step(),
            source_message=voice_source_message(),
            draft_lease=draft_lease(prepared),
        )
        task = await wait_for_runtime_task(coordinator)
        await task
        assert prepared.cancelled
        assert not prepared.release_calls
        assert len(llm.calls) == 1
        assert "语音表达标签" not in json.dumps(llm.calls, ensure_ascii=False)
        assert not coordinator.get_runtime(FISH_ID, SESSION_ID).speech_tags
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_promoted_voice_uses_shared_release_before_any_public_task_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["普通路径不应启动"]),
        system_prompt="system",
    )
    prepared = BlockingReleasePrepared()
    await prepared.sink.set_emotion(PetEmotion.HAPPY)
    await prepared.sink.append("预生成 T 确认。")
    await prepared.sink.flush()
    monkeypatch.setattr(
        coordinator._generation,
        "initial_input_hash",
        lambda messages, *, enable_home, **kwargs: "matching-input",
        raising=False,
    )
    answer_id = voice_answer_message_id(USER_MESSAGE_ID)

    try:
        coordinator.notify_voice_turn(
            FISH_ID,
            SESSION_ID,
            1,
            USER_MESSAGE_ID,
            answer_id,
            tracker_with_ask_step(),
            source_message=voice_source_message(),
            draft_lease=draft_lease(prepared),
        )
        task = await wait_for_runtime_task(coordinator)
        await asyncio.wait_for(prepared.release_entered.wait(), timeout=1)
        value = coordinator.get_runtime(FISH_ID, SESSION_ID)

        assert value.status is MessageStatus.GENERATING
        assert value.content == ""
        assert value.flush_offset == 0
        assert value.pet_emotion is PetEmotion.NEUTRAL
        assert value.pet_metadata_revision == 0

        prepared.release_gate.set()
        await task

        assert value.content == "预生成 T 确认。"
        assert value.flush_offset == len(value.content)
        assert value.pet_emotion is PetEmotion.HAPPY
        assert prepared.continue_calls == []
    finally:
        prepared.release_gate.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_voice_turn_context_drift_does_not_restart_a_valid_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["不得重启"])
    coordinator = AnswerCoordinator(
        store,
        llm,
        system_prompt="system",
    )
    prepared = FakeCommittedPrepared(output_tokens=7)
    await prepared.sink.append("暂停时已生成的回答")
    store.history_error = RuntimeError("later context is unavailable")
    monkeypatch.setattr(
        coordinator._generation,
        "initial_input_hash",
        lambda messages, *, enable_home, **kwargs: "different-input",
        raising=False,
    )

    tracker = tracker_with_ask_step()
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker,
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    task = await wait_for_runtime_task(coordinator)
    await task
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)

    assert prepared.cancelled is False
    assert not prepared.settled.is_set()
    assert prepared.continue_calls == []
    assert llm.calls == []
    assert runtime.content == "暂停时已生成的回答"
    assert not any(step.name == "voice_draft.fallback" for step in tracker.snapshot())
    await coordinator.close()


@pytest.mark.asyncio
async def test_failed_prepared_release_settles_before_lifecycle_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(FISH_ID)
    prepared = BlockingCommittedPrepared(output_tokens=7)
    llm = FakeChatClient(["不得重启"])
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    async def fail_release(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AnswerGenerationError

    monkeypatch.setattr(prepared, "release_to_public", fail_release)
    tracker = tracker_with_ask_step()
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker,
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    task = await wait_for_runtime_task(coordinator)
    await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)
    assert not prepared.settled.is_set()
    prepared.cancel_release.set()

    await task

    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert task.exception() is None
    assert runtime.status is MessageStatus.FAILED
    assert llm.calls == []
    assert prepared.settled.is_set()
    await coordinator.close()


@pytest.mark.asyncio
async def test_failed_answer_keeps_which_service_let_it_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_memory import failures

    store = FakeStore(FISH_ID)
    prepared = BlockingCommittedPrepared(output_tokens=7)
    coordinator = AnswerCoordinator(
        store, FakeChatClient(["不得重启"]), system_prompt="system"
    )

    async def fail_release(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise failures.noted(TimeoutError(), "jev")

    monkeypatch.setattr(prepared, "release_to_public", fail_release)
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker_with_ask_step(),
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    task = await wait_for_runtime_task(coordinator)
    await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)
    prepared.cancel_release.set()
    await task

    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.status is MessageStatus.FAILED
    assert runtime.failure == "jev:timeout"
    assert store.finish_failures[-1] == "jev:timeout"
    await coordinator.close()


@pytest.mark.asyncio
async def test_failed_prepared_release_does_not_read_fallback_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(FISH_ID)
    prepared = FakeCommittedPrepared(
        output_token_error=RuntimeError("metric sink failed")
    )
    llm = FakeChatClient(["不得重启"])
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    async def fail_release(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AnswerGenerationError

    monkeypatch.setattr(prepared, "release_to_public", fail_release)
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker_with_ask_step(),
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    task = await wait_for_runtime_task(coordinator)

    await task

    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert task.exception() is None
    assert runtime.status is MessageStatus.FAILED
    assert prepared.settled.is_set()
    assert llm.calls == []
    await coordinator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["complete", "fail", "cancel"])
async def test_prepared_performance_is_exported_after_settlement_before_finish(
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store, FakeChatClient(["不得重启"]), system_prompt="system"
    )
    prepared = FakeCommittedPrepared()
    entered = asyncio.Event()
    release = prepared.release_to_public
    tracker = tracker_with_ask_step()
    origin_ns = tracker.origin_ns
    exports: list[MessagePerformanceTracker] = []

    async def controlled_release(*args: object, **kwargs: object) -> None:
        entered.set()
        if outcome == "fail":
            raise AnswerGenerationError
        if outcome == "cancel":
            await asyncio.Future()
        await release(*args, **kwargs)

    def export(formal_tracker: MessagePerformanceTracker) -> None:
        assert store.finish_calls == []
        assert formal_tracker is tracker
        assert formal_tracker.origin_ns == origin_ns
        if outcome != "complete":
            assert prepared.settled.is_set()
        exports.append(formal_tracker)
        now_ns = formal_tracker.now_ns()
        formal_tracker.record(
            "voice_draft.preparation", started_ns=now_ns, finished_ns=now_ns
        )

    monkeypatch.setattr(prepared, "release_to_public", controlled_release)
    monkeypatch.setattr(
        prepared, "record_preparation_performance", export, raising=False
    )
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker,
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    task = await wait_for_runtime_task(coordinator)
    try:
        await asyncio.wait_for(entered.wait(), timeout=0.2)
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
            assert len(store.finish_calls) == 1
        assert exports == [tracker]
        assert any(
            step.name == "voice_draft.preparation" for step in tracker.snapshot()
        )
    finally:
        await coordinator.close()


@pytest.mark.asyncio
async def test_duplicate_voice_notifications_create_one_answer() -> None:
    store = FakeStore(FISH_ID)
    llm = FakeChatClient(["一次回答"])
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")
    answer_id = voice_answer_message_id(USER_MESSAGE_ID)
    source = voice_source_message()

    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        answer_id,
        tracker_with_ask_step(),
        source_message=source,
        draft_lease=None,
    )
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        answer_id,
        tracker_with_ask_step(),
        source_message=source,
        draft_lease=None,
    )
    task = await wait_for_runtime_task(coordinator)
    await task
    await asyncio.sleep(0)

    assert store.insert_count == 1
    assert store.answer_message_ids == [answer_id]
    assert len(llm.calls) == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_duplicate_voice_notification_settles_unconsumed_claimed_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["已完成"]),
        system_prompt="system",
    )
    answer_id = voice_answer_message_id(USER_MESSAGE_ID)
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        answer_id,
        tracker_with_ask_step(),
        source_message=voice_source_message(),
        draft_lease=None,
    )
    await (await wait_for_runtime_task(coordinator))

    prepared = FakeCommittedPrepared()
    monkeypatch.setattr(
        coordinator._generation,
        "initial_input_hash",
        lambda messages, *, enable_home, **kwargs: "matching-input",
        raising=False,
    )
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        answer_id,
        tracker_with_ask_step(),
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    await asyncio.wait_for(prepared.settled.wait(), timeout=0.2)

    assert prepared.cancelled is True
    assert prepared.continue_calls == []
    assert store.insert_count == 1
    await coordinator.close()


def only_background_task(coordinator: AnswerCoordinator) -> asyncio.Task[None]:
    assert len(coordinator._background_tasks) == 1
    return next(iter(coordinator._background_tasks))


async def wait_until(
    predicate: Callable[[], bool],
    *,
    deadline_seconds: float = 0.2,
) -> None:
    async def poll() -> None:
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=deadline_seconds)


@pytest.mark.asyncio
async def test_prestart_voice_notification_cancel_settles_registry() -> None:
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(["unused"]),
        system_prompt="system",
    )
    prepared = BlockingCommittedPrepared()
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker_with_ask_step(),
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    task = only_background_task(coordinator)
    task.cancel()
    try:
        await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)
        assert not prepared.settled.is_set()
        prepared.cancel_release.set()
        await asyncio.wait_for(prepared.settled.wait(), timeout=0.2)
    finally:
        prepared.cancel_release.set()
        await coordinator.close()


@pytest.mark.asyncio
async def test_generation_lock_voice_cancel_settles_registry() -> None:
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(["unused"]),
        system_prompt="system",
    )
    prepared = BlockingCommittedPrepared()
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    await runtime.generation_lock.acquire()
    try:
        coordinator.notify_voice_turn(
            FISH_ID,
            SESSION_ID,
            1,
            USER_MESSAGE_ID,
            voice_answer_message_id(USER_MESSAGE_ID),
            tracker_with_ask_step(),
            source_message=voice_source_message(),
            draft_lease=draft_lease(prepared),
        )
        task = only_background_task(coordinator)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)
        assert not prepared.settled.is_set()
        prepared.cancel_release.set()
        await asyncio.wait_for(prepared.settled.wait(), timeout=0.2)
    finally:
        prepared.cancel_release.set()
        runtime.generation_lock.release()
        await coordinator.close()


@pytest.mark.asyncio
async def test_close_settles_voice_lease_cancelled_before_handler_start() -> None:
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(["unused"]),
        system_prompt="system",
    )
    prepared = BlockingCommittedPrepared()
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker_with_ask_step(),
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    only_background_task(coordinator).cancel()
    close_task = asyncio.create_task(coordinator.close())
    try:
        await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)
        await asyncio.sleep(0)
        assert not close_task.done()
        prepared.cancel_release.set()
        await asyncio.wait_for(close_task, timeout=0.2)
        assert prepared.settled.is_set()
    finally:
        prepared.cancel_release.set()
        await asyncio.gather(close_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_delete_settles_voice_lease_while_handler_waits_for_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest_lookup_started = asyncio.Event()
    latest_lookup_release = asyncio.Event()
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["unused"]),
        system_prompt="system",
    )

    async def delayed_latest_sequence(fish_id: UUID, session_id: UUID) -> int:
        del fish_id, session_id
        latest_lookup_started.set()
        await latest_lookup_release.wait()
        return 1

    monkeypatch.setattr(store, "get_latest_user_sequence", delayed_latest_sequence)
    prepared = BlockingCommittedPrepared()
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    delete_task: asyncio.Task[bool] | None = None
    coordinator.notify_voice_turn(
        FISH_ID,
        SESSION_ID,
        1,
        USER_MESSAGE_ID,
        voice_answer_message_id(USER_MESSAGE_ID),
        tracker_with_ask_step(),
        source_message=voice_source_message(),
        draft_lease=draft_lease(prepared),
    )
    await asyncio.wait_for(latest_lookup_started.wait(), timeout=0.2)
    async with runtime.condition:
        latest_lookup_release.set()
        await wait_until(lambda: store.insert_count == 1)
        delete_task = asyncio.create_task(coordinator.delete_fish(FISH_ID))
        try:
            await asyncio.wait_for(prepared.cancel_started.wait(), timeout=0.2)
            await asyncio.sleep(0)
            assert not delete_task.done()
            prepared.cancel_release.set()
        finally:
            prepared.cancel_release.set()
    assert delete_task is not None
    deleted = await asyncio.wait_for(delete_task, timeout=0.2)

    assert deleted is True
    assert prepared.settled.is_set()
    await coordinator.close()


@pytest.mark.asyncio
async def test_pre_generation_failure_persists_timing_error() -> None:
    store = FakeStore(FISH_ID)
    store.history_error = RuntimeError("history unavailable")
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient([]),
        performance_store=performance_store,
        system_prompt="system",
    )

    with pytest.raises(RuntimeError, match="history unavailable"):
        await coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
        )
    await coordinator.close()

    _, steps = performance_store.writes[0]
    by_name = {step.name: step for step in steps}
    assert by_name["history.load"].error == "RuntimeError"
    assert by_name["context.load_total"].error == "RuntimeError"
    assert "llm.answer.initial" not in by_name
    assert "total" in by_name


@pytest.mark.asyncio
async def test_concurrent_distinct_asks_generate_for_latest_message_id() -> None:
    store = FakeStore(FISH_ID)
    store.latest_user_sequence = 2
    performance_store = FakePerformanceStore()
    llm = FakeChatClient(["最新回答"])
    coordinator = AnswerCoordinator(
        store,
        llm,
        performance_store=performance_store,
        system_prompt="system",
    )
    latest_message_id = UUID(int=202)

    await asyncio.gather(
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
        ),
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=2,
            message_id=latest_message_id,
            tracker=tracker_with_ask_step(),
        ),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert len(llm.calls) == 1
    latest_steps = next(
        steps
        for message_id, steps in performance_store.writes
        if message_id == latest_message_id
    )
    old_steps = next(
        steps
        for message_id, steps in performance_store.writes
        if message_id == USER_MESSAGE_ID
    )
    assert any(step.name == "llm.answer.initial" for step in latest_steps)
    assert all(step.name != "llm.answer.initial" for step in old_steps)


@pytest.mark.asyncio
async def test_history_is_bounded_to_owned_ask_sequence() -> None:
    store = FakeStore(FISH_ID)
    store.latest_sequence_after_answer_create = 2
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["回答"]),
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert store.history_before_sequences == [2]


@pytest.mark.asyncio
async def test_answer_persistence_failure_still_persists_performance_steps() -> None:
    store = FakeStore(FISH_ID)
    store.finish_error = RuntimeError("database unavailable")
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["回答"]),
        performance_store=performance_store,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    with pytest.raises(RuntimeError, match="database unavailable"):
        await runtime.llm_task
    assert runtime.status is MessageStatus.FAILED
    await coordinator.close()

    _, steps = performance_store.writes[0]
    by_name = {step.name: step for step in steps}
    assert by_name["answer.persist"].error == "RuntimeError"
    assert "total" in by_name


@pytest.mark.asyncio
async def test_newer_message_interrupts_and_persists_partial_answer() -> None:
    store = FakeStore(FISH_ID)
    llm = ControlledChatClient()
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        llm,
        performance_store=performance_store,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    old_task = runtime.llm_task
    assert old_task is not None
    await llm.wait_for_streams(1)
    await llm.emit(0, "旧回答")
    await wait_for_content(runtime, "旧回答")

    store.latest_user_sequence = 2
    await asyncio.wait_for(
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=2,
            message_id=UUID(int=202),
            tracker=tracker_with_ask_step(),
        ),
        timeout=1,
    )

    assert old_task.done()
    assert store.finish_calls[0] == {
        "message_id": UUID(int=101),
        "fish_id": FISH_ID,
        "content": "旧回答",
        "requested_status": MessageStatus.INTERRUPTED,
    }
    assert runtime.answer_message_id == UUID(int=102)
    assert runtime.content == ""

    await llm.wait_for_streams(2)
    await llm.emit(1, "新回答")
    await llm.finish(1)
    assert runtime.llm_task is not None
    await runtime.llm_task
    assert runtime.content == "新回答"
    assert runtime.status is MessageStatus.COMPLETED
    await coordinator.close()

    assert {message_id for message_id, _ in performance_store.writes} == {
        USER_MESSAGE_ID,
        UUID(int=202),
    }
    old_steps = next(
        steps
        for message_id, steps in performance_store.writes
        if message_id == USER_MESSAGE_ID
    )
    old_llm_step = next(step for step in old_steps if step.name == "llm.answer.initial")
    assert old_llm_step.error == "CancelledError"
    assert old_llm_step.output_payload == {
        "content": '{"m":"neutral","a":"旧回答',
        "finish_reason": None,
        "input_tokens": None,
        "output_tokens": None,
    }


@pytest.mark.asyncio
async def test_explicit_interrupt_persists_only_consumed_answer_content() -> None:
    store = FakeStore(FISH_ID)
    llm = ControlledChatClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.answer_message_id is not None
    await llm.wait_for_streams(1)
    await llm.emit(0, "一二三四五六")
    await wait_for_content(runtime, "一二三四五六")

    await coordinator.interrupt_answer(FISH_ID, runtime.answer_message_id, 5)

    assert runtime.status is MessageStatus.INTERRUPTED
    assert runtime.consumed_offset == 5
    assert runtime.content == "一二三四五"
    assert store.progress_calls == [(runtime.answer_message_id, FISH_ID, 5)]
    assert store.finish_calls[0]["content"] == "一二三四五"
    assert store.finish_calls[0]["requested_status"] is MessageStatus.INTERRUPTED


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt_with_new_message", [False, True])
async def test_interrupt_during_persistence_releases_next_answer(
    interrupt_with_new_message: bool,
) -> None:
    class BlockedCompletionStore(FakeStore):
        started = asyncio.Event()

        async def finish_answer(self, *args, **kwargs):
            if not self.started.is_set():
                self.started.set()
                await asyncio.Event().wait()
            return await super().finish_answer(*args, **kwargs)

        async def insert_generating_answer(self, *args, **kwargs):
            # Mirror the database's one-generating-answer constraint.
            if self.insert_count and not self.finish_calls:
                raise RuntimeError("previous answer still generating")
            return await super().insert_generating_answer(*args, **kwargs)

    store = BlockedCompletionStore(FISH_ID)
    performance = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        FakeChatClient(["我得好好捋捋哦，后半句"]),
        performance_store=performance,
        system_prompt="system",
    )
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
        playback_owner="companion",
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    await asyncio.wait_for(store.started.wait(), timeout=1)
    old_answer_id = runtime.answer_message_id
    assert old_answer_id is not None
    await coordinator.update_progress(FISH_ID, old_answer_id, 7)
    if not interrupt_with_new_message:
        await coordinator.interrupt_answer(FISH_ID, old_answer_id, 7)
        assert runtime.status is MessageStatus.INTERRUPTED

    store.latest_user_sequence = 2
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=2,
        message_id=UUID(int=202),
        tracker=tracker_with_ask_step(),
    )
    assert runtime.llm_task is not None
    await runtime.llm_task
    await coordinator.close()

    assert store.finish_calls[0] == {
        "message_id": old_answer_id,
        "fish_id": FISH_ID,
        "content": "我得好好捋捋哦",
        "requested_status": MessageStatus.INTERRUPTED,
    }
    assert store.insert_count == 2
    assert runtime.status is MessageStatus.COMPLETED
    assert len(performance.writes) == 2


@pytest.mark.asyncio
async def test_interrupt_still_finishes_a_generation_task_that_already_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store, FakeChatClient(["正常回应"]), system_prompt="system"
    )
    original = coordinator._run_generation

    async def crashed_generation(*args, **kwargs):
        raise ValueError("a must not be blank")

    monkeypatch.setattr(coordinator, "_run_generation", crashed_generation)
    try:
        await coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
            playback_owner="companion",
        )
        runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
        with pytest.raises(ValueError, match="a must not be blank"):
            await runtime.llm_task
        answer_id = runtime.answer_message_id
        runtime.content = "听到。还未说完。"
        await coordinator.interrupt_answer(FISH_ID, answer_id, 3)
        assert runtime.status is MessageStatus.INTERRUPTED
        assert store.finish_calls[0]["requested_status"] is MessageStatus.INTERRUPTED
        assert store.finish_calls[0]["content"] == "听到。"

        monkeypatch.setattr(coordinator, "_run_generation", original)
        store.latest_user_sequence = 2
        await coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            2,
            message_id=UUID(int=202),
            tracker=tracker_with_ask_step(),
        )
        await runtime.llm_task
        assert runtime.status is MessageStatus.COMPLETED
    finally:
        await asyncio.gather(coordinator.close(), return_exceptions=True)


@pytest.mark.asyncio
async def test_explicit_interrupt_stops_completed_answer_playback() -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store, FakeChatClient(["一二三四五六"]), system_prompt="system"
    )
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    assert runtime.answer_message_id is not None
    assert runtime.status is MessageStatus.COMPLETED

    await coordinator.interrupt_answer(FISH_ID, runtime.answer_message_id, 3)

    assert runtime.status is MessageStatus.INTERRUPTED
    assert runtime.content == "一二三"
    assert runtime.consumed_offset == 3
    assert store.terminal_interrupt_calls == [(runtime.answer_message_id, FISH_ID, 3)]


@pytest.mark.asyncio
@pytest.mark.parametrize("consumed_offset", [0, 1, 2])
async def test_explicit_interrupt_records_failed_answer_playback_progress(
    consumed_offset: int,
) -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(store, FailingChatClient(), system_prompt="system")
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    assert runtime.answer_message_id is not None
    assert runtime.status is MessageStatus.FAILED

    await coordinator.interrupt_answer(
        FISH_ID, runtime.answer_message_id, consumed_offset
    )

    assert runtime.status is MessageStatus.FAILED
    assert runtime.content == "部分"
    assert runtime.consumed_offset == consumed_offset
    assert store.progress_calls == [
        (runtime.answer_message_id, FISH_ID, consumed_offset)
    ]
    assert store.stored_progress == consumed_offset
    assert store.terminal_interrupt_calls == []
    assert store.finish_calls == [
        {
            "message_id": runtime.answer_message_id,
            "fish_id": FISH_ID,
            "content": "部分",
            "requested_status": MessageStatus.FAILED,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("consumed_offset", "error"),
    [(-1, "non-negative"), (3, "generated content")],
)
async def test_explicit_interrupt_rejects_invalid_failed_answer_playback_progress(
    consumed_offset: int,
    error: str,
) -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(store, FailingChatClient(), system_prompt="system")
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    assert runtime.answer_message_id is not None
    original_offset = runtime.consumed_offset

    with pytest.raises(ValueError, match=error):
        await coordinator.interrupt_answer(
            FISH_ID, runtime.answer_message_id, consumed_offset
        )

    assert runtime.status is MessageStatus.FAILED
    assert runtime.content == "部分"
    assert runtime.consumed_offset == original_offset
    assert store.progress_calls == []
    assert store.terminal_interrupt_calls == []
    assert len(store.finish_calls) == 1
    assert store.finish_calls[0]["requested_status"] is MessageStatus.FAILED


@pytest.mark.asyncio
async def test_discards_chunk_returned_after_stream_cancellation() -> None:
    store = FakeStore(FISH_ID)
    llm = LateChunkAfterCancelClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    await llm.first_chunk_sent.wait()
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    await wait_for_content(runtime, "旧回答")

    store.latest_user_sequence = 2
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=2,
        message_id=UUID(int=202),
        tracker=tracker_with_ask_step(),
    )
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert store.finish_calls[0]["requested_status"] is MessageStatus.INTERRUPTED
    assert store.finish_calls[0]["content"] == "旧回答"
    assert runtime.content == "新回答"


@pytest.mark.asyncio
async def test_llm_failure_persists_partial_content_as_failed() -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        FailingChatClient(),
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert runtime.content == "部分"
    assert runtime.status is MessageStatus.FAILED
    assert store.finish_calls == [
        {
            "message_id": UUID(int=101),
            "fish_id": FISH_ID,
            "content": "部分",
            "requested_status": MessageStatus.FAILED,
        }
    ]


@pytest.mark.asyncio
async def test_retries_malformed_initial_json_before_visible_content() -> None:
    store = FakeStore(FISH_ID)
    llm = MalformedThenValidJsonClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert llm.json_calls == 2
    assert runtime.content == "重试成功"
    assert runtime.status is MessageStatus.COMPLETED
    assert store.finish_calls == [
        {
            "message_id": UUID(int=101),
            "fish_id": FISH_ID,
            "content": "重试成功",
            "requested_status": MessageStatus.COMPLETED,
        }
    ]


@pytest.mark.asyncio
async def test_retries_provider_timeout_before_visible_content() -> None:
    store = FakeStore(FISH_ID)
    llm = TimeoutThenValidJsonClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert llm.json_calls == 2
    assert runtime.content == "超时后重试成功"
    assert runtime.status is MessageStatus.COMPLETED


@pytest.mark.asyncio
async def test_does_not_retry_malformed_initial_json_after_visible_content() -> None:
    store = FakeStore(FISH_ID)
    llm = InvalidAfterVisibleJsonClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task

    assert llm.json_calls == 1
    assert runtime.content == "部分"
    assert runtime.status is MessageStatus.FAILED


@pytest.mark.asyncio
async def test_concurrent_progress_keeps_maximum_for_generating_answer() -> None:
    store = FakeStore(FISH_ID)
    llm = ControlledChatClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    await llm.wait_for_streams(1)
    await llm.emit(0, "abcdef")
    await wait_for_content(runtime, "abcdef")
    assert runtime.answer_message_id is not None

    await asyncio.gather(
        coordinator.update_progress(FISH_ID, runtime.answer_message_id, 5),
        coordinator.update_progress(FISH_ID, runtime.answer_message_id, 3),
    )

    assert runtime.consumed_offset == 5
    assert set(store.progress_calls) == {
        (runtime.answer_message_id, FISH_ID, 3),
        (runtime.answer_message_id, FISH_ID, 5),
    }
    assert runtime.llm_task is not None
    runtime.llm_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runtime.llm_task


@pytest.mark.asyncio
async def test_rejects_progress_beyond_generated_content() -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(store, FakeChatClient([]), system_prompt="system")
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.answer_message_id is not None

    with pytest.raises(ValueError, match="generated content"):
        await coordinator.update_progress(FISH_ID, runtime.answer_message_id, 1)


@pytest.mark.asyncio
async def test_terminal_progress_uses_database_atomic_update() -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store, FakeChatClient(["done"]), system_prompt="system"
    )
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.llm_task is not None
    await runtime.llm_task
    assert runtime.answer_message_id is not None

    result = await coordinator.update_progress(
        FISH_ID,
        runtime.answer_message_id,
        2,
    )

    assert result == 2
    assert store.progress_calls == [(runtime.answer_message_id, FISH_ID, 2)]


@pytest.mark.asyncio
async def test_progress_cannot_advance_an_interrupted_answer() -> None:
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(store, FakeChatClient([]), system_prompt="system")
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    runtime.answer_message_id = UUID(int=101)
    runtime.content = "一二三四五"
    runtime.consumed_offset = 3
    runtime.status = MessageStatus.INTERRUPTED

    result = await coordinator.update_progress(
        FISH_ID,
        runtime.answer_message_id,
        5,
    )

    assert result == 3
    assert runtime.consumed_offset == 3
    assert store.progress_calls == []


@pytest.mark.asyncio
async def test_rejects_negative_progress() -> None:
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient([]),
        system_prompt="system",
    )

    with pytest.raises(ValueError, match="non-negative"):
        await coordinator.update_progress(FISH_ID, UUID(int=999), -1)


@pytest.mark.asyncio
async def test_close_cancels_and_waits_for_generation_tasks() -> None:
    store = FakeStore(FISH_ID)
    llm = ControlledChatClient()
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        llm,
        performance_store=performance_store,
        system_prompt="system",
    )
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        sequence=1,
        message_id=USER_MESSAGE_ID,
        tracker=tracker_with_ask_step(),
    )
    await llm.wait_for_streams(1)
    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    task = runtime.llm_task
    assert task is not None

    await coordinator.close()

    assert task.done()
    assert store.finish_calls[0]["requested_status"] is MessageStatus.INTERRUPTED
    assert performance_store.writes[0][0] == USER_MESSAGE_ID


@pytest.mark.asyncio
async def test_different_fish_generate_with_isolated_runtime_content() -> None:
    second_fish = UUID("00000000-0000-0000-0000-000000000002")
    store = MultiFishStore({FISH_ID: 1, second_fish: 2})
    llm = ControlledChatClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await asyncio.gather(
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
        ),
        coordinator.handle_user_message(
            second_fish,
            SESSION_ID,
            sequence=2,
            message_id=UUID(int=202),
            tracker=tracker_with_ask_step(),
        ),
    )
    await llm.wait_for_streams(2)
    first_runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    second_runtime = coordinator.get_runtime(second_fish, SESSION_ID)
    assert first_runtime.llm_task is not None
    assert second_runtime.llm_task is not None
    assert not first_runtime.llm_task.done()
    assert not second_runtime.llm_task.done()

    await llm.emit(0, "first")
    await llm.emit(1, "second")
    await llm.finish(0)
    await llm.finish(1)
    await asyncio.gather(first_runtime.llm_task, second_runtime.llm_task)

    assert {first_runtime.content, second_runtime.content} == {"first", "second"}


@pytest.mark.asyncio
async def test_sessions_of_same_fish_generate_concurrently_with_isolated_history() -> (
    None
):
    second_session = UUID("00000000-0000-0000-0000-000000000102")
    store = MultiSessionStore(FISH_ID, {SESSION_ID: 1, second_session: 2})
    llm = ControlledChatClient()
    coordinator = AnswerCoordinator(store, llm, system_prompt="system")

    await asyncio.gather(
        coordinator.handle_user_message(
            FISH_ID,
            SESSION_ID,
            sequence=1,
            message_id=USER_MESSAGE_ID,
            tracker=tracker_with_ask_step(),
        ),
        coordinator.handle_user_message(
            FISH_ID,
            second_session,
            sequence=2,
            message_id=UUID(int=202),
            tracker=tracker_with_ask_step(),
        ),
    )
    await llm.wait_for_streams(2)
    first_runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    second_runtime = coordinator.get_runtime(FISH_ID, second_session)

    assert first_runtime.llm_task is not None
    assert second_runtime.llm_task is not None
    assert not first_runtime.llm_task.done()
    assert not second_runtime.llm_task.done()
    assert {
        json.loads(call[-1]["content"].splitlines()[1])["content"].splitlines()[-1]
        for call in llm.calls
    } == {
        str(SESSION_ID),
        str(second_session),
    }

    await llm.emit(0, "first")
    await llm.emit(1, "second")
    await llm.finish(0)
    await llm.finish(1)
    await asyncio.gather(first_runtime.llm_task, second_runtime.llm_task)

    assert {first_runtime.content, second_runtime.content} == {"first", "second"}
    await coordinator.close()
