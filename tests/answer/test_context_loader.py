import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fish_memory.agent.models import (
    AgentEndpoint,
    AgentHistorySource,
    AgentResolvedTarget,
    AgentRun,
    AgentRunStatus,
    AgentTarget,
    AgentTask,
    AgentTaskCandidate,
    ConnectorTypeSlug,
)
from fish_memory.agent.routing import (
    AgentTaskRouteMode,
    LocalConnectorCapabilityPolicy,
)
from fish_memory.agent.runtime_state import AgentRuntimeState
from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.history_tokens import (
    estimate_history_message_tokens,
    retain_recent_message_window,
)
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    NormalizedStaticProfile,
    SnapshotItemState,
)
from fish_memory.soul import SoulProfile

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000101")
NOW = datetime(2026, 8, 9, 12, tzinfo=UTC)
TARGET_ID = UUID(int=201)


@pytest.mark.asyncio
async def test_task_switch_is_shared_by_chat_and_voice_draft_contexts():
    state = AgentRuntimeState(enabled=False, default_target_id=None)
    targets = RecordingAgentTargetStore(agent_target())
    tasks = RecordingAgentTaskStore(())
    history = ReadOnlyHistoryStore([user_message("帮我调查这个项目")])
    loader = AnswerContextLoader(
        history,
        now=lambda: NOW,
        system_prompt="base",
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_runtime_state=state,
    )
    off = await loader.load_for_generation(
        FISH_ID,
        SESSION_ID,
        through_sequence=10,
        tracker=MessagePerformanceTracker(),
    )
    assert off.agent_target is None and targets.calls == []
    state.configure(enabled=True, default_target_id=TARGET_ID)
    draft = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="帮我调查这个项目",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )
    assert draft.agent_target == agent_target()
    assert targets.calls == [(FISH_ID, TARGET_ID)]
    state.configure(enabled=False, default_target_id=TARGET_ID)
    off_again = await loader.load_for_generation(
        FISH_ID,
        SESSION_ID,
        through_sequence=10,
        tracker=MessagePerformanceTracker(),
    )
    assert off_again.agent_target is None and off_again.messages == off.messages
    assert targets.calls == [(FISH_ID, TARGET_ID)]


def user_message(
    content: str = "你好",
    *,
    sequence: int = 10,
    created_at: datetime = NOW,
    history_token_count: int = 0,
) -> Message:
    return Message(
        id=UUID(int=sequence),
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=sequence,
        role=Role.USER,
        content=content,
        status=MessageStatus.INITIALIZED,
        created_at=created_at,
        history_token_count=history_token_count,
    )


def dynamic_snapshot() -> NormalizedDynamicSnapshot:
    return NormalizedDynamicSnapshot(
        items=(
            SnapshotItemState(
                id=2,
                summary="用户正在整理 Answer 组件",
                support_fact_ids=(UUID(int=2),),
                lineage_fact_ids=(UUID(int=2), UUID(int=12)),
                last_supported_at=NOW,
                item_version=2,
            ),
        ),
        through_fact_sequence=2,
        version=1,
    )


def static_profile() -> NormalizedStaticProfile:
    return NormalizedStaticProfile(
        items=(
            SnapshotItemState(
                id=3,
                summary="用户重视低延迟",
                support_fact_ids=(UUID(int=3),),
                lineage_fact_ids=(UUID(int=3), UUID(int=13)),
                last_supported_at=NOW,
                item_version=4,
            ),
        ),
        through_fact_sequence=3,
        version=1,
    )


class BlockingHistoryStore:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.started = asyncio.Event()
        self.cancelled = False
        self.calls: list[tuple[UUID, UUID, int, int]] = []

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
    ) -> list[Message]:
        self.calls.append((fish_id, session_id, before_sequence, limit))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return [user_message()]


class ImmediateHistoryStore:
    def __init__(self, content: str) -> None:
        self.content = content

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
    ) -> list[Message]:
        del fish_id, session_id, before_sequence, limit
        return [user_message(self.content)]


class ReadOnlyHistoryStore:
    def __init__(self, rows: list[Message]) -> None:
        self.rows = rows
        self.mutable_calls = 0

    async def load_history(self, *args: object, **kwargs: object) -> list[Message]:
        del args, kwargs
        self.mutable_calls += 1
        return self.rows

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
        return self.rows


@pytest.mark.asyncio
async def test_active_plugin_context_reaches_chat_and_voice_draft_then_disappears():
    current = "\n当前活动：主持直播"
    store = ReadOnlyHistoryStore([user_message()])
    loader = AnswerContextLoader(
        store,
        now=lambda: NOW,
        system_prompt="base",
        activity_context_provider=lambda fish, session: (
            current if (fish, session) == (FISH_ID, SESSION_ID) else ""
        ),
    )
    baseline = AnswerContextLoader(store, now=lambda: NOW, system_prompt="base")

    async def load(target):
        return await target.load_for_generation(
            FISH_ID,
            SESSION_ID,
            through_sequence=10,
            tracker=MessagePerformanceTracker(),
        )

    active = await load(loader)
    draft = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="你好",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )
    assert active.activity_context == draft.activity_context == current
    assert current in active.messages[0]["content"]
    assert current in draft.messages[0]["content"]
    assert active.messages[0]["activity_context"] == current
    assert draft.messages[0]["activity_context"] == current
    current = ""
    stopped, original = await load(loader), await load(baseline)
    assert stopped.messages == original.messages
    assert stopped.activity_context == ""


def agent_target() -> AgentResolvedTarget:
    endpoint = AgentEndpoint(
        id=UUID(int=202),
        connector_type=ConnectorTypeSlug("hermes"),
        base_url="http://localhost:9000",
        created_at=NOW,
        updated_at=NOW,
    )
    target = AgentTarget(
        id=TARGET_ID,
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


def agent_candidate(*, connector_type: str = "hermes") -> AgentTaskCandidate:
    resolved = agent_target()
    endpoint = replace(
        resolved.endpoint,
        connector_type=ConnectorTypeSlug(connector_type),
    )
    task = AgentTask(
        id=UUID(int=203),
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        target_id=TARGET_ID,
        external_session_id="fm-task-test",
        title="运行中的测试任务",
        created_at=NOW,
        updated_at=NOW,
    )
    run = AgentRun(
        id=UUID(int=204),
        task_id=task.id,
        target_id=TARGET_ID,
        fish_id=task.fish_id,
        session_id=task.session_id,
        previous_run_id=None,
        source_message_id=UUID(int=205),
        confirmation_message_id=UUID(int=206),
        input="原始任务",
        external_run_id="external-run",
        history_source=AgentHistorySource.NONE,
        status=AgentRunStatus.RUNNING,
        output=None,
        result_message_id=None,
        error_code=None,
        error_message=None,
        provider_data=None,
        created_at=NOW,
        started_at=NOW,
        updated_at=NOW,
        finished_at=None,
    )
    return AgentTaskCandidate(
        task=task,
        run=run,
        interaction=None,
        target=resolved.target,
        endpoint=endpoint,
    )


class RecordingAgentTargetStore:
    def __init__(self, target: AgentResolvedTarget | None) -> None:
        self.target = target
        self.calls: list[tuple[UUID, UUID | None]] = []

    async def resolve_for_fish(
        self,
        fish_id: UUID,
        default_target_id: UUID | None,
    ) -> AgentResolvedTarget | None:
        self.calls.append((fish_id, default_target_id))
        return self.target


class RecordingAgentTaskStore:
    def __init__(
        self,
        candidates: tuple[AgentTaskCandidate, ...],
        *,
        has_recent_agent_reply: bool = False,
    ) -> None:
        self.candidates = candidates
        self._has_recent_agent_reply = has_recent_agent_reply
        self.calls: list[tuple[UUID, UUID, int, int, frozenset[str] | None]] = []
        self.recent_reply_calls: list[tuple[UUID, UUID, int | None, int]] = []

    async def list_task_candidates(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        recent_terminal_limit: int,
        limit: int,
        supported_connector_types: frozenset[str] | None = None,
    ) -> tuple[AgentTaskCandidate, ...]:
        self.calls.append(
            (
                fish_id,
                session_id,
                recent_terminal_limit,
                limit,
                supported_connector_types,
            )
        )
        return self.candidates

    async def has_recent_agent_reply(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> bool:
        self.recent_reply_calls.append((fish_id, session_id, before_sequence, limit))
        return self._has_recent_agent_reply


class RetainingHistoryStore:
    def __init__(
        self,
        rows: list[Message],
        *,
        window_start: int,
        window_total: int,
        token_limit: int,
        retain_limit: int,
    ) -> None:
        self.rows = rows
        self.window_start = window_start
        self.window_total = window_total
        self.token_limit = token_limit
        self.retain_limit = retain_limit

    def _token_count(self, message: Message) -> int:
        return message.history_token_count or estimate_history_message_tokens(
            message.content
        )

    def _retained_window(
        self,
        appended_token_count: int | None = None,
    ) -> tuple[int, int]:
        pairs = [
            (message.sequence, self._token_count(message)) for message in self.rows
        ]
        if appended_token_count is not None:
            pairs.append(
                (
                    max((sequence for sequence, _ in pairs), default=0) + 1,
                    appended_token_count,
                )
            )
        return retain_recent_message_window(
            pairs,
            retain_limit=self.retain_limit,
        )

    def _visible_history(
        self,
        *,
        before_sequence: int,
        start_sequence: int,
        limit: int,
    ) -> list[Message]:
        rows = [
            message
            for message in self.rows
            if start_sequence <= message.sequence < before_sequence
        ]
        return rows[-limit:]

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
    ) -> list[Message]:
        del fish_id, session_id
        return self._visible_history(
            before_sequence=before_sequence,
            start_sequence=self.window_start,
            limit=limit,
        )

    async def load_history_readonly(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int,
        appended_history_token_count: int | None = None,
    ) -> list[Message]:
        del fish_id, session_id
        if appended_history_token_count is None:
            start_sequence = self.window_start
        elif (
            self.window_start == 0
            or self.window_total + appended_history_token_count > self.token_limit
        ):
            start_sequence, _ = self._retained_window(appended_history_token_count)
        else:
            start_sequence = self.window_start
        return self._visible_history(
            before_sequence=before_sequence,
            start_sequence=start_sequence,
            limit=limit,
        )

    def commit_user(self, content: str, created_at: datetime) -> Message:
        message = user_message(
            content,
            sequence=max((item.sequence for item in self.rows), default=0) + 1,
            created_at=created_at,
        )
        self.rows.append(message)
        message_token_count = self._token_count(message)
        if self.window_start == 0:
            self.window_start, self.window_total = self._retained_window()
        elif self.window_total + message_token_count <= self.token_limit:
            self.window_total += message_token_count
        else:
            self.window_start, self.window_total = self._retained_window()
        return message


class BlockingSnapshotStore:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.dynamic_started = asyncio.Event()
        self.static_started = asyncio.Event()
        self.dynamic_cancelled = False
        self.static_cancelled = False
        self.dynamic_failure: Exception | None = None
        self.static_failure: Exception | None = None

    async def load_normalized_dynamic_snapshot(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> NormalizedDynamicSnapshot | None:
        assert fish_id == FISH_ID
        assert session_id == SESSION_ID
        self.dynamic_started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.dynamic_cancelled = True
            raise
        if self.dynamic_failure is not None:
            raise self.dynamic_failure
        return dynamic_snapshot()

    async def load_normalized_static_profile(
        self,
        fish_id: UUID,
    ) -> NormalizedStaticProfile | None:
        assert fish_id == FISH_ID
        self.static_started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.static_cancelled = True
            raise
        if self.static_failure is not None:
            raise self.static_failure
        return static_profile()


class BlockingSoulStore:
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release
        self.started = asyncio.Event()
        self.cancelled = False
        self.failure: Exception | None = None

    async def get_profile(self, fish_id: UUID) -> SoulProfile | None:
        assert fish_id == FISH_ID
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.failure is not None:
            raise self.failure
        return SoulProfile.from_values("记忆助手", "直接、简洁", "小鱼")


def create_loader(
    release: asyncio.Event,
) -> tuple[
    AnswerContextLoader,
    BlockingHistoryStore,
    BlockingSnapshotStore,
    BlockingSoulStore,
]:
    history = BlockingHistoryStore(release)
    snapshots = BlockingSnapshotStore(release)
    souls = BlockingSoulStore(release)
    return (
        AnswerContextLoader(
            history,
            snapshot_store=snapshots,
            fish_store=souls,
            history_message_limit=50,
            dynamic_snapshot_max_age=timedelta(days=7),
            now=lambda: NOW,
            system_prompt="base prompt",
        ),
        history,
        snapshots,
        souls,
    )


async def wait_until_all_started(
    history: BlockingHistoryStore,
    snapshots: BlockingSnapshotStore,
    souls: BlockingSoulStore,
) -> None:
    await asyncio.wait_for(
        asyncio.gather(
            history.started.wait(),
            snapshots.dynamic_started.wait(),
            snapshots.static_started.wait(),
            souls.started.wait(),
        ),
        timeout=1,
    )


@pytest.mark.asyncio
async def test_loads_context_sources_concurrently_and_builds_messages() -> None:
    release = asyncio.Event()
    loader, history, snapshots, souls = create_loader(release)
    tracker = MessagePerformanceTracker()

    task = asyncio.create_task(
        loader.load(
            FISH_ID,
            SESSION_ID,
            through_sequence=10,
            tracker=tracker,
        )
    )
    await wait_until_all_started(history, snapshots, souls)
    release.set()

    messages = await task

    assert history.calls == [(FISH_ID, SESSION_ID, 11, 50)]
    assert messages[0]["role"] == "system"
    assert "base prompt" in messages[0]["content"]
    assert "名字：小鱼" in messages[0]["content"]
    assert "用户重视低延迟" in messages[1]["content"]
    assert "用户正在整理 Answer 组件" not in str(messages)
    assert messages[-1]["content"].endswith("你好")
    step_names = [step.name for step in tracker.snapshot()]
    assert sorted(step_names) == sorted(
        [
            "history.load",
            "static.load",
            "snapshot.load",
            "soul.load",
            "context.load_total",
            "context.build",
        ]
    )


@pytest.mark.asyncio
async def test_appends_contextual_system_rule_when_current_message_matches() -> None:
    loader = AnswerContextLoader(
        ImmediateHistoryStore("把客厅空调打开"),
        now=lambda: NOW,
        system_prompt="base prompt",
        system_prompt_extension=lambda content: (
            "\n\nhome rule" if "空调" in content else ""
        ),
    )

    messages = await loader.load(
        FISH_ID,
        SESSION_ID,
        through_sequence=10,
        tracker=MessagePerformanceTracker(),
    )

    assert messages[0]["content"] == "base prompt\n\nhome rule"


@pytest.mark.asyncio
async def test_keeps_system_prompt_unchanged_when_current_message_does_not_match() -> (
    None
):
    loader = AnswerContextLoader(
        ImmediateHistoryStore("今天天气怎么样"),
        now=lambda: NOW,
        system_prompt="base prompt",
        system_prompt_extension=lambda content: (
            "\n\nhome rule" if "空调" in content else ""
        ),
    )

    messages = await loader.load(
        FISH_ID,
        SESSION_ID,
        through_sequence=10,
        tracker=MessagePerformanceTracker(),
    )

    assert messages[0]["content"] == "base prompt"


@pytest.mark.asyncio
async def test_draft_context_does_not_mutate_history_and_appends_current_user() -> None:
    history = ReadOnlyHistoryStore([user_message("之前的问题")])
    loader = AnswerContextLoader(
        history,
        now=lambda: NOW,
        system_prompt="base prompt",
        system_prompt_extension=lambda content: (
            "\nvoice" if content == "新问题" else ""
        ),
    )

    loaded = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="新问题",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )

    assert loaded.messages[0]["content"] == "base prompt\nvoice"
    assert loaded.messages[-1]["content"].endswith("新问题")
    assert history.mutable_calls == 0


@pytest.mark.asyncio
async def test_draft_context_matches_context_after_committing_identical_user() -> None:
    history = ReadOnlyHistoryStore([user_message("之前的问题")])
    loader = AnswerContextLoader(
        history,
        now=lambda: NOW,
        system_prompt="base prompt",
        system_prompt_extension=lambda content: (
            "\nvoice" if content == "新问题" else ""
        ),
    )

    draft = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="新问题",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )
    history.rows.append(user_message("新问题", sequence=11))
    committed = await loader.load_for_generation(
        FISH_ID,
        SESSION_ID,
        through_sequence=11,
        tracker=MessagePerformanceTracker(),
    )

    assert committed.messages == draft.messages
    assert committed.recall_conversation == draft.recall_conversation


@pytest.mark.asyncio
async def test_loads_resolved_target_and_immutable_task_context_once_per_snapshot() -> (
    None
):
    history = ReadOnlyHistoryStore([user_message("之前的问题")])
    targets = RecordingAgentTargetStore(agent_target())
    tasks = RecordingAgentTaskStore((agent_candidate(),))
    loader = AnswerContextLoader(
        history,
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_default_target_id=TARGET_ID,
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    loaded = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="把正在跑的版本改成 Python 3.13",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )

    assert loaded.agent_target == agent_target()
    assert loaded.task_lane_context.candidates[0].mode is AgentTaskRouteMode.STEER
    assert isinstance(loaded.task_lane_context.candidates, tuple)
    assert targets.calls == [(FISH_ID, TARGET_ID)]
    assert tasks.calls == [(FISH_ID, SESSION_ID, 5, 8, frozenset({"hermes"}))]
    assert tasks.recent_reply_calls == [(FISH_ID, SESSION_ID, None, 5)]


@pytest.mark.asyncio
async def test_committed_context_freezes_recent_agent_reply_signal() -> None:
    history = ReadOnlyHistoryStore([user_message("还要好吃的", sequence=10)])
    record = agent_candidate()
    record = replace(
        record,
        run=replace(
            record.run,
            status=AgentRunStatus.COMPLETED,
            result_message_id=UUID(int=207),
            finished_at=NOW,
        ),
    )
    targets = RecordingAgentTargetStore(agent_target())
    tasks = RecordingAgentTaskStore(
        (record,),
        has_recent_agent_reply=True,
    )
    loader = AnswerContextLoader(
        history,
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_default_target_id=TARGET_ID,
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    loaded = await loader.load_for_generation(
        FISH_ID,
        SESSION_ID,
        through_sequence=10,
        tracker=MessagePerformanceTracker(),
    )

    assert loaded.task_lane_context.has_recent_agent_reply is True
    assert loaded.task_lane_context.requires_probe is True
    assert tasks.recent_reply_calls == [(FISH_ID, SESSION_ID, 10, 5)]


@pytest.mark.asyncio
async def test_missing_target_skips_candidates_and_returns_empty_context() -> None:
    history = ReadOnlyHistoryStore([user_message("之前的问题")])
    targets = RecordingAgentTargetStore(None)
    tasks = RecordingAgentTaskStore((agent_candidate(),))
    loader = AnswerContextLoader(
        history,
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_default_target_id=TARGET_ID,
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    loaded = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="帮我调查故障",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )

    assert loaded.agent_target is None
    assert loaded.task_lane_context.candidates == ()
    assert targets.calls == [(FISH_ID, TARGET_ID)]
    assert tasks.calls == []


@pytest.mark.asyncio
async def test_unsupported_target_skips_candidates_and_returns_empty_context() -> None:
    history = ReadOnlyHistoryStore([user_message("之前的问题")])
    unsupported = agent_target()
    unsupported = replace(
        unsupported,
        endpoint=replace(
            unsupported.endpoint,
            connector_type=ConnectorTypeSlug("fake.valid"),
        ),
    )
    targets = RecordingAgentTargetStore(unsupported)
    tasks = RecordingAgentTaskStore((agent_candidate(),))
    loader = AnswerContextLoader(
        history,
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_default_target_id=TARGET_ID,
        agent_capability_policy=LocalConnectorCapabilityPolicy(
            supported_connector_types=frozenset({"hermes"}),
            steer_connector_types=frozenset({"hermes"}),
        ),
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    loaded = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="帮我调查故障",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )

    assert loaded.agent_target is None
    assert loaded.task_lane_context.candidates == ()
    assert targets.calls == [(FISH_ID, TARGET_ID)]
    assert tasks.calls == []


@pytest.mark.asyncio
async def test_supported_target_excludes_unsupported_task_candidates() -> None:
    history = ReadOnlyHistoryStore([user_message("之前的问题")])
    targets = RecordingAgentTargetStore(agent_target())
    tasks = RecordingAgentTaskStore((agent_candidate(connector_type="fake.valid"),))
    loader = AnswerContextLoader(
        history,
        agent_target_store=targets,
        agent_task_store=tasks,
        agent_default_target_id=TARGET_ID,
        agent_capability_policy=LocalConnectorCapabilityPolicy(
            supported_connector_types=frozenset({"hermes"}),
            steer_connector_types=frozenset({"hermes"}),
        ),
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    loaded = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="继续那个任务",
        created_at=NOW,
        tracker=MessagePerformanceTracker(),
    )

    assert loaded.agent_target == agent_target()
    assert loaded.task_lane_context.candidates == ()
    assert targets.calls == [(FISH_ID, TARGET_ID)]
    assert tasks.calls == [(FISH_ID, SESSION_ID, 5, 8, frozenset({"hermes"}))]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("window_start", "window_total"),
    [(0, 0), (1, 24)],
    ids=["uninitialized-window", "initialized-window-rebuild"],
)
async def test_draft_context_matches_committed_context_after_retention_moves(
    window_start: int,
    window_total: int,
) -> None:
    draft_created_at = NOW + timedelta(hours=1)
    history = RetainingHistoryStore(
        [
            user_message("old one", sequence=1, history_token_count=12),
            user_message("old two", sequence=2, history_token_count=12),
        ],
        window_start=window_start,
        window_total=window_total,
        token_limit=30,
        retain_limit=20,
    )
    loader = AnswerContextLoader(
        history,
        now=lambda: NOW,
        system_prompt="base prompt",
    )

    draft = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="abcdefghijkl",
        created_at=draft_created_at,
        tracker=MessagePerformanceTracker(),
    )
    committed_user = history.commit_user("abcdefghijkl", draft_created_at)
    committed = await loader.load_for_generation(
        FISH_ID,
        SESSION_ID,
        through_sequence=committed_user.sequence,
        tracker=MessagePerformanceTracker(),
    )

    assert (
        draft.messages
        == committed.messages
        == [
            {"role": "system", "content": "base prompt"},
            {
                "role": "user",
                "content": (
                    "<message_time>2026-08-09 13:00:00+00:00</message_time>\n"
                    "abcdefghijkl"
                ),
            },
        ]
    )
    assert draft.recall_conversation == committed.recall_conversation == ()


@pytest.mark.asyncio
async def test_snapshot_and_soul_failures_use_current_fallbacks() -> None:
    release = asyncio.Event()
    release.set()
    loader, _history, snapshots, souls = create_loader(release)
    snapshots.dynamic_failure = RuntimeError("dynamic unavailable")
    snapshots.static_failure = ValueError("static unavailable")
    souls.failure = LookupError("soul unavailable")
    tracker = MessagePerformanceTracker()

    messages = await loader.load(
        FISH_ID,
        SESSION_ID,
        through_sequence=10,
        tracker=tracker,
    )

    assert messages == [
        {"role": "system", "content": "base prompt"},
        {
            "role": "user",
            "content": "<message_time>2026-08-09 12:00:00+00:00</message_time>\n你好",
        },
    ]
    by_name = {step.name: step for step in tracker.snapshot()}
    assert by_name["static.load"].error == "ValueError"
    assert by_name["snapshot.load"].error == "RuntimeError"
    assert by_name["soul.load"].error == "LookupError"
    assert by_name["context.load_total"].error is None


@pytest.mark.asyncio
async def test_cancellation_propagates_and_cancels_context_reads() -> None:
    release = asyncio.Event()
    loader, history, snapshots, souls = create_loader(release)
    tracker = MessagePerformanceTracker()
    task = asyncio.create_task(
        loader.load(
            FISH_ID,
            SESSION_ID,
            through_sequence=10,
            tracker=tracker,
        )
    )
    await wait_until_all_started(history, snapshots, souls)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert history.cancelled is True
    assert snapshots.dynamic_cancelled is True
    assert snapshots.static_cancelled is True
    assert souls.cancelled is True
    by_name = {step.name: step for step in tracker.snapshot()}
    assert by_name["context.load_total"].error == "CancelledError"
