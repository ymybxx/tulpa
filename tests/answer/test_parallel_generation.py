import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.agent.models import (
    AgentEndpoint,
    AgentHistorySource,
    AgentInteraction,
    AgentInteractionKind,
    AgentResolvedTarget,
    AgentRun,
    AgentRunStatus,
    AgentTarget,
    AgentTask,
    AgentTaskCandidate,
    ConnectorTypeSlug,
)
from fish_memory.agent.routing import (
    AgentTaskDecisionKind,
    AgentTaskRouteMode,
    LocalConnectorCapabilityPolicy,
    PreparedAgentTaskRoute,
    TaskLaneContext,
)
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.generation import AnswerGenerationError
from fish_memory.answer.lane_context import (
    build_home_lane_messages,
    build_recall_lane_messages,
    build_web_lane_messages,
)
from fish_memory.answer.output_sink import PromotableAnswerSink
from fish_memory.answer.parallel_generation import (
    ParallelAnswerGenerationPipeline,
    _build_recall_result,
)
from fish_memory.answer.rquery import RQueryResolution
from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.voice_draft_types import semantic_initial_input_hash
from fish_memory.home.models import (
    HomeActionStatus,
    HomeIntentType,
    HomeRunOutcome,
    ResolvedQueryScope,
)
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.models import MessageStatus, Role
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion
from fish_memory.recall.models import (
    RecalledFact,
    RecallExecutionTrace,
    RecallResult,
)

FISH_ID = UUID(int=1)
ANSWER_ID = UUID(int=2)
SOURCE_ID = UUID(int=3)


def _timed_lane_messages(
    timestamp: str,
    suffix: str = "question",
    *,
    tag: str = "message_time",
) -> list[ChatMessage]:
    return [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": f"<{tag}>{timestamp}</{tag}>\n{suffix}",
        },
    ]


@pytest.mark.parametrize(
    "builder",
    [
        build_recall_lane_messages,
        build_web_lane_messages,
        build_home_lane_messages,
    ],
    ids=["recall", "web", "home"],
)
@pytest.mark.parametrize("tag", ["message_time", "message_time_range"])
def test_semantic_hash_normalizes_generated_time_inside_current_wrapper(
    builder: Callable[..., list[ChatMessage]],
    tag: str,
) -> None:
    first = {
        "model": "same-model",
        "messages": builder(_timed_lane_messages("2026-08-30 08:00 UTC", tag=tag)),
    }
    second = {
        "model": "same-model",
        "messages": builder(_timed_lane_messages("2026-08-30 08:01 UTC", tag=tag)),
    }

    assert semantic_initial_input_hash(first) == semantic_initial_input_hash(second)


def test_semantic_hash_keeps_later_user_supplied_time_tag_significant() -> None:
    first = build_recall_lane_messages(
        _timed_lane_messages(
            "2026-08-30 08:00 UTC",
            "question <message_time>user-a</message_time>",
        )
    )
    second = build_recall_lane_messages(
        _timed_lane_messages(
            "2026-08-30 08:01 UTC",
            "question <message_time>user-b</message_time>",
        )
    )

    assert semantic_initial_input_hash(first) != semantic_initial_input_hash(second)


def test_semantic_hash_rejects_unknown_or_changed_wrapper_content() -> None:
    prefix = "# 当前用户消息（应用已进行 JSON 转义）\n"
    unknown_first = (
        prefix
        + '{"content":"<message_time>one</message_time>\\nquestion"}'
        + "\n\n# 当前问题的 R 通道复核\n# 非生产规则"
    )
    unknown_second = unknown_first.replace(">one<", ">two<")
    known_first = build_web_lane_messages(
        _timed_lane_messages("2026-08-30 08:00 UTC", "question one")
    )
    known_second = build_web_lane_messages(
        _timed_lane_messages("2026-08-30 08:01 UTC", "question two")
    )

    assert semantic_initial_input_hash(unknown_first) != semantic_initial_input_hash(
        unknown_second
    )
    assert semantic_initial_input_hash(known_first) != semantic_initial_input_hash(
        known_second
    )


def test_recall_result_builder_rejects_depth_two() -> None:
    with pytest.raises(ValueError, match="recall_depth"):
        _build_recall_result(
            {
                "emotion": PetEmotion.NEUTRAL,
                "recall_depth": 2,
                "query": "用户最近使用的音乐服务",
                "end": False,
            }
        )


class LaneChatClient:
    def __init__(
        self,
        *,
        recall: str,
        web: str = '{"w":0}',
        home: str = '{"h":0}',
        task: str = '{"t":0,"m":"neutral","a":""}',
        search: str = '{"searchContext":"联网结果"}',
        search_error: Exception | None = None,
        continuation: str = '{"append":"最终补充。"}',
        model: str = "fake",
    ) -> None:
        self.responses = {
            "recall": recall,
            "web": web,
            "home": home,
            "task": task,
            "search": search,
            "continuation": continuation,
        }
        self.search_error = search_error
        self.model = model
        self.calls: list[tuple[str, Sequence[ChatMessage]]] = []

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search
        lane = self._lane(messages, force_search=force_search)
        self.calls.append((lane, messages))
        if lane == "search" and self.search_error is not None:
            raise self.search_error
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
            "model": self.model,
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
        if messages[-1]["content"].startswith("# 可补充的信息"):
            return "continuation"
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


class SplitTaskLaneChatClient(LaneChatClient):
    def __init__(self, *, recall: str, task_chunks: Sequence[str]) -> None:
        super().__init__(recall=recall, task="".join(task_chunks))
        self.task_chunks = tuple(task_chunks)

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        lane = self._lane(messages, force_search=force_search)
        if lane != "task":
            async for chunk in super().stream_json(
                messages,
                enable_search=enable_search,
                force_search=force_search,
            ):
                yield chunk
            return
        self.calls.append((lane, messages))
        for content in self.task_chunks:
            await asyncio.sleep(0)
            yield ChatStreamChunk(content=content)


class SequencedContinuationLaneChatClient(LaneChatClient):
    def __init__(
        self,
        *,
        recall: str,
        continuations: Sequence[str | Exception],
        web: str = '{"w":0}',
        home: str = '{"h":0}',
        search: str = '{"searchContext":"联网结果"}',
    ) -> None:
        super().__init__(recall=recall, web=web, home=home, search=search)
        self.continuations = list(continuations)

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        lane = self._lane(messages, force_search=force_search)
        if lane != "continuation":
            async for chunk in super().stream_json(
                messages,
                enable_search=enable_search,
                force_search=force_search,
            ):
                yield chunk
            return
        self.calls.append((lane, messages))
        result = self.continuations.pop(0)
        if isinstance(result, Exception):
            raise result
        yield ChatStreamChunk(content=result)


class UsageLaneChatClient(LaneChatClient):
    def __init__(self, usage: dict[str, int | None]) -> None:
        super().__init__(recall='{"m":"neutral","a":"R。","r":0,"e":0}')
        self.usage = usage

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search
        lane = self._lane(messages, force_search=force_search)
        self.calls.append((lane, messages))
        yield ChatStreamChunk(
            content=self.responses[lane],
            output_tokens=self.usage.get(lane),
        )


class BarrierLaneChatClient(LaneChatClient):
    def __init__(self) -> None:
        super().__init__(
            recall='{"m":"neutral","a":"R。","r":0,"e":0}',
            web='{"w":1,"m":"neutral","a":"W。","q":"query"}',
            home='{"h":1,"t":"control","m":"happy","a":"H。"}',
        )
        self.started: set[str] = set()
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search
        lane = self._lane(messages, force_search=force_search)
        if lane not in {"recall", "web", "home"}:
            async for chunk in super().stream_json(
                messages,
                force_search=force_search,
            ):
                yield chunk
            return
        self.calls.append((lane, messages))
        self.started.add(lane)
        if self.started == {"recall", "web", "home"}:
            self.all_started.set()
        await self.release.wait()
        yield ChatStreamChunk(content=self.responses[lane])


class PayloadOrderLaneChatClient(BarrierLaneChatClient):
    def __init__(self) -> None:
        super().__init__()
        self._hash_failed = False

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        del messages, enable_search, force_search
        if self.started != {"recall", "web", "home"}:
            raise RuntimeError("payload constructed before all lane streams started")
        if not self._hash_failed:
            self._hash_failed = True
            raise RuntimeError("deliberate payload hash failure")
        return {"model": self.model, "lane": "settlement"}


class CancellationSettlementLaneChatClient(LaneChatClient):
    def __init__(self) -> None:
        super().__init__(recall='{"m":"neutral","a":"R。","r":0,"e":0}')
        self.started = asyncio.Event()
        self.cancellation_started = asyncio.Event()
        self.cancellation_release = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search
        lane = self._lane(messages, force_search=force_search)
        self.calls.append((lane, messages))
        self.started.set()
        if False:
            yield ChatStreamChunk(content=self.responses[lane])
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancellation_started.set()
            await self.cancellation_release.wait()
            raise


class LateLosingRecallClient(LaneChatClient):
    def __init__(self) -> None:
        super().__init__(
            recall='{"m":"angry","a":"败选回答。","r":0,"e":0}',
            web=('{"w":1,"m":"shy","a":"联网回答。","q":"杭州下周活动"}'),
        )
        self.recall_cancelled = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        lane = self._lane(messages, force_search=force_search)
        if lane != "recall":
            async for chunk in super().stream_json(
                messages,
                enable_search=enable_search,
                force_search=force_search,
            ):
                yield chunk
            return
        self.calls.append((lane, messages))
        yield ChatStreamChunk(content='{"m":"angry","a":"败选')
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.recall_cancelled.set()
            # Simulate a provider transport that delivers one final buffered
            # packet while cancellation is unwinding.
            yield ChatStreamChunk(content='迟到回答。","r":0,"e":0}')


class BlockingSearchLaneChatClient(LaneChatClient):
    def __init__(self) -> None:
        super().__init__(
            recall='{"m":"neutral","a":"R。","r":0,"e":0}',
            web=(
                '{"w":1,"m":"neutral","a":"我去看看最新情况。","q":"台湾明天天气预报"}'
            ),
        )
        self.search_started = asyncio.Event()
        self.search_release = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        if not force_search:
            async for chunk in super().stream_json(
                messages,
                enable_search=enable_search,
                force_search=False,
            ):
                yield chunk
            return
        self.calls.append(("search", messages))
        self.search_started.set()
        await self.search_release.wait()
        yield ChatStreamChunk(content=self.responses["search"])


class BlockingRecallCompletionClient(LaneChatClient):
    def __init__(self) -> None:
        super().__init__(recall='{"m":"happy","a":"首段尾段","r":0,"e":0}')
        self.first_delta_sent = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search
        lane = self._lane(messages, force_search=force_search)
        self.calls.append((lane, messages))
        if lane != "recall":
            yield ChatStreamChunk(content=self.responses[lane])
            return
        yield ChatStreamChunk(content='{"m":"happy","a":"首段')
        self.first_delta_sent.set()
        await self.release.wait()
        yield ChatStreamChunk(content='尾段","r":0,"e":0}')


class BlockingTaskCompletionClient(LaneChatClient):
    def __init__(self) -> None:
        super().__init__(
            recall='{"m":"neutral","a":"R。","r":0,"e":0}',
            task='{"t":1,"m":"happy","a":"我来处理。"}',
        )
        self.partial_consumed = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search
        lane = self._lane(messages, force_search=force_search)
        self.calls.append((lane, messages))
        if lane != "task":
            yield ChatStreamChunk(content=self.responses[lane])
            return
        yield ChatStreamChunk(content='{"t":1,"m":"happy","a":"我来处理。"')
        self.partial_consumed.set()
        await self.release.wait()
        yield ChatStreamChunk(content="}")


class FakeRecallService:
    def __init__(
        self,
        *,
        fail: bool = False,
        include_facts: bool = True,
    ) -> None:
        self.fail = fail
        self.include_facts = include_facts
        self.calls: list[tuple[str, int]] = []
        self.planning_contents: list[str | None] = []

    async def recall(
        self,
        fish_id: UUID,
        content: str,
        depth: int,
        *,
        planning_content: str | None = None,
        entity_hints: Sequence[str] = (),
        conversation_context: Sequence[ChatMessage] = (),
        tracker: MessagePerformanceTracker | None = None,
    ) -> RecallResult:
        del fish_id, entity_hints, conversation_context, tracker
        self.calls.append((content, depth))
        self.planning_contents.append(planning_content)
        if self.fail:
            raise RuntimeError("recall unavailable")
        fact = RecalledFact(
            id=UUID(int=10),
            source_message_id=UUID(int=11),
            source_role=Role.USER,
            text="用户最近使用 Spotify",
            occurred_start=None,
            occurred_end=None,
            mentioned_at=datetime(2026, 8, 24, tzinfo=UTC),
            entities=("Spotify",),
            matched_by=("semantic",),
            semantic_score=0.9,
            rank_score=1.0,
        )
        return RecallResult(
            facts=(fact,) if self.include_facts else (),
            trace=RecallExecutionTrace(
                depth=depth,
                degraded=False,
                plan=None,
                plan_error=None,
                duration_ms={},
            ),
        )


class GatedRQueryIntegrityService:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[tuple[str, str]] = []

    async def resolve(
        self,
        fish_id: UUID,
        *,
        original_query: str,
        generated_query: str,
        tracker: MessagePerformanceTracker,
    ) -> RQueryResolution:
        del fish_id, tracker
        self.calls.append((original_query, generated_query))
        self.entered.set()
        await self.release.wait()
        return RQueryResolution(
            original_query=original_query,
            generated_query=generated_query,
            final_query="修复后的查询",
            original_entity_count=1,
            generated_entity_count=0,
            missing_aliases=("Spotify",),
            outcome="repaired",
        )


class FakeHomeRunHandle:
    def __init__(self, outcome: HomeRunOutcome) -> None:
        self.outcome = outcome

    async def result(self) -> HomeRunOutcome:
        return self.outcome


def messages(content: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": "soul"},
        {"role": "user", "content": "# 记忆摘要\nsummary"},
        {"role": "assistant", "content": "recent answer"},
        {"role": "user", "content": content},
    ]


def runtime() -> FishRuntime:
    value = FishRuntime()
    value.answer_message_id = ANSWER_ID
    value.status = MessageStatus.GENERATING
    return value


def resolved_agent_target(*, seed: int = 100) -> AgentResolvedTarget:
    endpoint = AgentEndpoint(
        id=UUID(int=seed + 1),
        connector_type=ConnectorTypeSlug("hermes"),
        base_url="http://localhost:9000",
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
        updated_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    target = AgentTarget(
        id=UUID(int=seed),
        endpoint_id=endpoint.id,
        external_key="default",
        route_prefix="",
        credential_ref="hermes:test",
        connector_config={},
        display_name="Hermes",
        enabled=True,
        created_at=datetime(2026, 8, 30, tzinfo=UTC),
        updated_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    return AgentResolvedTarget(target=target, endpoint=endpoint)


def task_candidate(
    status: AgentRunStatus,
    *,
    seed: int = 200,
    title: str = "候选任务",
) -> AgentTaskCandidate:
    resolved = resolved_agent_target(seed=seed + 20)
    now = datetime(2026, 8, 30, tzinfo=UTC)
    task = AgentTask(
        id=UUID(int=seed),
        fish_id=FISH_ID,
        session_id=UUID(int=seed + 10),
        target_id=resolved.target.id,
        external_session_id=f"fm-task-{seed}",
        title=title,
        created_at=now,
        updated_at=now,
    )
    run = AgentRun(
        id=UUID(int=seed + 1),
        task_id=task.id,
        target_id=resolved.target.id,
        fish_id=task.fish_id,
        session_id=task.session_id,
        previous_run_id=None,
        source_message_id=UUID(int=seed + 2),
        confirmation_message_id=UUID(int=seed + 3),
        input="原任务",
        external_run_id=(None if status is AgentRunStatus.QUEUED else f"run-{seed}"),
        history_source=AgentHistorySource.NONE,
        status=status,
        output="done" if status.is_terminal else None,
        result_message_id=UUID(int=seed + 4) if status.is_terminal else None,
        error_code=None,
        error_message=None,
        provider_data=None,
        created_at=now,
        started_at=now,
        updated_at=now,
        finished_at=now if status.is_terminal else None,
    )
    return AgentTaskCandidate(
        task=task,
        run=run,
        interaction=None,
        target=resolved.target,
        endpoint=resolved.endpoint,
    )


def frozen_task_context(
    records: Sequence[AgentTaskCandidate],
    *,
    content: str,
    policy_version: str = "agent-task-routing-v1",
    has_recent_agent_reply: bool = False,
) -> TaskLaneContext:
    return TaskLaneContext.from_records(
        tuple(records),
        current_content=content,
        capability_policy=LocalConnectorCapabilityPolicy(
            routing_policy_version=policy_version
        ),
        has_recent_agent_reply=has_recent_agent_reply,
    )


class RecordingTaskRouteService:
    def __init__(
        self, *, block: bool = False, failure: Exception | None = None
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.prepare_calls: list[dict[str, object]] = []
        self._prepared_inputs: dict[int, dict[str, object]] = {}
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.failure = failure
        if not block:
            self.release.set()

    async def prepare_selected_route(self, **kwargs) -> PreparedAgentTaskRoute:
        self.prepare_calls.append(dict(kwargs))
        indexes = kwargs["candidate_indexes"]
        prepared = PreparedAgentTaskRoute(
            fish_id=kwargs["fish_id"],
            target_id=kwargs["resolved_target"].target.id,
            kind=kwargs["kind"],
            content=kwargs["content"],
            candidate=(
                kwargs["task_context"].candidates[indexes[0]] if indexes else None
            ),
        )
        self._prepared_inputs[id(prepared)] = dict(kwargs)
        return prepared

    async def commit_prepared_route(
        self, prepared: PreparedAgentTaskRoute, **kwargs: object
    ) -> AgentRun:
        return await self.commit_selected_route(
            **self._prepared_inputs[id(prepared)], **kwargs
        )

    async def commit_selected_route(self, **kwargs: object) -> AgentRun:
        self.calls.append(dict(kwargs))
        self.entered.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        return task_candidate(AgentRunStatus.QUEUED, seed=900).run


@pytest.mark.asyncio
async def test_prepared_initial_answer_buffers_and_defers_routes() -> None:
    """Removing the commit gate would start Home before an owner accepts it."""
    calls: list[str] = []
    client = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        home='{"h":1,"t":"control","m":"happy","a":"H。"}',
    )
    pipeline = ParallelAnswerGenerationPipeline(
        client,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
    )
    sink = PromotableAnswerSink()
    prepared = pipeline.prepare_initial(
        FISH_ID,
        messages("打开客厅灯"),
        (),
        sink,
        MessagePerformanceTracker(),
        enable_home=True,
    )

    await prepared.initial_ready()

    assert sink.snapshot().content == "H。"
    assert prepared.route_evidence() == {
        "route": "home",
        "recall_depth": 0,
        "context_complete": True,
    }
    assert calls == []

    def start_home(intent: HomeIntentType) -> FakeHomeRunHandle:
        calls.append("home")
        return FakeHomeRunHandle(
            HomeRunOutcome(intent=intent, status=HomeActionStatus.SUCCEEDED)
        )

    await prepared.continue_after_commit(start_home_action=start_home)

    assert calls == ["home"]
    with pytest.raises(RuntimeError, match="already continued"):
        await prepared.continue_after_commit(start_home_action=start_home)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("usage", "expected"),
    [({"recall": 7, "web": 3}, 7), ({"recall": None, "web": None}, None)],
)
async def test_prepared_answer_reports_actual_provider_output_token_usage(
    usage: dict[str, int | None],
    expected: int | None,
) -> None:
    client = UsageLaneChatClient(usage)
    pipeline = ParallelAnswerGenerationPipeline(
        client,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
    )
    prepared = pipeline.prepare_initial(
        FISH_ID,
        messages("问题"),
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=False,
    )

    await prepared.initial_ready()

    assert prepared.output_token_usage() == expected
    await prepared.cancel()


@pytest.mark.asyncio
async def test_cancelled_prepared_initial_answer_never_starts_route_work() -> None:
    """A late lane result must not publish or cross the cancelled commit gate."""
    client = BarrierLaneChatClient()
    pipeline = ParallelAnswerGenerationPipeline(
        client,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=True,
    )
    sink = PromotableAnswerSink()
    prepared = pipeline.prepare_initial(
        FISH_ID,
        messages("打开客厅灯"),
        (),
        sink,
        MessagePerformanceTracker(),
        enable_home=True,
    )

    await asyncio.wait_for(client.all_started.wait(), timeout=1)
    prepared.cancel()
    client.release.set()

    with pytest.raises(asyncio.CancelledError):
        await prepared.initial_ready()
    with pytest.raises(RuntimeError, match="cancelled"):
        await prepared.continue_after_commit(start_home_action=lambda _intent: None)
    await asyncio.sleep(0)

    assert sink.snapshot().content == ""


@pytest.mark.asyncio
async def test_prepared_initial_answer_hashes_exact_payloads() -> None:
    """Ignoring payload content or model configuration could promote a stale draft."""
    common = {
        "recall": '{"m":"neutral","a":"R。","r":0,"e":0}',
    }
    punctuation_pipeline = ParallelAnswerGenerationPipeline(
        LaneChatClient(**common),
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
    )
    punctuation = punctuation_pipeline.prepare_initial(
        FISH_ID,
        messages("明天会下雨吗?"),
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=False,
    )
    full_width_punctuation = ParallelAnswerGenerationPipeline(
        LaneChatClient(**common),
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
    ).prepare_initial(
        FISH_ID,
        messages("明天会下雨吗？"),
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=False,
    )
    changed_model = ParallelAnswerGenerationPipeline(
        LaneChatClient(**common, model="other-generation-model"),
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
    ).prepare_initial(
        FISH_ID,
        messages("明天会下雨吗?"),
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=False,
    )

    await punctuation.initial_ready()
    await full_width_punctuation.initial_ready()
    await changed_model.initial_ready()

    assert punctuation.input_hash != full_width_punctuation.input_hash
    assert punctuation.input_hash != changed_model.input_hash
    assert (
        punctuation_pipeline.initial_input_hash(
            messages("明天会下雨吗?"),
            enable_home=False,
        )
        == punctuation.input_hash
    )

    punctuation.cancel()
    full_width_punctuation.cancel()
    changed_model.cancel()


@pytest.mark.asyncio
async def test_prepare_initial_starts_all_lanes_before_payload_hashing() -> None:
    """Hashing before stream startup could turn a preflight failure into no work."""
    client = PayloadOrderLaneChatClient()
    pipeline = ParallelAnswerGenerationPipeline(
        client,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=True,
    )
    prepared = pipeline.prepare_initial(
        FISH_ID,
        messages("打开客厅灯"),
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=True,
    )

    with pytest.raises(RuntimeError, match="deliberate payload hash failure"):
        await prepared.initial_ready()

    assert client.started == {"recall", "web", "home"}


@pytest.mark.asyncio
async def test_generate_cancellation_waits_for_lane_settlement() -> None:
    """Returning before lane cleanup can leak a provider task after cancellation."""
    client = CancellationSettlementLaneChatClient()
    generation = asyncio.create_task(
        ParallelAnswerGenerationPipeline(
            client,
            recall_service=None,
            recall_context_token_budget=1000,
            enable_search=False,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("你好"),
            (),
            runtime(),
            MessagePerformanceTracker(),
        )
    )

    await asyncio.wait_for(client.started.wait(), timeout=1)
    generation.cancel()
    await asyncio.wait_for(client.cancellation_started.wait(), timeout=1)

    assert not generation.done()

    client.cancellation_release.set()
    with pytest.raises(asyncio.CancelledError):
        await generation


@pytest.mark.asyncio
async def test_plain_turn_uses_recall_lane_direct_answer() -> None:
    llm = LaneChatClient(
        recall='{"m":"happy","a":"直接回答。","r":0,"e":0}',
    )
    value = runtime()
    home_intents: list[HomeIntentType] = []

    def start_home(intent: HomeIntentType) -> None:
        home_intents.append(intent)

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=FakeRecallService(),
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("你好"),
        (),
        value,
        MessagePerformanceTracker(),
        start_home_action=start_home,
    )

    assert value.content == "直接回答。"
    assert value.flush_offset == len("直接回答。")
    assert value.pet_emotion is PetEmotion.HAPPY
    assert home_intents == []
    assert {lane for lane, _ in llm.calls} == {"recall", "web"}


@pytest.mark.asyncio
async def test_plain_turn_defaults_missing_recall_end_to_false() -> None:
    llm = LaneChatClient(
        recall=(
            '{"m":"neutral","a":"我这就去查看您的当前位置。","r":1,'
            '"q":"查看用户当前位置"}'
        ),
    )
    value = runtime()
    recall = FakeRecallService()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=recall,
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("查看我的位置"),
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert value.content == "我这就去查看您的当前位置。最终补充。"
    assert value.end_conversation is False
    assert recall.calls == [("查看用户当前位置", 1)]


@pytest.mark.asyncio
async def test_home_lane_has_priority_and_starts_home_router() -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        web='{"w":1,"m":"neutral","a":"W。","q":"query"}',
        home='{"h":1,"t":"control","m":"happy","a":"我来处理一下。"}',
    )
    value = runtime()
    home_intents: list[HomeIntentType] = []

    def start_home(intent: HomeIntentType) -> None:
        home_intents.append(intent)

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=FakeRecallService(),
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("把客厅灯打开"),
        (),
        value,
        MessagePerformanceTracker(),
        start_home_action=start_home,
    )

    assert value.content == "我来处理一下。"
    assert value.flush_offset == len("我来处理一下。")
    assert value.pet_emotion is PetEmotion.HAPPY
    assert home_intents == [HomeIntentType.CONTROL]
    assert {lane for lane, _ in llm.calls} == {"recall", "web", "home"}


@pytest.mark.asyncio
async def test_home_query_appends_result_to_the_selected_answer() -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        home='{"h":1,"t":"query","m":"neutral","a":"我看看。"}',
        continuation='{"append":"客厅主灯开着。"}',
    )
    value = runtime()
    outcome = HomeRunOutcome(
        intent=HomeIntentType.QUERY,
        status=HomeActionStatus.SUCCEEDED,
        query=ResolvedQueryScope(entities=()),
    )
    source_messages = messages("客厅灯开着吗？")
    source_messages[-1]["environment_image"] = {
        "data_url": "data:image/jpeg;base64,private-frame",
        "digest": "a" * 64,
        "byte_count": 9,
        "width": 640,
        "height": 360,
        "observed_at": "2026-09-02T08:30:00+00:00",
    }

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=10_000,
        enable_search=False,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        source_messages,
        (),
        value,
        MessagePerformanceTracker(),
        start_home_action=lambda _intent: FakeHomeRunHandle(outcome),
    )

    assert value.content == "我看看。客厅主灯开着。"
    assert value.flush_offset == len("我看看。")
    assert [lane for lane, _messages in llm.calls].count("continuation") == 1
    home_call = next(call for lane, call in llm.calls if lane == "home")
    continuation_call = next(call for lane, call in llm.calls if lane == "continuation")
    assert all("environment_image" not in message for message in home_call)
    assert continuation_call[-1]["environment_image"]["digest"] == "a" * 64


@pytest.mark.asyncio
async def test_web_lane_has_priority_over_recall_lane() -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":1,"q":"memory","e":0}',
        web='{"w":1,"m":"shy","a":"我去了解一下。","q":"杭州下周活动"}',
    )
    recall = FakeRecallService()
    value = runtime()
    source_messages = messages("杭州下周有什么活动")
    source_messages[-1]["environment_image"] = {
        "data_url": "data:image/jpeg;base64,private-frame",
        "digest": "a" * 64,
        "byte_count": 9,
        "width": 640,
        "height": 360,
        "observed_at": "2026-09-02T08:30:00+00:00",
    }

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=recall,
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        source_messages,
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert value.content == "我去了解一下。最终补充。"
    assert value.flush_offset == len("我去了解一下。")
    assert value.pet_emotion is PetEmotion.SHY
    assert recall.calls == []
    search_call = next(call for lane, call in llm.calls if lane == "search")
    assert "杭州下周活动" in search_call[-1]["content"]
    web_call = next(call for lane, call in llm.calls if lane == "web")
    continuation_call = next(call for lane, call in llm.calls if lane == "continuation")
    assert all("environment_image" not in message for message in web_call)
    assert all("environment_image" not in message for message in search_call)
    assert continuation_call[-1]["environment_image"]["digest"] == "a" * 64


@pytest.mark.asyncio
async def test_recall_lane_query_drives_recall_after_transition() -> None:
    llm = LaneChatClient(
        recall=(
            '{"m":"neutral","a":"我想一下。","r":1,"q":"用户最近使用的音乐服务","e":0}'
        ),
    )
    recall = FakeRecallService()
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=recall,
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("我最近用什么听歌？"),
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert recall.calls == [("用户最近使用的音乐服务", 1)]
    assert recall.planning_contents == ["我最近用什么听歌？"]
    assert value.content == "我想一下。最终补充。"
    assert value.flush_offset == len("我想一下。")


@pytest.mark.asyncio
@pytest.mark.parametrize("recall_failure", [False, True], ids=["no-facts", "failed"])
async def test_unavailable_recall_completes_waiting_transition(
    recall_failure: bool,
) -> None:
    llm = LaneChatClient(
        recall=(
            '{"m":"neutral","a":"让我想想。","r":1,"q":"用户最近使用的音乐服务","e":0}'
        ),
        continuation='{"append":"这次没有足够可靠的信息让我确定答案，我先不乱猜。"}',
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=FakeRecallService(
            fail=recall_failure,
            include_facts=False,
        ),
        recall_context_token_budget=10_000,
        enable_search=False,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("我最近用什么听歌？"),
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert value.content == (
        "让我想想。这次没有足够可靠的信息让我确定答案，我先不乱猜。"
    )
    assert [lane for lane, _messages in llm.calls].count("continuation") == 1


@pytest.mark.asyncio
async def test_waiting_recall_retries_an_empty_continuation_before_publishing() -> None:
    llm = SequencedContinuationLaneChatClient(
        recall=(
            '{"m":"neutral","a":"关于那次旅行……","r":1,"q":"用户去年旅行的餐厅","e":0}'
        ),
        continuations=(
            '{"append":""}',
            '{"append":"这次没有足够可靠的信息让我确定答案，我先不乱猜。"}',
        ),
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=FakeRecallService(include_facts=False),
        recall_context_token_budget=10_000,
        enable_search=False,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("我去年旅行时去了哪家餐厅？"),
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert value.content == (
        "关于那次旅行……这次没有足够可靠的信息让我确定答案，我先不乱猜。"
    )
    assert [lane for lane, _messages in llm.calls].count("continuation") == 2


@pytest.mark.asyncio
async def test_rquery_repair_waits_for_shallow_answer() -> None:
    llm = LaneChatClient(
        recall=(
            '{"m":"neutral","a":"让我想想。","r":1,"q":"用户最近的音乐服务","e":0}'
        ),
    )
    recall = FakeRecallService()
    integrity = GatedRQueryIntegrityService()
    value = runtime()
    tracker = MessagePerformanceTracker()
    generation = asyncio.create_task(
        ParallelAnswerGenerationPipeline(
            llm,
            recall_service=recall,
            rquery_integrity_service=integrity,
            recall_context_token_budget=10_000,
            enable_search=False,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages(
                "<message_time>2026-08-28 08:00 CST</message_time>\n我最近用什么听歌？"
            ),
            (),
            value,
            tracker,
        )
    )

    await asyncio.wait_for(integrity.entered.wait(), timeout=1)

    assert value.content == "让我想想。"
    assert any(step.name == "answer.first_content" for step in tracker.snapshot())
    assert recall.calls == []

    integrity.release.set()
    await generation

    assert recall.calls == [("修复后的查询", 1)]
    assert recall.planning_contents == ["我最近用什么听歌？"]
    assert integrity.calls == [("我最近用什么听歌？", "用户最近的音乐服务")]


@pytest.mark.asyncio
async def test_all_configured_lanes_start_before_priority_arbitration() -> None:
    llm = BarrierLaneChatClient()
    value = runtime()
    home_intents: list[HomeIntentType] = []

    def start_home(intent: HomeIntentType) -> None:
        home_intents.append(intent)

    generation = asyncio.create_task(
        ParallelAnswerGenerationPipeline(
            llm,
            recall_service=FakeRecallService(),
            recall_context_token_budget=10_000,
            enable_search=True,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("把客厅灯打开"),
            (),
            value,
            MessagePerformanceTracker(),
            start_home_action=start_home,
        )
    )

    await asyncio.wait_for(llm.all_started.wait(), timeout=1)
    assert not generation.done()
    llm.release.set()
    await generation

    assert llm.started == {"recall", "web", "home"}
    assert value.content == "H。"
    assert value.pet_emotion is PetEmotion.HAPPY
    assert home_intents == [HomeIntentType.CONTROL]


@pytest.mark.asyncio
async def test_rejected_home_lane_never_starts_home_router() -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"普通回答。","r":0,"e":0}',
        home='{"h":0}',
    )
    value = runtime()
    home_intents: list[HomeIntentType] = []

    def start_home(intent: HomeIntentType) -> None:
        home_intents.append(intent)

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=10_000,
        enable_search=False,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("打开关于灯光设计的文章"),
        (),
        value,
        MessagePerformanceTracker(),
        start_home_action=start_home,
    )

    assert {lane for lane, _ in llm.calls} == {"recall", "home"}
    assert home_intents == []
    assert value.content == "普通回答。"


@pytest.mark.asyncio
async def test_cancelled_loser_late_packet_cannot_mutate_runtime() -> None:
    llm = LateLosingRecallClient()
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=FakeRecallService(),
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("杭州下周有什么活动"),
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert llm.recall_cancelled.is_set()
    assert value.content == "联网回答。最终补充。"
    assert value.flush_offset == len("联网回答。")
    assert value.pet_emotion is PetEmotion.SHY
    assert "败选" not in value.content


@pytest.mark.asyncio
async def test_failed_higher_gate_degrades_to_next_lane() -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        web='{"w":1,"m":"happy","a":"W。","q":"query"}',
        home="[",
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("打开客厅灯并查一下天气"),
        (),
        value,
        MessagePerformanceTracker(),
        start_home_action=lambda _intent: None,
    )

    assert value.content == "W。最终补充。"
    assert value.pet_emotion is PetEmotion.HAPPY


@pytest.mark.asyncio
@pytest.mark.parametrize("search_failure", [False, True], ids=["no-context", "failed"])
async def test_unavailable_web_search_completes_waiting_transition(
    search_failure: bool,
) -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        web=('{"w":1,"m":"neutral","a":"我去看看最新情况。","q":"台湾明天天气预报"}'),
        search='{"searchContext":""}',
        search_error=(RuntimeError("search unavailable") if search_failure else None),
        continuation='{"append":"这次没有找到可靠的天气信息，我先不乱猜。"}',
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("明天台湾天气怎么样？"),
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert value.content == (
        "我去看看最新情况。这次没有找到可靠的天气信息，我先不乱猜。"
    )
    assert [lane for lane, _messages in llm.calls].count("continuation") == 1


@pytest.mark.asyncio
async def test_waiting_web_route_fails_after_two_continuation_provider_errors() -> None:
    llm = SequencedContinuationLaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        web='{"w":1,"m":"neutral","a":"明天的天气……","q":"台湾明天天气"}',
        search='{"searchContext":""}',
        continuations=(
            RuntimeError("provider unavailable"),
            RuntimeError("provider unavailable"),
        ),
    )

    with pytest.raises(AnswerGenerationError):
        await ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=10_000,
            enable_search=True,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("明天台湾天气怎么样？"),
            (),
            runtime(),
            MessagePerformanceTracker(),
        )

    assert [lane for lane, _messages in llm.calls].count("continuation") == 2


@pytest.mark.asyncio
async def test_waiting_home_query_rejects_two_empty_continuations() -> None:
    llm = SequencedContinuationLaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        home='{"h":1,"t":"query","m":"neutral","a":"客厅灯的状态……"}',
        continuations=('{"append":""}', '{"append":""}'),
    )
    outcome = HomeRunOutcome(
        intent=HomeIntentType.QUERY,
        status=HomeActionStatus.SUCCEEDED,
        query=ResolvedQueryScope(entities=()),
    )

    with pytest.raises(AnswerGenerationError):
        await ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=10_000,
            enable_search=False,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("客厅灯开着吗？"),
            (),
            runtime(),
            MessagePerformanceTracker(),
            start_home_action=lambda _intent: FakeHomeRunHandle(outcome),
        )

    assert [lane for lane, _messages in llm.calls].count("continuation") == 2


@pytest.mark.asyncio
async def test_malformed_home_lane_after_intent_never_starts_home_work() -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        home='{"h":1,"t":"control","m":',
    )
    value = runtime()
    home_intents: list[HomeIntentType] = []

    with pytest.raises(AnswerGenerationError):
        await ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=10_000,
            enable_search=False,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("打开客厅灯"),
            (),
            value,
            MessagePerformanceTracker(),
            start_home_action=home_intents.append,
        )

    assert home_intents == []


@pytest.mark.asyncio
async def test_long_lane_marks_the_complete_a_without_including_continuation() -> None:
    transition = "这是一段超过普通分块长度的完整前置回答，用来验证语义边界。"
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        web=('{"w":1,"m":"neutral","a":"' + transition + '","q":"杭州明天天气"}'),
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=10_000,
        enable_search=True,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("杭州明天天气怎么样"),
        (),
        value,
        MessagePerformanceTracker(),
    )

    assert value.content == transition + "最终补充。"
    assert value.flush_offset == len(transition)


@pytest.mark.asyncio
async def test_web_transition_is_flushable_before_search_completes() -> None:
    llm = BlockingSearchLaneChatClient()
    value = runtime()
    transition = "我去看看最新情况。"
    generation = asyncio.create_task(
        ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=10_000,
            enable_search=True,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("明天台湾呢"),
            (),
            value,
            MessagePerformanceTracker(),
        )
    )

    await asyncio.wait_for(llm.search_started.wait(), timeout=1)

    assert not generation.done()
    assert value.content == transition
    assert value.flush_offset == len(transition)

    llm.search_release.set()
    await generation
    assert value.content == transition + "最终补充。"


@pytest.mark.asyncio
async def test_selected_task_commit_blocks_all_public_fields_until_release() -> None:
    content = "帮我调查这个故障"
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        web='{"w":0}',
        task=('{"t":1,"m":"happy","a":"我先接下这个任务。"}'),
    )
    routes = RecordingTaskRouteService(block=True)
    value = runtime()
    generation = asyncio.create_task(
        ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=1000,
            enable_search=True,
            task_route_service=routes,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages(content),
            (),
            value,
            MessagePerformanceTracker(),
            source_message_id=SOURCE_ID,
            agent_target=resolved_agent_target(),
            task_lane_context=TaskLaneContext.empty(),
        )
    )

    await asyncio.wait_for(routes.entered.wait(), timeout=1)

    assert value.content == ""
    assert value.flush_offset == 0
    assert value.pet_emotion is PetEmotion.NEUTRAL
    assert value.pet_metadata_revision == 0

    routes.release.set()
    await generation

    assert value.content == "我先接下这个任务。"
    assert value.flush_offset == len(value.content)
    assert value.pet_emotion is PetEmotion.HAPPY
    assert routes.calls[0]["source_message_id"] == SOURCE_ID
    assert routes.calls[0]["confirmation_message_id"] == ANSWER_ID


@pytest.mark.asyncio
async def test_new_task_waits_for_optional_c_or_object_close() -> None:
    llm = BlockingTaskCompletionClient()
    routes = RecordingTaskRouteService()
    value = runtime()
    generation = asyncio.create_task(
        ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=1000,
            enable_search=False,
            task_route_service=routes,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("帮我调查这个故障"),
            (),
            value,
            MessagePerformanceTracker(),
            source_message_id=SOURCE_ID,
            agent_target=resolved_agent_target(),
            task_lane_context=TaskLaneContext.empty(),
        )
    )

    await asyncio.wait_for(llm.partial_consumed.wait(), timeout=1)

    assert routes.calls == []
    assert generation.done() is False

    llm.release.set()
    await generation

    assert len(routes.calls) == 1
    assert routes.calls[0]["kind"] is AgentTaskDecisionKind.NEW_TASK


@pytest.mark.asyncio
async def test_recall_selection_publishes_first_delta_before_full_lane_completion() -> (
    None
):
    llm = BlockingRecallCompletionClient()
    value = runtime()
    generation = asyncio.create_task(
        ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=1000,
            enable_search=False,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("你好"),
            (),
            value,
            MessagePerformanceTracker(),
        )
    )

    await asyncio.wait_for(llm.first_delta_sent.wait(), timeout=1)
    async with value.condition:
        await asyncio.wait_for(
            value.condition.wait_for(lambda: value.content == "首段"), timeout=1
        )

    assert generation.done() is False
    assert value.pet_emotion is PetEmotion.HAPPY

    llm.release.set()
    await generation
    assert value.content == "首段尾段"


@pytest.mark.asyncio
async def test_home_stays_above_task_and_unselected_task_never_commits() -> None:
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        web='{"w":1,"m":"neutral","a":"W。","q":"query"}',
        home='{"h":1,"t":"control","m":"happy","a":"H。"}',
        task=('{"t":1,"m":"shy","a":"T 不应公开。"}'),
    )
    routes = RecordingTaskRouteService()
    value = runtime()
    home_calls: list[HomeIntentType] = []

    def start_home(intent: HomeIntentType) -> FakeHomeRunHandle:
        home_calls.append(intent)
        return FakeHomeRunHandle(
            HomeRunOutcome(intent=intent, status=HomeActionStatus.SUCCEEDED)
        )

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=True,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("帮我执行任务：打开客厅灯"),
        (),
        value,
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=TaskLaneContext.empty(),
        start_home_action=start_home,
    )

    assert value.content == "H。"
    assert value.pet_emotion is PetEmotion.HAPPY
    assert routes.calls == []
    assert home_calls == [HomeIntentType.CONTROL]
    assert {lane for lane, _ in llm.calls} == {"recall", "web", "home", "task"}


@pytest.mark.asyncio
async def test_web_stays_above_task_and_rejected_web_preserves_task_route() -> None:
    target = resolved_agent_target()
    for web_response, expected, expected_commits in (
        (
            '{"w":1,"m":"happy","a":"W。","q":"query"}',
            "W。最终补充。",
            0,
        ),
        ('{"w":0}', "T。", 1),
    ):
        routes = RecordingTaskRouteService()
        value = runtime()
        llm = LaneChatClient(
            recall='{"m":"neutral","a":"R。","r":0,"e":0}',
            web=web_response,
            task='{"t":1,"m":"shy","a":"T。"}',
        )

        await ParallelAnswerGenerationPipeline(
            llm,
            recall_service=None,
            recall_context_token_budget=1000,
            enable_search=True,
            task_route_service=routes,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("帮我调查这个任务"),
            (),
            value,
            MessagePerformanceTracker(),
            source_message_id=SOURCE_ID,
            agent_target=target,
            task_lane_context=TaskLaneContext.empty(),
        )

        assert value.content == expected
        assert len(routes.calls) == expected_commits
        assert ("search" in {lane for lane, _ in llm.calls}) is (expected_commits == 0)


@pytest.mark.asyncio
async def test_invalid_ambiguous_task_decision_falls_back_without_commit() -> None:
    content = "继续任务"
    context = frozen_task_context(
        (
            task_candidate(AgentRunStatus.COMPLETED, seed=300, title="任务 A"),
            task_candidate(AgentRunStatus.COMPLETED, seed=400, title="任务 B"),
        ),
        content=content,
    )
    routes = RecordingTaskRouteService()
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        LaneChatClient(
            recall='{"m":"neutral","a":"R。","r":0,"e":0}',
            task=('{"t":2,"m":"neutral","a":"你指的是任务 A 还是任务 B？"}'),
        ),
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        value,
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert value.content == "R。"
    assert routes.calls == []


@pytest.mark.asyncio
async def test_selected_task_commit_failure_is_closed_without_lane_fallback() -> None:
    routes = RecordingTaskRouteService(failure=RuntimeError("CAS lost"))
    value = runtime()

    with pytest.raises(AnswerGenerationError):
        await ParallelAnswerGenerationPipeline(
            LaneChatClient(
                recall='{"m":"neutral","a":"R。","r":0,"e":0}',
                web='{"w":0}',
                task=('{"t":1,"m":"happy","a":"成功确认不应发布。"}'),
            ),
            recall_service=None,
            recall_context_token_budget=1000,
            enable_search=True,
            task_route_service=routes,
        ).generate(
            FISH_ID,
            ANSWER_ID,
            messages("帮我调查这个任务"),
            (),
            value,
            MessagePerformanceTracker(),
            source_message_id=SOURCE_ID,
            agent_target=resolved_agent_target(),
            task_lane_context=TaskLaneContext.empty(),
        )

    assert len(routes.calls) == 1
    assert value.content == ""
    assert value.flush_offset == 0
    assert value.pet_emotion is PetEmotion.NEUTRAL
    assert value.pet_metadata_revision == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "has_target"),
    [("帮我调查这个任务", False), ("你好，今天心情怎么样？", True)],
)
async def test_cheap_gate_does_not_start_task_lane_without_both_signal_and_target(
    content: str,
    has_target: bool,
) -> None:
    routes = RecordingTaskRouteService()
    llm = LaneChatClient(recall='{"m":"neutral","a":"R。","r":0,"e":0}')

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        runtime(),
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target() if has_target else None,
        task_lane_context=TaskLaneContext.empty(),
    )

    assert [lane for lane, _ in llm.calls] == ["recall"]
    assert routes.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "has_recent_agent_reply", "expected_mode"),
    [
        (AgentRunStatus.RUNNING, False, AgentTaskRouteMode.STEER),
        (AgentRunStatus.COMPLETED, True, AgentTaskRouteMode.NEW_RUN),
    ],
)
async def test_task_focus_opens_t_lane_for_keyword_free_supplement(
    status: AgentRunStatus,
    has_recent_agent_reply: bool,
    expected_mode: AgentTaskRouteMode,
) -> None:
    content = "还要好吃的"
    context = frozen_task_context(
        (task_candidate(status, seed=440),),
        content=content,
        has_recent_agent_reply=has_recent_agent_reply,
    )
    routes = RecordingTaskRouteService()
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"普通回答。","r":0,"e":0}',
        task='{"t":1,"m":"happy","a":"我把这个要求补充进去。","c":0}',
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        value,
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert context.candidates[0].mode is expected_mode
    assert {lane for lane, _ in llm.calls} == {"recall", "task"}
    assert value.content == "我把这个要求补充进去。"
    assert routes.calls[0]["candidate_indexes"] == (0,)


@pytest.mark.asyncio
async def test_task_route_keeps_candidate_when_closing_brace_is_its_own_chunk() -> None:
    content = "中秋这几天你都看一下"
    context = frozen_task_context(
        (task_candidate(AgentRunStatus.COMPLETED, seed=444),),
        content=content,
        has_recent_agent_reply=True,
    )
    routes = RecordingTaskRouteService()
    llm = SplitTaskLaneChatClient(
        recall='{"m":"neutral","a":"普通回答。","r":0,"e":0}',
        task_chunks=(
            '{"t":1,"m":"neutral","a":"我来查。","c":0',
            "}",
        ),
    )

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        runtime(),
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert routes.calls[0]["candidate_indexes"] == (0,)


@pytest.mark.asyncio
async def test_stale_terminal_task_does_not_permanently_open_t_lane() -> None:
    content = "还要好吃的"
    context = frozen_task_context(
        (task_candidate(AgentRunStatus.COMPLETED, seed=445),),
        content=content,
    )
    routes = RecordingTaskRouteService()
    llm = LaneChatClient(recall='{"m":"neutral","a":"普通回答。","r":0,"e":0}')

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        runtime(),
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert [lane for lane, _ in llm.calls] == ["recall"]
    assert routes.calls == []


@pytest.mark.asyncio
async def test_recent_task_focus_probes_but_pronunciation_stays_in_recall() -> None:
    content = "obsdin怎么读"
    context = frozen_task_context(
        (task_candidate(AgentRunStatus.COMPLETED, seed=446),),
        content=content,
        has_recent_agent_reply=True,
    )
    routes = RecordingTaskRouteService()
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"Obsidian 读作“哦布西迪恩”。","r":0,"e":0}',
        task='{"t":0,"m":"neutral","a":""}',
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        value,
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert {lane for lane, _ in llm.calls} == {"recall", "task"}
    assert value.content == "Obsidian 读作“哦布西迪恩”。"
    assert routes.calls == []


@pytest.mark.asyncio
async def test_candidate_task_missing_c_remains_a_new_task() -> None:
    content = "对了，抖音的也看下"
    context = frozen_task_context(
        (task_candidate(AgentRunStatus.RUNNING, seed=447),),
        content=content,
        has_recent_agent_reply=True,
    )
    routes = RecordingTaskRouteService()
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        LaneChatClient(
            recall='{"m":"neutral","a":"普通回答。","r":0,"e":0}',
            task='{"t":1,"m":"happy","a":"我同步查看抖音热点。"}',
        ),
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        value,
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert len(routes.calls) == 1
    assert routes.calls[0]["kind"] is AgentTaskDecisionKind.NEW_TASK
    assert routes.calls[0]["candidate_indexes"] == ()
    assert value.content == "我同步查看抖音热点。"


@pytest.mark.asyncio
async def test_invalid_task_candidate_array_falls_back_to_recall() -> None:
    content = "那盘龙呢"
    context = frozen_task_context(
        (
            task_candidate(AgentRunStatus.COMPLETED, seed=448),
            task_candidate(AgentRunStatus.COMPLETED, seed=458),
        ),
        content=content,
        has_recent_agent_reply=True,
    )
    routes = RecordingTaskRouteService()
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        LaneChatClient(
            recall='{"m":"neutral","a":"《盘龙》的情况是……","r":0,"e":0}',
            task='{"t":1,"m":"neutral","a":"我继续处理。","c":[0,1]}',
        ),
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages(content),
        (),
        value,
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert value.content == "《盘龙》的情况是……"
    assert routes.calls == []


@pytest.mark.asyncio
async def test_pending_interaction_focus_starts_task_lane_for_short_reply() -> None:
    record = task_candidate(AgentRunStatus.WAITING_USER, seed=450)
    interaction = AgentInteraction(
        id=UUID(int=460),
        run_id=record.run.id,
        fish_id=record.run.fish_id,
        session_id=record.run.session_id,
        kind=AgentInteractionKind.INPUT,
        request_message_id=UUID(int=461),
        response_message_id=None,
        next_run_id=None,
        provider_data=None,
        created_at=record.run.created_at,
        resolved_at=None,
        closed_at=None,
    )
    record = replace(
        record,
        interaction=interaction,
        interaction_request_content="多久后重新检查内存？",
        interaction_request_session_id=record.task.session_id,
    )
    context = frozen_task_context((record,), content="立刻")
    routes = RecordingTaskRouteService()
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"普通回答。","r":0,"e":0}',
        task=('{"t":1,"m":"neutral","a":"好的，我马上检查。","c":0}'),
    )
    value = runtime()

    await ParallelAnswerGenerationPipeline(
        llm,
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    ).generate(
        FISH_ID,
        ANSWER_ID,
        messages("立刻"),
        (),
        value,
        MessagePerformanceTracker(),
        source_message_id=SOURCE_ID,
        agent_target=resolved_agent_target(),
        task_lane_context=context,
    )

    assert {lane for lane, _ in llm.calls} == {"recall", "task"}
    assert value.content == "好的，我马上检查。"
    assert routes.calls[0]["candidate_indexes"] == (0,)


@pytest.mark.asyncio
async def test_exact_hash_includes_target_candidate_mode_and_policy_fingerprint() -> (
    None
):
    content = "帮我继续这个任务"
    routes = RecordingTaskRouteService()
    pipeline = ParallelAnswerGenerationPipeline(
        LaneChatClient(recall='{"m":"neutral","a":"R。","r":0,"e":0}'),
        recall_service=None,
        recall_context_token_budget=1000,
        enable_search=False,
        task_route_service=routes,
    )
    first_context = frozen_task_context(
        (task_candidate(AgentRunStatus.COMPLETED, seed=500),),
        content=content,
    )
    changed_internal_id = frozen_task_context(
        (task_candidate(AgentRunStatus.COMPLETED, seed=600),),
        content=content,
    )
    changed_policy = frozen_task_context(
        (task_candidate(AgentRunStatus.COMPLETED, seed=500),),
        content=content,
        policy_version="agent-task-routing-v2",
    )
    changed_recent_reply = frozen_task_context(
        (task_candidate(AgentRunStatus.COMPLETED, seed=500),),
        content=content,
        has_recent_agent_reply=True,
    )
    changed_mode = frozen_task_context(
        (task_candidate(AgentRunStatus.RUNNING, seed=500),),
        content="把任务改成另一种实现",
    )
    target = resolved_agent_target(seed=700)

    def exact(
        agent_target: AgentResolvedTarget,
        context: TaskLaneContext,
        *,
        message_content: str = content,
    ) -> str:
        return pipeline.initial_input_hash(
            messages(message_content),
            enable_home=False,
            agent_target=agent_target,
            task_lane_context=context,
        )

    hashes = {
        exact(target, first_context),
        exact(resolved_agent_target(seed=701), first_context),
        exact(target, changed_internal_id),
        exact(target, changed_policy),
        exact(target, changed_recent_reply),
        exact(target, changed_mode, message_content="把任务改成另一种实现"),
    }

    assert len(hashes) == 6
    prepared = pipeline.prepare_initial(
        FISH_ID,
        messages(content),
        (),
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        enable_home=False,
        agent_target=target,
        task_lane_context=first_context,
    )
    await prepared.initial_ready()
    assert prepared.input_hash == exact(target, first_context)
    await prepared.cancel()
    assert pipeline.semantic_initial_input_hash(
        messages(content),
        enable_home=False,
        agent_target=target,
        task_lane_context=first_context,
    ) == pipeline.semantic_initial_input_hash(
        messages(content),
        enable_home=False,
        agent_target=resolved_agent_target(seed=701),
        task_lane_context=changed_policy,
    )
