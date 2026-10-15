import asyncio
import copy
from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import pytest

from fish_memory.agent.routing import TaskLaneContext
from fish_memory.answer.context import ChatMessage
from fish_memory.answer.output_sink import PromotableAnswerSink, RuntimeAnswerSink
from fish_memory.answer.parallel_generation import ParallelAnswerGenerationPipeline
from fish_memory.home.models import (
    HomeControlPlan,
    HomeIntentType,
    HomeOperation,
    HomeQueryDetail,
    HomeQueryPlan,
    HomeSelection,
    HomeStateFilter,
)
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion
from tests.answer.test_parallel_generation import (
    ANSWER_ID,
    FISH_ID,
    SOURCE_ID,
    BlockingSearchLaneChatClient,
    FakeRecallService,
    LaneChatClient,
    RecordingTaskRouteService,
    messages,
    resolved_agent_target,
    runtime,
)
from tests.home.test_scoped_service import make_service as make_home_service
from tests.home.test_scoped_service import scope

SESSION_ID = UUID(int=31)


def pipeline(llm, *, recall=None, search=False, home=None, task=None):
    return ParallelAnswerGenerationPipeline(
        llm,
        recall_service=recall,
        recall_context_token_budget=10_000,
        enable_search=search,
        home_service=home,
        task_route_service=task,
    )


async def release(prepared, public, tracker):
    await prepared.release_to_public(
        RuntimeAnswerSink(public, ANSWER_ID, tracker),
        source_message_id=SOURCE_ID,
        answer_message_id=ANSWER_ID,
        runtime=public,
        session_id=SESSION_ID,
    )


async def wait_for_public_content(public, expected):
    async with public.condition:
        await asyncio.wait_for(
            public.condition.wait_for(lambda: public.content == expected), timeout=1
        )


def assert_unpublished(public):
    assert public.content == ""
    assert public.flush_offset == 0
    assert public.pet_emotion is PetEmotion.NEUTRAL
    assert public.pet_metadata_revision == 0
    assert public.end_conversation is False


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["recall", "web"])
async def test_full_route_and_continuation_finish_privately_before_commit(route):
    llm = LaneChatClient(
        recall='{"m":"happy","a":"让我想想。","r":1,"q":"音乐服务","e":0}',
        web='{"w":1,"m":"shy","a":"我去了解。","q":"天气"}',
    )
    recall = FakeRecallService()
    runner = pipeline(llm, recall=recall, search=route == "web")
    sink, tracker, public = (
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        runtime(),
    )
    prepared = runner.prepare_initial(
        FISH_ID, messages("音乐服务或天气"), (), sink, tracker, enable_home=False
    )

    result = await asyncio.wait_for(prepared.preparation_ready(), timeout=1)
    assert result.name == route
    prefix = "让我想想。" if route == "recall" else "我去了解。"
    assert sink.snapshot().content == prefix + "最终补充。"
    assert_unpublished(public)
    before_calls = list(llm.calls)
    before_recall = list(recall.calls)
    assert [lane for lane, _ in llm.calls].count("continuation") == 1
    assert len(recall.calls) == (1 if route == "recall" else 0)
    assert [lane for lane, _ in llm.calls].count("search") == (route == "web")

    await release(prepared, public, tracker)

    assert public.content == prefix + "最终补充。"
    assert public.flush_offset == len(prefix)
    assert llm.calls == before_calls and recall.calls == before_recall
    with pytest.raises(RuntimeError, match="already continued"):
        await release(prepared, public, tracker)


class GatedRecall(FakeRecallService):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.conversations = []

    async def recall(self, *args, **kwargs):
        self.conversations.append(kwargs.get("conversation_context"))
        result = await super().recall(*args, **kwargs)
        self.entered.set()
        await self.release.wait()
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["recall", "web"])
async def test_commit_while_route_waits_adopts_the_same_request(route):
    recall = GatedRecall()
    llm = (
        BlockingSearchLaneChatClient()
        if route == "web"
        else LaneChatClient(
            recall='{"m":"neutral","a":"让我想想。","r":1,"q":"音乐服务","e":0}'
        )
    )
    entered = llm.search_started if route == "web" else recall.entered
    resume = llm.search_release if route == "web" else recall.release
    runner = pipeline(llm, recall=recall, search=route == "web")
    sink, tracker, public = (
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        runtime(),
    )
    prepared = runner.prepare_initial(
        FISH_ID,
        messages("我最近听什么或天气如何"),
        (),
        sink,
        tracker,
        enable_home=False,
    )
    await asyncio.wait_for(entered.wait(), timeout=1)
    assert not prepared.preparation_done()
    assert_unpublished(public)
    committing = asyncio.create_task(release(prepared, public, tracker))
    prefix = "我去看看最新情况。" if route == "web" else "让我想想。"
    await wait_for_public_content(public, prefix)
    assert not committing.done()
    assert len(recall.calls) == (1 if route == "recall" else 0)
    assert [lane for lane, _ in llm.calls].count("search") == (route == "web")
    resume.set()
    await asyncio.wait_for(committing, timeout=1)

    assert public.content == prefix + "最终补充。"
    assert len(recall.calls) == (1 if route == "recall" else 0)
    assert [lane for lane, _ in llm.calls].count("search") == (route == "web")
    assert [lane for lane, _ in llm.calls].count("continuation") == 1


class NonCooperativeContinuation(LaneChatClient):
    def __init__(self):
        super().__init__(
            recall='{"m":"neutral","a":"让我想想。","r":1,"q":"音乐服务","e":0}'
        )
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.late_emitted = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search=False,
        force_search=False,
    ) -> AsyncIterator[ChatStreamChunk]:
        lane = self._lane(messages, force_search=force_search)
        if lane != "continuation":
            async for chunk in super().stream_json(
                messages, enable_search=enable_search, force_search=force_search
            ):
                yield chunk
            return
        self.calls.append((lane, messages))
        self.entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
            self.late_emitted.set()
            yield ChatStreamChunk(content='{"append":"取消后才到达的内容。"}')


@pytest.mark.asyncio
@pytest.mark.parametrize("attach_public", [False, True])
async def test_cancel_fences_late_noncooperative_continuation(attach_public):
    llm = NonCooperativeContinuation()
    runner = pipeline(llm, recall=FakeRecallService())
    sink, tracker, public = (
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        runtime(),
    )
    prepared = runner.prepare_initial(
        FISH_ID, messages("最近用什么听音乐"), (), sink, tracker, enable_home=False
    )
    await asyncio.wait_for(llm.entered.wait(), timeout=1)
    committing = None
    if attach_public:
        committing = asyncio.create_task(release(prepared, public, tracker))
        await wait_for_public_content(public, "让我想想。")
    before = sink.snapshot()
    public_before = public.content

    cleanup = prepared.cancel()
    await asyncio.wait_for(llm.cancelled.wait(), timeout=1)
    assert not cleanup.done()
    llm.release.set()
    await asyncio.wait_for(cleanup, timeout=1)
    if committing is not None:
        committed = await asyncio.gather(committing, return_exceptions=True)
        assert isinstance(committed[0], (RuntimeError, asyncio.CancelledError))

    assert llm.late_emitted.is_set()
    assert sink.snapshot() == before
    assert public.content == public_before
    if not attach_public:
        assert_unpublished(public)
    with pytest.raises(RuntimeError, match="cancelled"):
        await release(prepared, public, tracker)


@pytest.mark.asyncio
@pytest.mark.parametrize("intent", [HomeIntentType.QUERY, HomeIntentType.CONTROL])
async def test_home_preparation_is_private_and_commit_does_not_repeat_work(
    intent, monkeypatch
):
    plan = (
        HomeQueryPlan(
            scope(HomeSelection.ALL), HomeStateFilter.ALL, HomeQueryDetail.FULL
        )
        if intent is HomeIntentType.QUERY
        else HomeControlPlan(scope(HomeSelection.SPECIFIC), HomeOperation.TURN_OFF)
    )
    home, devices, store, selector = make_home_service(plan)
    planner = home._scope_planner
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        home='{"h":1,"t":"' + intent.value + '","m":"happy","a":"家里的灯。"}',
    )
    runner = pipeline(llm, home=home)
    sink, tracker, public = (
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        runtime(),
    )
    prepared = runner.prepare_initial(
        FISH_ID,
        messages("查看灯状态" if intent is HomeIntentType.QUERY else "关掉灯一"),
        (),
        sink,
        tracker,
        enable_home=True,
    )
    await asyncio.wait_for(prepared.preparation_ready(), timeout=1)

    assert len(planner.calls) == 1
    assert store.created == store.updated == devices.calls == []
    assert_unpublished(public)
    assert [lane for lane, _ in llm.calls].count("continuation") == (
        intent is HomeIntentType.QUERY
    )
    buffered = sink.snapshot().content
    assert buffered == "家里的灯。" + (
        "最终补充。" if intent is HomeIntentType.QUERY else ""
    )
    before_calls = list(llm.calls)
    public_tracker = MessagePerformanceTracker()
    public.performance_tracker = public_tracker
    commit_origins = []
    start_prepared = home.start_prepared

    def record_start(*args, origin_ns=None):
        commit_origins.append(origin_ns)
        return start_prepared(*args, origin_ns=origin_ns)

    monkeypatch.setattr(home, "start_prepared", record_start)

    await release(prepared, public, public_tracker)

    assert public.content == buffered
    assert len(planner.calls) == 1 and llm.calls == before_calls
    assert len(store.created) == 1
    assert store.created[0]["source_message_id"] == SOURCE_ID
    assert commit_origins == [public_tracker.origin_ns]
    if intent is HomeIntentType.CONTROL:
        assert len(selector.calls) == 1
        assert [call[3]["entity_id"] for call in devices.calls] == ["light.one"]
    else:
        assert selector.calls == devices.calls == []
    with pytest.raises(RuntimeError, match="already continued"):
        await release(prepared, public, tracker)
    assert len(store.created) == 1
    await home.close()


@pytest.mark.asyncio
async def test_task_prepares_once_privately_and_commits_once_before_public_ack():
    routes = RecordingTaskRouteService(block=True)
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"R。","r":0,"e":0}',
        task='{"t":1,"m":"happy","a":"我来完成这件事。"}',
    )
    runner = pipeline(llm, task=routes)
    sink, tracker, public = (
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        runtime(),
    )
    prepared = runner.prepare_initial(
        FISH_ID,
        messages("帮我调查这个故障"),
        (),
        sink,
        tracker,
        enable_home=False,
        agent_target=resolved_agent_target(),
        task_lane_context=TaskLaneContext.empty(),
    )
    await asyncio.wait_for(prepared.preparation_ready(), timeout=1)
    assert len(routes.prepare_calls) == 1 and routes.calls == []
    assert_unpublished(public)
    committing = asyncio.create_task(release(prepared, public, tracker))
    await asyncio.wait_for(routes.entered.wait(), timeout=1)
    assert_unpublished(public)
    assert len(routes.prepare_calls) == len(routes.calls) == 1
    routes.release.set()
    await asyncio.wait_for(committing, timeout=1)
    assert public.content == "我来完成这件事。"
    assert routes.calls[0]["source_message_id"] == SOURCE_ID
    assert routes.calls[0]["confirmation_message_id"] == ANSWER_ID
    with pytest.raises(RuntimeError, match="already continued"):
        await release(prepared, public, tracker)
    assert len(routes.prepare_calls) == len(routes.calls) == 1


@pytest.mark.asyncio
async def test_preparation_deep_copies_messages_history_and_screen_attachment():
    recall = GatedRecall()
    llm = LaneChatClient(
        recall='{"m":"neutral","a":"让我想想。","r":1,"q":"音乐服务","e":0}'
    )
    runner = pipeline(llm, recall=recall)
    source = messages("原始当前问题")
    source[-1]["environment_image"] = {
        "data_url": "data:image/jpeg;base64,original-frame",
        "digest": "a" * 64,
        "byte_count": 14,
        "width": 640,
        "height": 360,
        "observed_at": "2026-09-02T08:30:00+00:00",
    }
    history = [{"role": "user", "content": "原始历史问题"}]
    original, original_history = copy.deepcopy(source), copy.deepcopy(history)
    sink, tracker = PromotableAnswerSink(), MessagePerformanceTracker()
    prepared = runner.prepare_initial(
        FISH_ID, source, history, sink, tracker, enable_home=False
    )
    source[0]["content"] = "changed system"
    source[-1]["content"] = "changed current message"
    source[-1]["environment_image"]["data_url"] = "data:image/jpeg;base64,changed"
    source[-1]["environment_image"]["digest"] = "b" * 64
    source.append({"role": "user", "content": "later message"})
    history[0]["content"] = "changed history"

    await asyncio.wait_for(recall.entered.wait(), timeout=1)
    assert prepared.input_context.messages == original
    assert prepared.input_context.recall_conversation == tuple(original_history)
    assert recall.conversations == [tuple(original_history)]
    assert await prepared.input_ready() == runner.initial_input_hash(
        original, enable_home=False
    )
    recall.release.set()
    await asyncio.wait_for(prepared.preparation_ready(), timeout=1)

    for lane, call in llm.calls:
        if lane in {"recall", "continuation"}:
            images = [
                message["environment_image"]
                for message in call
                if message.get("environment_image") is not None
            ]
            assert images == [original[-1]["environment_image"]]
            assert "changed" not in str(call)
    assert recall.planning_contents == ["原始当前问题"]


@pytest.mark.asyncio
async def test_cancel_stops_release_waiting_for_public_runtime_lock():
    sink, tracker, public = (
        PromotableAnswerSink(),
        MessagePerformanceTracker(),
        runtime(),
    )
    prepared = pipeline(
        LaneChatClient(recall='{"m":"happy","a":"预生成内容","r":0,"e":1}')
    ).prepare_initial(FISH_ID, messages("你好"), (), sink, tracker, enable_home=False)
    await prepared.preparation_ready()
    entered = asyncio.Event()

    class SignalingRuntimeSink(RuntimeAnswerSink):
        async def set_emotion(self, emotion):
            entered.set()
            await super().set_emotion(emotion)

    await public.condition.acquire()
    publishing = asyncio.create_task(
        prepared.release_to_public(
            SignalingRuntimeSink(public, ANSWER_ID, tracker),
            source_message_id=SOURCE_ID,
            answer_message_id=ANSWER_ID,
            runtime=public,
            session_id=SESSION_ID,
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(prepared.cancel(), timeout=1)
    finally:
        public.condition.release()
    with pytest.raises(asyncio.CancelledError):
        await publishing

    assert_unpublished(public)


@pytest.mark.asyncio
async def test_private_performance_audit_preserves_public_origin_and_full_steps():
    now_ns = 1_000_000
    private_tracker = MessagePerformanceTracker(clock=lambda: now_ns)
    recall = GatedRecall()
    sink, public = PromotableAnswerSink(), runtime()
    prepared = pipeline(
        LaneChatClient(
            recall='{"m":"neutral","a":"让我想想。","r":1,"q":"音乐服务","e":0}'
        ),
        recall=recall,
    ).prepare_initial(
        FISH_ID, messages("音乐服务"), (), sink, private_tracker, enable_home=False
    )
    await asyncio.wait_for(recall.entered.wait(), timeout=1)
    now_ns = 2_000_000
    public_tracker = MessagePerformanceTracker(clock=lambda: now_ns)
    with pytest.raises(RuntimeError, match="must settle"):
        prepared.record_preparation_performance(public_tracker)
    assert public_tracker.snapshot() == ()
    recall.release.set()
    await prepared.preparation_ready()
    assert_unpublished(public)
    assert not any(
        step.name == "answer.first_append" for step in private_tracker.snapshot()
    )

    now_ns = 3_000_000
    await release(prepared, public, public_tracker)
    prepared.record_preparation_performance(public_tracker)
    prepared.record_preparation_performance(public_tracker)
    steps = public_tracker.finish()
    by_name = {step.name: step for step in steps}
    assert public_tracker.origin_ns == 2_000_000
    assert by_name["total"].duration_us == 1_000
    assert by_name["answer.first_content"].duration_us == 1_000
    assert by_name["answer.first_append"].duration_us == 1_000
    assert sum(step.name == "voice_draft.preparation" for step in steps) == 1
    audit = by_name["voice_draft.preparation"]
    assert audit.duration_us == 0
    assert audit.output_payload["originRelativeToMessageUs"] == -1_000
    recorded = audit.output_payload["steps"]
    assert [item["stepName"] for item in recorded] == [
        step.name for step in private_tracker.snapshot()
    ]
    assert {"llm.answer.initial", "recall.total", "llm.answer.continuation"} <= {
        item["stepName"] for item in recorded
    }
    for item, step in zip(recorded, private_tracker.snapshot(), strict=True):
        assert item["startedOffsetUs"] == step.started_offset_us
        assert item["durationUs"] == step.duration_us
        assert item["inputPayload"] == step.input_payload
        assert item["outputPayload"] == step.output_payload
        assert item["error"] == step.error
