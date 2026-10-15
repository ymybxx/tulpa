import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.answer.context import (
    ChatMessage,
    build_web_search_context_messages,
    parse_search_context,
)
from fish_memory.answer.coordinator import AnswerCoordinator
from fish_memory.answer.generation import latest_user_content
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.recall.models import (
    RecalledFact,
    RecallExecutionTrace,
    RecallQueryPlan,
    RecallResult,
    RecallSourceRole,
)

from .fakes import (
    DEFAULT_SESSION_ID,
    FakeChatClient,
    FakePerformanceStore,
    FakeStore,
    fake_request_payload,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = DEFAULT_SESSION_ID
USER_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000201")


def store_message(
    fish_id: UUID,
    session_id: UUID,
    sequence: int,
    role: Role,
    content: str,
) -> Message:
    return Message(
        id=UUID(int=sequence + 10),
        fish_id=fish_id,
        session_id=session_id,
        sequence=sequence,
        role=role,
        content=content,
        status=(
            MessageStatus.INITIALIZED if role is Role.USER else MessageStatus.COMPLETED
        ),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )


def test_latest_user_content_excludes_chat_time_marker() -> None:
    assert (
        latest_user_content(
            [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": (
                        "<message_time>2026-07-29 07:15:00+00:00</message_time>\n"
                        "What was my previous occupation?"
                    ),
                },
            ]
        )
        == "What was my previous occupation?"
    )


def test_web_search_context_prompt_and_response_are_strict() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "今天有什么新闻？"},
    ]

    prompt = build_web_search_context_messages(messages)

    assert prompt[-1]["role"] == "user"
    assert '"searchContext"' in prompt[-1]["content"]
    assert "只返回 JSON 对象" in prompt[-1]["content"]
    assert "当前问题" in prompt[-1]["content"]
    assert parse_search_context('{"searchContext":"最新消息。"}') == "最新消息。"


def test_web_search_context_prompt_does_not_mix_local_recall() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "这件事现在有什么新进展？"},
    ]

    prompt = build_web_search_context_messages(messages)

    assert "用户以前养过一条金鱼" not in prompt[-1]["content"]
    assert "searchContext" in prompt[-1]["content"]
    assert "当前问题" in prompt[-1]["content"]
    assert parse_search_context('{"searchContext":"实时事实"}') == "实时事实"


def empty_result(depth: int) -> RecallResult:
    return RecallResult(
        facts=(),
        trace=RecallExecutionTrace(
            depth=depth,
            degraded=False,
            plan=None,
            plan_error=None,
            duration_ms={},
        ),
    )


def result_with_fact(depth: int) -> RecallResult:
    fact_id = UUID(int=301)
    return RecallResult(
        facts=(
            RecalledFact(
                id=fact_id,
                source_message_id=UUID(int=201),
                source_role=Role.USER,
                text="用户以前养过一条金鱼",
                occurred_start=None,
                occurred_end=None,
                mentioned_at=datetime(2026, 7, 1, tzinfo=UTC),
                entities=("金鱼",),
                matched_by=("semantic",),
                semantic_score=0.9,
                rank_score=None,
            ),
        ),
        trace=RecallExecutionTrace(
            depth=depth,
            degraded=False,
            plan=None,
            plan_error=None,
            duration_ms={},
        ),
        direct_fact_ids=(fact_id,),
    )


def result_with_fact_and_plan(
    depth: int,
    source_role: RecallSourceRole = RecallSourceRole.EITHER,
) -> RecallResult:
    result = result_with_fact(depth)
    return RecallResult(
        facts=result.facts,
        trace=RecallExecutionTrace(
            depth=depth,
            degraded=False,
            plan=RecallQueryPlan.model_validate(
                {
                    "semanticQuery": "用户以前养过什么",
                    "sourceRole": source_role.value,
                    "time": {"start": None, "end": None},
                }
            ),
            plan_error=None,
            duration_ms={},
        ),
        direct_fact_ids=result.direct_fact_ids,
    )


class FakeRecallService:
    def __init__(
        self,
        result: RecallResult,
        *,
        blocked: bool = False,
    ) -> None:
        self.result = result
        self.calls: list[tuple[UUID, str, int]] = []
        self.entity_hint_calls: list[tuple[str, ...]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        if not blocked:
            self.release.set()

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
        del planning_content, tracker
        self.calls.append((fish_id, content, depth))
        self.entity_hint_calls.append(tuple(entity_hints))
        self.conversation_context = tuple(conversation_context)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.result


class FailingRecallService:
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
        del (
            fish_id,
            content,
            depth,
            planning_content,
            entity_hints,
            conversation_context,
            tracker,
        )
        raise RuntimeError("recall unavailable")


class ScriptedChatClient:
    def __init__(
        self,
        *,
        initial_chunks: list[str | ChatStreamChunk],
        continuation_chunks: list[str | ChatStreamChunk],
        web_chunks: list[str | ChatStreamChunk] | None = None,
        home_chunks: list[str | ChatStreamChunk] | None = None,
        search_chunks: list[str | ChatStreamChunk] | None = None,
        block_initial: bool = False,
    ) -> None:
        self.initial_chunks = initial_chunks
        self.continuation_chunks = continuation_chunks
        self.web_chunks = web_chunks or ['{"w":0}']
        self.home_chunks = home_chunks or ['{"h":0}']
        self.search_chunks = search_chunks or []
        self.initial_calls: list[Sequence[ChatMessage]] = []
        self.web_calls: list[Sequence[ChatMessage]] = []
        self.home_calls: list[Sequence[ChatMessage]] = []
        self.continuation_calls: list[Sequence[ChatMessage]] = []
        self.search_calls: list[Sequence[ChatMessage]] = []
        self.search_flags: list[bool] = []
        self.force_search_flags: list[bool] = []
        self.search_started = asyncio.Event()
        self.search_release = asyncio.Event()
        self.search_cancelled = False
        self.block_search = False
        self.initial_release = asyncio.Event()
        if not block_initial:
            self.initial_release.set()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        self.search_flags.append(enable_search)
        self.force_search_flags.append(force_search)
        chunks: list[str | ChatStreamChunk]
        wait_for_initial_release = False
        if force_search:
            self.search_calls.append(messages)
            chunks = self.search_chunks
        elif messages[0]["content"].startswith("# R 通道"):
            self.initial_calls.append(messages)
            chunks = self.initial_chunks
            wait_for_initial_release = True
        elif messages[0]["content"].startswith("# W 通道"):
            self.web_calls.append(messages)
            chunks = self.web_chunks
        elif messages[0]["content"].startswith("# H 通道"):
            self.home_calls.append(messages)
            chunks = self.home_chunks
        else:
            self.continuation_calls.append(messages)
            chunks = self.continuation_chunks
        for chunk in chunks:
            yield (
                chunk
                if isinstance(chunk, ChatStreamChunk)
                else ChatStreamChunk(content=chunk)
            )
        if force_search:
            self.search_started.set()
            if self.block_search:
                try:
                    await self.search_release.wait()
                except asyncio.CancelledError:
                    self.search_cancelled = True
                    raise
        if wait_for_initial_release:
            await self.initial_release.wait()

    def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages

        async def fail() -> AsyncIterator[ChatStreamChunk]:
            raise AssertionError("continuation must use JSON mode")
            yield ChatStreamChunk()

        return fail()

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return fake_request_payload(messages)

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        parameters: dict[str, JsonValue] = {
            "stream": True,
            "response_format": {"type": "json_object"},
        }
        if enable_search or force_search:
            parameters["enable_search"] = True
        if force_search:
            parameters["search_options"] = {"forced_search": True}
        return {
            "model": "fake",
            "messages": [dict(message) for message in messages],
            "parameters": parameters,
        }

    async def close(self) -> None:
        return None


async def run_answer(coordinator: AnswerCoordinator) -> None:
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        1,
        message_id=USER_MESSAGE_ID,
        tracker=MessagePerformanceTracker(),
    )
    task = coordinator.get_runtime(FISH_ID, SESSION_ID).llm_task
    assert task is not None
    await task


@pytest.mark.asyncio
async def test_depth_zero_does_not_call_recall() -> None:
    recall = FakeRecallService(empty_result(0))
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        FakeChatClient(["你好"]),
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert recall.calls == []
    assert coordinator.get_runtime(FISH_ID, SESSION_ID).content == "你好"


@pytest.mark.asyncio
async def test_lean_depth_zero_derives_context_complete_for_diagnostics() -> None:
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        ScriptedChatClient(
            initial_chunks=['{"m":"neutral","a":"你好","r":0,"e":0}'],
            continuation_chunks=[],
        ),
        performance_store=performance_store,
        system_prompt="system",
    )

    await run_answer(coordinator)
    await coordinator.close()

    step = next(
        step
        for step in performance_store.writes[0][1]
        if step.name == "answer.recall_depth"
    )
    assert step.output_payload["context_complete"] is True


@pytest.mark.asyncio
async def test_web_search_without_recall_uses_non_search_continuation() -> None:
    llm = ScriptedChatClient(
        initial_chunks=['{"m":"neutral","a":"普通回答。","r":0,"e":0}'],
        web_chunks=['{"w":1,"m":"neutral","a":"我先了解一下。","q":"最新天气"}'],
        search_chunks=['{"searchContext":"最新情况是晴天。"}'],
        continuation_chunks=['{"append":"最新情况是晴天。"}'],
    )
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        performance_store=performance_store,
        answer_enable_search=True,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert (
        coordinator.get_runtime(FISH_ID, SESSION_ID).content
        == "我先了解一下。最新情况是晴天。"
    )
    assert llm.force_search_flags.count(True) == 1
    assert llm.search_flags == [False] * 4
    assert len(llm.search_calls) == 1
    assert all(
        "# 当前用户消息（应用已进行 JSON 转义）" not in message["content"]
        for message in llm.search_calls[0]
    )
    await coordinator.close()
    search_steps = [
        step
        for step in performance_store.writes[0][1]
        if step.name == "llm.web_search.context"
    ]
    assert len(search_steps) == 1
    assert search_steps[0].input_payload["parameters"]["enable_search"] is True
    assert search_steps[0].input_payload["parameters"]["search_options"] == {
        "forced_search": True
    }


@pytest.mark.asyncio
async def test_web_lane_priority_does_not_run_recall_when_both_lanes_accept() -> None:
    recall = FakeRecallService(result_with_fact(1))
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"我记得这件事。","r":1,"q":"用户过去提到的天气","e":0}',
        ],
        web_chunks=['{"w":1,"m":"neutral","a":"我先查一下。","q":"最新天气"}'],
        search_chunks=['{"searchContext":"最新天气是晴天。"}'],
        continuation_chunks=['{"append":"综合来看，是晴天。"}'],
    )
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        answer_enable_search=True,
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert coordinator.get_runtime(FISH_ID, SESSION_ID).content == (
        "我先查一下。综合来看，是晴天。"
    )
    assert len(llm.search_calls) == 1
    assert len(llm.continuation_calls) == 1
    assert "最新天气是晴天" in llm.continuation_calls[0][-1]["content"]
    assert "用户以前养过一条金鱼" not in llm.continuation_calls[0][-1]["content"]
    assert recall.calls == []
    await coordinator.close()


@pytest.mark.asyncio
async def test_web_lane_uses_continuation_without_recall_context() -> None:
    recall = FakeRecallService(empty_result(2))
    llm = ScriptedChatClient(
        initial_chunks=['{"m":"neutral","a":"普通回答。","r":0,"e":0}'],
        web_chunks=['{"w":1,"m":"neutral","a":"我先了解一下。","q":"昨天的主要新闻"}'],
        search_chunks=['{"searchContext":"昨天的主要新闻是……。"}'],
        continuation_chunks=['{"append":"我整理好了，主要新闻是……。"}'],
    )
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        answer_enable_search=True,
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert coordinator.get_runtime(FISH_ID, SESSION_ID).content == (
        "我先了解一下。我整理好了，主要新闻是……。"
    )
    assert len(llm.continuation_calls) == 1
    assert '"u":[]' in llm.continuation_calls[0][-1]["content"]
    assert "昨天的主要新闻是" in llm.continuation_calls[0][-1]["content"]
    assert recall.calls == []
    await coordinator.close()


@pytest.mark.asyncio
async def test_web_search_and_continuation_failure_marks_waiting_answer_failed() -> (
    None
):
    llm = ScriptedChatClient(
        initial_chunks=['{"m":"neutral","a":"普通回答。","r":0,"e":0}'],
        web_chunks=['{"w":1,"m":"neutral","a":"先给你一个简短回答。","q":"最新情况"}'],
        search_chunks=["["],
        continuation_chunks=[],
    )
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        answer_enable_search=True,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert (
        coordinator.get_runtime(FISH_ID, SESSION_ID).content == "先给你一个简短回答。"
    )
    assert coordinator.get_runtime(FISH_ID, SESSION_ID).status is MessageStatus.FAILED
    assert len(llm.continuation_calls) == 2
    await coordinator.close()


@pytest.mark.asyncio
async def test_interrupt_cancels_pending_web_search() -> None:
    llm = ScriptedChatClient(
        initial_chunks=['{"m":"neutral","a":"普通回答。","r":0,"e":0}'],
        web_chunks=['{"w":1,"m":"neutral","a":"先等等。","q":"最新情况"}'],
        search_chunks=[],
        continuation_chunks=[],
    )
    llm.block_search = True
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        answer_enable_search=True,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        1,
        message_id=USER_MESSAGE_ID,
        tracker=MessagePerformanceTracker(),
    )
    await asyncio.wait_for(llm.search_started.wait(), timeout=1)
    task = coordinator.get_runtime(FISH_ID, SESSION_ID).llm_task
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert llm.search_cancelled is True
    await coordinator.close()


@pytest.mark.asyncio
async def test_web_search_starts_before_recall_lane_finishes() -> None:
    llm = ScriptedChatClient(
        initial_chunks=['{"m":"neutral","a":"尚未完成'],
        web_chunks=['{"w":1,"m":"neutral","a":"我先了解一下。","q":"最新情况"}'],
        search_chunks=['{"searchContext":"最新情况。"}'],
        continuation_chunks=[],
        block_initial=True,
    )
    llm.block_search = True
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        answer_enable_search=True,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        1,
        message_id=USER_MESSAGE_ID,
        tracker=MessagePerformanceTracker(),
    )
    await asyncio.wait_for(llm.search_started.wait(), timeout=1)

    task = coordinator.get_runtime(FISH_ID, SESSION_ID).llm_task
    assert task is not None
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.search_cancelled is True
    await coordinator.close()


@pytest.mark.asyncio
async def test_end_flag_is_available_to_continuous_conversation_clients() -> None:
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        ScriptedChatClient(
            initial_chunks=['{"m":"neutral","a":"先休息吧。","r":0,"e":1}'],
            continuation_chunks=[],
        ),
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert coordinator.get_runtime(FISH_ID, SESSION_ID).end_conversation is True
    await coordinator.close()


@pytest.mark.asyncio
async def test_r_end_flag_stays_true_after_append_only_continuation() -> None:
    recall = FakeRecallService(result_with_fact(1))
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        ScriptedChatClient(
            initial_chunks=[
                '{"m":"neutral","a":"我想想。","r":1,"q":"用户记忆","e":1}'
            ],
            continuation_chunks=['{"append":""}'],
        ),
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert coordinator.get_runtime(FISH_ID, SESSION_ID).end_conversation is True
    await coordinator.close()


@pytest.mark.asyncio
async def test_recall_starts_after_lane_query_before_generation_finishes() -> None:
    recall = FakeRecallService(empty_result(1), blocked=True)
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"我先想想。","r":1,',
            '"q":"用户记忆","e":0}',
        ],
        continuation_chunks=[],
    )
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        recall_service=recall,
        system_prompt="system",
    )

    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        1,
        message_id=USER_MESSAGE_ID,
        tracker=MessagePerformanceTracker(),
    )
    await asyncio.wait_for(recall.started.wait(), timeout=1)

    task = coordinator.get_runtime(FISH_ID, SESSION_ID).llm_task
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recall.cancelled is True


@pytest.mark.asyncio
async def test_recall_uses_current_question_and_recent_conversation() -> None:
    recall = FakeRecallService(empty_result(1))
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"我想想。","r":1,"q":"新耳机后来怎么了","e":0}'
        ],
        continuation_chunks=[],
    )
    store = FakeStore(FISH_ID)

    async def load_history(
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list:
        del before_sequence, limit
        return [
            store_message(fish_id, session_id, 0, Role.USER, "我买了一副新耳机"),
            store_message(fish_id, session_id, 1, Role.USER, "它后来怎么了？"),
        ]

    store.load_history = load_history  # type: ignore[method-assign]
    coordinator = AnswerCoordinator(
        store,
        llm,
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert recall.calls == [(FISH_ID, "新耳机后来怎么了", 1)]
    assert recall.conversation_context == (
        {"role": "user", "content": "我买了一副新耳机"},
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_recalled_facts_append_natural_continuation_to_same_message() -> None:
    recall = FakeRecallService(result_with_fact(1))
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"让我想想。","r":1,"q":"用户以前养过什么","e":0}',
        ],
        continuation_chunks=['{"append":"你以前养过一条金鱼。"}'],
    )
    store = FakeStore(FISH_ID)
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        store,
        llm,
        recall_service=recall,
        performance_store=performance_store,
        system_prompt="system",
    )

    await run_answer(coordinator)

    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.content == "让我想想。你以前养过一条金鱼。"
    assert runtime.status is MessageStatus.COMPLETED
    assert store.finish_calls[-1]["content"] == runtime.content
    continuation = llm.continuation_calls[0]
    assert continuation[-2] == {
        "role": "assistant",
        "content": "让我想想。",
    }
    assert "用户以前养过一条金鱼" in continuation[-1]["content"]
    assert continuation[-1]["role"] == "user"
    assert "只返回 JSON 对象" in continuation[-1]["content"]
    assert "不要提及记录、记忆、历史、系统或检索" in continuation[-1]["content"]
    assert "不要说“根据我的记录”" in continuation[-1]["content"]
    assert '"occurredStart"' not in continuation[-1]["content"]
    assert '"occurredEnd"' not in continuation[-1]["content"]
    assert '"u":[{"t":"f","x":"用户以前养过一条金鱼"' in continuation[-1]["content"]
    assert '"m":"2026-07-01T00+00:00"' in continuation[-1]["content"]
    assert '"entities"' not in continuation[-1]["content"]
    assert "u 是按相关性排列的 Memory Units" in continuation[-1]["content"]
    assert "m 是 mentioned hour" in continuation[-1]["content"]
    assert "时间精确到小时，使用用户当前系统时区并带偏移" in continuation[-1]["content"]
    assert "Reserve Relation 的 f 只含 Anchor Facts" in continuation[-1]["content"]
    assert "f 是 Relation 的权威成员 Facts" in continuation[-1]["content"]
    assert "s 是 Relation 概览 Summary" in continuation[-1]["content"]
    assert "s 与 f 冲突或 s 遗漏成员命题时，以 f 为准；" in continuation[-1]["content"]
    assert "priority" not in continuation[-1]["content"]
    await coordinator.close()
    assert any(
        step.name == "recall.context_build" for step in performance_store.writes[0][1]
    )


@pytest.mark.asyncio
async def test_english_recall_continuation_inserts_word_boundary_space() -> None:
    recall = FakeRecallService(result_with_fact(1))
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        ScriptedChatClient(
            initial_chunks=[
                '{"m":"neutral","a":"I am trying to remember.","r":1,'
                '"q":"where the user met Sophia","e":0}'
            ],
            continuation_chunks=[
                '{"append":"Actually, you met Sophia at a coffee shop."}'
            ],
        ),
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.content == (
        "I am trying to remember. Actually, you met Sophia at a coffee shop."
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_records_first_packet_decision_and_first_continuation_append() -> None:
    recall = FakeRecallService(result_with_fact(1))
    llm = ScriptedChatClient(
        initial_chunks=[
            ChatStreamChunk(content="", input_tokens=20),
            '{"m":"neutral","a":"让我想想。","r":1,"q":"用户以前养过什么","e":0}',
        ],
        continuation_chunks=[
            '{"append":"你以前养过',
            '一条金鱼。"}',
        ],
    )
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        recall_service=recall,
        performance_store=performance_store,
        system_prompt="system",
    )

    await run_answer(coordinator)
    await coordinator.close()

    steps = {step.name: step for step in performance_store.writes[0][1]}
    assert (
        steps["llm.first_packet"].started_offset_us
        <= steps["llm.first_content"].started_offset_us
    )
    assert (
        steps["llm.first_packet"].duration_us <= steps["llm.first_content"].duration_us
    )
    assert (
        steps["answer.decision_ready"].started_offset_us
        <= steps["answer.first_content"].started_offset_us
    )
    assert (
        steps["answer.decision_ready"].duration_us
        <= steps["answer.first_content"].duration_us
    )
    assert steps["llm.continuation.first_content"].duration_us >= 0
    assert steps["answer.first_append"].duration_us >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_role",
    [
        RecallSourceRole.USER,
        RecallSourceRole.ASSISTANT,
        RecallSourceRole.EITHER,
    ],
)
async def test_recall_total_records_explainable_bounded_result(
    source_role: RecallSourceRole,
) -> None:
    recall = FakeRecallService(result_with_fact_and_plan(1, source_role))
    performance_store = FakePerformanceStore()
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        ScriptedChatClient(
            initial_chunks=[
                '{"m":"neutral","a":"我想想。","r":1,"q":"用户以前养过什么","e":0}'
            ],
            continuation_chunks=['{"append":"你以前养过一条金鱼。"}'],
        ),
        recall_service=recall,
        performance_store=performance_store,
        system_prompt="system",
    )

    await run_answer(coordinator)
    await coordinator.close()

    recall_step = next(
        step for step in performance_store.writes[0][1] if step.name == "recall.total"
    )
    assert recall_step.input_payload == {
        "originalQuery": "你好",
        "generatedQuery": "用户以前养过什么",
        "finalQuery": "用户以前养过什么",
        "queryOutcome": "generated",
        "depth": 1,
        "conversationContext": [],
    }
    assert recall_step.output_payload == {
        "depth": 1,
        "degraded": False,
        "plan": {
            "semanticQuery": "用户以前养过什么",
            "sourceRole": source_role.value,
            "entityHints": [],
            "time": {"start": None, "end": None},
        },
        "directFactIds": [str(UUID(int=301))],
        "channels": ["semantic"],
        "evidence": [
            {
                "factId": str(UUID(int=301)),
                "sourceMessageId": str(UUID(int=201)),
                "text": "用户以前养过一条金鱼",
                "matchedBy": ["semantic"],
                "occurredStart": None,
                "mentionedAt": "2026-07-01T00:00:00+00:00",
                "validity": None,
            }
        ],
    }


@pytest.mark.asyncio
async def test_recalled_continuation_uses_empty_append_when_no_new_content() -> None:
    recall = FakeRecallService(result_with_fact(1))
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"你以前养过一条金鱼。","r":1,'
            '"q":"用户以前养过什么","e":0}'
        ],
        continuation_chunks=['{"append":""}'],
    )
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert (
        coordinator.get_runtime(FISH_ID, SESSION_ID).content == "你以前养过一条金鱼。"
    )


@pytest.mark.asyncio
async def test_empty_recall_completes_waiting_shallow_answer_without_invention() -> (
    None
):
    recall = FakeRecallService(empty_result(1))
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"我暂时想不起来。","r":1,"q":"用户记忆","e":0}',
        ],
        continuation_chunks=[
            '{"append":"这次没有足够可靠的信息让我确定答案，我先不乱猜。"}'
        ],
    )
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        recall_service=recall,
        system_prompt="system",
    )

    await run_answer(coordinator)

    assert coordinator.get_runtime(FISH_ID, SESSION_ID).content == (
        "我暂时想不起来。这次没有足够可靠的信息让我确定答案，我先不乱猜。"
    )
    assert len(llm.continuation_calls) == 1


@pytest.mark.asyncio
async def test_new_ask_cancels_pending_recall_and_interrupts_shallow_answer() -> None:
    recall = FakeRecallService(empty_result(1), blocked=True)
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"我正在回想。","r":1,"q":"用户记忆","e":0}',
        ],
        continuation_chunks=[],
    )
    store = FakeStore(FISH_ID)
    coordinator = AnswerCoordinator(
        store,
        llm,
        recall_service=recall,
        system_prompt="system",
    )
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        1,
        message_id=USER_MESSAGE_ID,
        tracker=MessagePerformanceTracker(),
    )
    await asyncio.wait_for(recall.started.wait(), timeout=1)

    store.latest_user_sequence = 2
    await coordinator.handle_user_message(
        FISH_ID,
        SESSION_ID,
        2,
        message_id=UUID(int=202),
        tracker=MessagePerformanceTracker(),
    )

    assert recall.cancelled is True
    assert store.finish_calls[0]["content"] == "我正在回想。"
    assert store.finish_calls[0]["requested_status"] is MessageStatus.INTERRUPTED
    await coordinator.close()


@pytest.mark.asyncio
async def test_recall_and_continuation_failure_marks_waiting_answer_failed() -> None:
    llm = ScriptedChatClient(
        initial_chunks=[
            '{"m":"neutral","a":"我暂时想不起来。","r":1,"q":"用户记忆","e":0}',
        ],
        continuation_chunks=[],
    )
    coordinator = AnswerCoordinator(
        FakeStore(FISH_ID),
        llm,
        recall_service=FailingRecallService(),
        system_prompt="system",
    )

    await run_answer(coordinator)

    runtime = coordinator.get_runtime(FISH_ID, SESSION_ID)
    assert runtime.content == "我暂时想不起来。"
    assert runtime.status is MessageStatus.FAILED
    assert len(llm.continuation_calls) == 2
