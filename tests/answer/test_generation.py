import asyncio
from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.answer.generation import (
    AnswerGenerationError,
    AnswerGenerationPipeline,
    _parse_continuation_append,
)
from fish_memory.answer.runtime import FishRuntime
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.models import MessageStatus
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.recall.models import RecallExecutionTrace, RecallResult

from .fakes import (
    FakeChatClient,
    InvalidAfterVisibleJsonClient,
    MalformedThenValidJsonClient,
    fake_json_request_payload,
    fake_request_payload,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
ANSWER_ID = UUID("00000000-0000-0000-0000-000000000002")
MESSAGES: list[ChatMessage] = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "你好"},
]


def generating_runtime(answer_id: UUID = ANSWER_ID) -> FishRuntime:
    runtime = FishRuntime()
    runtime.answer_message_id = answer_id
    runtime.status = MessageStatus.GENERATING
    return runtime


def empty_recall_result(depth: int) -> RecallResult:
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


def continuation_prompt(shallow_answer: str) -> str:
    pipeline = AnswerGenerationPipeline(
        FakeChatClient([]),
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )
    return pipeline._build_continuation_messages(  # noqa: SLF001
        MESSAGES,
        shallow_answer,
        empty_recall_result(1),
    )[-1]["content"]


def test_empty_append_requires_complete_shallow_answer_and_no_new_content() -> None:
    prompt = continuation_prompt("已经完整回答。")

    assert "浅回答已经完整回答当前问题" in prompt
    assert "补充信息没有提供尚未表达的新内容" in prompt
    assert "允许 append 为空字符串" in prompt


def test_waiting_shallow_answer_requires_nonempty_completion() -> None:
    prompt = continuation_prompt("让我想想。")

    assert "浅回答只是等待型过渡" in prompt
    assert "append 必须完成当前回答，不能返回空字符串" in prompt


def test_empty_supplement_completes_without_inventing_an_answer() -> None:
    prompt = continuation_prompt("Give me a moment.")

    assert "补充信息为空且浅回答是等待型过渡" in prompt
    assert "append 仍必须非空" in prompt
    assert "不得猜测或编造" in prompt


def test_continuation_style_remains_owned_by_role_and_context() -> None:
    prompt = continuation_prompt("让我想想。")

    assert "最终语言风格继续完全服从角色设定、上下文和当前对话" in prompt
    assert "我能确认聊到的有" not in prompt


def test_continuation_language_follows_current_user_content_not_protocol() -> None:
    pipeline = AnswerGenerationPipeline(
        FakeChatClient([]),
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )
    prompt = pipeline._build_continuation_messages(  # noqa: SLF001
        [
            {"role": "system", "content": "中文协议"},
            {"role": "user", "content": "What did I do yesterday?"},
        ],
        "Give me a moment.",
        empty_recall_result(1),
    )[-1]["content"]

    assert "# 原始当前用户消息" in prompt
    assert '{"content":"What did I do yesterday?"}' in prompt
    assert "当前用户消息 content" in prompt
    assert "用户明确指定的语言" in prompt
    assert "当前消息的主要语言" in prompt
    assert "协议、示例、工具结果的语言" in prompt
    assert "append" in prompt


def test_continuation_contract_does_not_request_end_decision() -> None:
    prompt = continuation_prompt("让我想想。")

    assert '"end"' not in prompt
    assert "end 必须是最后一个字段" not in prompt


def test_continuation_parser_rejects_end_field() -> None:
    with pytest.raises(ValueError, match="only append"):
        _parse_continuation_append('{"append":"补充。","end":true}')


class BlockingRecallService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []

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
        assert fish_id == FISH_ID
        del conversation_context
        self.calls.append((content, depth, tuple(entity_hints)))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return empty_recall_result(depth)


class BlockingInitialClient:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages, enable_search, force_search
        yield ChatStreamChunk(content=('{"r":1,"a":"我想想'))
        await self.release.wait()
        yield ChatStreamChunk(content='。","e":0}')

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages, enable_search, force_search
        if False:
            yield ChatStreamChunk()

    def request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        del enable_search, force_search
        return fake_request_payload(messages)

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        del enable_search, force_search
        return fake_json_request_payload(messages)

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_depth_zero_streams_direct_answer_into_owned_runtime() -> None:
    runtime = generating_runtime()
    pipeline = AnswerGenerationPipeline(
        FakeChatClient(["你好"]),
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )

    await pipeline.generate(
        FISH_ID,
        ANSWER_ID,
        MESSAGES,
        (),
        runtime,
        MessagePerformanceTracker(),
    )

    assert runtime.content == "你好"
    assert runtime.status is MessageStatus.GENERATING


@pytest.mark.asyncio
async def test_complete_answer_without_end_field_does_not_fail_generation() -> None:
    class MissingEndClient(FakeChatClient):
        async def stream_json(
            self,
            messages: Sequence[ChatMessage],
            *,
            enable_search: bool = False,
            force_search: bool = False,
        ) -> AsyncIterator[ChatStreamChunk]:
            del messages, enable_search, force_search
            yield ChatStreamChunk(content=('{"r":0,"m":"neutral","a":"完整回答。"}'))

    runtime = generating_runtime()
    pipeline = AnswerGenerationPipeline(
        MissingEndClient([]),
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )

    await pipeline.generate(
        FISH_ID,
        ANSWER_ID,
        MESSAGES,
        (),
        runtime,
        MessagePerformanceTracker(),
    )

    assert runtime.content == "完整回答。"
    assert runtime.end_conversation is False


@pytest.mark.asyncio
async def test_malformed_response_retries_before_visible_content() -> None:
    client = MalformedThenValidJsonClient()
    runtime = generating_runtime()
    pipeline = AnswerGenerationPipeline(
        client,
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )

    await pipeline.generate(
        FISH_ID,
        ANSWER_ID,
        MESSAGES,
        (),
        runtime,
        MessagePerformanceTracker(),
    )

    assert client.json_calls == 2
    assert runtime.content == "重试成功"


@pytest.mark.asyncio
async def test_invalid_response_after_visible_content_is_not_retried() -> None:
    client = InvalidAfterVisibleJsonClient()
    runtime = generating_runtime()
    pipeline = AnswerGenerationPipeline(
        client,
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )

    with pytest.raises(AnswerGenerationError):
        await pipeline.generate(
            FISH_ID,
            ANSWER_ID,
            MESSAGES,
            (),
            runtime,
            MessagePerformanceTracker(),
        )

    assert client.json_calls == 1
    assert runtime.content == "部分"


@pytest.mark.asyncio
async def test_recall_starts_before_initial_stream_finishes_and_is_cancelled() -> None:
    recall = BlockingRecallService()
    client = BlockingInitialClient()
    runtime = generating_runtime()
    pipeline = AnswerGenerationPipeline(
        client,
        recall_service=recall,
        recall_context_token_budget=50_000,
        enable_search=False,
    )
    task = asyncio.create_task(
        pipeline.generate(
            FISH_ID,
            ANSWER_ID,
            MESSAGES,
            (),
            runtime,
            MessagePerformanceTracker(),
        )
    )

    await asyncio.wait_for(recall.started.wait(), timeout=1)
    assert recall.calls == [("你好", 1, ())]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recall.cancelled is True


@pytest.mark.asyncio
async def test_stale_answer_cannot_append_visible_content() -> None:
    runtime = generating_runtime(UUID(int=99))
    pipeline = AnswerGenerationPipeline(
        FakeChatClient(["不应出现"]),
        recall_service=None,
        recall_context_token_budget=50_000,
        enable_search=False,
    )

    await pipeline.generate(
        FISH_ID,
        ANSWER_ID,
        MESSAGES,
        (),
        runtime,
        MessagePerformanceTracker(),
    )

    assert runtime.content == ""
