import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from uuid import UUID

from fish_memory.answer.context import ChatMessage
from fish_memory.llm.client import ChatStreamChunk
from fish_memory.models import Message, MessageStatus, Role
from fish_memory.performance.models import JsonValue, PerformanceStep
from fish_memory.soul import SoulProfile

DEFAULT_SESSION_ID = UUID("00000000-0000-0000-0000-000000000101")


def fake_request_payload(messages: Sequence[ChatMessage]) -> JsonValue:
    return {
        "model": "fake",
        "messages": [dict(message) for message in messages],
        "parameters": {"stream": True},
    }


def fake_json_request_payload(
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


def _json_string_content(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)[1:-1]


def _lane_kind(messages: Sequence[ChatMessage]) -> str:
    system = messages[0]["content"] if messages else ""
    if system.startswith("# R 通道"):
        return "r"
    if system.startswith("# W 通道"):
        return "w"
    if system.startswith("# H 通道"):
        return "h"
    return "other"


class FakeStore:
    def __init__(
        self,
        fish_id: UUID,
        session_id: UUID = DEFAULT_SESSION_ID,
    ) -> None:
        self.fish_id = fish_id
        self.session_id = session_id
        self.latest_user_sequence = 1
        self.finish_calls: list[dict[str, object]] = []
        self.finish_failures: list[str | None] = []
        self.progress_calls: list[tuple[UUID, UUID, int]] = []
        self.terminal_interrupt_calls: list[tuple[UUID, UUID, int]] = []
        self.stored_progress: int | None = None
        self.next_completion_status: MessageStatus | None = None
        self.history_error: Exception | None = None
        self.finish_error: Exception | None = None
        self.latest_sequence_after_answer_create: int | None = None
        self.history_before_sequences: list[int] = []
        self.insert_count = 0
        self.soft_delete_calls: list[UUID] = []
        self.soft_delete_result = True
        self.history_limits: list[int] = []
        self.answer_message_ids: list[UUID] = []

    async def get_latest_user_sequence(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> int | None:
        assert fish_id == self.fish_id
        assert session_id == self.session_id
        return self.latest_user_sequence

    async def insert_generating_answer(
        self,
        fish_id: UUID,
        session_id: UUID,
        context_through_sequence: int,
        *,
        message_id: UUID | None = None,
        consumed_offset: int = -1,
    ) -> Message:
        self.insert_count += 1
        if self.latest_sequence_after_answer_create is not None:
            self.latest_user_sequence = self.latest_sequence_after_answer_create
        effective_message_id = message_id or UUID(int=100 + self.insert_count)
        self.answer_message_ids.append(effective_message_id)
        return Message(
            id=effective_message_id,
            fish_id=fish_id,
            session_id=session_id,
            sequence=10 + self.insert_count,
            role=Role.ASSISTANT,
            content="",
            status=MessageStatus.GENERATING,
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
            context_through_sequence=context_through_sequence,
            consumed_offset=consumed_offset,
        )

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        self.history_limits.append(limit)
        self.history_before_sequences.append(before_sequence)
        assert session_id == self.session_id
        if self.history_error is not None:
            raise self.history_error
        return [
            Message(
                id=UUID(int=1),
                fish_id=fish_id,
                session_id=session_id,
                sequence=1,
                role=Role.USER,
                content="你好",
                status=MessageStatus.INITIALIZED,
                created_at=datetime(2026, 7, 19, tzinfo=UTC),
            )
        ]

    async def load_history_readonly(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
        appended_history_token_count: int | None = None,
    ) -> list[Message]:
        del appended_history_token_count
        return await self.load_history(
            fish_id,
            session_id,
            before_sequence=before_sequence,
            limit=limit,
        )

    async def finish_answer(
        self,
        message_id: UUID,
        fish_id: UUID,
        *,
        content: str,
        requested_status: MessageStatus,
        failure: str | None = None,
    ) -> MessageStatus:
        self.finish_failures.append(failure)
        if self.finish_error is not None:
            raise self.finish_error
        self.finish_calls.append(
            {
                "message_id": message_id,
                "fish_id": fish_id,
                "content": content,
                "requested_status": requested_status,
            }
        )
        if (
            requested_status is MessageStatus.COMPLETED
            and self.next_completion_status is not None
        ):
            status = self.next_completion_status
            self.next_completion_status = None
            return status
        return requested_status

    async def update_consumed_offset(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
    ) -> int | None:
        self.progress_calls.append((message_id, fish_id, consumed_offset))
        self.stored_progress = max(self.stored_progress or -1, consumed_offset)
        return self.stored_progress

    async def interrupt_terminal_answer(
        self,
        message_id: UUID,
        fish_id: UUID,
        consumed_offset: int,
    ) -> int | None:
        self.terminal_interrupt_calls.append((message_id, fish_id, consumed_offset))
        self.stored_progress = consumed_offset
        return consumed_offset

    async def soft_delete_fish(self, fish_id: UUID) -> bool:
        self.soft_delete_calls.append(fish_id)
        return self.soft_delete_result


class FakeFishSoulStore:
    def __init__(self, profile: SoulProfile | None) -> None:
        self.profile = profile
        self.calls: list[UUID] = []
        self.failure: Exception | None = None

    async def get_profile(self, fish_id: UUID) -> SoulProfile | None:
        self.calls.append(fish_id)
        if self.failure is not None:
            raise self.failure
        return self.profile


class BlockingLatestMessageStore(FakeStore):
    def __init__(self, fish_id: UUID) -> None:
        super().__init__(fish_id)
        self.latest_lookup_started = asyncio.Event()
        self.latest_lookup_cancelled = False

    async def get_latest_user_sequence(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> int | None:
        assert fish_id == self.fish_id
        assert session_id == self.session_id
        self.latest_lookup_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.latest_lookup_cancelled = True
            raise


class MultiFishStore(FakeStore):
    def __init__(self, sequences: dict[UUID, int]) -> None:
        super().__init__(next(iter(sequences)))
        self.sequences = sequences

    async def get_latest_user_sequence(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> int | None:
        del session_id
        return self.sequences.get(fish_id)

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        del before_sequence, limit
        return [
            Message(
                id=UUID(int=self.sequences[fish_id]),
                fish_id=fish_id,
                session_id=session_id,
                sequence=self.sequences[fish_id],
                role=Role.USER,
                content=str(fish_id),
                status=MessageStatus.INITIALIZED,
                created_at=datetime(2026, 7, 19, tzinfo=UTC),
            )
        ]


class MultiSessionStore(FakeStore):
    def __init__(self, fish_id: UUID, sequences: dict[UUID, int]) -> None:
        super().__init__(fish_id, next(iter(sequences)))
        self.sequences = sequences

    async def get_latest_user_sequence(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> int | None:
        assert fish_id == self.fish_id
        return self.sequences.get(session_id)

    async def insert_generating_answer(
        self,
        fish_id: UUID,
        session_id: UUID,
        context_through_sequence: int,
        *,
        message_id: UUID | None = None,
        consumed_offset: int = -1,
    ) -> Message:
        del message_id
        self.insert_count += 1
        return Message(
            id=UUID(int=100 + self.insert_count),
            fish_id=fish_id,
            session_id=session_id,
            sequence=10 + self.insert_count,
            role=Role.ASSISTANT,
            content="",
            status=MessageStatus.GENERATING,
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
            context_through_sequence=context_through_sequence,
            consumed_offset=consumed_offset,
        )

    async def load_history(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int,
        limit: int = 500,
    ) -> list[Message]:
        del before_sequence, limit
        assert fish_id == self.fish_id
        return [
            Message(
                id=UUID(int=self.sequences[session_id]),
                fish_id=fish_id,
                session_id=session_id,
                sequence=self.sequences[session_id],
                role=Role.USER,
                content=str(session_id),
                status=MessageStatus.INITIALIZED,
                created_at=datetime(2026, 7, 19, tzinfo=UTC),
            )
        ]


class FakeChatClient:
    def __init__(self, chunks: list[str], *, search_aware: bool = False) -> None:
        self.chunks = chunks
        self.search_aware = search_aware
        self.calls: list[Sequence[ChatMessage]] = []
        self.lane_calls: list[tuple[str, Sequence[ChatMessage]]] = []
        self.search_flags: list[bool] = []
        self.force_search_flags: list[bool] = []

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        self.calls.append(messages)
        for chunk in self.chunks:
            yield ChatStreamChunk(content=chunk)

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        lane = _lane_kind(messages)
        self.lane_calls.append((lane, messages))
        if lane == "w":
            yield ChatStreamChunk(content='{"w":0}')
            return
        if lane == "h":
            yield ChatStreamChunk(content='{"h":0}')
            return
        self.search_flags.append(enable_search)
        self.force_search_flags.append(force_search)
        self.calls.append(messages)
        decision = (
            '{"m":"neutral","a":"'
            if lane == "r"
            else ('{"r":0,"w":0,"a":"' if self.search_aware else '{"r":0,"a":"')
        )
        yield ChatStreamChunk(content=decision)
        for chunk in self.chunks:
            yield ChatStreamChunk(content=_json_string_content(chunk))
        yield ChatStreamChunk(content='","r":0,"e":0}' if lane == "r" else '","e":0}')

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return fake_request_payload(messages)

    def json_stream_request_payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> JsonValue:
        return fake_json_request_payload(
            messages,
            enable_search=enable_search,
            force_search=force_search,
        )

    async def close(self) -> None:
        return None


class ControlledChatClient:
    def __init__(self) -> None:
        self.calls: list[Sequence[ChatMessage]] = []
        self.queues: list[asyncio.Queue[str | None]] = []
        self.stream_condition = asyncio.Condition()

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        async with self.stream_condition:
            self.calls.append(messages)
            self.queues.append(queue)
            self.stream_condition.notify_all()
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield ChatStreamChunk(content=chunk)

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search, force_search
        lane = _lane_kind(messages)
        if lane == "w":
            yield ChatStreamChunk(content='{"w":0}')
            return
        if lane == "h":
            yield ChatStreamChunk(content='{"h":0}')
            return
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        async with self.stream_condition:
            self.calls.append(messages)
            self.queues.append(queue)
            self.stream_condition.notify_all()
        yield ChatStreamChunk(
            content='{"m":"neutral","a":"' if lane == "r" else '{"r":1,"a":"'
        )
        while True:
            chunk = await queue.get()
            if chunk is None:
                yield ChatStreamChunk(
                    content='","r":0,"e":0}' if lane == "r" else '","e":0}'
                )
                return
            yield ChatStreamChunk(content=_json_string_content(chunk))

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
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

    async def wait_for_streams(self, count: int) -> None:
        async with self.stream_condition:
            await self.stream_condition.wait_for(lambda: len(self.queues) >= count)

    async def emit(self, stream_index: int, chunk: str) -> None:
        await self.queues[stream_index].put(chunk)

    async def finish(self, stream_index: int) -> None:
        await self.queues[stream_index].put(None)

    async def close(self) -> None:
        return None


class LateChunkAfterCancelClient:
    def __init__(self) -> None:
        self.calls = 0
        self.first_chunk_sent = asyncio.Event()

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages
        call_index = self.calls
        self.calls += 1
        if call_index > 0:
            yield ChatStreamChunk(content="新回答")
            return

        yield ChatStreamChunk(content="旧回答")
        self.first_chunk_sent.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            yield ChatStreamChunk(content="迟到内容")

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search, force_search
        lane = _lane_kind(messages)
        if lane == "w":
            yield ChatStreamChunk(content='{"w":0}')
            return
        if lane == "h":
            yield ChatStreamChunk(content='{"h":0}')
            return
        yield ChatStreamChunk(
            content='{"m":"neutral","a":"' if lane == "r" else '{"r":0,"a":"'
        )
        async for chunk in self.stream(messages):
            if chunk.content:
                yield ChatStreamChunk(content=_json_string_content(chunk.content))
        yield ChatStreamChunk(content='","r":0,"e":0}' if lane == "r" else '","e":0}')

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
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


class FailingChatClient:
    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages
        yield ChatStreamChunk(content="部分")
        raise RuntimeError("provider failed")

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search, force_search
        lane = _lane_kind(messages)
        if lane == "w":
            yield ChatStreamChunk(content='{"w":0}')
            return
        if lane == "h":
            yield ChatStreamChunk(content='{"h":0}')
            return
        yield ChatStreamChunk(
            content=('{"m":"neutral","a":"部分' if lane == "r" else '{"r":0,"a":"部分')
        )
        raise RuntimeError("provider failed")

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
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


class MalformedThenValidJsonClient:
    def __init__(self) -> None:
        self.json_calls = 0

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages
        if False:
            yield ChatStreamChunk(content="")

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search, force_search
        lane = _lane_kind(messages)
        if lane == "w":
            yield ChatStreamChunk(content='{"w":0}')
            return
        if lane == "h":
            yield ChatStreamChunk(content='{"h":0}')
            return
        self.json_calls += 1
        if self.json_calls == 1:
            yield ChatStreamChunk(content="[")
            return
        yield ChatStreamChunk(
            content=(
                '{"m":"neutral","a":"重试成功","r":0,"e":0}'
                if lane == "r"
                else '{"r":0,"a":"重试成功","e":0}'
            )
        )

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
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


class TimeoutThenValidJsonClient:
    def __init__(self) -> None:
        self.json_calls = 0

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages
        if False:
            yield ChatStreamChunk(content="")

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search, force_search
        lane = _lane_kind(messages)
        if lane == "w":
            yield ChatStreamChunk(content='{"w":0}')
            return
        if lane == "h":
            yield ChatStreamChunk(content='{"h":0}')
            return
        self.json_calls += 1
        if self.json_calls == 1:
            raise TimeoutError("provider timed out")
        yield ChatStreamChunk(
            content=(
                '{"m":"neutral","a":"超时后重试成功","r":0,"e":0}'
                if lane == "r"
                else '{"r":0,"a":"超时后重试成功","e":0}'
            )
        )

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
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


class InvalidAfterVisibleJsonClient:
    def __init__(self) -> None:
        self.json_calls = 0

    async def stream(
        self,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[ChatStreamChunk]:
        del messages
        if False:
            yield ChatStreamChunk(content="")

    async def stream_json(
        self,
        messages: Sequence[ChatMessage],
        *,
        enable_search: bool = False,
        force_search: bool = False,
    ) -> AsyncIterator[ChatStreamChunk]:
        del enable_search, force_search
        lane = _lane_kind(messages)
        if lane == "w":
            yield ChatStreamChunk(content='{"w":0}')
            return
        if lane == "h":
            yield ChatStreamChunk(content='{"h":0}')
            return
        self.json_calls += 1
        yield ChatStreamChunk(
            content=('{"m":"neutral","a":"部分' if lane == "r" else '{"r":0,"a":"部分')
        )

    def request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
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


class FakePerformanceStore:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.writes: list[tuple[UUID, tuple[PerformanceStep, ...]]] = []

    async def insert_steps(
        self,
        message_id: UUID,
        steps: Sequence[PerformanceStep],
    ) -> None:
        self.started.set()
        if self.block:
            await self.release.wait()
        self.writes.append((message_id, tuple(steps)))
