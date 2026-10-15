import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Literal, cast
from uuid import UUID

import pytest

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.audio import PlayedOffsetTracker
from fish_memory.companion.client import (
    CompanionAgentMessage,
    CompanionAskResult,
    CompanionMessage,
    CompanionReplyContext,
)
from fish_memory.companion.performance import PetVoiceStatus
from fish_memory.companion.playback import PlaybackArbiter, PlaybackOwner
from fish_memory.companion.session import (
    CompanionSession,
    CompanionState,
    TtsTextBuffer,
    VoiceReplyContext,
    _AgentProgressBarrier,
)
from fish_memory.pet.types import PetEmotion

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SECOND_FISH_ID = UUID("00000000-0000-0000-0000-000000000002")
ANSWER_ID = UUID("00000000-0000-0000-0000-000000000101")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000201")


class FakeApi:
    def __init__(self) -> None:
        self.ask_calls: list[tuple[UUID, str]] = []
        self.ask_input_modes: list[str | None] = []
        self.ask_session_ids: list[UUID] = []
        self.answer_session_ids: list[UUID] = []
        self.resolve_calls: list[UUID] = []
        self.progress_calls: list[tuple[UUID, int]] = []
        self.interrupt_calls: list[tuple[UUID, int]] = []
        self.interrupt_started = asyncio.Event()
        self.answer_events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        self.home_events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        self.resolved_session_ids: list[UUID] = [SESSION_ID]
        self.voice_commit_calls: list[tuple[VoiceDraftKey, int, int, str]] = []
        self.voice_commit_error: Exception | None = None

    async def resolve_session(self, fish_id: UUID) -> UUID:
        self.resolve_calls.append(fish_id)
        return self.resolved_session_ids.pop(0)

    async def ask(
        self,
        fish_id: UUID,
        content: str,
        *,
        session_id: UUID,
        reply_context: CompanionReplyContext | None = None,
        input_mode: str | None = None,
    ) -> CompanionAskResult:
        del reply_context
        self.ask_calls.append((fish_id, content))
        self.ask_session_ids.append(session_id)
        self.ask_input_modes.append(input_mode)
        return CompanionAskResult(ANSWER_ID, True)

    async def commit_voice_turn(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        promote_revision: int,
        content: str,
        reply_context: CompanionReplyContext | None = None,
    ) -> CompanionAskResult:
        del reply_context
        self.voice_commit_calls.append((key, revision, promote_revision, content))
        if self.voice_commit_error is not None:
            raise self.voice_commit_error
        return CompanionAskResult(ANSWER_ID, True)

    async def iter_answer(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
    ) -> AsyncIterator[dict[str, object]]:
        del fish_id
        self.answer_session_ids.append(session_id)
        while True:
            event = await self.answer_events.get()
            if event is None:
                return
            yield event

    async def iter_home_actions(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
    ) -> AsyncIterator[dict[str, object]]:
        del fish_id, session_id
        while True:
            event = await self.home_events.get()
            if event is None:
                return
            yield event

    async def update_progress(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None:
        del fish_id
        self.progress_calls.append((message_id, consumed_offset))

    async def interrupt_answer(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None:
        del fish_id
        self.interrupt_started.set()
        self.interrupt_calls.append((message_id, consumed_offset))


class FakeTts:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | int | None]] = []
        self.played_offset = 0

    async def start(self, *, text_offset: int = 0) -> None:
        self.calls.append(("start", text_offset or None))
        self.played_offset = text_offset

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))
        self.played_offset += len(text)

    async def flush(self) -> None:
        pass

    async def finish(self) -> None:
        self.calls.append(("finish", None))

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def resume(self) -> None:
        self.calls.append(("resume", None))

    async def duck(self) -> None:
        self.calls.append(("duck", None))

    async def unduck(self) -> None:
        self.calls.append(("unduck", None))


class FakeDraftPromotion:
    def __init__(self, output: FakeTts, submitted_offset: int) -> None:
        self.output = output
        self.submitted_offset = submitted_offset


class FakeDraftTtsSession:
    def __init__(self, promotion: FakeDraftPromotion | None) -> None:
        self.promotion = promotion
        self.promotions: list[tuple[VoiceDraftKey, int, str, int]] = []
        self.promotion_owners: list[PlaybackOwner | None] = []
        self.cancels: list[tuple[VoiceDraftKey, int]] = []

    async def promote(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        content: str,
        flush_offset: int,
        playback_owner: PlaybackOwner | None = None,
        before_playback: Callable[[], None] | None = None,
    ) -> FakeDraftPromotion | None:
        self.promotions.append((key, revision, content, flush_offset))
        self.promotion_owners.append(playback_owner)
        result = self.promotion
        self.promotion = None
        if result is not None and before_playback is not None:
            before_playback()
        return result

    async def cancel(
        self,
        key: VoiceDraftKey,
        *,
        through_revision: int,
    ) -> None:
        self.cancels.append((key, through_revision))

    async def close(self) -> None:
        return None


class AgentFakeApi(FakeApi):
    def __init__(self) -> None:
        super().__init__()
        self.agent_events: asyncio.Queue[
            CompanionAgentMessage | BaseException | None
        ] = asyncio.Queue()
        self.agent_connections: list[tuple[UUID, UUID, int]] = []
        self.agent_consumption_calls: list[tuple[UUID, int, bool]] = []
        self.agent_consumption_error: Exception | None = None
        self.companion_consumption_sources: list[str] = []
        self.answer_expected = True
        self.ask_reply_contexts: list[CompanionReplyContext | None] = []
        self.voice_reply_contexts: list[CompanionReplyContext | None] = []

    async def ask(
        self,
        fish_id: UUID,
        content: str,
        *,
        session_id: UUID,
        reply_context: CompanionReplyContext | None = None,
        input_mode: str | None = None,
    ) -> CompanionAskResult:
        self.ask_calls.append((fish_id, content))
        self.ask_session_ids.append(session_id)
        self.ask_reply_contexts.append(reply_context)
        return CompanionAskResult(ANSWER_ID, self.answer_expected)

    async def commit_voice_turn(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        promote_revision: int,
        content: str,
        reply_context: CompanionReplyContext | None = None,
    ) -> CompanionAskResult:
        self.voice_commit_calls.append((key, revision, promote_revision, content))
        self.voice_reply_contexts.append(reply_context)
        if self.voice_commit_error is not None:
            raise self.voice_commit_error
        return CompanionAskResult(ANSWER_ID, self.answer_expected)

    async def iter_agent_messages(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        after_sequence: int,
    ) -> AsyncIterator[CompanionAgentMessage]:
        self.agent_connections.append((fish_id, session_id, after_sequence))
        while True:
            item = await self.agent_events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def consume_agent_message(
        self,
        fish_id: UUID,
        message_id: UUID,
        *,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> None:
        del fish_id, session_id
        self.agent_consumption_calls.append((message_id, consumed_offset, interrupted))
        if self.agent_consumption_error is not None:
            raise self.agent_consumption_error

    async def iter_companion_messages(
        self,
        fish_id: UUID,
        *,
        session_id: UUID,
        after_sequence: int,
    ) -> AsyncIterator[CompanionMessage]:
        async for message in self.iter_agent_messages(
            fish_id,
            session_id=session_id,
            after_sequence=after_sequence,
        ):
            yield message

    async def consume_companion_message(
        self,
        fish_id: UUID,
        message_id: UUID,
        *,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
        source: Literal["agent", "proactive_observation"],
    ) -> None:
        self.companion_consumption_sources.append(source)
        await self.consume_agent_message(
            fish_id,
            message_id,
            session_id=session_id,
            consumed_offset=consumed_offset,
            interrupted=interrupted,
        )


class BlockingAgentAskApi(AgentFakeApi):
    def __init__(self) -> None:
        super().__init__()
        self.ask_started = asyncio.Event()
        self.release_ask = asyncio.Event()

    async def ask(
        self,
        fish_id: UUID,
        content: str,
        *,
        session_id: UUID,
        reply_context: CompanionReplyContext | None = None,
        input_mode: str | None = None,
    ) -> CompanionAskResult:
        self.ask_calls.append((fish_id, content))
        self.ask_session_ids.append(session_id)
        self.ask_reply_contexts.append(reply_context)
        self.ask_started.set()
        await self.release_ask.wait()
        return CompanionAskResult(ANSWER_ID, self.answer_expected)


class CallbackTts(FakeTts):
    def __init__(self) -> None:
        super().__init__()
        self.on_first_frame: Callable[[], None] | None = None
        self._first_frame_reported = False

    async def start(self, *, text_offset: int = 0) -> None:
        await super().start(text_offset=text_offset)
        self._first_frame_reported = False

    async def append(self, text: str) -> None:
        await super().append(text)
        callback = self.on_first_frame
        if callback is not None and not self._first_frame_reported:
            self._first_frame_reported = True
            callback()


class BlockingAgentTts(CallbackTts):
    def __init__(self) -> None:
        super().__init__()
        self.finish_started = asyncio.Event()
        self.release_finish = asyncio.Event()

    async def finish(self) -> None:
        self.calls.append(("finish", None))
        self.finish_started.set()
        await self.release_finish.wait()

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))
        self.release_finish.set()


class PartialThenFullTts(FakeTts):
    def __init__(self, first_played_offset: int) -> None:
        super().__init__()
        self.first_played_offset = first_played_offset
        self.start_count = 0
        self.second_start = asyncio.Event()
        self.release_second_start = asyncio.Event()

    async def start(self, *, text_offset: int = 0) -> None:
        await super().start(text_offset=text_offset)
        self.start_count += 1
        if self.start_count == 2:
            self.second_start.set()
            await self.release_second_start.wait()

    async def append(self, text: str) -> None:
        await super().append(text)
        if self.start_count == 1:
            self.played_offset = self.first_played_offset


class FailOnceThenFullTts(FakeTts):
    def __init__(self, failing_operation: str) -> None:
        super().__init__()
        self.failing_operation = failing_operation
        self.start_count = 0
        self.second_start = asyncio.Event()
        self.release_second_start = asyncio.Event()

    async def start(self, *, text_offset: int = 0) -> None:
        self.start_count += 1
        if self.start_count == 1 and self.failing_operation == "start":
            self.calls.append(("start_failed", text_offset))
            raise RuntimeError("start failed")
        await super().start(text_offset=text_offset)
        if self.start_count == 2:
            self.second_start.set()
            await self.release_second_start.wait()

    async def append(self, text: str) -> None:
        await super().append(text)
        if self.start_count == 1:
            self.played_offset = 3
            if self.failing_operation == "append":
                raise RuntimeError("append failed")
            return

    async def finish(self) -> None:
        await super().finish()
        if self.start_count == 1 and self.failing_operation == "finish":
            raise RuntimeError("finish failed")


class FullFrameTts(CallbackTts):
    async def append(self, text: str) -> None:
        await super().append(text)


class FirstFrameThenFailTts(CallbackTts):
    def __init__(self) -> None:
        super().__init__()
        self.start_count = 0
        self.second_start = asyncio.Event()
        self.release_second_start = asyncio.Event()

    async def start(self, *, text_offset: int = 0) -> None:
        await super().start(text_offset=text_offset)
        self.start_count += 1
        if self.start_count == 2:
            self.second_start.set()
            await self.release_second_start.wait()

    async def append(self, text: str) -> None:
        await super().append(text)
        if self.start_count == 1:
            self.played_offset = 3
            raise RuntimeError("append failed after first frame")


class FullBlockingAgentTts(BlockingAgentTts):
    async def append(self, text: str) -> None:
        await super().append(text)


class TimedNoProgressTts(FakeTts):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.started_at: list[float] = []
        self.fourth_start = asyncio.Event()
        self.release_fourth_start = asyncio.Event()

    async def start(self, *, text_offset: int = 0) -> None:
        self.calls.append(("start", text_offset or None))
        self.played_offset = text_offset
        self.started_at.append(asyncio.get_running_loop().time())
        if len(self.started_at) == 4:
            self.fourth_start.set()
            await self.release_fourth_start.wait()
        elif self.mode == "start_fail":
            raise RuntimeError("start failed without progress")

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))
        self.release_fourth_start.set()


class PacketTrackedAgentTts:
    consumption_is_confirmed = True

    def __init__(self, mode: Literal["append_error", "block_second"]) -> None:
        self.mode = mode
        self.calls: list[tuple[str, object]] = []
        self.on_first_frame: Callable[[], None] | None = None
        self.packet_consumed = asyncio.Event()
        self.release_append = asyncio.Event()
        self._tracker = PlayedOffsetTracker()
        self._next_offset = 0
        self._append_count = 0
        self._first_frame_reported = False

    @property
    def played_offset(self) -> int:
        return self._tracker.played_offset

    async def start(self, *, text_offset: int = 0) -> None:
        self.calls.append(("start", text_offset or None))
        self._tracker = PlayedOffsetTracker(initial_offset=text_offset)
        self._next_offset = text_offset

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))
        self._append_count += 1
        start_offset = self._next_offset
        end_offset = start_offset + len(text)
        # Real providers can send multiple PCM packets for the same submitted
        # text segment. One device byte must not make the whole segment durable.
        for _ in range(2):
            self._tracker.add_segment(
                byte_length=8,
                start_offset=start_offset,
                end_offset=end_offset,
            )
        self._tracker.mark(end_offset)
        self._next_offset = end_offset
        if self.mode == "block_second" and self._append_count == 1:
            self._tracker.consume(16)
        else:
            self._tracker.consume(1)
        callback = self.on_first_frame
        if callback is not None and not self._first_frame_reported:
            self._first_frame_reported = True
            callback()
        if self.mode == "append_error" or self._append_count == 2:
            self.packet_consumed.set()
        if self.mode == "append_error":
            raise RuntimeError("provider failed after the first PCM packet")
        if self._append_count == 1:
            return
        await self.release_append.wait()

    async def finish(self) -> None:
        self.calls.append(("finish", None))

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))
        self.release_append.set()

    async def stop(self) -> None:
        self.calls.append(("stop", None))
        self.release_append.set()

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def resume(self) -> None:
        self.calls.append(("resume", None))

    async def duck(self) -> None:
        self.calls.append(("duck", None))

    async def unduck(self) -> None:
        self.calls.append(("unduck", None))


def agent_message(
    sequence: int,
    *,
    message_id: UUID | None = None,
    session_id: UUID = SESSION_ID,
    content: str = "任务已经完成，请查看结果。",
    consumed_offset: int = 0,
    subject_kind: Literal["interaction", "run_result"] = "run_result",
    activation_policy: Literal[
        "playback_start", "playback_complete"
    ] = "playback_start",
) -> CompanionAgentMessage:
    return CompanionAgentMessage(
        message_id=message_id or UUID(int=1000 + sequence),
        sequence=sequence,
        session_id=session_id,
        content=content,
        consumed_offset=consumed_offset,
        subject_kind=subject_kind,
        subject_id=UUID(int=2000 + sequence),
        activation_policy=activation_policy,
    )


def proactive_message(
    sequence: int,
    *,
    message_id: UUID | None = None,
    content: str = "你已经专注一阵子了，要不要休息一下？",
    consumed_offset: int = 0,
    expires_at: datetime | None = None,
    supersession_key: str | None = "attention",
) -> CompanionMessage:
    return CompanionMessage(
        message_id=message_id or UUID(int=3000 + sequence),
        sequence=sequence,
        session_id=SESSION_ID,
        content=content,
        consumed_offset=consumed_offset,
        source="proactive_observation",
        emotion="happy",
        interrupt_policy="drop",
        expires_at=expires_at,
        supersession_key=supersession_key,
    )


async def wait_until(
    predicate: Callable[[], bool],
    *,
    max_wait_seconds: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + max_wait_seconds
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("test condition did not become true")
        await asyncio.sleep(0.001)


class RecordingPerformanceReporter:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID, PetVoiceStatus]] = []

    def submit(
        self,
        fish_id: UUID,
        answer_message_id: UUID,
        status: PetVoiceStatus,
    ) -> None:
        self.calls.append((fish_id, answer_message_id, status))


class RecordingPetEvents:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def state(self, value: str, generation: UUID | None) -> None:
        self.calls.append(("state", value, generation))

    def expression(
        self,
        emotion: PetEmotion,
        generation: UUID | None,
    ) -> None:
        self.calls.append(("expression", emotion, generation))

    def audio_start(self, generation: UUID, sample_rate: int) -> None:
        self.calls.append(("audio_start", generation, sample_rate))

    def audio_pause(self, generation: UUID | None) -> None:
        self.calls.append(("audio_pause", generation))

    def audio_resume(self, generation: UUID | None) -> None:
        self.calls.append(("audio_resume", generation))

    def audio_stop(self, generation: UUID | None) -> None:
        self.calls.append(("audio_stop", generation))

    def cancel(self, generation: UUID | None) -> None:
        self.calls.append(("cancel", generation))

    def pcm(self, value: bytes) -> None:
        self.calls.append(("pcm", value))


@pytest.mark.asyncio
async def test_session_emits_generation_scoped_pet_state_metadata_and_audio() -> None:
    api = FakeApi()
    events = RecordingPetEvents()
    session = CompanionSession(api, FakeTts(), pet_events=events)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.metadata",
            "data": {
                "messageId": str(ANSWER_ID),
                "emotion": "happy",
            },
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "你好呀"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    thinking = next(call for call in events.calls if call[:2] == ("state", "thinking"))
    generation = thinking[2]
    assert isinstance(generation, UUID)
    assert ("expression", PetEmotion.HAPPY, generation) in events.calls
    assert ("audio_start", generation, 16_000) in events.calls
    assert ("audio_stop", generation) in events.calls


@pytest.mark.asyncio
async def test_pause_promotes_duck_without_leaving_stale_output_state() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("上一句")
    await asyncio.sleep(0)
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "正在回答"},
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)

    assert await session.attenuate_output_for_speech()
    assert await session.pause_output_for_speech()
    assert session.output_paused_for_speech is True

    await session.resume_output_after_speech()

    assert tts.calls[-4:] == [
        ("duck", None),
        ("pause", None),
        ("unduck", None),
        ("resume", None),
    ]
    assert session.output_attenuated_for_speech is False


@pytest.mark.asyncio
async def test_home_query_result_does_not_start_a_second_status_speech() -> None:
    api = FakeApi()
    tts = FakeTts()
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(
        api,
        tts,
        home_actions_enabled=True,
        performance_reporter=reporter,
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("玄关灯开着吗")
    await asyncio.sleep(0)

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": True},
        }
    )
    await api.home_events.put(
        {
            "event": "home.action.succeeded",
            "data": {
                "sourceMessageId": str(ANSWER_ID),
                "status": "succeeded",
                "action": {"entityId": "light.entry", "operation": "query"},
                "result": {"state": "off"},
            },
        }
    )
    await session.wait_for_idle()

    assert ("append", "设备当前是关闭的。") not in tts.calls
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.COMPLETED)]


@pytest.mark.asyncio
async def test_binding_change_discards_old_home_status_waiting_on_turn() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts, home_actions_enabled=True)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("打开玄关灯")
    await asyncio.sleep(0)

    await api.home_events.put(
        {
            "event": "home.action.failed",
            "data": {
                "sourceMessageId": str(ANSWER_ID),
                "status": "failed",
                "action": {"entityId": "light.entry", "operation": "turn_on"},
                "result": {"error": "offline"},
            },
        }
    )
    for _ in range(50):
        if session._status_speech_task is not None:  # noqa: SLF001
            break
        await asyncio.sleep(0.001)
    assert session._status_speech_task is not None  # noqa: SLF001

    await session.set_binding(SECOND_FISH_ID)

    assert session.fish_id == SECOND_FISH_ID
    assert session.state is CompanionState.STANDBY
    assert session._home_tasks == set()  # noqa: SLF001
    assert session._status_speech_task is None  # noqa: SLF001
    assert session._status_speech_queue.empty()  # noqa: SLF001
    assert ("append", "设备操作没有成功。") not in tts.calls


@pytest.mark.asyncio
async def test_late_old_home_event_cannot_enter_rebound_fish_queue() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts, home_actions_enabled=True)
    await session.set_binding(FISH_ID)
    old_generation = session._binding_generation  # noqa: SLF001
    await session.set_binding(SECOND_FISH_ID)
    await api.home_events.put(
        {
            "event": "home.action.succeeded",
            "data": {
                "sourceMessageId": str(ANSWER_ID),
                "status": "succeeded",
                "action": {"entityId": "light.entry", "operation": "query"},
                "result": {"state": "off"},
            },
        }
    )

    await session._watch_home_action(  # noqa: SLF001
        FISH_ID,
        SESSION_ID,
        ANSWER_ID,
        old_generation,
    )

    assert session._status_speech_queue.empty()  # noqa: SLF001
    assert session._status_speech_task is None  # noqa: SLF001
    assert ("append", "设备当前是关闭的。") not in tts.calls


@pytest.mark.asyncio
async def test_binding_change_stops_old_home_status_already_speaking() -> None:
    class BlockingStatusTts(FakeTts):
        def __init__(self) -> None:
            super().__init__()
            self.finish_started = asyncio.Event()

        async def finish(self) -> None:
            self.calls.append(("finish", None))
            self.finish_started.set()
            await asyncio.Event().wait()

    api = FakeApi()
    tts = BlockingStatusTts()
    session = CompanionSession(api, tts, home_actions_enabled=True)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("打开玄关灯")
    await asyncio.sleep(0)

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": True},
        }
    )
    await api.home_events.put(
        {
            "event": "home.action.failed",
            "data": {
                "sourceMessageId": str(ANSWER_ID),
                "status": "failed",
                "action": {"entityId": "light.entry", "operation": "turn_on"},
                "result": {"error": "offline"},
            },
        }
    )
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)

    await asyncio.wait_for(session.set_binding(SECOND_FISH_ID), timeout=1)

    assert tts.calls[-4:] == [
        ("start", None),
        ("append", "设备操作没有成功。"),
        ("finish", None),
        ("stop", None),
    ]
    assert session.fish_id == SECOND_FISH_ID
    assert session.state is CompanionState.STANDBY
    assert session._home_tasks == set()  # noqa: SLF001
    assert session._status_speech_task is None  # noqa: SLF001
    assert session._status_speech_queue.empty()  # noqa: SLF001


@pytest.mark.asyncio
async def test_successful_control_does_not_speak_a_second_response() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts, home_actions_enabled=True)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("打开玄关灯")
    await asyncio.sleep(0)

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await api.home_events.put(
        {
            "event": "home.action.succeeded",
            "data": {
                "sourceMessageId": str(ANSWER_ID),
                "status": "succeeded",
                "action": {"entityId": "light.entry", "operation": "turn_on"},
                "result": {},
            },
        }
    )
    await session.wait_for_idle()

    assert not [call for call in tts.calls if call[0] == "append"]


class InterruptibleFakeTts(FakeTts):
    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))


class RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.frames: list[tuple[bytes, bytes, bytes]] = []

    def record_frame(
        self,
        capture_pcm: bytes,
        render_pcm: bytes,
        cleaned_pcm: bytes,
        **_: object,
    ) -> int:
        self.frames.append((capture_pcm, render_pcm, cleaned_pcm))
        return len(self.frames) - 1

    def record_event(self, name: str, **kwargs: object) -> None:
        self.events.append((name, kwargs))


class FailingTts(FakeTts):
    async def start(self) -> None:
        raise RuntimeError("audio device unavailable")


class BlockingWakeAckTts(FakeTts):
    def __init__(self) -> None:
        super().__init__()
        self.append_started = asyncio.Event()
        self.release_append = asyncio.Event()

    async def append(self, text: str) -> None:
        await super().append(text)
        self.append_started.set()
        await self.release_append.wait()


class BlockingStopTts(FakeTts):
    def __init__(self) -> None:
        super().__init__()
        self.stop_started = asyncio.Event()
        self.release_stop = asyncio.Event()

    async def stop(self) -> None:
        await super().stop()
        self.stop_started.set()
        await self.release_stop.wait()


@pytest.mark.asyncio
async def test_audio_recovery_reset_stops_wake_ack_without_unbinding() -> None:
    api = FakeApi()
    tts = BlockingWakeAckTts()
    events = RecordingPetEvents()
    session = CompanionSession(api, tts, pet_events=events)
    await session.set_binding(FISH_ID)
    assert await session.on_wake_word()
    await asyncio.wait_for(tts.append_started.wait(), timeout=1)
    assert session.state is CompanionState.ACKNOWLEDGING
    assert session.session_id == SESSION_ID

    assert await session.reset_for_audio_recovery(FISH_ID)

    assert session.fish_id == FISH_ID
    assert session.session_id is None
    assert session.state is CompanionState.STANDBY
    assert ("stop", None) in tts.calls
    assert not [
        call
        for call in events.calls
        if call[:2] == ("state", CompanionState.UNBOUND.value)
    ]


@pytest.mark.asyncio
async def test_audio_recovery_reset_clears_listening_session() -> None:
    api = FakeApi()
    events = RecordingPetEvents()
    session = CompanionSession(api, FakeTts(), pet_events=events)
    await session.set_binding(FISH_ID)
    assert await session.on_wake_word(play_ack=False)
    assert session.state is CompanionState.LISTENING
    assert session.session_id == SESSION_ID

    assert await session.reset_for_audio_recovery(FISH_ID)

    assert session.fish_id == FISH_ID
    assert session.session_id is None
    assert session.state is CompanionState.STANDBY
    assert not [
        call
        for call in events.calls
        if call[:2] == ("state", CompanionState.UNBOUND.value)
    ]


@pytest.mark.asyncio
async def test_new_conversation_stops_the_turn_and_moves_to_the_new_session() -> None:
    new_session = UUID("00000000-0000-0000-0000-0000000000aa")
    api = FakeApi()
    events = RecordingPetEvents()
    session = CompanionSession(api, FakeTts(), pet_events=events)
    await session.set_binding(FISH_ID)
    assert await session.on_wake_word(play_ack=False)
    assert session.session_id == SESSION_ID

    assert not await session.start_new_session(
        UUID("00000000-0000-0000-0000-000000000002"), new_session
    )
    assert session.session_id == SESSION_ID

    assert await session.start_new_session(FISH_ID, new_session)

    assert session.fish_id == FISH_ID
    assert session.session_id == new_session
    assert session.state is CompanionState.STANDBY
    assert not [
        call
        for call in events.calls
        if call[:2] == ("state", CompanionState.UNBOUND.value)
    ]
    # The next wake keeps the new session instead of resolving the old one.
    assert await session.on_wake_word(play_ack=False)
    assert session.session_id == new_session


@pytest.mark.asyncio
async def test_audio_recovery_reset_ignores_obsolete_binding() -> None:
    session = CompanionSession(FakeApi(), FakeTts())
    await session.set_binding(FISH_ID)
    assert await session.on_wake_word(play_ack=False)

    recovered = await session.reset_for_audio_recovery(
        UUID("00000000-0000-0000-0000-000000000002")
    )

    assert recovered is False
    assert session.fish_id == FISH_ID
    assert session.session_id == SESSION_ID
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
@pytest.mark.parametrize("awake", [False, True])
async def test_healthy_audio_reconnects_background_messages_without_wake(awake):
    api = AgentFakeApi()
    api.resolved_session_ids = [SESSION_ID, SESSION_ID]
    session = CompanionSession(api, FakeTts())
    try:
        await session.set_binding(FISH_ID)
        if awake:
            await session.on_wake_word(play_ack=False)
        assert await session.reset_for_audio_recovery(FISH_ID)
        # A failed replacement must not erase the previous conversation state.
        assert await session.reset_for_audio_recovery(FISH_ID)
        assert session._agent_listener_task is None
        assert await session.resume_after_audio_recovery(FISH_ID)
        assert session.session_id == SESSION_ID
        assert session._agent_listener_task is not None
        assert session.state is (
            CompanionState.LISTENING if awake else CompanionState.STANDBY
        )
        if awake:
            assert session.freeze_turn_binding() is not None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_recovery_cannot_override_an_explicit_same_fish_standby_binding():
    api = AgentFakeApi()
    api.resolved_session_ids = [SESSION_ID, SESSION_ID]
    session = CompanionSession(api, FakeTts())
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        await session.reset_for_audio_recovery(FISH_ID)
        await session.set_binding(FISH_ID)
        assert await session.resume_after_audio_recovery(FISH_ID)
        assert session.state is CompanionState.STANDBY
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_audio_recovery_reset_cancels_active_answer_turn() -> None:
    api = FakeApi()
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(
        api,
        FakeTts(),
        performance_reporter=reporter,
    )
    await session.set_binding(FISH_ID)
    assert await session.on_wake_word(play_ack=False)
    assert await session.submit_transcript("你好")
    await asyncio.sleep(0)
    assert session.state is CompanionState.THINKING

    assert await session.reset_for_audio_recovery(FISH_ID)

    assert session.fish_id == FISH_ID
    assert session.session_id is None
    assert session.state is CompanionState.STANDBY
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)


@pytest.mark.asyncio
async def test_answer_turn_logs_tts_failures(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="fish_memory.companion.session")
    api = FakeApi()
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(
        api,
        FailingTts(),
        performance_reporter=reporter,
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("你好")
    await asyncio.sleep(0)

    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "你好呀"},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert "语音回答处理失败" in caplog.text
    assert "audio device unavailable" in caplog.text
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.FAILED)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_name", "expected_status"),
    [
        ("answer.completed", PetVoiceStatus.COMPLETED),
        ("answer.interrupted", PetVoiceStatus.STOPPED),
        ("answer.failed", PetVoiceStatus.FAILED),
    ],
)
async def test_answer_terminal_event_reports_pet_voice_status(
    event_name: str,
    expected_status: PetVoiceStatus,
) -> None:
    api = FakeApi()
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(
        api,
        FakeTts(),
        performance_reporter=reporter,
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "你好呀"},
        }
    )
    await api.answer_events.put(
        {
            "event": event_name,
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )

    await asyncio.wait_for(session.wait_for_idle(), timeout=1)
    await session.close()

    assert reporter.calls == [(FISH_ID, ANSWER_ID, expected_status)]


@pytest.mark.asyncio
@pytest.mark.parametrize("final_played_offset", [None, 3])
async def test_failed_continuation_finishes_existing_text_without_ending_conversation(
    final_played_offset: int | None,
) -> None:
    class MeasuredTts(FakeTts):
        async def finish(self) -> None:
            await super().finish()
            if final_played_offset is not None:
                self.played_offset = final_played_offset

    api = FakeApi()
    tts = MeasuredTts()
    reporter = RecordingPerformanceReporter()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api, tts, performance_reporter=reporter, playback_arbiter=arbiter
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    text = "第一句。后面这句还没有读完"
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": text},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.failed",
            "data": {"messageId": str(ANSWER_ID), "end": True},
        }
    )

    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert "".join(str(value) for name, value in tts.calls if name == "append") == text
    assert ("finish", None) in tts.calls
    assert ("stop", None) not in tts.calls
    assert api.progress_calls == [
        (ANSWER_ID, len(text) if final_played_offset is None else final_played_offset)
    ]
    assert api.interrupt_calls == []
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.FAILED)]
    assert session.state is CompanionState.LISTENING
    assert session.session_id == SESSION_ID
    assert arbiter.current_owner is None
    await session.close()


@pytest.mark.asyncio
async def test_failed_continuation_can_be_interrupted_while_draining_audio() -> None:
    api = FakeApi()

    class PartialTts(BlockingAgentTts):
        async def append(self, text):
            self.calls.append(("append", text))

    tts = PartialTts()
    reporter = RecordingPerformanceReporter()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api, tts, performance_reporter=reporter, playback_arbiter=arbiter
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "已经开始播放的回答"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.failed",
            "data": {"messageId": str(ANSWER_ID)},
        }
    )
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 3
    assert session.state is CompanionState.SPEAKING
    assert arbiter.current_owner is not None
    assert reporter.calls == []

    await asyncio.wait_for(session.interrupt_from_speech(), timeout=0.2)
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert ("interrupt", None) in tts.calls
    assert api.interrupt_calls == [(ANSWER_ID, 3)]
    assert all(offset <= 3 for _, offset in api.progress_calls)
    assert session.state is CompanionState.LISTENING
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.STOPPED)]
    await session.close()


@pytest.mark.asyncio
async def test_failed_answer_without_content_does_not_start_audio() -> None:
    api = FakeApi()
    tts = FakeTts()
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(api, tts, performance_reporter=reporter)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.failed",
            "data": {"messageId": str(ANSWER_ID), "end": True},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert not any(name in {"start", "append", "finish"} for name, _ in tts.calls)
    assert api.progress_calls == [(ANSWER_ID, 0)]
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.FAILED)]
    assert session.state is CompanionState.LISTENING
    await session.close()


@pytest.mark.asyncio
async def test_failed_continuation_still_cleans_up_a_real_tts_finish_failure() -> None:
    api = FakeApi()
    tts = FailOnceThenFullTts("finish")
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(api, tts, performance_reporter=reporter)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "已经生成的回答"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.failed",
            "data": {"messageId": str(ANSWER_ID)},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert ("finish", None) in tts.calls
    assert ("stop", None) in tts.calls
    assert api.progress_calls == [(ANSWER_ID, 3)]
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.FAILED)]
    assert session.state is CompanionState.LISTENING
    await session.close()


@pytest.mark.asyncio
async def test_failed_answer_progress_error_does_not_leave_playback_owner_held() -> (
    None
):
    class FailingProgressApi(FakeApi):
        async def update_progress(
            self, fish_id: UUID, message_id: UUID, consumed_offset: int
        ) -> None:
            raise RuntimeError("progress temporarily unavailable")

    api = FailingProgressApi()
    tts = FakeTts()
    reporter = RecordingPerformanceReporter()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api, tts, performance_reporter=reporter, playback_arbiter=arbiter
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "已经生成的回答"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.failed",
            "data": {"messageId": str(ANSWER_ID)},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert ("finish", None) in tts.calls
    assert ("stop", None) not in tts.calls
    assert arbiter.current_owner is None
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.FAILED)]
    assert session.state is CompanionState.LISTENING
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [False, True])
async def test_failed_answer_progress_wait_is_bounded_and_interruptible(
    interrupt: bool,
) -> None:
    class HangingProgressApi(FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.progress_started = asyncio.Event()
            self.progress_canceled = asyncio.Event()

        async def update_progress(
            self,
            fish_id: UUID,
            message_id: UUID,
            consumed_offset: int,
        ) -> None:
            self.progress_started.set()
            try:
                await asyncio.Future()
            finally:
                self.progress_canceled.set()

    api = HangingProgressApi()
    tts = BlockingAgentTts()
    tts.release_finish.set()
    reporter = RecordingPerformanceReporter()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api, tts, performance_reporter=reporter, playback_arbiter=arbiter
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    text = "已经生成的回答"
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": text},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.failed",
            "data": {"messageId": str(ANSWER_ID)},
        }
    )
    await asyncio.wait_for(api.progress_started.wait(), timeout=1)
    if interrupt:
        await asyncio.wait_for(session.interrupt_from_speech(), timeout=0.2)
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    await asyncio.wait_for(api.progress_canceled.wait(), timeout=1)
    assert arbiter.current_owner is None
    assert session.state is CompanionState.LISTENING
    assert reporter.calls == [
        (
            FISH_ID,
            ANSWER_ID,
            PetVoiceStatus.STOPPED if interrupt else PetVoiceStatus.FAILED,
        )
    ]
    assert api.interrupt_calls == ([(ANSWER_ID, len(text))] if interrupt else [])
    await session.close()


@pytest.mark.asyncio
async def test_wake_ack_does_not_report_pet_voice_performance() -> None:
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(
        FakeApi(),
        FakeTts(),
        performance_reporter=reporter,
    )
    await session.set_binding(FISH_ID)

    await session.on_wake_word()
    await session.wait_for_idle()

    assert reporter.calls == []


@pytest.mark.asyncio
async def test_transcript_stops_wake_ack_before_starting_answer() -> None:
    api = FakeApi()
    tts = BlockingWakeAckTts()
    session = CompanionSession(api, tts)

    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await asyncio.wait_for(tts.append_started.wait(), timeout=1)

    assert await session.submit_transcript("你好")

    assert tts.calls[-1] == ("stop", None)
    await session.close()


@pytest.mark.asyncio
async def test_closing_the_mic_ends_listening_and_a_new_wake_starts_again() -> None:
    tts = BlockingWakeAckTts()
    api = FakeApi()
    # Ending the conversation lets the next wake resolve its session again.
    api.resolved_session_ids = [SESSION_ID, SESSION_ID]
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    assert not await session.stop_listening()
    await session.on_wake_word()
    await asyncio.wait_for(tts.append_started.wait(), timeout=1)
    assert session.state is CompanionState.ACKNOWLEDGING

    assert await session.stop_listening()
    assert session.state is CompanionState.STANDBY
    assert tts.calls[-1] == ("stop", None)
    assert not await session.stop_listening()

    assert await session.on_wake_word(play_ack=False)
    assert session.state is CompanionState.LISTENING
    assert await session.stop_listening()
    assert session.state is CompanionState.STANDBY
    await session.close()


@pytest.mark.asyncio
async def test_background_listener_rejoins_the_session_after_a_conversation_ends() -> (
    None
):
    # The listener's subscription keeps 内心 and 主动陪伴 running between
    # conversations; ending one must not leave TA unsubscribed until the next wake.
    api = AgentFakeApi()
    api.resolved_session_ids = [SESSION_ID, SESSION_ID]
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await wait_until(lambda: len(api.agent_connections) == 1)

    assert await session.on_wake_word(play_ack=False)
    assert await session.stop_listening()

    await wait_until(lambda: len(api.agent_connections) == 2)
    assert api.agent_connections[-1][:2] == (FISH_ID, SESSION_ID)
    assert session.state is CompanionState.STANDBY
    await session.close()


@pytest.mark.asyncio
async def test_speech_interrupt_releases_wake_finish_before_next_answer() -> None:
    class FinishingWakeTts(FakeTts):
        def __init__(self) -> None:
            super().__init__()
            self.wake_finishing = asyncio.Event()
            self.release_wake = asyncio.Event()

        async def finish(self) -> None:
            await super().finish()
            if not self.wake_finishing.is_set():
                self.wake_finishing.set()
                await self.release_wake.wait()

        async def interrupt(self) -> None:
            # Like ApiTtsOutput, stopping audio does not resolve the provider's
            # pending finish. Its owning task must also be cancelled.
            self.calls.append(("interrupt", None))

    api = FakeApi()
    tts = FinishingWakeTts()
    playback = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=playback)
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word()
        await asyncio.wait_for(tts.wake_finishing.wait(), timeout=1)

        await asyncio.wait_for(session.interrupt_from_speech(), timeout=0.2)

        assert session.state is CompanionState.LISTENING
        assert playback.current_owner is None
        assert not tts.release_wake.is_set()
        assert await session.submit_transcript("说话呀")
        await api.answer_events.put(
            {
                "event": "answer.snapshot",
                "data": {"messageId": str(ANSWER_ID), "content": "我在听。"},
            }
        )
        await api.answer_events.put(
            {
                "event": "answer.completed",
                "data": {"messageId": str(ANSWER_ID), "end": False},
            }
        )
        await asyncio.wait_for(session.wait_for_idle(), timeout=1)

        assert ("append", "我在听。") in tts.calls
        assert playback.current_owner is None
        assert session.state is CompanionState.LISTENING
        assert api.interrupt_calls == []
    finally:
        tts.release_wake.set()
        await session.close()
        await playback.close()


@pytest.mark.asyncio
async def test_wake_word_then_transcript_starts_one_answer_turn() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)

    await session.set_binding(FISH_ID)
    assert session.state is CompanionState.STANDBY
    await session.on_wake_word()
    assert session.state is CompanionState.ACKNOWLEDGING

    await session.submit_transcript("你好")
    await asyncio.sleep(0)

    assert api.ask_calls == [(FISH_ID, "你好")]
    assert api.ask_session_ids == [SESSION_ID]
    assert api.answer_session_ids == [SESSION_ID]
    assert session.state is CompanionState.THINKING

    await api.answer_events.put(
        {
            "event": "answer.snapshot",
            "data": {"messageId": str(ANSWER_ID), "content": "你好呀"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "content": "你好呀", "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert tts.calls == [("start", None), ("append", "你好呀"), ("finish", None)]
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
async def test_voice_draft_submit_owns_the_ordinary_answer_lifecycle() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)

    accepted = await session.submit_voice_turn(
        UUID(int=301),
        revision=4,
        promote_revision=2,
        content="最终问题",
    )

    assert accepted is True
    assert session.state is CompanionState.THINKING
    assert api.ask_calls == []
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert api.voice_commit_calls == [
        (
            VoiceDraftKey(FISH_ID, SESSION_ID, UUID(int=301)),
            4,
            2,
            "最终问题",
        )
    ]
    assert api.answer_session_ids == [SESSION_ID]
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
async def test_zero_pcm_draft_fallback_synthesizes_prefix_once_ordinary() -> None:
    api = FakeApi()
    ordinary_tts = FakeTts()
    diagnostics = RecordingDiagnostics()
    pet_events = RecordingPetEvents()
    prefix = "已经提前合成。"
    draft_tts = FakeDraftTtsSession(None)
    session = CompanionSession(
        api,
        ordinary_tts,
        draft_tts=draft_tts,
        diagnostics=diagnostics,
        pet_events=pet_events,
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    turn_id = UUID(int=303)
    assert await session.submit_voice_turn(
        turn_id,
        revision=5,
        promote_revision=4,
        content="最终问题",
    )
    assert "tts_start_requested" not in [event[0] for event in diagnostics.events]
    assert not [call for call in pet_events.calls if call[0] == "audio_start"]
    await api.answer_events.put(
        {
            "event": "answer.snapshot",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": prefix + "后续内容。",
                "flushOffset": len(prefix),
            },
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert draft_tts.promotions == [
        (
            VoiceDraftKey(FISH_ID, SESSION_ID, turn_id),
            4,
            prefix + "后续内容。",
            len(prefix),
        )
    ]
    assert ordinary_tts.calls == [
        ("start", None),
        ("append", prefix),
        ("append", "后续内容。"),
        ("finish", None),
    ]
    assert ordinary_tts.calls.count(("append", prefix)) == 1
    assert "tts_start_requested" in [event[0] for event in diagnostics.events]
    recorded_events = cast(
        list[tuple[str, dict[str, object]]],
        diagnostics.events,
    )
    assert "voice_draft_fallback" in [name for name, _ in recorded_events]
    assert len([call for call in pet_events.calls if call[0] == "audio_start"]) == 1


@pytest.mark.asyncio
async def test_draft_tts_miss_falls_back_to_ordinary_tts_once() -> None:
    api = FakeApi()
    ordinary_tts = FakeTts()
    draft_tts = FakeDraftTtsSession(None)
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(
        api,
        ordinary_tts,
        draft_tts=draft_tts,
        diagnostics=diagnostics,
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    turn_id = UUID(int=304)
    assert await session.submit_voice_turn(
        turn_id,
        revision=5,
        promote_revision=4,
        content="最终问题",
    )
    await api.answer_events.put(
        {
            "event": "answer.snapshot",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": "普通回答。",
                "flushOffset": 5,
            },
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert ordinary_tts.calls == [
        ("start", None),
        ("append", "普通回答。"),
        ("finish", None),
    ]
    recorded_events = cast(
        list[tuple[str, dict[str, object]]],
        diagnostics.events,
    )
    assert "voice_draft_fallback" in [name for name, _ in recorded_events]


@pytest.mark.asyncio
async def test_voice_draft_commit_failure_uses_normal_turn_failure_recovery() -> None:
    api = FakeApi()
    api.voice_commit_error = RuntimeError("commit unavailable")
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)

    assert await session.submit_voice_turn(
        UUID(int=302),
        revision=2,
        promote_revision=0,
        content="最终问题",
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert api.ask_calls == []
    assert len(api.voice_commit_calls) == 1
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
async def test_answer_tts_starts_at_first_pause_then_buffers_tail() -> None:
    class PauseTts(FakeTts):
        async def flush(self) -> None:
            self.calls.append(("flush", None))

    api = FakeApi()
    tts = PauseTts()
    session = CompanionSession(api, tts)

    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": "你好，12345,67890,",
            },
        }
    )
    await asyncio.sleep(0)
    assert tts.calls == [("start", None), ("append", "你好，"), ("flush", None)]

    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": "ABCDE.尾巴",
            },
        }
    )
    await asyncio.sleep(0)
    assert ("append", "12345,67890,ABCDE.") in tts.calls

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "content": "", "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert tts.calls[-2:] == [("append", "尾巴"), ("finish", None)]
    assert tts.calls.count(("flush", None)) == 1


def test_tts_text_buffer_waits_for_a_boundary_after_fifteen_characters() -> None:
    buffer = TtsTextBuffer()

    assert buffer.append("12345,67890,") == []
    assert buffer.append("ABCDE.") == ["12345,67890,ABCDE."]
    assert buffer.finish() == []


def test_tts_text_buffer_flushes_a_short_tail_only_when_answer_finishes() -> None:
    buffer = TtsTextBuffer()

    assert buffer.append("12345,67890,") == []
    assert buffer.finish() == ["12345,67890,"]


def test_tts_text_buffer_flushes_a_complete_short_lane_segment() -> None:
    buffer = TtsTextBuffer()
    text = "我去看看最新情况。"

    assert buffer.append(text) == []
    assert buffer.flush_through(len(text)) == [text]
    assert buffer.flush_through(len(text)) == []
    assert buffer.finish() == []


def test_tts_text_buffer_flushes_only_unsent_long_segment_tail() -> None:
    buffer = TtsTextBuffer()
    prefix = "123456789012345,"
    tail = "已经发送。短尾巴"

    assert buffer.append(prefix + tail) == [prefix]
    assert buffer.flush_through(len(prefix + tail)) == [tail]
    assert buffer.flush_through(len(prefix + tail)) == []


@pytest.mark.asyncio
async def test_segment_event_flushes_short_answer_before_terminal_event() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    transition = "我去看看最新情况"

    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("明天台湾呢")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": transition},
        }
    )
    await asyncio.sleep(0)
    assert tts.calls == [("start", None)]

    await api.answer_events.put(
        {
            "event": "answer.segment",
            "data": {
                "messageId": str(ANSWER_ID),
                "flushOffset": len(transition),
                "reason": "lane_answer_complete",
            },
        }
    )
    await asyncio.sleep(0)
    assert tts.calls[-1] == ("append", transition)

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "content": transition, "end": False},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)
    assert tts.calls[-1] == ("finish", None)


@pytest.mark.asyncio
async def test_unpunctuated_segment_commits_provider_before_recall_returns() -> None:
    class SegmentTts(FakeTts):
        async def flush(self) -> None:
            self.calls.append(("flush", None))

    api = FakeApi()
    tts = SegmentTts()
    session = CompanionSession(api, tts)
    transition = "我正在试着回忆之前查明天杭州天气的相关情况哦"
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("查一下明天杭州的天气")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": transition},
        }
    )
    await wait_until(lambda: session.state is CompanionState.SPEAKING)
    assert tts.calls == [("start", None)]
    await api.answer_events.put(
        {
            "event": "answer.segment",
            "data": {"messageId": str(ANSWER_ID), "flushOffset": len(transition)},
        }
    )
    await wait_until(lambda: ("flush", None) in tts.calls)
    assert tts.calls == [("start", None), ("append", transition), ("flush", None)]
    # No continuation or terminal event has arrived yet.
    await session.close()


@pytest.mark.asyncio
async def test_session_reports_answer_and_tts_boundaries_to_diagnostics() -> None:
    api = FakeApi()
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(api, FakeTts(), diagnostics=diagnostics)

    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "你好呀"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "content": "你好呀"},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    names = [event[0] for event in diagnostics.events]
    assert "ask_submitted" in names
    assert "answer_event_received" in names
    assert "answer_first_content" in names
    assert "tts_start_requested" in names
    assert "tts_playback_started" not in names
    assert "tts_playback_stopped" in names

    answer_event = next(
        event for event in diagnostics.events if event[0] == "answer_event_received"
    )
    assert answer_event[1]["metadata"] == {"messageId": str(ANSWER_ID)}


@pytest.mark.asyncio
async def test_wake_word_plays_voice_acknowledgement() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)

    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    assert session.state is CompanionState.ACKNOWLEDGING

    await asyncio.sleep(0)

    assert tts.calls == [
        ("start", None),
        ("append", "我在呢。"),
        ("finish", None),
    ]
    assert session.state is CompanionState.LISTENING


@pytest.mark.asyncio
async def test_wake_ack_reports_tts_playback_stop_to_diagnostics() -> None:
    api = FakeApi()
    diagnostics = RecordingDiagnostics()
    session = CompanionSession(api, FakeTts(), diagnostics=diagnostics)

    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    names = [event[0] for event in diagnostics.events]
    assert names.count("tts_start_requested") == 1
    assert "tts_playback_started" not in names
    assert names.count("tts_playback_stopped") == 1


@pytest.mark.asyncio
async def test_wake_word_can_start_listening_without_ack_for_replay() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)

    await session.set_binding(FISH_ID)
    assert await session.on_wake_word(play_ack=False)

    assert session.state is CompanionState.LISTENING
    assert tts.calls == []


@pytest.mark.asyncio
async def test_end_true_returns_to_wake_word_standby() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("再见")

    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "再见"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "content": "再见", "end": True},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert session.state is CompanionState.STANDBY
    assert session.session_id is None
    assert session.voice_conversation_id is None


@pytest.mark.asyncio
async def test_continuous_voice_turns_reuse_the_wake_session() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)

    for content in ("第一句", "第二句"):
        assert await session.submit_transcript(content)
        await api.answer_events.put(
            {
                "event": "answer.completed",
                "data": {
                    "messageId": str(ANSWER_ID),
                    "content": "收到",
                    "end": False,
                },
            }
        )
        await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert api.resolve_calls == [FISH_ID]
    assert api.ask_session_ids == [SESSION_ID, SESSION_ID]


@pytest.mark.asyncio
async def test_next_wake_resolves_a_new_session_after_conversation_ends() -> None:
    next_session_id = UUID("00000000-0000-0000-0000-000000000202")
    api = FakeApi()
    api.resolved_session_ids.append(next_session_id)
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    assert await session.submit_transcript("再见")
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": "再见",
                "end": True,
            },
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert await session.on_wake_word(play_ack=False)

    assert api.resolve_calls == [FISH_ID, FISH_ID]
    assert session.session_id == next_session_id


@pytest.mark.asyncio
async def test_speech_interrupts_playback_at_current_offset() -> None:
    api = FakeApi()
    tts = FakeTts()
    reporter = RecordingPerformanceReporter()
    session = CompanionSession(api, tts, performance_reporter=reporter)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("继续")

    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "一段回答"},
        }
    )
    await asyncio.sleep(0)
    tts.played_offset = 3
    await session.interrupt_from_speech()
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)

    assert api.interrupt_calls == [(ANSWER_ID, 3)]
    assert tts.calls[-1] == ("stop", None)
    assert session.state is CompanionState.LISTENING
    assert reporter.calls == [(FISH_ID, ANSWER_ID, PetVoiceStatus.STOPPED)]


@pytest.mark.asyncio
async def test_speech_interrupt_diagnostics_include_played_offset() -> None:
    api = FakeApi()
    diagnostics = RecordingDiagnostics()
    tts = InterruptibleFakeTts()
    session = CompanionSession(api, tts, diagnostics=diagnostics)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("继续")

    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "一段回答"},
        }
    )
    await asyncio.sleep(0)
    tts.played_offset = 7
    await session.interrupt_from_speech()
    await session.wait_for_idle()

    events = {
        name: kwargs
        for name, kwargs in diagnostics.events
        if name in {"tts_interrupt_requested", "tts_playback_stopped"}
    }
    assert events["tts_interrupt_requested"]["metadata"]["playedOffset"] == 7
    assert events["tts_playback_stopped"]["metadata"]["playedOffset"] == 7


@pytest.mark.asyncio
async def test_speech_interrupt_persists_before_audio_cleanup_finishes() -> None:
    api = FakeApi()
    tts = BlockingStopTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("继续")

    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "一段回答"},
        }
    )
    await asyncio.sleep(0)
    tts.played_offset = 3

    interrupt_task = asyncio.create_task(session.interrupt_from_speech())
    await asyncio.wait_for(tts.stop_started.wait(), timeout=1)
    await asyncio.wait_for(api.interrupt_started.wait(), timeout=0.1)

    tts.release_stop.set()
    await asyncio.wait_for(interrupt_task, timeout=1)
    assert api.interrupt_calls == [(ANSWER_ID, 3)]


@pytest.mark.asyncio
async def test_speech_interrupt_does_not_delay_the_next_ask() -> None:
    api = FakeApi()
    tts = BlockingStopTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "旧回答"},
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    await asyncio.wait_for(session.interrupt_from_speech(), timeout=0.1)

    assert session.state is CompanionState.LISTENING
    assert await session.submit_transcript("孔子")
    await asyncio.sleep(0)
    assert api.ask_calls == [(FISH_ID, "上一句"), (FISH_ID, "孔子")]

    tts.release_stop.set()
    await session.close()


@pytest.mark.asyncio
async def test_non_interrupt_output_settles_before_next_tts_start() -> None:
    api = FakeApi()
    tts = BlockingStopTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word()
    await session.submit_transcript("上一句")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "旧回答"},
        }
    )
    for _ in range(50):
        if session.state is CompanionState.SPEAKING:
            break
        await asyncio.sleep(0.001)
    assert session.state is CompanionState.SPEAKING

    await session.interrupt_from_speech()
    assert await session.submit_transcript("孔子")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "新回答"},
        }
    )
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": "新回答",
                "end": False,
            },
        }
    )

    await asyncio.sleep(0.01)
    assert ("append", "新回答") not in tts.calls

    tts.release_stop.set()
    await wait_until(lambda: ("append", "新回答") in tts.calls)
    await session.close()


@pytest.mark.asyncio
async def test_agent_listener_starts_on_binding_and_plays_without_wake() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )

    await session.set_binding(FISH_ID)
    await wait_until(lambda: bool(api.agent_connections))
    message = agent_message(1)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)

    assert session.state is CompanionState.SPEAKING
    assert api.ask_calls == []
    assert arbiter.current_owner is not None
    assert arbiter.current_owner.kind == "agent"
    assert session.reply_context == VoiceReplyContext(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        voice_conversation_id=cast(UUID, session.voice_conversation_id),
        subject_kind="run_result",
        subject_id=message.subject_id,
        source_message_id=message.message_id,
        expires_at=session.reply_context.expires_at,
    )

    tts.release_finish.set()
    await wait_until(lambda: bool(api.agent_consumption_calls))
    assert api.agent_consumption_calls == [
        (message.message_id, len(message.content), False)
    ]
    await session.close()


@pytest.mark.asyncio
async def test_proactive_message_uses_dedicated_owner_without_reply_context() -> None:
    api = AgentFakeApi()
    tts = FakeTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    message = proactive_message(1)

    await session.set_binding(FISH_ID)
    await api.agent_events.put(message)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert api.agent_consumption_calls == [
        (message.message_id, len(message.content), False)
    ]
    assert api.companion_consumption_sources == ["proactive_observation"]
    assert session.reply_context is None
    assert sum(call[0] == "start" for call in tts.calls) == 1
    await session.close()


@pytest.mark.asyncio
async def test_new_proactive_message_supersedes_queued_message_without_replay() -> None:
    api = AgentFakeApi()
    tts = FakeTts()
    arbiter = PlaybackArbiter()
    blocker = await arbiter.acquire("answer", UUID(int=9999))
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    old = proactive_message(1, supersession_key="focus")
    replacement = proactive_message(2, supersession_key="focus")

    await session.set_binding(FISH_ID)
    await api.agent_events.put(old)
    await api.agent_events.put(replacement)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert api.agent_consumption_calls == [(old.message_id, 0, True)]
    assert [item.message_id for item in session._agent_queue] == [  # noqa: SLF001
        replacement.message_id
    ]

    await arbiter.release(blocker)
    session._ensure_agent_drain()  # noqa: SLF001
    await wait_until(lambda: len(api.agent_consumption_calls) == 2)
    assert api.agent_consumption_calls[-1] == (
        replacement.message_id,
        len(replacement.content),
        False,
    )
    assert sum(call[0] == "start" for call in tts.calls) == 1
    await session.close()


@pytest.mark.asyncio
async def test_agent_message_drops_older_queued_proactive_before_playback() -> None:
    api = AgentFakeApi()
    tts = FakeTts()
    arbiter = PlaybackArbiter()
    blocker = await arbiter.acquire("answer", UUID(int=9998))
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    proactive = proactive_message(1)
    agent = agent_message(2)

    await session.set_binding(FISH_ID)
    await api.agent_events.put(proactive)
    await wait_until(lambda: bool(session._agent_queue))  # noqa: SLF001
    await api.agent_events.put(agent)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert api.agent_consumption_calls == [(proactive.message_id, 0, True)]
    assert [item.message_id for item in session._agent_queue] == [  # noqa: SLF001
        agent.message_id
    ]
    await arbiter.release(blocker)
    session._ensure_agent_drain()  # noqa: SLF001
    await wait_until(lambda: len(api.agent_consumption_calls) == 2)
    assert api.agent_consumption_calls[-1] == (
        agent.message_id,
        len(agent.content),
        False,
    )
    await session.close()


@pytest.mark.asyncio
async def test_home_status_interrupts_active_proactive_at_actual_offset() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    proactive = proactive_message(1)
    home_text = "设备操作没有成功。"

    await session.set_binding(FISH_ID)
    await api.agent_events.put(proactive)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 4
    await session._status_speech_queue.put(  # noqa: SLF001
        (session.binding_generation, FISH_ID, UUID(int=911), home_text)
    )
    session._ensure_status_speech_worker()  # noqa: SLF001

    await wait_until(lambda: bool(api.agent_consumption_calls))
    await wait_until(lambda: ("append", home_text) in tts.calls)
    assert api.agent_consumption_calls == [(proactive.message_id, 4, True)]
    assert arbiter.current_owner is None or arbiter.current_owner.kind == "home"
    await session.close()


@pytest.mark.asyncio
async def test_answer_delta_interrupts_active_proactive_before_speaking() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    proactive = proactive_message(1)

    await session.set_binding(FISH_ID)
    await api.agent_events.put(proactive)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 5
    await session._consume_event(  # noqa: SLF001
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "新的回答。"},
        }
    )
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert api.agent_consumption_calls == [(proactive.message_id, 5, True)]
    assert arbiter.current_owner is not None
    assert arbiter.current_owner.kind == "answer"
    assert arbiter.current_spoken_text == "新的回答。"
    await session._consume_event(  # noqa: SLF001
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await session.close()


@pytest.mark.asyncio
async def test_proactive_arriving_during_foreground_speech_is_dropped() -> None:
    api = AgentFakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    message = proactive_message(1)

    await session.set_binding(FISH_ID)
    reservation = session.reserve_foreground_speech()
    await api.agent_events.put(message)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert api.agent_consumption_calls == [(message.message_id, 0, True)]
    session.release_foreground_speech(reservation)
    await asyncio.sleep(0.02)
    assert tts.calls == []
    await session.close()


@pytest.mark.asyncio
async def test_codex_notice_waits_for_foreground_instead_of_being_dropped() -> None:
    api = AgentFakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    message = proactive_message(1, supersession_key="codex-completion:example")
    await session.set_binding(FISH_ID)
    reservation = session.reserve_foreground_speech()
    await api.agent_events.put(message)
    await wait_until(lambda: bool(session._agent_queue))
    session._drop_proactive_for_priority_speech()
    assert api.agent_consumption_calls == []
    assert tts.calls == []
    session.release_foreground_speech(reservation)
    await wait_until(lambda: bool(tts.calls))
    await session.close()


@pytest.mark.asyncio
async def test_proactive_supersession_cancels_message_waiting_to_acquire() -> None:
    class GatedProactiveArbiter(PlaybackArbiter):
        def __init__(self) -> None:
            super().__init__()
            self.acquire_started = asyncio.Event()
            self.release_acquire = asyncio.Event()

        async def acquire(
            self,
            kind: str,
            message_id: UUID | None,
        ) -> PlaybackOwner:
            if kind == "proactive":
                self.acquire_started.set()
                await self.release_acquire.wait()
            return await super().acquire(cast(object, kind), message_id)

    api = AgentFakeApi()
    tts = FakeTts()
    arbiter = GatedProactiveArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    old = proactive_message(1, supersession_key="focus")
    replacement = proactive_message(2, supersession_key="focus")

    await session.set_binding(FISH_ID)
    await api.agent_events.put(old)
    await asyncio.wait_for(arbiter.acquire_started.wait(), timeout=1)
    await api.agent_events.put(replacement)
    await wait_until(lambda: bool(api.agent_consumption_calls))
    arbiter.release_acquire.set()
    await wait_until(lambda: len(api.agent_consumption_calls) == 2)

    assert api.agent_consumption_calls == [
        (old.message_id, 0, True),
        (replacement.message_id, len(replacement.content), False),
    ]
    assert sum(call[0] == "start" for call in tts.calls) == 1
    await session.close()


@pytest.mark.asyncio
async def test_foreground_reservation_drops_active_proactive_remainder() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    message = proactive_message(1, supersession_key=None)

    await session.set_binding(FISH_ID)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 4

    reservation = session.reserve_foreground_speech()
    await session.interrupt_from_speech()
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert api.agent_consumption_calls == [(message.message_id, 4, True)]
    assert session.reply_context is None
    session.release_foreground_speech(reservation)
    await asyncio.sleep(0.02)
    assert sum(call[0] == "start" for call in tts.calls) == 1
    assert session._agent_queue == []  # noqa: SLF001
    await session.close()


@pytest.mark.asyncio
async def test_active_proactive_supersession_settles_actual_offset_then_replaces() -> (
    None
):
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    session = CompanionSession(api, tts)
    old = proactive_message(1, supersession_key="focus")
    replacement = proactive_message(2, supersession_key="focus")

    await session.set_binding(FISH_ID)
    await api.agent_events.put(old)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 3
    await api.agent_events.put(replacement)
    await wait_until(lambda: len(api.agent_consumption_calls) == 2)

    assert api.agent_consumption_calls[0] == (old.message_id, 3, True)
    assert api.agent_consumption_calls[1] == (
        replacement.message_id,
        len(replacement.content),
        False,
    )
    assert sum(call[0] == "start" for call in tts.calls) == 2
    await session.close()


@pytest.mark.asyncio
async def test_proactive_expiry_is_rechecked_after_playback_acquire() -> None:
    class GatedProactiveArbiter(PlaybackArbiter):
        def __init__(self) -> None:
            super().__init__()
            self.acquire_started = asyncio.Event()
            self.release_acquire = asyncio.Event()

        async def acquire(
            self,
            kind: str,
            message_id: UUID | None,
        ) -> PlaybackOwner:
            if kind == "proactive":
                self.acquire_started.set()
                await self.release_acquire.wait()
            return await super().acquire(cast(object, kind), message_id)

    current = datetime(2026, 9, 1, tzinfo=UTC)

    def now() -> datetime:
        return current

    api = AgentFakeApi()
    tts = FakeTts()
    arbiter = GatedProactiveArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter, now=now)
    message = proactive_message(
        1,
        expires_at=current + timedelta(minutes=1),
    )

    await session.set_binding(FISH_ID)
    await api.agent_events.put(message)
    await asyncio.wait_for(arbiter.acquire_started.wait(), timeout=1)
    current += timedelta(minutes=2)
    arbiter.release_acquire.set()
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert tts.calls == []
    assert api.agent_consumption_calls == [(message.message_id, 0, True)]
    await session.close()


@pytest.mark.asyncio
async def test_close_cancels_proactive_expiry_and_consumption_retry_tasks() -> None:
    api = AgentFakeApi()
    api.agent_consumption_error = RuntimeError("offline")
    session = CompanionSession(
        api,
        FakeTts(),
        agent_reconnect_delays_seconds=(10.0,),
    )
    message = proactive_message(
        1,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )

    await session.set_binding(FISH_ID)
    await api.agent_events.put(message)
    await wait_until(lambda: bool(api.agent_consumption_calls))
    await wait_until(
        lambda: bool(session._proactive_settlement_tasks)  # noqa: SLF001
    )

    await session.close()

    assert session._proactive_expiry_tasks == {}  # noqa: SLF001
    assert session._proactive_settlement_tasks == set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_wake_reuses_binding_session_and_agent_listener() -> None:
    api = AgentFakeApi()
    session = CompanionSession(api, FakeTts())

    await session.set_binding(FISH_ID)
    await wait_until(lambda: bool(api.agent_connections))
    listener = session._agent_listener_task  # noqa: SLF001

    assert await session.on_wake_word(play_ack=False)
    await asyncio.sleep(0)

    assert api.resolve_calls == [FISH_ID]
    assert session.session_id == SESSION_ID
    assert session._agent_listener_task is listener  # noqa: SLF001
    assert api.agent_connections == [(FISH_ID, SESSION_ID, 0)]
    await session.close()


@pytest.mark.asyncio
async def test_slow_old_binding_resolution_cannot_start_old_session_listener() -> None:
    second_session_id = UUID("00000000-0000-0000-0000-000000000202")

    class SwitchingResolveApi(AgentFakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.first_resolve_started = asyncio.Event()
            self.release_first_resolve = asyncio.Event()

        async def resolve_session(self, fish_id: UUID) -> UUID:
            self.resolve_calls.append(fish_id)
            if fish_id == FISH_ID:
                self.first_resolve_started.set()
                await self.release_first_resolve.wait()
                return SESSION_ID
            return second_session_id

    api = SwitchingResolveApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    first_binding = asyncio.create_task(session.set_binding(FISH_ID))
    await asyncio.wait_for(api.first_resolve_started.wait(), timeout=1)

    await session.set_binding(SECOND_FISH_ID)
    await wait_until(lambda: bool(api.agent_connections))
    api.release_first_resolve.set()
    await first_binding

    assert session.fish_id == SECOND_FISH_ID
    assert session.session_id == second_session_id
    assert api.agent_connections == [(SECOND_FISH_ID, second_session_id, 0)]

    stale = agent_message(1, session_id=SESSION_ID, content="旧会话结果。")
    current = agent_message(2, session_id=second_session_id, content="新会话结果。")
    await api.agent_events.put(stale)
    await api.agent_events.put(current)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert ("append", stale.content) not in tts.calls
    assert ("append", current.content) in tts.calls
    assert api.agent_consumption_calls == [
        (current.message_id, len(current.content), False)
    ]
    await session.close()


@pytest.mark.asyncio
async def test_agent_message_for_another_session_is_ignored_without_consumption() -> (
    None
):
    api = AgentFakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    old_session_id = UUID("00000000-0000-0000-0000-000000000202")

    await session.set_binding(FISH_ID)
    assert await session.on_wake_word(play_ack=False)
    await api.agent_events.put(agent_message(1, session_id=old_session_id))
    await asyncio.sleep(0)

    assert session.session_id == SESSION_ID
    assert session.reply_context is None
    assert api.agent_consumption_calls == []
    assert tts.calls == []
    assert session._agent_after_cursor == 0  # noqa: SLF001
    await session.close()


@pytest.mark.asyncio
async def test_session_end_clears_agent_listener_cursor_and_queue() -> None:
    api = AgentFakeApi()
    session = CompanionSession(api, FakeTts())

    await session.set_binding(FISH_ID)
    assert await session.on_wake_word(play_ack=False)
    await wait_until(lambda: session._agent_listener_task is not None)  # noqa: SLF001
    assert await session.submit_transcript("再见")
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": True},
        }
    )
    await session.wait_for_idle()

    assert session.state is CompanionState.STANDBY
    assert session.session_id is None
    assert session._agent_listener_task is None  # noqa: SLF001
    assert session._agent_listener_session_id is None  # noqa: SLF001
    assert session._agent_after_cursor == 0  # noqa: SLF001
    assert session._agent_queue == []  # noqa: SLF001
    await session.close()


@pytest.mark.asyncio
async def test_active_context_blocks_next_agent_until_monotonic_ttl_expires() -> None:
    api = AgentFakeApi()
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reply_context_ttl_seconds=0.02,
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    first = agent_message(1, content="第一条任务结果。")
    second = agent_message(2, content="第二条任务结果。")

    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(first)
    await api.agent_events.put(second)
    await wait_until(lambda: len(api.agent_consumption_calls) == 1)

    assert [call for call in tts.calls if call[0] == "append"] == [
        ("append", first.content)
    ]
    assert session.reply_context is not None

    await wait_until(lambda: len(api.agent_consumption_calls) == 2)
    assert [call for call in tts.calls if call[0] == "append"] == [
        ("append", first.content),
        ("append", second.content),
    ]
    assert session.reply_context is not None
    assert session.reply_context.source_message_id == second.message_id
    await session.close()


@pytest.mark.asyncio
async def test_agent_listener_reconnects_with_max_cursor_and_sorts_unseen_ids() -> None:
    api = AgentFakeApi()
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reconnect_delays_seconds=(0.0,),
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    later_sequence = agent_message(2, content="序号二。")
    earlier_sequence = agent_message(1, content="序号一。")

    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(later_sequence)
    await api.agent_events.put(earlier_sequence)
    await api.agent_events.put(later_sequence)
    await api.agent_events.put(None)
    await wait_until(lambda: len(api.agent_connections) >= 2)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert api.agent_connections[:2] == [
        (FISH_ID, SESSION_ID, 0),
        (FISH_ID, SESSION_ID, 2),
    ]
    assert [call for call in tts.calls if call[0] == "append"] == [
        ("append", earlier_sequence.content)
    ]
    await session.set_binding(FISH_ID)
    await asyncio.sleep(0)
    assert len(api.agent_connections) == 2
    await session.close()


@pytest.mark.asyncio
async def test_agent_rechecks_foreground_blockers_after_acquire_returns() -> None:
    class GatedAgentArbiter(PlaybackArbiter):
        def __init__(self) -> None:
            super().__init__()
            self.agent_acquire_started = asyncio.Event()
            self.release_agent_acquire = asyncio.Event()

        async def acquire(
            self,
            kind: str,
            message_id: UUID | None,
        ) -> PlaybackOwner:
            if kind == "agent":
                self.agent_acquire_started.set()
                await self.release_agent_acquire.wait()
            return await super().acquire(cast(object, kind), message_id)

    api = AgentFakeApi()
    tts = FakeTts()
    arbiter = GatedAgentArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    message = agent_message(1)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(arbiter.agent_acquire_started.wait(), timeout=1)

    reservation = session.reserve_foreground_speech()
    session._set_state(CompanionState.THINKING)  # noqa: SLF001
    arbiter.release_agent_acquire.set()
    await asyncio.sleep(0)
    session.release_foreground_speech(reservation)
    await asyncio.sleep(0.02)

    assert tts.calls == []
    assert arbiter.current_owner is None
    assert [item.message_id for item in session._agent_queue] == [  # noqa: SLF001
        message.message_id
    ]
    await session.close()


@pytest.mark.asyncio
async def test_frozen_context_ignores_later_agent_and_false_skips_sse() -> None:
    api = BlockingAgentAskApi()
    api.answer_expected = False
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        home_actions_enabled=True,
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    first = agent_message(1, content="第一条任务结果。")
    second = agent_message(2, content="第二条任务结果。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(first)
    await wait_until(lambda: session.reply_context is not None)
    frozen = session.freeze_turn_binding()
    assert frozen is not None

    assert await session.submit_transcript("这是对第一条的回复", frozen_turn=frozen)
    await asyncio.wait_for(api.ask_started.wait(), timeout=1)
    await api.agent_events.put(second)
    await asyncio.sleep(0)
    assert ("append", second.content) not in tts.calls

    api.release_ask.set()
    await session.wait_for_idle()
    assert api.ask_reply_contexts == [
        CompanionReplyContext("run_result", first.subject_id)
    ]
    assert api.ask_session_ids == [first.session_id]
    assert api.answer_session_ids == []
    assert session._home_tasks == set()  # noqa: SLF001
    assert session.reply_context is not None
    assert session.reply_context.source_message_id == second.message_id
    await session.close()


@pytest.mark.asyncio
async def test_voice_commit_no_answer_retires_private_tts_and_skips_answer_sse() -> (
    None
):
    api = AgentFakeApi()
    api.answer_expected = False
    draft_tts = FakeDraftTtsSession(None)
    session = CompanionSession(api, FakeTts(), draft_tts=draft_tts)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    turn_id = UUID(int=333)

    assert await session.submit_voice_turn(
        turn_id,
        revision=5,
        promote_revision=4,
        content="批准这个任务",
    )
    await session.wait_for_idle()

    assert api.answer_session_ids == []
    assert draft_tts.cancels == [(VoiceDraftKey(FISH_ID, SESSION_ID, turn_id), 4)]
    assert session.state is CompanionState.LISTENING
    await session.close()


@pytest.mark.asyncio
async def test_context_ttl_without_queue_keeps_explicit_voice_conversation_open() -> (
    None
):
    api = AgentFakeApi()
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reply_context_ttl_seconds=0.01,
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await wait_until(lambda: session.reply_context is not None)
    conversation_id = session.voice_conversation_id

    await wait_until(lambda: session.reply_context is None)

    assert session.voice_conversation_id == conversation_id
    assert session.session_id == message.session_id
    assert session.state is CompanionState.LISTENING
    await session.close()


@pytest.mark.asyncio
async def test_rejected_final_can_release_context_reservation_for_next_turn() -> None:
    api = AgentFakeApi()
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(agent_message(1))
    await wait_until(lambda: session.reply_context is not None)

    rejected = session.freeze_turn_binding()
    assert rejected is not None and rejected.reply_context is not None
    session.release_frozen_turn(rejected)
    retried = session.freeze_turn_binding()

    assert retried is not None
    assert retried.reply_context == rejected.reply_context
    await session.close()


@pytest.mark.asyncio
async def test_wake_and_answer_each_hold_and_release_their_playback_owner() -> None:
    api = FakeApi()
    tts = BlockingWakeAckTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    await session.set_binding(FISH_ID)

    assert await session.on_wake_word()
    await asyncio.wait_for(tts.append_started.wait(), timeout=1)
    assert arbiter.current_owner is not None
    assert arbiter.current_owner.kind == "wake"
    assert arbiter.current_spoken_text == "我在呢。"

    tts.release_append.set()
    await session.wait_for_idle()
    assert arbiter.current_owner is None
    assert await session.submit_transcript("继续")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "新的回答。"},
        }
    )
    await wait_until(lambda: session.state is CompanionState.SPEAKING)
    assert arbiter.current_owner is not None
    assert arbiter.current_owner.kind == "answer"
    assert arbiter.current_spoken_text == "新的回答。"

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await session.wait_for_idle()
    assert arbiter.current_owner is None
    assert arbiter.current_spoken_text == ""
    await session.close()


@pytest.mark.asyncio
async def test_home_status_uses_home_owner_after_answer_releases() -> None:
    api = FakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        home_actions_enabled=True,
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("打开玄关灯")
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await api.home_events.put(
        {
            "event": "home.action.failed",
            "data": {
                "sourceMessageId": str(ANSWER_ID),
                "status": "failed",
                "action": {"entityId": "light.entry", "operation": "turn_on"},
                "result": {"error": "offline"},
            },
        }
    )
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)

    assert arbiter.current_owner is not None
    assert arbiter.current_owner.kind == "home"
    assert arbiter.current_spoken_text == "设备操作没有成功。"
    tts.release_finish.set()
    await session.wait_for_idle()
    assert arbiter.current_owner is None
    await session.close()


@pytest.mark.asyncio
async def test_home_waiter_does_not_speak_while_agent_context_is_active() -> None:
    api = AgentFakeApi()
    tts = FullBlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reply_context_ttl_seconds=0.05,
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="A任务结果已经完整播报。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    assert session.reply_context is not None
    home_text = "设备操作没有成功。"
    await session._status_speech_queue.put(  # noqa: SLF001
        (session.binding_generation, FISH_ID, UUID(int=909), home_text)
    )
    session._ensure_status_speech_worker()  # noqa: SLF001
    await asyncio.sleep(0)
    tts.release_finish.set()

    await asyncio.sleep(0.02)
    assert ("append", home_text) not in tts.calls
    await wait_until(lambda: ("append", home_text) in tts.calls)
    await session.close()


@pytest.mark.asyncio
async def test_home_rechecks_turn_idle_after_waiting_for_agent_context() -> None:
    api = BlockingAgentAskApi()
    tts = FullFrameTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="A任务结果已经完整播报。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await wait_until(lambda: session.reply_context is not None)
    home_text = "设备操作没有成功。"
    await session._status_speech_queue.put(  # noqa: SLF001
        (session.binding_generation, FISH_ID, UUID(int=910), home_text)
    )
    session._ensure_status_speech_worker()  # noqa: SLF001
    await asyncio.sleep(0)

    assert await session.submit_transcript("这是对A的回复") is True
    await asyncio.wait_for(api.ask_started.wait(), timeout=1)
    api.release_ask.set()
    await wait_until(lambda: session.reply_context is None)
    await asyncio.sleep(0.02)
    assert session.state is CompanionState.THINKING
    assert ("append", home_text) not in tts.calls

    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await wait_until(lambda: ("append", home_text) in tts.calls)
    await session.close()


@pytest.mark.asyncio
async def test_agent_spoken_text_is_the_only_echo_source_and_clears_on_release() -> (
    None
):
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="任务已经完成，可以查看结果。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)

    assert session.is_likely_echo("任务已经完成") is True
    tts.release_finish.set()
    await wait_until(lambda: arbiter.current_owner is None)
    assert session.is_likely_echo("任务已经完成") is False
    await session.close()


@pytest.mark.asyncio
async def test_promoted_draft_receives_the_same_answer_owner_token() -> None:
    api = FakeApi()
    arbiter = PlaybackArbiter()
    promoted_output = FakeTts()
    draft_tts = FakeDraftTtsSession(FakeDraftPromotion(promoted_output, 4))
    session = CompanionSession(
        api,
        FakeTts(),
        draft_tts=draft_tts,
        playback_arbiter=arbiter,
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    assert await session.submit_voice_turn(
        UUID(int=444),
        revision=5,
        promote_revision=4,
        content="最终问题",
    )
    await api.answer_events.put(
        {
            "event": "answer.snapshot",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": "前缀后续。",
                "flushOffset": 4,
            },
        }
    )
    await wait_until(lambda: bool(draft_tts.promotion_owners))

    owner = draft_tts.promotion_owners[0]
    assert owner is not None
    assert owner.kind == "answer"
    assert owner.message_id == ANSWER_ID
    assert arbiter.current_owner == owner
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await session.wait_for_idle()
    assert arbiter.current_owner is None
    await session.close()


@pytest.mark.asyncio
async def test_unrelated_reply_promotes_only_the_frozen_context_identity() -> None:
    api = AgentFakeApi()
    arbiter = PlaybackArbiter()
    ordinary = CallbackTts()
    promoted_output = FakeTts()
    draft_tts = FakeDraftTtsSession(FakeDraftPromotion(promoted_output, 4))
    session = CompanionSession(
        api,
        ordinary,
        draft_tts=draft_tts,
        playback_arbiter=arbiter,
    )
    ordinary.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    source = agent_message(1, content="原任务结果。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(source)
    await wait_until(lambda: session.reply_context is not None)
    frozen = session.freeze_turn_binding()
    assert frozen is not None and frozen.reply_context is not None

    assert await session.submit_voice_turn(
        UUID(int=445),
        revision=5,
        promote_revision=4,
        content="这是一条无关的新问题",
        frozen_turn=frozen,
    )
    await api.answer_events.put(
        {
            "event": "answer.snapshot",
            "data": {
                "messageId": str(ANSWER_ID),
                "content": "前缀后续。",
                "flushOffset": 4,
            },
        }
    )
    await wait_until(lambda: bool(draft_tts.promotion_owners))

    assert api.voice_reply_contexts == [
        CompanionReplyContext("run_result", source.subject_id)
    ]
    assert draft_tts.promotions[0][1] == 4
    assert draft_tts.promotion_owners[0] == arbiter.current_owner
    await api.answer_events.put(
        {
            "event": "answer.completed",
            "data": {"messageId": str(ANSWER_ID), "end": False},
        }
    )
    await session.wait_for_idle()
    await session.close()


@pytest.mark.asyncio
async def test_agent_resume_uses_absolute_offset_and_segmented_tts_ranges() -> None:
    api = AgentFakeApi()
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    prefix = "已经播过"
    first_segment = "这是恢复后的第一段内容而且长度已经足够了。"
    second_segment = "这里是恢复后的第二段内容同样足够长。"
    message = agent_message(
        1,
        content=prefix + first_segment + second_segment,
        consumed_offset=len(prefix),
    )

    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert ("start", len(prefix)) in tts.calls
    assert [call for call in tts.calls if call[0] == "append"] == [
        ("append", first_segment),
        ("append", second_segment),
    ]
    assert api.agent_consumption_calls == [
        (message.message_id, len(message.content), False)
    ]
    await session.close()


@pytest.mark.asyncio
async def test_partial_approval_replays_current_segment_without_context() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    approval = agent_message(
        1,
        content="是否允许执行这个需要明确授权的操作？",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(approval)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    reservation = session.reserve_foreground_speech()
    tts.played_offset = 5

    await session.interrupt_from_speech()
    assert session.reply_context is None
    assert api.agent_consumption_calls == [(approval.message_id, 0, False)]
    assert api.interrupt_calls == []

    session.release_foreground_speech(reservation)
    await wait_until(lambda: sum(call[0] == "start" for call in tts.calls) >= 2)
    await wait_until(lambda: len(api.agent_consumption_calls) == 2)
    assert api.agent_consumption_calls[-1] == (
        approval.message_id,
        len(approval.content),
        False,
    )
    assert session.reply_context is not None
    assert session.reply_context.source_message_id == approval.message_id
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("played_offset", [0, 4])
async def test_approval_finish_uses_actual_offset_and_requeues_partial(
    played_offset: int,
) -> None:
    api = AgentFakeApi()
    tts = PartialThenFullTts(played_offset)
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    approval = agent_message(
        1,
        content="是否允许执行这个需要明确授权的操作？",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(approval)

    await asyncio.wait_for(tts.second_start.wait(), timeout=1)
    assert api.agent_consumption_calls[0] == (
        approval.message_id,
        played_offset,
        False,
    )
    assert session.reply_context is None
    assert ("start", played_offset or None) in tts.calls

    tts.release_second_start.set()
    await wait_until(lambda: len(api.agent_consumption_calls) == 2)
    assert api.agent_consumption_calls[-1] == (
        approval.message_id,
        len(approval.content),
        False,
    )
    assert session.reply_context is not None
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["start", "append", "finish"])
async def test_agent_tts_failure_requeues_safe_segment_start_before_later_message(
    operation: str,
) -> None:
    api = AgentFakeApi()
    tts = FailOnceThenFullTts(operation)
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    first = agent_message(
        1,
        content="第一条需要完整播报的审批问题。",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )
    later = agent_message(2, content="第二条不能越过第一条。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(first)
    await api.agent_events.put(later)

    await asyncio.wait_for(tts.second_start.wait(), timeout=1)
    expected_offset = 0
    assert api.agent_consumption_calls[0] == (
        first.message_id,
        expected_offset,
        False,
    )
    assert ("append", later.content) not in tts.calls
    assert ("start", expected_offset or None) in tts.calls

    tts.release_second_start.set()
    await wait_until(lambda: len(api.agent_consumption_calls) == 2)
    assert session.reply_context is not None
    await session.close()


@pytest.mark.asyncio
async def test_agent_append_error_replays_current_segment_from_its_start() -> None:
    api = AgentFakeApi()
    tts = PacketTrackedAgentTts("append_error")
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reconnect_delays_seconds=(1.0,),
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    prefix = "已经确认的前缀。"
    approval = agent_message(
        1,
        content=prefix + "这一整段审批问题不能因为一个PCM字节就算播放完成。",
        consumed_offset=len(prefix),
        subject_kind="interaction",
        activation_policy="playback_complete",
    )

    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(approval)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert tts.played_offset == len(prefix)
    assert tts._tracker.display_offset == len(approval.content)
    assert api.agent_consumption_calls == [(approval.message_id, len(prefix), False)]
    assert session.reply_context is None
    barrier = session._agent_progress_barrier  # noqa: SLF001
    assert barrier is not None
    assert barrier.consumed_offset == len(prefix)
    await session.close()


@pytest.mark.asyncio
async def test_approval_barge_in_replays_partially_heard_segment() -> None:
    api = AgentFakeApi()
    tts = PacketTrackedAgentTts("block_second")
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reconnect_delays_seconds=(1.0,),
    )
    prefix = "已经确认的前缀。"
    first_segment = "第一段审批说明已经完整播放并且长度足够了。"
    second_segment = "第二段审批问题仍然只听到了第一个PCM字节。"
    approval = agent_message(
        1,
        content=prefix + first_segment + second_segment,
        consumed_offset=len(prefix),
        subject_kind="interaction",
        activation_policy="playback_complete",
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(approval)
    await asyncio.wait_for(tts.packet_consumed.wait(), timeout=1)
    reservation = session.reserve_foreground_speech()

    await session.interrupt_from_speech()

    assert tts._tracker.display_offset == len(approval.content)
    safe_offset = len(prefix) + len(first_segment)
    assert api.agent_consumption_calls == [(approval.message_id, safe_offset, False)]
    assert session.reply_context is None
    session.release_foreground_speech(reservation)
    await session.close()


@pytest.mark.asyncio
async def test_partial_failure_clears_first_frame_context_before_requeue() -> None:
    api = AgentFakeApi()
    tts = FirstFrameThenFailTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="需要补充信息的任务问题。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)

    await asyncio.wait_for(tts.second_start.wait(), timeout=1)
    assert session.reply_context is None
    assert api.agent_consumption_calls[0] == (message.message_id, 0, False)

    tts.release_second_start.set()
    await wait_until(lambda: session.reply_context is not None)
    await session.close()


@pytest.mark.asyncio
async def test_partial_progress_retry_survives_seen_eviction_and_blocks_b() -> None:
    api = AgentFakeApi()
    api.agent_consumption_error = RuntimeError("offline")
    tts = PartialThenFullTts(4)
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reconnect_delays_seconds=(0.05,),
    )
    first = agent_message(
        1,
        content="第一条审批问题不能被越过。",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )
    later = agent_message(2, content="第二条不能越过第一条。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(first)
    await api.agent_events.put(later)
    await wait_until(lambda: bool(api.agent_consumption_calls))
    for sequence in range(10, 10 + 2050):
        session._accept_agent_message(  # noqa: SLF001
            agent_message(
                sequence,
                content="已经完整消费。",
                consumed_offset=-1,
            )
        )
    assert first.message_id not in session._agent_seen_ids  # noqa: SLF001
    assert ("append", later.content) not in tts.calls

    api.agent_consumption_error = None
    await asyncio.wait_for(tts.second_start.wait(), timeout=1)
    assert ("start", 4) in tts.calls
    assert ("append", later.content) not in tts.calls

    tts.release_second_start.set()
    await wait_until(lambda: session.reply_context is not None)
    assert session.reply_context.source_message_id == first.message_id
    await session.close()


@pytest.mark.asyncio
async def test_same_message_barrier_upgrade_replaces_retry_without_stalling() -> None:
    api = AgentFakeApi()
    session = CompanionSession(
        api,
        FakeTts(),
        agent_reconnect_delays_seconds=(0.0,),
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    reservation = session.reserve_foreground_speech()
    message = agent_message(1, content="同一条消息需要继续恢复。")
    old_barrier = _AgentProgressBarrier(
        message=message,
        consumed_offset=3,
        resume_playback=True,
        activate_context=False,
        persist_required=False,
    )
    upgraded_barrier = _AgentProgressBarrier(
        message=message,
        consumed_offset=5,
        resume_playback=True,
        activate_context=False,
        persist_required=True,
    )

    session._block_on_agent_progress(old_barrier)  # noqa: SLF001
    session._block_on_agent_progress(upgraded_barrier)  # noqa: SLF001

    await wait_until(lambda: bool(api.agent_consumption_calls))
    assert api.agent_consumption_calls == [(message.message_id, 5, False)]
    assert session._agent_progress_barrier is None  # noqa: SLF001
    assert [item.consumed_offset for item in session._agent_queue] == [  # noqa: SLF001
        5
    ]
    session.release_foreground_speech(reservation)
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["silent", "start_fail"])
async def test_no_progress_playback_retry_keeps_capped_backoff(mode: str) -> None:
    api = AgentFakeApi()
    tts = TimedNoProgressTts(mode)
    session = CompanionSession(
        api,
        tts,
        agent_reconnect_delays_seconds=(0.01, 0.03, 0.06),
    )
    message = agent_message(
        1,
        content="审批问题持续没有音频进展。",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)

    await asyncio.wait_for(tts.fourth_start.wait(), timeout=1)
    intervals = [
        right - left
        for left, right in zip(
            tts.started_at,
            tts.started_at[1:],
            strict=False,
        )
    ]
    assert intervals[0] >= 0.008
    assert intervals[1] >= 0.025
    assert intervals[2] >= 0.05
    assert intervals == sorted(intervals)

    tts.release_fourth_start.set()
    await session.close()


@pytest.mark.asyncio
async def test_contextual_ask_syncs_failed_full_progress_before_send() -> None:
    api = AgentFakeApi()
    api.agent_consumption_error = RuntimeError("offline")
    api.answer_expected = False
    tts = FullFrameTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reconnect_delays_seconds=(10.0,),
    )
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="请回复这条完整播报的问题。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await wait_until(lambda: session.reply_context is not None)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert await session.submit_transcript("第一次回复") is False
    assert api.ask_calls == []

    api.agent_consumption_error = None
    assert await session.submit_transcript("第二次回复") is True
    await session.wait_for_idle()
    assert api.agent_consumption_calls[-1] == (
        message.message_id,
        len(message.content),
        False,
    )
    assert api.ask_calls == [(FISH_ID, "第二次回复")]
    await session.close()


@pytest.mark.asyncio
async def test_approval_context_activates_only_after_drain_and_full_progress() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    approval = agent_message(
        1,
        content="是否允许执行这个操作？",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )

    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(approval)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    assert session.reply_context is None

    tts.release_finish.set()
    await wait_until(lambda: bool(api.agent_consumption_calls))
    assert session.reply_context is not None
    assert session.reply_context.subject_kind == "interaction"
    assert session.reply_context.subject_id == approval.subject_id
    await session.close()


@pytest.mark.asyncio
async def test_approval_progress_failure_does_not_activate_context() -> None:
    api = AgentFakeApi()
    api.agent_consumption_error = RuntimeError("offline")
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    approval = agent_message(
        1,
        content="是否允许执行这个操作？",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )

    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(approval)
    await wait_until(lambda: bool(api.agent_consumption_calls))

    assert session.reply_context is None
    await session.close()


@pytest.mark.asyncio
async def test_failed_approval_progress_retries_before_later_message() -> None:
    api = AgentFakeApi()
    api.agent_consumption_error = RuntimeError("offline")
    tts = CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    approval = agent_message(
        1,
        content="是否允许执行这个操作？",
        subject_kind="interaction",
        activation_policy="playback_complete",
    )
    later = agent_message(2, content="稍后的任务结果。")

    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(approval)
    await api.agent_events.put(later)
    await wait_until(lambda: len(api.agent_consumption_calls) == 1)
    assert ("append", later.content) not in tts.calls
    assert session.reply_context is None

    api.agent_consumption_error = None
    await wait_until(lambda: len(api.agent_consumption_calls) >= 2)
    assert session.reply_context is not None
    assert session.reply_context.source_message_id == approval.message_id
    assert ("append", later.content) not in tts.calls
    await session.close()


@pytest.mark.asyncio
async def test_agent_partial_finalize_failure_preserves_context_and_sends_no_ask() -> (
    None
):
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="还需要你补充一个信息。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    assert session.reply_context is not None
    api.agent_consumption_error = RuntimeError("offline")
    tts.played_offset = 4

    await session.interrupt_from_speech()
    accepted = await session.submit_transcript("这是补充信息")

    assert accepted is False
    assert api.ask_calls == []
    assert api.interrupt_calls == []
    assert session.reply_context is not None
    assert api.agent_consumption_calls[-1] == (message.message_id, 0, True)

    api.agent_consumption_error = None
    api.answer_expected = False
    assert await session.submit_transcript("再次提交补充信息") is True
    await session.wait_for_idle()
    assert api.ask_calls == [(FISH_ID, "再次提交补充信息")]
    assert api.ask_reply_contexts == [
        CompanionReplyContext("run_result", message.subject_id)
    ]
    await session.close()


@pytest.mark.asyncio
async def test_post_finalize_state_race_releases_frozen_context_reservation() -> None:
    class BlockingConsumeApi(AgentFakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.consume_started = asyncio.Event()
            self.release_consume = asyncio.Event()

        async def consume_agent_message(
            self,
            fish_id: UUID,
            message_id: UUID,
            *,
            session_id: UUID,
            consumed_offset: int,
            interrupted: bool,
        ) -> None:
            self.consume_started.set()
            await self.release_consume.wait()
            await super().consume_agent_message(
                fish_id,
                message_id,
                session_id=session_id,
                consumed_offset=consumed_offset,
                interrupted=interrupted,
            )

    api = BlockingConsumeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="还需要你补充一个信息。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(message)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 4
    await session.interrupt_from_speech()
    frozen = session.freeze_turn_binding()
    assert frozen is not None and frozen.reply_context is not None

    submitting = asyncio.create_task(
        session.submit_transcript("补充信息", frozen_turn=frozen)
    )
    await asyncio.wait_for(api.consume_started.wait(), timeout=1)
    session._set_state(CompanionState.THINKING)  # noqa: SLF001
    api.release_consume.set()
    assert await submitting is False

    retried = session.freeze_turn_binding()
    assert retried is not None
    assert retried.reply_context == frozen.reply_context
    await session.close()


@pytest.mark.asyncio
async def test_audio_recovery_clears_agent_listener_cursor_and_queue() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    active = agent_message(1, content="当前任务结果还在播放。")
    later = agent_message(2, content="下一条任务结果。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await wait_until(lambda: session._agent_listener_task is not None)  # noqa: SLF001
    await api.agent_events.put(active)
    await api.agent_events.put(later)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 3

    assert await session.reset_for_audio_recovery(FISH_ID) is True

    assert session._agent_listener_task is None  # noqa: SLF001
    assert session._agent_listener_session_id is None  # noqa: SLF001
    assert session._agent_after_cursor == 0  # noqa: SLF001
    assert session._agent_queue == []  # noqa: SLF001
    assert session.reply_context is None
    assert session.session_id is None
    assert tts.calls.count(("append", active.content)) == 1
    assert ("append", later.content) not in tts.calls
    assert (active.message_id, 0, False) in api.agent_consumption_calls
    await session.close()


@pytest.mark.asyncio
async def test_audio_recovery_does_not_resume_old_session_agent_messages() -> None:
    api = AgentFakeApi()
    api.agent_consumption_error = RuntimeError("offline")
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(
        api,
        tts,
        playback_arbiter=arbiter,
        agent_reconnect_delays_seconds=(0.05,),
    )
    active = agent_message(1, content="第一条恢复后必须继续播放。")
    later = agent_message(2, content="第二条不能越过第一条。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(active)
    await api.agent_events.put(later)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 3

    assert await session.reset_for_audio_recovery(FISH_ID) is True
    assert ("append", later.content) not in tts.calls
    assert session._agent_listener_task is None  # noqa: SLF001
    assert session._agent_queue == []  # noqa: SLF001
    assert session._agent_progress_barrier is None  # noqa: SLF001
    assert session.session_id is None
    await session.close()


@pytest.mark.asyncio
async def test_fish_switch_stops_agent_and_clears_context_queue_and_conversation() -> (
    None
):
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    active = agent_message(1, content="当前任务结果还在播放。")
    queued = agent_message(2, content="下一条任务结果。")
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await api.agent_events.put(active)
    await api.agent_events.put(queued)
    await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
    tts.played_offset = 3
    stale_owner = cast(PlaybackOwner, arbiter.current_owner)

    await session.set_binding(SECOND_FISH_ID)
    session.on_playback_started(stale_owner)

    assert session.fish_id == SECOND_FISH_ID
    assert session.reply_context is None
    assert session.voice_conversation_id is None
    assert arbiter.current_owner is None
    assert (active.message_id, 0, False) in api.agent_consumption_calls
    assert ("append", queued.content) not in tts.calls
    assert not any(
        fish_id == SECOND_FISH_ID
        for fish_id, _session_id, _cursor in api.agent_connections
    )
    await session.close()


@pytest.mark.asyncio
async def test_quick_text_wakes_without_ack_and_uses_native_answer() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    result = await session.submit_text(FISH_ID, "hello from keyboard")
    assert result["messageId"] == str(ANSWER_ID)
    assert result["sessionId"] == str(SESSION_ID)
    assert api.ask_calls == [(FISH_ID, "hello from keyboard")]
    # The chat page marks it as typed, unlike what the pet heard.
    assert api.ask_input_modes == ["typed"]
    assert api.answer_session_ids == [SESSION_ID]
    assert not tts.calls  # no wake acknowledgement or duplicate web TTS
    assert session.state is CompanionState.THINKING
    await session.close()


@pytest.mark.asyncio
async def test_quick_text_waits_for_ask_acceptance_and_reports_failure() -> None:
    api = FakeApi()
    entered, release = asyncio.Event(), asyncio.Event()

    async def failing_ask(*args: object, **kwargs: object) -> CompanionAskResult:
        entered.set()
        await release.wait()
        raise ConnectionError("synthetic unavailable API")

    api.ask = failing_ask
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    task = asyncio.create_task(session.submit_text(FISH_ID, "keep this draft"))
    await entered.wait()
    assert not task.done()
    release.set()
    with pytest.raises(RuntimeError, match="草稿已保留"):
        await asyncio.wait_for(task, timeout=1)
    assert session.state is CompanionState.LISTENING
    await session.close()


@pytest.mark.asyncio
async def test_quick_text_rejects_a_stale_pet_before_any_ask() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    with pytest.raises(ValueError, match="宠物已切换"):
        await session.submit_text(SECOND_FISH_ID, "do not send")
    assert not api.ask_calls
    await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("while_speaking", [False, True])
async def test_quick_text_can_reply_to_agent_without_waiting_for_context_expiry(
    while_speaking: bool,
) -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts() if while_speaking else CallbackTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    tts.on_first_frame = lambda: session.on_playback_started(
        cast(PlaybackOwner, arbiter.current_owner)
    )
    message = agent_message(1, content="请告诉我下一步。")
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        await api.agent_events.put(message)
        await wait_until(lambda: session.reply_context is not None)
        if while_speaking:
            await tts.finish_started.wait()
            tts.played_offset = 3
        else:
            await wait_until(lambda: session.state is CompanionState.LISTENING)

        result = await asyncio.wait_for(
            session.submit_text(FISH_ID, "这是键盘回复"), timeout=1
        )

        assert result["messageId"] == str(ANSWER_ID)
        assert api.ask_calls == [(FISH_ID, "这是键盘回复")]
        assert api.ask_reply_contexts == [
            CompanionReplyContext("run_result", message.subject_id)
        ]
        if while_speaking:
            assert (message.message_id, 0, True) in api.agent_consumption_calls
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_quick_text_interrupts_plugin_speech_before_submitting() -> None:
    api = AgentFakeApi()
    tts = BlockingAgentTts()
    arbiter = PlaybackArbiter()
    session = CompanionSession(api, tts, playback_arbiter=arbiter)
    message = replace(proactive_message(1), source="plugin")
    try:
        await session.set_binding(FISH_ID)
        await session.on_wake_word(play_ack=False)
        await api.agent_events.put(message)
        await asyncio.wait_for(tts.finish_started.wait(), timeout=1)
        result = await asyncio.wait_for(
            session.submit_text(FISH_ID, "换个话题吧"), timeout=1
        )
        assert result["messageId"] == str(ANSWER_ID)
        assert api.ask_calls == [(FISH_ID, "换个话题吧")]
        assert api.ask_reply_contexts == [None]
        assert ("interrupt", None) in tts.calls
        assert session.state is CompanionState.THINKING
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_proactive_speech_also_publishes_completed_caption() -> None:
    class CaptionEvents(RecordingPetEvents):
        def __init__(self) -> None:
            super().__init__()
            self.captions: list[dict[str, object]] = []

        def playback_progress(self, payload: dict[str, object]) -> None:
            self.captions.append(payload)

    api = AgentFakeApi()
    events = CaptionEvents()
    session = CompanionSession(api, FakeTts(), pet_events=events)
    message = proactive_message(1)
    await session.set_binding(FISH_ID)
    await api.agent_events.put(message)
    await wait_until(lambda: any(p["status"] == "completed" for p in events.captions))
    assert events.captions[-1]["visibleContent"] == message.content
    assert events.captions[-1]["consumedOffset"] == len(message.content)
    assert all(p["messageSequence"] == message.sequence for p in events.captions)
    assert api.companion_consumption_sources == ["proactive_observation"]
    await session.close()


@pytest.mark.asyncio
async def test_stalled_draft_speech_falls_back_to_normal_synthesis() -> None:
    api = FakeApi()
    tts = FakeTts()
    session = CompanionSession(api, tts)
    await session.set_binding(FISH_ID)
    session._answer_message_id = ANSWER_ID
    session._assistant_text = "好好好，是我听错了嘛。"
    session._draft_tts_waiting = True

    await session._fall_back_from_stalled_draft()

    assert session._draft_tts_waiting is False
    assert session._tts_started is True
    assert tts.calls[0] == ("start", None)
    spoken = "".join(str(value) for kind, value in tts.calls if kind == "append")
    assert spoken.startswith("好好好，")  # later phrases follow as the answer streams
    await session.close()


@pytest.mark.asyncio
async def test_a_heard_sentence_is_marked_as_spoken() -> None:
    api = FakeApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await wait_until(lambda: bool(api.ask_calls))
    assert api.ask_input_modes == ["spoken"]
    await session.close()


class FailureCaptionEvents(RecordingPetEvents):
    def __init__(self) -> None:
        super().__init__()
        self.captions: list[dict[str, object]] = []

    def playback_progress(self, payload: dict[str, object]) -> None:
        self.captions.append(payload)


@pytest.mark.asyncio
async def test_failed_answer_shows_which_service_let_it_down() -> None:
    api = FakeApi()
    events = FailureCaptionEvents()
    session = CompanionSession(api, FakeTts(), pet_events=events)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.failed",
            "data": {"messageId": str(ANSWER_ID), "failure": "jev:timeout"},
        }
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)
    assert events.captions[-1]["status"] == "failed"
    assert events.captions[-1]["failure"] == "jev:timeout"
    await session.close()


@pytest.mark.asyncio
async def test_unspoken_reply_shows_and_keeps_why_it_was_not_heard() -> None:
    from fish_memory import failures

    class RefusingTts(FakeTts):
        async def finish(self) -> None:
            raise failures.ReportedFailure("bad text", "voice:rejected")

    class RecordingApi(FakeApi):
        def __init__(self) -> None:
            super().__init__()
            self.failures: list[tuple[object, ...]] = []

        async def record_failure(self, fish_id, message_id, failure) -> None:
            self.failures.append((fish_id, message_id, failure))

    api = RecordingApi()
    events = FailureCaptionEvents()
    session = CompanionSession(api, RefusingTts(), pet_events=events)
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    await session.submit_transcript("你好")
    await api.answer_events.put(
        {
            "event": "answer.delta",
            "data": {"messageId": str(ANSWER_ID), "content": "helello."},
        }
    )
    await api.answer_events.put(
        {"event": "answer.completed", "data": {"messageId": str(ANSWER_ID)}}
    )
    await asyncio.wait_for(session.wait_for_idle(), timeout=1)
    await wait_until(lambda: api.failures)
    assert events.captions[-1]["failure"] == "voice:rejected"
    assert api.failures == [(FISH_ID, ANSWER_ID, "voice:rejected")]
    await session.close()
