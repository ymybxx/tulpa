"""Content-free processor lifecycle events must not alter voice decisions."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pytest

from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.processor import (
    AsrInput,
    CompanionAudioProcessor,
    SmartTurnInput,
    _FinalizedTurn,
)
from fish_memory.companion.session import CompanionSession, CompanionState

from .test_processor import (
    FakeFrameProcessor,
    FakeInterruptionVad,
    FakeKws,
    FakeSmartTurn,
    FrameAsr,
    FrozenTurnSessionHarness,
    frozen_turn,
)


@dataclass(frozen=True)
class RuntimeEvent:
    name: str
    text: str | None
    metadata: dict[str, object]


class RuntimeEvents:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []
        self.frame_count = 0

    def record_frame(self, *_: bytes, **__: object) -> int:
        self.frame_count += 1
        return self.frame_count - 1

    def record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "processor",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        del frame_index, source
        self.events.append(RuntimeEvent(name, text, dict(metadata or {})))

    def named(self, name: str) -> list[RuntimeEvent]:
        return [event for event in self.events if event.name == name]


_DEFAULT_SNAPSHOT_UPDATE = AsrUpdate("私密内容。", is_final=True, event="Speech")


class SnapshotAsr(FrameAsr):
    def __init__(
        self,
        *,
        snapshot_update: AsrUpdate | None = _DEFAULT_SNAPSHOT_UPDATE,
        consumed: bool = True,
    ) -> None:
        super().__init__("在线最终内容", partial_text=None)
        self.snapshot_update = snapshot_update
        self.consumed = consumed
        self.consume_calls = 0

    def recognize_snapshot_update(self, _pcm: bytes) -> AsrUpdate | None:
        self.snapshot_calls += 1
        return self.snapshot_update

    def consume_snapshot(self, _pcm: bytes) -> bool:
        self.consume_calls += 1
        return self.consumed


def make_processor(
    *,
    state: CompanionState = CompanionState.LISTENING,
    asr: FrameAsr | None = None,
    voice_values: list[bool] | None = None,
    commit_delay: float = 0,
    smart_turn: SmartTurnInput | None = None,
    interruption_vad: FakeInterruptionVad | None = None,
) -> tuple[CompanionAudioProcessor, FrozenTurnSessionHarness, RuntimeEvents]:
    session = FrozenTurnSessionHarness(frozen_turn(generation=1))
    session.state = state
    events = RuntimeEvents()
    processor = CompanionAudioProcessor(
        cast(CompanionSession, session),
        FakeKws(),
        cast(AsrInput, asr or FrameAsr("完整问题", partial_text=None)),
        frame_processor=FakeFrameProcessor(voice_values or []),
        diagnostics=events,
        utterance_commit_delay_seconds=commit_delay,
        smart_turn=smart_turn,
        smart_turn_reopen_seconds=0.1,
        interruption_vad=interruption_vad,
        speech_start_frames=1,
        speech_end_frames=1,
    )
    return processor, session, events


@pytest.mark.parametrize(
    ("state", "old_flag", "expected"),
    [
        (CompanionState.LISTENING, True, False),
        (CompanionState.SPEAKING, False, True),
    ],
)
async def test_speech_boundaries_report_current_state_not_previous_candidate(
    state: CompanionState, old_flag: bool, expected: bool
) -> None:
    processor, _, events = make_processor(state=state, voice_values=[True, False])
    processor._speech_candidate_requires_confirmation = old_flag

    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)

    for name in ("speech_start", "speech_end"):
        event = events.named(name)[0]
        assert event.metadata["speechGeneration"] == 1
        assert event.metadata["state"] == state.value
        assert event.metadata["requiresConfirmation"] is expected
    await processor.close()


async def test_formal_snapshot_decode_events_have_counts_not_content() -> None:
    asr = SnapshotAsr()
    processor, session, events = make_processor(asr=asr)
    processor._speech_generation = 4

    await processor._finish_utterance_snapshot(
        record_speech_end=True,
        pcm_snapshot=b"a" * 320,
        expected_generation=4,
    )

    assert asr.snapshot_calls == 1
    assert asr.consume_calls == 1
    assert asr.finish_calls == 0
    assert session.submissions[0][0] == "私密内容。"
    started = events.named("asr_decode_started")[0]
    completed = events.named("asr_decode_completed")[0]
    assert started.text is None
    assert completed.text is None
    assert completed.metadata["speechGeneration"] == 4
    assert completed.metadata["decodeKind"] == "snapshot"
    assert completed.metadata["charCount"] == 5
    assert completed.metadata["compactChars"] == 4
    assert completed.metadata["asrEvent"] == "Speech"
    assert cast(float, completed.metadata["durationMs"]) >= 0
    assert "私密内容" not in repr((started, completed))
    # Existing private diagnostics retain their original final text contract.
    assert events.named("asr_final")[0].text == "私密内容。"


async def test_online_fallback_logs_only_existing_decode_calls() -> None:
    asr = SnapshotAsr(snapshot_update=None)
    processor, session, events = make_processor(asr=asr)
    processor._speech_generation = 3

    await processor._finish_utterance_snapshot(
        record_speech_end=False,
        pcm_snapshot=b"a" * 320,
        expected_generation=3,
    )

    assert asr.snapshot_calls == 1
    assert asr.finish_calls == 1
    assert asr.consume_calls == 0
    assert session.submissions[0][0] == "在线最终内容"
    assert [
        event.metadata["decodeKind"] for event in events.named("asr_decode_started")
    ] == ["snapshot", "finish"]
    completed = events.named("asr_decode_completed")
    assert [event.metadata["decodeKind"] for event in completed] == [
        "snapshot",
        "finish",
    ]
    assert completed[0].metadata["charCount"] == 0
    assert completed[0].metadata["asrEvent"] == ""


@pytest.mark.parametrize(
    ("generation", "consumed", "reason", "consume_calls"),
    [
        (5, True, "stale_generation", 0),
        (4, False, "snapshot_not_consumed", 1),
    ],
)
async def test_discarded_snapshot_keeps_generation_and_original_decisions(
    generation: int, consumed: bool, reason: str, consume_calls: int
) -> None:
    asr = SnapshotAsr(consumed=consumed)
    processor, session, events = make_processor(asr=asr)
    processor._speech_generation = generation

    await processor._finish_utterance_snapshot(
        record_speech_end=False,
        pcm_snapshot=b"a" * 320,
        expected_generation=4,
    )

    discarded = events.named("finish_discarded")[0]
    assert discarded.metadata["reason"] == reason
    assert discarded.metadata["speechGeneration"] == 4
    assert discarded.metadata["currentSpeechGeneration"] == generation
    assert discarded.text is None
    assert asr.snapshot_calls == 1
    assert asr.consume_calls == consume_calls
    assert asr.finish_calls == 0
    assert session.submissions == []
    assert events.named("asr_final") == []


async def test_restarted_speech_logs_cancelled_old_finish_generation() -> None:
    asr = SnapshotAsr()
    processor, _, events = make_processor(
        asr=asr, voice_values=[True, False, True], commit_delay=60
    )
    await processor.process_frame(b"a" * 320, b"r" * 320)
    await processor.process_frame(b"b" * 320, b"r" * 320)
    finish_task = processor._frame_finish_task
    assert finish_task is not None
    await processor.process_frame(b"c" * 320, b"r" * 320)
    await asyncio.sleep(0)

    scheduled = events.named("finish_scheduled")[0]
    assert scheduled.metadata["speechGeneration"] == 1
    assert scheduled.metadata["commitDelayMs"] == 60_000
    cancelled = events.named("finish_cancelled")[0]
    assert cancelled.metadata["speechGeneration"] == 1
    assert cancelled.metadata["currentSpeechGeneration"] == 2
    assert cancelled.metadata["reason"] == "speech_restart"
    assert finish_task.cancelled()
    assert asr.finish_calls == 0
    await processor.close()


@pytest.mark.parametrize("use_smart_turn", [False, True])
async def test_finish_waiting_reports_delay_reason_without_extra_decode(
    use_smart_turn: bool,
) -> None:
    asr = SnapshotAsr()
    processor, _, events = make_processor(
        asr=asr,
        commit_delay=0 if use_smart_turn else 0.1,
        smart_turn=(
            cast(SmartTurnInput, FakeSmartTurn(False)) if use_smart_turn else None
        ),
    )
    processor._speech_generation = 2

    await processor._finish_frame_utterance_after_delay(2)

    waiting = events.named("finish_waiting")
    delay_event = waiting[-1]
    assert delay_event.metadata["speechGeneration"] == 2
    assert delay_event.metadata["reason"] == (
        "smart_turn_incomplete" if use_smart_turn else "commit_delay"
    )
    assert 0 < cast(float, delay_event.metadata["remainingMs"]) <= 100
    if use_smart_turn:
        assert waiting[0].metadata["reason"] == "smart_turn_inference"
        assert waiting[0].metadata["timeoutMs"] == 500
    assert asr.snapshot_calls == 1
    assert asr.finish_calls == 0


async def test_stale_delayed_finish_is_discarded_before_decode() -> None:
    asr = SnapshotAsr()
    processor, session, events = make_processor(asr=asr)
    processor._speech_generation = 5

    await processor._finish_frame_utterance_after_delay(4)

    event = events.named("finish_discarded")[0]
    assert event.metadata["reason"] == "stale_generation"
    assert event.metadata["speechGeneration"] == 4
    assert event.metadata["currentSpeechGeneration"] == 5
    assert asr.snapshot_calls == 0
    assert asr.finish_calls == 0
    assert session.submissions == []


@pytest.mark.parametrize("reason", ["near_end_not_confirmed", "likely_echo"])
async def test_final_rejection_branches_emit_content_free_fixed_reasons(
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor, session, events = make_processor(state=CompanionState.SPEAKING)
    processor._speech_generation = 8
    monkeypatch.setattr(session, "is_likely_echo", lambda _: reason == "likely_echo")
    turn = _FinalizedTurn(
        content="私密影片声音",
        asr_event="Speech",
        speech_ms=500,
        requires_confirmation=True,
        confirmed=False,
        requires_near_end=reason == "near_end_not_confirmed",
        near_end_confirmed=False,
        frozen_turn=session.current_frozen_turn,
    )

    await processor._deliver_finalized_turn(turn)

    event = events.named("asr_final_rejected")[0]
    assert event.text is None
    assert event.metadata["reason"] == reason
    assert event.metadata["speechGeneration"] == 8
    assert "私密影片声音" not in repr(event)
    assert session.submissions == []


async def test_near_end_event_identifies_vad_and_is_not_per_frame() -> None:
    processor, _, events = make_processor(
        state=CompanionState.SPEAKING,
        interruption_vad=FakeInterruptionVad([True, True]),
    )
    processor._speech_generation = 3
    processor._interruption_requires_near_end = True
    await processor._observe_near_end(b"a" * 320)
    await processor._observe_near_end(b"b" * 320)

    near_end = events.named("near_end_confirmed")
    assert len(near_end) == 1
    assert near_end[0].metadata["detector"] == "silero_vad"
    assert near_end[0].metadata["speechGeneration"] == 3


def test_event_context_preserves_existing_metadata_and_private_text() -> None:
    processor, _, events = make_processor()
    original: dict[str, object] = {
        "speechGeneration": 19,
        "state": "provided",
        "rawText": "私密原文",
    }
    processor._record_event("asr_final", text="私密正文", metadata=original)

    assert events.events[0].text == "私密正文"
    assert events.events[0].metadata == original
    assert original == {
        "speechGeneration": 19,
        "state": "provided",
        "rawText": "私密原文",
    }
