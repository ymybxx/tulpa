import asyncio
import logging
from typing import cast
from uuid import UUID

import pytest

from fish_memory.companion.performance import (
    CompanionLatencyTracker,
    CompositeAudioDiagnostics,
    PetVoicePerformanceReporter,
    PetVoiceStatus,
    summarize_voice_draft_attempt,
)

USER_1 = UUID("00000000-0000-0000-0000-000000000001")
ANSWER_1 = UUID("00000000-0000-0000-0000-000000000101")
USER_2 = UUID("00000000-0000-0000-0000-000000000002")
ANSWER_2 = UUID("00000000-0000-0000-0000-000000000102")


class ManualClock:
    def __init__(self) -> None:
        self.at_ms = 0.0

    def set(self, at_ms: float) -> None:
        self.at_ms = at_ms

    def now_ns(self) -> int:
        return round(self.at_ms * 1_000_000)


def record(
    tracker: CompanionLatencyTracker,
    clock: ManualClock,
    name: str,
    at_ms: float,
    *,
    message_id: UUID | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    clock.set(at_ms)
    event_metadata = dict(metadata or {})
    if message_id is not None:
        event_metadata["messageId"] = str(message_id)
    tracker.record_event(name, metadata=event_metadata or None)


def test_complete_turn_reports_the_full_pet_voice_chain() -> None:
    clock = ManualClock()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)
    record(tracker, clock, "speech_start", 0.0)
    record(tracker, clock, "speech_end", 100.0)
    record(tracker, clock, "asr_final", 148.5)
    record(tracker, clock, "ask_submitted", 156.4, message_id=USER_1)
    record(tracker, clock, "answer_event_received", 1100.0, message_id=ANSWER_1)
    record(tracker, clock, "answer_first_content", 1419.0, message_id=ANSWER_1)
    record(tracker, clock, "tts_start_requested", 1420.0, message_id=ANSWER_1)
    record(
        tracker,
        clock,
        "tts_ready",
        1460.0,
        metadata={
            "connectionReused": True,
            "provider": "dashscope",
            "model": "qwen-audio-3.0-tts-flash",
            "voice": "longanhuan_v3.6",
        },
    )
    record(tracker, clock, "tts_text_submitted", 1462.0)
    record(tracker, clock, "tts_first_audio_received", 1630.0)
    record(tracker, clock, "tts_first_pcm_queued", 1631.0)
    record(tracker, clock, "tts_playback_started", 1663.6)
    record(tracker, clock, "tts_playback_stopped", 1700.0, message_id=ANSWER_1)

    assert tracker.metrics_for(ANSWER_1, PetVoiceStatus.COMPLETED) == {
        "status": "completed",
        "userMessageId": str(USER_1),
        "answerMessageId": str(ANSWER_1),
        "provider": "dashscope",
        "model": "qwen-audio-3.0-tts-flash",
        "voice": "longanhuan_v3.6",
        "connectionReused": True,
        "nativeMs": {
            "endpointToAsrFinal": 48.5,
            "endpointToAsk": 56.4,
            "askToAnswerEvent": 943.6,
            "askToFirstContent": 1262.6,
            "firstContentToTtsStart": 1.0,
            "ttsStartToReady": 40.0,
            "ttsReadyToTextSubmitted": 2.0,
            "ttsTextToFirstAudio": 168.0,
            "firstAudioToPcmQueued": 1.0,
            "pcmQueuedToPlayback": 32.6,
            "askToPlayback": 1507.2,
            "endpointToPlayback": 1563.6,
        },
    }


def test_draft_metrics_report_useful_hit_and_discarded_bytes() -> None:
    metrics = summarize_voice_draft_attempt(
        started_ms=200.0,
        commit_ms=480.0,
        first_content_ms=430.0,
        promoted=True,
        discarded_pcm_bytes=0,
    )

    assert metrics == {
        "draftLeadMs": 280.0,
        "draftFirstContentLeadMs": 50.0,
        "draftUsefulHit": True,
        "draftPromoted": True,
        "draftDiscardedPcmBytes": 0,
    }


def test_pet_voice_persists_first_audio_breakdown_and_separate_sentence_waits():
    clock = ManualClock()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)
    record(tracker, clock, "ask_submitted", 0, message_id=USER_1)
    record(tracker, clock, "answer_event_received", 20, message_id=ANSWER_1)
    record(
        tracker,
        clock,
        "tts_first_audio_timing",
        1200,
        metadata={"providerFirstPacketMs": 400, "facialWaitMs": 0},
    )
    record(
        tracker, clock, "tts_facial_ready", 1200, metadata={"waitMs": 0, "endOffset": 5}
    )
    record(
        tracker,
        clock,
        "tts_facial_ready",
        1800,
        metadata={"waitMs": 600, "endOffset": 10},
    )
    record(
        tracker,
        clock,
        "tts_facial_ready",
        2000,
        metadata={"waitMs": 200, "endOffset": 15, "errorType": "TimeoutError"},
    )
    metrics = tracker.metrics_for(ANSWER_1, PetVoiceStatus.COMPLETED)
    assert metrics["nativeMs"]["ttsProviderFirstAudio"] == 400
    assert metrics["nativeMs"]["ttsFacialWaitForFirstAudio"] == 0
    assert metrics["facial"] == {
        "sentenceCount": 3,
        "waitedSentenceCount": 2,
        "failedSentenceCount": 1,
        "totalWaitMs": 800,
        "maxWaitMs": 600,
    }


def test_draft_first_content_after_commit_keeps_signed_negative_lead() -> None:
    metrics = summarize_voice_draft_attempt(
        started_ms=200.0,
        commit_ms=480.0,
        first_content_ms=500.0,
        promoted=True,
        discarded_pcm_bytes=0,
    )

    assert metrics["draftFirstContentLeadMs"] == -20.0
    assert metrics["draftUsefulHit"] is False


def test_draft_metrics_select_promoted_revision_without_reusing_old_content() -> None:
    clock = ManualClock()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)
    turn_id = USER_1
    record(tracker, clock, "speech_start", 0.0)
    record(
        tracker,
        clock,
        "voice_draft_started",
        200.0,
        metadata={"turnId": str(turn_id), "revision": 1},
    )
    record(
        tracker,
        clock,
        "voice_draft_first_content",
        300.0,
        metadata={"turnId": str(turn_id), "revision": 1},
    )
    record(
        tracker,
        clock,
        "voice_draft_canceled",
        350.0,
        metadata={"turnId": str(turn_id), "revision": 2, "attemptRevision": 1},
    )
    record(
        tracker,
        clock,
        "voice_draft_started",
        400.0,
        metadata={"turnId": str(turn_id), "revision": 3},
    )
    record(tracker, clock, "ask_submitted", 480.0, message_id=turn_id)
    record(tracker, clock, "answer_event_received", 490.0, message_id=ANSWER_1)
    record(
        tracker,
        clock,
        "voice_draft_first_content",
        500.0,
        metadata={"turnId": str(turn_id), "revision": 3},
    )
    record(
        tracker,
        clock,
        "voice_draft_promoted",
        510.0,
        metadata={"turnId": str(turn_id), "revision": 3, "promoted": True},
    )

    metrics = tracker.metrics_for(ANSWER_1, PetVoiceStatus.COMPLETED)

    assert metrics is not None
    draft = cast(dict[str, object], metrics["voiceDraft"])
    assert draft["selectedRevision"] == 3
    assert draft["draftFirstContentLeadMs"] == -20.0
    assert draft["draftUsefulHit"] is False
    assert draft["attempts"] == [
        {
            "revision": 1,
            "startedCount": 1,
            "firstContentCount": 1,
            "promotedCount": 0,
            "canceledCount": 1,
            "fallbackCount": 0,
            "lateEventDroppedCount": 0,
            "draftFirstContentLeadMs": 180.0,
            "draftUsefulHit": False,
        },
        {
            "revision": 3,
            "startedCount": 1,
            "firstContentCount": 1,
            "promotedCount": 1,
            "canceledCount": 0,
            "fallbackCount": 0,
            "lateEventDroppedCount": 0,
            "draftFirstContentLeadMs": -20.0,
            "draftUsefulHit": False,
        },
    ]


def test_late_old_turn_event_never_attaches_to_active_answer() -> None:
    clock = ManualClock()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)
    record(tracker, clock, "speech_start", 0.0)
    record(
        tracker,
        clock,
        "voice_draft_started",
        10.0,
        metadata={"turnId": str(USER_1), "revision": 1},
    )
    record(tracker, clock, "ask_submitted", 20.0, message_id=USER_1)
    record(tracker, clock, "answer_event_received", 30.0, message_id=ANSWER_1)
    record(tracker, clock, "speech_start", 40.0)
    record(
        tracker,
        clock,
        "voice_draft_started",
        50.0,
        metadata={"turnId": str(USER_2), "revision": 1},
    )
    record(tracker, clock, "ask_submitted", 60.0, message_id=USER_2)
    record(tracker, clock, "answer_event_received", 70.0, message_id=ANSWER_2)
    record(
        tracker,
        clock,
        "voice_draft_late_event_dropped",
        80.0,
        metadata={"turnId": str(USER_1), "revision": 1, "reason": "stale_owner"},
    )

    first = tracker.metrics_for(ANSWER_1, PetVoiceStatus.COMPLETED)
    second = tracker.metrics_for(ANSWER_2, PetVoiceStatus.COMPLETED)

    assert first is not None and second is not None
    first_draft = cast(dict[str, object], first["voiceDraft"])
    second_draft = cast(dict[str, object], second["voiceDraft"])
    assert first_draft["lateEventDroppedCount"] == 1
    assert second_draft["lateEventDroppedCount"] == 0


def test_unmatched_draft_turn_state_is_bounded() -> None:
    tracker = CompanionLatencyTracker(max_draft_turns=2)

    for value in (1, 2, 3):
        tracker.record_event(
            "voice_draft_started",
            metadata={"turnId": str(UUID(int=value)), "revision": 1},
        )

    assert tracker.unmatched_draft_turn_count == 2


def test_turn_metrics_include_real_draft_lifecycle_counts() -> None:
    clock = ManualClock()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)
    draft_metadata: dict[str, object] = {
        "turnId": str(USER_1),
        "revision": 1,
    }
    record(tracker, clock, "speech_start", 0.0)
    record(
        tracker,
        clock,
        "voice_draft_started",
        200.0,
        metadata=draft_metadata,
    )
    record(
        tracker,
        clock,
        "voice_draft_transcript_ready",
        230.0,
        metadata={**draft_metadata, "charCount": 4, "utf8Bytes": 12},
    )
    record(
        tracker,
        clock,
        "voice_draft_request_started",
        240.0,
        metadata=draft_metadata,
    )
    record(
        tracker,
        clock,
        "voice_draft_first_content",
        430.0,
        metadata=draft_metadata,
    )
    record(tracker, clock, "speech_end", 450.0)
    record(tracker, clock, "ask_submitted", 480.0, message_id=USER_1)
    record(tracker, clock, "answer_event_received", 500.0, message_id=ANSWER_1)
    record(
        tracker,
        clock,
        "voice_draft_promoted",
        510.0,
        metadata={**draft_metadata, "promoted": True},
    )
    record(
        tracker,
        clock,
        "voice_draft_tts_pcm_discarded",
        520.0,
        metadata={**draft_metadata, "pcmBytes": 640},
    )
    record(
        tracker,
        clock,
        "voice_draft_cleanup",
        530.0,
        metadata={**draft_metadata, "reason": "promoted"},
    )

    metrics = tracker.metrics_for(ANSWER_1, PetVoiceStatus.COMPLETED)

    assert metrics is not None
    assert metrics["voiceDraft"] == {
        "draftLeadMs": 280.0,
        "draftFirstContentLeadMs": 50.0,
        "draftUsefulHit": True,
        "draftPromoted": True,
        "draftDiscardedPcmBytes": 640,
        "selectedRevision": 1,
        "startedCount": 1,
        "transcriptReadyCount": 1,
        "requestStartedCount": 1,
        "firstContentCount": 1,
        "promotedCount": 1,
        "canceledCount": 0,
        "fallbackCount": 0,
        "lateEventDroppedCount": 0,
        "cleanupCount": 1,
        "ttsProviderReadyCount": 0,
        "ttsTextSubmittedCount": 0,
        "ttsFirstAudioBufferedCount": 0,
        "ttsPcmPromotedCount": 0,
        "ttsPcmDiscardedCount": 1,
        "attempts": [
            {
                "revision": 1,
                "startedCount": 1,
                "firstContentCount": 1,
                "promotedCount": 1,
                "canceledCount": 0,
                "fallbackCount": 0,
                "lateEventDroppedCount": 0,
                "draftFirstContentLeadMs": 50.0,
                "draftUsefulHit": True,
            }
        ],
    }


def test_reopened_speech_does_not_replace_the_previous_turn_endpoint() -> None:
    clock = ManualClock()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)

    record(tracker, clock, "speech_start", 0.0)
    record(tracker, clock, "speech_end", 100.0)
    record(tracker, clock, "asr_final", 120.0)
    record(tracker, clock, "ask_submitted", 130.0, message_id=USER_1)
    record(tracker, clock, "answer_event_received", 200.0, message_id=ANSWER_1)
    record(tracker, clock, "answer_first_content", 220.0, message_id=ANSWER_1)
    record(tracker, clock, "tts_start_requested", 221.0, message_id=ANSWER_1)
    record(tracker, clock, "tts_ready", 230.0)
    record(tracker, clock, "tts_text_submitted", 240.0)
    record(tracker, clock, "tts_first_audio_received", 300.0)
    record(tracker, clock, "tts_first_pcm_queued", 301.0)
    record(tracker, clock, "tts_playback_started", 310.0)

    record(tracker, clock, "speech_start", 320.0)
    record(tracker, clock, "speech_end", 400.0)
    record(tracker, clock, "asr_final", 420.0)
    record(tracker, clock, "tts_playback_stopped", 430.0, message_id=ANSWER_1)
    record(tracker, clock, "ask_submitted", 440.0, message_id=USER_2)
    record(tracker, clock, "answer_event_received", 500.0, message_id=ANSWER_2)
    record(tracker, clock, "answer_first_content", 520.0, message_id=ANSWER_2)
    record(tracker, clock, "tts_start_requested", 521.0, message_id=ANSWER_2)
    record(tracker, clock, "tts_ready", 530.0)
    record(tracker, clock, "tts_text_submitted", 540.0)
    record(tracker, clock, "tts_first_audio_received", 590.0)
    record(tracker, clock, "tts_first_pcm_queued", 591.0)
    record(tracker, clock, "tts_playback_started", 600.0)
    record(tracker, clock, "tts_playback_stopped", 610.0, message_id=ANSWER_2)

    first = tracker.metrics_for(ANSWER_1, PetVoiceStatus.STOPPED)
    second = tracker.metrics_for(ANSWER_2, PetVoiceStatus.COMPLETED)

    assert first is not None
    assert second is not None
    assert first["nativeMs"]["endpointToPlayback"] == 210.0
    assert second["nativeMs"]["endpointToPlayback"] == 200.0
    assert second["nativeMs"]["endpointToAsk"] == 40.0


def test_composite_observer_tracks_events_when_audio_diagnostics_are_disabled() -> None:
    class DisabledAudioDiagnostics:
        def __init__(self) -> None:
            self.events: list[str] = []

        def record_frame(self, *_args: object, **_kwargs: object) -> int:
            return -1

        def record_event(self, name: str, **_kwargs: object) -> None:
            self.events.append(name)

    clock = ManualClock()
    audio = DisabledAudioDiagnostics()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)
    observer = CompositeAudioDiagnostics(audio, tracker)

    record(observer, clock, "speech_end", 100.0)
    record(observer, clock, "asr_final", 120.0)
    record(observer, clock, "ask_submitted", 130.0, message_id=USER_1)
    record(observer, clock, "answer_event_received", 200.0, message_id=ANSWER_1)

    assert audio.events == [
        "speech_end",
        "asr_final",
        "ask_submitted",
        "answer_event_received",
    ]
    assert tracker.metrics_for(ANSWER_1, PetVoiceStatus.FAILED) is not None


def traced_tracker() -> CompanionLatencyTracker:
    clock = ManualClock()
    tracker = CompanionLatencyTracker(clock_ns=clock.now_ns)
    record(tracker, clock, "speech_end", 100.0)
    record(tracker, clock, "asr_final", 120.0)
    record(tracker, clock, "ask_submitted", 130.0, message_id=USER_1)
    record(tracker, clock, "answer_event_received", 200.0, message_id=ANSWER_1)
    return tracker


@pytest.mark.asyncio
async def test_reporter_submits_in_the_background_and_close_flushes() -> None:
    class BlockingClient:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls: list[tuple[UUID, UUID, dict[str, object]]] = []

        async def persist_pet_voice_performance(
            self,
            fish_id: UUID,
            answer_message_id: UUID,
            metrics: dict[str, object],
        ) -> None:
            self.calls.append((fish_id, answer_message_id, metrics))
            self.started.set()
            await self.release.wait()

    client = BlockingClient()
    reporter = PetVoicePerformanceReporter(client, traced_tracker())

    reporter.submit(USER_2, ANSWER_1, PetVoiceStatus.COMPLETED)
    await asyncio.wait_for(client.started.wait(), timeout=1)
    close_task = asyncio.create_task(reporter.close())
    await asyncio.sleep(0)

    assert not close_task.done()
    assert client.calls[0][:2] == (USER_2, ANSWER_1)
    assert client.calls[0][2]["status"] == "completed"
    assert client.calls[0][2]["answerMessageId"] == str(ANSWER_1)

    client.release.set()
    await asyncio.wait_for(close_task, timeout=1)


@pytest.mark.asyncio
async def test_reporter_ignores_missing_trace() -> None:
    class UnexpectedClient:
        async def persist_pet_voice_performance(self, *_args: object) -> None:
            raise AssertionError("missing traces must not be persisted")

    reporter = PetVoicePerformanceReporter(
        UnexpectedClient(),
        CompanionLatencyTracker(),
    )

    reporter.submit(USER_2, ANSWER_1, PetVoiceStatus.FAILED)
    await reporter.close()


@pytest.mark.asyncio
async def test_reporter_logs_persistence_failure_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingClient:
        async def persist_pet_voice_performance(self, *_args: object) -> None:
            raise RuntimeError("diagnostics unavailable")

    reporter = PetVoicePerformanceReporter(FailingClient(), traced_tracker())

    with caplog.at_level(logging.WARNING):
        reporter.submit(USER_2, ANSWER_1, PetVoiceStatus.STOPPED)
        await reporter.close()

    assert "diagnostics unavailable" in caplog.text
