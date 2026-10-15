import json
import logging
import struct
from collections.abc import Mapping

import pytest

from fish_memory.companion.runtime_trace import VoiceRuntimeTrace


class Observer:
    def __init__(self) -> None:
        self.frames = 0
        self.events: list[tuple[str, str | None, Mapping[str, object] | None]] = []

    def record_frame(self, *args: object, **kwargs: object) -> int:
        self.frames += 1
        return self.frames

    def record_event(
        self,
        name: str,
        *,
        frame_index: int | None = None,
        text: str | None = None,
        source: str = "unknown",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.events.append((name, text, metadata))


class Clock:
    now = 0

    def __call__(self) -> int:
        return self.now


def payloads(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, object]]:
    prefix = f"voice_trace event={event} "
    return [
        json.loads(record.getMessage().removeprefix(prefix))
        for record in caplog.records
        if record.getMessage().startswith(prefix)
    ]


def test_summary_is_low_frequency_and_marks_post_vpio_without_recording_pcm(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    delegate = Observer()
    clock = Clock()
    trace = VoiceRuntimeTrace(
        delegate,
        context=lambda: {"state": "listening", "captureStage": "post_vpio"},
        clock_ns=clock,
    )
    pcm = struct.pack("<160h", *([1000] * 160))
    trace.record_event("speech_start", metadata={"speechGeneration": 3})
    for frame in range(501):
        clock.now = frame * 10_000_000
        assert (
            trace.record_frame(pcm, bytes(320), pcm, voice_detected=True) == frame + 1
        )
    summaries = payloads(caplog, "input_window")
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["frames"] == 501
    assert summary["voicedFrames"] == 501
    assert summary["vadEvaluatedFrames"] == 501
    assert summary["candidateActiveMs"] == 5000
    assert summary["speechGeneration"] == 3
    assert summary["sampledPeak"] == 1000
    assert summary["sampledRmsDbfs"] == -30.3
    assert summary["sampledPcmSource"] == "processor_cleaned_pcm"
    assert summary["captureStage"] == "post_vpio"
    assert all(
        not isinstance(value, (bytes, bytearray)) for value in vars(trace).values()
    )


def test_events_keep_delegate_semantics_but_never_log_text_or_unknown_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    delegate = Observer()
    trace = VoiceRuntimeTrace(delegate, context=lambda: {"apiKey": "secret-context"})
    metadata = {
        "reason": "punctuation_only",
        "rawText": "secret-raw",
        "asrEvent": "<|Speech|><|secret-event|>",
        "messageId": "secret-invalid-id",
        "durationMs": float("nan"),
    }
    trace.record_event(
        "asr_final_rejected", text="secret-transcript", metadata=metadata
    )
    trace.record_event("voice_draft_first_content", text="secret-draft")
    trace.record_event("unknown-event-secret", metadata={"reason": "secret-reason"})
    assert delegate.events[0] == ("asr_final_rejected", "secret-transcript", metadata)
    assert len(delegate.events) == 3
    result = payloads(caplog, "asr_final_rejected")[0]
    assert result["reason"] == "punctuation_only"
    assert result["asrEvent"] == ["speech"]
    assert result["charCount"] == len("secret-transcript")
    assert "secret" not in caplog.text
    assert "durationMs" not in result


def test_nonlexical_admission_reason_is_observable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})

    trace.record_event(
        "asr_final_rejected",
        metadata={"reason": "nonlexical_vocalization"},
    )

    result = payloads(caplog, "asr_final_rejected")[0]
    assert result["reason"] == "nonlexical_vocalization"


def test_facial_wait_and_provider_timing_are_logged_without_speech_or_targets(caplog):
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})
    trace.record_event(
        "tts_facial_ready",
        text="private speech",
        metadata={
            "waitMs": 748.542,
            "endOffset": 35,
            "errorType": "TimeoutError",
            "targets": {"ParamAngleX": 1},
            "text": "private speech",
        },
    )
    trace.record_event(
        "tts_first_audio_timing",
        metadata={
            "providerFirstPacketMs": 400.0,
            "facialWaitMs": 748.542,
        },
    )
    face = payloads(caplog, "tts_facial_ready")[0]
    assert face["waitMs"] == 748.542 and face["endOffset"] == 35
    assert face["errorType"] == "TimeoutError"
    timing = payloads(caplog, "tts_first_audio_timing")[0]
    assert timing["providerFirstPacketMs"] == 400
    assert timing["facialWaitMs"] == 748.542
    assert "private" not in caplog.text and "ParamAngleX" not in caplog.text


def test_reused_asr_result_is_observable_without_logging_transcript(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})
    trace.record_event(
        "asr_draft_reused",
        text="private-transcript",
        metadata={"pcmBytes": 640, "charCount": 4, "asrEvent": "Speech"},
    )
    result = payloads(caplog, "asr_draft_reused")[0]
    assert result["pcmBytes"] == 640
    assert result["charCount"] == 4
    assert result["asrEvent"] == ["speech"]
    assert "private-transcript" not in caplog.text


@pytest.mark.parametrize(
    "event",
    [
        "voice_draft_started",
        "voice_draft_transcript_ready",
        "voice_draft_request_started",
        "voice_draft_first_content",
        "voice_draft_promoted",
        "voice_draft_fallback",
        "voice_draft_canceled",
        "voice_draft_cleanup",
        "voice_draft_late_event_dropped",
        "voice_draft_tts_provider_ready",
        "voice_draft_tts_skipped",
        "voice_draft_tts_text_submitted",
        "voice_draft_tts_first_audio_buffered",
        "voice_draft_tts_pcm_promoted",
        "voice_draft_tts_pcm_discarded",
        "voice_draft_asr_prepare_triggered",
        "voice_draft_preparation_started",
        "voice_draft_preparation_queued",
        "voice_draft_candidate_triggered",
        "voice_draft_gate_summary",
        "voice_draft_resumed",
        "voice_draft_asr_reuse_missed",
    ],
)
def test_draft_lifecycle_keeps_shared_safe_fields_without_content(
    caplog: pytest.LogCaptureFixture, event: str
) -> None:
    caplog.set_level(logging.INFO)
    delegate = Observer()
    trace = VoiceRuntimeTrace(delegate, context=lambda: {})
    safe = {
        "turnId": "00000000-0000-0000-0000-000000000101",
        "revision": 4,
        "attemptRevision": 3,
        "candidateEpoch": 2,
        "speechGeneration": 9,
        "bindingGeneration": 1,
        "reason": "audio_queue_age",
        "queueAgeMs": 151.5,
        "queueDepth": 7,
        "droppedFrames": 2,
        "pcmBytes": 640,
        "utf8Bytes": 12,
        "submittedChars": 4,
        "providerReady": True,
        "accepted": False,
    }
    metadata = {
        **safe,
        "content": "private-draft-content",
        "rawText": "private-raw-transcript",
        "prompt": "private-prompt",
        "pcm": b"private-pcm",
        "text": "private-nested-text",
        "untrusted": {"reason": "private-nested-reason"},
    }
    trace.record_event(event, text="private-transcript", metadata=metadata)

    result = payloads(caplog, event)[0]
    assert {key: result[key] for key in safe} == safe
    assert set(result) == {"runId", "atMs", *safe}
    assert "private" not in caplog.text
    assert delegate.events == [(event, "private-transcript", metadata)]


@pytest.mark.parametrize("reason", ["speech_end", "committed", "abandoned"])
def test_draft_gate_summary_preserves_shared_counts_without_pcm(
    caplog: pytest.LogCaptureFixture, reason: str
) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})
    safe = {
        "reason": reason,
        "gateFrames": 85,
        "gateSpeechFrames": 51,
        "gateSilentFrames": 9,
        "gateLongestSilenceFrames": 17,
        "gateRequiredSilenceFrames": 20,
        "gateCandidateCount": 0,
        "gateResumedCount": 1,
    }
    trace.record_event(
        "voice_draft_gate_summary",
        metadata={**safe, "pcm": "private-pcm", "content": "private-text"},
    )

    result = payloads(caplog, "voice_draft_gate_summary")[0]
    assert {key: result[key] for key in safe} == safe
    assert "private" not in caplog.text


@pytest.mark.parametrize(
    ("event", "reason"),
    [
        ("voice_draft_fallback", "decode_busy"),
        ("voice_draft_tts_skipped", "committed_tts_active"),
        ("voice_draft_resumed", "fast_vad"),
        ("voice_draft_resumed", "primary_start"),
    ],
)
def test_draft_skip_and_resume_reasons_survive_the_runtime_log(
    caplog: pytest.LogCaptureFixture, event: str, reason: str
) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})
    trace.record_event(event, metadata={"reason": reason, "attemptRevision": 2})
    result = payloads(caplog, event)[0]
    assert result["reason"] == reason
    assert result["attemptRevision"] == 2


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "reason": "private-unlisted-reason",
            "turnId": "private-invalid-uuid",
            "revision": True,
            "attemptRevision": -1,
            "queueAgeMs": float("nan"),
            "queueDepth": "private-depth",
            "accepted": "private-accepted",
        },
        {
            "reason": b"private-reason-bytes",
            "queueAgeMs": float("inf"),
            "queueDepth": -1,
            "pcmBytes": b"private-audio",
            "submittedChars": 1.5,
        },
    ],
)
def test_draft_metadata_rejects_unknown_content_and_invalid_numeric_fields(
    caplog: pytest.LogCaptureFixture, metadata: dict[str, object]
) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})
    trace.record_event("voice_draft_fallback", text="private-text", metadata=metadata)
    trace.record_event("voice_draft_private_event_name", metadata={"revision": 1})

    result = payloads(caplog, "voice_draft_fallback")[0]
    assert set(result) == {"runId", "atMs"}
    assert "private" not in caplog.text


def test_draft_events_share_the_bounded_lifecycle_rate_limit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    clock = Clock()
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {}, clock_ns=clock)
    trace.record_frame(bytes(320), bytes(320), bytes(320))
    for _ in range(60):
        trace.record_event("speech_end")
    for _ in range(60):
        trace.record_event(
            "voice_draft_fallback", metadata={"reason": "audio_frame_drop"}
        )

    assert len(payloads(caplog, "speech_end")) == 60
    assert len(payloads(caplog, "voice_draft_fallback")) == 20
    clock.now = 5_000_000_000
    trace.record_frame(bytes(320), bytes(320), bytes(320))
    assert payloads(caplog, "input_window")[0]["suppressedEvents"] == 40
    trace.record_event("voice_draft_started", metadata={"revision": 5})
    assert payloads(caplog, "voice_draft_started")[0]["revision"] == 5


def test_event_flood_is_bounded_and_reported_in_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    clock = Clock()
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {}, clock_ns=clock)
    trace.record_frame(bytes(320), bytes(320), bytes(320))
    for _ in range(100):
        trace.record_event("speech_end")
    assert len(payloads(caplog, "speech_end")) == 80
    clock.now = 5_000_000_000
    trace.record_frame(bytes(320), bytes(320), bytes(320))
    assert payloads(caplog, "input_window")[0]["suppressedEvents"] == 20
    trace.record_event("speech_start")
    assert len(payloads(caplog, "speech_start")) == 1


def test_diagnostic_failure_does_not_disable_delegate_or_propagate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    delegate = Observer()

    def failing_context() -> Mapping[str, object]:
        raise RuntimeError("secret-exception-content")

    trace = VoiceRuntimeTrace(delegate, context=failing_context)
    trace.record_event("speech_start")
    trace.record_event("speech_end")
    assert trace.record_frame(bytes(320), bytes(320), bytes(320)) == 1
    assert [item[0] for item in delegate.events] == ["speech_start", "speech_end"]
    assert "secret-exception-content" not in caplog.text
    assert caplog.text.count("trace disabled") == 1


def test_disabled_info_logging_does_not_sample_frames(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
    clock = Clock()
    delegate = Observer()
    trace = VoiceRuntimeTrace(delegate, context=lambda: {}, clock_ns=clock)
    for _ in range(600):
        clock.now += 10_000_000
        trace.record_frame(bytes(320), bytes(320), bytes(320), voice_detected=True)
    assert delegate.frames == 600
    assert trace._frames == 0
    assert not caplog.records


@pytest.mark.parametrize(
    ("event", "metadata"),
    [
        (
            "finish_scheduled",
            {"speechGeneration": 4, "commitDelayMs": 0, "smartTurnEnabled": True},
        ),
        (
            "finish_waiting",
            {"reason": "smart_turn_inference", "timeoutMs": 500},
        ),
        (
            "finish_cancelled",
            {
                "reason": "speech_restart",
                "speechGeneration": 4,
                "currentSpeechGeneration": 5,
            },
        ),
        (
            "asr_final_rejected",
            {
                "reason": "near_end_not_confirmed",
                "requiresNearEnd": True,
                "nearEndConfirmed": False,
                "detector": "silero_vad",
            },
        ),
        (
            "submit_turn_rejected",
            {"reason": "state_not_listening", "state": "standby", "inputChars": 6},
        ),
    ],
)
def test_diagnostic_contract_keeps_real_stage_fields(
    caplog: pytest.LogCaptureFixture, event: str, metadata: dict[str, object]
) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})
    trace.record_event(event, metadata=metadata)
    result = payloads(caplog, event)[0]
    for field, value in metadata.items():
        assert result[field] == value


@pytest.mark.parametrize("asr_event", ["Speech", "<|Speech|>", " speech "])
def test_asr_labels_accept_existing_plain_and_tagged_forms(
    caplog: pytest.LogCaptureFixture, asr_event: str
) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(Observer(), context=lambda: {})
    trace.record_event("asr_decode_completed", metadata={"asrEvent": asr_event})
    assert payloads(caplog, "asr_decode_completed")[0]["asrEvent"] == ["speech"]


def test_existing_smart_turn_error_type_is_visible_without_changing_delegate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    delegate = Observer()
    trace = VoiceRuntimeTrace(delegate, context=lambda: {})
    trace.record_event("smart_turn_failed", metadata={"error": "TimeoutError"})
    assert payloads(caplog, "smart_turn_failed")[0]["errorType"] == "TimeoutError"
    assert delegate.events[0][2] == {"error": "TimeoutError"}


def test_standby_placeholder_is_not_counted_as_vad_evaluation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    clock = Clock()
    state = "standby"
    trace = VoiceRuntimeTrace(
        Observer(), context=lambda: {"state": state}, clock_ns=clock
    )
    trace.record_event("voice_trace_started")
    trace.record_frame(bytes(320), bytes(320), bytes(320), voice_detected=False)
    clock.now = 3_000_000_000
    trace.record_frame(bytes(320), bytes(320), bytes(320), voice_detected=False)
    state = "listening"
    trace.record_event("session_state_changed", metadata={"state": state})
    clock.now = 5_000_000_000
    trace.record_frame(bytes(320), bytes(320), bytes(320), voice_detected=True)
    summary = payloads(caplog, "input_window")[0]
    assert summary["frames"] == 3
    assert summary["vadEvaluatedFrames"] == 1
    assert summary["voicedFrames"] == 1


def test_owner_admission_logs_only_safe_decision_fields(caplog) -> None:
    caplog.set_level(logging.INFO)
    trace = VoiceRuntimeTrace(
        Observer(), context=lambda: {"ownerRecognitionEnabled": True}
    )
    trace.record_event(
        "owner_voice_decision",
        metadata={
            "reason": "not_owner",
            "score": 0.23,
            "revoked": True,
            "inferenceMs": 20.0,
            "releasedFrames": 0,
            "embedding": ["private-embedding"],
            "pcm": "private-pcm",
        },
    )
    result = payloads(caplog, "owner_voice_decision")[0]
    assert result["score"] == 0.23
    assert result["reason"] == "not_owner"
    assert result["revoked"] is True
    assert result["ownerRecognitionEnabled"] is True
    assert "private" not in caplog.text


def test_owner_enabled_standby_counts_real_vad_instead_of_placeholder(caplog) -> None:
    caplog.set_level(logging.INFO)
    clock = Clock()
    trace = VoiceRuntimeTrace(
        Observer(),
        context=lambda: {"state": "standby", "ownerRecognitionEnabled": True},
        clock_ns=clock,
    )
    trace.record_event("voice_trace_started")
    trace.record_frame(bytes(320), bytes(320), bytes(320), voice_detected=True)
    clock.now = 5_000_000_000
    trace.record_frame(bytes(320), bytes(320), bytes(320), voice_detected=False)
    result = payloads(caplog, "input_window")[0]
    assert result["vadEvaluatedFrames"] == 2
    assert result["voicedFrames"] == 1
