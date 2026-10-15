import pytest

from fish_memory.companion.latency import summarize_turn_latency
from fish_memory.performance.models import PerformanceStep


def event(
    name: str,
    at_ms: float,
    *,
    message_id: str | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {"name": name, "atMs": at_ms}
    if message_id is not None:
        value["metadata"] = {"messageId": message_id}
    return value


def test_summarizes_one_correlated_native_turn() -> None:
    timeline = [
        event("speech_end", 100.0),
        event("asr_final", 148.5),
        event("ask_submitted", 156.4, message_id="user-1"),
        event("answer_event_received", 1100.0, message_id="answer-1"),
        event("answer_first_content", 1419.0, message_id="answer-1"),
        event("tts_start_requested", 1420.0, message_id="answer-1"),
        event("tts_ready", 1460.0),
        event("tts_text_submitted", 1462.0),
        event("tts_first_audio_received", 1630.0),
        event("tts_first_pcm_queued", 1631.0),
        event("tts_playback_started", 1663.6),
    ]

    result = summarize_turn_latency(timeline)

    assert result["userMessageId"] == "user-1"
    assert result["answerMessageId"] == "answer-1"
    assert result["nativeMs"] == {
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
    }
    assert result["serverStepEndMs"] == {}


def test_missing_events_remain_none_and_duplicates_use_first_event() -> None:
    result = summarize_turn_latency(
        [
            event("speech_end", 10.0),
            event("ask_submitted", 20.0, message_id="user-1"),
            event("ask_submitted", 30.0, message_id="user-2"),
        ]
    )

    assert result["userMessageId"] == "user-1"
    assert result["answerMessageId"] is None
    assert result["nativeMs"]["endpointToAsk"] == 10.0
    assert result["nativeMs"]["askToPlayback"] is None


def test_server_offsets_are_reported_without_mixing_clock_domains() -> None:
    result = summarize_turn_latency(
        [event("ask_submitted", 20.0, message_id="user-1")],
        server_steps=[
            PerformanceStep(
                name="context.build",
                index=0,
                started_offset_us=10_000,
                duration_us=900,
            ),
            PerformanceStep(
                name="answer.first_content",
                index=0,
                started_offset_us=0,
                duration_us=1_262_600,
            ),
        ],
    )

    assert result["serverStepEndMs"] == {
        "context.build": 10.9,
        "answer.first_content": 1262.6,
    }
    assert "serverToPlayback" not in result["nativeMs"]


@pytest.mark.parametrize("wait", [0.0, 748.542])
def test_first_audio_breakdown_uses_measured_durations_and_not_later_face_wait(wait):
    timeline = [
        event("tts_text_submitted", 100),
        {"name": "tts_facial_ready", "atMs": 1240, "metadata": {"waitMs": wait}},
        {
            "name": "tts_first_audio_timing",
            "atMs": 1248,
            "metadata": {"providerFirstPacketMs": 400.0, "facialWaitMs": wait},
        },
        event("tts_first_audio_received", 1248.542),
        {"name": "tts_facial_ready", "atMs": 2100, "metadata": {"waitMs": 900}},
    ]
    native = summarize_turn_latency(timeline)["nativeMs"]
    assert native["ttsTextToFirstAudio"] == 1148.542
    assert native["ttsProviderFirstAudio"] == 400.0
    assert native["ttsFacialWaitForFirstAudio"] == wait


@pytest.mark.parametrize("invalid", [True, -1, float("nan"), float("inf"), "400", None])
def test_missing_or_invalid_provider_timings_are_not_fabricated(invalid):
    timeline = [
        {
            "name": "tts_first_audio_timing",
            "atMs": 100,
            "metadata": {"providerFirstPacketMs": invalid, "facialWaitMs": invalid},
        }
    ]
    native = summarize_turn_latency(timeline)["nativeMs"]
    assert "ttsProviderFirstAudio" not in native
    assert "ttsFacialWaitForFirstAudio" not in native
