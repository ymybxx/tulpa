"""Deterministic latency summaries for one native companion turn."""

import math
from collections.abc import Mapping, Sequence

from fish_memory.performance.models import JsonValue, PerformanceStep

_NATIVE_SEGMENTS = {
    "endpointToAsrFinal": ("speech_end", "asr_final"),
    "endpointToAsk": ("speech_end", "ask_submitted"),
    "askToAnswerEvent": ("ask_submitted", "answer_event_received"),
    "askToFirstContent": ("ask_submitted", "answer_first_content"),
    "firstContentToTtsStart": ("answer_first_content", "tts_start_requested"),
    "ttsStartToReady": ("tts_start_requested", "tts_ready"),
    "ttsReadyToTextSubmitted": ("tts_ready", "tts_text_submitted"),
    "ttsTextToFirstAudio": ("tts_text_submitted", "tts_first_audio_received"),
    "firstAudioToPcmQueued": (
        "tts_first_audio_received",
        "tts_first_pcm_queued",
    ),
    "pcmQueuedToPlayback": ("tts_first_pcm_queued", "tts_playback_started"),
    "askToPlayback": ("ask_submitted", "tts_playback_started"),
    "endpointToPlayback": ("speech_end", "tts_playback_started"),
}


def summarize_turn_latency(
    timeline: Sequence[Mapping[str, object]],
    *,
    server_steps: Sequence[PerformanceStep] = (),
) -> dict[str, JsonValue]:
    """Summarize measured boundaries without crossing clock domains."""

    events: dict[str, Mapping[str, object]] = {}
    for event in timeline:
        name = event.get("name")
        if isinstance(name, str) and name not in events:
            events[name] = event

    native: dict[str, JsonValue] = {
        label: _difference_ms(events.get(start), events.get(end))
        for label, (start, end) in _NATIVE_SEGMENTS.items()
    }
    timing_event = events.get("tts_first_audio_timing", {})
    timing = timing_event.get("metadata")
    if isinstance(timing, Mapping):
        for output, source in (
            ("ttsProviderFirstAudio", "providerFirstPacketMs"),
            ("ttsFacialWaitForFirstAudio", "facialWaitMs"),
        ):
            value = timing.get(source)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            ):
                # Keep the API's measured duration; do not subtract timestamps
                # across processes or substitute later sentence waits.
                native[output] = round(float(value), 3)
    server: dict[str, JsonValue] = {}
    for step in server_steps:
        key = step.name if step.index == 0 else f"{step.name}[{step.index}]"
        if key in server:
            continue
        server[key] = round(
            (step.started_offset_us + step.duration_us) / 1000,
            3,
        )

    return {
        "userMessageId": _message_id(events.get("ask_submitted")),
        "answerMessageId": _message_id(events.get("answer_event_received")),
        "nativeMs": native,
        "serverStepEndMs": server,
    }


def _difference_ms(
    start: Mapping[str, object] | None,
    end: Mapping[str, object] | None,
) -> float | None:
    start_ms = _event_time_ms(start)
    end_ms = _event_time_ms(end)
    if start_ms is None or end_ms is None or end_ms < start_ms:
        return None
    return round(end_ms - start_ms, 3)


def _event_time_ms(event: Mapping[str, object] | None) -> float | None:
    if event is None:
        return None
    value = event.get("atMs")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _message_id(event: Mapping[str, object] | None) -> str | None:
    if event is None:
        return None
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("messageId")
    return value if isinstance(value, str) and value else None
