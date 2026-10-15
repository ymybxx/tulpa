import base64
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from fish_memory.observations.environment_history_audio import SystemAudioTranscript
from fish_memory.observations.environment_history_client import (
    EnvironmentHistoryModelClient,
)
from fish_memory.observations.environment_history_contact_sheet import (
    build_contact_sheet,
)
from fish_memory.observations.environment_history_models import (
    EnvironmentHistoryWindow,
    SemanticDocument,
)
from tests.observations.test_environment_history import (
    FISH,
    START,
    settle,
    setup_service,
)
from tests.observations.test_environment_history_client import JPEG, JPEG_2, FakeModel
from tests.observations.test_environment_history_idle import SilentTranscriber
from tests.observations.test_environment_history_rolling_speakers import (
    make_diarizer,
    pcm,
)
from tests.test_environment_frames import _request


def window(index=0, duration=5):
    start = START.timestamp() + index * 5
    return EnvironmentHistoryWindow(
        capture_session_id=UUID(int=2),
        window_index=index,
        started_at=start,
        ended_at=start + duration,
        frames=tuple(
            _request(
                observed_at=start + second,
                frame_id=100 + index * 5 + second,
            ).model_copy(
                update={
                    "pixel_width": 2,
                    "pixel_height": 1,
                    "byte_length": len(JPEG),
                    "jpeg_base64": base64.b64encode(JPEG).decode(),
                }
            )
            for second in range(int(duration))
        ),
        audio_status="complete",
        audio_silence=True,
        audio_captured_samples=round(duration * 16000),
    )


def test_host_session_routing_does_not_change_capture_retry_identity():
    captured = window()
    assigned = captured.model_copy(update={"session_id": UUID(int=50)})
    assert captured.input_digest() == assigned.input_digest()


@pytest.mark.asyncio
async def test_five_second_silence_skips_model_and_never_starts_aggregate_jobs():
    service, repository, model = setup_service(transcriber=SilentTranscriber())
    try:
        for index in range(12):
            await service.enqueue(FISH, window(index))
        await settle(service)
        assert len(model.calls) == 1
        assert not model.merges
        assert len(repository.values) == 12
        assert all(value.period_seconds == 5 for value in repository.values.values())
        assert len(window().decode_pcm()) == 160_000
        assert len(window(duration=2).decode_pcm()) == 64_000
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_screen_only_windows_skip_asr_and_reuse_unchanged_visuals():
    transcriber = SimpleNamespace(
        close=AsyncMock(),
        transcribe=AsyncMock(side_effect=AssertionError("audio input is disabled")),
        transcribe_window=AsyncMock(
            side_effect=AssertionError("audio input is disabled")
        ),
    )
    service, repository, model = setup_service(transcriber=transcriber)
    try:
        for index in range(3):
            visual = window(index).model_copy(
                update={
                    "audio_status": "unavailable",
                    "audio_silence": False,
                    "audio_captured_samples": 0,
                    "audio_pcm_base64": None,
                    "audio_gap_reason": "system_audio_disabled",
                }
            )
            await service.enqueue(FISH, visual)
        await settle(service)
        assert len(model.calls) == 1
        assert len(repository.values) == 3
        assert not model.merges
        transcriber.transcribe.assert_not_called()
        transcriber.transcribe_window.assert_not_called()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_silent_visual_comparison_can_return_empty_with_previous_images(
    system_timezone,
):
    system_timezone("Asia/Shanghai")
    provider = FakeModel()
    provider.content = json.dumps({"audiovisualSummary": "", "activitySummary": ""})
    client = EnvironmentHistoryModelClient(
        api_key="unused",
        base_url="https://unused.invalid",
        model="test",
        client=cast(Any, provider),
    )
    previous = window()
    current = window(1)
    current = current.model_copy(
        update={
            "frames": tuple(
                frame.model_copy(
                    update={
                        "byte_length": len(JPEG_2),
                        "jpeg_base64": base64.b64encode(JPEG_2).decode(),
                    }
                )
                for frame in current.frames
            )
        }
    )
    result = await client.summarize_window(
        current, SystemAudioTranscript("", "silence"), previous_window=previous
    )
    assert not result.summary and not result.audio
    parts = provider.calls[0]["messages"][1]["content"]
    assert [part["type"] for part in parts] == [
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    reference, current_input = (parts[i]["text"] for i in (0, 2))
    assert reference == (
        "下面是上一时段看到的画面：时间"
        "2026-09-03T08:00:00+08:00 — 2026-09-03T08:00:05+08:00"
    )
    assert current_input == (
        "这是当前时段看到的画面：时间"
        "2026-09-03T08:00:05+08:00 — 2026-09-03T08:00:10+08:00"
    )
    assert "你听到的声音" not in reference + current_input
    for offset, source in ((0, previous), (2, current)):
        sheet = build_contact_sheet(source)
        assert parts[offset + 1]["image_url"]["url"].endswith(sheet.jpeg_base64)
    await client.close()


@pytest.mark.asyncio
async def test_summary_input_keeps_past_then_current_image_labels(monkeypatch):
    client = EnvironmentHistoryModelClient(
        api_key="unused",
        base_url="https://unused.invalid",
        model="test",
        client=cast(Any, FakeModel()),
    )
    complete = AsyncMock(return_value=SemanticDocument(summary_version=4))
    monkeypatch.setattr(client, "_complete_document", complete)
    try:
        await client.summarize_window(
            window(1),
            SystemAudioTranscript("本段新对白", "Speech"),
            previous_window=window(),
            continuity_context="较早的声源与对白",
        )
        parts = complete.call_args.args[0]
        assert "较早的声源与对白" in parts[0]["text"]
        assert parts[1]["text"].startswith("下面是上一时段看到的画面：时间")
        assert "你听到的声音" not in parts[1]["text"]
        assert parts[2]["type"] == "image_url"
        assert parts[3]["text"].startswith("这是当前时段看到的画面：时间")
        assert parts[3]["text"].endswith("\n你听到的声音：本段新对白")
        assert parts[4]["type"] == "image_url"
        assert len(parts) == 5
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_empty_visual_summary_cannot_drop_new_aligned_speech():
    provider = FakeModel()
    provider.content = json.dumps({"audiovisualSummary": "", "activitySummary": ""})
    client = EnvironmentHistoryModelClient(
        api_key="unused",
        base_url="https://unused.invalid",
        model="test",
        client=cast(Any, provider),
    )
    words = "明天我会把书还给奶奶。"
    try:
        result = await client.summarize_window(
            window(1), SystemAudioTranscript(words, "Speech"), previous_window=window()
        )
        assert words in result.audiovisual_summary and words in result.audio
        assert result.activity_summary == ""
        assert len(provider.calls) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_next_summary_receives_prior_role_context_and_continuous_voice_labels():
    service, repository, _ = setup_service(transcriber=SilentTranscriber())
    calls = []

    class Model:
        async def summarize_window(self, current, transcript, **options):
            calls.append(options)
            return SemanticDocument(
                summary_version=4, audiovisual_summary="男孩向奶奶借书。"
            )

        async def close(self):
            pass

    class Audio:
        async def transcribe(self, pcm):
            return SystemAudioTranscript(
                "声源abc-S00：奶奶，下周我还你这本书。", "Speech"
            )

        async def close(self):
            pass

    service._summarizer, service._transcriber = Model(), Audio()
    try:
        await service.enqueue(FISH, window())
        await settle(service)
        await service.enqueue(FISH, window(1))
        await settle(service)
        assert "男孩向奶奶借书" in calls[1]["continuity_context"]
        assert "声源abc-S00" in calls[1]["continuity_context"]
        assert "2026-09-03T00:00:00" in calls[1]["continuity_context"]
    finally:
        await service.close()


def test_voice_timeline_uses_real_duration_across_short_and_five_second_windows():
    diarizer = make_diarizer()
    diarizer.segment_at(
        pcm((0, 2, 1000))[:64000], stream_id="media", window_index=0, started_at=100
    )
    diarizer.segment_at(
        pcm((0, 5, 1000))[:160000], stream_id="media", window_index=1, started_at=102
    )
    diarizer.segment_at(
        pcm((0, 5, -1000))[:160000], stream_id="media", window_index=2, started_at=107
    )
    diarizer.observe_silence_at(
        stream_id="media", window_index=3, started_at=112, duration=5
    )
    assert diarizer._state.windows[1] == (32000, 112000)
    assert diarizer._state.windows[2] == (112000, 192000)
    for revision in diarizer.window_revisions:
        duration = 2 if revision.window_index == 0 else 5
        assert all(
            0 <= span.start_sample < span.end_sample <= duration * 16000
            for span in revision.spans
        )
