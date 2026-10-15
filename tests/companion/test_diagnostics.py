import asyncio
import json
from pathlib import Path

import pytest

from fish_memory.companion.diagnostics import (
    AudioDiagnostics,
    AudioDiagnosticsConfig,
    privacy_safe_event_fields,
)


def _frame(value: bytes = b"c") -> tuple[bytes, bytes, bytes]:
    return value * 320, b"r" * 320, b"x" * 320


def test_diagnostics_config_captures_tuning_parameters() -> None:
    class Settings:
        audio_diagnostics_enabled = True
        audio_diagnostics_dir = "/tmp/audio-diagnostics"
        audio_diagnostics_retention_days = 9
        audio_diagnostics_pre_seconds = 4.0
        audio_diagnostics_post_seconds = 6.0
        companion_aec_delay_ms = 70
        companion_aec_delay_mode = "auto"
        companion_vad_start_frames = 2
        companion_vad_threshold = 0.4
        companion_vad_min_silence_ms = 300
        companion_vad_min_speech_ms = 80
        companion_vad_window_size = 512
        companion_vad_max_speech_seconds = 20.0
        companion_vad_num_threads = 1
        asr_model = "sensevoice"
        asr_num_threads = 4
        companion_audio_block_size = 160

    config = AudioDiagnosticsConfig.from_settings(Settings())

    assert config.enabled is True
    assert config.directory == Path("/tmp/audio-diagnostics")
    assert config.parameters == {
        "aecDelayMs": None,
        "aecDelayMode": "auto",
        "vadBackend": "silero",
        "vadStartFrames": 1,
        "vadThreshold": 0.4,
        "vadMinSilenceMs": 300,
        "vadMinSpeechMs": 80,
        "vadSpeechPreRollFrames": 22,
        "vadWindowSize": 512,
        "vadMaxSpeechSeconds": 20.0,
        "vadNumThreads": 1,
        "asrModel": "sensevoice",
        "asrNumThreads": 4,
        "audioBlockSize": 160,
    }


@pytest.mark.asyncio
async def test_disabled_diagnostics_do_not_create_files(tmp_path: Path) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(enabled=False, directory=tmp_path)
    )

    await diagnostics.start()
    capture, render, cleaned = _frame()
    diagnostics.record_frame(capture, render, cleaned)
    diagnostics.record_event("interruption_confirmed", text="你好")
    await diagnostics.close()

    entries = await asyncio.to_thread(lambda: list(tmp_path.iterdir()))
    assert entries == []


def test_ring_keeps_only_the_configured_pre_window(tmp_path: Path) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0.02,
            post_seconds=0,
        )
    )

    first = diagnostics.record_frame(*_frame(b"1"))
    second = diagnostics.record_frame(*_frame(b"2"))
    third = diagnostics.record_frame(*_frame(b"3"))

    assert (first, second, third) == (0, 1, 2)
    assert [frame.capture_pcm for frame in diagnostics._ring] == [
        b"2" * 320,
        b"3" * 320,
    ]


def test_speech_start_creates_pending_question_snapshot(tmp_path: Path) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(enabled=True, directory=tmp_path)
    )

    index = diagnostics.record_frame(*_frame())
    diagnostics.record_event("speech_start", frame_index=index, source="vad")

    assert diagnostics._active_snapshot is not None
    assert diagnostics._active_snapshot.kind == "question"
    assert diagnostics._active_snapshot.question_start_frame_index == index


@pytest.mark.asyncio
async def test_question_snapshot_writes_raw_and_cleaned_question_audio(
    tmp_path: Path,
) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0,
            post_seconds=0,
        )
    )

    await diagnostics.start()
    first = diagnostics.record_frame(*_frame(b"1"))
    diagnostics.record_event(
        "speech_start",
        frame_index=first,
        source="vad",
        metadata={"preRollFrames": 1},
    )
    second = diagnostics.record_frame(*_frame(b"2"))
    diagnostics.record_event("speech_end", frame_index=second, source="vad")
    diagnostics.record_event("asr_final", text="你好", source="asr")

    await diagnostics._queue.join()
    manifests = await asyncio.to_thread(lambda: list(tmp_path.rglob("manifest.json")))
    assert manifests == []

    diagnostics.record_event("ask_submitted", source="api")
    await diagnostics.close()

    manifest_path = await asyncio.to_thread(
        lambda: next(tmp_path.rglob("manifest.json"))
    )
    event_dir = manifest_path.parent
    event_files = await asyncio.to_thread(lambda: list(event_dir.iterdir()))
    assert {path.name for path in event_files} == {
        "capture.wav",
        "render.wav",
        "cleaned.wav",
        "question.wav",
        "question-cleaned.wav",
        "manifest.json",
    }
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "question"
    assert manifest["question"]["startFrame"] == first
    assert manifest["question"]["endFrame"] == second
    assert manifest["question"]["captureFile"] == "question.wav"


@pytest.mark.asyncio
async def test_draft_events_strip_speculative_content_before_manifest_write(
    tmp_path: Path,
) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0,
            post_seconds=0,
        )
    )

    await diagnostics.start()
    frame = diagnostics.record_frame(*_frame())
    diagnostics.record_event("speech_start", frame_index=frame, source="vad")
    diagnostics.record_event(
        "voice_draft_transcript_ready",
        text="绝不能落盘的候选转写",
        source="voice_draft",
        metadata={
            "turnId": "00000000-0000-0000-0000-000000000123",
            "revision": 2,
            "charCount": 10,
            "utf8Bytes": 30,
            "prompt": "绝不能落盘的提示词",
            "generatedText": "绝不能落盘的草稿",
            "pcm": "绝不能落盘的音频载荷",
        },
    )
    diagnostics.record_event("asr_final", text="权威问题", source="asr")
    diagnostics.record_event("ask_submitted", source="api")
    await diagnostics.close()

    manifest_path = await asyncio.to_thread(
        lambda: next(tmp_path.rglob("manifest.json"))
    )
    manifest = json.loads(manifest_path.read_text())
    event = next(
        item
        for item in manifest["timeline"]
        if item["name"] == "voice_draft_transcript_ready"
    )
    serialized = json.dumps(event, ensure_ascii=False)

    assert event == {
        "name": "voice_draft_transcript_ready",
        "atMs": event["atMs"],
        "source": "voice_draft",
        "metadata": {
            "turnId": "00000000-0000-0000-0000-000000000123",
            "revision": 2,
            "charCount": 10,
            "utf8Bytes": 30,
        },
    }
    assert "候选转写" not in serialized
    assert "提示词" not in serialized
    assert "草稿" not in serialized
    assert "音频载荷" not in serialized


def test_draft_turn_id_is_retained_only_when_it_is_a_strict_uuid() -> None:
    _, valid = privacy_safe_event_fields(
        "voice_draft_started",
        text=None,
        metadata={
            "turnId": "00000000-0000-0000-0000-000000000123",
            "revision": 1,
        },
    )
    _, invalid = privacy_safe_event_fields(
        "voice_draft_started",
        text=None,
        metadata={"turnId": "not-a-uuid", "revision": 1},
    )

    assert valid == {
        "turnId": "00000000-0000-0000-0000-000000000123",
        "revision": 1,
    }
    assert invalid == {"revision": 1}


@pytest.mark.parametrize(
    "reason",
    ["non_speech_event", "nonlexical_vocalization"],
)
def test_draft_admission_reasons_are_retained(reason: str) -> None:
    _, metadata = privacy_safe_event_fields(
        "voice_draft_fallback",
        text=None,
        metadata={"reason": reason},
    )

    assert metadata == {"reason": reason}


@pytest.mark.asyncio
async def test_empty_asr_candidate_is_not_saved_as_a_question(
    tmp_path: Path,
) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0,
            post_seconds=0,
        )
    )

    await diagnostics.start()
    first = diagnostics.record_frame(*_frame(b"1"))
    diagnostics.record_event("speech_start", frame_index=first, source="vad")
    diagnostics.record_event("speech_end", frame_index=first, source="vad")
    diagnostics.record_event("asr_final", source="asr")

    await diagnostics._queue.join()
    assert diagnostics._active_snapshot is None
    manifests = await asyncio.to_thread(lambda: list(tmp_path.rglob("manifest.json")))
    assert manifests == []

    await diagnostics.close()


@pytest.mark.asyncio
async def test_tts_playback_snapshot_keeps_unconfirmed_interruption_audio(
    tmp_path: Path,
) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0,
            post_seconds=0,
        )
    )

    await diagnostics.start()
    first = diagnostics.record_frame(*_frame(b"1"))
    diagnostics.record_event(
        "tts_playback_started",
        source="tts",
        metadata={"messageId": "answer-1"},
    )
    second = diagnostics.record_frame(*_frame(b"2"))
    diagnostics.record_event("speech_start", frame_index=second, source="vad")
    diagnostics.record_event(
        "asr_first_partial",
        frame_index=second,
        text="停止",
        source="asr",
    )
    diagnostics.record_event("speech_end", frame_index=second, source="vad")
    diagnostics.record_event("asr_final", text="停止", source="asr")
    diagnostics.record_frame(*_frame(b"3"))
    diagnostics.record_event(
        "tts_playback_stopped",
        source="tts",
        metadata={"messageId": "answer-1"},
    )

    await diagnostics.close()

    manifest_path = await asyncio.to_thread(
        lambda: next(tmp_path.rglob("manifest.json"))
    )
    event_dir = manifest_path.parent
    event_files = await asyncio.to_thread(lambda: list(event_dir.iterdir()))
    assert {path.name for path in event_files} == {
        "capture.wav",
        "render.wav",
        "cleaned.wav",
        "manifest.json",
    }
    manifest = json.loads(manifest_path.read_text())
    assert manifest["kind"] == "playback"
    assert manifest["sourceMessageId"] == "answer-1"
    assert [event["name"] for event in manifest["timeline"]] == [
        "tts_playback_started",
        "speech_start",
        "asr_first_partial",
        "speech_end",
        "asr_final",
        "tts_playback_stopped",
    ]
    assert [frame["frameIndex"] for frame in manifest["frames"]] == [
        first,
        second,
        second + 1,
    ]


@pytest.mark.asyncio
async def test_playback_uses_latest_answer_event_message_id(
    tmp_path: Path,
) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0,
            post_seconds=0,
        )
    )

    await diagnostics.start()
    diagnostics.record_event(
        "answer_event_received",
        source="sse",
        metadata={"messageId": "answer-2"},
    )
    diagnostics.record_event("tts_playback_started", source="audio")
    diagnostics.record_frame(*_frame(b"1"))
    diagnostics.record_event("tts_playback_stopped", source="tts")
    await diagnostics.close()

    manifest_path = await asyncio.to_thread(
        lambda: next(tmp_path.rglob("manifest.json"))
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sourceMessageId"] == "answer-2"


def test_confirmed_interruption_copies_timeline_and_waits_for_post_window(
    tmp_path: Path,
) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0,
            post_seconds=0.02,
        )
    )

    index = diagnostics.record_frame(*_frame())
    diagnostics.record_event("speech_start", frame_index=index, source="vad")
    diagnostics.record_event(
        "asr_first_partial",
        frame_index=index,
        text="停止",
        source="asr",
    )
    diagnostics.record_event(
        "interruption_confirmed",
        frame_index=index,
        text="停止",
        source="asr",
    )

    assert diagnostics._active_snapshot is not None
    assert len(diagnostics._active_snapshot.frames) == 1
    diagnostics.record_frame(*_frame(b"2"))
    diagnostics.record_frame(*_frame(b"3"))

    assert diagnostics._active_snapshot is None
    snapshot = diagnostics._ready_snapshots[0]
    assert [event["name"] for event in snapshot.timeline] == [
        "speech_start",
        "asr_first_partial",
        "interruption_confirmed",
    ]


@pytest.mark.asyncio
async def test_confirmed_interruption_writes_three_wav_files_and_manifest(
    tmp_path: Path,
) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            pre_seconds=0,
            post_seconds=0,
        )
    )

    await diagnostics.start()
    diagnostics.record_frame(*_frame())
    diagnostics.record_event(
        "interruption_confirmed",
        text="停止",
        source="asr",
    )
    await diagnostics.close()

    manifest_path = await asyncio.to_thread(
        lambda: next(tmp_path.rglob("manifest.json"))
    )
    event_dir = manifest_path.parent
    event_files = await asyncio.to_thread(lambda: list(event_dir.iterdir()))
    assert {path.name for path in event_files} == {
        "capture.wav",
        "render.wav",
        "cleaned.wav",
        "manifest.json",
    }
    manifest = json.loads(manifest_path.read_text())
    assert manifest["timeline"][0]["name"] == "interruption_confirmed"
    assert manifest["audio"]["sampleRate"] == 16_000


@pytest.mark.asyncio
async def test_full_queue_drops_diagnostics_without_raising(tmp_path: Path) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path,
            queue_size=1,
            pre_seconds=0,
            post_seconds=0,
        )
    )
    await diagnostics.start()
    for _ in range(3):
        diagnostics.record_frame(*_frame())
        diagnostics.record_event("interruption_confirmed", source="asr")

    await diagnostics.close()

    assert diagnostics.dropped_events >= 1


@pytest.mark.asyncio
async def test_diagnostic_writer_errors_do_not_escape_close(tmp_path: Path) -> None:
    diagnostics = AudioDiagnostics(
        AudioDiagnosticsConfig(
            enabled=True,
            directory=tmp_path / "file",
            pre_seconds=0,
            post_seconds=0,
        )
    )
    await diagnostics.start()
    diagnostics.record_frame(*_frame())
    diagnostics.record_event("interruption_confirmed", source="asr")
    await diagnostics.close()
