from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from fish_memory.companion.shared_asr import SharedAsrResponse
from fish_memory.observations import environment_history_audio as audio
from fish_memory.observations.environment_history_speakers import SpeakerSpan

PCM = b"\x80\x01" * 160_000
SPANS = (
    SpeakerSpan(0, 32_000, "声音abcdef-01"),
    SpeakerSpan(32_000, 80_000, "声音abcdef-02"),
    SpeakerSpan(80_000, 160_000, None, overlap=True),
)


def setup_transcriber(monkeypatch, *, spans=SPANS, responses=None, enabled=True):
    class Diarizer:
        def __init__(self, settings):
            self.calls = []
            self.closed = False
            self.reset_count = 0
            self.boundary_clear_count = 0
            self.silence_calls = []
            self.revisions = ()
            self.track_aliases = ()

        def segment(self, pcm, *, stream_id, window_index):
            self.calls.append((pcm, stream_id, window_index))
            if isinstance(spans, Exception):
                raise spans
            return spans

        def observe_silence(self, *, stream_id, window_index):
            self.silence_calls.append((stream_id, window_index))

        def close(self):
            self.closed = True

        def reset(self):
            self.reset_count += 1

        def clear_boundary(self):
            self.boundary_clear_count += 1

    class SharedClient:
        def __init__(self, path):
            self.calls = []
            self.responses = list(responses or [])

        async def recognize(self, pcm):
            self.calls.append(pcm)
            await asyncio.sleep(0)
            if self.responses:
                return self.responses.pop(0)
            text = f"台词{len(self.calls)}"
            return response(text, tuple(0.5 + i * 0.15 for i in range(len(text))))

    monkeypatch.setenv("FISH_MEMORY_SHARED_ASR_SOCKET", "/tmp/unused-speaker-test.sock")
    monkeypatch.setattr(audio, "LocalSpeakerDiarizer", Diarizer)
    monkeypatch.setattr(audio, "SharedAsrClient", SharedClient)
    monkeypatch.setattr(
        audio,
        "StreamingAsr",
        lambda _: (_ for _ in ()).throw(AssertionError("duplicate ASR model")),
    )
    return audio.LocalSystemAudioTranscriber(
        SimpleNamespace(environment_history_speaker_enabled=enabled)
    )


def response(text, times):
    return SharedAsrResponse("ok", text, "Speech", tuple(text), tuple(times))


async def window(transcriber, index=0, *, pcm=PCM):
    return await transcriber.transcribe_window(
        pcm, stream_id="fish:capture", window_index=index, started_at=100 + index * 10
    )


@pytest.mark.asyncio
async def test_complete_pcm_is_decoded_once_before_voice_time_association(monkeypatch):
    text = "第一句。第二句。一起说。"
    transcriber = setup_transcriber(
        monkeypatch,
        responses=[
            response(
                text, (0.5, 0.65, 0.8, 0.95, 2.5, 2.65, 2.8, 2.95, 6, 6.15, 6.3, 6.45)
            )
        ],
    )
    try:
        result = await window(transcriber)
        assert transcriber._shared_client.calls == [PCM]
        assert transcriber._speaker_diarizer.calls == [(PCM, "fish:capture", 0)]
        assert (
            result.diarization_status == "ok" and result.alignment_status == "aligned"
        )
        assert result.text == text
        assert "".join(segment.text for segment in result.segments) == text
        assert [segment.speaker_id for segment in result.segments] == [
            "声音abcdef-01",
            "声音abcdef-02",
            None,
        ]
        assert result.segments[-1].overlap
        rendered = audio.format_system_audio_transcript(result)
        assert "声音abcdef-02：第二句。" in rendered
        assert "多人同时说话（归属未确定）：一起说。" in rendered
        assert transcriber._decoder is None
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_microphone_priority_does_not_queue_crops_or_claim_voice(monkeypatch):
    transcriber = setup_transcriber(monkeypatch, responses=[SharedAsrResponse("busy")])
    try:
        result = await window(transcriber)
        assert result.text == "" and result.event == "unavailable"
        assert not result.segments
        assert transcriber._shared_client.calls == [PCM]
        assert transcriber._speaker_diarizer is None
        assert transcriber._decoder is None
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_missing_alignment_preserves_whole_text_without_cropped_retry(
    monkeypatch,
):
    text = "奶奶，你能借我钱吗？今天想吃什么？"
    transcriber = setup_transcriber(
        monkeypatch, responses=[SharedAsrResponse("ok", text, "Speech")]
    )
    try:
        result = await window(transcriber)
        assert result.diarization_status == "unassigned"
        assert result.alignment_status == "missing_alignment"
        assert result.text == text and result.segments == ()
        assert audio.format_system_audio_transcript(result) == text
        assert transcriber._shared_client.calls == [PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spans", [(), tuple(SpeakerSpan(n, n + 1, None) for n in range(17))]
)
async def test_empty_or_fragmented_segmentation_preserves_whole_transcript(
    monkeypatch, spans
):
    transcriber = setup_transcriber(
        monkeypatch,
        spans=spans,
        responses=[SharedAsrResponse("ok", "<Music>配乐里的对白</Music>", "Music")],
    )
    try:
        result = await window(transcriber)
        assert result.diarization_status == "unassigned"
        assert transcriber._shared_client.calls == [PCM]
        assert audio.format_system_audio_transcript(result) == "配乐里的对白"
        assert result.segments == ()
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_failed_segmentation_uses_plain_asr_and_bounded_retry(
    monkeypatch, caplog
):
    transcriber = setup_transcriber(monkeypatch, spans=RuntimeError("secret audio"))
    try:
        first = await window(transcriber)
        second = await window(transcriber, 1)
        assert first.diarization_status == second.diarization_status == "failed"
        assert first.text == "台词1"
        assert first.segments == ()
        assert len(transcriber._speaker_diarizer.calls) == 1
        assert transcriber._shared_client.calls == [PCM, PCM]
        assert "secret audio" not in caplog.text
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_invalid_segment_time_cannot_claim_speaker_for_other_audio(monkeypatch):
    transcriber = setup_transcriber(
        monkeypatch, spans=(SpeakerSpan(0, 160_001, "wrong"),)
    )
    try:
        result = await window(transcriber)
        assert result.diarization_status == "failed"
        assert not result.segments
        assert "wrong" not in result.text
        assert transcriber._shared_client.calls == [PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_silence_does_not_load_any_speaker_or_asr_model(monkeypatch):
    transcriber = setup_transcriber(monkeypatch)
    try:
        result = await window(transcriber, pcm=b"\x00\x00" * 160_000)
        assert result.diarization_status == "silence"
        assert transcriber._speaker_diarizer is None
        assert transcriber._shared_client.calls == []
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_disabled_speakers_keep_existing_plain_asr(monkeypatch):
    transcriber = setup_transcriber(monkeypatch, enabled=False)
    try:
        result = await window(transcriber)
        assert result.diarization_status == "disabled"
        assert transcriber._speaker_diarizer is None
        assert result.text == "台词1"
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_windows_are_serialized_without_interleaving_voice_turns(monkeypatch):
    transcriber = setup_transcriber(monkeypatch)
    try:
        first, second = await asyncio.gather(
            window(transcriber), window(transcriber, 1)
        )
        assert first.text == "台词1" and second.text == "台词2"
        assert [segment.text for segment in first.segments] == ["台词1"]
        assert [segment.text for segment in second.segments] == ["台词2"]
        assert transcriber._shared_client.calls == [PCM, PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed_at", [500, 90])
async def test_wall_clock_gap_or_rewind_resets_even_with_adjacent_index(
    monkeypatch, resumed_at
):
    transcriber = setup_transcriber(monkeypatch)
    try:
        await window(transcriber)
        await transcriber.transcribe_window(
            PCM, stream_id="fish:capture", window_index=1, started_at=resumed_at
        )
        assert transcriber._speaker_diarizer.reset_count == 1
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_cancelled_whole_window_never_starts_segmentation(monkeypatch):
    transcriber = setup_transcriber(monkeypatch)
    entered = asyncio.Event()
    calls = []

    async def recognize(pcm):
        calls.append(pcm)
        entered.set()
        await asyncio.Future()

    transcriber._shared_client.recognize = recognize
    task = asyncio.create_task(window(transcriber))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await transcriber.release()
        assert transcriber._speaker_diarizer is None
        assert calls == [PCM]
    finally:
        task.cancel()
        await transcriber.close()


@pytest.mark.asyncio
async def test_release_waits_for_cancelled_native_segmentation_then_resets(monkeypatch):
    transcriber = setup_transcriber(monkeypatch)
    entered, unblock = threading.Event(), threading.Event()
    original_segment = audio.LocalSpeakerDiarizer.segment

    def segment(self, *args, **kwargs):
        entered.set()
        assert unblock.wait(2)
        return original_segment(self, *args, **kwargs)

    monkeypatch.setattr(audio.LocalSpeakerDiarizer, "segment", segment)
    task = asyncio.create_task(window(transcriber))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release = asyncio.create_task(transcriber.release())
        await asyncio.sleep(0)
        assert not release.done()
        unblock.set()
        await asyncio.wait_for(release, 1)
        assert transcriber._speaker_diarizer is None
        assert transcriber._shared_client.calls == [PCM]
    finally:
        unblock.set()
        await transcriber.close()


def test_unknown_and_overlapping_speech_cannot_be_rendered_as_named_speaker():
    result = audio.SystemAudioTranscript(
        "discard stale flattened text",
        "Speech",
        (
            audio.SystemAudioSegment(
                0, 1, "<Music>对白</Music>", "声音abcdef-01", True
            ),
            audio.SystemAudioSegment(1, 2, "回答", None),
        ),
    )
    text = audio.format_system_audio_transcript(result)
    assert "声音abcdef-01" not in text
    assert "多人同时说话（归属未确定）：对白" in text
    assert "说话人未确定：回答" in text
    assert "Music" not in text


@pytest.mark.asyncio
async def test_partial_capture_clears_boundary_but_not_voice_templates(monkeypatch):
    transcriber = setup_transcriber(monkeypatch)
    try:
        await window(transcriber)
        await transcriber.transcribe_window(
            PCM,
            stream_id="fish:capture",
            window_index=1,
            started_at=110,
            audio_complete=False,
        )
        assert transcriber._speaker_diarizer.boundary_clear_count == 2
        assert transcriber._speaker_diarizer.reset_count == 0
        await window(transcriber, 2)
        assert transcriber._speaker_diarizer.boundary_clear_count == 2
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_small_clock_gap_disables_boundary_without_forgetting_voices(monkeypatch):
    transcriber = setup_transcriber(monkeypatch)
    try:
        await window(transcriber)
        await transcriber.transcribe_window(
            PCM,
            stream_id="fish:capture",
            window_index=1,
            started_at=111,
        )
        assert transcriber._speaker_diarizer.boundary_clear_count == 1
        assert transcriber._speaker_diarizer.reset_count == 0
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_unconfirmed_local_turns_and_diagnostics_survive_transcription(
    monkeypatch,
):
    spans = (
        SpeakerSpan(
            0,
            32_000,
            None,
            local_speaker_id="local:one:0:0:0",
            reason="short_evidence",
            clean_seconds=0.8,
            identity_source="local",
        ),
        SpeakerSpan(
            32_000,
            80_000,
            None,
            local_speaker_id="local:one:0:1:1",
            reason="match_ambiguous",
            clean_seconds=1.5,
            best_similarity=0.71,
            second_similarity=0.70,
            identity_source="local",
        ),
    )
    transcriber = setup_transcriber(
        monkeypatch,
        spans=spans,
        responses=[
            response("第一句。第二句。", (0.5, 0.6, 0.7, 0.8, 2.5, 2.6, 2.7, 2.8))
        ],
    )
    try:
        result = await window(transcriber)
        assert len(result.segments) == 2
        assert result.segments[1].reason == "match_ambiguous"
        assert result.segments[1].best_similarity == 0.71
        rendered = audio.format_system_audio_transcript(result)
        assert "说话人未确定：第一句。" in rendered
        assert "说话人未确定：第二句。" in rendered
        assert all(
            term not in rendered
            for term in ("本片段声音", "local:", "0.71", "match_ambiguous")
        )
        assert transcriber._shared_client.calls == [PCM]
    finally:
        await transcriber.close()


def test_unverified_islands_keep_turn_boundaries_without_inventing_people():
    def render(scope):
        return audio.format_system_audio_transcript(
            audio.SystemAudioTranscript(
                "",
                "Speech",
                (
                    audio.SystemAudioSegment(
                        0, 1, "第一句", None, local_speaker_id=scope
                    ),
                    audio.SystemAudioSegment(
                        2, 3, "第二句", None, local_speaker_id=scope
                    ),
                ),
            )
        )

    first, second = render("local:capture:0:0:0"), render("local:capture:1:2:0")
    assert first.count("说话人未确定") == second.count("说话人未确定") == 2
    assert "[00:00.00—00:01.00]" in first
    assert "[00:02.00—00:03.00]" in first
    assert "local:" not in first + second


def test_track_label_survives_later_global_identity_confirmation():
    before = audio.SystemAudioTranscript(
        "",
        "Speech",
        (
            audio.SystemAudioSegment(
                0,
                2,
                "我先说明情况。",
                None,
                track_id="track:abc123:7",
                track_reason="created",
            ),
        ),
    )
    after = audio.SystemAudioTranscript(
        "",
        "Speech",
        (
            audio.SystemAudioSegment(
                0,
                1,
                "我给你加钱。",
                "声音abc123-03",
                track_id="track:abc123:7",
                track_reason="matched",
            ),
        ),
    )
    for result in (before, after):
        rendered = audio.format_system_audio_transcript(result)
        assert "声音abc123-T07：" in rendered
        assert "track:" not in rendered
        assert "声音abc123-03" not in rendered


@pytest.mark.asyncio
async def test_track_metadata_and_silence_boundaries_survive_whole_asr(monkeypatch):
    spans = (
        SpeakerSpan(0, 32_000, None, track_id="track:abc123:1", track_reason="created"),
        SpeakerSpan(
            48_000, 64_000, None, track_id="track:abc123:1", track_reason="matched"
        ),
    )
    transcriber = setup_transcriber(
        monkeypatch,
        spans=spans,
        responses=[
            response("第一句。第二句。", (0.5, 0.6, 0.7, 0.8, 3.2, 3.3, 3.4, 3.5))
        ],
    )
    try:
        result = await window(transcriber)
        assert [s.track_id for s in result.segments] == ["track:abc123:1"] * 2
        assert [s.track_reason for s in result.segments] == ["created", "matched"]
        assert (
            audio.format_system_audio_transcript(result).count("声音abc123-T01：") == 2
        )
        assert transcriber._shared_client.calls == [PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_decoder_metadata_does_not_become_text_or_people(monkeypatch):
    transcriber = setup_transcriber(
        monkeypatch,
        responses=[
            SharedAsrResponse(
                "ok",
                "<|zh|><|NEUTRAL|><|Speech|>好",
                "Speech",
                ("<|zh|>", "<|NEUTRAL|>", "<|Speech|>", "好"),
                (0, 0.1, 0.2, 0.5),
            )
        ],
    )
    try:
        result = await window(transcriber)
        assert len(result.segments) == 1
        assert result.segments[0].text == result.text == "好"
        assert "<|" not in audio.format_system_audio_transcript(result)
        assert transcriber._shared_client.calls == [PCM]
    finally:
        await transcriber.close()


@pytest.mark.parametrize("text", ["<|Speech|>", "<|zh|><|NEUTRAL|>", "。", "?!…", " "])
def test_empty_audio_evidence_is_filtered(text):
    assert audio.format_system_audio_text(text) == ""


@pytest.mark.parametrize("text", ["好", "不", "嗯", "No", "42", "奶奶，你能借我钱吗？"])
def test_meaningful_short_audio_is_preserved(text):
    assert audio.format_system_audio_text(f"<|zh|><|Speech|>{text}") == text


def test_old_formatted_transcripts_drop_empty_speakers_and_empty_time_blocks():
    header = "2026-09-06 06:27:18 — 2026-09-06 06:27:28 UTC"
    empty = (
        header + "\n[00:02.44—00:02.49] 本片段声音B：<|Speech|>"
        "\n[00:03.00—00:04.00] 声音C：。"
    )
    assert audio.format_system_audio_text(empty) == ""
    meaningful = "[00:00.00—00:00.30] 声音abc123-T01：好"
    assert audio.format_system_audio_text(empty + "\n\n" + meaningful) == meaningful


@pytest.mark.asyncio
@pytest.mark.parametrize("times", [(0.8, 0.2), (0.2, float("nan")), (0.2, 10.1)])
async def test_bad_token_times_keep_complete_text_without_claiming_a_speaker(
    monkeypatch, times
):
    transcriber = setup_transcriber(
        monkeypatch,
        responses=[SharedAsrResponse("ok", "你好", "Speech", ("你", "好"), times)],
    )
    try:
        result = await window(transcriber)
        assert result.text == "你好" and not result.segments
        assert result.alignment_status == "invalid_alignment"
        assert audio.format_system_audio_transcript(result) == "你好"
        assert transcriber._shared_client.calls == [PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_short_window_uses_all_received_pcm_and_bounds_token_evidence(
    monkeypatch,
):
    pcm = PCM[:16_000]
    transcriber = setup_transcriber(
        monkeypatch,
        spans=(SpeakerSpan(0, 8000, "声音abcdef-01"),),
        responses=[response("您好。", (0.12, 0.24, 0.36))],
    )
    try:
        result = await window(transcriber, pcm=pcm)
        assert result.text == "您好。" and len(result.segments) == 1
        assert (
            0
            <= result.segments[0].start_seconds
            < result.segments[0].end_seconds
            <= 0.5
        )
        assert transcriber._shared_client.calls == [pcm]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_later_independent_speaker_evidence_revises_cached_tokens_without_asr(
    monkeypatch,
):
    spans = (SpeakerSpan(0, 32_000, None, local_speaker_id="local:earlier"),)
    transcriber = setup_transcriber(monkeypatch, spans=spans)
    try:
        first = await window(transcriber)
        assert first.segments[0].speaker_id is None
        transcriber._speaker_diarizer.revisions = (
            SimpleNamespace(
                window_index=0,
                start_sample=0,
                end_sample=32_000,
                speaker_id="声音abc123-01",
                track_id="track:abc123:1",
                reason="retrospective_match",
            ),
        )
        second = await window(transcriber, 1)
        assert len(second.revisions) == 1
        revised = second.revisions[0]
        assert revised.window_index == 0 and revised.started_at == 100
        assert revised.text == first.text == "台词1"
        assert "".join(segment.text for segment in revised.segments) == first.text
        assert revised.segments[0].speaker_id == "声音abc123-01"
        assert revised.segments[0].track_id == "track:abc123:1"
        assert revised.segments[0].identity_source == "retrospective"
        assert transcriber._shared_client.calls == [PCM, PCM]
        assert len(transcriber._speaker_diarizer.calls) == 2
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_track_alias_updates_the_cached_label_without_new_transcription(
    monkeypatch,
):
    spans = (SpeakerSpan(0, 32_000, None, track_id="track:abc123:2"),)
    transcriber = setup_transcriber(monkeypatch, spans=spans)
    try:
        first = await window(transcriber)
        transcriber._speaker_diarizer.track_aliases = (
            ("track:abc123:2", "track:abc123:1"),
        )
        second = await window(transcriber, 1)
        assert second.revisions[0].text == first.text
        assert second.revisions[0].segments[0].track_id == "track:abc123:1"
        assert second.revisions[0].segments[0].track_reason == "track_alias"
        assert transcriber._shared_client.calls == [PCM, PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_revision_does_not_borrow_a_neighbouring_unknown_intervals_identity(
    monkeypatch,
):
    spans = (
        SpeakerSpan(0, 32_000, None),
        SpeakerSpan(32_000, 160_000, "声音abc123-02"),
    )
    transcriber = setup_transcriber(
        monkeypatch, spans=spans, responses=[response("你好", (1.95, 2.05))]
    )
    try:
        first = await window(transcriber)
        assert [segment.speaker_id for segment in first.segments] == [
            None,
            "声音abc123-02",
        ]
        transcriber._speaker_diarizer.revisions = (
            SimpleNamespace(
                window_index=0,
                start_sample=0,
                end_sample=32_000,
                speaker_id="声音abc123-01",
                track_id="track:abc123:1",
                reason="retrospective_match",
            ),
        )
        second = await window(transcriber, 1)
        assert second.revisions[0].text == "你好"
        assert [segment.speaker_id for segment in second.revisions[0].segments] == [
            "声音abc123-01",
            "声音abc123-02",
        ]
        assert transcriber._shared_client.calls == [PCM, PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("next_stream", "next_index", "next_time"),
    [
        ("fish:other", 1, 110),
        ("fish:capture", 31, 410),
    ],
)
async def test_retrospective_cache_cannot_revise_another_stream_or_expired_audio(
    monkeypatch, next_stream, next_index, next_time
):
    spans = (SpeakerSpan(0, 32_000, None, track_id="track:abc123:2"),)
    transcriber = setup_transcriber(monkeypatch, spans=spans)
    try:
        await window(transcriber)
        transcriber._speaker_diarizer.track_aliases = (
            ("track:abc123:2", "track:abc123:1"),
        )
        result = await transcriber.transcribe_window(
            PCM, stream_id=next_stream, window_index=next_index, started_at=next_time
        )
        assert result.revisions == ()
        assert len(transcriber._alignment_history) == 1
        assert transcriber._shared_client.calls == [PCM, PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_complete_silence_advances_window_without_loading_or_clearing_models(
    monkeypatch,
):
    transcriber = setup_transcriber(monkeypatch)
    try:
        await window(transcriber)
        diarizer = transcriber._speaker_diarizer
        result = await window(transcriber, 1, pcm=b"\0\0" * 160_000)
        assert result.event == "silence"
        assert diarizer.silence_calls == [("fish:capture", 1)]
        assert transcriber._last_speaker_window == ("fish:capture", 1, 110)
        await window(transcriber, 2)
        assert transcriber._speaker_diarizer is diarizer
        assert diarizer.reset_count == diarizer.boundary_clear_count == 0
        assert transcriber._shared_client.calls == [PCM, PCM]
    finally:
        await transcriber.close()


@pytest.mark.asyncio
async def test_partial_silence_clears_continuity_without_claiming_completed_window(
    monkeypatch,
):
    transcriber = setup_transcriber(monkeypatch)
    try:
        await window(transcriber)
        diarizer = transcriber._speaker_diarizer
        await transcriber.transcribe_window(
            b"\0\0" * 160_000,
            stream_id="fish:capture",
            window_index=1,
            started_at=110,
            audio_complete=False,
        )
        assert diarizer.silence_calls == []
        assert diarizer.boundary_clear_count == 1
        assert transcriber._last_speaker_window == ("fish:capture", 0, 100)
        await window(transcriber, 2)
        assert diarizer.boundary_clear_count == 2 and diarizer.reset_count == 0
        assert transcriber._shared_client.calls == [PCM, PCM]
    finally:
        await transcriber.close()


def test_text_cap_never_keeps_times_for_the_uncapped_transcript():
    transcript = audio._transcript("好" * 1001, "Speech", ("好",) * 1001, (0.2,) * 1001)
    assert len(transcript.text) == 1000
    assert transcript.tokens == transcript.timestamps == ()


@pytest.mark.asyncio
async def test_complete_window_boundary_revisions_arrive_during_silence_without_asr(
    monkeypatch,
):
    from fish_memory.observations.environment_history_rolling_speakers import (
        SpeakerWindowRevision,
    )

    transcriber = setup_transcriber(
        monkeypatch,
        spans=(SpeakerSpan(0, 32000, None),),
        responses=[response("你好", (0.2, 0.7))],
    )
    try:
        original = await window(transcriber)
        diarizer = transcriber._speaker_diarizer
        for index in range(1, 18):
            await window(transcriber, index, pcm=bytes(320000))
        diarizer.window_revisions = (
            SpeakerWindowRevision(
                0,
                (SpeakerSpan(0, 8000, "voice-A"), SpeakerSpan(8000, 32000, "voice-B")),
            ),
        )
        result = await window(transcriber, 18, pcm=bytes(320000))
        assert result.event == "silence"
        assert len(transcriber._shared_client.calls) == 1
        assert len(result.revisions) == 1
        correction = result.revisions[0]
        assert correction.text == original.text == "你好"
        assert [s.speaker_id for s in correction.segments] == ["voice-A", "voice-B"]
        assert "".join(s.text for s in correction.segments) == "你好"
    finally:
        await transcriber.close()
