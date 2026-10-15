from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest

from fish_memory.observations import environment_history_audio as audio
from fish_memory.observations.environment_history_models import (
    SEMANTIC_DOCUMENT_MAX_BYTES,
    SemanticDocument,
    snapshot_id,
)
from fish_memory.observations.environment_history_service import (
    EnvironmentHistoryService,
    _audio_words,
    _redacted_audio_revision,
    _redacted_audio_transcript,
)
from fish_memory.observations.environment_history_speakers import SpeakerSpan
from tests.observations.test_environment_history import (
    CAPTURE,
    FISH,
    START,
    Repository,
    Summarizer,
    Transcriber,
    window,
)
from tests.observations.test_environment_history_audio_speakers import (
    PCM,
    setup_transcriber,
)
from tests.observations.test_environment_history_audio_speakers import (
    window as transcribe_window,
)


@pytest.mark.parametrize(
    ("text", "tokens", "times"),
    [
        ("好😀。", ("好", "😀", "。"), (0.5, 2.5, 2.6)),
        ("C++", ("C", "+", "+"), (0.5, 2.5, 2.6)),
        ("$9000", ("$", "9000"), (0.5, 2.5)),
    ],
)
def test_aligned_symbols_survive_formatting_and_later_context_sanitizing(
    text, tokens, times
):
    transcript = audio.LocalSystemAudioTranscriber._aligned_transcript(
        audio.SystemAudioTranscript(text, "Speech", tokens=tokens, timestamps=times),
        (SpeakerSpan(0, 16_000, "A"), SpeakerSpan(32_000, 48_000, "B")),
        3,
    )
    assert "".join(part.text for part in transcript.segments) == text
    rendered = audio.format_system_audio_transcript(transcript)
    assert _audio_words(rendered) == text
    assert _audio_words(audio.format_system_audio_text(rendered)) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed_at", [500, 90])
async def test_complete_silence_cannot_hide_a_long_gap_or_clock_rewind(
    monkeypatch, resumed_at
):
    transcriber = setup_transcriber(monkeypatch)
    try:
        await transcribe_window(transcriber)
        assert transcriber._alignment_history
        diarizer = transcriber._speaker_diarizer
        await transcriber.transcribe_window(
            b"\0\0" * 160_000,
            stream_id="fish:capture",
            window_index=1,
            started_at=resumed_at,
        )
        assert diarizer.reset_count == 1
        assert diarizer.silence_calls == [("fish:capture", 1)]
        assert not transcriber._alignment_history
        await transcriber.transcribe_window(
            PCM,
            stream_id="fish:capture",
            window_index=2,
            started_at=resumed_at + 10,
        )
        assert diarizer.reset_count == 1
        assert transcriber._shared_client.calls == [PCM, PCM]
    finally:
        await transcriber.close()


def _split_secret_transcript() -> audio.SystemAudioTranscript:
    # Synthetic provider-shaped text only, never real credentials.
    fake = "sk-" + "a" * 20
    return audio.SystemAudioTranscript(
        fake,
        "Speech",
        (
            audio.SystemAudioSegment(
                0, 1, fake[:10], "A", alignment_reason="token_time"
            ),
            audio.SystemAudioSegment(
                1, 2, fake[10:], "B", alignment_reason="token_time"
            ),
        ),
        tokens=tuple(fake),
        timestamps=tuple(index * 0.05 for index in range(len(fake))),
    )


def test_global_redaction_cannot_be_undone_by_separately_cleaned_parts():
    transcript = _split_secret_transcript()
    result = _redacted_audio_transcript(transcript)
    assert result.text == "[REDACTED]"
    assert result.segments == ()
    assert result.tokens == result.timestamps == ()
    assert result.alignment_status == "redacted_alignment"
    assert audio.format_system_audio_transcript(result) == "[REDACTED]"


def test_retrospective_redaction_uses_the_same_whole_text_boundary():
    transcript = _split_secret_transcript()
    revision = audio.SystemAudioRevision(
        1, START.timestamp(), transcript.text, transcript.segments, "Speech"
    )
    result = _redacted_audio_revision(revision)
    assert result.text == "[REDACTED]" and result.segments == ()
    assert result.window_index == 1 and result.started_at == START.timestamp()


def test_safe_transcript_keeps_alignment_but_redacted_text_drops_original_tokens():
    plain = audio.SystemAudioTranscript(
        "你好", "Speech", tokens=("你", "好"), timestamps=(0.2, 0.4)
    )
    assert _redacted_audio_transcript(plain) == plain
    secret = _split_secret_transcript()
    secret = replace(secret, segments=())
    result = _redacted_audio_transcript(secret)
    assert result.text == "[REDACTED]"
    assert result.tokens == result.timestamps == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("full_budget", [False, True])
async def test_persisted_diagnostics_match_the_revision_that_fit_in_the_document(
    full_budget,
):
    original = audio.SystemAudioTranscript(
        "好",
        "Speech",
        (audio.SystemAudioSegment(0, 1, "好", "A", alignment_reason="token_time"),),
        alignment_status="aligned",
    )
    revised = replace(
        original, segments=(replace(original.segments[0], speaker_id="声音abcdef-T01"),)
    )
    original_rendered = audio.format_system_audio_transcript(original)
    revised_rendered = audio.format_system_audio_transcript(revised)
    document = SemanticDocument(
        summary_version=2,
        audiovisual_summary="听到的话：\n" + original_rendered,
        audio=original_rendered,
        visual_summary="x",
    )
    if full_budget:
        padding = SEMANTIC_DOCUMENT_MAX_BYTES - len(document.model_dump_json().encode())
        document = SemanticDocument.model_validate(
            {
                **document.model_dump(),
                "visualSummary": "x" * (padding + 1),
            }
        )
        assert len(document.model_dump_json().encode()) == SEMANTIC_DOCUMENT_MAX_BYTES

    class AudioSource(Transcriber):
        async def transcribe(self, pcm):
            return original

    class InflightVisualSummary(Summarizer):
        async def summarize_window(self, window, transcript):
            # A later acoustic result arrives while this visual request is in
            # flight. The source recording remains unchanged until _run saves.
            previous = service._recent_audio[FISH][(CAPTURE, 0)]
            service._recent_audio[FISH][(CAPTURE, 0)] = replace(
                previous,
                text=revised_rendered,
                segments=revised.segments,
            )
            return document

    repository = Repository()
    service = EnvironmentHistoryService(
        repository,
        InflightVisualSummary(),
        AudioSource(),
        now=lambda: START + timedelta(seconds=30),
    )
    service.set_enabled(FISH, True)
    try:
        assert await service.enqueue(FISH, window()) == "pending"
        key = snapshot_id(FISH, CAPTURE, 0, 10)
        await asyncio.wait_for(service._jobs[key][1], 1)
        stored = repository.values[(FISH, key)]
        evidence = repository.evidence[(FISH, key)]
        if full_budget:
            assert stored.document.audio == original_rendered
            assert evidence["speakerAttributionRevised"] is False
            assert evidence["audioSegments"][0]["speakerId"] == "A"
        else:
            assert stored.document.audio == revised_rendered
            assert evidence["speakerAttributionRevised"] is True
            assert evidence["audioSegments"][0]["speakerId"] == "声音abcdef-T01"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_silence_delivers_old_speaker_correction_to_persisted_parent_sources():
    import json

    from fish_memory.observations.environment_history_client import _narrative_content
    from fish_memory.observations.environment_history_narrative import (
        NarrativeParagraph,
    )
    from fish_memory.observations.environment_history_service import _audio_range_header
    from tests.observations.test_environment_history_hour import (
        START as HISTORY_START,
    )
    from tests.observations.test_environment_history_hour import (
        revision_job,
        semantic_snapshot,
        start_service,
    )

    service, repository, _ = await start_service(
        now=HISTORY_START + timedelta(seconds=200)
    )
    original = audio.SystemAudioTranscript(
        "仍在等待。", "Speech", (audio.SystemAudioSegment(0, 1, "仍在等待。", None),)
    )
    original_text = audio.format_system_audio_transcript(original)
    leaf = replace(
        semantic_snapshot(10, 0, "unused"),
        document=SemanticDocument(
            summary_version=3,
            audio=original_text,
            audiovisual_summary=f"听到的话：\n{original_text}",
        ),
    )
    parent = replace(
        semantic_snapshot(60, 0, "unused"),
        document=SemanticDocument(
            summary_version=3,
            narrative_paragraphs=(
                NarrativeParagraph(text="有人还在等候。", source_ids=("source-0",)),
            ),
            narrative_sources={
                "source-0": _audio_range_header(leaf.started_at, leaf.ended_at)
                + "\n"
                + original_text
            },
        ),
    )
    persisted = {leaf.id: leaf, parent.id: parent}
    versions = []

    async def persist(value, *, through_index, segments):
        persisted[value.id] = value
        versions.append(through_index)
        return value

    repository.revise_attribution = persist
    try:
        service._remember_audio_transcript(revision_job(0), original)
        service._remember_snapshot(leaf, publish=False)
        service._remember_snapshot(parent, publish=False)
        correction = audio.SystemAudioRevision(
            0,
            leaf.started_at.timestamp(),
            original.text,
            (replace(original.segments[0], speaker_id="声音abcdef-01"),),
            original.event,
        )
        silent = audio.SystemAudioTranscript("", "silence", revisions=(correction,))
        job = revision_job(18)
        service._remember_audio_transcript(job, silent)
        await service._persist_attribution_revisions(job, silent)
        assert versions == [18, 18]
        assert "声音abcdef-01" in persisted[leaf.id].document.audio
        updated = persisted[parent.id].document
        assert "声音abcdef-01" in updated.narrative_sources["source-0"]
        assert updated.narrative_paragraphs == parent.document.narrative_paragraphs
        content, sources, _ = _narrative_content((persisted[parent.id],), None)
        assert (
            json.loads(content)["segments"][0]["turns"][0]["speaker"] == "声音abcdef-01"
        )
        assert list(sources) == ["source-0"]
        assert _audio_words(persisted[leaf.id].document.audio) == _audio_words(
            original_text
        )
    finally:
        await service.close()
