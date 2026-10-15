from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.observations.environment_history_audio import (
    SystemAudioSegment,
    SystemAudioTranscript,
    _transcript,
    format_system_audio_text,
)
from fish_memory.observations.environment_history_models import (
    EnvironmentHistoryWindow,
    SemanticDocument,
    SemanticSnapshot,
    snapshot_id,
)
from fish_memory.observations.environment_history_service import (
    EnvironmentHistoryService,
    HistoryDisabled,
    HistoryQueueFull,
)
from fish_memory.observations.environment_history_store import (
    EnvironmentHistoryConflict,
)
from tests.test_environment_frames import _request

FISH = UUID(int=1)
CAPTURE = UUID(int=2)
START = datetime(2026, 9, 3, tzinfo=UTC)


def expected_local_range(start: datetime, end: datetime, *, date: bool = False):
    # Windows cannot force process TZ with tzset; these summaries intentionally
    # use the current system zone, while stored timestamps remain UTC.
    if date:
        return " — ".join(
            value.astimezone().isoformat(timespec="seconds") for value in (start, end)
        )
    return " — ".join(value.astimezone().strftime("%H:%M:%S") for value in (start, end))


class Repository:
    def __init__(self):
        self.values = {}
        self.evidence = {}
        self.recovery_calls = 0
        self.recovery_scanned = asyncio.Event()

    async def get(self, fish_id, snapshot_id):
        return self.values.get((fish_id, snapshot_id))

    async def save(self, value, *, evidence=None):
        self.values[(value.fish_id, value.id)] = value
        self.evidence[(value.fish_id, value.id)] = evidence
        return value

    async def minute_windows(self, fish_id, capture_session_id, first_index):
        return tuple(
            sorted(
                (
                    v
                    for v in self.values.values()
                    if v.fish_id == fish_id
                    and v.capture_session_id == capture_session_id
                    and v.period_seconds == 10
                    and first_index <= v.window_index < first_index + 6
                ),
                key=lambda v: v.window_index,
            )
        )

    async def child_snapshots(
        self,
        fish_id,
        capture_session_id,
        first_index,
        *,
        child_period_seconds,
        parent_period_seconds,
    ):
        if {60: 10, 300: 60, 1800: 300}.get(parent_period_seconds) != (
            child_period_seconds
        ):
            raise ValueError("invalid hierarchy edge")
        child_count = parent_period_seconds // child_period_seconds
        step = child_period_seconds // 10
        expected = [first_index + offset * step for offset in range(child_count)]
        values = tuple(
            sorted(
                (
                    value
                    for value in self.values.values()
                    if value.fish_id == fish_id
                    and value.capture_session_id == capture_session_id
                    and value.period_seconds == child_period_seconds
                    and value.window_index in expected
                ),
                key=lambda value: value.window_index,
            )
        )
        if [value.window_index for value in values] != expected:
            return ()
        if any(
            left.ended_at != right.started_at
            for left, right in zip(values[:-1], values[1:], strict=True)
        ):
            return ()
        return values

    async def recent_hour_snapshots(self, fish_id, *, since, through):
        return tuple(
            sorted(
                (
                    value
                    for value in self.values.values()
                    if value.fish_id == fish_id
                    and value.ended_at >= since
                    and value.ended_at <= through
                ),
                key=lambda value: (
                    value.started_at,
                    -value.period_seconds,
                    str(value.id),
                ),
            )
        )

    async def delete_compacted_windows(self, fish_id, *, ended_before):
        del fish_id, ended_before
        return 0

    async def context_snapshots(self, fish_id):
        values = [v for v in self.values.values() if v.fish_id == fish_id]
        minute = max(
            (v for v in values if v.period_seconds == 60),
            key=lambda v: v.ended_at,
            default=None,
        )
        tail = tuple(
            sorted(
                (
                    v
                    for v in values
                    if v.period_seconds == 10
                    and (minute is None or v.ended_at > minute.ended_at)
                ),
                key=lambda v: v.started_at,
            )
        )
        return minute, tail[-12:]

    async def pending_minute_windows(
        self,
        fish_id,
        limit=4,
        *,
        since=None,
        through=None,
    ):
        self.recovery_calls += 1
        self.recovery_scanned.set()
        groups = {}
        for value in self.values.values():
            if (
                value.fish_id == fish_id
                and value.period_seconds == 10
                and (since is None or value.started_at >= since)
                and (through is None or value.ended_at <= through)
            ):
                key = (value.capture_session_id, (value.window_index // 6) * 6)
                groups.setdefault(key, []).append(value)
        pending = []
        for (capture, first), values in groups.items():
            ordered = sorted(values, key=lambda value: value.window_index)
            if [value.window_index for value in ordered] != list(
                range(first, first + 6)
            ):
                continue
            if any(
                left.ended_at != right.started_at
                for left, right in zip(ordered[:-1], ordered[1:], strict=True)
            ):
                continue
            if (fish_id, snapshot_id(fish_id, capture, first, 60)) not in self.values:
                pending.append(ordered[-1])
        return tuple(sorted(pending, key=lambda value: value.ended_at)[:limit])


class Summarizer:
    def __init__(self):
        self.calls = []
        self.merges = []
        self.five_minute_merges = []
        self.half_hour_merges = []
        self.merged = asyncio.Event()
        self.block = None
        self.fail = set()

    async def summarize_window(self, window, transcript, **kwargs):
        self.calls.append((window.window_index, transcript))
        if self.block:
            await self.block.wait()
        if window.window_index in self.fail:
            raise ValueError("synthetic failure")
        return SemanticDocument(summary=f"window {window.window_index}")

    async def merge_snapshots(
        self, windows, *, parent_period_seconds, previous_document=None
    ):
        # Existing lifecycle fixtures override these helpers to block/fail a
        # particular merge; the production service uses one generic interface.
        merge = {
            60: self.merge_minute,
            300: self.merge_five_minutes,
            1800: self.merge_thirty_minutes,
        }[parent_period_seconds]
        return await merge(
            windows,
            previous_summary=(
                previous_document.summary if previous_document is not None else None
            ),
        )

    async def merge_minute(self, windows, *, previous_summary=None):
        self.merges.append([w.window_index for w in windows])
        self.merged.set()
        return SemanticDocument(summary="minute")

    async def merge_five_minutes(self, windows, *, previous_summary=None):
        self.five_minute_merges.append([w.window_index for w in windows])
        return SemanticDocument(summary="five minutes")

    async def merge_thirty_minutes(self, windows, *, previous_summary=None):
        self.half_hour_merges.append([w.window_index for w in windows])
        return SemanticDocument(summary="half hour")

    async def close(self):
        pass


class Transcriber:
    async def transcribe(self, pcm):
        return SystemAudioTranscript("local transcript" if pcm else "", "unknown")

    async def close(self):
        pass


class StatusFailure(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"synthetic status {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("event", ["BGM", "Music"])
def test_system_audio_music_contract_preserves_plain_text_and_empty_windows(event):
    assert format_system_audio_text("歌词或对白", event) == "歌词或对白"
    assert format_system_audio_text("", event) == ""


def test_system_audio_non_music_is_plain_text_and_transcript_is_bounded():
    assert format_system_audio_text("普通对白", "Speech") == "普通对白"
    assert _transcript("字" * 1001, "<|Speech|>") == SystemAudioTranscript(
        "字" * 1000, "Speech"
    )


def test_system_audio_prefers_music_when_sensevoice_returns_multiple_events():
    assert _transcript("带配乐的对白", "<|Speech|><|BGM|>") == (
        SystemAudioTranscript("带配乐的对白", "BGM")
    )


def window(index=0, capture=CAPTURE):
    return EnvironmentHistoryWindow(
        capture_session_id=capture,
        window_index=index,
        started_at=START.timestamp() + index * 10,
        ended_at=START.timestamp() + (index + 1) * 10,
        frames=(_request(observed_at=START.timestamp() + index * 10 + 1),),
        audio_status="unavailable",
        audio_captured_samples=0,
        audio_gap_reason="os_unsupported",
    )


def audible_window(index=0, capture=CAPTURE):
    return window(index, capture).model_copy(
        update={
            "audio_status": "complete",
            "audio_pcm_base64": base64.b64encode(b"\x01\x00" * 160_000).decode(),
            "audio_captured_samples": 160_000,
            "audio_gap_reason": None,
        }
    )


def stored_window(index=0, capture=CAPTURE):
    return SemanticSnapshot(
        id=snapshot_id(FISH, capture, index, 10),
        fish_id=FISH,
        capture_session_id=capture,
        window_index=index,
        period_seconds=10,
        started_at=START + timedelta(seconds=index * 10),
        ended_at=START + timedelta(seconds=(index + 1) * 10),
        document=SemanticDocument(summary=f"stored window {index}"),
        input_digest=window(index, capture).input_digest(),
        created_at=START,
    )


def setup_service(*, repository=None, summarizer=None, transcriber=None, **kwargs):
    repository = repository or Repository()
    summarizer = summarizer or Summarizer()
    service = EnvironmentHistoryService(
        repository,
        summarizer,
        transcriber or Transcriber(),
        now=kwargs.pop("now", lambda: START + timedelta(minutes=5)),
        retry_delays=kwargs.pop("retry_delays", ()),
        **kwargs,
    )
    service.set_enabled(FISH, True)
    return service, repository, summarizer


async def settle(service):
    for _ in range(100):
        tasks = [task for _, task in service._jobs.values()]
        if not tasks:
            return
        await asyncio.gather(*tasks)
    raise AssertionError("history work did not settle")


@pytest.mark.asyncio
async def test_complete_music_bridge_is_saved_without_periodic_aggregation():
    from tests.observations.test_environment_history_client import EventModel, _client

    class MusicBridge(Transcriber):
        async def transcribe_window(self, pcm, *, window_index, **kwargs):
            del pcm, kwargs
            if window_index in (1, 2, 3, 4):
                return SystemAudioTranscript("", "Music")
            return SystemAudioTranscript(
                "我明天来。" if window_index == 0 else "临时有事，来不了了。", "Speech"
            )

    clock = [START]
    service, repository, _ = setup_service(
        summarizer=_client(EventModel()),
        transcriber=MusicBridge(),
        now=lambda: clock[0],
    )
    try:
        for index in range(6):
            clock[0] = START + timedelta(seconds=(index + 1) * 10)
            await service.enqueue(
                FISH, audible_window(index).model_copy(update={"frames": ()})
            )
            await settle(service)
        assert service.status(FISH)["completedWindows"] == 6
        assert service.status(FISH)["failedWindows"] == 0
        parent = await repository.get(FISH, snapshot_id(FISH, CAPTURE, 0, 60))
        assert parent is None
        key = (FISH, snapshot_id(FISH, CAPTURE, 2, 10))
        assert not repository.values[key].document.summary
        assert (
            repository.evidence[key]["transcriptionTextStatus"] == "no_recognized_words"
        )
        assert any(
            "来不了了" in item.document.audio for item in repository.values.values()
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_window_without_media_does_not_call_model_or_persist_fake_history():
    service, repository, model = setup_service()
    try:
        empty = window().model_copy(update={"frames": ()})
        assert await service.enqueue(FISH, empty) == "skipped"
        await settle(service)
        assert model.calls == []
        assert not repository.values
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_completed_windows_and_restart_never_schedule_periodic_summaries():
    service, repository, model = setup_service()
    try:
        for i in [3, 1, 5, 0, 4, 2]:
            await service.enqueue(FISH, window(i))
        await settle(service)
        assert model.merges == []
        assert len(repository.values) == 6
        assert await service.enqueue(FISH, window(5)) == "completed"
        await settle(service)
        assert len(model.calls) == 6
        service.set_enabled(FISH, False)
        service.set_enabled(FISH, True)
        await settle(service)
        assert model.merges == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_snapshot_context_range_uses_seconds_and_one_line_per_summary():
    repository = Repository()
    value = replace(
        stored_window(),
        started_at=START + timedelta(microseconds=687000),
        ended_at=START + timedelta(seconds=10, microseconds=687000),
        document=SemanticDocument(summary="first line\nsecond line"),
    )
    repository.values[(FISH, value.id)] = value
    service, _, _ = setup_service(repository=repository)
    try:
        await service._recovery_tasks[FISH]
        context = await service.get_latest_context_summary(FISH)

        assert context is not None
        expected_range = expected_local_range(
            START, START + timedelta(seconds=10), date=True
        )
        assert f"你之前看到和听到的事（{expected_range}）" in context.summary
        assert "- first line\n  second line" in context.summary
        assert ".687000" not in context.summary
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_ask_context_uses_snapshot_summaries_and_keeps_recent_raw_asr():
    class RichSummarizer(Summarizer):
        async def summarize_window(self, value, transcript, **kwargs):
            self.calls.append((value.window_index, transcript))
            index = value.window_index
            return SemanticDocument(
                summary=f"window-{index}-summary-public",
                audio=f"window-{index}-audio-detail-only",
            )

        async def merge_minute(self, windows, *, previous_summary=None):
            self.merges.append([value.window_index for value in windows])
            self.merged.set()
            return SemanticDocument(
                summary="minute-summary-public",
                audio="minute-audio-detail-only",
            )

    class IndexedTranscriber(Transcriber):
        def __init__(self):
            self.index = 0

        async def transcribe(self, pcm):
            assert len(pcm) == 320_000
            transcript = SystemAudioTranscript(f"raw-system-asr-{self.index}", "Speech")
            self.index += 1
            return transcript

    current_time = START

    def clock():
        return current_time

    service, _, _ = setup_service(
        summarizer=RichSummarizer(),
        transcriber=IndexedTranscriber(),
        now=clock,
    )
    try:
        for index in range(7):
            current_time = START + timedelta(seconds=(index + 1) * 10 + 1)
            await service.enqueue(FISH, audible_window(index))
            await settle(service)

        context = await service.get_latest_context_summary(FISH)

        assert context is not None
        assert "minute-summary-public" not in context.summary
        assert all(
            f"window-{index}-summary-public" in context.summary for index in range(7)
        )
        assert "minute-audio-detail-only" not in context.summary
        assert "window-6-audio-detail-only" not in context.summary
        assert "- raw-system-asr-6" in context.summary
        assert "recent_system_playback_transcripts" not in context.summary
        assert "startedAt" not in context.summary
        assert "endedAt" not in context.summary
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_incomplete_minute_keeps_completed_ten_second_snapshots_available():
    service, repository, model = setup_service()
    model.fail.add(2)
    try:
        for i in range(6):
            await service.enqueue(FISH, window(i))
        await settle(service)
        assert not model.merges
        assert len(repository.values) == 5
        assert service.status(FISH)["failedWindows"] == 1
        context = await service.get_latest_context_summary(FISH)
        assert (
            context
            and "window 5" in context.summary
            and "window 2" not in context.summary
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_transient_window_failure_retries_then_persists():
    class FlakySummarizer(Summarizer):
        attempts = 0

        async def summarize_window(self, window, transcript, **kwargs):
            self.calls.append((window.window_index, transcript))
            self.attempts += 1
            if self.attempts == 1:
                raise StatusFailure(503)
            return SemanticDocument(summary="recovered window")

    model = FlakySummarizer()
    service, repository, _ = setup_service(summarizer=model, retry_delays=(0,))
    try:
        await service.enqueue(FISH, window())
        await settle(service)
        assert model.attempts == 2
        assert len(repository.values) == 1
        assert service.status(FISH)["completedWindows"] == 1
        assert service.status(FISH)["failedWindows"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capability", "frames"), [("unsupported", 0), ("supported", 1), ("unknown", 1)]
)
async def test_screen_is_left_out_only_for_a_model_that_refuses_images(
    capability, frames
):
    class Recording(Summarizer):
        frames = None

        async def summarize_window(self, window, transcript, **kwargs):
            self.frames = len(window.frames)
            return await super().summarize_window(window, transcript, **kwargs)

    async def image_input():
        return capability

    model = Recording()
    service, _, _ = setup_service(summarizer=model, image_input=image_input)
    try:
        await service.enqueue(FISH, audible_window())
        await settle(service)
        assert model.frames == frames
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["bad_request", "validation"])
async def test_deterministic_window_failure_is_not_retried_and_next_window_runs(
    failure_kind,
):
    if failure_kind == "bad_request":
        failure: Exception = StatusFailure(400)
    else:
        try:
            SemanticDocument.model_validate({"summary": ""})
        except ValidationError as error:
            failure = error

    class FlakySummarizer(Summarizer):
        async def summarize_window(self, window, transcript, **kwargs):
            self.calls.append((window.window_index, transcript))
            if window.window_index == 0:
                raise failure
            return SemanticDocument(summary=f"window {window.window_index}")

    model = FlakySummarizer()
    service, repository, _ = setup_service(summarizer=model, retry_delays=(0, 0))
    try:
        await service.enqueue(FISH, window(0))
        await settle(service)
        await service.enqueue(FISH, window(1))
        await settle(service)

        assert [index for index, _ in model.calls] == [0, 1]
        assert [value.window_index for value in repository.values.values()] == [1]
        assert service.status(FISH)["failedWindows"] == 1
        assert service.status(FISH)["completedWindows"] == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_pending_window_never_blocks_ask_and_queue_is_bounded():
    service, _, model = setup_service(max_pending=1)
    model.block = asyncio.Event()
    try:
        await service.enqueue(FISH, window())
        with pytest.raises(HistoryQueueFull):
            await service.enqueue(FISH, window(1))
        assert (
            await asyncio.wait_for(service.get_latest_context_summary(FISH), 0.1)
            is None
        )
        assert await service.enqueue(FISH, window()) == "pending"
        service.set_enabled(FISH, False)
        assert await service.get_latest_context_summary(FISH) is None
        with pytest.raises(HistoryDisabled):
            await service.enqueue(FISH, window(2))
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spoken",
    [
        "If I could die, would I cry?",
        "普通叙述。" * 100 + "重要结论在最后，必须完整保留。",
    ],
)
async def test_recent_system_audio_is_available_before_flash_snapshot_finishes(spoken):
    class VideoTranscriber(Transcriber):
        def __init__(self):
            self.calls = 0

        async def transcribe(self, pcm):
            assert len(pcm) == 320_000
            self.calls += 1
            return SystemAudioTranscript(spoken, "BGM")

    def clock():
        return START + timedelta(seconds=11)

    transcriber = VideoTranscriber()
    service, repository, model = setup_service(
        transcriber=transcriber,
        now=clock,
    )
    model.block = asyncio.Event()
    try:
        await service.enqueue(FISH, audible_window())
        for _ in range(100):
            if model.calls:
                break
            await asyncio.sleep(0)
        assert model.calls
        assert repository.values == {}

        context = await asyncio.wait_for(service.get_latest_context_summary(FISH), 0.1)

        assert context is not None
        expected_range = expected_local_range(
            START, START + timedelta(seconds=10), date=True
        )
        assert f"你刚才听到的声音（{expected_range}）" in context.summary
        assert context.summary.count(spoken) == 1
        assert "- " + spoken in context.summary
        assert "<Music>" not in context.summary
        assert "ASR" not in context.summary
        assert "本次 Ask" not in context.summary
        assert "不应重复计算" not in context.summary
        assert "…" not in context.summary
        assert "startedAt" not in context.summary
        assert "endedAt" not in context.summary
        assert transcriber.calls == 1

        service.set_enabled(FISH, False)
        assert await service.get_latest_context_summary(FISH) is None
    finally:
        model.block.set()
        await settle(service)
        await service.close()


@pytest.mark.asyncio
async def test_recent_system_audio_context_is_one_minute_ordered_and_bounded():
    class IndexedTranscriber(Transcriber):
        def __init__(self):
            self.index = 0

        async def transcribe(self, pcm):
            text = f"line-{self.index}"
            self.index += 1
            return SystemAudioTranscript(text, "Speech")

    current_time = START

    def clock():
        return current_time

    transcriber = IndexedTranscriber()
    service, repository, model = setup_service(
        transcriber=transcriber,
        now=clock,
    )
    model.fail.update(range(7))
    try:
        for index in range(7):
            current_time = START + timedelta(seconds=(index + 1) * 10 + 1)
            await service.enqueue(FISH, audible_window(index))
            await settle(service)
        assert repository.values == {}

        current_time = START + timedelta(seconds=71)
        context = await service.get_latest_context_summary(FISH)
        assert context is not None
        assert "line-0" not in context.summary
        assert all(f"line-{index}" in context.summary for index in range(1, 7))
        positions = [context.summary.index(f"line-{index}") for index in range(1, 7)]
        assert positions == sorted(positions)
        assert "lookbackSeconds" not in context.summary
        assert context.summary.count("- line-") == 6
        assert context.summary.count("2026-09-03T") == 2
        assert context.through_event_count == 6
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_audio_is_redacted_before_summary_and_not_duplicated_in_evidence():
    transcript = "视频台词 password=super-secret-value"

    class ExactTranscriber(Transcriber):
        async def transcribe(self, pcm):
            return SystemAudioTranscript(transcript, "Speech")

    service, repository, model = setup_service(transcriber=ExactTranscriber())
    try:
        await service.enqueue(FISH, audible_window())
        await settle(service)

        key = snapshot_id(FISH, CAPTURE, 0, 10)
        evidence = repository.evidence[(FISH, key)]
        assert model.calls == [
            (0, SystemAudioTranscript("视频台词 password=[REDACTED]", "Speech"))
        ]
        assert "audioTranscript" not in evidence
        assert "asrEvent" not in evidence
        assert evidence == {
            "frameTimes": [START.timestamp() + 1],
            "audioStatus": "complete",
            "audioCapturedSamples": 160_000,
            "audioGapReason": None,
            "audioSource": "system_playback_local_asr",
            "audioSilence": False,
            "visualReuseFrom": None,
            "diarizationStatus": "unavailable",
            "alignmentStatus": "unavailable",
            "speakerRevisedWindows": [],
            "transcriptionRecoveryStatus": "not_needed",
            "transcriptionTextStatus": "recognized",
            "recoveredSpeechIntervals": [],
            "speakerAttributionRevised": False,
            "audioSegments": [],
        }
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_window_transcription_keeps_scoped_speakers_redacted_before_summary():
    original = SystemAudioTranscript(
        "password=secret-old-text",
        "Speech",
        segments=(
            SystemAudioSegment(
                0.5,
                2.0,
                '奶奶，我的 password="secret-segment"',
                "声音abc123-01",
                local_speaker_id="local:abc123:0:0:0",
                reason="matched",
                clean_seconds=1.35469,
                best_similarity=0.876543,
                second_similarity=0.532198,
                identity_source="embedding",
                track_id="track:abc123:1",
                track_reason="matched",
            ),
            SystemAudioSegment(2.5, 5.0, "先把收据给我。", "声音abc123-02"),
            SystemAudioSegment(6.0, 7.0, "等等。", None, overlap=True),
        ),
        diarization_status="partial",
    )

    class SegmentTranscriber(Transcriber):
        def __init__(self):
            self.calls = []

        async def transcribe(self, pcm):
            raise AssertionError("the window API must preserve the capture identity")

        async def transcribe_window(self, pcm, **kwargs):
            assert len(pcm) == 320_000
            self.calls.append(kwargs)
            return original

    transcriber = SegmentTranscriber()
    service, repository, model = setup_service(
        transcriber=transcriber,
        now=lambda: START + timedelta(seconds=11),
    )
    model.block = asyncio.Event()
    try:
        await service.enqueue(FISH, audible_window())
        for _ in range(100):
            if model.calls:
                break
            await asyncio.sleep(0)
        assert model.calls
        assert transcriber.calls == [
            {
                "stream_id": f"{FISH}:{CAPTURE}",
                "window_index": 0,
                "started_at": START.timestamp(),
                "audio_complete": True,
            }
        ]
        redacted = model.calls[0][1]
        assert redacted.event == "Speech"
        assert redacted.diarization_status == "partial"
        assert redacted.text == "password=[REDACTED]"
        assert redacted.segments[0] == replace(
            original.segments[0], text="奶奶，我的 password=[REDACTED]"
        )
        assert redacted.segments[1:] == original.segments[1:]
        assert "secret-segment" in original.segments[0].text

        # Ask can use the timed, redacted dialogue while Flash is still working.
        context = await service.get_latest_context_summary(FISH)
        assert context is not None
        assert (
            expected_local_range(START, START + timedelta(seconds=10))
            in context.summary
        )
        assert (
            "[00:00.50—00:02.00] 声音abc123-T01：奶奶，我的 password=[REDACTED]"
            in context.summary
        )
        assert "多人同时说话（归属未确定）：等等。" in context.summary
        assert "secret-" not in context.summary
        assert str(CAPTURE) not in context.summary
        assert repository.values == {}

        model.block.set()
        await settle(service)
        evidence = repository.evidence[(FISH, snapshot_id(FISH, CAPTURE, 0, 10))]
        assert evidence["diarizationStatus"] == "partial"
        core_keys = {"startSeconds", "endSeconds", "speakerId", "overlap"}
        assert [
            {key: value for key, value in segment.items() if key in core_keys}
            for segment in evidence["audioSegments"]
        ] == [
            {
                "startSeconds": 0.5,
                "endSeconds": 2.0,
                "speakerId": "声音abc123-01",
                "overlap": False,
            },
            {
                "startSeconds": 2.5,
                "endSeconds": 5.0,
                "speakerId": "声音abc123-02",
                "overlap": False,
            },
            {
                "startSeconds": 6.0,
                "endSeconds": 7.0,
                "speakerId": None,
                "overlap": True,
            },
        ]
        assert evidence["audioSegments"][0] == {
            "startSeconds": 0.5,
            "endSeconds": 2.0,
            "speakerId": "声音abc123-01",
            "overlap": False,
            "localSpeakerId": "local:abc123:0:0:0",
            "reason": "matched",
            "cleanSeconds": 1.355,
            "bestSimilarity": 0.8765,
            "secondSimilarity": 0.5322,
            "identitySource": "embedding",
            "boundaryReason": None,
            "trackId": "track:abc123:1",
            "trackReason": "matched",
            "alignmentReason": "legacy_interval",
        }
        assert "0.8765" not in context.summary
        assert "local:abc123" not in context.summary
        assert "secret-" not in repr(evidence)
        assert "奶奶" not in repr(evidence)
        assert "audioPcmBase64" not in evidence
    finally:
        model.block.set()
        await settle(service)
        await service.close()


@pytest.mark.asyncio
async def test_recent_dialogue_keeps_window_times_and_distinct_speaker_namespaces():
    class SegmentTranscriber(Transcriber):
        def __init__(self):
            self.calls = []

        async def transcribe_window(
            self, pcm, *, stream_id, window_index, started_at, audio_complete
        ):
            assert audio_complete
            self.calls.append((stream_id, window_index, started_at))
            speaker_id = "声音abc123-01" if window_index < 2 else "声音def456-01"
            return SystemAudioTranscript(
                "",
                "Speech",
                segments=(
                    SystemAudioSegment(
                        0.0, 3.0, f"第{window_index}段对白。", speaker_id
                    ),
                ),
                diarization_status="ok",
            )

    transcriber = SegmentTranscriber()
    service, _, model = setup_service(
        transcriber=transcriber, now=lambda: START + timedelta(seconds=31)
    )
    model.fail.update(range(3))
    second_capture = UUID(int=3)
    try:
        for index in range(3):
            current = audible_window(index)
            if index == 2:
                current = current.model_copy(
                    update={"capture_session_id": second_capture}
                )
            await service.enqueue(FISH, current)
            await settle(service)

        assert [call[0] for call in transcriber.calls] == [
            f"{FISH}:{CAPTURE}",
            f"{FISH}:{CAPTURE}",
            f"{FISH}:{second_capture}",
        ]
        context = await service.get_latest_context_summary(FISH)
        assert context is not None
        assert context.summary.count("声音abc123-01") == 2
        assert context.summary.count("声音def456-01") == 1
        assert all(
            expected_local_range(
                START + timedelta(seconds=index * 10),
                START + timedelta(seconds=(index + 1) * 10),
            )
            in context.summary
            for index in range(3)
        )
        assert [
            context.summary.index(f"第{index}段对白") for index in range(3)
        ] == sorted(context.summary.index(f"第{index}段对白") for index in range(3))
        assert str(CAPTURE) not in context.summary
        assert str(second_capture) not in context.summary
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_identical_ids_with_changed_input_are_rejected():
    service, _, _ = setup_service()
    try:
        await service.enqueue(FISH, window())
        await settle(service)
        with pytest.raises(EnvironmentHistoryConflict):
            await service.enqueue(
                FISH, window().model_copy(update={"audio_gap_reason": "no_samples"})
            )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_minute_never_combines_capture_sessions_or_discontinuous_windows():
    service, repository, model = setup_service()
    try:
        for i in range(6):
            item = window(i, CAPTURE if i < 3 else UUID(int=3))
            await service.enqueue(FISH, item)
        await settle(service)
        assert not model.merges
        values = list(repository.values.values())
        assert all(v.period_seconds == 10 for v in values)
    finally:
        await service.close()


def test_pcm_and_timestamps_are_checked_without_accepting_fake_complete_audio():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EnvironmentHistoryWindow.model_validate(
            {**window().model_dump(), "audio_status": "complete"}
        )
    with pytest.raises(ValidationError):
        EnvironmentHistoryWindow.model_validate(
            {**window().model_dump(), "ended_at": START.timestamp() + 11}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reenable", [False, True])
async def test_enqueue_cannot_cross_disable_while_lookup_is_pending(reenable):
    entered, release = asyncio.Event(), asyncio.Event()

    class PausedRepository(Repository):
        async def get(self, fish_id, snapshot_id):
            entered.set()
            await release.wait()
            return await super().get(fish_id, snapshot_id)

    service, _, model = setup_service(repository=PausedRepository())
    pending = asyncio.create_task(service.enqueue(FISH, window()))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        service.set_enabled(FISH, False)
        if reenable:
            service.set_enabled(FISH, True)
        release.set()
        with pytest.raises(HistoryDisabled):
            await pending
        assert service.status(FISH)["pendingWindows"] == 0
        assert model.calls == []
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await service.close()


@pytest.mark.asyncio
async def test_disable_cleans_a_window_task_cancelled_before_it_starts():
    service, repository, model = setup_service()
    try:
        await service.enqueue(FISH, window())
        task = next(iter(service._jobs.values()))[1]
        service.set_enabled(FISH, False)
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        assert service._jobs == {}
        assert service._tasks == set()
        assert service.status(FISH)["pendingWindows"] == 0
        assert repository.values == {}
        assert model.calls == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_disabling_last_fish_releases_transcriber_after_jobs_cancel():
    class ReleasableTranscriber(Transcriber):
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.released = asyncio.Event()
            self.closed = False

        async def transcribe(self, pcm):
            del pcm
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

        async def release(self):
            assert self.cancelled.is_set()
            self.released.set()

        async def close(self):
            self.closed = True

    transcriber = ReleasableTranscriber()
    service, _, _ = setup_service(transcriber=transcriber)
    try:
        await service.enqueue(FISH, window())
        await asyncio.wait_for(transcriber.started.wait(), 1)
        service.set_enabled(FISH, False)
        await asyncio.wait_for(transcriber.released.wait(), 1)
        assert transcriber.cancelled.is_set()
    finally:
        await service.close()
    assert transcriber.closed


@pytest.mark.asyncio
async def test_immediate_reenable_cancels_decoder_release():
    class ReleasableTranscriber(Transcriber):
        def __init__(self):
            self.release_count = 0
            self.released = asyncio.Event()

        async def release(self):
            self.release_count += 1
            self.released.set()

    transcriber = ReleasableTranscriber()
    service, _, _ = setup_service(transcriber=transcriber)
    try:
        service.set_enabled(FISH, False)
        service.set_enabled(FISH, True)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert transcriber.release_count == 0

        service.set_enabled(FISH, False)
        await asyncio.wait_for(transcriber.released.wait(), 1)
        assert transcriber.release_count == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_switching_active_fish_retires_old_hour_state_and_releases_when_off():
    class ReleasableTranscriber(Transcriber):
        def __init__(self):
            self.released = asyncio.Event()

        async def release(self):
            self.released.set()

    other_fish = UUID(int=3)
    transcriber = ReleasableTranscriber()
    service, _, _ = setup_service(transcriber=transcriber)
    try:
        await service._recovery_tasks[FISH]
        service._remember_snapshot(stored_window(0))

        service.set_enabled(other_fish, True)

        assert service.status(FISH)["enabled"] is False
        assert service.status(other_fish)["enabled"] is True
        assert FISH not in service._recent_snapshots
        assert FISH not in service._context_views
        await service._recovery_tasks[other_fish]

        service.set_enabled(other_fish, False)
        await asyncio.wait_for(transcriber.released.wait(), 1)
        assert service.status(other_fish)["enabled"] is False
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_close_cleans_unstarted_jobs_and_cannot_reopen():
    service, _, model = setup_service()
    await service.enqueue(FISH, window())
    await service.close()
    assert service._jobs == {}
    assert service._last_windows == {}
    assert service._recovery_tasks == {}
    assert service._tasks == set()
    assert model.calls == model.merges == []
    service.set_enabled(FISH, True)
    with pytest.raises(HistoryDisabled):
        await service.enqueue(FISH, window(1))


class LateResultGate:
    def __init__(self):
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self):
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()


@pytest.mark.asyncio
async def test_late_transcription_cannot_start_a_model_after_disable_and_reenable():
    gate = LateResultGate()

    class LateTranscriber(Transcriber):
        async def transcribe(self, pcm):
            await gate.wait()
            return await super().transcribe(pcm)

    service, repository, model = setup_service(transcriber=LateTranscriber())
    try:
        await service.enqueue(FISH, window())
        await asyncio.wait_for(gate.entered.wait(), 1)
        task = next(iter(service._jobs.values()))[1]
        service.set_enabled(FISH, False)
        service.set_enabled(FISH, True)
        await asyncio.wait_for(gate.cancelled.wait(), 1)
        gate.release.set()
        await task
        assert model.calls == []
        assert repository.values == {}
        assert service.status(FISH)["failedWindows"] == 0
    finally:
        gate.release.set()
        await service.close()


@pytest.mark.asyncio
async def test_retired_window_done_callback_cannot_remove_replacement_job():
    gate = LateResultGate()
    second_entered, second_release = asyncio.Event(), asyncio.Event()

    class LateSummarizer(Summarizer):
        attempts = 0

        async def summarize_window(self, window, transcript, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                await gate.wait()
            else:
                second_entered.set()
                await second_release.wait()
            return await super().summarize_window(window, transcript)

    service, repository, _ = setup_service(summarizer=LateSummarizer())
    try:
        await service.enqueue(FISH, window())
        await asyncio.wait_for(gate.entered.wait(), 1)
        retired = next(iter(service._jobs.values()))[1]
        service.set_enabled(FISH, False)
        service.set_enabled(FISH, True)
        await asyncio.wait_for(gate.cancelled.wait(), 1)
        await service.enqueue(FISH, window())
        await asyncio.wait_for(second_entered.wait(), 1)
        replacement = next(iter(service._jobs.values()))[1]
        gate.release.set()
        await retired
        assert list(service._jobs.values())[0][1] is replacement
        assert service.status(FISH)["pendingWindows"] == 1
        assert repository.values == {}
        second_release.set()
        await settle(service)
        assert len(repository.values) == 1
    finally:
        gate.release.set()
        second_release.set()
        await service.close()


@pytest.mark.asyncio
async def test_enqueue_freezes_window_before_background_work_starts():
    service, repository, _ = setup_service()
    original = window()
    digest = original.input_digest()
    try:
        await service.enqueue(FISH, original)
        original.audio_gap_reason = "changed_after_enqueue"
        pending = next(iter(service._jobs.values()))[0]
        assert pending.window.audio_gap_reason == "os_unsupported"
        await settle(service)
        assert next(iter(repository.values.values())).input_digest == digest
    finally:
        await service.close()
