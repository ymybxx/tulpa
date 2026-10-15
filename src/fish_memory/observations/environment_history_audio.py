"""Local system-audio transcription, isolated from microphone turn decoding."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from fish_memory.companion.asr import (
    AsrModelConfig,
    AsrModelDownloader,
    StreamingAsr,
)
from fish_memory.companion.shared_asr import (
    SharedAsrClient,
    SharedAsrResponse,
    _set_utility_qos,
    configured_shared_asr_socket,
)
from fish_memory.observations.environment_history_alignment import align_transcript
from fish_memory.observations.environment_history_recovery import (
    RecoveredTranscript,
    sparse_speech_region,
    splice_recovered_speech,
)
from fish_memory.observations.environment_history_rolling_speakers import (
    RollingSpeakerDiarizer as LocalSpeakerDiarizer,
)
from fish_memory.observations.environment_history_speakers import (
    SpeakerSpan,
    SpeakerWindowOrderError,
)

logger = logging.getLogger(__name__)

# ScreenCaptureKit can produce low-level integer dither while no system audio is
# playing. Sending ten seconds of that noise through SenseVoice every window is
# expensive and cannot yield useful speech. The threshold is deliberately near
# digital zero (roughly -78 dBFS), so quiet dialogue still reaches the recognizer
# while zero-fill and tiny quantisation noise do not.
_SILENCE_PEAK_PCM16 = 4
_MAX_TRANSCRIPT_CHARS = 1000
_SAMPLE_RATE = 16_000
_MAX_SPEECH_SEGMENTS = 16
_SPEAKER_RETRY_SECONDS = 60.0
_SHARED_RECONNECT_SECONDS = 5.0
_DECODER_TAG = re.compile(r"<\|[^<>\r\n]{1,64}\|>")
_MUSIC_TAG = re.compile(r"</?Music>", re.IGNORECASE)
_TRACK_ID = re.compile(r"track:([a-zA-Z0-9_-]{1,32}):([1-9][0-9]{0,8})")
_TIMED_SPEECH_LINE = re.compile(
    r"^\s*\[\d{2,}:\d{2}\.\d{2}[—–-]\d{2,}:\d{2}\.\d{2}\]"
    r"\s*[^：\n]{1,160}：(?P<body>.*)$"
)
_AUDIO_WINDOW_HEADER = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s*[—–-]\s*"
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?: UTC)?\s*$"
)


@dataclass(frozen=True, slots=True)
class SystemAudioSegment:
    """Text associated conservatively with token points and a speaker interval."""

    start_seconds: float
    end_seconds: float
    text: str
    speaker_id: str | None
    overlap: bool = False
    local_speaker_id: str | None = None
    reason: str = "unavailable"
    clean_seconds: float = 0.0
    best_similarity: float | None = None
    second_similarity: float | None = None
    identity_source: str = "unknown"
    boundary_reason: str | None = None
    track_id: str | None = None
    track_reason: str | None = None
    alignment_reason: str = "legacy_interval"


@dataclass(frozen=True, slots=True)
class SystemAudioRevision:
    window_index: int
    started_at: float
    text: str
    segments: tuple[SystemAudioSegment, ...]
    event: str


@dataclass(frozen=True, slots=True)
class SystemAudioTranscript:
    text: str
    event: str
    segments: tuple[SystemAudioSegment, ...] = ()
    diarization_status: str = "unavailable"
    tokens: tuple[str, ...] = ()
    timestamps: tuple[float, ...] = ()
    alignment_status: str = "unavailable"
    revisions: tuple[SystemAudioRevision, ...] = ()
    recovery_status: str = "not_needed"
    recovered_intervals: tuple[tuple[float, float], ...] = ()
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _AlignedWindow:
    started_at: float
    duration: float
    transcript: SystemAudioTranscript
    spans: tuple[SpeakerSpan, ...]


class LocalSystemAudioTranscriber:
    def __init__(self, settings: object) -> None:
        self._config = replace(AsrModelConfig.from_settings(settings), num_threads=1)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="environment-asr",
            initializer=_set_utility_qos,
        )
        self._settings = settings
        self._speaker_enabled = bool(
            getattr(settings, "environment_history_speaker_enabled", True)
        )
        self._speaker_diarizer: LocalSpeakerDiarizer | None = None
        self._last_speaker_window: tuple[str, int, float] | None = None
        self._speaker_retry_after = 0.0
        self._alignment_stream: str | None = None
        self._alignment_history: OrderedDict[int, _AlignedWindow] = OrderedDict()
        # One window owns its full transcription and independent segmentation. Model
        # summaries remain concurrent in the service after this lock is freed.
        self._window_lock = asyncio.Lock()
        self._last_speaker_duration = 10.0
        self._decoder: StreamingAsr | None = None
        self._state_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._last_shared_diagnostic: str | None = None
        shared_socket = configured_shared_asr_socket()
        self._shared_client = (
            SharedAsrClient(shared_socket) if shared_socket is not None else None
        )

    async def transcribe_window(
        self,
        pcm: bytes,
        *,
        stream_id: str,
        window_index: int,
        started_at: float,
        audio_complete: bool = True,
    ) -> SystemAudioTranscript:
        if not math.isfinite(started_at):
            raise ValueError("system audio timestamp must be finite")
        if type(pcm) is not bytes or len(pcm) % 2 or len(pcm) > 320_000:
            raise ValueError("system audio must be at most ten seconds of PCM16")
        async with self._window_lock:
            self._check_open()
            if _effectively_silent_pcm16(pcm):
                if self._speaker_diarizer is not None:
                    await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self._observe_silence,
                        stream_id,
                        window_index,
                        started_at,
                        audio_complete,
                        len(pcm) / 32000,
                    )
                result = self._align_window(
                    SystemAudioTranscript("", "silence"),
                    (),
                    len(pcm) / 32000,
                    stream_id,
                    window_index,
                    started_at,
                )
                return replace(result, diarization_status="silence")
            # Preserve the complete decode. Speaker attribution does not cut
            # this into tiny recognition jobs; a sparse whole speech passage
            # may receive one bounded recovery attempt below.
            transcript = await self.transcribe(pcm)
            if transcript.event == "unavailable":
                return transcript
            if not self._speaker_enabled:
                return replace(transcript, diarization_status="disabled")
            if time.monotonic() < self._speaker_retry_after:
                return replace(transcript, diarization_status="failed")
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("system audio transcriber is closed")
                self._generation += 1
                future = self._executor.submit(
                    self._segment,
                    pcm,
                    stream_id,
                    window_index,
                    started_at,
                    audio_complete,
                )
            try:
                spans = await asyncio.wrap_future(future)
                _validate_spans(spans, len(pcm) // 2)
            except asyncio.CancelledError:
                raise
            except SpeakerWindowOrderError:
                return replace(transcript, diarization_status="unassigned")
            except Exception as error:
                self._speaker_retry_after = time.monotonic() + _SPEAKER_RETRY_SECONDS
                logger.warning(
                    "影音说话人分段暂不可用，保留无人物标签的转录: type=%s",
                    type(error).__name__,
                )
                return replace(transcript, diarization_status="failed")
            self._check_open()
            if len(spans) > 128:
                spans = ()
            if audio_complete:
                transcript = await self._recover_sparse_speech(pcm, transcript, spans)
            return self._align_window(
                transcript,
                spans,
                len(pcm) / (_SAMPLE_RATE * 2),
                stream_id,
                window_index,
                started_at,
            )

    async def _recover_sparse_speech(
        self,
        pcm: bytes,
        transcript: SystemAudioTranscript,
        spans: tuple[SpeakerSpan, ...],
    ) -> SystemAudioTranscript:
        region = sparse_speech_region(
            transcript.text,
            transcript.tokens,
            transcript.timestamps,
            spans,
            len(pcm) / (_SAMPLE_RATE * 2),
        )
        if region is None:
            return transcript
        first = round(region.start * _SAMPLE_RATE)
        last = round(region.end * _SAMPLE_RATE)
        region = replace(region, start=first / _SAMPLE_RATE, end=last / _SAMPLE_RATE)
        try:
            # The shared recognizer keeps foreground voice priority. A busy or
            # failed recovery leaves the successful primary transcript intact.
            candidate = await self.transcribe(pcm[first * 2 : last * 2])
        except asyncio.CancelledError:
            raise
        except Exception:
            self._check_open()
            return replace(transcript, recovery_status="unavailable")
        self._check_open()
        if candidate.event == "unavailable":
            return replace(transcript, recovery_status="unavailable")
        recovered = splice_recovered_speech(
            RecoveredTranscript(
                transcript.text, transcript.tokens, transcript.timestamps
            ),
            RecoveredTranscript(candidate.text, candidate.tokens, candidate.timestamps),
            region,
        )
        if recovered is None or len(recovered.text) > _MAX_TRANSCRIPT_CHARS:
            return replace(transcript, recovery_status="rejected")
        return replace(
            transcript,
            text=recovered.text,
            tokens=recovered.tokens,
            timestamps=recovered.timestamps,
            recovery_status="applied",
            recovered_intervals=((region.start, region.end),),
        )

    def _observe_silence(
        self,
        stream_id: str,
        window_index: int,
        started_at: float,
        audio_complete: bool,
        duration: float = 10.0,
    ) -> None:
        self._check_open()
        diarizer = self._speaker_diarizer
        if diarizer is not None:
            previous = self._last_speaker_window
            if (
                previous is not None
                and previous[0] == stream_id
                and window_index > previous[1]
                and not 0 < started_at - previous[2] <= 70
            ):
                diarizer.reset()
                self._alignment_history.clear()
            if audio_complete:
                timed_silence = getattr(diarizer, "observe_silence_at", None)
                if callable(timed_silence):
                    timed_silence(
                        stream_id=stream_id,
                        window_index=window_index,
                        started_at=started_at,
                        duration=duration,
                    )
                else:
                    diarizer.observe_silence(
                        stream_id=stream_id, window_index=window_index
                    )
                self._last_speaker_duration = duration
                self._last_speaker_window = (stream_id, window_index, started_at)
            else:
                diarizer.clear_boundary()

    def _check_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("system audio transcriber is closed")

    def _segment(
        self,
        pcm: bytes,
        stream_id: str,
        window_index: int,
        started_at: float,
        audio_complete: bool,
    ) -> tuple[SpeakerSpan, ...]:
        self._check_open()
        if self._speaker_diarizer is None:
            self._speaker_diarizer = LocalSpeakerDiarizer(self._settings)
        previous = self._last_speaker_window
        boundary_gap = (
            previous is not None
            and previous[0] == stream_id
            and window_index > previous[1]
            and (
                window_index != previous[1] + 1
                or abs(started_at - previous[2] - self._last_speaker_duration) >= 0.05
            )
        )
        if not audio_complete or boundary_gap:
            self._speaker_diarizer.clear_boundary()
        if (
            previous is not None
            and previous[0] == stream_id
            and window_index > previous[1]
            and not 0 < started_at - previous[2] <= 70
        ):
            # Some capture clients resume their index after a wall-clock gap.
            # Treat a long pause/time rewind like a new acoustic scene even
            # when the client did not rotate its capture identity.
            self._speaker_diarizer.reset()
        timed_segment = getattr(self._speaker_diarizer, "segment_at", None)
        spans = (
            timed_segment(
                pcm,
                stream_id=stream_id,
                window_index=window_index,
                started_at=started_at,
            )
            if callable(timed_segment)
            else self._speaker_diarizer.segment(
                pcm, stream_id=stream_id, window_index=window_index
            )
        )
        self._last_speaker_duration = len(pcm) / 32000
        if not audio_complete:
            self._speaker_diarizer.clear_boundary()
        self._last_speaker_window = (stream_id, window_index, started_at)
        return spans

    @staticmethod
    def _aligned_transcript(
        transcript: SystemAudioTranscript,
        spans: tuple[SpeakerSpan, ...],
        duration: float,
    ) -> SystemAudioTranscript:
        alignment = align_transcript(
            transcript.text, transcript.tokens, transcript.timestamps, spans, duration
        )
        if not alignment.parts:
            return replace(
                transcript,
                segments=(),
                diarization_status="unassigned",
                alignment_status=alignment.status,
            )
        segments = []
        for part in alignment.parts:
            span = spans[part.span_index] if part.span_index is not None else None
            segments.append(
                SystemAudioSegment(
                    part.start_seconds,
                    part.end_seconds,
                    part.text,
                    span.speaker_id if span else None,
                    overlap=part.overlap,
                    local_speaker_id=span.local_speaker_id if span else None,
                    reason=span.reason if span else part.reason,
                    clean_seconds=span.clean_seconds if span else 0.0,
                    best_similarity=span.best_similarity if span else None,
                    second_similarity=span.second_similarity if span else None,
                    identity_source=span.identity_source if span else "unknown",
                    boundary_reason=span.boundary_reason if span else None,
                    track_id=span.track_id if span else None,
                    track_reason=span.track_reason if span else None,
                    alignment_reason=part.reason,
                )
            )
        return replace(
            transcript,
            segments=tuple(segments),
            diarization_status="ok",
            alignment_status=alignment.status,
        )

    def _align_window(
        self,
        transcript: SystemAudioTranscript,
        spans: tuple[SpeakerSpan, ...],
        duration: float,
        stream_id: str,
        window_index: int,
        started_at: float,
    ) -> SystemAudioTranscript:
        if self._alignment_stream != stream_id:
            self._alignment_history.clear()
            self._alignment_stream = stream_id
        # Keep token evidence and independent embeddings' source intervals only;
        # there is no replay of old PCM or a second transcription of a window.
        for index, value in tuple(self._alignment_history.items()):
            if index >= window_index or not 0 < started_at - value.started_at <= 300:
                self._alignment_history.pop(index, None)
        revisions = []
        diarizer = self._speaker_diarizer
        changes = getattr(diarizer, "revisions", ())
        windows = {
            revision.window_index: revision.spans
            for revision in getattr(diarizer, "window_revisions", ())
        }
        aliases = dict(getattr(diarizer, "track_aliases", ()))
        for index, value in tuple(self._alignment_history.items()):
            changed = False
            revised_spans = []
            for span in value.spans:
                update = next(
                    (
                        change
                        for change in changes
                        if change.window_index == index
                        and change.start_sample == span.start_sample
                        and change.end_sample == span.end_sample
                    ),
                    None,
                )
                if update is not None and not span.overlap:
                    span = replace(
                        span,
                        speaker_id=update.speaker_id,
                        track_id=update.track_id,
                        reason=update.reason,
                        track_reason=update.reason,
                        identity_source="retrospective",
                    )
                    changed = True
                elif span.track_id in aliases and not span.overlap:
                    span = replace(
                        span,
                        track_id=aliases[span.track_id],
                        track_reason="track_alias",
                        identity_source="retrospective",
                    )
                    changed = True
                revised_spans.append(span)
            if index in windows:
                revised_spans = list(windows[index])
                _validate_spans(
                    tuple(revised_spans), round(value.duration * _SAMPLE_RATE)
                )
                changed = tuple(revised_spans) != value.spans
            if not changed:
                continue
            revised = self._aligned_transcript(
                value.transcript, tuple(revised_spans), value.duration
            )
            self._alignment_history[index] = replace(value, spans=tuple(revised_spans))
            revisions.append(
                SystemAudioRevision(
                    index,
                    value.started_at,
                    revised.text,
                    revised.segments,
                    revised.event,
                )
            )
        result = self._aligned_transcript(transcript, spans, duration)
        if transcript.text:
            self._alignment_history[window_index] = _AlignedWindow(
                started_at, duration, transcript, spans
            )
        while len(self._alignment_history) > 31:
            self._alignment_history.popitem(last=False)
        return replace(result, revisions=tuple(revisions))

    async def transcribe(self, pcm: bytes) -> SystemAudioTranscript:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("system audio transcriber is closed")
            if _effectively_silent_pcm16(pcm):
                return SystemAudioTranscript("", "silence")
            self._generation += 1
            generation = self._generation
        if self._shared_client is not None:
            return await self._transcribe_shared(pcm)
        with self._state_lock:
            if self._closed or self._generation != generation:
                raise RuntimeError("system audio transcriber is closed")
            future = self._executor.submit(self._decode, pcm)
        return await asyncio.wrap_future(future)

    async def _transcribe_shared(self, pcm: bytes) -> SystemAudioTranscript:
        shared_client = self._shared_client
        if shared_client is None:  # pragma: no cover - caller invariant
            raise RuntimeError("shared environment ASR is not configured")
        deadline = time.monotonic() + _SHARED_RECONNECT_SECONDS
        while True:
            self._check_open()
            try:
                response = await shared_client.recognize(pcm)
            except (
                asyncio.IncompleteReadError,
                OSError,
                TimeoutError,
                ValueError,
            ) as error:
                # A voice toggle replaces the socket owner. Keep this same PCM
                # briefly in memory so the handoff does not erase a whole window.
                if isinstance(error, (ConnectionRefusedError, FileNotFoundError)):
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(min(0.1, remaining))
                        continue
                response = SharedAsrResponse("unavailable")
                unavailable_reason = f"transport_{type(error).__name__}"
            else:
                if response.status == "unavailable":
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(min(0.1, remaining))
                        continue
                unavailable_reason = (
                    "companion_voice_priority"
                    if response.status == "busy"
                    else "companion_recognizer_not_ready"
                    if response.status == "unavailable"
                    else "shared_recognizer_error"
                )
            break
        with self._state_lock:
            if self._closed:
                raise RuntimeError("system audio transcription was retired")
        if response.status != "ok":
            self._record_shared_diagnostic(response.status, unavailable_reason)
            return SystemAudioTranscript(
                "", "unavailable", unavailable_reason=unavailable_reason
            )
        if self._last_shared_diagnostic is not None:
            logger.info(
                "共享影音 ASR 已恢复: previous=%s", self._last_shared_diagnostic
            )
        self._last_shared_diagnostic = None
        return _transcript(
            response.text,
            response.event,
            getattr(response, "tokens", ()),
            getattr(response, "timestamps", ()),
        )

    def _record_shared_diagnostic(self, status: str, reason: str) -> None:
        if status == "busy":
            reason = "companion_voice_priority"
        elif status != "unavailable":
            reason = "shared_recognizer_error"
        diagnostic = f"{status}:{reason}"
        if diagnostic == self._last_shared_diagnostic:
            return
        self._last_shared_diagnostic = diagnostic
        logger.warning(
            "共享影音 ASR 本窗口不可用: status=%s, reason=%s; "
            "当前不加载第二份本地 ASR 模型",
            status,
            reason,
        )

    def _decode(self, pcm: bytes) -> SystemAudioTranscript:
        if self._decoder is None:
            AsrModelDownloader(self._config).ensure()
            self._decoder = StreamingAsr(self._config)
        try:
            self._decoder.accept_pcm(pcm)
            result = self._decoder.finish_utterance_update()
        except Exception:
            self._decoder = None
            raise
        return _transcript(
            result.text,
            result.event,
            getattr(result, "tokens", ()),
            getattr(result, "timestamps", ()),
        )

    async def release(self) -> None:
        """Drop the native decoder after queued work, unless newer work arrived."""
        with self._state_lock:
            if self._closed:
                return
            generation = self._generation
            future = self._executor.submit(self._release_if_current, generation)
        await asyncio.wrap_future(future)

    def _release_if_current(self, generation: int) -> None:
        with self._state_lock:
            if self._closed or self._generation != generation:
                return
            self._drop_decoder()

    async def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            future = self._executor.submit(self._drop_decoder)
        await asyncio.wrap_future(future)
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    def _drop_decoder(self) -> None:
        self._decoder = None
        self._alignment_history.clear()
        self._alignment_stream = None
        if self._speaker_diarizer is not None:
            self._speaker_diarizer.close()
            self._speaker_diarizer = None
        self._speaker_retry_after = 0.0
        self._last_speaker_window = None


def _validate_spans(spans: tuple[SpeakerSpan, ...], sample_count: int) -> None:
    previous = 0
    for span in spans:
        if (
            type(span.start_sample) is not int
            or type(span.end_sample) is not int
            or not previous <= span.start_sample < span.end_sample <= sample_count
        ):
            raise ValueError(
                "speaker intervals must be ordered inside their PCM window"
            )
        previous = span.end_sample


def format_system_audio_transcript(transcript: SystemAudioTranscript) -> str:
    if not transcript.segments:
        return format_system_audio_text(transcript.text)
    lines = []
    for segment in transcript.segments:
        text = (
            _MUSIC_TAG.sub("", _DECODER_TAG.sub("", segment.text)).strip()
            if segment.alignment_reason != "legacy_interval"
            else format_system_audio_text(segment.text).strip()
        )
        if not text:
            continue
        if segment.overlap:
            speaker = "多人同时说话（归属未确定）"
        elif segment.track_id and (match := _TRACK_ID.fullmatch(segment.track_id)):
            # The track stays stable while its optional historical identity is
            # resolved. An utterance/island ID is never a new-person label.
            speaker = f"声音{match[1]}-T{int(match[2]):02d}"
        elif segment.speaker_id:
            speaker = segment.speaker_id
        else:
            speaker = "说话人未确定"
        start = _format_segment_time(segment.start_seconds)
        end = _format_segment_time(segment.end_seconds)
        lines.append(f"[{start}—{end}] {speaker}：{text}")
    return "\n".join(lines)


def _format_segment_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    minutes, remainder = divmod(centiseconds, 6000)
    return f"{minutes:02d}:{remainder / 100:05.2f}"


def _transcript(
    text: str,
    raw_event: str,
    tokens: tuple[str, ...] = (),
    timestamps: tuple[float, ...] = (),
) -> SystemAudioTranscript:
    known_events = (
        "BGM",
        "Music",
        "Speech",
        "Applause",
        "Laughter",
        "Cough",
        "Sneeze",
        "Breath",
    )
    event = next((item for item in known_events if item in raw_event), "unknown")
    # Keep decoder metadata separate from recognized speech/lyrics. Background
    # music must not change how the transcript is presented to the model.
    clean = format_system_audio_text(text)
    if len(clean) > _MAX_TRANSCRIPT_CHARS:
        tokens, timestamps = (), ()
    return SystemAudioTranscript(
        clean[:_MAX_TRANSCRIPT_CHARS],
        event,
        tokens=tokens,
        timestamps=timestamps,
    )


def format_system_audio_text(text: str, event: str = "") -> str:
    """Return plain transcription, including when reading older tagged windows."""
    clean = _MUSIC_TAG.sub("", _DECODER_TAG.sub("", text)).strip()
    blocks = []
    for block in clean.split("\n\n"):
        lines = []
        meaningful = False
        for line in block.splitlines():
            match = _TIMED_SPEECH_LINE.fullmatch(line)
            body = match["body"] if match else line
            has_words = any(
                character.isalnum() or unicodedata.category(character).startswith("S")
                for character in body
            )
            if match and not has_words:
                continue
            lines.append(line)
            if not _AUDIO_WINDOW_HEADER.fullmatch(line):
                meaningful = meaningful or has_words
        if meaningful:
            blocks.append("\n".join(lines))
    # Existing stored transcripts have time/voice labels. Those labels must not
    # make a metadata-only utterance look like meaningful speech during replay.
    return "\n\n".join(blocks).strip()


def system_audio_words(text: str) -> str:
    """Content identity for label-only revisions, including passage headers."""
    lines = []
    for line in text.splitlines():
        match = _TIMED_SPEECH_LINE.fullmatch(line)
        lines.append(match["body"] if match else line)
    return re.sub(r"\s+", "", "".join(lines))


def _effectively_silent_pcm16(pcm: bytes) -> bool:
    """Reject only near-digital silence before claiming the shared ASR."""
    if not pcm:
        return True
    if len(pcm) % 2:
        # Preserve the decoder's existing malformed-PCM error boundary.
        return False
    samples = np.frombuffer(pcm, dtype="<i2")
    if samples.size == 0:
        return True
    minimum = int(samples.min())
    maximum = int(samples.max())
    peak = max(abs(minimum), abs(maximum))
    return peak <= _SILENCE_PEAK_PCM16
