"""Exercise early ASR against real processor/controller ownership boundaries."""

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.asr import AsrUpdate
from fish_memory.companion.audio_processing import ProcessedAudioFrame
from fish_memory.companion.client import CompanionAskResult, CompanionReplyContext
from fish_memory.companion.processor import CompanionAudioProcessor
from fish_memory.companion.session import CompanionSession
from fish_memory.companion.voice_drafts import (
    SpeculativeSilenceGate,
    VoiceDraftOwner,
    VoiceDraftTurnController,
)

from .test_draft_transcript_reuse import (
    SILENCE,
    SPEECH,
    BufferedAsr,
    PausingSmartTurn,
    SharedVadFrameProcessor,
)
from .test_processor import (
    FISH_ID,
    FakeApi,
    FakeDraftTransport,
    FakeDraftVad,
    FakeKws,
    FakeTts,
    RecordingDiagnostics,
)


class ControlledDecoder:
    def __init__(self, *, blocked: frozenset[int] = frozenset({0})) -> None:
        self.inputs: list[bytes] = []
        self.started = [asyncio.Event() for _ in range(4)]
        self.release = [threading.Event() for _ in range(4)]
        for index, release in enumerate(self.release):
            if index not in blocked:
                release.set()
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    @property
    def calls(self) -> int:
        return len(self.inputs)

    def recognize(self, pcm: bytes) -> str:
        return self.recognize_update(pcm).text

    def recognize_update(self, pcm: bytes) -> AsrUpdate:
        with self._lock:
            index = len(self.inputs)
            self.inputs.append(pcm)
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        self._loop.call_soon_threadsafe(self.started[index].set)
        try:
            assert self.release[index].wait(5), "test did not release native decode"
            return AsrUpdate(f"第{index + 1}次完整识别", is_final=True, event="Speech")
        finally:
            with self._lock:
                self._active -= 1

    def release_all(self) -> None:
        for release in self.release:
            release.set()


class ObservedTransport(FakeDraftTransport):
    def __init__(self) -> None:
        super().__init__()
        self.uploaded = asyncio.Queue()

    async def upsert_voice_draft(
        self, key: object, *, revision: int, content: str, **metadata: object
    ) -> bool:
        accepted = await super().upsert_voice_draft(
            key, revision=revision, content=content, **metadata
        )
        self.uploaded.put_nowait((key, revision, content))
        return accepted


class ObservedApi(FakeApi):
    def __init__(self) -> None:
        super().__init__()
        self.committed = asyncio.Event()

    async def commit_voice_turn(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
        promote_revision: int,
        content: str,
        reply_context: CompanionReplyContext | None = None,
    ) -> CompanionAskResult:
        result = await super().commit_voice_turn(
            key,
            revision=revision,
            promote_revision=promote_revision,
            content=content,
            reply_context=reply_context,
        )
        self.committed.set()
        return result


class ObservedController(VoiceDraftTurnController):
    async def wait_for_transcript(
        self, owner: VoiceDraftOwner, pcm_snapshot: bytes
    ) -> AsrUpdate | None:
        self.transcript_wait_started.set()
        return await super().wait_for_transcript(owner, pcm_snapshot)


class ScriptedFrames(SharedVadFrameProcessor):
    def __init__(
        self,
        primary: list[bool],
        candidate: list[bool],
        cleaned_overrides: dict[int, bytes],
    ) -> None:
        super().__init__(primary, candidate)
        self._index = 0
        self._cleaned_overrides = cleaned_overrides

    def process(self, *, capture_pcm: bytes, render_pcm: bytes) -> ProcessedAudioFrame:
        frame = super().process(capture_pcm=capture_pcm, render_pcm=render_pcm)
        if self._index in self._cleaned_overrides:
            frame = replace(frame, cleaned_pcm=self._cleaned_overrides[self._index])
        self._index += 1
        return frame


class MeasuredBufferedAsr(BufferedAsr):
    def __init__(self) -> None:
        super().__init__()
        self.size_probe_calls = 0

    def buffered_pcm_bytes(self) -> int:
        self.size_probe_calls += 1
        return len(self.accepted) * len(SPEECH)


@asynccontextmanager
async def early_case(
    primary: list[bool],
    candidate: list[bool],
    *,
    decoder: ControlledDecoder,
    cleaned_overrides: dict[int, bytes] | None = None,
    smart_turn: PausingSmartTurn | None = None,
    voice_draft_max_pcm_bytes: int | None = None,
) -> AsyncIterator[SimpleNamespace]:
    api = ObservedApi()
    session = CompanionSession(api, FakeTts())
    await session.set_binding(FISH_ID)
    await session.on_wake_word(play_ack=False)
    transport = ObservedTransport()
    diagnostics = RecordingDiagnostics()
    controller = ObservedController(
        client=transport, decoder=decoder, diagnostics=diagnostics
    )
    controller.transcript_wait_started = asyncio.Event()
    asr = MeasuredBufferedAsr()
    processor = CompanionAudioProcessor(
        session,
        FakeKws(),
        asr,
        frame_processor=ScriptedFrames(primary, candidate, cleaned_overrides or {}),
        smart_turn=smart_turn,
        voice_draft_vad=FakeDraftVad([]),
        voice_draft_gate=SpeculativeSilenceGate(
            silence_frames=10, prepare_silence_frames=1
        ),
        voice_draft_controller=controller,
        voice_draft_max_pcm_bytes=voice_draft_max_pcm_bytes,
        speech_start_frames=1,
        speech_end_frames=1,
        diagnostics=diagnostics,
    )
    try:
        yield SimpleNamespace(
            api=api,
            session=session,
            processor=processor,
            controller=controller,
            decoder=decoder,
            asr=asr,
            transport=transport,
            diagnostics=diagnostics,
        )
    finally:
        decoder.release_all()
        if smart_turn is not None:
            smart_turn.release.set()
        await processor.close()
        await controller.close()
        await session.close()


async def feed(c: SimpleNamespace, frames: list[bytes]) -> None:
    for pcm in frames:
        await c.processor.process_frame(pcm, SILENCE)


@pytest.mark.asyncio
async def test_draft_pcm_budget_avoids_copy_and_preserves_final_asr() -> None:
    decoder = ControlledDecoder(blocked=frozenset())
    async with early_case(
        [True] * 36 + [False],
        [True] * 2 + [False] * 35,
        decoder=decoder,
        voice_draft_max_pcm_bytes=len(SPEECH) * 2,
    ) as c:
        await feed(c, [SPEECH] * 2 + [SILENCE])

        assert c.asr.size_probe_calls == 1
        assert c.asr.snapshot_calls == 0
        assert decoder.calls == 0
        assert "voice_draft_asr_prepare_triggered" not in [
            name for name, _kwargs in c.diagnostics.events
        ]
        limits = [
            kwargs["metadata"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_fallback"
            and kwargs["metadata"].get("reason") == "buffer_limit"
        ]
        assert len(limits) == 1
        assert limits[0]["reason"] == "buffer_limit"
        assert limits[0]["pcmBytes"] == len(SPEECH) * 3

        # Later candidate frames remain circuit-broken for this logical turn.
        await feed(c, [SILENCE] * 33)
        assert c.asr.size_probe_calls == 1
        assert decoder.calls == 0
        assert "voice_draft_candidate_triggered" not in [
            name for name, _kwargs in c.diagnostics.events
        ]
        assert (
            sum(
                name == "voice_draft_fallback"
                and kwargs["metadata"].get("reason") == "buffer_limit"
                for name, kwargs in c.diagnostics.events
            )
            == 1
        )

        await feed(c, [SILENCE])
        await asyncio.wait_for(c.api.committed.wait(), 2)
        assert c.api.voice_commit_calls[0][2] == 0
        assert c.api.voice_commit_calls[0][3] == "正式识别"
        assert c.asr.snapshot_calls == 1
        assert c.asr.consumed == [SPEECH * 2 + SILENCE * 34]


@pytest.mark.asyncio
async def test_draft_pcm_budget_includes_exact_boundary() -> None:
    decoder = ControlledDecoder(blocked=frozenset())
    async with early_case(
        [True] * 3,
        [True] * 2 + [False],
        decoder=decoder,
        voice_draft_max_pcm_bytes=len(SPEECH) * 3,
    ) as c:
        await feed(c, [SPEECH] * 2 + [SILENCE])
        await asyncio.wait_for(decoder.started[0].wait(), 2)
        await c.controller.wait_idle()

        assert decoder.inputs == [SPEECH * 2 + SILENCE]
        assert c.asr.snapshot_calls == 0
        assert not any(
            name == "voice_draft_fallback"
            and kwargs["metadata"].get("reason") == "buffer_limit"
            for name, kwargs in c.diagnostics.events
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_ready_at", ["before_candidate", "candidate", "end"])
async def test_first_quiet_frame_prepares_without_upload_until_100ms_and_end_reuses(
    decode_ready_at: str,
) -> None:
    decoder = ControlledDecoder()
    async with early_case(
        [True] * 36 + [False],
        [True] * 2 + [False] * 35,
        decoder=decoder,
    ) as c:
        await feed(c, [SPEECH] * 2 + [SILENCE])
        await asyncio.wait_for(decoder.started[0].wait(), 2)
        assert decoder.inputs == [SPEECH * 2 + SILENCE]
        assert c.transport.upserts == c.api.voice_commit_calls == []

        if decode_ready_at == "before_candidate":
            decoder.release[0].set()
            await c.controller.wait_idle()
        await feed(c, [SILENCE] * 8)
        # Nine 10 ms frames are still below the upload threshold, even if
        # recognition has already completed. No wall-clock timing assertion.
        assert c.transport.upserts == c.api.voice_commit_calls == []
        assert decoder.calls == 1

        await feed(c, [SILENCE])
        if decode_ready_at == "candidate":
            decoder.release[0].set()
        if decode_ready_at != "end":
            await asyncio.wait_for(c.transport.uploaded.get(), 2)
        else:
            assert c.transport.upserts == []
        assert c.api.voice_commit_calls == []

        await feed(c, [SILENCE] * 24)
        assert c.api.voice_commit_calls == []
        finishing = asyncio.create_task(feed(c, [SILENCE]))
        try:
            if decode_ready_at == "end":
                await asyncio.wait_for(c.controller.transcript_wait_started.wait(), 2)
                assert not finishing.done()
                assert c.asr.snapshot_calls == c.asr.finish_calls == 0
                assert c.asr.consumed == []
                decoder.release[0].set()
            await asyncio.wait_for(finishing, 2)
            await asyncio.wait_for(c.api.committed.wait(), 2)
        finally:
            decoder.release_all()
            await asyncio.gather(finishing, return_exceptions=True)

        assert decoder.calls == decoder.max_active == 1
        assert len(c.transport.upserts) == len(c.api.voice_commit_calls) == 1
        assert c.api.voice_commit_calls[0][3] == "第1次完整识别"
        assert c.api.voice_commit_calls[0][2] == c.transport.upserts[0][1]
        assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        assert c.asr.consumed == [SPEECH * 2 + SILENCE * 34]
        assert c.asr.snapshot_utterance() == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("old_decode_completed", [False, True])
async def test_resumption_before_100ms_never_uploads_old_asr_and_keeps_full_pcm(
    old_decode_completed: bool,
) -> None:
    decoder = ControlledDecoder()
    prefix = SPEECH * 2 + SILENCE * 9 + SPEECH * 3
    async with early_case(
        [True] * 48 + [False],
        [True] * 2 + [False] * 9 + [True] * 3 + [False] * 35,
        decoder=decoder,
    ) as c:
        await feed(c, [SPEECH] * 2 + [SILENCE])
        await asyncio.wait_for(decoder.started[0].wait(), 2)
        if old_decode_completed:
            decoder.release[0].set()
            await c.controller.wait_idle()
        await feed(c, [SILENCE] * 8 + [SPEECH] * 3)
        assert c.transport.upserts == []
        assert c.asr.consumed == []
        assert c.asr.snapshot_utterance() == prefix

        await feed(c, [SILENCE] * 10)
        if not old_decode_completed:
            assert decoder.calls == 1
            assert c.transport.upserts == []
            decoder.release[0].set()
            await c.controller.wait_idle()
        else:
            await asyncio.wait_for(decoder.started[1].wait(), 2)
            uploaded = await asyncio.wait_for(c.transport.uploaded.get(), 2)
            assert uploaded[2] == "第2次完整识别"
            assert len(decoder.inputs) == 2
            assert decoder.inputs[0] == SPEECH * 2 + SILENCE
            assert decoder.inputs[1].startswith(prefix + SILENCE)
            assert set(decoder.inputs[1][len(prefix) :]) <= {0}
            assert decoder.max_active == 1
            assert c.api.voice_commit_calls == []

        await feed(c, [SILENCE] * 25)
        await asyncio.wait_for(c.api.committed.wait(), 2)
        if old_decode_completed:
            assert [call[2] for call in c.transport.upserts] == ["第2次完整识别"]
            assert c.api.voice_commit_calls[0][3] == "第2次完整识别"
            assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        else:
            assert c.transport.upserts == []
            assert c.api.voice_commit_calls[0][3] == "正式识别"
            assert c.asr.snapshot_calls == 1
            assert c.asr.finish_calls == 0
        assert c.asr.consumed == [prefix + SILENCE * 34]


@pytest.mark.asyncio
async def test_asr_track_change_invalidates_preparation_before_upload() -> None:
    decoder = ControlledDecoder(blocked=frozenset())
    other_track = b"b" * 320
    async with early_case(
        [True] * 36 + [False],
        [True] * 2 + [False] * 35,
        decoder=decoder,
        cleaned_overrides={3: SILENCE},
    ) as c:
        await feed(c, [SPEECH] * 2 + [SILENCE])
        await c.controller.wait_idle()
        assert decoder.calls == 1
        # The silence classifier sees zeros while authoritative ASR receives
        # another track. The old prefix match alone must not authorize reuse.
        await feed(c, [other_track] + [SILENCE] * 8)
        uploaded = await asyncio.wait_for(c.transport.uploaded.get(), 2)
        assert uploaded[2] == "第2次完整识别"
        assert decoder.inputs[1] == SPEECH * 2 + SILENCE + other_track + SILENCE * 8
        assert [call[2] for call in c.transport.upserts] == ["第2次完整识别"]

        await feed(c, [SILENCE] * 25)
        await asyncio.wait_for(c.api.committed.wait(), 2)
        assert c.api.voice_commit_calls[0][3] == "第2次完整识别"
        assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        assert c.asr.consumed == [SPEECH * 2 + SILENCE + other_track + SILENCE * 32]


@pytest.mark.asyncio
async def test_short_post_end_voice_invalidates_preparation_without_new_start() -> None:
    decoder = ControlledDecoder(blocked=frozenset())
    smart = PausingSmartTurn()
    async with early_case(
        [True] * 36 + [False, False],
        [True] * 2 + [False] * 35 + [True],
        decoder=decoder,
        smart_turn=smart,
    ) as c:
        await feed(c, [SPEECH] * 2 + [SILENCE])
        await c.controller.wait_idle()
        await feed(c, [SILENCE] * 9)
        await asyncio.wait_for(c.transport.uploaded.get(), 2)
        await feed(c, [SILENCE] * 25)
        assert await asyncio.to_thread(smart.started.wait, 1)
        assert c.api.voice_commit_calls == []
        # This activity is deliberately shorter than a new primary START.
        await feed(c, [SPEECH])
        smart.release.set()
        await asyncio.wait_for(c.api.committed.wait(), 2)
        assert c.api.voice_commit_calls[0][3] == "正式识别"
        assert c.asr.snapshot_calls == 1
        assert "asr_draft_reused" not in [name for name, _ in c.diagnostics.events]
        assert "primary_inference_tail" in [
            kwargs["metadata"]["reason"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_asr_reuse_missed"
        ]


@pytest.mark.asyncio
async def test_candidate_can_decode_after_overloaded_prepare_was_skipped() -> None:
    decoder = ControlledDecoder(blocked=frozenset())
    async with early_case(
        [True] * 36 + [False],
        [True] * 2 + [False] * 35,
        decoder=decoder,
    ) as c:
        c.queue_age_ms = 60.0
        c.processor._frame_queue_age_ms = lambda: c.queue_age_ms
        c.processor._frame_queue_depth = lambda: 0
        await feed(c, [SPEECH] * 2 + [SILENCE])
        await c.controller.wait_idle()
        assert decoder.calls == 0
        assert c.transport.upserts == []
        assert "audio_queue_age" in [
            kwargs["metadata"]["reason"]
            for name, kwargs in c.diagnostics.events
            if name == "voice_draft_fallback"
        ]

        await feed(c, [SILENCE] * 8)
        c.queue_age_ms = 0.0
        await feed(c, [SILENCE])
        await asyncio.wait_for(c.transport.uploaded.get(), 2)
        assert decoder.inputs == [SPEECH * 2 + SILENCE * 10]
        assert c.api.voice_commit_calls == []

        await feed(c, [SILENCE] * 25)
        await asyncio.wait_for(c.api.committed.wait(), 2)
        assert c.api.voice_commit_calls[0][3] == "第1次完整识别"
        assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        assert decoder.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("dropped_frame", [False, True])
async def test_ready_preparation_survives_queue_age_but_not_audio_loss(
    dropped_frame: bool,
) -> None:
    decoder = ControlledDecoder(blocked=frozenset())
    async with early_case(
        [True] * 36 + [False],
        [True] * 2 + [False] * 35,
        decoder=decoder,
    ) as c:
        c.queue_age_ms = 0.0
        c.dropped_frames = 0
        c.processor._frame_queue_age_ms = lambda: c.queue_age_ms
        c.processor._frame_queue_depth = lambda: 0
        c.processor._dropped_frames = lambda: c.dropped_frames
        await feed(c, [SPEECH] * 2 + [SILENCE])
        await c.controller.wait_idle()
        assert decoder.inputs == [SPEECH * 2 + SILENCE]
        assert c.transport.upserts == []

        await feed(c, [SILENCE] * 8)
        c.queue_age_ms = 60.0
        c.dropped_frames = int(dropped_frame)
        await feed(c, [SILENCE])
        await c.controller.wait_idle()
        if dropped_frame:
            assert c.transport.upserts == []
            assert "audio_frame_drop" in [
                kwargs["metadata"]["reason"]
                for name, kwargs in c.diagnostics.events
                if name == "voice_draft_fallback"
            ]
        else:
            assert len(c.transport.upserts) == 1
            assert c.transport.upserts[0][2] == "第1次完整识别"
        assert decoder.calls == 1
        assert c.api.voice_commit_calls == []

        await feed(c, [SILENCE] * 25)
        await asyncio.wait_for(c.api.committed.wait(), 2)
        if dropped_frame:
            assert c.api.voice_commit_calls[0][2] == 0
            assert c.api.voice_commit_calls[0][3] == "正式识别"
            assert c.asr.snapshot_calls == 1
        else:
            assert c.api.voice_commit_calls[0][3] == "第1次完整识别"
            assert c.asr.snapshot_calls == c.asr.finish_calls == 0
        assert decoder.calls == 1
