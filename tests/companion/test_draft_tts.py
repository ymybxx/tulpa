import asyncio
import gc
import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import cast
from uuid import UUID

import pytest

from fish_memory.answer.voice_draft_types import VoiceDraftKey
from fish_memory.companion.audio import PcmPlayback
from fish_memory.companion.draft_tts import (
    DraftTtsOutputFactory,
    DraftTtsSession,
    PromotablePcmSink,
)
from fish_memory.companion.playback import PlaybackArbiter
from fish_memory.companion.session import CompanionSession
from fish_memory.companion.tts import ApiTtsOutput, TtsConfiguration

from .test_session import FakeApi, FakeTts

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000002")
TURN_ID = UUID("00000000-0000-0000-0000-000000000003")
KEY = VoiceDraftKey(FISH_ID, SESSION_ID, TURN_ID)
CONFIG = TtsConfiguration("model", "voice", 16_000)


async def wait_for_condition(
    predicate: Callable[[], bool],
    *,
    max_wait_seconds: float = 1.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + max_wait_seconds
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition did not become true")
        await asyncio.sleep(0.001)


async def load_configuration() -> TtsConfiguration:
    return CONFIG


@pytest.mark.asyncio
async def test_pure_effect_draft_reports_bytes_only_after_owned_live_playback():
    class LivePcm(PcmPlayback):
        async def start(self):
            pass

    arbiter = PlaybackArbiter()
    live = LivePcm()
    sink = PromotablePcmSink(live, 100, playback_arbiter=arbiter)
    await sink.append(b"a" * 8, start_offset=0, end_offset=0)
    assert sink.buffered_bytes == 8
    assert sink.played_audio_bytes == 0
    owner = await arbiter.acquire("answer", TURN_ID)
    assert await sink.promote(expected_end_offset=0, owner=owner)
    assert sink.played_audio_bytes == 0
    live._callback(bytearray(4), 2)
    assert sink.played_audio_bytes == 4
    assert sink.played_offset == sink.display_offset == 0
    await arbiter.release(owner)
    assert sink.played_audio_bytes == 0
    sink.reset()
    assert sink.played_audio_bytes == 0


class FakeSink:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object | None]] = []
        self._played_offset = 0

    @property
    def played_offset(self) -> int:
        return self._played_offset

    @property
    def display_offset(self) -> int:
        return self._played_offset

    async def mark(self, end_offset: int) -> None:
        self._played_offset = max(self._played_offset, end_offset)

    async def start(self) -> None:
        self.calls.append(("start", None))

    def reset(self) -> None:
        self.calls.append(("reset", None))
        self._played_offset = 0

    async def append(
        self,
        pcm: bytes,
        *,
        start_offset: int,
        end_offset: int,
    ) -> None:
        self.calls.append(("append", (pcm, start_offset, end_offset)))
        self._played_offset = end_offset

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def pause(self) -> None:
        self.calls.append(("pause", None))

    async def resume(self) -> None:
        self.calls.append(("resume", None))

    async def duck(self) -> None:
        self.calls.append(("duck", None))

    async def unduck(self) -> None:
        self.calls.append(("unduck", None))

    async def wait_until_drained(self) -> None:
        self.calls.append(("drain", None))


class BlockingDrainSink(FakeSink):
    def __init__(self) -> None:
        super().__init__()
        self.drain_started = asyncio.Event()
        self.release_drain = asyncio.Event()

    async def wait_until_drained(self) -> None:
        self.calls.append(("drain_started", None))
        self.drain_started.set()
        await self.release_drain.wait()
        self.calls.append(("drain_finished", None))


class ResetWakesBlockingDrainSink(BlockingDrainSink):
    def reset(self) -> None:
        super().reset()
        self.release_drain.set()


@pytest.mark.asyncio
async def test_promoted_pcm_pause_preempts_passive_drain_wait() -> None:
    arbiter = PlaybackArbiter()
    live = BlockingDrainSink()
    sink = PromotablePcmSink(
        live,
        max_buffer_bytes=96_000,
        playback_arbiter=arbiter,
    )
    await sink.append(b"pcm!", start_offset=0, end_offset=4)
    owner = await arbiter.acquire("answer", TURN_ID)
    assert await sink.promote(expected_end_offset=4, owner=owner) is True
    live.calls.clear()
    live.release_drain.clear()

    drain_task = asyncio.create_task(sink.wait_until_drained())
    await asyncio.wait_for(live.drain_started.wait(), timeout=0.1)

    await asyncio.wait_for(sink.pause(), timeout=0.1)

    assert drain_task.done() is False
    assert live.calls == [("drain_started", None), ("pause", None)]
    live.release_drain.set()
    await drain_task
    assert await arbiter.release(owner) is True


@pytest.mark.asyncio
async def test_promoted_pcm_revoke_preempts_passive_drain_wait() -> None:
    live = ResetWakesBlockingDrainSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)
    await sink.append(b"pcm!", start_offset=0, end_offset=4)
    assert await sink.promote(expected_end_offset=4) is True
    live.calls.clear()
    live.release_drain.clear()

    drain_task = asyncio.create_task(sink.wait_until_drained())
    await asyncio.wait_for(live.drain_started.wait(), timeout=0.1)

    await asyncio.wait_for(sink.revoke(), timeout=0.1)
    await asyncio.wait_for(drain_task, timeout=0.1)

    assert live.calls == [
        ("drain_started", None),
        ("reset", None),
        ("drain_finished", None),
    ]
    assert sink.retirement_may_mutate_live_sink is False


@pytest.mark.asyncio
async def test_promoted_pcm_drain_snapshots_after_inflight_append_barrier() -> None:
    class BlockingAppendAndDrainSink(BlockingDrainSink):
        def __init__(self) -> None:
            super().__init__()
            self.append_started = asyncio.Event()
            self.release_append = asyncio.Event()

        async def append(
            self,
            pcm: bytes,
            *,
            start_offset: int,
            end_offset: int,
        ) -> None:
            self.calls.append(("append_started", (pcm, start_offset, end_offset)))
            self.append_started.set()
            await self.release_append.wait()
            await super().append(
                pcm,
                start_offset=start_offset,
                end_offset=end_offset,
            )

    live = BlockingAppendAndDrainSink()
    # Do not block the initial promotion append.
    live.release_append.set()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)
    await sink.append(b"seed", start_offset=0, end_offset=4)
    assert await sink.promote(expected_end_offset=4) is True
    live.calls.clear()
    live.append_started.clear()
    live.release_append.clear()

    append_task = asyncio.create_task(
        sink.append(b"late", start_offset=4, end_offset=8)
    )
    await asyncio.wait_for(live.append_started.wait(), timeout=0.1)
    drain_task = asyncio.create_task(sink.wait_until_drained())
    await asyncio.sleep(0)

    assert drain_task.done() is False
    assert ("drain_started", None) not in live.calls

    live.release_append.set()
    await append_task
    await asyncio.wait_for(live.drain_started.wait(), timeout=0.1)
    live.release_drain.set()
    await drain_task


@pytest.mark.asyncio
async def test_pcm_is_private_until_exact_promotion() -> None:
    live = FakeSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)

    await sink.start()
    await sink.append(b"\x01\x02", start_offset=0, end_offset=6)

    assert live.calls == []
    assert sink.played_offset == 0

    promoted = await sink.promote(expected_end_offset=6)

    assert promoted is True
    assert live.calls == [
        ("start", None),
        ("reset", None),
        ("append", (b"\x01\x02", 0, 6)),
    ]


@pytest.mark.asyncio
async def test_private_pcm_acquires_no_owner_and_promotion_keeps_one_token() -> None:
    arbiter = PlaybackArbiter()
    live = FakeSink()
    sink = PromotablePcmSink(
        live,
        max_buffer_bytes=96_000,
        playback_arbiter=arbiter,
    )

    await sink.start()
    await sink.append(b"\x01\x02", start_offset=0, end_offset=4)
    assert arbiter.current_owner is None
    assert live.calls == []

    owner = await arbiter.acquire("answer", TURN_ID)
    assert await sink.promote(expected_end_offset=4, owner=owner) is True
    calls_after_promotion = list(live.calls)
    assert await arbiter.release(owner) is True
    await arbiter.acquire("agent", UUID(int=404))
    await sink.append(b"\x03\x04", start_offset=4, end_offset=6)
    sink.reset()

    assert live.calls == calls_after_promotion


@pytest.mark.asyncio
async def test_pcm_sink_validates_the_first_packet_after_empty_promotion() -> None:
    live = FakeSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)

    assert await sink.promote(expected_end_offset=6) is True
    assert not any(name == "append" for name, _ in live.calls)
    await sink.append(b"late", start_offset=1, end_offset=6)
    await sink.append(b"late", start_offset=0, end_offset=6)
    assert not any(name == "append" for name, _ in live.calls)


@pytest.mark.asyncio
async def test_pcm_range_mismatch_discards_everything_and_cannot_promote() -> None:
    live = FakeSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)

    await sink.append(b"aa", start_offset=0, end_offset=3)
    await sink.append(b"bb", start_offset=4, end_offset=6)
    await sink.append(b"late", start_offset=0, end_offset=3)

    assert await sink.promote(expected_end_offset=6) is False
    assert sink.buffered_bytes == 0
    assert live.calls == []


@pytest.mark.asyncio
async def test_pcm_repeated_sentence_ranges_merge_but_promote_only_once() -> None:
    live = FakeSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)

    await sink.append(b"aa", start_offset=0, end_offset=3)
    await sink.append(b"bb", start_offset=0, end_offset=3)

    assert await sink.promote(expected_end_offset=3) is True
    assert await sink.promote(expected_end_offset=3) is False
    await sink.append(b"cc", start_offset=3, end_offset=5)
    assert [call for call in live.calls if call[0] == "append"] == [
        ("append", (b"aabb", 0, 3)),
        ("append", (b"cc", 3, 5)),
    ]


@pytest.mark.asyncio
async def test_pcm_cap_is_all_or_nothing_and_late_packets_stay_private() -> None:
    live = FakeSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=6)

    await sink.append(b"aaaa", start_offset=0, end_offset=2)
    await sink.append(b"bbbb", start_offset=0, end_offset=2)
    await sink.append(b"late", start_offset=0, end_offset=2)

    assert sink.buffered_bytes == 0
    assert await sink.promote(expected_end_offset=2) is False
    assert live.calls == []


@pytest.mark.asyncio
async def test_cancel_during_readiness_cannot_rearm_discarded_sink() -> None:
    live = FakeSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)

    sink.discard()
    await sink.start()
    sink.reset()
    await sink.append(b"late", start_offset=0, end_offset=4)

    assert sink.buffered_bytes == 0
    assert await sink.promote(expected_end_offset=4) is False
    assert live.calls == []


@pytest.mark.asyncio
async def test_cancel_while_attaching_never_replays_private_pcm() -> None:
    class BlockingStartSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def start(self) -> None:
            self.calls.append(("start", None))
            self.started.set()
            await self.release.wait()

    live = BlockingStartSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)
    await sink.append(b"pcm!", start_offset=0, end_offset=4)

    promotion = asyncio.create_task(sink.promote(expected_end_offset=4))
    await live.started.wait()
    sink.discard()
    live.release.set()

    assert await promotion is False
    assert not [call for call in live.calls if call[0] == "append"]


@pytest.mark.asyncio
async def test_cancel_while_attaching_reports_discarded_private_bytes() -> None:
    class BlockingStartSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def start(self) -> None:
            self.started.set()
            await self.release.wait()

    events: list[tuple[str, dict[str, object]]] = []
    live = BlockingStartSink()
    sink = PromotablePcmSink(
        live,
        max_buffer_bytes=96_000,
        on_event=lambda name, metadata: events.append((name, metadata)),
    )
    await sink.append(b"pcm!", start_offset=0, end_offset=4)

    promotion = asyncio.create_task(sink.promote(expected_end_offset=4))
    await live.started.wait()
    sink.discard()
    live.release.set()

    assert await promotion is False
    assert events[-1] == (
        "voice_draft_tts_pcm_discarded",
        {"pcmBytes": 4, "reason": "closed"},
    )


@pytest.mark.asyncio
async def test_packet_during_blocked_attach_waits_then_forwards_exact_range() -> None:
    class BlockingStartSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def start(self) -> None:
            self.calls.append(("start", None))
            self.started.set()
            await self.release.wait()

    live = BlockingStartSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)
    await sink.append(b"aa", start_offset=0, end_offset=4)
    promotion = asyncio.create_task(sink.promote(expected_end_offset=4))
    await live.started.wait()

    concurrent_packet = asyncio.create_task(
        sink.append(b"bb", start_offset=0, end_offset=4)
    )
    await asyncio.sleep(0)
    assert concurrent_packet.done() is False

    live.release.set()
    assert await promotion is True
    await concurrent_packet

    assert [call for call in live.calls if call[0] == "append"] == [
        ("append", (b"aa", 0, 4)),
        ("append", (b"bb", 0, 4)),
    ]


@pytest.mark.asyncio
async def test_cancel_during_blocked_live_append_resets_mutation() -> None:
    class BlockingAppendSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.append_started = asyncio.Event()
            self.release_append = asyncio.Event()
            self.queued: list[bytes] = []

        def reset(self) -> None:
            super().reset()
            self.queued.clear()

        async def append(
            self,
            pcm: bytes,
            *,
            start_offset: int,
            end_offset: int,
        ) -> None:
            self.calls.append(("append", (pcm, start_offset, end_offset)))
            if pcm == b"pcm!":
                self.append_started.set()
                await self.release_append.wait()
            self.queued.append(pcm)

    live = BlockingAppendSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)
    await sink.append(b"pcm!", start_offset=0, end_offset=4)
    promotion = asyncio.create_task(sink.promote(expected_end_offset=4))
    await live.append_started.wait()

    cancellation = asyncio.create_task(sink.revoke())
    await asyncio.sleep(0)
    assert cancellation.done() is False
    live.release_append.set()

    assert await promotion is False
    await cancellation
    assert live.queued == []
    assert live.calls[-1] == ("reset", None)


@pytest.mark.asyncio
async def test_sync_reset_fences_then_async_revoke_drains_blocked_append() -> None:
    class BlockingAppendSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.append_started = asyncio.Event()
            self.release_append = asyncio.Event()
            self.queued: list[bytes] = []

        def reset(self) -> None:
            super().reset()
            self.queued.clear()

        async def append(
            self,
            pcm: bytes,
            *,
            start_offset: int,
            end_offset: int,
        ) -> None:
            self.calls.append(("append", (pcm, start_offset, end_offset)))
            if pcm == b"pcm!":
                self.append_started.set()
                await self.release_append.wait()
            self.queued.append(pcm)

    live = BlockingAppendSink()
    sink = PromotablePcmSink(live, max_buffer_bytes=96_000)
    await sink.append(b"seed", start_offset=0, end_offset=4)
    assert await sink.promote(expected_end_offset=4)
    forwarded = asyncio.create_task(sink.append(b"pcm!", start_offset=0, end_offset=4))
    await live.append_started.wait()

    sink.reset()
    late = asyncio.create_task(sink.append(b"late", start_offset=0, end_offset=4))
    await asyncio.sleep(0)
    assert late.done() is True
    cleanup = asyncio.create_task(sink.revoke())
    live.release_append.set()

    await forwarded
    await cleanup
    assert live.queued == []


class FakeDraftClient:
    def __init__(self) -> None:
        self.requests: list[tuple[VoiceDraftKey, int]] = []
        self.events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()

    async def iter_voice_draft_events(
        self,
        key: VoiceDraftKey,
        *,
        revision: int,
    ) -> AsyncIterator[dict[str, object]]:
        self.requests.append((key, revision))
        while True:
            event = await self.events.get()
            if event is None:
                return
            yield event


class FakeDraftOutput:
    def __init__(
        self,
        sink: PromotablePcmSink,
        *,
        emit_pcm: bool = True,
        provider_state: object | None = None,
    ) -> None:
        self.sink = sink
        self.configuration = CONFIG
        self.provider_state = provider_state or SimpleNamespace(
            ready=True,
            active=True,
            error_free=True,
            provider="dashscope",
            configuration=CONFIG,
        )
        self.emit_pcm = emit_pcm
        self.calls: list[tuple[str, object | None]] = []
        self.played_offset = 0
        self.flush_count = 0

    def configure(self, *, model: str, voice: str) -> None:
        self.calls.append(("configure", (model, voice)))
        self.configuration = TtsConfiguration(
            model,
            voice,
            self.configuration.sample_rate,
        )

    async def start(self) -> None:
        self.calls.append(("start", None))
        await self.sink.start()

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))
        if self.emit_pcm:
            await self.sink.append(
                text.encode(),
                start_offset=0,
                end_offset=len(text),
            )

    async def finish(self) -> None:
        self.calls.append(("finish", None))

    async def flush(self) -> None:
        self.flush_count += 1

    async def stop(self) -> None:
        await self.interrupt()

    async def interrupt(self) -> None:
        self.calls.append(("interrupt", None))

    async def pause(self) -> None:
        await self.sink.pause()

    async def resume(self) -> None:
        await self.sink.resume()

    async def duck(self) -> None:
        await self.sink.duck()

    async def unduck(self) -> None:
        await self.sink.unduck()

    async def close_transport(self) -> None:
        self.calls.append(("close_transport", None))


class RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_frame(self, *_args: object, **_kwargs: object) -> int:
        return -1

    def record_event(self, name: str, **kwargs: object) -> None:
        self.events.append((name, kwargs))


def draft_snapshot(
    content: str,
    *,
    revision: int = 4,
    flush_offset: int = 0,
    state: str = "active",
) -> dict[str, object]:
    return {
        "event": "draft.snapshot",
        "data": {
            "turnId": str(TURN_ID),
            "revision": revision,
            "state": state,
            "content": content,
            "startOffset": 0,
            "endOffset": len(content),
            "flushOffset": flush_offset,
        },
    }


def draft_segment(flush_offset: int, *, revision: int = 4) -> dict[str, object]:
    return {
        "event": "draft.segment",
        "data": {
            "turnId": str(TURN_ID),
            "revision": revision,
            "flushOffset": flush_offset,
        },
    }


@pytest.mark.asyncio
async def test_draft_tts_consumes_exact_revision_and_submits_only_first_segment() -> (
    None
):
    client = FakeDraftClient()
    live = FakeSink()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )

    assert await draft.start(KEY, revision=4)
    await client.events.put(draft_snapshot("第一段。第二段。"))
    await client.events.put(draft_segment(len("第一段。")))
    await client.events.put(draft_segment(len("第一段。第二段。")))
    for _ in range(50):
        if any(call[0] == "append" for call in outputs[0].calls):
            break
        await asyncio.sleep(0.001)

    assert client.requests == [(KEY, 4)]
    assert [call for call in outputs[0].calls if call[0] == "append"] == [
        ("append", "第一段。")
    ]
    assert live.calls == []
    await draft.close()


@pytest.mark.asyncio
async def test_private_first_pause_commits_before_the_lane_finishes() -> None:
    client, live = FakeDraftClient(), FakeSink()
    outputs = []

    def factory(sink):
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=factory,
        load_configuration=load_configuration,
    )
    try:
        assert await draft.start(KEY, revision=4)
        prefix = "你好，"
        content = prefix + "后面还没说完"
        await draft.accept_event(draft_snapshot(content, flush_offset=0))
        assert [c for c in outputs[0].calls if c[0] == "append"] == [("append", prefix)]
        assert outputs[0].flush_count == 1
        assert live.calls == []
        promotion = await draft.promote(
            KEY, revision=4, content=content, flush_offset=0
        )
        assert promotion is not None
        assert promotion.output is outputs[0]
        assert promotion.submitted_offset == len(prefix)
        assert len(outputs) == 1
    finally:
        await draft.close()


@pytest.mark.asyncio
async def test_private_segment_without_punctuation_flushes_original_tts() -> None:
    client, live = FakeDraftClient(), FakeSink()
    outputs = []

    def factory(sink):
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=factory,
        load_configuration=load_configuration,
    )
    try:
        assert await draft.start(KEY, revision=4)
        text = "我再试着帮你查哦"
        await draft.accept_event(draft_snapshot(text))
        assert outputs[0].flush_count == 0
        await draft.accept_event(draft_segment(len(text)))
        assert outputs[0].flush_count == 1
        assert [c for c in outputs[0].calls if c[0] == "append"] == [("append", text)]
        assert live.calls == []
        promotion = await draft.promote(
            KEY, revision=4, content=text, flush_offset=len(text)
        )
        assert promotion is not None
        assert promotion.output is outputs[0]
        assert promotion.submitted_offset == len(text)
        assert len(outputs) == 1
    finally:
        await draft.close()


@pytest.mark.asyncio
async def test_draft_tts_records_provider_text_pcm_and_promotion_without_content() -> (
    None
):
    client = FakeDraftClient()
    live = FakeSink()
    diagnostics = RecordingDiagnostics()

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=cast(
            DraftTtsOutputFactory,
            lambda sink: FakeDraftOutput(sink),
        ),
        load_configuration=load_configuration,
        diagnostics=diagnostics,
    )

    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("私有第一段。", flush_offset=6))
    promotion = await draft.promote(
        KEY,
        revision=4,
        content="私有第一段。公开后续。",
        flush_offset=6,
    )

    assert promotion is not None
    assert [name for name, _ in diagnostics.events] == [
        "voice_draft_tts_provider_ready",
        "voice_draft_first_content",
        "voice_draft_tts_first_audio_buffered",
        "voice_draft_tts_text_submitted",
        "voice_draft_tts_pcm_promoted",
    ]
    assert all("text" not in kwargs for _, kwargs in diagnostics.events)
    assert diagnostics.events[-1][1]["metadata"] == {
        "turnId": str(TURN_ID),
        "revision": 4,
        "pcmBytes": 18,
    }
    assert all(
        cast(dict[str, object], kwargs["metadata"])["turnId"] == str(TURN_ID)
        and cast(dict[str, object], kwargs["metadata"])["revision"] == 4
        for _, kwargs in diagnostics.events
    )
    await draft.close()


@pytest.mark.asyncio
async def test_private_event_from_wrong_revision_invalidates_generation() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    output: FakeDraftOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal output
        output = FakeDraftOutput(sink)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)

    await draft.accept_event(draft_snapshot("旧内容。", revision=3, flush_offset=4))

    assert output is not None
    assert ("append", "旧内容。") not in output.calls
    assert ("interrupt", None) in output.calls
    assert live.calls == []
    await draft.close()


@pytest.mark.asyncio
async def test_exact_snapshot_promotes_same_ready_provider_and_pcm_prefix() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    output: FakeDraftOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal output
        output = FakeDraftOutput(sink)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))
    assert output is not None

    promotion = await draft.promote(
        KEY,
        revision=4,
        content="第一段。后续",
        flush_offset=4,
    )

    assert promotion is not None
    assert promotion.output is output
    assert promotion.submitted_offset == 4
    assert [call for call in output.calls if call[0] == "start"] == [("start", None)]
    assert live.calls == [
        ("start", None),
        ("reset", None),
        ("append", ("第一段。".encode(), 0, 4)),
    ]
    assert (
        await draft.promote(
            KEY,
            revision=4,
            content="第一段。后续",
            flush_offset=4,
        )
        is None
    )
    assert ("interrupt", None) not in output.calls
    await draft.close()


@pytest.mark.asyncio
async def test_draft_session_promotion_returns_output_bound_to_public_owner() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    arbiter = PlaybackArbiter()
    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=cast(
            DraftTtsOutputFactory,
            lambda sink: FakeDraftOutput(sink),
        ),
        load_configuration=load_configuration,
        playback_arbiter=arbiter,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))
    assert arbiter.current_owner is None
    owner = await arbiter.acquire("answer", UUID(int=404))

    promotion = await draft.promote(
        KEY,
        revision=4,
        content="第一段。后续",
        flush_offset=4,
        playback_owner=owner,
    )

    assert promotion is not None
    calls_after_promotion = list(live.calls)
    assert await arbiter.release(owner) is True
    await promotion.output.append("过期后续")
    assert live.calls == calls_after_promotion
    await draft.close()


@pytest.mark.asyncio
async def test_real_api_tts_late_pcm_uses_one_owner_layer_after_promotion() -> None:
    class ControlledSocket:
        def __init__(self) -> None:
            self.incoming: asyncio.Queue[str | bytes | None] = asyncio.Queue()
            self.closed = False

        async def send(self, value: str) -> None:
            message = json.loads(value)
            if message["type"] == "start":
                await self.incoming.put(
                    json.dumps(
                        {
                            "type": "ready",
                            "provider": "fake",
                            "model": CONFIG.model,
                            "voice": CONFIG.voice,
                            "sampleRate": CONFIG.sample_rate,
                        }
                    )
                )
            elif message["type"] == "append":
                await self.incoming.put(
                    json.dumps({"type": "audio", "text": message["text"]})
                )
                await self.incoming.put(b"\x01\x02")
            elif message["type"] in {"finish", "stop"}:
                await self.incoming.put(json.dumps({"type": "finished"}))

        async def recv(self) -> str | bytes:
            value = await self.incoming.get()
            if value is None:
                raise RuntimeError("closed")
            return value

        async def close(self) -> None:
            self.closed = True
            await self.incoming.put(None)

    client = FakeDraftClient()
    live = FakeSink()
    arbiter = PlaybackArbiter()
    socket = ControlledSocket()
    outputs: list[ApiTtsOutput] = []

    def output_factory(sink: PromotablePcmSink) -> ApiTtsOutput:
        output = ApiTtsOutput(
            "http://test",
            sink=sink,
            model=CONFIG.model,
            voice=CONFIG.voice,
            sample_rate=CONFIG.sample_rate,
            websocket_connector=lambda _url: asyncio.sleep(0, result=socket),
        )
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
        playback_arbiter=arbiter,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("甲段。", flush_offset=3))
    await wait_for_condition(lambda: bool(outputs[0]._audio_packet_count))
    owner = await arbiter.acquire("answer", UUID(int=405))
    promotion = await draft.promote(
        KEY,
        revision=4,
        content="甲段。公开后续。",
        flush_offset=3,
        playback_owner=owner,
    )
    assert promotion is not None

    try:
        await promotion.output.append("乙段。")
        await asyncio.wait_for(
            wait_for_condition(
                lambda: len([call for call in live.calls if call[0] == "append"]) == 2
            ),
            timeout=0.2,
        )
    finally:
        await outputs[0].interrupt()
        await asyncio.wait_for(outputs[0].close_transport(), timeout=1)
        await arbiter.release(owner)
        await asyncio.wait_for(draft.close(), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("draft_has_old_arbiter", [False, True])
async def test_provided_session_propagates_runner_arbiter_to_draft(
    draft_has_old_arbiter: bool,
) -> None:
    old_arbiter = PlaybackArbiter()
    runner_arbiter = PlaybackArbiter()
    client = FakeDraftClient()
    live = FakeSink()
    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=cast(
            DraftTtsOutputFactory,
            lambda sink: FakeDraftOutput(sink),
        ),
        load_configuration=load_configuration,
        playback_arbiter=old_arbiter if draft_has_old_arbiter else None,
    )
    session = CompanionSession(
        FakeApi(),
        FakeTts(),
        draft_tts=draft,
        playback_arbiter=old_arbiter,
    )

    session.attach_playback_arbiter(runner_arbiter)

    assert draft._playback_arbiter is runner_arbiter
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))
    owner = await runner_arbiter.acquire("answer", UUID(int=406))
    promotion = await draft.promote(
        KEY,
        revision=4,
        content="第一段。后续",
        flush_offset=4,
        playback_owner=owner,
    )
    assert promotion is not None
    await runner_arbiter.release(owner)
    await session.close()


@pytest.mark.asyncio
async def test_ready_active_provider_without_first_pcm_is_adopted() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    output: FakeDraftOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal output
        output = FakeDraftOutput(sink, emit_pcm=False)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    assert output is not None
    assert output.calls[:2] == [
        ("configure", (CONFIG.model, CONFIG.voice)),
        ("start", None),
    ]
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))
    assert output.sink.buffered_bytes == 0

    promotion = await draft.promote(
        KEY,
        revision=4,
        content="第一段。后续",
        flush_offset=4,
    )

    assert promotion is not None
    assert promotion.output is output
    assert promotion.submitted_offset == 4
    assert not any(name == "append" for name, _ in live.calls)
    assert ("interrupt", None) not in output.calls

    await output.sink.append(b"late", start_offset=0, end_offset=4)
    assert [call for call in live.calls if call[0] == "append"] == [
        ("append", (b"late", 0, 4))
    ]
    await draft.close()


@pytest.mark.asyncio
async def test_provider_finished_with_zero_audio_falls_back_without_live_start() -> (
    None
):
    client = FakeDraftClient()
    live = FakeSink()
    output: FakeDraftOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal output
        output = FakeDraftOutput(sink, emit_pcm=False)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))
    assert output is not None
    output.provider_state = SimpleNamespace(
        ready=True,
        active=False,
        error_free=True,
        provider="dashscope",
        configuration=CONFIG,
    )

    promotion = await draft.promote(
        KEY,
        revision=4,
        content="第一段。",
        flush_offset=4,
    )

    assert promotion is None
    assert live.calls == []
    assert ("interrupt", None) in output.calls
    await draft.close()


@pytest.mark.asyncio
async def test_in_flight_text_submission_is_joined_by_promotion() -> None:
    class BlockingSubmitOutput(FakeDraftOutput):
        def __init__(self, sink: PromotablePcmSink) -> None:
            super().__init__(sink, emit_pcm=False)
            self.submit_started = asyncio.Event()
            self.release_submit = asyncio.Event()

        async def append(self, text: str) -> None:
            self.calls.append(("append", text))
            self.submit_started.set()
            await self.release_submit.wait()

    client = FakeDraftClient()
    live = FakeSink()
    output: BlockingSubmitOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> BlockingSubmitOutput:
        nonlocal output
        output = BlockingSubmitOutput(sink)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    assert output is not None
    acceptance = asyncio.create_task(
        draft.accept_event(draft_snapshot("第一段。", flush_offset=4))
    )
    await output.submit_started.wait()

    promotion = asyncio.create_task(
        draft.promote(
            KEY,
            revision=4,
            content="第一段。",
            flush_offset=4,
        )
    )
    try:
        await asyncio.sleep(0)
        assert not promotion.done()
        assert live.calls == []
        assert ("interrupt", None) not in output.calls
        output.release_submit.set()
        result = await asyncio.wait_for(promotion, timeout=0.5)
        assert result is not None
        assert result.output is output
        assert output.calls.count(("append", "第一段。")) == 1
    finally:
        output.release_submit.set()
        await asyncio.gather(acceptance, promotion, return_exceptions=True)
        await draft.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_state",
    [
        SimpleNamespace(
            ready=True,
            active=False,
            error_free=True,
            provider="dashscope",
            configuration=CONFIG,
        ),
        SimpleNamespace(
            ready=True,
            active=True,
            error_free=False,
            provider="dashscope",
            configuration=CONFIG,
        ),
        SimpleNamespace(
            ready=False,
            active=True,
            error_free=True,
            provider="dashscope",
            configuration=CONFIG,
        ),
    ],
)
async def test_terminal_or_error_provider_without_pcm_falls_back(
    provider_state: object,
) -> None:
    client = FakeDraftClient()
    live = FakeSink()

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        return FakeDraftOutput(
            sink,
            emit_pcm=False,
            provider_state=provider_state,
        )

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))

    assert (
        await draft.promote(
            KEY,
            revision=4,
            content="第一段。",
            flush_offset=4,
        )
        is None
    )
    assert live.calls == []
    await draft.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actual_configuration",
    [
        TtsConfiguration("other-model", "voice", 16_000),
        TtsConfiguration("model", "other-voice", 16_000),
        TtsConfiguration("model", "voice", 22_050),
    ],
)
async def test_actual_ready_configuration_mismatch_falls_back(
    actual_configuration: TtsConfiguration,
) -> None:
    client = FakeDraftClient()
    live = FakeSink()
    mismatched = SimpleNamespace(
        ready=True,
        active=True,
        error_free=True,
        provider="local",
        configuration=actual_configuration,
    )

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        return FakeDraftOutput(sink, provider_state=mismatched)

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))

    assert (
        await draft.promote(
            KEY,
            revision=4,
            content="第一段。",
            flush_offset=4,
        )
        is None
    )
    assert live.calls == []
    await draft.close()


@pytest.mark.asyncio
async def test_voice_change_between_draft_and_public_snapshot_falls_back() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    changed = TtsConfiguration("model", "new-voice", 16_000)
    configurations = iter((CONFIG, changed))

    async def load_current_configuration() -> TtsConfiguration:
        return next(configurations)

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=FakeDraftOutput,
        load_configuration=load_current_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))

    assert (
        await draft.promote(
            KEY,
            revision=4,
            content="第一段。",
            flush_offset=4,
        )
        is None
    )
    assert live.calls == []
    await draft.close()


@pytest.mark.asyncio
async def test_unexpected_private_sse_eof_makes_pcm_unpromotable() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    output: FakeDraftOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal output
        output = FakeDraftOutput(sink)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await client.events.put(draft_snapshot("第一段。", flush_offset=4))
    await client.events.put(None)
    assert output is not None
    for _ in range(50):
        if ("interrupt", None) in output.calls:
            break
        await asyncio.sleep(0.001)

    assert (
        await draft.promote(
            KEY,
            revision=4,
            content="第一段。",
            flush_offset=4,
        )
        is None
    )
    assert live.calls == []
    await draft.close()


@pytest.mark.asyncio
async def test_committed_private_sse_terminal_remains_promotable() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    output: FakeDraftOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal output
        output = FakeDraftOutput(sink)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await client.events.put(
        draft_snapshot("第一段。", flush_offset=4, state="committed")
    )
    await client.events.put(None)
    for _ in range(50):
        if output is not None and ("append", "第一段。") in output.calls:
            break
        await asyncio.sleep(0.001)

    promotion = await draft.promote(
        KEY,
        revision=4,
        content="第一段。",
        flush_offset=4,
    )

    assert promotion is not None
    await draft.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("public_revision", "public_content", "public_flush", "public_config"),
    [
        (5, "第一段。", 4, CONFIG),
        (4, "不同内容", 4, CONFIG),
        (4, "第一段。", 4, TtsConfiguration("other", "voice", 16_000)),
    ],
)
async def test_draft_tts_mismatch_discards_private_pcm(
    public_revision: int,
    public_content: str,
    public_flush: int,
    public_config: TtsConfiguration,
) -> None:
    client = FakeDraftClient()
    live = FakeSink()
    output: FakeDraftOutput | None = None

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal output
        output = FakeDraftOutput(sink)
        return output

    configurations = iter((CONFIG, public_config))

    async def load_current_configuration() -> TtsConfiguration:
        return next(configurations)

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_current_configuration,
    )
    assert await draft.start(KEY, revision=4)
    await draft.accept_event(draft_snapshot("第一段。", flush_offset=4))
    assert output is not None

    promotion = await draft.promote(
        KEY,
        revision=public_revision,
        content=public_content,
        flush_offset=public_flush,
    )

    assert promotion is None
    assert live.calls == []
    assert ("interrupt", None) in output.calls
    await draft.close()


@pytest.mark.asyncio
async def test_old_generation_pcm_cannot_enter_new_or_public_sink() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=1)
    old_sink = outputs[0].sink
    await draft.cancel(KEY, through_revision=2)
    assert await draft.start(KEY, revision=3)

    await old_sink.append(b"late", start_offset=0, end_offset=4)
    await outputs[1].sink.append(b"new!", start_offset=0, end_offset=3)

    assert old_sink.buffered_bytes == 0
    assert outputs[1].sink.buffered_bytes == 4
    assert live.calls == []
    await draft.close()


@pytest.mark.asyncio
async def test_stale_cancel_or_promotion_cannot_retire_newer_revision() -> None:
    client = FakeDraftClient()
    live = FakeSink()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=3)

    await draft.cancel(KEY, through_revision=2)
    assert (
        await draft.promote(
            KEY,
            revision=1,
            content="旧内容",
            flush_offset=0,
        )
        is None
    )
    await draft.accept_event(draft_snapshot("新内容。", revision=3, flush_offset=4))

    assert ("interrupt", None) not in outputs[0].calls
    assert ("append", "新内容。") in outputs[0].calls
    await draft.close()


@pytest.mark.asyncio
async def test_newer_start_owns_revision_while_older_retirement_is_blocked() -> None:
    previous_key = VoiceDraftKey(FISH_ID, SESSION_ID, UUID(int=99))
    client = FakeDraftClient()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=FakeSink(),
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(previous_key, revision=9)
    retirement_started = asyncio.Event()
    release_retirement = asyncio.Event()
    original_revoke = outputs[0].sink.revoke

    async def blocked_revoke() -> None:
        retirement_started.set()
        await release_retirement.wait()
        await original_revoke()

    outputs[0].sink.revoke = blocked_revoke  # type: ignore[method-assign]
    old_start = asyncio.create_task(draft.start(KEY, revision=1))
    await retirement_started.wait()
    new_start = asyncio.create_task(draft.start(KEY, revision=3))
    try:
        for _ in range(50):
            if new_start.done():
                break
            await asyncio.sleep(0.001)
        assert new_start.done() is True
        assert await new_start is True
        assert draft._key == KEY
        assert draft._revision == 3
        assert draft._output is outputs[-1]
        assert draft._event_task is not None

        release_retirement.set()
        assert await old_start is False
        assert draft._key == KEY
        assert draft._revision == 3
        assert draft._output is outputs[-1]
    finally:
        release_retirement.set()
        await asyncio.gather(old_start, new_start, return_exceptions=True)
        await draft.close()


@pytest.mark.asyncio
async def test_cancel_during_pending_start_prevents_stale_installation() -> None:
    previous_key = VoiceDraftKey(FISH_ID, SESSION_ID, UUID(int=98))
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=FakeSink(),
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(previous_key, revision=9)
    retirement_started = asyncio.Event()
    release_retirement = asyncio.Event()
    original_revoke = outputs[0].sink.revoke

    async def blocked_revoke() -> None:
        retirement_started.set()
        await release_retirement.wait()
        await original_revoke()

    outputs[0].sink.revoke = blocked_revoke  # type: ignore[method-assign]
    pending = asyncio.create_task(draft.start(KEY, revision=1))
    await retirement_started.wait()

    await draft.cancel(KEY, through_revision=1)
    release_retirement.set()

    assert await pending is False
    assert draft._key is None
    assert draft._output is None
    assert len(outputs) == 1
    await draft.close()


@pytest.mark.asyncio
async def test_start_task_cancel_retains_promoted_sink_cleanup_exactly_once() -> None:
    previous_key = KEY

    class BlockingLiveSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.append_started = asyncio.Event()
            self.release_append = asyncio.Event()
            self.queued: list[bytes] = []

        def reset(self) -> None:
            super().reset()
            self.queued.clear()

        async def append(
            self,
            pcm: bytes,
            *,
            start_offset: int,
            end_offset: int,
        ) -> None:
            self.calls.append(("append", (pcm, start_offset, end_offset)))
            if pcm == b"pcm!":
                self.append_started.set()
                await self.release_append.wait()
            self.queued.append(pcm)

    live = BlockingLiveSink()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(previous_key, revision=9)
    await draft.accept_event(draft_snapshot("第一段。", revision=9, flush_offset=4))
    await outputs[0].sink.append(b"seed", start_offset=0, end_offset=4)
    assert (
        await draft.promote(
            previous_key,
            revision=9,
            content="第一段。",
            flush_offset=4,
        )
        is not None
    )
    forwarded = asyncio.create_task(
        outputs[0].sink.append(b"pcm!", start_offset=0, end_offset=4)
    )
    await live.append_started.wait()

    start_task = asyncio.create_task(draft.start(KEY, revision=10))
    await asyncio.sleep(0)
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    live.release_append.set()
    await forwarded
    for _ in range(100):
        if ("close_transport", None) in outputs[0].calls:
            break
        await asyncio.sleep(0.001)

    assert live.queued == []
    assert outputs[0].calls.count(("interrupt", None)) == 1
    assert outputs[0].calls.count(("close_transport", None)) == 1
    assert draft._pending_key is None
    assert draft._key is None
    await draft.close()


@pytest.mark.asyncio
async def test_close_claims_and_drains_pending_readiness_resource_once() -> None:
    class BlockingReadinessOutput(FakeDraftOutput):
        def __init__(self, sink: PromotablePcmSink) -> None:
            super().__init__(sink, emit_pcm=False)
            self.readiness_started = asyncio.Event()

        async def start(self) -> None:
            self.calls.append(("start", None))
            await self.sink.start()
            self.readiness_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.calls.append(("readiness_cancelled", None))
                raise

    outputs: list[BlockingReadinessOutput] = []

    def output_factory(sink: PromotablePcmSink) -> BlockingReadinessOutput:
        output = BlockingReadinessOutput(sink)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=FakeSink(),
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    start_task = asyncio.create_task(draft.start(KEY, revision=1))
    for _ in range(100):
        if outputs and outputs[0].readiness_started.is_set():
            break
        await asyncio.sleep(0.001)
    assert outputs and outputs[0].readiness_started.is_set()

    await draft.close()

    assert start_task.done() is True
    assert await start_task is False
    assert outputs[0].calls.count(("readiness_cancelled", None)) == 1
    assert outputs[0].calls.count(("interrupt", None)) == 1
    assert outputs[0].calls.count(("close_transport", None)) == 1
    assert draft._pending_key is None
    assert draft._key is None
    assert draft._cleanup_tasks == set()


@pytest.mark.asyncio
async def test_start_task_cancel_claims_pending_readiness_resource_once() -> None:
    class BlockingReadinessOutput(FakeDraftOutput):
        def __init__(self, sink: PromotablePcmSink) -> None:
            super().__init__(sink, emit_pcm=False)
            self.readiness_started = asyncio.Event()

        async def start(self) -> None:
            self.calls.append(("start", None))
            await self.sink.start()
            self.readiness_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.calls.append(("readiness_cancelled", None))
                raise

    outputs: list[BlockingReadinessOutput] = []

    def output_factory(sink: PromotablePcmSink) -> BlockingReadinessOutput:
        output = BlockingReadinessOutput(sink)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=FakeSink(),
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    start_task = asyncio.create_task(draft.start(KEY, revision=1))
    for _ in range(100):
        if outputs and outputs[0].readiness_started.is_set():
            break
        await asyncio.sleep(0.001)
    assert outputs and outputs[0].readiness_started.is_set()

    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task
    for _ in range(100):
        if ("close_transport", None) in outputs[0].calls:
            break
        await asyncio.sleep(0.001)

    assert outputs[0].calls.count(("readiness_cancelled", None)) == 1
    assert outputs[0].calls.count(("interrupt", None)) == 1
    assert outputs[0].calls.count(("close_transport", None)) == 1
    assert draft._pending_key is None
    assert draft._pending_resource is None
    assert draft._key is None
    await draft.close()


@pytest.mark.asyncio
async def test_newer_start_progresses_while_canceled_retirement_is_tracked() -> None:
    previous_key = VoiceDraftKey(FISH_ID, SESSION_ID, UUID(int=96))
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=FakeSink(),
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(previous_key, revision=9)
    retirement_started = asyncio.Event()
    release_retirement = asyncio.Event()
    original_revoke = outputs[0].sink.revoke

    async def blocked_revoke() -> None:
        retirement_started.set()
        await release_retirement.wait()
        await original_revoke()

    outputs[0].sink.revoke = blocked_revoke  # type: ignore[method-assign]
    canceled_start = asyncio.create_task(draft.start(KEY, revision=1))
    await retirement_started.wait()
    canceled_start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await canceled_start

    assert any(not task.done() for task in draft._cleanup_tasks)
    assert await draft.start(KEY, revision=3) is True
    assert draft._key == KEY
    assert draft._revision == 3
    assert draft._output is outputs[-1]

    release_retirement.set()
    for _ in range(100):
        if not draft._cleanup_tasks:
            break
        await asyncio.sleep(0.001)
    assert draft._key == KEY
    assert draft._revision == 3
    assert outputs[0].calls.count(("interrupt", None)) == 1
    assert outputs[0].calls.count(("close_transport", None)) == 1
    await draft.close()


@pytest.mark.asyncio
async def test_new_promotion_waits_for_old_live_sink_retirement_barrier() -> None:
    class BlockingLiveSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.old_append_started = asyncio.Event()
            self.release_old_append = asyncio.Event()
            self.queued: list[bytes] = []

        def reset(self) -> None:
            super().reset()
            self.queued.clear()

        async def append(
            self,
            pcm: bytes,
            *,
            start_offset: int,
            end_offset: int,
        ) -> None:
            self.calls.append(("append", (pcm, start_offset, end_offset)))
            if pcm == b"old!":
                self.old_append_started.set()
                await self.release_old_append.wait()
            self.queued.append(pcm)

    live = BlockingLiveSink()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=1)
    await draft.accept_event(draft_snapshot("第一段。", revision=1, flush_offset=4))
    await outputs[0].sink.append(b"seed", start_offset=0, end_offset=4)
    assert (
        await draft.promote(
            KEY,
            revision=1,
            content="第一段。",
            flush_offset=4,
        )
        is not None
    )
    old_append = asyncio.create_task(
        outputs[0].sink.append(b"old!", start_offset=0, end_offset=4)
    )
    await live.old_append_started.wait()

    replacement = asyncio.create_task(draft.start(KEY, revision=2))
    await asyncio.sleep(0)
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement
    assert await draft.start(KEY, revision=3)
    await draft.accept_event(draft_snapshot("新回答。", revision=3, flush_offset=4))
    await outputs[-1].sink.append(b"new!", start_offset=0, end_offset=4)
    reset_count = live.calls.count(("reset", None))

    promotion = asyncio.create_task(
        draft.promote(
            KEY,
            revision=3,
            content="新回答。",
            flush_offset=4,
        )
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert promotion.done() is False
    assert live.calls.count(("reset", None)) == reset_count
    assert not [call for call in live.calls if call == ("append", (b"new!", 0, 4))]

    live.release_old_append.set()
    await old_append
    result = await promotion

    assert result is not None
    assert live.queued == [b"new!"]
    assert live.calls[-1] == ("append", (b"new!", 0, 4))
    await draft.close()


@pytest.mark.asyncio
async def test_cancel_invalidates_promotion_waiting_on_live_retirement() -> None:
    class BlockingLiveSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.old_append_started = asyncio.Event()
            self.release_old_append = asyncio.Event()

        async def append(
            self,
            pcm: bytes,
            *,
            start_offset: int,
            end_offset: int,
        ) -> None:
            self.calls.append(("append", (pcm, start_offset, end_offset)))
            if pcm == b"old!":
                self.old_append_started.set()
                await self.release_old_append.wait()

    live = BlockingLiveSink()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=1)
    await draft.accept_event(draft_snapshot("第一段。", revision=1, flush_offset=4))
    await outputs[0].sink.append(b"seed", start_offset=0, end_offset=4)
    assert (
        await draft.promote(
            KEY,
            revision=1,
            content="第一段。",
            flush_offset=4,
        )
        is not None
    )
    old_append = asyncio.create_task(
        outputs[0].sink.append(b"old!", start_offset=0, end_offset=4)
    )
    await live.old_append_started.wait()
    replacement = asyncio.create_task(draft.start(KEY, revision=2))
    await asyncio.sleep(0)
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement
    assert await draft.start(KEY, revision=3)
    await draft.accept_event(draft_snapshot("新回答。", revision=3, flush_offset=4))
    await outputs[-1].sink.append(b"new!", start_offset=0, end_offset=4)

    promotion = asyncio.create_task(
        draft.promote(
            KEY,
            revision=3,
            content="新回答。",
            flush_offset=4,
        )
    )
    await asyncio.sleep(0)
    cancellation = asyncio.create_task(draft.cancel(KEY, through_revision=3))
    for _ in range(50):
        if cancellation.done():
            break
        await asyncio.sleep(0.001)

    assert cancellation.done() is True
    assert await cancellation is None
    assert promotion.done() is False
    assert not [call for call in live.calls if call == ("append", (b"new!", 0, 4))]

    live.release_old_append.set()
    await old_append
    assert await promotion is None
    assert not [call for call in live.calls if call == ("append", (b"new!", 0, 4))]
    await draft.close()


@pytest.mark.asyncio
async def test_never_attached_readiness_cleanup_does_not_block_promotion() -> None:
    class StaleReadinessOutput(FakeDraftOutput):
        def __init__(self, sink: PromotablePcmSink) -> None:
            super().__init__(sink, emit_pcm=False)
            self.readiness_started = asyncio.Event()
            self.interrupt_started = asyncio.Event()
            self.release_interrupt = asyncio.Event()

        async def start(self) -> None:
            self.calls.append(("start", None))
            await self.sink.start()
            self.readiness_started.set()
            await asyncio.Event().wait()

        async def interrupt(self) -> None:
            self.calls.append(("interrupt", None))
            self.interrupt_started.set()
            await self.release_interrupt.wait()

    live = FakeSink()
    stale: StaleReadinessOutput | None = None
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        nonlocal stale
        if stale is None:
            stale = StaleReadinessOutput(sink)
            outputs.append(stale)
            return stale
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    stale_start = asyncio.create_task(draft.start(KEY, revision=1))
    for _ in range(100):
        if stale is not None and stale.readiness_started.is_set():
            break
        await asyncio.sleep(0.001)
    assert stale is not None and stale.readiness_started.is_set()

    assert await draft.start(KEY, revision=2)
    assert await stale_start is False
    await stale.interrupt_started.wait()
    assert any(not task.done() for task in draft._cleanup_tasks)
    assert draft._live_sink_retirement_tasks == set()

    await draft.accept_event(draft_snapshot("新回答。", revision=2, flush_offset=4))
    await outputs[-1].sink.append(b"new!", start_offset=0, end_offset=4)
    promotion = await asyncio.wait_for(
        draft.promote(
            KEY,
            revision=2,
            content="新回答。",
            flush_offset=4,
        ),
        timeout=0.1,
    )

    assert promotion is not None
    assert live.calls[-1] == ("append", (b"new!", 0, 4))
    stale.release_interrupt.set()
    await draft.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("miss_kind", ["prefix", "provider"])
async def test_promotion_miss_waits_old_live_retirement_before_public_fallback(
    miss_kind: str,
) -> None:
    class BlockingLiveSink(FakeSink):
        def __init__(self) -> None:
            super().__init__()
            self.old_append_started = asyncio.Event()
            self.release_old_append = asyncio.Event()
            self.queued: list[bytes] = []

        def reset(self) -> None:
            super().reset()
            self.queued.clear()

        async def append(
            self,
            pcm: bytes,
            *,
            start_offset: int,
            end_offset: int,
        ) -> None:
            self.calls.append(("append", (pcm, start_offset, end_offset)))
            if pcm == b"old!":
                self.old_append_started.set()
                await self.release_old_append.wait()
            self.queued.append(pcm)

    live = BlockingLiveSink()
    outputs: list[FakeDraftOutput] = []

    def output_factory(sink: PromotablePcmSink) -> FakeDraftOutput:
        output = FakeDraftOutput(sink, emit_pcm=False)
        outputs.append(output)
        return output

    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=output_factory,
        load_configuration=load_configuration,
    )
    assert await draft.start(KEY, revision=1)
    await draft.accept_event(draft_snapshot("第一段。", revision=1, flush_offset=4))
    await outputs[0].sink.append(b"seed", start_offset=0, end_offset=4)
    assert (
        await draft.promote(
            KEY,
            revision=1,
            content="第一段。",
            flush_offset=4,
        )
        is not None
    )
    old_append = asyncio.create_task(
        outputs[0].sink.append(b"old!", start_offset=0, end_offset=4)
    )
    await live.old_append_started.wait()
    replacement = asyncio.create_task(draft.start(KEY, revision=2))
    await asyncio.sleep(0)
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement
    assert await draft.start(KEY, revision=3)
    await draft.accept_event(draft_snapshot("新回答。", revision=3, flush_offset=4))
    if miss_kind == "provider":
        outputs[-1].provider_state = SimpleNamespace(
            ready=True,
            active=False,
            error_free=True,
            provider="dashscope",
            configuration=CONFIG,
        )
    public_content = "不同。" if miss_kind == "prefix" else "新回答。"

    promotion = asyncio.create_task(
        draft.promote(
            KEY,
            revision=3,
            content=public_content,
            flush_offset=4,
        )
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(asyncio.shield(promotion), timeout=0.02)
    live.release_old_append.set()
    await old_append
    assert await promotion is None

    # CompanionSession ordinary fallback starts only after every old live reset.
    await live.start()
    live.reset()
    await live.append(b"ordinary", start_offset=0, end_offset=8)
    for _ in range(20):
        await asyncio.sleep(0)
    assert live.queued == [b"ordinary"]
    await draft.close()


@pytest.mark.asyncio
async def test_transport_cleanup_failure_is_consumed_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingCleanupOutput(FakeDraftOutput):
        async def close_transport(self) -> None:
            raise RuntimeError("cleanup failed")

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    caplog.set_level("DEBUG", logger="fish_memory.companion.draft_tts")
    draft = DraftTtsSession(
        client=FakeDraftClient(),
        live_sink=FakeSink(),
        max_buffer_bytes=96_000,
        output_factory=FailingCleanupOutput,
        load_configuration=load_configuration,
    )
    try:
        assert await draft.start(KEY, revision=4)
        await draft.cancel(KEY, through_revision=4)
        for _ in range(20):
            await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

        assert unhandled == []
        assert "cleanup failed" in caplog.text
    finally:
        await draft.close()
        loop.set_exception_handler(previous_handler)
