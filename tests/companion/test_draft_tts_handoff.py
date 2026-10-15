"""A public answer takes over the same in-flight synthesis and its late PCM."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import UUID

import pytest

from fish_memory.companion.draft_tts import DraftTtsSession, PromotablePcmSink
from fish_memory.companion.playback import PlaybackArbiter

from .test_draft_tts import (
    KEY,
    FakeDraftClient,
    FakeDraftOutput,
    FakeSink,
    draft_snapshot,
    load_configuration,
)

TEXT = "第一段。"
PCM = b"\x01\x00\x02\x00"


class PausedOutput(FakeDraftOutput):
    def __init__(
        self,
        sink: PromotablePcmSink,
        *,
        block_start: bool,
        block_append: bool,
    ) -> None:
        super().__init__(sink, emit_pcm=False)
        self.start_entered = asyncio.Event()
        self.start_release = asyncio.Event()
        self.append_entered = asyncio.Event()
        self.append_release = asyncio.Event()
        self.start_cancelled = False
        self.append_cancelled = False
        if not block_start:
            self.start_release.set()
        if not block_append:
            self.append_release.set()

    async def start(self, *, text_offset: int = 0) -> None:
        assert text_offset == 0
        self.calls.append(("start", None))
        self.start_entered.set()
        try:
            await self.start_release.wait()
            await self.sink.start()
        except asyncio.CancelledError:
            self.start_cancelled = True
            raise

    async def append(self, text: str) -> None:
        self.calls.append(("append", text))
        self.append_entered.set()
        try:
            await self.append_release.wait()
        except asyncio.CancelledError:
            self.append_cancelled = True
            raise

    async def emit_late_pcm(self) -> None:
        await self.sink.append(PCM, start_offset=0, end_offset=len(TEXT))


@asynccontextmanager
async def handoff_case(
    *, block_start: bool = False, block_append: bool = False
) -> AsyncIterator[SimpleNamespace]:
    client = FakeDraftClient()
    live = FakeSink()
    arbiter = PlaybackArbiter()
    created = asyncio.Event()
    outputs: list[PausedOutput] = []
    tasks: list[asyncio.Task[object]] = []

    def factory(sink: PromotablePcmSink) -> PausedOutput:
        output = PausedOutput(sink, block_start=block_start, block_append=block_append)
        outputs.append(output)
        created.set()
        return output

    draft = DraftTtsSession(
        client=client,
        live_sink=live,
        max_buffer_bytes=96_000,
        output_factory=factory,
        load_configuration=load_configuration,
        playback_arbiter=arbiter,
    )
    try:
        yield SimpleNamespace(
            client=client,
            live=live,
            arbiter=arbiter,
            draft=draft,
            outputs=outputs,
            created=created,
            tasks=tasks,
        )
    finally:
        for output in outputs:
            output.start_release.set()
            output.append_release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await draft.close()


async def start_first_append(case: SimpleNamespace) -> PausedOutput:
    assert await case.draft.start(KEY, revision=4)
    output = case.outputs[0]
    await case.client.events.put(draft_snapshot(TEXT, flush_offset=len(TEXT)))
    await asyncio.wait_for(output.append_entered.wait(), timeout=0.5)
    return output


async def promote(case: SimpleNamespace):
    owner = await case.arbiter.acquire("answer", UUID(int=801))
    return await case.draft.promote(
        KEY,
        revision=4,
        content=TEXT,
        flush_offset=len(TEXT),
        playback_owner=owner,
    )


@pytest.mark.asyncio
async def test_zero_pcm_handoff_keeps_provider_and_plays_its_late_first_packet() -> (
    None
):
    async with handoff_case() as case:
        output = await start_first_append(case)
        assert output.sink.buffered_bytes == 0

        promotion = await promote(case)

        assert promotion is not None
        assert promotion.submitted_offset == len(TEXT)
        assert len(case.outputs) == 1
        assert output.calls.count(("start", None)) == 1
        assert output.calls.count(("append", TEXT)) == 1
        assert ("interrupt", None) not in output.calls
        assert not any(name == "append" for name, _ in case.live.calls)
        await output.emit_late_pcm()
        assert [call for call in case.live.calls if call[0] == "append"] == [
            ("append", (PCM, 0, len(TEXT)))
        ]


@pytest.mark.asyncio
async def test_handoff_keeps_the_event_readers_first_append_alive() -> None:
    async with handoff_case(block_append=True) as case:
        output = await start_first_append(case)
        promotion = asyncio.create_task(promote(case))
        case.tasks.append(promotion)
        for _ in range(3):
            await asyncio.sleep(0)

        assert not promotion.done()
        assert not output.append_cancelled
        assert ("interrupt", None) not in output.calls
        await output.emit_late_pcm()
        assert not any(name == "append" for name, _ in case.live.calls)
        output.append_release.set()
        result = await asyncio.wait_for(promotion, timeout=0.5)

        assert result is not None
        assert result.submitted_offset == len(TEXT)
        assert output.calls.count(("start", None)) == 1
        assert output.calls.count(("append", TEXT)) == 1
        assert not output.append_cancelled
        assert [call for call in case.live.calls if call[0] == "append"] == [
            ("append", (PCM, 0, len(TEXT)))
        ]


@pytest.mark.asyncio
async def test_handoff_joins_the_already_started_provider_readiness() -> None:
    async with handoff_case(block_start=True) as case:
        starting = asyncio.create_task(case.draft.start(KEY, revision=4))
        case.tasks.append(starting)
        await asyncio.wait_for(case.created.wait(), timeout=0.5)
        output = case.outputs[0]
        await asyncio.wait_for(output.start_entered.wait(), timeout=0.5)
        promotion = asyncio.create_task(promote(case))
        case.tasks.append(promotion)
        for _ in range(3):
            await asyncio.sleep(0)

        assert not promotion.done()
        assert not output.start_cancelled
        assert ("interrupt", None) not in output.calls
        output.start_release.set()
        assert await asyncio.wait_for(starting, timeout=0.5)
        result = await asyncio.wait_for(promotion, timeout=0.5)

        assert result is not None
        assert result.submitted_offset == 0
        assert len(case.outputs) == 1
        assert output.calls.count(("start", None)) == 1
        assert output.calls.count(("append", TEXT)) == 0
        await result.output.append(TEXT)
        assert output.calls.count(("append", TEXT)) == 1
        await output.emit_late_pcm()
        assert [call for call in case.live.calls if call[0] == "append"] == [
            ("append", (PCM, 0, len(TEXT)))
        ]


@pytest.mark.asyncio
async def test_public_flush_can_advance_beyond_the_private_first_segment() -> None:
    async with handoff_case() as case:
        output = await start_first_append(case)
        owner = await case.arbiter.acquire("answer", UUID(int=802))
        public_content = TEXT + "第二段。"

        promotion = await case.draft.promote(
            KEY,
            revision=4,
            content=public_content,
            flush_offset=len(public_content),
            playback_owner=owner,
        )

        assert promotion is not None
        assert promotion.submitted_offset == len(TEXT)
        assert output.calls.count(("append", TEXT)) == 1
        assert ("interrupt", None) not in output.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("buffer_first_packet", [False, True])
async def test_partial_sentence_pcm_keeps_its_range_across_handoff(
    buffer_first_packet: bool,
) -> None:
    async with handoff_case() as case:
        output = await start_first_append(case)
        if buffer_first_packet:
            await output.sink.append(PCM, start_offset=0, end_offset=2)
        promotion = await promote(case)
        assert promotion is not None
        if not buffer_first_packet:
            await output.sink.append(PCM, start_offset=0, end_offset=2)
        await output.sink.append(PCM, start_offset=2, end_offset=len(TEXT))

        assert [call for call in case.live.calls if call[0] == "append"] == [
            ("append", (PCM, 0, 2)),
            ("append", (PCM, 2, len(TEXT))),
        ]
        assert output.calls.count(("append", TEXT)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_field", ["active", "error_free"])
async def test_provider_failure_during_live_attach_falls_back_with_the_full_prefix(
    provider_field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with handoff_case() as case:
        output = await start_first_append(case)
        assert output.sink.buffered_bytes == 0
        entered, resume = asyncio.Event(), asyncio.Event()
        original_start = case.live.start

        async def blocked_start() -> None:
            entered.set()
            await resume.wait()
            await original_start()

        monkeypatch.setattr(case.live, "start", blocked_start)
        owner = await case.arbiter.acquire("answer", UUID(int=803))
        promoting = asyncio.create_task(
            case.draft.promote(
                KEY,
                revision=4,
                content=TEXT,
                flush_offset=len(TEXT),
                playback_owner=owner,
            )
        )
        case.tasks.append(promoting)
        try:
            await asyncio.wait_for(entered.wait(), timeout=0.5)
            setattr(output.provider_state, provider_field, False)
            resume.set()
            promotion = await asyncio.wait_for(promoting, timeout=0.5)
        finally:
            resume.set()

        assert promotion is None
        assert output.calls.count(("append", TEXT)) == 1
        assert output.calls.count(("interrupt", None)) == 1
        assert not any(name == "append" for name, _ in case.live.calls)
        assert case.live.played_offset == 0
        calls_after_miss = list(case.live.calls)
        await output.emit_late_pcm()
        assert case.live.calls == calls_after_miss

        # A missed handoff gives the caller no submitted offset to skip. Its
        # normal output must remain able to submit and play the complete prefix.
        monkeypatch.setattr(case.live, "start", original_start)
        assert await case.arbiter.release(owner)
        fallback_owner = await case.arbiter.acquire("answer", UUID(int=804))
        fallback_output = FakeDraftOutput(case.live)
        fallback = case.arbiter.bind_tts(fallback_owner, fallback_output)
        await fallback.start()
        await fallback.append(TEXT)
        assert fallback_output.calls.count(("append", TEXT)) == 1
        assert [call for call in case.live.calls if call[0] == "append"] == [
            ("append", (TEXT.encode(), 0, len(TEXT)))
        ]
        after_fallback = list(case.live.calls)
        await output.emit_late_pcm()
        assert case.live.calls == after_fallback


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["starting", "appending", "promoted"])
async def test_cancelled_handoff_never_plays_late_pcm_from_the_old_provider(
    phase: str,
) -> None:
    async with handoff_case(
        block_start=phase == "starting", block_append=phase == "appending"
    ) as case:
        if phase == "starting":
            starting = asyncio.create_task(case.draft.start(KEY, revision=4))
            case.tasks.append(starting)
            await asyncio.wait_for(case.created.wait(), timeout=0.5)
            output = case.outputs[0]
            await asyncio.wait_for(output.start_entered.wait(), timeout=0.5)
        else:
            output = await start_first_append(case)
        promotion = asyncio.create_task(promote(case))
        case.tasks.append(promotion)
        if phase == "promoted":
            assert await asyncio.wait_for(promotion, timeout=0.5) is not None
        else:
            for _ in range(3):
                await asyncio.sleep(0)

        await asyncio.wait_for(case.draft.cancel(KEY, through_revision=4), timeout=0.5)
        if phase != "promoted":
            assert await asyncio.wait_for(promotion, timeout=0.5) is None
        calls_after_cancel = list(case.live.calls)
        await output.emit_late_pcm()

        assert case.live.calls == calls_after_cancel
        assert not any(name == "append" for name, _ in case.live.calls)
        assert output.calls.count(("start", None)) == 1
        assert output.calls.count(("interrupt", None)) == 1
