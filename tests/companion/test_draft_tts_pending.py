"""Public stream lag preserves private synthesis until its prefix catches up."""

import asyncio
from uuid import UUID

import pytest

from fish_memory.companion.draft_tts import DraftTtsPromotion
from fish_memory.companion.tts import DraftTtsPending, TtsConfiguration

from .test_draft_tts import KEY, draft_snapshot, wait_for_condition
from .test_draft_tts_handoff import PCM, TEXT, handoff_case, start_first_append


@pytest.mark.asyncio
@pytest.mark.parametrize("lag", ["prefix", "flush"])
async def test_pending_public_prefix_keeps_reader_and_synthesizes_once(
    lag: str,
) -> None:
    async with handoff_case() as case:
        output = await start_first_append(case)
        owner = await case.arbiter.acquire("answer", UUID(int=901))
        public_content = TEXT[:2] if lag == "prefix" else TEXT

        result = await case.draft.promote(
            KEY,
            revision=4,
            content=public_content,
            flush_offset=0,
            playback_owner=owner,
        )

        assert isinstance(result, DraftTtsPending)
        assert case.live.calls == []
        assert ("interrupt", None) not in output.calls
        await output.emit_late_pcm()
        assert case.live.calls == []

        # An ordinary private SSE update must still be consumed after Pending.
        complete_text = TEXT + "第二段。"
        await case.client.events.put(
            draft_snapshot(complete_text, flush_offset=len(complete_text))
        )
        await wait_for_condition(lambda: case.draft._snapshot_content == complete_text)
        result = await case.draft.promote(
            KEY,
            revision=4,
            content=complete_text,
            flush_offset=len(complete_text),
            playback_owner=owner,
        )

        assert isinstance(result, DraftTtsPromotion)
        assert result.submitted_offset == len(TEXT)
        assert len(case.outputs) == 1
        assert output.calls.count(("start", None)) == 1
        assert output.calls.count(("append", TEXT)) == 1
        assert ("interrupt", None) not in output.calls
        assert [call for call in case.live.calls if call[0] == "append"] == [
            ("append", (PCM, 0, len(TEXT)))
        ]


@pytest.mark.asyncio
async def test_pending_prefix_does_not_wait_for_or_cancel_private_append() -> None:
    async with handoff_case(block_append=True) as case:
        output = await start_first_append(case)
        owner = await case.arbiter.acquire("answer", UUID(int=902))

        result = await asyncio.wait_for(
            case.draft.promote(
                KEY,
                revision=4,
                content=TEXT[:2],
                flush_offset=0,
                playback_owner=owner,
            ),
            timeout=0.5,
        )

        assert isinstance(result, DraftTtsPending)
        assert not output.append_release.is_set()
        assert not output.append_cancelled
        assert case.live.calls == []
        output.append_release.set()
        result = await case.draft.promote(
            KEY,
            revision=4,
            content=TEXT,
            flush_offset=len(TEXT),
            playback_owner=owner,
        )
        assert isinstance(result, DraftTtsPromotion)
        assert output.calls.count(("start", None)) == 1
        assert output.calls.count(("append", TEXT)) == 1
        assert ("interrupt", None) not in output.calls


@pytest.mark.asyncio
async def test_private_prefix_growth_during_append_wait_returns_pending() -> None:
    async with handoff_case(block_append=True) as case:
        output = await start_first_append(case)
        owner = await case.arbiter.acquire("answer", UUID(int=903))
        promotion = asyncio.create_task(
            case.draft.promote(
                KEY,
                revision=4,
                content=TEXT,
                flush_offset=len(TEXT),
                playback_owner=owner,
            )
        )
        case.tasks.append(promotion)
        for _ in range(3):
            await asyncio.sleep(0)
        assert not promotion.done()

        complete_text = TEXT + "后续内容"
        await case.draft.accept_event(
            draft_snapshot(complete_text, flush_offset=len(TEXT))
        )
        output.append_release.set()
        result = await asyncio.wait_for(promotion, timeout=0.5)

        assert isinstance(result, DraftTtsPending)
        assert not output.append_cancelled
        assert case.live.calls == []
        assert ("interrupt", None) not in output.calls
        # The actual reader is still active, including after the append wait.
        await case.client.events.put(
            draft_snapshot(complete_text, flush_offset=len(complete_text))
        )
        await wait_for_condition(lambda: case.draft._flush_offset == len(complete_text))
        result = await case.draft.promote(
            KEY,
            revision=4,
            content=complete_text,
            flush_offset=len(complete_text),
            playback_owner=owner,
        )
        assert isinstance(result, DraftTtsPromotion)
        assert output.calls.count(("append", TEXT)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["prefix", "config", "provider", "owner"])
async def test_pending_does_not_hide_a_real_promotion_mismatch(failure: str) -> None:
    async with handoff_case() as case:
        output = await start_first_append(case)
        owner = await case.arbiter.acquire("answer", UUID(int=904))
        result = await case.draft.promote(
            KEY,
            revision=4,
            content=TEXT[:2],
            flush_offset=0,
            playback_owner=owner,
        )
        assert isinstance(result, DraftTtsPending)
        content = TEXT[:2]
        if failure == "prefix":
            content = "不同"
        elif failure == "config":
            output.configuration = TtsConfiguration("other", "voice", 16_000)
        elif failure == "provider":
            output.provider_state.error_free = False
        else:
            assert await case.arbiter.release(owner)

        result = await case.draft.promote(
            KEY,
            revision=4,
            content=content,
            flush_offset=0,
            playback_owner=owner,
        )

        assert result is None
        await wait_for_condition(lambda: ("interrupt", None) in output.calls)
        await output.emit_late_pcm()
        assert case.live.calls == []
        assert output.calls.count(("append", TEXT)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["cancel", "draft.failed"])
async def test_pending_keeps_terminal_events_effective_and_late_pcm_private(
    terminal: str,
) -> None:
    async with handoff_case() as case:
        output = await start_first_append(case)
        owner = await case.arbiter.acquire("answer", UUID(int=905))
        result = await case.draft.promote(
            KEY,
            revision=4,
            content=TEXT[:2],
            flush_offset=0,
            playback_owner=owner,
        )
        assert isinstance(result, DraftTtsPending)

        if terminal == "cancel":
            await case.draft.cancel(KEY, through_revision=4)
        else:
            await case.client.events.put(
                {
                    "event": terminal,
                    "data": {"turnId": str(KEY.turn_id), "revision": 4},
                }
            )
        await wait_for_condition(lambda: ("interrupt", None) in output.calls)

        result = await case.draft.promote(
            KEY,
            revision=4,
            content=TEXT,
            flush_offset=len(TEXT),
            playback_owner=owner,
        )
        assert result is None
        await output.emit_late_pcm()
        assert case.live.calls == []
        assert output.calls.count(("append", TEXT)) == 1
