"""Ordered ephemeral screen frames are frozen once per voice draft revision."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.answer.context import ChatMessage, get_environment_images
from fish_memory.answer.context_loader import AnswerContextLoader
from fish_memory.answer.environment_context import (
    EnvironmentAnswerContextLoader,
    ImageInputCapabilityProvider,
)
from fish_memory.answer.voice_drafts import (
    VoiceDraftCoordinator,
    VoiceDraftRevisionConflict,
)
from fish_memory.api.models import VoiceDraftUpsertRequest
from fish_memory.companion.voice_drafts import VoiceDraftTurnController
from fish_memory.environment_frames import (
    MAX_ENVIRONMENT_FRAME_HISTORY,
    EnvironmentFrameCache,
    EnvironmentPerceptionPolicy,
)
from fish_memory.llm.client import (
    _diagnostic_messages,
    _provider_messages,
    _without_environment_images,
)
from fish_memory.performance.tracker import MessagePerformanceTracker
from tests.answer.test_environment_context import Capability, History, SummaryLoader
from tests.answer.test_voice_drafts import FakePrepared
from tests.companion.test_voice_drafts import FakeVoiceDraftClient, ImmediateDecoder
from tests.test_environment_frames import FISH_ID, JPEG, SESSION_ID, FakeClock, _request

TURN_ID = UUID(int=801)


def populate(cache: EnvironmentFrameCache, clock: FakeClock, seconds: range) -> None:
    for second in seconds:
        clock.now = second
        assert cache.put(FISH_ID, _request(observed_at=second, frame_id=second), JPEG)


def voice_lease(
    cache: EnvironmentFrameCache, *, revision: int, through: float, now: float
):
    return cache.lease_voice_frames(
        FISH_ID,
        SESSION_ID,
        TURN_ID,
        revision,
        speech_started_at=100,
        frame_snapshot_at=through,
        now=now,
        capability=True,
    )


def test_voice_frames_sample_every_second_and_include_initial_frame() -> None:
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(
        FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True, audio_enabled=True)
    )
    populate(cache, clock, range(100, 106))
    first = voice_lease(cache, revision=1, through=105, now=105)
    assert [frame.observed_at for frame in first] == [100, 101, 102, 103, 104, 105]

    populate(cache, clock, range(106, 108))
    assert voice_lease(cache, revision=1, through=105, now=107) is first
    resumed = voice_lease(cache, revision=3, through=107, now=107)
    assert [frame.observed_at for frame in resumed] == list(range(100, 108))
    # A fresh ordinary Ask retains the existing latest-single-frame behavior.
    ordinary = cache.lease_frame(
        FISH_ID, SESSION_ID, "ordinary", now=107, capability=True
    )
    assert ordinary is not None and ordinary.observed_at == 107
    assert voice_lease(cache, revision=2, through=105, now=107)[-1].observed_at == 105
    with pytest.raises(ValueError, match="different capture range"):
        voice_lease(cache, revision=1, through=106, now=107)


def test_empty_voice_frame_bundle_cannot_change_after_it_is_frozen() -> None:
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(
        FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True, audio_enabled=True)
    )
    assert voice_lease(cache, revision=1, through=100, now=100) == ()
    populate(cache, clock, range(100, 101))
    assert voice_lease(cache, revision=1, through=100, now=100) == ()
    assert len(voice_lease(cache, revision=2, through=100, now=100)) == 1


@pytest.mark.parametrize(
    "duration,expected", [(0.8, [100]), (1.1, [100, 101]), (2.1, [100, 101, 102])]
)
def test_speech_start_and_one_second_slots_tolerate_capture_timer_offset(
    duration, expected
):
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(
        FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True, audio_enabled=True)
    )
    populate(cache, clock, range(100, 104))
    frames = cache.lease_voice_frames(
        FISH_ID,
        SESSION_ID,
        TURN_ID,
        1,
        speech_started_at=100.4,
        frame_snapshot_at=100.4 + duration,
        now=104,
        capability=True,
    )
    assert [frame.observed_at for frame in frames] == expected


def test_ring_is_bounded_and_clear_disable_or_fish_switch_removes_all_frames() -> None:
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(
        FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True, audio_enabled=True)
    )
    populate(cache, clock, range(100, 240))
    assert len(cache._history[FISH_ID]) == MAX_ENVIRONMENT_FRAME_HISTORY
    assert cache._history[FISH_ID][0].observed_at == 120
    assert voice_lease(cache, revision=1, through=239, now=239)
    cache.clear(FISH_ID)
    assert FISH_ID not in cache._history
    assert not cache._voice_leases
    populate(cache, clock, range(240, 241))
    cache.set_enabled(FISH_ID, False)
    assert cache.snapshot(FISH_ID) is None
    assert not cache._history
    cache.set_policy(
        FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True, audio_enabled=True)
    )
    populate(cache, clock, range(241, 242))
    cache.set_enabled(UUID(int=999), True)
    assert not cache.is_enabled(FISH_ID)
    assert cache.snapshot(FISH_ID) is None
    assert not cache._history
    assert not cache._voice_leases


@pytest.mark.asyncio
async def test_frozen_frames_reach_provider_in_order_after_summary_await() -> None:
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(
        FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True, audio_enabled=True)
    )
    populate(cache, clock, range(100, 106))
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingSummary(SummaryLoader):
        async def get_latest_context_summary(self, fish_id):
            entered.set()
            await release.wait()
            return None

    loader = EnvironmentAnswerContextLoader(
        WaitingSummary(None),
        cache,
        cast(ImageInputCapabilityProvider, Capability("supported")),
        now=lambda: datetime.fromtimestamp(clock.now, UTC),
    )
    context_task = asyncio.create_task(
        loader.load(
            FISH_ID,
            SESSION_ID,
            turn_id=TURN_ID,
            environment_revision=1,
            speech_started_at=datetime.fromtimestamp(100, UTC),
            frame_snapshot_at=datetime.fromtimestamp(105, UTC),
        )
    )
    await entered.wait()
    populate(cache, clock, range(106, 107))
    release.set()
    context = await context_task
    assert len(context.images) == 6
    assert context.image == context.images[-1]
    message: ChatMessage = {
        "role": "user",
        "content": "看这些变化",
        "environment_images": context.images,
        "environment_image": context.image,
    }
    assert get_environment_images(message) == context.images
    provider = _provider_messages([message])[0]["content"]
    urls = [
        part["image_url"]["url"] for part in provider if part["type"] == "image_url"
    ]
    assert urls == [image["data_url"] for image in context.images]
    labels = [
        provider[index - 1]["text"]
        for index, part in enumerate(provider)
        if part["type"] == "image_url"
    ]
    assert all(
        image["observed_at"] in label
        for image, label in zip(context.images, labels, strict=True)
    )
    diagnostic = json.dumps(_diagnostic_messages([message]))
    assert "data:image" not in diagnostic
    assert "environmentImages" in diagnostic
    assert _without_environment_images([message]) == [
        {"role": "user", "content": "看这些变化"}
    ]


@pytest.mark.asyncio
async def test_draft_context_attaches_frozen_sequence_without_polluting_recall() -> (
    None
):
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(
        FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True, audio_enabled=True)
    )
    populate(cache, clock, range(100, 106))
    loader = AnswerContextLoader(
        History([]),
        now=lambda: datetime.fromtimestamp(clock.now, UTC),
        environment_context_loader=EnvironmentAnswerContextLoader(
            SummaryLoader(None),
            cache,
            cast(ImageInputCapabilityProvider, Capability("supported")),
            now=lambda: datetime.fromtimestamp(clock.now, UTC),
        ),
        system_prompt="system",
    )
    context = await loader.load_for_draft(
        FISH_ID,
        SESSION_ID,
        content="看这些变化",
        created_at=datetime.fromtimestamp(105, UTC),
        tracker=MessagePerformanceTracker(),
        turn_id=TURN_ID,
        environment_revision=1,
        speech_started_at=datetime.fromtimestamp(100, UTC),
        frame_snapshot_at=datetime.fromtimestamp(105, UTC),
    )
    assert len(get_environment_images(context.messages[-1])) == 6
    assert context.messages[-1]["environment_image"]["observed_at"] == (
        datetime.fromtimestamp(105, UTC).isoformat()
    )
    # The current user text is routed separately from Recall's prior turns.
    assert context.recall_conversation == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"frameSnapshotAt": None},
        {"speechStartedAt": "2026-09-03T12:00:00"},
        {"frameSnapshotAt": "2026-09-03T11:59:59Z"},
    ],
)
def test_voice_window_schema_requires_paired_aware_ordered_timestamps(
    changes: dict,
) -> None:
    value = {
        "revision": 1,
        "content": "问题",
        "speechStartedAt": "2026-09-03T12:00:00Z",
        "frameSnapshotAt": "2026-09-03T12:00:02Z",
    }
    with pytest.raises(ValidationError):
        VoiceDraftUpsertRequest.model_validate(value | changes)


@pytest.mark.asyncio
async def test_controller_freezes_cutoff_at_candidate_and_retains_start_on_resume() -> (
    None
):
    start = datetime(2026, 9, 3, 12, tzinfo=UTC)
    clock = [start + timedelta(seconds=2)]
    calls: list[dict] = []

    class FrameClient(FakeVoiceDraftClient):
        async def upsert_voice_draft(self, key, **kwargs):
            calls.append(kwargs)
            return True

    controller = VoiceDraftTurnController(
        client=FrameClient(), decoder=ImmediateDecoder(), now=lambda: clock[0]
    )
    owner = controller.open_turn(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        binding_generation=1,
        speech_generation=1,
        speech_started_at=start,
    )
    try:
        assert await controller.prepare(owner, pcm=b"a" * 320) is not None
        first = controller.candidate(owner, pcm=b"a" * 640)
        clock[0] = start + timedelta(seconds=3)
        assert await first is not None
        assert calls[0]["speech_started_at"] == start
        assert calls[0]["frame_snapshot_at"] == start + timedelta(seconds=2)
        resumed = controller.resume(owner)
        reopened = controller.reopen(speech_generation=2)
        assert resumed.speech_started_at == reopened.speech_started_at == start
        assert await controller.candidate(reopened, pcm=b"a" * 960) is not None
        assert calls[1]["speech_started_at"] == start
        assert calls[1]["frame_snapshot_at"] == start + timedelta(seconds=3)
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_server_passes_frozen_window_and_rejects_same_revision_change() -> None:
    from fish_memory.answer.voice_draft_types import VoiceDraftKey

    start = datetime(2026, 9, 3, 12, tzinfo=UTC)
    through = start + timedelta(seconds=4)
    called = asyncio.Event()
    metadata: list[dict] = []

    async def prepare(
        key, revision, content, *, speech_started_at=None, frame_snapshot_at=None
    ):
        metadata.append(
            {
                "speech_started_at": speech_started_at,
                "frame_snapshot_at": frame_snapshot_at,
            }
        )
        called.set()
        result = FakePrepared(revision)
        result.ready.set()
        return result

    server = VoiceDraftCoordinator(prepare=prepare)
    key = VoiceDraftKey(FISH_ID, SESSION_ID, TURN_ID)
    try:
        await server.upsert(
            key,
            revision=1,
            content="问题",
            speech_started_at=start,
            frame_snapshot_at=through,
        )
        await asyncio.wait_for(called.wait(), 1)
        assert metadata[0]["speech_started_at"] == start
        assert metadata[0]["frame_snapshot_at"] == through
        with pytest.raises(VoiceDraftRevisionConflict):
            await server.upsert(
                key,
                revision=1,
                content="问题",
                speech_started_at=start,
                frame_snapshot_at=through + timedelta(seconds=1),
            )
    finally:
        await server.close()
