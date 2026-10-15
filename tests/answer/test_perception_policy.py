"""Independent perception choices control both capture and Ask evidence."""

import asyncio
from datetime import UTC, datetime
from itertools import product
from typing import cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.answer.environment_context import (
    EnvironmentAnswerContextLoader,
    ImageInputCapabilityProvider,
)
from fish_memory.environment_frames import (
    EnvironmentFrameCache,
    EnvironmentPerceptionPolicy,
    EnvironmentPerceptionStateRequest,
)
from fish_memory.observations.computer_history import ComputerHistoryContextSummary
from fish_memory.observations.environment_history_service import (
    CombinedEnvironmentSummaryLoader,
)
from tests.answer.test_environment_context import Capability, SummaryLoader
from tests.test_environment_frames import FISH_ID, JPEG, SESSION_ID, FakeClock, _request


def summary(text: str) -> ComputerHistoryContextSummary:
    return ComputerHistoryContextSummary(
        summary=text,
        segment_id=UUID(int=3),
        through_event_count=1,
        ended_at=datetime.fromtimestamp(100, UTC),
        last_successful_at=datetime.fromtimestamp(100, UTC),
        source="observation",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,screen,audio", list(product([False, True], repeat=3))
)
async def test_all_switch_combinations_supply_only_enabled_evidence(
    operation, screen, audio
):
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    policy = EnvironmentPerceptionPolicy(operation, screen, audio)
    cache.set_policy(FISH_ID, policy)
    for second in range(100, 104):
        clock.now = second
        assert (
            cache.put(FISH_ID, _request(observed_at=second, frame_id=second), JPEG)
            == policy.frames_enabled
        )
    operations = SummaryLoader(summary("operation-evidence"))
    video = SummaryLoader(summary("audiovisual-evidence"))
    summaries = CombinedEnvironmentSummaryLoader(video, operations, policy=cache.policy)
    loader = EnvironmentAnswerContextLoader(
        summaries,
        cache,
        cast(ImageInputCapabilityProvider, Capability("supported")),
        now=lambda: datetime.fromtimestamp(clock.now, UTC),
    )
    result = await loader.load(
        FISH_ID,
        SESSION_ID,
        turn_id=UUID(int=9),
        environment_revision=1,
        speech_started_at=datetime.fromtimestamp(100, UTC),
        frame_snapshot_at=datetime.fromtimestamp(103, UTC),
    )
    assert result.enabled == policy.enabled
    assert bool(result.image) == policy.frames_enabled
    assert len(result.images) == (4 if screen else 0)
    assert ("operation-evidence" in result.summary_context) == operation
    assert ("audiovisual-evidence" in result.summary_context) == (screen or audio)
    assert operations.calls == ([FISH_ID] if operation else [])
    assert video.calls == ([FISH_ID] if screen or audio else [])
    assert cache.policy(FISH_ID).screen_enabled == screen


@pytest.mark.asyncio
async def test_audiovisual_has_no_disabled_operation_fallback():
    operations = SummaryLoader(summary("old private operations"))
    video = SummaryLoader(None)
    summaries = CombinedEnvironmentSummaryLoader(
        video,
        operations,
        policy=lambda _: EnvironmentPerceptionPolicy(audio_enabled=True),
    )
    assert await summaries.get_latest_context_summary(FISH_ID) is None
    assert not operations.calls


@pytest.mark.asyncio
async def test_combined_context_has_one_timestamped_operation_heading() -> None:
    value = ComputerHistoryContextSummary(
        summary="evidence",
        segment_id=UUID(int=3),
        through_event_count=1,
        ended_at=datetime.fromtimestamp(100.75, UTC),
        last_successful_at=datetime.fromtimestamp(100.75, UTC),
        source="observation",
    )
    operations = SummaryLoader(value)
    video = SummaryLoader(value)
    summaries = CombinedEnvironmentSummaryLoader(
        video,
        operations,
        policy=lambda _: EnvironmentPerceptionPolicy(True, False, True),
    )

    result = await summaries.get_latest_context_summary(FISH_ID)

    assert result is not None
    assert "## 最近的电脑操作（截至 1970-01-01T00:01:40+00:00）" in result.summary
    assert result.summary.endswith("\n\nevidence")
    assert ".750000" not in result.summary


def test_legacy_policy_migrates_to_operation_and_screen_without_audio():
    value = EnvironmentPerceptionStateRequest.model_validate({"enabled": True})
    assert value.policy() == EnvironmentPerceptionPolicy(True, True, False)


def test_old_audiovisual_request_preserves_its_screen_and_audio_inputs():
    value = EnvironmentPerceptionStateRequest.model_validate(
        {"operationEnabled": False, "screenEnabled": False, "audiovisualEnabled": True}
    )
    assert value.policy() == EnvironmentPerceptionPolicy(False, True, True)


def test_unsupported_audio_policy_preserves_screen_summaries_and_reports_unavailable():
    from fish_memory.answer.environment_context import render_perception_policy

    policy = EnvironmentPerceptionStateRequest.model_validate(
        {
            "operationEnabled": False,
            "screenEnabled": True,
            "audioEnabled": True,
            "audioSupported": False,
        }
    ).policy()
    assert policy.screen_enabled and policy.history_enabled
    assert not policy.audio_enabled
    assert render_perception_policy(policy).splitlines()[-1] == (
        "听不到电脑里的声音：这台电脑的系统不支持。"
    )
    cache = EnvironmentFrameCache()
    cache.set_policy(FISH_ID, EnvironmentPerceptionPolicy(audio_supported=False))
    assert not cache.policy(FISH_ID).audio_supported
    assert not cache.policy(FISH_ID).enabled


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"enabled": None},
        {"screenEnabled": True},
        {
            "enabled": True,
            "operationEnabled": True,
            "screenEnabled": True,
            "audioEnabled": False,
        },
        {"operationEnabled": False, "screenEnabled": True, "audioEnabled": None},
    ],
)
def test_state_requires_complete_unambiguous_policy(body):
    with pytest.raises(ValidationError):
        EnvironmentPerceptionStateRequest.model_validate(body)


def test_operation_only_fish_switch_cannot_keep_previous_capture_permission():
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(FISH_ID, EnvironmentPerceptionPolicy(screen_enabled=True))
    cache.put(FISH_ID, _request(observed_at=100), JPEG)
    other = UUID(int=800)
    cache.set_policy(other, EnvironmentPerceptionPolicy(operation_enabled=True))
    assert not cache.is_enabled(FISH_ID)
    assert cache.snapshot(FISH_ID) is None
    assert not cache.put(FISH_ID, _request(observed_at=100), JPEG)
    assert cache.is_enabled(other) and not cache.policy(other).frames_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,screen,audio", list(product([False, True], repeat=3))
)
async def test_switches_can_revoke_evidence_during_pending_summary_read(
    operation, screen, audio
):
    clock = FakeClock(100)
    cache = EnvironmentFrameCache(now=clock)
    cache.set_policy(FISH_ID, EnvironmentPerceptionPolicy(True, True, True))
    for second in range(100, 104):
        clock.now = second
        cache.put(FISH_ID, _request(observed_at=second, frame_id=second), JPEG)
    entered, release = asyncio.Event(), asyncio.Event()

    class WaitingSummary(SummaryLoader):
        async def get_latest_context_summary(self, fish_id):
            entered.set()
            await release.wait()
            return await super().get_latest_context_summary(fish_id)

    operations = WaitingSummary(summary("operation-evidence"))
    video = WaitingSummary(summary("audiovisual-evidence"))
    loader = EnvironmentAnswerContextLoader(
        CombinedEnvironmentSummaryLoader(video, operations, policy=cache.policy),
        cache,
        cast(ImageInputCapabilityProvider, Capability("supported")),
        now=lambda: datetime.fromtimestamp(clock.now, UTC),
    )
    task = asyncio.create_task(
        loader.load(
            FISH_ID,
            SESSION_ID,
            turn_id=UUID(int=10),
            environment_revision=1,
            speech_started_at=datetime.fromtimestamp(100, UTC),
            frame_snapshot_at=datetime.fromtimestamp(103, UTC),
        )
    )
    await entered.wait()
    cache.set_policy(FISH_ID, EnvironmentPerceptionPolicy(operation, screen, audio))
    release.set()
    result = await task
    assert ("operation-evidence" in result.summary_context) == operation
    assert ("audiovisual-evidence" in result.summary_context) == (screen or audio)
    assert bool(result.image) == screen
    assert len(result.images) == (4 if screen else 0)


def test_inner_life_off_keeps_senses_but_no_audiovisual_summaries() -> None:
    value = EnvironmentPerceptionStateRequest.model_validate(
        {
            "operationEnabled": False,
            "screenEnabled": True,
            "audioEnabled": True,
            "summariesEnabled": False,
        }
    )
    policy = value.policy()

    assert policy.enabled
    assert policy.frames_enabled
    assert not policy.history_enabled


def test_summaries_default_on_for_older_desktop_clients() -> None:
    value = EnvironmentPerceptionStateRequest.model_validate(
        {"operationEnabled": False, "screenEnabled": True, "audioEnabled": False}
    )

    assert value.policy().history_enabled
