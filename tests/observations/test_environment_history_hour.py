from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fish_memory.answer.environment_context import render_environment_summary_context
from fish_memory.answer.history_tokens import estimate_text_tokens
from fish_memory.observations.environment_history_audio import SystemAudioTranscript
from fish_memory.observations.environment_history_models import (
    SemanticDocument,
    SemanticSnapshot,
    snapshot_id,
)
from fish_memory.observations.environment_history_service import (
    EnvironmentHistoryService,
    _media_recap_audio,
    _render_snapshot_context,
    _select_hour_snapshots,
    _snapshot_context_blocks,
)
from fish_memory.observations.media_questions import requests_media_recap

FISH = UUID("00000000-0000-0000-0000-000000000101")
CAPTURE = UUID("00000000-0000-0000-0000-000000000102")
START = datetime(2026, 9, 5, tzinfo=UTC)


class MemoryRepository:
    def __init__(self, values: tuple[SemanticSnapshot, ...] = ()) -> None:
        self.values = {(value.fish_id, value.id): value for value in values}
        self.calls: Counter[str] = Counter()
        self.cleanup_cutoffs: list[datetime] = []

    async def get(self, fish_id, value_id):
        self.calls["get"] += 1
        return self.values.get((fish_id, value_id))

    async def save(self, value, *, evidence=None):
        del evidence
        self.calls["save"] += 1
        self.values[(value.fish_id, value.id)] = value
        return value

    async def minute_windows(self, fish_id, capture_session_id, first_index):
        self.calls["minute_windows"] += 1
        return await self.child_snapshots(
            fish_id,
            capture_session_id,
            first_index,
            child_period_seconds=10,
            parent_period_seconds=60,
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
        self.calls["child_snapshots"] += 1
        from fish_memory.observations.environment_history_hierarchy import (
            level_for_period,
        )

        assert (
            level_for_period(parent_period_seconds).child_period_seconds
            == child_period_seconds
        )
        child_count = parent_period_seconds // child_period_seconds
        index_step = child_period_seconds // 10
        expected = tuple(
            first_index + offset * index_step for offset in range(child_count)
        )
        children = tuple(
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
        if tuple(value.window_index for value in children) != expected:
            return ()
        if any(
            left.ended_at != right.started_at
            for left, right in zip(children[:-1], children[1:], strict=True)
        ):
            return ()
        return children

    async def recent_hour_snapshots(self, fish_id, *, since, through):
        self.calls["recent_hour_snapshots"] += 1
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
        del fish_id
        self.calls["delete_compacted_windows"] += 1
        self.cleanup_cutoffs.append(ended_before)
        return 0

    async def context_snapshots(self, fish_id):
        del fish_id
        self.calls["context_snapshots"] += 1
        raise AssertionError("Ask must read the already-published hot view")

    async def pending_minute_windows(
        self,
        fish_id,
        limit=4,
        *,
        since=None,
        through=None,
    ):
        self.calls["pending_minute_windows"] += 1
        groups: dict[tuple[UUID, int], list[SemanticSnapshot]] = {}
        for value in self.values.values():
            if (
                value.fish_id != fish_id
                or value.period_seconds != 10
                or (since is not None and value.started_at < since)
                or (through is not None and value.ended_at > through)
            ):
                continue
            group = (value.capture_session_id, (value.window_index // 6) * 6)
            groups.setdefault(group, []).append(value)
        pending = []
        for (capture_session_id, first_index), values in groups.items():
            ordered = sorted(values, key=lambda value: value.window_index)
            if [value.window_index for value in ordered] != list(
                range(first_index, first_index + 6)
            ):
                continue
            parent_id = snapshot_id(FISH, capture_session_id, first_index, 60)
            if (FISH, parent_id) not in self.values:
                pending.append(ordered[-1])
        return tuple(sorted(pending, key=lambda value: value.window_index)[:limit])


class RecordingSummarizer:
    def __init__(self, *, fail_period: int | None = None) -> None:
        self.fail_period = fail_period
        self.window_calls: list[int] = []
        self.merge_calls: dict[int, list[list[int]]] = {
            60: [],
            300: [],
            1800: [],
        }
        self.merge_contexts: dict[int, list[str | None]] = {60: [], 300: [], 1800: []}

    async def summarize_window(self, window, transcript):
        del transcript
        self.window_calls.append(window.window_index)
        return SemanticDocument(summary=f"ten-{window.window_index}")

    async def merge_snapshots(
        self, windows, *, parent_period_seconds, previous_document=None
    ):
        self.merge_contexts.setdefault(parent_period_seconds, []).append(
            previous_document.summary if previous_document is not None else None
        )
        prefix = {60: "minute", 300: "five", 1800: "half"}.get(
            parent_period_seconds, str(parent_period_seconds)
        )
        return self._merge(windows, parent_period_seconds, prefix)

    def _merge(self, windows, period_seconds, prefix):
        indexes = [value.window_index for value in windows]
        self.merge_calls.setdefault(period_seconds, []).append(indexes)
        if self.fail_period == period_seconds:
            raise ValueError(f"synthetic {period_seconds}s failure")
        return SemanticDocument(summary=f"{prefix}-{indexes[0]}")

    async def close(self):
        pass


class SilentTranscriber:
    async def transcribe(self, pcm):
        del pcm
        return SystemAudioTranscript("", "unknown")

    async def close(self):
        pass


def semantic_snapshot(
    period_seconds: int,
    window_index: int,
    summary: str,
    *,
    created_at: datetime | None = None,
) -> SemanticSnapshot:
    started_at = START + timedelta(seconds=window_index * 10)
    ended_at = started_at + timedelta(seconds=period_seconds)
    return SemanticSnapshot(
        id=snapshot_id(FISH, CAPTURE, window_index, period_seconds),
        fish_id=FISH,
        capture_session_id=CAPTURE,
        window_index=window_index,
        period_seconds=period_seconds,
        started_at=started_at,
        ended_at=ended_at,
        document=SemanticDocument(summary=summary),
        input_digest=f"digest-{period_seconds}-{window_index}",
        created_at=created_at or ended_at,
    )


async def start_service(
    *,
    repository: MemoryRepository | None = None,
    summarizer: RecordingSummarizer | None = None,
    now: datetime,
) -> tuple[EnvironmentHistoryService, MemoryRepository, RecordingSummarizer]:
    repository = repository or MemoryRepository()
    summarizer = summarizer or RecordingSummarizer()
    service = EnvironmentHistoryService(
        repository,
        summarizer,
        SilentTranscriber(),
        now=lambda: now,
        concurrency=4,
        max_pending=256,
        retry_delays=(),
    )
    service.set_enabled(FISH, True)
    await service._recovery_tasks[FISH]
    return service, repository, summarizer


async def settle(service: EnvironmentHistoryService) -> None:
    for _ in range(300):
        tasks = [task for _, task in service._jobs.values()]
        if not tasks:
            await asyncio.sleep(0)
            if not service._jobs:
                return
            continue
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)
    raise AssertionError("environment history hierarchy did not settle")


async def seed_ten_second_windows(
    service: EnvironmentHistoryService,
    repository: MemoryRepository,
    count: int,
) -> None:
    leaves = [semantic_snapshot(10, index, f"ten-{index}") for index in range(count)]
    for value in leaves:
        await repository.save(value)
        service._remember_snapshot(value, publish=False)
    await settle(service)


def test_rendered_history_keeps_categories_and_legacy_separate_without_empty_sections():
    story = "奶奶发现借款，孙子解释钱用于修车，两人约定下次先商量。"
    activity = "用户在编辑器修改文件，然后切换浏览器。"
    current = replace(
        semantic_snapshot(300, 0, "unused"),
        document=SemanticDocument(audiovisual_summary=story, activity_summary=activity),
    )
    rendered = _render_snapshot_context((current,))
    assert f"影音内容：\n- {story}" in rendered
    assert f"电脑活动：\n- {activity}" in rendered
    assert rendered.count(story) == rendered.count(activity) == 1
    assert "未分类内容" not in rendered
    assert "audiovisualSummary" not in rendered

    legacy = semantic_snapshot(10, 30, "旧的混合观察，无法可靠自动分类。")
    rendered = _render_snapshot_context((current, legacy))
    assert "未分类内容：\n- 旧的混合观察，无法可靠自动分类。" in rendered
    media_only = replace(current, document=SemanticDocument(audiovisual_summary=story))
    assert "电脑活动" not in _render_snapshot_context((media_only,))


@pytest.mark.asyncio
async def test_ask_reads_hot_hour_view_without_any_repository_query():
    service, repository, _ = await start_service(now=START + timedelta(minutes=1))
    try:
        value = semantic_snapshot(60, 0, "hot minute")
        service._remember_snapshot(value)
        calls_before_asks = repository.calls.copy()

        first = await service.get_latest_context_summary(FISH)
        second = await service.get_latest_context_summary(FISH)

        assert first == second
        assert first is not None and "hot minute" in first.summary
        assert repository.calls == calls_before_asks
        assert repository.calls["context_snapshots"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_hour_view_uses_coarse_old_coverage_and_ten_second_recent_minute():
    as_of = START + timedelta(hours=1)
    service, _, _ = await start_service(now=as_of)
    try:
        # Older history has overlapping representations; only the widest valid
        # node should survive in the Ask view.
        old_half = semantic_snapshot(1800, 0, "old-half-hour")
        old_five = semantic_snapshot(300, 0, "old-five-minute")
        old_minute = semantic_snapshot(60, 0, "old-minute")
        for value in (old_half, old_five, old_minute):
            service._remember_snapshot(value, publish=False)

        # Overlapping coarse nodes must not hide the last six scene/dialogue
        # windows, even when every level of compaction has already completed.
        for first_index in (300, 330):
            service._remember_snapshot(
                semantic_snapshot(300, first_index, f"recent-five-{first_index}"),
                publish=False,
            )
        for index in range(354, 360):
            service._remember_snapshot(
                semantic_snapshot(10, index, f"recent-ten-{index}"),
                publish=False,
            )
        for first_index in range(300, 360, 6):
            service._remember_snapshot(
                semantic_snapshot(
                    60,
                    first_index,
                    f"recent-minute-{first_index}",
                ),
                publish=False,
            )
        service._publish_context_view(FISH, as_of=as_of)

        context = await service.get_latest_context_summary(FISH)

        assert context is not None
        assert "old-half-hour" in context.summary
        assert "old-five-minute" not in context.summary
        assert "old-minute" not in context.summary
        assert "recent-five-300" in context.summary
        assert "recent-five-330" not in context.summary
        assert all(
            f"recent-minute-{index}" in context.summary for index in range(330, 354, 6)
        )
        assert "recent-minute-354" not in context.summary
        assert all(
            f"recent-ten-{index}" in context.summary for index in range(354, 360)
        )
    finally:
        await service.close()


def test_44_minute_pyramid_matches_chronological_time_groups():
    offset = timedelta(hours=10)
    values = []
    for period, step in ((10, 1), (60, 6), (300, 30), (1800, 180)):
        for index in range(0, 264, step):
            if index * 10 + period > 2640:
                continue
            value = semantic_snapshot(period, index, f"p{period}-{index}")
            values.append(
                replace(
                    value,
                    started_at=value.started_at + offset,
                    ended_at=value.ended_at + offset,
                )
            )

    selected = _select_hour_snapshots(
        tuple(reversed(values)), as_of=START + offset + timedelta(minutes=44)
    )

    assert [(value.period_seconds, value.window_index) for value in selected] == [
        (1800, 0),
        (300, 180),
        (300, 210),
        (60, 240),
        (60, 246),
        (60, 252),
        *((10, index) for index in range(258, 264)),
    ]
    rendered = _render_snapshot_context(selected)
    assert rendered.replace("\n\n未分类内容：", "").split("\n\n")[1:] == [
        "10:00:00 — 10:30:00｜30min 粒度\n- p1800-0",
        "10:30:00 — 10:40:00｜5min 粒度\n- p300-180\n- p300-210",
        "10:40:00 — 10:43:00｜1min 粒度\n- p60-240\n- p60-246\n- p60-252",
        "10:43:00 — 10:44:00｜10s 粒度\n"
        + "\n".join(f"- p10-{index}" for index in range(258, 264)),
    ]
    assert "…" not in rendered


def test_pyramid_labels_actual_fallback_granularity_without_bridging_gaps():
    values = (
        semantic_snapshot(60, 0, "before-gap"),
        semantic_snapshot(60, 12, "after-gap"),
        semantic_snapshot(60, 18, "latest-parent"),
        semantic_snapshot(10, 18, "latest-child"),
        # Other recent children have not completed: do not replace the known
        # child with its broad parent or imply the missing detail is available.
    )
    selected = _select_hour_snapshots(values, as_of=START + timedelta(minutes=4))
    assert [value.document.summary for value in selected] == [
        "before-gap",
        "after-gap",
        "latest-child",
    ]
    rendered = _render_snapshot_context(selected)
    assert rendered.replace("\n\n未分类内容：", "").split("\n\n")[1:] == [
        "00:00:00 — 00:01:00｜1min 粒度\n- before-gap",
        "00:02:00 — 00:03:00｜1min 粒度\n- after-gap",
        "00:03:00 — 00:03:10｜10s 粒度\n- latest-child",
    ]


def test_pyramid_time_range_remains_unambiguous_across_midnight():
    values = tuple(
        semantic_snapshot(10, index, f"window-{index}") for index in (8639, 8640)
    )
    rendered = _render_snapshot_context(values)
    assert "23:59:50 — 2026-09-06 00:00:10｜10s 粒度" in rendered


def test_hour_boundary_keeps_complete_half_hour_and_more_than_one_hour():
    as_of = START + timedelta(minutes=63)
    values = tuple(
        semantic_snapshot(period, index, f"p{period}-{index}")
        for period, step in ((10, 1), (60, 6), (300, 30), (1800, 180))
        for index in range(0, 378, step)
        if index * 10 + period <= 3780
    )

    selected = _select_hour_snapshots(values, as_of=as_of)

    assert selected[0].period_seconds == 1800
    assert selected[0].started_at == START
    assert selected[-1].ended_at == as_of
    assert selected[-1].ended_at - selected[0].started_at > timedelta(hours=1)
    periods = [value.period_seconds for value in selected]
    assert periods == sorted(periods, reverse=True)
    assert sum(value.period_seconds for value in selected) == 63 * 60


def test_capture_restart_keeps_coarse_parents_and_groups_the_earlier_partial_tail():
    # A partial old capture must not discard the next capture's ready parent.
    second_capture = UUID(int=999)
    values = [
        semantic_snapshot(1800, 0, "older half hour"),
        semantic_snapshot(60, 180, "first capture tail"),
    ]
    for period, indexes in (
        (300, [186]),
        (60, range(186, 216, 6)),
        (10, range(216, 222)),
    ):
        for index in indexes:
            values.append(
                replace(
                    semantic_snapshot(period, index, f"p{period}-{index}"),
                    capture_session_id=second_capture,
                )
            )

    selected = _select_hour_snapshots(values, as_of=START + timedelta(minutes=37))

    assert [value.period_seconds for value in selected] == [
        1800,
        60,
        300,
        *([10] * 6),
    ]
    assert all(
        left.ended_at == right.started_at
        for left, right in zip(selected[:-1], selected[1:], strict=True)
    )
    assert sum(value.period_seconds for value in selected) == 37 * 60
    blocks = _snapshot_context_blocks(selected)
    assert [block.period_seconds for block in blocks] == [
        1800,
        300,
        300,
        *([10] * 6),
    ]
    assert blocks[1].snapshots == (values[1],)
    rendered = _render_snapshot_context(selected)
    assert "00:30:00 — 00:31:00｜5min 粒度" in rendered
    assert "（00:30:00 — 00:31:00，未满5分钟）first capture tail" in rendered
    assert "00:31:00 — 00:36:00｜5min 粒度\n\n未分类内容：\n- p300-186" in rendered


def test_cutoff_in_restart_gap_still_starts_with_the_complete_half_hour():
    values = [semantic_snapshot(1800, 0, "oldest complete half hour")]
    values.extend(
        semantic_snapshot(60, index, f"old tail {index}")
        for index in range(180, 204, 6)
    )
    values.append(semantic_snapshot(1800, 240, "new coarse parent"))
    values.extend(
        semantic_snapshot(60, index, f"new minute {index}")
        for index in range(240, 564, 6)
    )
    values.extend(
        semantic_snapshot(10, index, f"latest detail {index}")
        for index in range(564, 570)
    )

    selected = _select_hour_snapshots(values, as_of=START + timedelta(minutes=95))

    assert selected[0].period_seconds == 1800
    assert selected[0].started_at == START
    assert selected[-1].ended_at == START + timedelta(minutes=95)
    blocks = _snapshot_context_blocks(selected)
    assert [block.period_seconds for block in blocks] == sorted(
        (block.period_seconds for block in blocks), reverse=True
    )
    assert "new coarse parent" in [value.document.summary for value in selected]
    assert all(
        f"new minute {index}" in [value.document.summary for value in selected]
        for index in range(420, 564, 6)
    )
    assert sum(value.period_seconds for value in selected) == (95 - 6) * 60
    assert "00:30:00 — 00:34:00，未满30分钟" in _render_snapshot_context(selected)


def test_fifteen_minutes_after_restart_uses_existing_five_minute_and_minute_nodes():
    old_start = START + timedelta(hours=9, minutes=49, seconds=10)
    new_start = START + timedelta(hours=10, minutes=39, seconds=38)
    second_capture = UUID(int=999)
    values = []
    for capture, origin, layers in (
        (CAPTURE, old_start, ((300, [0]), (60, [30]), (10, range(36, 39)))),
        (
            second_capture,
            new_start,
            ((300, [0, 30]), (60, range(0, 90, 6)), (10, range(90))),
        ),
    ):
        for period, indexes in layers:
            for index in indexes:
                source = semantic_snapshot(period, index, f"{capture}-{period}-{index}")
                values.append(
                    replace(
                        source,
                        id=snapshot_id(FISH, capture, index, period),
                        capture_session_id=capture,
                        started_at=origin + timedelta(seconds=index * 10),
                        ended_at=origin + timedelta(seconds=index * 10 + period),
                    )
                )

    as_of = new_start + timedelta(minutes=15)
    selected = _select_hour_snapshots(tuple(reversed(values)), as_of=as_of)
    assert Counter(
        value.period_seconds
        for value in selected
        if value.capture_session_id == second_capture
    ) == {300: 2, 60: 4, 10: 6}
    assert selected[-1].ended_at - selected[0].started_at > timedelta(hours=1)
    blocks = _snapshot_context_blocks(selected)
    assert [block.period_seconds for block in blocks] == [
        *([300] * 4),
        *([60] * 4),
        *([10] * 6),
    ]
    # View grouping never changes source identities, durations, or contents.
    assert tuple(source for block in blocks for source in block.snapshots) == selected
    rendered = _render_snapshot_context(selected)
    assert "09:49:10 — 09:55:40｜5min 粒度" in rendered
    assert "09:54:10 — 09:55:40，未满5分钟" in rendered
    assert "10:39:38 — 10:49:38｜5min 粒度" in rendered
    assert "10:49:38 — 10:53:38｜1min 粒度" in rendered
    assert "10:53:38 — 10:54:38｜10s 粒度" in rendered
    for value in selected:
        assert rendered.count(value.document.summary) == 1


def test_missing_parent_groups_earlier_children_without_losing_text_or_filling_gaps():
    values = (
        semantic_snapshot(10, 0, "before gap"),
        semantic_snapshot(10, 2, "after gap\ncomplete ending"),
        semantic_snapshot(300, 30, "completed five minutes"),
        semantic_snapshot(10, 60, "latest detail"),
    )
    selected = _select_hour_snapshots(
        values, as_of=START + timedelta(minutes=10, seconds=10)
    )
    blocks = _snapshot_context_blocks(selected)
    assert [block.period_seconds for block in blocks] == [300, 300, 300, 10]
    assert tuple(source for block in blocks for source in block.snapshots) == values
    rendered = _render_snapshot_context(selected)
    assert "00:00:00 — 00:00:10｜5min 粒度" in rendered
    assert "00:00:20 — 00:00:30｜5min 粒度" in rendered
    assert "after gap\n  complete ending" in rendered
    assert "completed five minutes" in rendered


@pytest.mark.asyncio
async def test_restart_restores_a_parent_starting_before_the_hydration_boundary():
    now = START + timedelta(minutes=88)
    parent = semantic_snapshot(1800, 0, "complete oldest half hour")
    latest = semantic_snapshot(10, 527, "latest scene")
    service, _, _ = await start_service(
        repository=MemoryRepository((parent, latest)),
        now=now,
    )
    try:
        context = await service.get_latest_context_summary(FISH)
        assert context is not None
        assert "00:00:00 — 00:30:00｜30min 粒度" in context.summary
        assert "complete oldest half hour" in context.summary
        assert "latest scene" in context.summary
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_active_hour_anchors_to_completed_window_instead_of_dropping_buckets():
    service, _, _ = await start_service(now=START + timedelta(hours=1, seconds=1))
    try:
        for period_seconds, index_step in (
            (10, 1),
            (60, 6),
            (300, 30),
            (1800, 180),
        ):
            for index in range(0, 360, index_step):
                service._remember_snapshot(
                    semantic_snapshot(
                        period_seconds,
                        index,
                        f"p{period_seconds}-{index}",
                    ),
                    publish=False,
                )

        context = await service.get_latest_context_summary(FISH)

        assert context is not None
        assert context.through_event_count == 16
        assert "p1800-0" in context.summary
        assert all(
            f"p300-{index}" in context.summary for index in (180, 210, 240, 270, 300)
        )
        assert all(f"p60-{index}" in context.summary for index in range(330, 354, 6))
        assert all(f"p10-{index}" in context.summary for index in range(354, 360))
        assert "p1800-180" not in context.summary
        assert "p300-330" not in context.summary
        assert "p60-354" not in context.summary
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_quiet_source_reapplies_wall_clock_hour_boundary_after_anchor_grace():
    repository = MemoryRepository()
    summarizer = RecordingSummarizer()
    now = {"value": START + timedelta(hours=1, seconds=1)}
    service = EnvironmentHistoryService(
        repository,
        summarizer,
        SilentTranscriber(),
        now=lambda: now["value"],
        retry_delays=(),
    )
    service.set_enabled(FISH, True)
    await service._recovery_tasks[FISH]
    try:
        service._remember_snapshot(semantic_snapshot(10, 0, "old-edge"))
        service._remember_snapshot(semantic_snapshot(10, 359, "latest-edge"))
        anchored = await service.get_latest_context_summary(FISH)
        assert anchored is not None and "old-edge" in anchored.summary

        now["value"] = START + timedelta(hours=1, seconds=21)
        wall_clock = await service.get_latest_context_summary(FISH)

        assert wall_clock is not None
        # A complete head may now extend before the nominal cutoff.
        assert "old-edge" in wall_clock.summary
        assert "latest-edge" in wall_clock.summary
        now["value"] = START + timedelta(minutes=71)
        expired = await service.get_latest_context_summary(FISH)
        assert expired is not None
        assert "old-edge" not in expired.summary
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_hour_view_keeps_complete_fine_history_when_compaction_is_unavailable():
    as_of = START + timedelta(hours=1)
    service, _, _ = await start_service(now=as_of)
    try:
        # A total minute-compaction outage leaves 360 ten-second nodes. The hot
        # view is bounded by time; every completed window remains available.
        for index in range(360):
            service._remember_snapshot(
                semantic_snapshot(
                    10,
                    index,
                    f"window-{index}-" + '\\\\"日志字段\\\\"' * 100,
                ),
                publish=False,
            )
        service._publish_context_view(FISH, as_of=as_of)

        context = await service.get_latest_context_summary(FISH)

        assert context is not None
        assert context.through_event_count == 360
        assert context.summary.count("\n- ") == 360
        assert estimate_text_tokens(context.summary) > 2_800
        assert estimate_text_tokens(render_environment_summary_context(context)) > 2_800
        assert context.summary.count("2026-09-05T") == 2
        assert all(f"window-{index}-" in context.summary for index in range(354, 360))
        assert "…" not in context.summary
        for index in range(360):
            assert f"window-{index}-" + '\\\\"日志字段\\\\"' * 100 in context.summary
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_startup_hydrates_recent_snapshots_then_serves_them_from_memory():
    as_of = START + timedelta(hours=1)
    persisted = semantic_snapshot(10, 359, "persisted latest scene")
    repository = MemoryRepository((persisted,))
    service, _, _ = await start_service(repository=repository, now=as_of)
    try:
        calls_after_hydration = repository.calls.copy()

        context = await service.get_latest_context_summary(FISH)

        assert context is not None
        assert "persisted latest scene" in context.summary
        assert repository.calls == calls_after_hydration
        assert repository.calls["recent_hour_snapshots"] == 1
        assert repository.calls["context_snapshots"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_complete_hierarchy_restart_does_not_requery_existing_parent_ranges():
    persisted = tuple(
        semantic_snapshot(period_seconds, index, f"p{period_seconds}-{index}")
        for period_seconds, index_step in (
            (10, 1),
            (60, 6),
            (300, 30),
            (1800, 180),
        )
        for index in range(0, 360, index_step)
    )
    repository = MemoryRepository(persisted)
    service, _, _ = await start_service(
        repository=repository,
        now=START + timedelta(hours=1),
    )
    try:
        await settle(service)

        assert repository.calls["child_snapshots"] == 0
        assert repository.calls["get"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_missing_later_speech_is_explicit_after_loading_stored_evidence():
    from fish_memory.observations.environment_history_store import _snapshot

    opening = replace(
        semantic_snapshot(10, 0, "开场画面"),
        document=SemanticDocument(summary="开场画面", audio="在这家店干了三年"),
    )
    leaves = [opening]
    for index in range(1, 6):
        value = semantic_snapshot(10, index, "后续画面")
        # Exercise legacy evidence too: no new document field/migration needed.
        leaves.append(
            _snapshot(
                {
                    "id": str(value.id),
                    "fish_id": str(value.fish_id),
                    "started_at": value.started_at,
                    "ended_at": value.ended_at,
                    "created_at": value.created_at,
                    "payload": {
                        "captureSessionId": str(CAPTURE),
                        "windowIndex": index,
                        "periodSeconds": 10,
                        "inputDigest": value.input_digest,
                        "document": value.document.model_dump(),
                        "sourceSnapshotIds": [],
                        "evidence": {
                            "audioStatus": "complete",
                            "transcriptionTextStatus": "unavailable",
                        },
                    },
                }
            )
        )
    leaves.append(semantic_snapshot(10, 6, "视频已停止，安静画面"))
    service, _, _ = await start_service(now=START + timedelta(seconds=71))
    try:
        for value in leaves:
            service._remember_snapshot(value)
        recap = await service.get_context_for_question(FISH, "刚才这集讲了什么故事？")
        assert recap is not None
        assert "在这家店干了三年" in recap.summary
        assert "听觉记录不完整：00:00:10 — 00:01:00" in recap.summary
        assert "不能代表整段经过" in recap.summary
        assert "00:00:10 — 00:01:10 期间的声音没有听清" not in recap.summary
        ordinary = await service.get_latest_context_summary(FISH)
        assert ordinary is not None
        assert "听觉记录不完整" in ordinary.summary
    finally:
        await service.close()


def test_recap_audio_recovers_earlier_words_without_duplicate_leaf_or_future_data():
    minute = replace(
        semantic_snapshot(60, 0, "incorrect old interpretation"),
        document=SemanticDocument(
            audiovisualSummary="旧概括", audio="我先取消这个订单。"
        ),
    )
    leaf = replace(
        semantic_snapshot(10, 0, "leaf"),
        document=SemanticDocument(summary="leaf", audio="我先取消这个订单。"),
    )
    ending = replace(
        semantic_snapshot(60, 24, "ending"),
        document=SemanticDocument(
            summary="ending", audio="对方已经离开，我只能另想办法。"
        ),
    )
    future = replace(
        semantic_snapshot(10, 31, "future"),
        document=SemanticDocument(summary="future", audio="尚未发生的对白"),
    )
    result = _media_recap_audio(
        (ending, future, leaf, minute), (), as_of=START + timedelta(seconds=305)
    )
    assert [value.text for value in result] == [
        minute.document.audio,
        ending.document.audio,
    ]
    assert _media_recap_audio((minute,), (), as_of=START + timedelta(minutes=12)) == ()
    bounded = _media_recap_audio(
        (minute, ending),
        (),
        as_of=START + timedelta(seconds=305),
        max_characters=len(ending.document.audio) + 80,
    )
    assert [value.text for value in bounded] == [ending.document.audio]


def test_recap_recovers_compacted_dialogue_without_promoting_media_summary_to_speech():
    first = replace(
        semantic_snapshot(10, 0, "first"),
        document=SemanticDocument(
            summary_version=4,
            audiovisual_summary="开头",
            audio="开头原话",
        ),
    )
    sources = {}
    for index, words in [(1, "中间对白"), (27, "最后的反转对白")]:
        start = START + timedelta(seconds=index * 10)
        end = start + timedelta(seconds=10)
        key = str(snapshot_id(FISH, CAPTURE, index, 10))
        sources[f"{key}:0"] = (
            f"{start:%Y-%m-%d %H:%M:%S} — {end:%Y-%m-%d %H:%M:%S} UTC\n{words}"
        )
        sources[f"{key}:media"] = (
            f"{start:%Y-%m-%d %H:%M:%S} — {end:%Y-%m-%d %H:%M:%S} UTC\n模型概括不是原话"
        )
    # Multiple parts from one window must survive JSONB key reordering.
    last_key = str(snapshot_id(FISH, CAPTURE, 27, 10))
    sources[f"{last_key}:1"] = (
        sources[f"{last_key}:0"].split("\n")[0] + "\n紧接着的回答"
    )
    sources[f"{last_key}:summary"] = (
        sources[f"{last_key}:0"].split("\n")[0] + "\n更高层剧情概括"
    )
    parent = replace(
        semantic_snapshot(300, 0, "parent"),
        document=SemanticDocument(
            summary_version=3,
            audiovisual_summary="整段剧情",
            narrative_sources=dict(reversed(list(sources.items()))),
        ),
    )
    result = _media_recap_audio(
        (first, parent), (), as_of=START + timedelta(seconds=305)
    )
    assert [value.text for value in result] == [
        "开头原话",
        "中间对白",
        "最后的反转对白\n\n紧接着的回答",
    ]
    assert result[-1].started_at == START + timedelta(seconds=270)
    # The newest complete passage wins when the text budget is small.
    limited = _media_recap_audio(
        (first, parent),
        (),
        as_of=START + timedelta(seconds=305),
        max_characters=len(result[-1].text) + 80,
    )
    assert [value.text for value in limited] == [result[-1].text]


def test_recap_source_times_reject_header_only_and_outside_parent():
    header = (
        f"{START:%Y-%m-%d %H:%M:%S} — "
        f"{START + timedelta(seconds=10):%Y-%m-%d %H:%M:%S} UTC"
    )
    earlier = (
        f"{START - timedelta(seconds=10):%Y-%m-%d %H:%M:%S} — "
        f"{START:%Y-%m-%d %H:%M:%S} UTC"
    )
    parent = replace(
        semantic_snapshot(300, 0, "parent"),
        started_at=START + timedelta(milliseconds=610),
        ended_at=START + timedelta(seconds=300, milliseconds=610),
        document=SemanticDocument(
            summary_version=3,
            narrative_sources={
                f"{UUID(int=1)}:0": header + "\n完整原话",
                f"{UUID(int=2)}:0": header,
                f"{UUID(int=3)}:0": earlier + "\n范围外原话",
            },
        ),
    )
    result = _media_recap_audio((parent,), (), as_of=START + timedelta(seconds=301))
    assert len(result) == 1
    assert result[0].started_at == parent.started_at
    assert result[0].ended_at == parent.started_at + timedelta(seconds=10)
    assert _media_recap_audio((parent,), (), as_of=START + timedelta(seconds=10)) == ()


@pytest.mark.asyncio
async def test_media_question_restores_original_words_without_activity_or_extra_io():
    service, repository, _ = await start_service(now=START + timedelta(seconds=305))
    try:
        raw = replace(
            semantic_snapshot(60, 0, "old"),
            document=SemanticDocument(
                audiovisualSummary="最初的事情", audio="先前的原始对白"
            ),
        )
        parent = replace(
            semantic_snapshot(300, 0, "parent"),
            document=SemanticDocument(
                audiovisualSummary="较长的剧情概括",
                activitySummary="正在查看旧调试记录",
                visualSummary="独立观察到的无声动作",
                summary="未分类的旧调试文字",
                summaryVersion=2,
            ),
        )
        for value in (raw, parent):
            service._remember_snapshot(value)
        counts = repository.calls.copy()
        context = await service.get_context_for_question(FISH, "完整的视频讲了什么？")
        assert context is not None
        assert "先前的原始对白" in context.summary
        assert "较长的剧情概括" in context.summary
        assert "独立观察到的无声动作" in context.summary
        assert "正在查看旧调试记录" not in context.summary
        assert "未分类的旧调试文字" not in context.summary
        assert repository.calls == counts
        ordinary = await service.get_context_for_question(FISH, "我刚才操作了什么？")
        assert ordinary is not None and "正在查看旧调试记录" in ordinary.summary
    finally:
        await service.close()


def test_recap_keeps_new_tail_of_overlapping_capture_and_hides_unverified_old_story():
    minute = replace(
        semantic_snapshot(60, 0, "minute"),
        document=SemanticDocument(
            audiovisualSummary="旧版本错误声称事情已成功",
            audio="原先的请求",
        ),
    )
    partial = replace(
        semantic_snapshot(10, 5, "overlap"),
        capture_session_id=UUID(int=999),
        started_at=START + timedelta(seconds=55),
        ended_at=START + timedelta(seconds=65),
        document=SemanticDocument(summary="partial", audio="后来请求被拒绝了"),
    )
    raw = _media_recap_audio((minute, partial), (), as_of=START + timedelta(seconds=66))
    assert len(raw) == 2
    assert raw[-1].text == "后来请求被拒绝了"
    assert _render_snapshot_context((minute,), include_activity=False) == ""


@pytest.mark.asyncio
async def test_plugin_media_view_excludes_activity_without_replacing_normal_context():
    service, repository, _ = await start_service(now=START + timedelta(seconds=305))
    try:
        value = replace(
            semantic_snapshot(300, 0, "parent"),
            document=SemanticDocument(
                audiovisualSummary="电影剧情",
                visualSummary="角色推开房门",
                activitySummary="ACS操作记录",
                summary="无法分类的旧操作",
                summaryVersion=2,
            ),
        )
        service._remember_snapshot(value)
        before = await service.get_latest_context_summary(FISH)
        counts = repository.calls.copy()
        media = await service.get_latest_media_context_summary(FISH)
        assert media is not None
        assert "电影剧情" in media.summary and "角色推开房门" in media.summary
        assert "ACS操作记录" not in media.summary
        assert "无法分类的旧操作" not in media.summary
        assert repository.calls == counts
        after = await service.get_latest_context_summary(FISH)
        assert before.summary == after.summary and "ACS操作记录" in after.summary
    finally:
        await service.close()


@pytest.mark.parametrize(
    "question",
    [
        "总结一下刚才的那个视频",
        "完整的视频",
        "这个故事告诉我们什么道理",
        "What was the whole clip about?",
        "刚才那个视频为什么最后走路回家？",
    ],
)
def test_recent_media_question_scope(question):
    assert requests_media_recap(question)


@pytest.mark.parametrize(
    "question",
    [
        "帮我写一个故事",
        "现在画面里是谁？",
        "修复音频采样率设置",
        "今天几号？",
        "Summarize my browser history",
        "Tell me about my browsing history",
        "这个视频设置页面里的内容是什么意思",
    ],
)
def test_other_requests_keep_normal_environment_scope(question):
    assert not requests_media_recap(question)


def revision_job(index: int):
    from fish_memory.observations.environment_history_models import (
        EnvironmentHistoryWindow,
    )
    from fish_memory.observations.environment_history_service import _Job

    return _Job(
        FISH,
        EnvironmentHistoryWindow(
            capture_session_id=CAPTURE,
            window_index=index,
            started_at=(START + timedelta(seconds=index * 10)).timestamp(),
            ended_at=(START + timedelta(seconds=(index + 1) * 10)).timestamp(),
            frames=(),
            audio_status="unavailable",
            audio_captured_samples=0,
        ),
        "revision-test",
        0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["changed_words", "wrong_time", "stale_window"])
async def test_invalid_speaker_revision_cannot_rewrite_cached_transcript(fault):
    from fish_memory.observations.environment_history_audio import SystemAudioRevision

    service, _, _ = await start_service(now=START + timedelta(seconds=100))
    try:
        service._remember_audio_transcript(
            revision_job(1), SystemAudioTranscript("还没有付款。", "Speech")
        )
        original = service._recent_audio[FISH][(CAPTURE, 1)]
        revision = SystemAudioRevision(
            1,
            original.started_at.timestamp() + (1 if fault == "wrong_time" else 0),
            "已经付款。" if fault == "changed_words" else "还没有付款。",
            (),
            "Speech",
        )
        service._remember_audio_transcript(
            revision_job(32 if fault == "stale_window" else 2),
            SystemAudioTranscript("继续。", "Speech", revisions=(revision,)),
        )
        assert service._recent_audio[FISH][(CAPTURE, 1)] == original
    finally:
        await service.close()
