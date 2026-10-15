import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.observations import environment_history_hierarchy as hierarchy
from fish_memory.observations.environment_history_hierarchy import (
    SnapshotLevel,
    snapshot_levels,
)
from fish_memory.observations.environment_history_models import (
    SemanticDocument,
    SemanticSnapshot,
    snapshot_id,
)
from fish_memory.observations.environment_history_store import (
    EnvironmentHistoryConflict,
    EnvironmentHistoryStore,
)

FISH = UUID(int=201)
CAPTURE = UUID(int=202)
START = datetime(2026, 9, 3, tzinfo=UTC)


def semantic(
    index: int,
    *,
    fish_id: UUID = FISH,
    capture_session_id: UUID = CAPTURE,
) -> SemanticSnapshot:
    return SemanticSnapshot(
        id=snapshot_id(fish_id, capture_session_id, index, 10),
        fish_id=fish_id,
        capture_session_id=capture_session_id,
        window_index=index,
        period_seconds=10,
        started_at=START + timedelta(seconds=index * 10),
        ended_at=START + timedelta(seconds=(index + 1) * 10),
        document=SemanticDocument(summary=f"测试画面 {index}", audio="本地转写"),
        input_digest=str(index).zfill(64),
        created_at=START,
    )


def aggregate(
    period_seconds: int,
    first_index: int,
    *,
    capture_session_id: UUID = CAPTURE,
    fish_id: UUID = FISH,
    source_ids: tuple[UUID, ...] = (),
    shift_seconds: float = 0,
) -> SemanticSnapshot:
    started_at = START + timedelta(seconds=first_index * 10 + shift_seconds)
    return SemanticSnapshot(
        id=snapshot_id(fish_id, capture_session_id, first_index, period_seconds),
        fish_id=fish_id,
        capture_session_id=capture_session_id,
        window_index=first_index,
        period_seconds=period_seconds,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=period_seconds),
        document=SemanticDocument(summary=f"{period_seconds} 秒摘要 {first_index}"),
        input_digest=f"{period_seconds}:{first_index}".ljust(64, "0"),
        created_at=START,
        source_ids=source_ids,
    )


@pytest.mark.asyncio
async def test_unchanged_silent_windows_remain_readable_and_complete_for_merging(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)
    baseline = replace(
        semantic(0),
        document=SemanticDocument(
            summary_version=4, activity_summary="桌面停留在文档阅读页面。"
        ),
    )
    await store.save(baseline, evidence={"frameTimes": [START.timestamp() + 1]})
    for index in range(1, 6):
        value = replace(
            semantic(index),
            document=SemanticDocument(summary_version=4, unchanged_from=baseline.id),
        )
        await store.save(
            value,
            evidence={
                "audioStatus": "complete",
                "audioCapturedSamples": 160_000,
                "audioSilence": True,
                "visualReuseFrom": str(baseline.id),
                "frameTimes": [value.started_at.timestamp() + 1],
            },
        )
        restored = await store.get(FISH, value.id)
        assert restored is not None
        assert restored.document.unchanged_from == baseline.id
        assert not restored.document.audio
        assert restored.document.summary == "画面保持不变，未听到新的声音。"
    children = await store.minute_windows(FISH, CAPTURE, 0)
    assert [child.window_index for child in children] == list(range(6))
    assert children[-1].ended_at == START + timedelta(minutes=1)
    async with postgres_pool.acquire() as connection:
        payload = await connection.fetchval(
            "SELECT payload FROM observations WHERE id=$1", children[-1].id
        )
    document = json.loads(payload)["document"]
    assert document["unchangedFrom"] == str(baseline.id)
    assert "audioPcmBase64" not in payload and "jpegBase64" not in payload


@pytest.mark.asyncio
async def test_speaker_diagnostics_round_trip_without_entering_semantic_text(
    postgres_pool: asyncpg.Pool,
) -> None:
    from fish_memory.observations.environment_history_narrative import (
        NarrativeEvent,
        NarrativeEvidence,
    )
    from fish_memory.observations.environment_history_service import (
        _render_snapshot_context,
    )

    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)
    value = replace(
        semantic(0),
        document=SemanticDocument(
            audio="[00:00.00—00:00.80] 声音abcdef-T07：吃饭了吗？",
            visual_summary="两人坐在桌边。",
            summary_version=2,
            narrative_events=(
                NarrativeEvent(
                    eventId="event:lunch:0",
                    subject="声音abcdef-T07",
                    action="询问",
                    object="对方是否吃饭",
                    status="requested",
                    narration="有人询问对方是否吃饭。",
                    evidence=(
                        NarrativeEvidence(
                            sourceId="source:lunch:0", quote="吃饭了吗？"
                        ),
                    ),
                ),
            ),
        ),
    )
    diagnostic = {
        "startSeconds": 0.0,
        "endSeconds": 0.8,
        "speakerId": None,
        "localSpeakerId": "local:abcdef:0:1:0",
        "overlap": False,
        "reason": "match_ambiguous",
        "identitySource": "local",
        "cleanSeconds": 0.6,
        "bestSimilarity": 0.73,
        "secondSimilarity": 0.72,
        "boundaryReason": "embedding_conflict",
        "trackId": "track:abcdef:7",
        "trackReason": "matched_anchor",
    }
    saved = await store.save(
        value, evidence={"diarizationStatus": "ok", "audioSegments": [diagnostic]}
    )
    payload = await postgres_pool.fetchval(
        "SELECT payload FROM observations WHERE id=$1", saved.id
    )
    payload = json.loads(payload) if isinstance(payload, str) else payload
    assert payload["evidence"]["audioSegments"] == [diagnostic]
    assert "audiovisualSummary" not in payload["document"]
    assert saved.document == value.document
    loaded = await store.get(FISH, saved.id)
    assert loaded is not None and loaded.document == value.document
    rendered = _render_snapshot_context((loaded,))
    assert "有人询问对方是否吃饭。（提出请求）" in rendered
    assert "两人坐在桌边" in rendered
    assert "source:lunch:0" not in rendered
    assert "narrativeEvents" not in rendered
    for metadata in ("local:abcdef", "match_ambiguous", "0.73", "embedding_conflict"):
        assert metadata not in saved.document.model_dump_json()


@pytest.mark.asyncio
async def test_history_observations_are_idempotent_scoped_and_compact_atomically(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    other = UUID(int=203)
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1),($2)", FISH, other)
    store = EnvironmentHistoryStore(postgres_pool)
    for index in [3, 0, 5, 2, 1, 4, 6]:
        await store.save(semantic(index))
    await store.save(semantic(0))
    await store.save(semantic(0, fish_id=other))
    with pytest.raises(EnvironmentHistoryConflict):
        await store.save(replace(semantic(0), input_digest="f" * 64))
    windows = await store.minute_windows(FISH, CAPTURE, 0)
    assert [item.window_index for item in windows] == list(range(6))
    minute = replace(
        semantic(0),
        id=snapshot_id(FISH, CAPTURE, 0, 60),
        period_seconds=60,
        ended_at=START + timedelta(seconds=60),
        document=SemanticDocument(summary="完整一分钟"),
        source_ids=tuple(item.id for item in windows),
    )
    await store.save(minute)
    await store.save(minute)
    latest, tail = await store.context_snapshots(FISH)
    assert latest is not None and latest.id == minute.id
    assert [item.window_index for item in tail] == [6]
    other_minute, other_tail = await store.context_snapshots(other)
    assert other_minute is None and len(other_tail) == 1
    rows = await postgres_pool.fetch(
        "SELECT source, payload FROM observations WHERE fish_id=$1", FISH
    )
    assert len(rows) == 8
    assert all(row["source"] == "environment_history" for row in rows)
    assert all(
        "pcmBase64" not in row["payload"] and "jpegBase64" not in row["payload"]
        for row in rows
    )
    assert await postgres_pool.fetchval("SELECT count(*) FROM messages") == 0
    assert await postgres_pool.fetchval("SELECT count(*) FROM facts") == 0


@pytest.mark.asyncio
async def test_completed_ten_second_snapshots_survive_before_minute_is_ready(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)
    for index in [0, 2, 4]:
        await store.save(semantic(index))
    fresh_store = EnvironmentHistoryStore(postgres_pool)
    minute, tail = await fresh_store.context_snapshots(FISH)
    assert minute is None
    assert [item.window_index for item in tail] == [0, 2, 4]


@pytest.mark.asyncio
async def test_recovery_finds_bounded_complete_groups_without_crossing_scope_or_gaps(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    other = UUID(int=203)
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1),($2)", FISH, other)
    store = EnvironmentHistoryStore(postgres_pool)
    for index in range(35):
        # Five complete minutes, then an incomplete five-window group.
        await store.save(semantic(index))
    for index in range(36, 42):
        value = semantic(index)
        if index == 41:
            value = replace(
                value,
                started_at=value.started_at + timedelta(seconds=1),
                ended_at=value.ended_at + timedelta(seconds=1),
            )
        await store.save(value)
    for index in range(6):
        await store.save(semantic(index, fish_id=other))
    for index in range(42, 48):
        capture = UUID(int=300 if index < 45 else 301)
        await store.save(
            replace(
                semantic(index),
                id=snapshot_id(FISH, capture, index, 10),
                capture_session_id=capture,
            )
        )

    fresh_store = EnvironmentHistoryStore(postgres_pool)
    pending = await fresh_store.pending_minute_windows(FISH)
    assert [value.window_index for value in pending] == [5, 11, 17, 23]
    assert all(value.capture_session_id == CAPTURE for value in pending)
    assert all(value.fish_id == FISH for value in pending)
    assert len(await fresh_store.pending_minute_windows(FISH, limit=99)) == 4
    assert await fresh_store.pending_minute_windows(FISH, limit=0) == ()
    await store.save(
        replace(
            semantic(0),
            id=snapshot_id(FISH, CAPTURE, 0, 60),
            period_seconds=60,
            ended_at=START + timedelta(seconds=60),
        )
    )
    assert [
        value.window_index for value in await fresh_store.pending_minute_windows(FISH)
    ] == [11, 17, 23, 29]
    assert [
        value.window_index
        for value in await fresh_store.pending_minute_windows(FISH, limit=2)
    ] == [11, 17]
    assert [
        value.window_index for value in await fresh_store.pending_minute_windows(other)
    ] == [5]
    bounded = await fresh_store.pending_minute_windows(
        FISH,
        since=START + timedelta(seconds=120),
        through=START + timedelta(seconds=300),
    )
    assert [value.window_index for value in bounded] == [17, 23, 29]


@pytest.mark.asyncio
async def test_no_media_history_is_ignored_without_deleting_evidence(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)

    # A valid completed minute must remain visible after newer empty runs.
    valid_windows = []
    for index in range(6):
        valid_windows.append(
            await store.save(
                semantic(index),
                evidence={
                    "frameTimes": [semantic(index).started_at.timestamp()],
                    "audioCapturedSamples": 0,
                },
            )
        )
    valid_minute = replace(
        semantic(0),
        id=snapshot_id(FISH, CAPTURE, 0, 60),
        period_seconds=60,
        ended_at=START + timedelta(seconds=60),
        source_ids=tuple(value.id for value in valid_windows),
    )
    await store.save(valid_minute)

    # A complete old group produced while ScreenCaptureKit/TCC delivered no
    # media must neither enter Ask context nor get resubmitted for a merge.
    empty_windows = []
    for index in range(6, 12):
        empty_windows.append(
            await store.save(
                semantic(index),
                evidence={"frameTimes": [], "audioCapturedSamples": 0},
            )
        )
    empty_minute = replace(
        semantic(6),
        id=snapshot_id(FISH, CAPTURE, 6, 60),
        period_seconds=60,
        ended_at=START + timedelta(seconds=120),
        source_ids=tuple(value.id for value in empty_windows),
    )
    await store.save(empty_minute)
    for index in range(12, 18):
        await store.save(semantic(index), evidence={"audioCapturedSamples": 0})

    # Audio alone, a frame alone, and unknown old evidence are all usable.
    for index in range(18, 24):
        evidence = (
            {"frameTimes": [], "audioCapturedSamples": 160_000}
            if index % 3 == 0
            else {
                "frameTimes": [semantic(index).started_at.timestamp()],
                "audioCapturedSamples": 0,
            }
            if index % 3 == 1
            else None
        )
        await store.save(semantic(index), evidence=evidence)
    # One explicitly empty source also disqualifies its old mixed minute;
    # valid source windows remain individually available instead.
    mixed_windows = []
    for index in range(24, 30):
        mixed_windows.append(
            await store.save(
                semantic(index),
                evidence={
                    "frameTimes": []
                    if index == 29
                    else [semantic(index).started_at.timestamp()],
                    "audioCapturedSamples": 0,
                },
            )
        )
    await store.save(
        replace(
            semantic(24),
            id=snapshot_id(FISH, CAPTURE, 24, 60),
            period_seconds=60,
            ended_at=START + timedelta(seconds=300),
            source_ids=tuple(value.id for value in mixed_windows),
        )
    )

    fresh = EnvironmentHistoryStore(postgres_pool)
    minute, tail = await fresh.context_snapshots(FISH)
    assert minute is not None and minute.id == valid_minute.id
    assert [value.window_index for value in tail] == list(range(18, 29))
    assert await fresh.minute_windows(FISH, CAPTURE, 6) == ()
    assert await fresh.minute_windows(FISH, CAPTURE, 12) == ()
    assert [
        value.window_index for value in await fresh.minute_windows(FISH, CAPTURE, 24)
    ] == list(range(24, 29))
    assert [
        value.window_index for value in await fresh.pending_minute_windows(FISH)
    ] == [23]
    # Identity/status lookup and SQL rows are retained; filtering is not cleanup.
    assert await fresh.get(FISH, empty_windows[0].id) is not None
    assert await fresh.get(FISH, empty_minute.id) is not None
    assert (
        await postgres_pool.fetchval(
            "SELECT count(*) FROM observations WHERE fish_id=$1", FISH
        )
        == 33
    )


@pytest.mark.asyncio
async def test_hierarchy_children_require_complete_contiguous_same_session_ranges(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)

    minute_snapshots: list[SemanticSnapshot] = []
    for first_index in range(0, 30, 6):
        windows = []
        for index in range(first_index, first_index + 6):
            windows.append(await store.save(semantic(index)))
        minute_snapshots.append(
            await store.save(
                aggregate(
                    60,
                    first_index,
                    source_ids=tuple(window.id for window in windows),
                )
            )
        )

    minutes = await store.child_snapshots(
        FISH,
        CAPTURE,
        0,
        child_period_seconds=60,
        parent_period_seconds=300,
    )
    assert [snapshot.window_index for snapshot in minutes] == [0, 6, 12, 18, 24]

    # An identically indexed child from another capture session cannot fill a
    # hole. The lookup uses actual scoped children rather than source UUIDs.
    missing = minute_snapshots[2]
    await postgres_pool.execute("DELETE FROM observations WHERE id=$1", missing.id)
    other_capture = UUID(int=204)
    await store.save(aggregate(60, 12, capture_session_id=other_capture))
    assert (
        await store.child_snapshots(
            FISH,
            CAPTURE,
            0,
            child_period_seconds=60,
            parent_period_seconds=300,
        )
        == ()
    )

    for first_index in range(0, 180, 30):
        await store.save(aggregate(300, first_index))
    assert [
        snapshot.window_index
        for snapshot in await store.child_snapshots(
            FISH,
            CAPTURE,
            0,
            child_period_seconds=300,
            parent_period_seconds=1800,
        )
    ] == [0, 30, 60, 90, 120, 150]

    broken_capture = UUID(int=205)
    for first_index in range(0, 180, 30):
        await store.save(
            aggregate(
                300,
                first_index,
                capture_session_id=broken_capture,
                shift_seconds=1 if first_index == 90 else 0,
            )
        )
    assert (
        await store.child_snapshots(
            FISH,
            broken_capture,
            0,
            child_period_seconds=300,
            parent_period_seconds=1800,
        )
        == ()
    )
    with pytest.raises(ValueError, match="direct hierarchy"):
        await store.child_snapshots(
            FISH,
            CAPTURE,
            0,
            child_period_seconds=10,
            parent_period_seconds=300,
        )
    with pytest.raises(ValueError, match="aligned"):
        await store.child_snapshots(
            FISH,
            CAPTURE,
            6,
            child_period_seconds=60,
            parent_period_seconds=300,
        )


@pytest.mark.asyncio
async def test_recent_hour_returns_all_layers_in_stable_start_and_period_order(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    other = UUID(int=203)
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1),($2)", FISH, other)
    store = EnvironmentHistoryStore(postgres_pool)

    values = [
        semantic(0),
        aggregate(60, 0),
        aggregate(300, 0),
        aggregate(1800, 0),
        semantic(180),
    ]
    for value in values:
        await store.save(value)
    await store.save(
        semantic(1), evidence={"frameTimes": [], "audioCapturedSamples": 0}
    )
    await store.save(semantic(0, fish_id=other))

    recent = await store.recent_hour_snapshots(
        FISH,
        since=START,
        through=START + timedelta(hours=1),
    )

    assert [(value.window_index, value.period_seconds) for value in recent] == [
        (0, 1800),
        (0, 300),
        (0, 60),
        (0, 10),
        (180, 10),
    ]
    assert all(value.fish_id == FISH for value in recent)
    # Keep complete nodes at the left boundary so the hour selector can retain
    # a thirty-minute block instead of shredding it into tiny leading windows.
    clipped = await store.recent_hour_snapshots(
        FISH,
        since=START + timedelta(seconds=10),
        through=START + timedelta(hours=1),
    )
    assert [(value.window_index, value.period_seconds) for value in clipped] == [
        (0, 1800),
        (0, 300),
        (0, 60),
        (0, 10),
        (180, 10),
    ]
    with pytest.raises(ValueError, match="timezone-aware"):
        await store.recent_hour_snapshots(
            FISH,
            since=START.replace(tzinfo=None),
            through=START + timedelta(hours=1),
        )


@pytest.mark.asyncio
async def test_compacted_window_cleanup_requires_named_covering_same_session_parent(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)

    first_group_values = []
    for index in range(6):
        first_group_values.append(await store.save(semantic(index)))
    first_group = tuple(first_group_values)
    equal_cutoff_values = []
    for index in range(6, 12):
        equal_cutoff_values.append(await store.save(semantic(index)))
    equal_cutoff_group = tuple(equal_cutoff_values)
    other_capture = UUID(int=206)
    unparented = await store.save(semantic(0, capture_session_id=other_capture))
    first_parent = await store.save(
        aggregate(
            60,
            0,
            source_ids=tuple(value.id for value in first_group),
        )
    )
    second_parent = await store.save(
        aggregate(
            60,
            6,
            source_ids=tuple(value.id for value in equal_cutoff_group),
        )
    )
    five_minute_parent = await store.save(
        aggregate(300, 0, source_ids=(first_parent.id, second_parent.id))
    )

    deleted = await store.delete_compacted_windows(
        FISH,
        ended_before=START + timedelta(seconds=70),
    )

    assert deleted == 6
    for value in first_group:
        assert await store.get(FISH, value.id) is None
    # Strictly earlier means the covered child ending exactly at cutoff stays.
    assert await store.get(FISH, equal_cutoff_group[0].id) is not None
    # A different capture session without its own parent is never compacted.
    assert await store.get(FISH, unparented.id) is not None
    assert await store.get(FISH, first_parent.id) is not None
    assert await store.get(FISH, second_parent.id) is not None
    assert await store.get(FISH, five_minute_parent.id) is not None


@pytest.mark.asyncio
async def test_cleanup_compacts_all_detail_layers_but_keeps_half_hour_parent(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)

    minutes = []
    for first_index in range(0, 30, 6):
        leaves = []
        for index in range(first_index, first_index + 6):
            leaves.append(await store.save(semantic(index)))
        minutes.append(
            await store.save(
                aggregate(
                    60,
                    first_index,
                    source_ids=tuple(value.id for value in leaves),
                )
            )
        )
    five_minutes = [
        await store.save(
            aggregate(300, 0, source_ids=tuple(value.id for value in minutes))
        )
    ]
    for first_index in range(30, 180, 30):
        five_minutes.append(await store.save(aggregate(300, first_index)))
    half_hour = await store.save(
        aggregate(
            1800,
            0,
            source_ids=tuple(value.id for value in five_minutes),
        )
    )

    deleted = await store.delete_compacted_windows(
        FISH,
        ended_before=START + timedelta(seconds=1801),
    )

    assert deleted == 41
    for value in minutes:
        assert await store.get(FISH, value.id) is None
    for value in five_minutes:
        assert await store.get(FISH, value.id) is None
    assert await store.get(FISH, half_hour.id) == half_hour


@pytest.mark.asyncio
async def test_split_summary_documents_persist_without_overwriting_legacy_history(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)
    legacy = await store.save(semantic(0))
    document = SemanticDocument(
        audiovisual_summary=("奶奶说明明天由爸爸接小明。\n小明答应先写完作业。" * 100),
        activity_summary="用户切换到编辑器，视频继续播放。",
        audio="明天让你爸爸接你。",
    )
    assert len(document.summary) > 2000
    value = replace(semantic(1), document=document)
    await store.save(value)
    # An already durable legacy identity is immutable even if a caller later
    # supplies a categorized document with the same input digest.
    preserved = await store.save(replace(legacy, document=document))
    assert preserved.document.legacy_summary == legacy.document.legacy_summary
    assert preserved.document.audiovisual_summary == ""

    fresh = EnvironmentHistoryStore(postgres_pool)
    loaded = await fresh.get(FISH, value.id)
    assert loaded is not None and loaded.document == document
    payload = await postgres_pool.fetchval(
        "SELECT payload->'document' FROM observations WHERE id=$1", value.id
    )
    stored_document = json.loads(payload) if isinstance(payload, str) else payload
    assert stored_document == document.model_dump()
    assert "summary" not in stored_document
    assert (
        await postgres_pool.fetchval(
            "SELECT summary FROM observations WHERE id=$1", value.id
        )
        == document.summary
    )


@pytest.mark.asyncio
async def test_configured_extra_level_flows_through_queries_and_safe_cleanup(
    postgres_pool: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = SnapshotLevel(7200, 1800, "completedTwoHours", "failedTwoHours")
    monkeypatch.setattr(hierarchy, "SNAPSHOT_LEVELS", (*snapshot_levels(), extra))
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)
    children = [
        await store.save(aggregate(1800, first_index))
        for first_index in (0, 180, 360, 540)
    ]
    parent = await store.save(
        replace(
            aggregate(7200, 0, source_ids=tuple(child.id for child in children)),
            document=SemanticDocument(
                audiovisual_summary="持续剧情：冲突出现后，人物逐步澄清误会。",
                activity_summary="用户其间切换应用，仍继续播放同一视频。",
            ),
        )
    )

    assert await store.child_snapshots(
        FISH, CAPTURE, 0, child_period_seconds=1800, parent_period_seconds=7200
    ) == tuple(children)
    snapshots = await store.recent_hour_snapshots(
        FISH, since=START, through=START + timedelta(hours=2)
    )
    assert parent in snapshots
    assert all(child in snapshots for child in children)
    assert (
        await store.delete_compacted_windows(
            FISH, ended_before=START + timedelta(hours=2, seconds=1)
        )
        == 4
    )
    assert await store.get(FISH, parent.id) == parent
    for child in children:
        assert await store.get(FISH, child.id) is None


@pytest.mark.asyncio
async def test_speaker_revision_round_trip_preserves_words_audit_and_newer_version(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)
    words = "[00:00.00—00:01.00] 说话人未确定：还没有寄出。"
    original = replace(
        semantic(0),
        document=SemanticDocument(
            summary_version=3,
            audio=words,
            audiovisual_summary=f"听到的话：\n{words}",
        ),
    )
    await store.save(original)
    corrected_words = words.replace("说话人未确定", "声音abcdef-01")
    corrected = replace(
        original,
        document=SemanticDocument(
            summary_version=3,
            audio=corrected_words,
            audiovisual_summary=f"听到的话：\n{corrected_words}",
        ),
    )
    diagnostics = [
        {"speakerId": "声音abcdef-01", "startSeconds": 0.0, "endSeconds": 1.0}
    ]
    saved = await store.revise_attribution(
        corrected, through_index=20, segments=diagnostics
    )
    assert saved is not None and saved.document == corrected.document
    payload = await postgres_pool.fetchval(
        "SELECT payload FROM observations WHERE id=$1", original.id
    )
    payload = json.loads(payload) if isinstance(payload, str) else payload
    assert payload["evidence"]["originalAttribution"] == words
    assert payload["evidence"]["speakerRevisionWindow"] == 20
    assert payload["evidence"]["audioSegments"] == diagnostics
    # A late callback cannot restore the old unknown label.
    stale = await store.revise_attribution(original, through_index=19)
    assert stale is not None and stale.document == corrected.document
    altered = replace(
        corrected,
        document=SemanticDocument(
            summary_version=3,
            audio=corrected_words.replace("还没有", "已经"),
            audiovisual_summary=corrected.document.audiovisual_summary,
        ),
    )
    with pytest.raises(EnvironmentHistoryConflict, match="words or story"):
        await store.revise_attribution(altered, through_index=21)
    with pytest.raises(EnvironmentHistoryConflict, match="input changed"):
        await store.revise_attribution(
            replace(corrected, capture_session_id=UUID(int=999)), through_index=21
        )


@pytest.mark.asyncio
async def test_jsonb_source_order_cannot_scramble_the_next_narrative_merge(
    postgres_pool: asyncpg.Pool,
) -> None:
    from fish_memory.observations.environment_history_client import _narrative_content
    from fish_memory.observations.environment_history_narrative import (
        NarrativeParagraph,
    )

    await run_migrations(cast(MigrationPool, postgres_pool))
    await postgres_pool.execute("INSERT INTO fishes(id) VALUES($1)", FISH)
    store = EnvironmentHistoryStore(postgres_pool)
    sources = {
        "z:0": "2026-09-03 00:00:00 — 2026-09-03 00:00:10 UTC\n约好明天见面。",
        "a:0": "2026-09-03 00:00:20 — 2026-09-03 00:00:30 UTC\n临时取消了约定。",
        "b:0": "2026-09-03 00:00:10 — 2026-09-03 00:00:20 UTC\n对方答应了。",
    }
    parent = replace(
        aggregate(60, 0),
        document=SemanticDocument(
            summary_version=3,
            narrative_paragraphs=(
                NarrativeParagraph(
                    text="约定得到同意，后来又被取消。", source_ids=tuple(sources)
                ),
            ),
            narrative_sources=sources,
        ),
    )
    await store.save(parent)
    loaded = await EnvironmentHistoryStore(postgres_pool).get(FISH, parent.id)
    assert loaded is not None
    assert loaded.document == parent.document
    assert list(loaded.document.narrative_sources) != list(sources)
    content, ordered, _ = _narrative_content([loaded], loaded.document)
    assert list(ordered) == ["z:0", "b:0", "a:0"]
    request = json.loads(content)
    assert [item["audio"].splitlines()[-1] for item in request["segments"]] == [
        "约好明天见面。",
        "对方答应了。",
        "临时取消了约定。",
    ]
    assert request["previousWords"] == [
        "2026-09-03T00:00:00+00:00 — 2026-09-03T00:00:10+00:00\n约好明天见面。",
        "2026-09-03T00:00:10+00:00 — 2026-09-03T00:00:20+00:00\n对方答应了。",
        "2026-09-03T00:00:20+00:00 — 2026-09-03T00:00:30+00:00\n临时取消了约定。",
    ]
