from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.observations import environment_history_hierarchy as hierarchy
from fish_memory.observations.environment_history_hierarchy import (
    SnapshotLevel,
    level_for_period,
    parent_level_for_period,
    snapshot_levels,
)
from fish_memory.observations.environment_history_models import (
    FIVE_MINUTE_TYPE,
    HALF_HOUR_TYPE,
    MINUTE_TYPE,
    SEMANTIC_AUDIO_MAX_CHARACTERS,
    SEMANTIC_DOCUMENT_MAX_BYTES,
    WINDOW_TYPE,
    SemanticDocument,
    SemanticSnapshot,
    observation_type_for_period,
    snapshot_id,
)
from fish_memory.observations.environment_history_store import (
    EnvironmentHistoryStore,
)

FISH = UUID(int=501)
CAPTURE = UUID(int=502)
START = datetime(2026, 9, 4, tzinfo=UTC)


class RecordingPool:
    def __init__(self) -> None:
        self.parameters: tuple[object, ...] | None = None

    async def execute(self, query: str, *parameters: object) -> str:
        del query
        self.parameters = parameters
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *parameters: object) -> dict[str, object]:
        del query
        assert self.parameters is not None
        written = self.parameters
        assert parameters[:2] == (FISH, written[0])
        return {
            "id": written[0],
            "fish_id": written[1],
            "payload": written[7],
            "started_at": written[8],
            "ended_at": written[9],
            "created_at": START,
        }


class StoredRowPool:
    def __init__(self, row: dict[str, object]) -> None:
        self.row = row

    async def fetchrow(self, query: str, *parameters: object) -> dict[str, object]:
        del query, parameters
        return self.row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,visible",
    [
        ({"summaryVersion": 4}, ""),
        ({"summaryVersion": 4, "audio": "尚未形成总结的转写"}, ""),
        ({"summaryVersion": 4, "activitySummary": " \n\t"}, ""),
        ({"summaryVersion": 4, "unchangedFrom": str(UUID(int=401))}, ""),
        ({"summary": "旧版画面描述。"}, "旧版画面描述。"),
        ({"summaryVersion": 4, "activitySummary": "用户打开文档。"}, "用户打开文档。"),
        (
            {
                "summaryVersion": 3,
                "narrativeParagraphs": [
                    {"text": "角色走进教室。", "sourceIds": ["s1"]}
                ],
                "narrativeSources": {"s1": "角色走进教室。"},
            },
            "角色走进教室。",
        ),
    ],
)
async def test_capture_receipt_visibility_matches_rendered_history(raw, visible):
    document = SemanticDocument.model_validate(raw)
    pool = RecordingPool()
    snapshot = replace(
        minute_snapshot(document),
        period_seconds=5,
        ended_at=START + timedelta(seconds=5),
    )
    saved = await EnvironmentHistoryStore(cast(Any, pool)).save(snapshot)
    assert saved.document.history_summary == visible
    assert pool.parameters is not None
    payload = json.loads(cast(str, pool.parameters[7]))
    assert payload.get("include_in_context", True) is bool(visible)


@pytest.mark.asyncio
@pytest.mark.parametrize("period", [10, 60, 300, 1800])
async def test_split_summaries_round_trip_without_reclassification_or_shortening(
    period,
):
    audiovisual = "奶奶解释明天由爸爸接孩子。\n" + "对白中的关键转折。" * 100
    activity = "用户切换到编辑器；视频仍在后台播放。"
    legacy = "旧记录没有可靠的分类边界。"
    document = SemanticDocument(
        audiovisual_summary=audiovisual,
        activity_summary=activity,
        legacy_summary=legacy,
        audio="明天让你爸爸接你。",
    )
    pool = RecordingPool()
    saved = await EnvironmentHistoryStore(cast(Any, pool)).save(
        replace(
            minute_snapshot(document),
            id=snapshot_id(FISH, CAPTURE, 0, period),
            period_seconds=period,
            ended_at=START + timedelta(seconds=period),
        )
    )

    assert saved.document == document
    assert saved.document.audiovisual_summary == audiovisual
    assert saved.document.activity_summary == activity
    assert saved.document.legacy_summary == legacy
    assert pool.parameters is not None
    payload = json.loads(cast(str, pool.parameters[7]))
    assert payload["document"] == {
        "audiovisualSummary": audiovisual,
        "activitySummary": activity,
        "audio": "明天让你爸爸接你。",
        "summary": legacy,
    }
    assert pool.parameters[6] == document.summary
    assert "影音内容：" in document.summary
    assert "电脑操作：" in document.summary
    assert "未分类内容：" in document.summary


def test_split_summaries_accept_aliases_and_do_not_invent_a_legacy_summary():
    document = SemanticDocument.model_validate(
        {"audiovisualSummary": "影片中两人决定和好。", "activitySummary": ""}
    )
    assert document.audiovisual_summary == "影片中两人决定和好。"
    assert document.activity_summary == document.legacy_summary == ""
    assert document.model_dump() == {
        "audiovisualSummary": "影片中两人决定和好。",
        "activitySummary": "",
        "audio": "",
    }
    with pytest.raises(AttributeError):
        document.summary = "不能把展示文字写回分类"  # type: ignore[misc]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"audio": "仅有转录还不是摘要"},
        {"audiovisualSummary": "  ", "activitySummary": "\n", "summary": ""},
    ],
)
def test_semantic_document_requires_some_nonempty_summary(payload):
    with pytest.raises(ValidationError, match="nonempty summary"):
        SemanticDocument.model_validate(payload)


def test_split_summaries_share_the_utf8_storage_budget():
    with pytest.raises(ValidationError, match="storage budget"):
        SemanticDocument(
            audiovisual_summary="中" * 6_000,
            activity_summary="文" * 6_000,
        )


@pytest.mark.asyncio
async def test_store_redacts_each_category_audio_and_legacy_independently():
    document = SemanticDocument(
        audiovisual_summary="影片人物说 password=story-secret",
        activity_summary="编辑器显示 api_key=activity-secret",
        legacy_summary="旧条目 password=legacy-secret",
        audio="password=audio-secret",
    )
    pool = RecordingPool()
    saved = await EnvironmentHistoryStore(cast(Any, pool)).save(
        minute_snapshot(document)
    )
    assert saved.document.audiovisual_summary == "影片人物说 password=[REDACTED]"
    assert saved.document.activity_summary == "编辑器显示 api_key=[REDACTED]"
    assert saved.document.legacy_summary == "旧条目 password=[REDACTED]"
    assert saved.document.audio == "password=[REDACTED]"


@pytest.mark.asyncio
async def test_store_preserves_typed_event_provenance_and_redacts_nested_words():
    from fish_memory.observations.environment_history_narrative import (
        NarrativeEvent,
        NarrativeEvidence,
    )

    event_id = "event:" + "a" * 32
    source_id = "source:" + "b" * 32
    document = SemanticDocument(
        audiovisualSummary="有人提及账户信息。",
        summaryVersion=2,
        visualSummary="画面出现 password=visual-secret",
        narrativeEvents=(
            NarrativeEvent(
                eventId=event_id,
                subject="发言者",
                action="提及",
                object="信息",
                status="reported",
                narration="password=event-secret",
                evidence=(
                    NarrativeEvidence(
                        sourceId=source_id, quote="password=quote-secret"
                    ),
                ),
            ),
        ),
    )
    saved = await EnvironmentHistoryStore(cast(Any, RecordingPool())).save(
        minute_snapshot(document)
    )
    event = saved.document.narrative_events[0]
    assert saved.document.summary_version == 2
    assert event.event_id == event_id and event.evidence[0].source_id == source_id
    assert event.status == "reported"
    assert "secret" not in saved.document.model_dump_json()
    assert "quote-secret" in document.narrative_events[0].evidence[0].quote


@pytest.mark.asyncio
async def test_empty_processed_window_has_index_text_but_no_invented_context():
    document = SemanticDocument(summaryVersion=2)
    saved = await EnvironmentHistoryStore(cast(Any, RecordingPool())).save(
        minute_snapshot(document)
    )
    assert saved.document.summary == ""
    assert saved.document.audio == ""
    assert saved.document.summary_version == 2


def test_hierarchy_exposes_configured_direct_edges():
    assert [
        (level.period_seconds, level.child_count) for level in snapshot_levels()
    ] == [(5, 0), (10, 0), (60, 6), (300, 5), (1800, 6)]
    assert parent_level_for_period(10) == level_for_period(60)
    assert parent_level_for_period(1800) is None
    with pytest.raises(ValueError, match="unsupported"):
        level_for_period(120)
    with pytest.raises(ValueError, match="whole shorter children"):
        SnapshotLevel(100, 60, "completed", "failed")


@pytest.mark.asyncio
async def test_a_configured_extra_level_needs_no_storage_type_whitelist(monkeypatch):
    extra = SnapshotLevel(7200, 1800, "completedTwoHours", "failedTwoHours")
    monkeypatch.setattr(hierarchy, "SNAPSHOT_LEVELS", (*snapshot_levels(), extra))
    value = replace(
        minute_snapshot(SemanticDocument(activity_summary="两小时操作概括")),
        id=snapshot_id(FISH, CAPTURE, 0, 7200),
        period_seconds=7200,
        ended_at=START + timedelta(seconds=7200),
    )
    pool = RecordingPool()
    saved = await EnvironmentHistoryStore(cast(Any, pool)).save(value)
    assert saved.document.activity_summary == "两小时操作概括"
    assert parent_level_for_period(1800) == extra
    assert pool.parameters is not None
    assert pool.parameters[3] == "semantic_snapshot_7200s"


def minute_snapshot(document: SemanticDocument) -> SemanticSnapshot:
    return SemanticSnapshot(
        id=snapshot_id(FISH, CAPTURE, 0, 60),
        fish_id=FISH,
        capture_session_id=CAPTURE,
        window_index=0,
        period_seconds=60,
        started_at=START,
        ended_at=START + timedelta(seconds=60),
        document=document,
        input_digest="f" * 64,
        created_at=START,
        source_ids=tuple(UUID(int=600 + index) for index in range(6)),
    )


@pytest.mark.asyncio
async def test_minute_audio_contract_fits_utf8_transcripts_under_observation_limit():
    audio = "\n".join(f"<Music>{'中' * 1000}</Music>" for _ in range(6))
    assert len(audio) == 6095
    document = SemanticDocument(summary="概" * 600, audio=audio)
    assert (
        len(document.model_dump_json().encode("utf-8")) <= SEMANTIC_DOCUMENT_MAX_BYTES
    )

    pool = RecordingPool()
    store = EnvironmentHistoryStore(cast(Any, pool))
    saved = await store.save(minute_snapshot(document))

    assert saved.document == document
    assert pool.parameters is not None
    encoded = cast(str, pool.parameters[7])
    payload = json.loads(encoded)
    assert len(encoded.encode("utf-8")) <= 32_768
    assert set(payload["document"]) == {"summary", "audio"}
    assert "documentMarkdown" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(("period", "count"), [(300, 5), (1800, 6)])
async def test_complete_merge_fallback_fits_the_document_and_observation_budgets(
    period, count
):
    summary = "；".join(f"{index}:" + "细节" * 290 for index in range(count))
    document = SemanticDocument(summary=summary)
    assert len(summary) > 1600
    snapshot = replace(
        minute_snapshot(document),
        id=snapshot_id(FISH, CAPTURE, 0, period),
        period_seconds=period,
        ended_at=START + timedelta(seconds=period),
        source_ids=tuple(UUID(int=600 + index) for index in range(count)),
    )
    pool = RecordingPool()
    saved = await EnvironmentHistoryStore(cast(Any, pool)).save(snapshot)
    assert saved.document.summary == summary
    assert pool.parameters is not None
    assert len(cast(str, pool.parameters[7]).encode()) <= 32_768


def test_merge_fallback_still_rejects_excessive_utf8_bytes():
    with pytest.raises(ValidationError, match="storage budget"):
        SemanticDocument(summary="详" * (SEMANTIC_DOCUMENT_MAX_BYTES // 3))


@pytest.mark.parametrize(
    ("period_seconds", "expected_type"),
    [
        (5, "semantic_snapshot_5s"),
        (10, WINDOW_TYPE),
        (60, MINUTE_TYPE),
        (300, FIVE_MINUTE_TYPE),
        (1800, HALF_HOUR_TYPE),
    ],
)
@pytest.mark.asyncio
async def test_save_uses_explicit_observation_type_for_every_hierarchy_period(
    period_seconds: int,
    expected_type: str,
):
    pool = RecordingPool()
    store = EnvironmentHistoryStore(cast(Any, pool))
    value = replace(
        minute_snapshot(SemanticDocument(summary="分层摘要")),
        id=snapshot_id(FISH, CAPTURE, 0, period_seconds),
        period_seconds=period_seconds,
        ended_at=START + timedelta(seconds=period_seconds),
    )

    await store.save(value)

    assert observation_type_for_period(period_seconds) == expected_type
    assert pool.parameters is not None
    assert pool.parameters[3] == expected_type
    assert pool.parameters[4] == f"capture:{CAPTURE}:{period_seconds}"


@pytest.mark.asyncio
async def test_save_rejects_an_unmapped_period_before_writing():
    pool = RecordingPool()
    store = EnvironmentHistoryStore(cast(Any, pool))
    value = replace(
        minute_snapshot(SemanticDocument(summary="未知层级")),
        id=snapshot_id(FISH, CAPTURE, 0, 120),
        period_seconds=120,
        ended_at=START + timedelta(seconds=120),
    )

    with pytest.raises(ValueError, match="unsupported semantic snapshot period"):
        await store.save(value)

    assert pool.parameters is None


def test_audio_character_capacity_and_document_utf8_budget_are_independent():
    assert (
        SemanticDocument(
            summary="summary",
            audio="a" * SEMANTIC_AUDIO_MAX_CHARACTERS,
        ).audio
        == "a" * 12_000
    )
    with pytest.raises(ValidationError):
        SemanticDocument(summary="summary", audio="a" * 12_001)
    with pytest.raises(ValidationError, match="storage budget"):
        SemanticDocument(summary="😀" * 1_600, audio="😀" * 8_000)


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_size", [29_000, 30_500])
async def test_optional_speaker_scores_never_discard_a_full_summary(summary_size):
    document = SemanticDocument(summary="word " * (summary_size // 5), audio="完整对白")
    pool = RecordingPool()
    intervals = [
        {
            "startSeconds": index / 2,
            "endSeconds": (index + 1) / 2,
            "speakerId": "声音abcdef-01",
            "overlap": False,
            "localSpeakerId": f"local:abcdef:100:0:{index}",
            "reason": "match_ambiguous",
            "cleanSeconds": 0.5,
            "bestSimilarity": 0.7123,
            "secondSimilarity": 0.7034,
            "identitySource": "local",
        }
        for index in range(16)
    ]
    saved = await EnvironmentHistoryStore(cast(Any, pool)).save(
        minute_snapshot(document),
        evidence={"diarizationStatus": "ok", "audioSegments": intervals},
    )
    assert saved.document == document
    assert pool.parameters is not None
    encoded = cast(str, pool.parameters[7])
    assert len(encoded.encode()) <= 32_768
    payload = json.loads(encoded)
    assert payload["document"] == document.model_dump()
    assert payload["evidence"]["speakerDiagnosticsOmitted"] == "payload_budget"
    if summary_size == 29_000:
        assert len(payload["evidence"]["audioSegments"]) == 16
    else:
        assert payload["evidence"]["audioSegmentCount"] == 16
    # Compact only the written envelope, not the caller's diagnostic objects.
    assert intervals[0]["bestSimilarity"] == 0.7123


@pytest.mark.asyncio
async def test_legacy_visual_uncertainties_and_document_markdown_are_ignored():
    snapshot = minute_snapshot(SemanticDocument(summary="旧概括", audio="旧转写"))
    row = {
        "id": snapshot.id,
        "fish_id": snapshot.fish_id,
        "payload": {
            "captureSessionId": str(snapshot.capture_session_id),
            "windowIndex": snapshot.window_index,
            "periodSeconds": snapshot.period_seconds,
            "inputDigest": snapshot.input_digest,
            "document": {
                "summary": "旧概括",
                "visual": "旧画面细节",
                "audio": "旧转写",
                "uncertainties": "旧内部解释",
            },
            "documentMarkdown": "不得重新进入新文档",
            "sourceSnapshotIds": [],
            "evidence": {},
        },
        "started_at": snapshot.started_at,
        "ended_at": snapshot.ended_at,
        "created_at": snapshot.created_at,
    }
    store = EnvironmentHistoryStore(cast(Any, StoredRowPool(row)))

    loaded = await store.get(FISH, snapshot.id)

    assert loaded is not None
    assert loaded.document.model_dump() == {"summary": "旧概括", "audio": "旧转写"}


@pytest.mark.asyncio
async def test_legacy_compatibility_does_not_accept_unrelated_document_fields():
    snapshot = minute_snapshot(SemanticDocument(summary="旧概括"))
    row = {
        "id": snapshot.id,
        "fish_id": snapshot.fish_id,
        "payload": {
            "captureSessionId": str(snapshot.capture_session_id),
            "windowIndex": snapshot.window_index,
            "periodSeconds": snapshot.period_seconds,
            "inputDigest": snapshot.input_digest,
            "document": {"summary": "旧概括", "unexpected": "仍应拒绝"},
            "documentMarkdown": "旧字段",
            "sourceSnapshotIds": [],
        },
        "started_at": snapshot.started_at,
        "ended_at": snapshot.ended_at,
        "created_at": snapshot.created_at,
    }
    store = EnvironmentHistoryStore(cast(Any, StoredRowPool(row)))

    with pytest.raises(ValidationError):
        await store.get(FISH, snapshot.id)
