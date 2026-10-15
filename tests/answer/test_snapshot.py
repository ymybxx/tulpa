from datetime import UTC, datetime, timedelta
from uuid import UUID

from fish_memory.answer.snapshot import render_dynamic_snapshot, render_static_profile
from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    NormalizedStaticProfile,
    SnapshotItemState,
)

NOW = datetime(2026, 7, 24, 12, tzinfo=UTC)


def item(item_id: int, summary: str, *, days_ago: int) -> SnapshotItemState:
    support_id = UUID(int=item_id)
    return SnapshotItemState(
        id=item_id,
        summary=summary,
        support_fact_ids=(support_id,),
        lineage_fact_ids=(support_id, UUID(int=100 + item_id)),
        last_supported_at=NOW - timedelta(days=days_ago),
        item_version=3,
    )


def test_dynamic_renders_unexpired_summaries_by_support_time_then_id() -> None:
    """Catches age filtering in storage, unstable ordering, or metadata rendering."""
    snapshot = NormalizedDynamicSnapshot(
        items=(
            item(5, "等待回复", days_ago=0),
            item(3, "开始讨论", days_ago=2),
            item(4, "同日第二项", days_ago=1),
            item(2, "同日第一项", days_ago=1),
            item(1, "已过期", days_ago=8),
        ),
        through_fact_sequence=5,
        version=2,
    )

    rendered = render_dynamic_snapshot(snapshot, NOW, timedelta(days=7))

    assert rendered == (
        "<dynamic_memory>\n"
        "开始讨论\n"
        "同日第一项\n"
        "同日第二项\n"
        "等待回复\n"
        "</dynamic_memory>"
    )
    assert "sourceFactIds" not in rendered
    assert "lineageFactIds" not in rendered
    assert "lastSupportedAt" not in rendered
    assert "itemVersion" not in rendered
    assert "已过期" not in rendered
    assert "<events>" not in rendered
    assert "<states>" not in rendered


def test_empty_normalized_snapshot_renders_nothing() -> None:
    snapshot = NormalizedDynamicSnapshot(
        items=(),
        through_fact_sequence=0,
        version=1,
    )

    assert render_dynamic_snapshot(snapshot, NOW, timedelta(days=7)) == ""


def test_static_renders_every_summary_without_category_sections_or_expiration() -> None:
    """Catches accidental Dynamic age rules or category sections in Static memory."""
    profile = NormalizedStaticProfile(
        items=(
            item(8, "用户喜欢篮球", days_ago=800),
            item(6, "小明是用户的大学室友", days_ago=1000),
            item(7, "fish 的名字是杰洛特", days_ago=900),
        ),
        through_fact_sequence=8,
        version=2,
    )

    rendered = render_static_profile(profile)

    assert rendered == (
        "<static_memory>\n"
        "小明是用户的大学室友\n"
        "fish 的名字是杰洛特\n"
        "用户喜欢篮球\n"
        "</static_memory>"
    )
    assert "<user_profile>" not in rendered
    assert "<assistant_profile>" not in rendered
    assert "<world_profile>" not in rendered
