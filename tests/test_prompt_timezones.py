import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.answer.context import build_chat_messages
from fish_memory.clock import format_prompt_timestamp, localize_passage_timestamps
from fish_memory.experimental.channels import render_state
from fish_memory.extensions.event_context import event_data_view, event_history_view
from fish_memory.extensions.models import EventDefinition
from fish_memory.extensions.review_evidence import pack_review_evidence
from fish_memory.facts.prompt import build_fact_extraction_messages
from fish_memory.history.models import HistoryChunk, HistoryUnit
from fish_memory.models import MessageStatus, Role
from fish_memory.observations.context import _query_window
from fish_memory.recall.context import build_recall_context
from fish_memory.recall.prompt import build_recall_plan_messages
from fish_memory.snapshots.confirmation_prompt import (
    build_snapshot_confirmation_messages,
)
from fish_memory.snapshots.prompt import build_snapshot_preupdate_messages
from fish_memory.snapshots.static_confirmation_prompt import (
    build_static_profile_confirmation_messages,
)
from fish_memory.snapshots.static_prompt import build_static_profile_preupdate_messages
from tests.answer.test_context import message
from tests.facts.test_extractor import make_message
from tests.recall.test_context import fact, result
from tests.snapshots.test_updater import make_fact, make_item

NOW = datetime(2026, 9, 19, 16, 18, 43, tzinfo=UTC)


@pytest.mark.parametrize(
    ("zone", "instant", "expected"),
    [
        ("Asia/Shanghai", NOW, "2026-09-20T00:18:43+08:00"),
        ("Asia/Kathmandu", NOW, "2026-09-19T22:03:43+05:45"),
        ("America/New_York", NOW, "2026-09-19T12:18:43-04:00"),
        (
            "America/New_York",
            datetime(2026, 1, 1, tzinfo=UTC),
            "2025-12-31T19:00:00-05:00",
        ),
    ],
)
def test_model_clock_uses_system_zone_for_each_instant(
    system_timezone, zone, instant, expected
):
    system_timezone(zone)
    assert format_prompt_timestamp(instant) == expected
    assert datetime.fromisoformat(expected) == instant
    system_timezone("UTC")
    assert format_prompt_timestamp(instant) == instant.isoformat()
    with pytest.raises(ValueError, match="timezone-aware"):
        format_prompt_timestamp(instant.replace(tzinfo=None))


def test_current_user_and_flat_history_share_the_local_date(system_timezone):
    system_timezone("Asia/Shanghai")
    user = message(
        sequence=1,
        role=Role.USER,
        content="原话 16:18 不应被改写",
        status=MessageStatus.COMPLETED,
        created_at=NOW,
    )
    messages = build_chat_messages("system", [user])
    assert "2026-09-20 00:18:43+08:00" in messages[-1]["content"]
    assert "原话 16:18 不应被改写" in messages[-1]["content"]
    state = render_state(messages, NOW, "用户已说完")
    assert state.endswith("当前时间：2026-09-20T00:18:43+08:00")
    chunk = HistoryChunk(
        UUID(int=3),
        NOW,
        NOW,
        summary="此前一起看过节目。",
        units=(HistoryUnit(user.id, "u", user.content, NOW, NOW),),
    )
    assert chunk.render().startswith(
        "2026-09-20 00:18:43+08:00 - 2026-09-20 00:18:43+08:00"
    )
    assert chunk.started_at is NOW and user.created_at is NOW


def test_recall_calendar_tracks_local_month_and_weekday(system_timezone):
    system_timezone("Asia/Shanghai")
    messages = build_recall_plan_messages(
        "上周做了什么？", datetime(2024, 2, 29, 17, tzinfo=UTC)
    )
    assert json.loads(messages[-1]["content"])["currentTime"] == (
        "2024-03-01T01:00:00+08:00"
    )
    assert "March 2024" in messages[0]["content"]
    assert "(Friday)" in messages[0]["content"]
    assert "February 2024" not in messages[0]["content"]


def test_recall_fact_transport_uses_local_hours(system_timezone):
    system_timezone("Asia/Shanghai")
    recalled = fact(1, mentioned_at=NOW, occurred_start=NOW, occurred_end=NOW)
    payload = build_recall_context(
        result((recalled,), direct_fact_ids=(recalled.id,)), token_budget=1000
    )
    assert payload["u"][0]["m"] == "2026-09-20T00+08:00"
    assert payload["u"][0]["o"] == "2026-09-20T00+08:00"
    assert recalled.mentioned_at is NOW


def test_fact_extraction_reference_time_uses_system_zone(system_timezone):
    system_timezone("America/New_York")
    source = replace(make_message(), created_at=NOW)
    payload = json.loads(
        build_fact_extraction_messages(source, "今天发生的事")[-1]["content"]
    )
    assert payload["referenceTime"] == "2026-09-19T12:18:43-04:00"


@pytest.mark.parametrize("static", [False, True])
def test_snapshot_update_and_confirmation_use_local_fact_times(system_timezone, static):
    system_timezone("Asia/Shanghai")
    item = replace(make_item(), last_supported_at=NOW)
    support = replace(
        make_fact(0), mentioned_at=NOW, occurred_start=NOW, occurred_end=NOW
    )
    update = (
        build_static_profile_preupdate_messages
        if static
        else build_snapshot_preupdate_messages
    )
    confirm = (
        build_static_profile_confirmation_messages
        if static
        else build_snapshot_confirmation_messages
    )
    payload = json.loads(update([item], [support], NOW)[-1]["content"])
    expected = "2026-09-20T00:18:43+08:00"
    assert payload["currentTime"] == expected
    assert payload["existingItems"][0]["lastSupportedAt"] == expected
    for field in ("mentionedAt", "occurredStart", "occurredEnd"):
        assert payload["newFacts"][0][field] == expected
    confirmation = json.loads(
        confirm(
            item_index=0,
            item=item,
            support_facts=[support],
            new_facts=[support],
            recent_turns=[],
        )[-1]["content"]
    )
    assert confirmation["supportFacts"][0]["mentionedAt"] == expected
    assert confirmation["newFacts"][0]["occurredStart"] == expected


def test_plugin_timestamp_projection_does_not_mutate_source_or_plain_text(
    system_timezone,
):
    system_timezone("Asia/Shanghai")
    raw = {"type": "test", "received_at": "2026-09-19T16:18:43Z"}
    assert (
        event_history_view([raw], {})[0]["received_at"] == "2026-09-20T00:18:43+08:00"
    )
    assert raw["received_at"].endswith("Z")
    definition = EventDefinition.model_validate(
        {
            "type": "test",
            "schema": {
                "type": "object",
                "properties": {
                    "at": {"type": "string", "format": "date-time"},
                    "text": {"type": "string"},
                },
            },
        }
    )
    projected = event_data_view(
        {"at": raw["received_at"], "text": raw["received_at"]}, definition
    )
    assert projected == {"at": "2026-09-20T00:18:43+08:00", "text": raw["received_at"]}
    evidence = pack_review_evidence(
        [{"id": "one", "at": raw["received_at"], "text": "hi"}]
    )
    assert evidence["evidence"][0]["at"] == "2026-09-20T00:18:43+08:00"


def test_old_av_provenance_is_projected_without_rewriting_words(system_timezone):
    system_timezone("Asia/Shanghai")
    source = "2026-09-19 16:18:40 — 2026-09-19 16:18:45 UTC\n声音A：16:18 见。"
    assert localize_passage_timestamps(source) == (
        "2026-09-20T00:18:40+08:00 — 2026-09-20T00:18:45+08:00\n声音A：16:18 见。"
    )
    assert source.startswith("2026-09-19 16:18:40")


def test_observation_yesterday_uses_local_dst_midnights(system_timezone):
    system_timezone("America/New_York")
    span = _query_window(
        "我昨天在电脑上干了什么？", datetime(2026, 3, 9, 16, tzinfo=UTC)
    )
    assert span is not None
    start, end = span
    assert start.isoformat() == "2026-03-08T00:00:00-05:00"
    assert end.isoformat() == "2026-03-09T00:00:00-04:00"
    assert (end - start).total_seconds() == 23 * 3600
