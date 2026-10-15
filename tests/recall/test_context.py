import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from fish_memory.models import Role
from fish_memory.recall.context import build_recall_context
from fish_memory.recall.models import (
    RecalledFact,
    RecallExecutionTrace,
    RecallRelationUnit,
    RecallResult,
)


def fact(
    value: int,
    *,
    mentioned_at: datetime,
    occurred_start: datetime | None = None,
    occurred_end: datetime | None = None,
    invalidated_at: datetime | None = None,
) -> RecalledFact:
    return RecalledFact(
        id=UUID(int=value),
        source_message_id=UUID(int=100 + value),
        source_role=Role.USER,
        text=f"fact-{value}",
        occurred_start=occurred_start,
        occurred_end=occurred_end,
        mentioned_at=mentioned_at,
        entities=("不应输出",),
        matched_by=("semantic",),
        semantic_score=0.9,
        rank_score=0.9,
        invalidated_at=invalidated_at,
    )


def result(
    facts: tuple[RecalledFact, ...],
    *,
    direct_fact_ids: tuple[UUID, ...],
    relation_units: tuple[RecallRelationUnit, ...] = (),
    memory_unit_order: tuple[int | UUID, ...] = (),
) -> RecallResult:
    return RecallResult(
        facts=facts,
        trace=RecallExecutionTrace(
            depth=2,
            degraded=False,
            plan=None,
            plan_error=None,
            duration_ms={},
        ),
        direct_fact_ids=direct_fact_ids,
        relation_units=relation_units,
        memory_unit_order=memory_unit_order,
    )


def relation_unit(
    facts: tuple[RecalledFact, ...],
    *,
    primary: bool,
    summary_eligible: bool = True,
) -> RecallRelationUnit:
    return RecallRelationUnit(
        id=10,
        snapshot_type="dynamic",
        snapshot_version=3,
        snapshot_text="西装待取、新靴待取、旧靴待退",
        text_hash="a" * 64,
        last_supported_at=datetime(2026, 8, 25, 8, tzinfo=UTC),
        facts=facts,
        anchor_fact_ids=(facts[0].id,),
        relation_text_score=0.9,
        unit_score=0.95,
        summary_eligible=summary_eligible,
        primary=primary,
    )


def _selected_unit_texts(
    payload: dict[str, object],
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Expose selected unit identity, order, and Primary/Reserve expansion."""
    selected: list[tuple[str, str, tuple[str, ...]]] = []
    for item in cast(list[dict[str, object]], payload["u"]):
        if item["t"] == "r":
            facts = cast(list[dict[str, object]], item["f"])
            selected.append(
                (
                    "relation",
                    cast(str, item["s"]),
                    tuple(cast(str, fact["x"]) for fact in facts),
                )
            )
        else:
            text = cast(str, item["x"])
            selected.append(("fact", text, (text,)))
    return selected


def test_compact_context_uses_hour_precision_and_relation_member_range() -> None:
    first = replace(
        fact(1, mentioned_at=datetime(2026, 7, 1, 9, 42, tzinfo=UTC)),
        text="该用户提出事项一",
    )
    second = replace(
        fact(
            2,
            mentioned_at=datetime(2026, 7, 2, 12, 59, tzinfo=UTC),
            occurred_start=datetime(2026, 6, 30, 23, 15, tzinfo=UTC),
            occurred_end=datetime(2026, 7, 1, 1, 45, tzinfo=UTC),
        ),
        source_role=Role.ASSISTANT,
        text="该助手说明事项二",
    )
    standalone = replace(
        fact(3, mentioned_at=datetime(2026, 7, 3, 8, 30, tzinfo=UTC)),
        text="该用户还有事项三",
    )
    unit = replace(
        relation_unit((first, second), primary=True),
        snapshot_text="摘要",
    )

    payload = build_recall_context(
        result(
            (first, second, standalone),
            direct_fact_ids=(first.id, standalone.id),
            relation_units=(unit,),
            memory_unit_order=(unit.id, standalone.id),
        ),
        token_budget=12_000,
    )

    assert payload == {
        "u": [
            {
                "t": "r",
                "f": [
                    {"x": "该用户提出事项一", "m": "2026-07-01T09+00:00"},
                    {
                        "x": "该助手说明事项二",
                        "m": "2026-07-02T12+00:00",
                        "o": ["2026-06-30T23+00:00", "2026-07-01T01+00:00"],
                    },
                ],
                "s": "摘要",
                "start": "2026-07-01T09+00:00",
                "end": "2026-07-02T12+00:00",
            },
            {"t": "f", "x": "该用户还有事项三", "m": "2026-07-03T08+00:00"},
        ]
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert serialized.index('"f":') < serialized.index('"s":')
    assert '"t":"r","f":' in serialized


def test_compact_context_collapses_same_hour_occurred_range() -> None:
    standalone = replace(
        fact(
            1,
            mentioned_at=datetime(2026, 7, 3, 8, 59, tzinfo=UTC),
            occurred_start=datetime(2026, 6, 30, 23, 1, tzinfo=UTC),
            occurred_end=datetime(2026, 6, 30, 23, 58, tzinfo=UTC),
        ),
        text="该用户提出事项一",
    )

    payload = build_recall_context(
        result(
            (standalone,),
            direct_fact_ids=(standalone.id,),
        ),
        token_budget=12_000,
    )

    assert payload == {
        "u": [
            {
                "t": "f",
                "x": "该用户提出事项一",
                "m": "2026-07-03T08+00:00",
                "o": "2026-06-30T23+00:00",
            }
        ]
    }


def test_relation_unit_selection_does_not_expand_with_compact_transport() -> None:
    primary_facts = (
        replace(
            fact(1, mentioned_at=datetime(2026, 7, 1, 9, tzinfo=UTC)),
            text="primary-member-one-" * 3,
        ),
        replace(
            fact(2, mentioned_at=datetime(2026, 7, 2, 9, tzinfo=UTC)),
            text="primary-member-two-" * 3,
        ),
    )
    reserve_facts = (
        replace(
            fact(3, mentioned_at=datetime(2026, 7, 3, 9, tzinfo=UTC)),
            text="reserve-anchor-" * 3,
        ),
        replace(
            fact(4, mentioned_at=datetime(2026, 7, 4, 9, tzinfo=UTC)),
            text="reserve-hidden-" * 3,
        ),
    )
    standalone = replace(
        fact(5, mentioned_at=datetime(2026, 7, 5, 9, tzinfo=UTC)),
        text="standalone-fact-" * 3,
    )
    primary = replace(
        relation_unit(primary_facts, primary=True),
        id=11,
        snapshot_text="primary-summary-" * 3,
    )
    reserve = replace(
        relation_unit(reserve_facts, primary=False),
        id=12,
        snapshot_text="reserve-summary-" * 3,
    )
    recall_result = result(
        (*primary_facts, *reserve_facts, standalone),
        direct_fact_ids=(primary_facts[0].id, reserve_facts[0].id, standalone.id),
        relation_units=(primary, reserve),
        memory_unit_order=(reserve.id, standalone.id, primary.id),
    )

    near_budget_payload = build_recall_context(recall_result, token_budget=200)
    all_units_payload = build_recall_context(recall_result, token_budget=296)

    assert _selected_unit_texts(near_budget_payload) == [
        (
            "fact",
            "standalone-fact-" * 3,
            ("standalone-fact-" * 3,),
        ),
        (
            "relation",
            "primary-summary-" * 3,
            ("primary-member-one-" * 3, "primary-member-two-" * 3),
        ),
    ]
    assert _selected_unit_texts(all_units_payload) == [
        (
            "relation",
            "reserve-summary-" * 3,
            ("reserve-anchor-" * 3,),
        ),
        (
            "fact",
            "standalone-fact-" * 3,
            ("standalone-fact-" * 3,),
        ),
        (
            "relation",
            "primary-summary-" * 3,
            ("primary-member-one-" * 3, "primary-member-two-" * 3),
        ),
    ]


def test_primary_relation_context_contains_summary_and_all_members() -> None:
    first = fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    second = fact(2, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC))
    unit = relation_unit((first, second), primary=True)

    payload = build_recall_context(
        result(
            (first, second),
            direct_fact_ids=(first.id,),
            relation_units=(unit,),
        ),
        token_budget=12_000,
    )

    assert payload == {
        "u": [
            {
                "t": "r",
                "s": "西装待取、新靴待取、旧靴待退",
                "start": "2026-08-24T00+00:00",
                "end": "2026-08-25T00+00:00",
                "f": [
                    {
                        "x": "fact-1",
                        "m": "2026-08-24T00+00:00",
                    },
                    {
                        "x": "fact-2",
                        "m": "2026-08-25T00+00:00",
                    },
                ],
            }
        ]
    }


def test_relation_fact_dedupe_uses_final_rank_and_fact_uuid_without_mutation() -> None:
    shared = replace(
        fact(1, mentioned_at=datetime(2026, 8, 23, tzinfo=UTC)),
        text="higher-shared",
    )
    lower_shared_copy = replace(shared, text="lower-copy-same-id")
    higher_unique = fact(2, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    lower_unique = fact(3, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC))
    higher = replace(
        relation_unit((shared, higher_unique), primary=False),
        id=11,
        snapshot_text="higher-summary",
        anchor_fact_ids=(shared.id, higher_unique.id),
    )
    lower = replace(
        relation_unit((lower_shared_copy, lower_unique), primary=True),
        id=12,
        snapshot_text="lower-summary",
    )
    recall_result = result(
        (shared, higher_unique, lower_unique),
        direct_fact_ids=(shared.id, higher_unique.id, lower_unique.id),
        relation_units=(higher, lower),
        memory_unit_order=(higher.id, lower.id),
    )
    before = deepcopy(recall_result)

    payload = build_recall_context(recall_result, token_budget=12_000)

    assert _selected_unit_texts(payload) == [
        ("relation", "higher-summary", ("higher-shared", "fact-2")),
        ("relation", "lower-summary", ("fact-3",)),
    ]
    assert recall_result == before


def test_relation_fact_dedupe_runs_after_budget_without_refilling_units() -> None:
    shared = replace(
        fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC)),
        text="shared-evidence-" * 20,
    )
    third_fact = replace(
        fact(2, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC)),
        text="third",
    )
    higher = replace(
        relation_unit((shared,), primary=True),
        id=11,
        snapshot_text="high",
    )
    duplicate = replace(
        relation_unit((shared,), primary=True),
        id=12,
        snapshot_text="duplicate",
    )
    third = replace(
        relation_unit((third_fact,), primary=True),
        id=13,
        snapshot_text="third",
    )

    payload = build_recall_context(
        result(
            (shared, third_fact),
            direct_fact_ids=(shared.id, third_fact.id),
            relation_units=(higher, duplicate, third),
            memory_unit_order=(higher.id, duplicate.id, third.id),
        ),
        token_budget=353,
    )

    assert _selected_unit_texts(payload) == [
        ("relation", "high", ("shared-evidence-" * 20,)),
        ("relation", "duplicate", ()),
    ]


def test_reserve_single_support_without_anchor_has_empty_transport_facts() -> None:
    only = fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    unit = replace(
        relation_unit((only,), primary=False),
        anchor_fact_ids=(),
    )

    payload = build_recall_context(
        result(
            (only,),
            direct_fact_ids=(only.id,),
            relation_units=(unit,),
        ),
        token_budget=12_000,
    )

    item = cast(list[dict[str, object]], payload["u"])[0]
    assert item["s"] == "西装待取、新靴待取、旧靴待退"
    assert item["f"] == []


def test_reserve_relation_only_expands_direct_anchor_facts() -> None:
    first = fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    second = fact(2, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC))
    unit = relation_unit((first, second), primary=False)

    payload = build_recall_context(
        result(
            (first, second),
            direct_fact_ids=(first.id,),
            relation_units=(unit,),
        ),
        token_budget=12_000,
    )

    item = cast(list[dict[str, object]], payload["u"])[0]
    assert "priority" not in item
    assert item["s"] == "西装待取、新靴待取、旧靴待退"
    assert (item["start"], item["end"]) == (
        "2026-08-24T00+00:00",
        "2026-08-25T00+00:00",
    )
    assert [fact["x"] for fact in cast(list[dict[str, object]], item["f"])] == [
        "fact-1"
    ]


def test_reserve_relation_expands_anchor_source_siblings_only() -> None:
    anchor = fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    sibling = replace(
        fact(2, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC)),
        source_message_id=anchor.source_message_id,
    )
    unrelated = fact(3, mentioned_at=datetime(2026, 8, 26, tzinfo=UTC))
    unit = relation_unit((anchor, sibling, unrelated), primary=False)

    payload = build_recall_context(
        result(
            (anchor, sibling, unrelated),
            direct_fact_ids=(anchor.id,),
            relation_units=(unit,),
        ),
        token_budget=12_000,
    )

    item = cast(list[dict[str, object]], payload["u"])[0]
    assert [fact["x"] for fact in cast(list[dict[str, object]], item["f"])] == [
        "fact-1",
        "fact-2",
    ]


def test_reserve_relation_budget_charges_anchor_source_siblings() -> None:
    anchor = replace(
        fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC)),
        text="anchor",
    )
    sibling = replace(
        fact(2, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC)),
        source_message_id=anchor.source_message_id,
        text="sibling-" * 120,
    )
    standalone = replace(
        fact(3, mentioned_at=datetime(2026, 8, 26, tzinfo=UTC)),
        text="standalone",
    )
    unit = replace(
        relation_unit((anchor, sibling), primary=False),
        snapshot_text="summary",
    )

    payload = build_recall_context(
        result(
            (anchor, sibling, standalone),
            direct_fact_ids=(anchor.id, standalone.id),
            relation_units=(unit,),
            memory_unit_order=(unit.id, standalone.id),
        ),
        token_budget=250,
    )

    assert _selected_unit_texts(payload) == [
        ("fact", "standalone", ("standalone",)),
    ]


def test_reserve_without_summary_eligibility_or_anchors_has_empty_transport_facts() -> (
    None
):
    first = fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    second = fact(2, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    unit = replace(
        relation_unit((first, second), primary=False, summary_eligible=False),
        anchor_fact_ids=(),
    )

    payload = build_recall_context(
        result(
            (first, second),
            direct_fact_ids=(),
            relation_units=(unit,),
        ),
        token_budget=12_000,
    )

    item = cast(list[dict[str, object]], payload["u"])[0]
    assert item["s"] == "西装待取、新靴待取、旧靴待退"
    assert item["f"] == []


def test_reserve_no_anchor_transport_does_not_reduce_legacy_selection_cost() -> None:
    standalone = replace(
        fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC)),
        text="standalone-" * 3,
    )
    singleton_fact = replace(
        fact(2, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC)),
        text="singleton-evidence-" * 6,
    )
    fallback_facts = (
        replace(
            fact(3, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC)),
            text="fallback-one-" * 6,
        ),
        replace(
            fact(4, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC)),
            text="fallback-two-" * 6,
        ),
    )
    singleton = replace(
        relation_unit((singleton_fact,), primary=False),
        id=11,
        snapshot_text="singleton-summary",
        anchor_fact_ids=(),
    )
    fallback = replace(
        relation_unit(
            fallback_facts,
            primary=False,
            summary_eligible=False,
        ),
        id=12,
        snapshot_text="fallback-summary",
        anchor_fact_ids=(),
    )
    recall_result = result(
        (standalone, singleton_fact, *fallback_facts),
        direct_fact_ids=(standalone.id,),
        relation_units=(singleton, fallback),
        memory_unit_order=(singleton.id, fallback.id, standalone.id),
    )

    def selected_labels(token_budget: int) -> list[tuple[str, object]]:
        payload = build_recall_context(recall_result, token_budget=token_budget)
        return [
            (cast(str, item["t"]), item.get("s", item.get("x")))
            for item in cast(list[dict[str, object]], payload["u"])
        ]

    assert selected_labels(140) == [("f", "standalone-" * 3)]
    assert selected_labels(162) == [
        ("r", "singleton-summary"),
        ("f", "standalone-" * 3),
    ]
    assert selected_labels(298) == [
        ("r", "singleton-summary"),
        ("r", "fallback-summary"),
        ("f", "standalone-" * 3),
    ]


def test_relation_context_preserves_service_memory_unit_order() -> None:
    relation_fact = fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    standalone = replace(
        fact(2, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC)),
        rank_score=0.1,
    )
    unit = relation_unit((relation_fact,), primary=True)

    payload = build_recall_context(
        result(
            (standalone, relation_fact),
            direct_fact_ids=(standalone.id,),
            relation_units=(unit,),
            memory_unit_order=(standalone.id, unit.id),
        ),
        token_budget=12_000,
    )

    items = cast(list[dict[str, object]], payload["u"])
    assert [item["t"] for item in items] == ["f", "r"]


def test_relation_context_budget_selects_primary_before_earlier_reserve() -> None:
    reserve_fact = fact(1, mentioned_at=datetime(2026, 8, 25, tzinfo=UTC))
    primary_fact = fact(2, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))
    reserve = relation_unit((reserve_fact,), primary=False)
    primary = replace(
        relation_unit((primary_fact,), primary=True),
        id=11,
    )

    payload = build_recall_context(
        result(
            (reserve_fact, primary_fact),
            direct_fact_ids=(reserve_fact.id, primary_fact.id),
            relation_units=(reserve, primary),
            memory_unit_order=(reserve.id, primary.id),
        ),
        token_budget=1,
    )

    items = cast(list[dict[str, object]], payload["u"])
    assert len(items) == 1
    assert items[0]["s"] == "西装待取、新靴待取、旧靴待退"
    assert cast(list[dict[str, object]], items[0]["f"])[0]["x"] == "fact-2"


def test_relation_first_standalone_fact_uses_memory_unit_payload() -> None:
    standalone = fact(1, mentioned_at=datetime(2026, 8, 24, tzinfo=UTC))

    payload = build_recall_context(
        result(
            (standalone,),
            direct_fact_ids=(standalone.id,),
        ),
        token_budget=12_000,
    )

    assert payload == {
        "u": [
            {
                "t": "f",
                "x": "fact-1",
                "m": "2026-08-24T00+00:00",
            }
        ]
    }
