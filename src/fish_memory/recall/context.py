import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from fish_memory.answer.history_tokens import estimate_text_tokens
from fish_memory.clock import format_prompt_timestamp
from fish_memory.recall.models import RecalledFact, RecallRelationUnit, RecallResult


@dataclass(frozen=True, slots=True)
class _ContextCandidate:
    object_id: int | UUID
    primary: bool
    budget_payload: dict[str, object]
    transport_payload: dict[str, object]


def build_recall_context(
    result: RecallResult,
    *,
    token_budget: int,
) -> dict[str, object]:
    """Render selected recall memory units in their service-defined order."""
    if token_budget < 1:
        raise ValueError("recall context token budget must be positive")
    if not result.facts:
        return {"u": []}
    return _build_relation_unit_context(result, token_budget)


def _build_relation_unit_context(
    result: RecallResult,
    token_budget: int,
) -> dict[str, object]:
    member_ids = {fact.id for unit in result.relation_units for fact in unit.facts}
    candidates_by_id: dict[int | UUID, _ContextCandidate] = {
        unit.id: _ContextCandidate(
            object_id=unit.id,
            primary=unit.primary,
            budget_payload=_relation_unit_budget_payload(unit),
            transport_payload=_relation_unit_transport(unit),
        )
        for unit in result.relation_units
    }
    candidates_by_id.update(
        {
            fact.id: _ContextCandidate(
                object_id=fact.id,
                primary=True,
                budget_payload={
                    "kind": "fact",
                    "priority": "primary",
                    "fact": _fact_budget_payload(fact),
                },
                transport_payload={
                    "t": "f",
                    **_fact_transport(fact),
                },
            )
            for fact in result.facts
            if fact.id not in member_ids
        }
    )
    default_order = (
        *(unit.id for unit in result.relation_units),
        *(fact.id for fact in result.facts if fact.id not in member_ids),
    )
    ordered_ids = tuple(dict.fromkeys((*result.memory_unit_order, *default_order)))
    candidates = [
        candidates_by_id[item_id]
        for item_id in ordered_ids
        if item_id in candidates_by_id
    ]
    transport_fact_ids_by_relation = {
        unit.id: tuple(fact.id for fact in _transport_relation_facts(unit))
        for unit in result.relation_units
    }

    selected = _select_candidates(candidates, token_budget)
    return {
        "u": _deduplicate_relation_transport_facts(
            selected,
            transport_fact_ids_by_relation,
        )
    }


def _select_candidates(
    candidates: Sequence[_ContextCandidate],
    token_budget: int,
) -> list[_ContextCandidate]:
    selected_ids: set[int | UUID] = set()
    prioritized = sorted(candidates, key=lambda candidate: not candidate.primary)
    for candidate in prioritized:
        proposed_items = [
            item.budget_payload
            for item in candidates
            if item.object_id in selected_ids or item.object_id == candidate.object_id
        ]
        proposed: dict[str, object] = {"items": proposed_items}
        if selected_ids and not _fits(proposed, token_budget):
            continue
        selected_ids.add(candidate.object_id)

    return [
        candidate for candidate in candidates if candidate.object_id in selected_ids
    ]


def _deduplicate_relation_transport_facts(
    candidates: Sequence[_ContextCandidate],
    fact_ids_by_relation: dict[int, tuple[UUID, ...]],
) -> list[dict[str, object]]:
    seen_fact_ids: set[UUID] = set()
    payloads: list[dict[str, object]] = []
    for candidate in candidates:
        fact_ids = fact_ids_by_relation.get(candidate.object_id)
        if fact_ids is None:
            payloads.append(candidate.transport_payload)
            continue
        facts = candidate.transport_payload["f"]
        if not isinstance(facts, list):
            raise TypeError("relation transport facts must be a list")
        unique_facts: list[object] = []
        for fact_id, fact_payload in zip(fact_ids, facts, strict=True):
            if fact_id in seen_fact_ids:
                continue
            seen_fact_ids.add(fact_id)
            unique_facts.append(fact_payload)
        payloads.append({**candidate.transport_payload, "f": unique_facts})
    return payloads


def _relation_unit_budget_payload(unit: RecallRelationUnit) -> dict[str, object]:
    facts = _budget_relation_facts(unit)
    payload: dict[str, object] = {
        "kind": "relation",
        "priority": "primary" if unit.primary else "reserve",
        "snapshotType": unit.snapshot_type,
        "lastSupportedAt": _format_budget_time(unit.last_supported_at),
        "facts": [_fact_budget_payload(fact) for fact in facts],
    }
    if unit.summary_eligible:
        payload["summary"] = unit.snapshot_text
    return payload


def _relation_unit_transport(unit: RecallRelationUnit) -> dict[str, object]:
    facts = _transport_relation_facts(unit)
    return {
        "t": "r",
        "f": [_fact_transport(fact) for fact in facts],
        "s": unit.snapshot_text,
        "start": _format_hour(min(fact.mentioned_at for fact in unit.facts)),
        "end": _format_hour(max(fact.mentioned_at for fact in unit.facts)),
    }


def _budget_relation_facts(
    unit: RecallRelationUnit,
) -> tuple[RecalledFact, ...]:
    anchor_source_facts = _anchor_source_relation_facts(unit)
    return (
        unit.facts
        if len(unit.facts) == 1
        or unit.primary
        or (not unit.summary_eligible and not anchor_source_facts)
        else anchor_source_facts
    )


def _transport_relation_facts(
    unit: RecallRelationUnit,
) -> tuple[RecalledFact, ...]:
    if unit.primary:
        return unit.facts
    return _anchor_source_relation_facts(unit)


def _anchor_source_relation_facts(
    unit: RecallRelationUnit,
) -> tuple[RecalledFact, ...]:
    anchor_ids = set(unit.anchor_fact_ids)
    anchor_source_message_ids = {
        fact.source_message_id for fact in unit.facts if fact.id in anchor_ids
    }
    return tuple(
        fact
        for fact in unit.facts
        if fact.source_message_id in anchor_source_message_ids
    )


def _fact_budget_payload(fact: RecalledFact) -> dict[str, object]:
    payload: dict[str, object] = {
        "sourceRole": fact.source_role.value,
        "text": fact.text,
    }
    if fact.occurred_start is not None:
        payload["occurredStart"] = _format_budget_time(fact.occurred_start)
    if fact.occurred_end is not None:
        payload["occurredEnd"] = _format_budget_time(fact.occurred_end)
    payload["mentionedAt"] = _format_budget_time(fact.mentioned_at)
    if fact.invalidated_at is not None:
        payload["validity"] = "invalidated"
    return payload


def _fact_transport(fact: RecalledFact) -> dict[str, object]:
    payload: dict[str, object] = {
        "x": fact.text,
        "m": _format_hour(fact.mentioned_at),
    }
    if fact.occurred_start is not None:
        start = _format_hour(fact.occurred_start)
        end = _format_hour(fact.occurred_end or fact.occurred_start)
        payload["o"] = start if start == end else [start, end]
    return payload


def _fits(payload: dict[str, object], token_budget: int) -> bool:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return estimate_text_tokens(text) <= token_budget


def _format_budget_time(value: datetime) -> str:
    # This legacy selection cost is never sent to a model. Keep admission
    # independent of display timezone and the compact transport representation.
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def _format_hour(value: datetime) -> str:
    return format_prompt_timestamp(value, timespec="hours")
