"""Keep the review timeline intact; spend the remaining budget on shared context."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fish_memory.clock import format_prompt_timestamp_text

_TIMELINE_FIELDS = (
    "id",
    "source",
    "type",
    "at",
    "finished_at",
    "status",
    "reason",
    "text",
    "text_truncated",
    "spoken_chars",
    "delivery",
    "message_id",
    "response_event_ids",
    "shared_response",
    "reply_to_message_id",
    "model",
    "generation_ms",
)


def pack_review_evidence(
    evidence: list[dict[str, Any]], *, max_chars: int = 32_000
) -> dict[str, Any]:
    """Intern repeated excerpts without dropping speech or user instructions.

    Raw evidence remains in the saved report. Plans never become speech evidence.
    When context still exceeds the budget, remove older auxiliary excerpts first
    and mark every affected turn. Oversized speech alone fails explicitly instead
    of silently choosing a different set of turns to judge.
    """
    timeline: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    identities: dict[str, str] = {}
    priorities: dict[str, tuple[int, int]] = {}

    def reference(kind: str, value: Any, priority: tuple[int, int]) -> str:
        entry = {"kind": kind, "value": value}
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        identifier = identities.get(key)
        if identifier is None:
            identifier = f"C{len(identities) + 1}"
            identities[key] = identifier
            contexts[identifier] = entry
        priorities[identifier] = max(priorities.get(identifier, priority), priority)
        return identifier

    for item in sorted(evidence, key=lambda e: datetime.fromisoformat(e["at"])):
        row: dict[str, Any] = {
            k: item[k] for k in _TIMELINE_FIELDS if item.get(k) is not None
        }
        for key in ("at", "finished_at"):
            if isinstance(row.get(key), str):
                row[key] = format_prompt_timestamp_text(row[key])
        frame = item.get("screen_frame")
        if isinstance(frame, dict) and frame:
            # Keep attachment provenance, not cache/transport diagnostics. The
            # original metadata remains in the saved evidence for inspection.
            screen_frame: dict[str, bool | str] = {"attached": True}
            row["screen_frame"] = screen_frame
            observed_at = frame.get("observedAt")
            if isinstance(observed_at, str) and observed_at:
                screen_frame["observed_at"] = format_prompt_timestamp_text(observed_at)
        # External messages are material. An idle plan is not something the pet
        # has said, and must not prime the judge's assessment of its contribution.
        if isinstance(item.get("data"), dict) and "text" in item["data"]:
            row["data"] = item["data"]
        refs: list[str] = []
        if item.get("references"):
            refs.append(reference("event_references", item["references"], (2, 0)))
        audit = item.get("context_audit")
        perception = item.get("perception")
        if isinstance(audit, dict):
            # v5 audits also carry perception. Intern that through the same
            # fragment path, so one changing paragraph cannot duplicate the
            # whole snapshot or evict another turn's available evidence.
            if not isinstance(perception, dict):
                perception = audit.get("perception")
            refs.append(
                reference(
                    "context_audit",
                    {k: v for k, v in audit.items() if k != "perception"},
                    (2, 0),
                )
            )
        if isinstance(perception, dict):
            row["perception_available"] = perception.get("available", False)
            row["perception_omitted_chars"] = perception.get("omitted_chars", 0)
            for field in ("earlier_excerpt", "text"):
                parts = [p for p in str(perception.get(field) or "").split("\n\n") if p]
                for index, part in enumerate(parts):
                    refs.append(
                        reference(
                            "perception_earlier"
                            if field == "earlier_excerpt"
                            else "perception",
                            part,
                            (
                                0 if field == "earlier_excerpt" else 1,
                                index - len(parts),
                            ),
                        )
                    )
        if refs:
            row["context_refs"] = refs
        timeline.append(row)

    packed: dict[str, Any] = {"evidence": timeline, "contexts": contexts}
    original_chars = len(
        json.dumps(
            [
                {k: v for k, v in e.items() if k not in {"plan", "data"}}
                for e in evidence
            ],
            ensure_ascii=False,
        )
    )
    omitted = 0
    # Shared/newer fragments and actual request audits survive longest. Never
    # reuse a later turn's context for an earlier turn: refs retain provenance.
    for identifier in sorted(contexts, key=lambda c: priorities[c]):
        if len(json.dumps(packed, ensure_ascii=False)) <= max_chars:
            break
        contexts.pop(identifier)
        omitted += 1
        for row in timeline:
            if identifier in row.get("context_refs", []):
                row["context_refs"] = [
                    c for c in row["context_refs"] if c != identifier
                ]
                row["context_budget_omitted"] = True
    packed_chars = len(json.dumps(packed, ensure_ascii=False))
    if packed_chars > max_chars:
        raise ValueError("Review speech timeline exceeds budget; no turns were dropped")
    packed["context_compression"] = {
        "original_chars": original_chars,
        "packed_chars": packed_chars,
        "shared_fragments": len(contexts),
        "omitted_fragments": omitted,
    }
    return packed
