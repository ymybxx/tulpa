"""Quote-backed audio events, isolated from desktop and visual observations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

NARRATIVE_VERSION = 3
# Version four marks leaves produced from contemporaneous frames and words.
# Paragraph/provenance serialization remains version three and backward readable.
JOINT_WINDOW_VERSION = 4


class NarrativeParagraph(BaseModel):
    """A coherent passage with provenance kept out of the displayed prose."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    text: str = Field(min_length=1, max_length=8000)
    source_ids: tuple[str, ...] = Field(alias="sourceIds", min_length=1, max_length=64)

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("narrative paragraph is blank")
        return value.strip()


def render_paragraphs(paragraphs: tuple[NarrativeParagraph, ...]) -> str:
    return "\n\n".join(paragraph.text for paragraph in paragraphs)


def parse_narrative(
    raw: str, sources: Mapping[str, str]
) -> tuple[NarrativeParagraph, ...]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"paragraphs"}:
        raise ValueError("narrative response must contain paragraphs")
    entries = value["paragraphs"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 32:
        raise ValueError("invalid narrative paragraph count")
    keys = tuple(sources)
    voice_annotations = _voice_annotations(sources)
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"text", "sources"}:
            raise ValueError("invalid narrative paragraph")
        indexes = entry["sources"]
        if not isinstance(indexes, list) or not 1 <= len(indexes) <= 64:
            raise ValueError("narrative needs current sources")
        if any(type(i) is not int or not 1 <= i <= len(keys) for i in indexes):
            raise ValueError("narrative reference is not a current source")
        result.append(
            NarrativeParagraph(
                text=(
                    voice_annotations.sub("", entry["text"])
                    if voice_annotations is not None and isinstance(entry["text"], str)
                    else entry["text"]
                ),
                source_ids=tuple(dict.fromkeys(keys[i - 1] for i in indexes)),
            )
        )
    return tuple(result)


def _voice_annotations(sources: Mapping[str, str]) -> re.Pattern[str] | None:
    """Remove only machine labels parenthesized after an existing role noun.

    Never remove standalone subjects or labels actually spoken in the source.
    The original source passages/IDs retain all attribution for inspection.
    """
    labels = set()
    bodies = []
    pattern = re.compile(
        r"^\[\d{2,}:\d{2}\.\d{2}[—–-]\d{2,}:\d{2}\.\d{2}\]"
        r"\s*(声音[a-zA-Z0-9_-]{1,64})：(.*)$"
    )
    for words in sources.values():
        for line in words.splitlines():
            if match := pattern.fullmatch(line):
                labels.add(match[1])
                bodies.append(match[2])
    labels = {label for label in labels if not any(label in body for body in bodies)}
    if not labels:
        return None
    alternatives = "|".join(re.escape(label) for label in sorted(labels))
    return re.compile(r"(?<=\w)[（(]\s*(?:" + alternatives + r")\s*[）)]")


def narrative_story_schema(max_sources: int | None = None) -> str:
    source_index: dict[str, object] = {"type": "integer", "minimum": 1}
    if max_sources is not None:
        source_index["maximum"] = max_sources
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["paragraphs"],
            "properties": {
                "paragraphs": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["text", "sources"],
                        "properties": {
                            "text": {"type": "string", "minLength": 1},
                            "sources": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 64,
                                "items": source_index,
                            },
                        },
                    },
                }
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class NarrativeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_id: str = Field(alias="sourceId", min_length=1, max_length=100)
    quote: str = Field(min_length=1, max_length=1200)


class NarrativeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: str = Field(alias="eventId", min_length=1, max_length=100)
    subject: str = Field(default="", max_length=160)
    action: str = Field(default="", max_length=240)
    object: str = Field(default="", max_length=160)
    status: Literal[
        "requested",
        "planned",
        "waiting",
        "reported",
        "completed",
        "cancelled",
        "refused",
        "uncertain",
    ]
    narration: str = Field(min_length=1, max_length=1200)
    evidence: tuple[NarrativeEvidence, ...] = Field(min_length=1, max_length=6)


_STATE_LABELS = {
    "requested": "提出请求",
    "planned": "计划",
    "waiting": "等待中",
    "reported": "提及或评价",
    "completed": "已发生",
    "cancelled": "已取消",
    "refused": "被拒绝",
    "uncertain": "尚不确定",
}


def render_events(events: tuple[NarrativeEvent, ...]) -> str:
    """Keep the state visible instead of trusting a second free-form summary."""
    return "\n".join(
        f"{event.narration}（{_STATE_LABELS[event.status]}）" for event in events
    )


def parse_events(raw: str, sources: Mapping[str, str]) -> tuple[NarrativeEvent, ...]:
    """Reject the whole response on unsupported evidence; never drop a bad turn."""
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"events"}:
        raise ValueError("narrative response must contain only events")
    if not isinstance(value["events"], list) or not 1 <= len(value["events"]) <= 80:
        raise ValueError("narrative response needs between one and eighty events")
    result = []
    for index, event_value in enumerate(value["events"]):
        if not isinstance(event_value, dict) or "eventId" in event_value:
            raise ValueError("invalid narrative event")
        digest = hashlib.sha256(
            json.dumps(event_value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:24]
        references = event_value.get("evidence")
        if not isinstance(references, list) or not references:
            raise ValueError("narrative needs current source references")
        evidence = []
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {"sourceId"}:
                raise ValueError("narrative references must select sourceId only")
            source_index = reference.get("sourceId")
            if type(source_index) is not int or not 1 <= source_index <= len(sources):
                raise ValueError("narrative reference is not a current source")
            source_id = tuple(sources)[source_index - 1]
            if not sources[source_id].strip():
                raise ValueError("narrative reference has no current words")
            evidence.append({"sourceId": source_id, "quote": sources[source_id]})
        event_value = dict(event_value)
        if "summary" not in event_value or "narration" in event_value:
            raise ValueError("narrative event needs summary")
        event_value["narration"] = event_value.pop("summary")
        event = NarrativeEvent.model_validate(
            {**event_value, "eventId": f"event:{digest}:{index}", "evidence": evidence}
        )
        result.append(event)
    return tuple(result)


def narrative_schema() -> str:
    schema = NarrativeEvent.model_json_schema(by_alias=True)
    schema["properties"].pop("eventId")
    for field in ("subject", "action", "object"):
        schema["properties"].pop(field)
    schema["required"].remove("eventId")
    schema["properties"]["summary"] = schema["properties"].pop("narration")
    schema["properties"]["summary"]["title"] = "Summary"
    schema["required"].remove("narration")
    schema["required"].append("summary")
    # Inline defs live at the outer schema so standard JSON Schema references work.
    defs = schema.pop("$defs", {})
    evidence = defs["NarrativeEvidence"]
    evidence["properties"].pop("quote")
    evidence["required"].remove("quote")
    evidence["properties"]["sourceId"] = {"type": "integer", "minimum": 1}
    return json.dumps(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["events"],
            "properties": {
                "events": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 80,
                    "items": schema,
                }
            },
            "$defs": defs,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def preserve_unmentioned_sources(
    events: tuple[NarrativeEvent, ...],
    sources: Mapping[str, str],
) -> tuple[NarrativeEvent, ...]:
    """Keep omitted utterances as unresolved evidence, not invented story facts."""
    mentioned = {e.source_id for event in events for e in event.evidence}
    pending = [
        (index, source_id, quote)
        for index, (source_id, quote) in enumerate(sources.items())
        if source_id not in mentioned
    ]
    extras = []
    while pending:
        group = [pending.pop(0)]
        while (
            pending
            and len(group) < 6
            and pending[0][0] == group[-1][0] + 1
            and sum(len(text) for _, _, text in [*group, pending[0]]) < 1000
        ):
            group.append(pending.pop(0))
        # Do not group across an interpreted turn: that could move a later
        # decision before an intervening cancellation when sorted by first time.
        evidence = tuple(
            NarrativeEvidence(source_id=key, quote=text) for _, key, text in group
        )
        digest = hashlib.sha256(
            "\n".join(key for _, key, _ in group).encode()
        ).hexdigest()[:24]
        extras.append(
            NarrativeEvent(
                event_id=f"unresolved:{digest}",
                subject="未确定的声音",
                action="听到原话",
                object="",
                status="uncertain",
                narration="补充听到的话：\n" + "\n".join(text for _, _, text in group),
                evidence=evidence,
            )
        )
    order = {key: index for index, key in enumerate(sources)}
    return tuple(
        sorted(
            (*events, *extras),
            key=lambda event: min(
                order[evidence.source_id] for evidence in event.evidence
            ),
        )
    )
