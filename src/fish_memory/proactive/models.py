"""Expression and evidence records shared by Jev, delivery and autonomous tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

SourceKind = Literal[
    "screen", "system_audio", "dialogue", "operation", "memory", "recall", "web", "task"
]


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    kind: SourceKind
    text: str


@dataclass(frozen=True, slots=True)
class Focus:
    id: str
    revision: int
    topic: str
    progress: str
    sources: tuple[str, ...]
    state: Literal["observing", "awaiting_information"] = "observing"
    recheck_at: datetime | None = None
    settled: bool = False


@dataclass(frozen=True, slots=True)
class Supplement:
    kind: Literal["recall", "web", "task"]
    query: str
    text: str
    sources: tuple[str, ...] = ()


@dataclass(slots=True)
class ExpressionIntent:
    id: UUID
    fish_id: UUID
    session_id: UUID
    generation: int
    opportunity_id: str
    focus_id: str
    focus_revision: int
    kind: Literal["speak", "recall", "web", "task"]
    purpose: str
    created_at: datetime
    window_ended_at: datetime | None = None
    user_turn: int = 0
    contribution: str | None = None
    expects_reply: bool | None = None
    review: dict[str, object] | None = None
    follows: UUID | None = None
    state: Literal[
        "pending",
        "researching",
        "playing",
        "completed",
        "interrupted",
        "expired",
        "failed",
    ] = "pending"
    consumed_offset: int = 0
    text: str = ""
    supplement: Supplement | None = None
    timings: dict[str, float] = field(default_factory=dict)
    basis: Focus | None = None
    evidence: tuple[Evidence, ...] = ()
    # The Observation (a sight/sound summary or event) that made TA speak, if any.
    source_observation_id: UUID | None = None
