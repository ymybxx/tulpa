from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from fish_memory.entities.normalization import normalize_entity_alias
from fish_memory.models import Role
from fish_memory.performance.models import JsonValue


class RecallTimeField(StrEnum):
    OCCURRED = "occurred"
    MENTIONED = "mentioned"
    EITHER = "either"


class RecallOrder(StrEnum):
    RELEVANCE = "relevance"
    LATEST = "latest"
    EARLIEST = "earliest"


class RecallSourceRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    EITHER = "either"


class RecallTimeFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: RecallTimeField | None
    start: AwareDatetime | None
    end: AwareDatetime | None

    @model_validator(mode="after")
    def validate_filter(self) -> Self:
        if self.field is None and (self.start is not None or self.end is not None):
            raise ValueError("time field is required when a time bound is set")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("time end must not precede start")
        return self


class RecallTimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: AwareDatetime | None
    end: AwareDatetime | None

    @field_validator("start", "end", mode="before")
    @classmethod
    def reject_coercive_time_bounds(cls, value: object) -> object:
        if value is None or isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            raise ValueError("time bound must be a timezone-qualified ISO string")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exception:
            raise ValueError(
                "time bound must be a timezone-qualified ISO string"
            ) from exception
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("time bound must be a timezone-qualified ISO string")
        return value

    @property
    def has_bounds(self) -> bool:
        return self.start is not None or self.end is not None

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("time end must not precede start")
        return self


class RecallQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    semantic_query: str = Field(alias="semanticQuery", min_length=1)
    source_role: RecallSourceRole = Field(alias="sourceRole")
    entity_hints: tuple[str, ...] = Field(
        alias="entityHints",
        default_factory=tuple,
    )
    time: RecallTimeRange

    @field_validator("semantic_query")
    @classmethod
    def reject_blank_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("semantic query must not be blank")
        return value

    @field_validator("entity_hints")
    @classmethod
    def normalize_entity_hints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                hint for value in values if (hint := normalize_entity_alias(value))
            )
        )


@dataclass(frozen=True, slots=True)
class PlannedRecall:
    plan: RecallQueryPlan
    request_payload: JsonValue
    response_payload: JsonValue


@dataclass(frozen=True, slots=True)
class RecallCandidate:
    id: UUID
    source_message_id: UUID
    source_role: Role
    text: str
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime
    entities: tuple[str, ...]
    semantic_score: float | None = None
    invalidated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RecallFactHead:
    fact: RecallCandidate
    relation_item_ids: tuple[int, ...]
    embedding_text: str | None
    text_hash: str | None
    entity_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallRelationHead:
    id: int
    snapshot_type: str
    snapshot_version: int
    snapshot_text: str
    text_hash: str
    last_supported_at: datetime
    support_fact_ids: tuple[UUID, ...]
    support_count: int
    semantic_score: float


@dataclass(frozen=True, slots=True)
class RecalledFact:
    id: UUID
    source_message_id: UUID
    source_role: Role
    text: str
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime
    entities: tuple[str, ...]
    matched_by: tuple[str, ...]
    semantic_score: float | None
    rank_score: float | None
    invalidated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RecallRelationItem:
    id: int
    snapshot_type: str
    support_fact_ids: tuple[UUID, ...]
    snapshot_version: int


@dataclass(frozen=True, slots=True)
class RecallRelationUnit:
    id: int
    snapshot_type: str
    snapshot_version: int
    snapshot_text: str
    text_hash: str
    last_supported_at: datetime
    facts: tuple[RecalledFact, ...]
    anchor_fact_ids: tuple[UUID, ...]
    relation_text_score: float
    unit_score: float
    summary_eligible: bool
    primary: bool = True


@dataclass(frozen=True, slots=True)
class RecallExecutionTrace:
    depth: int
    degraded: bool
    plan: RecallQueryPlan | None
    plan_error: str | None
    duration_ms: dict[str, float]
    planner_request_payload: JsonValue = None
    planner_response_payload: JsonValue = None


@dataclass(frozen=True, slots=True)
class RecallResult:
    facts: tuple[RecalledFact, ...]
    trace: RecallExecutionTrace
    direct_fact_ids: tuple[UUID, ...] = ()
    relation_items: tuple[RecallRelationItem, ...] = ()
    relation_units: tuple[RecallRelationUnit, ...] = ()
    memory_unit_order: tuple[int | UUID, ...] = ()
