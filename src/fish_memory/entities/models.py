from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from fish_memory.models import Role


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    id: UUID
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.aliases or any(not alias for alias in self.aliases):
            raise ValueError("entity candidate requires aliases")


@dataclass(frozen=True, slots=True)
class EntityContextMessage:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class SnapshotEntityContext:
    candidates: tuple[EntityCandidate, ...] = ()
    fact_entity_indexes: dict[UUID, tuple[int, ...]] = field(default_factory=dict)
    fact_source_roles: dict[UUID, Role] = field(default_factory=dict)
    fact_texts: dict[UUID, str] = field(default_factory=dict)
    recent_context: tuple[EntityContextMessage, ...] = ()

    def indexes_for_fact(self, fact_id: UUID) -> tuple[int, ...]:
        return self.fact_entity_indexes.get(fact_id, ())


@dataclass(frozen=True, slots=True)
class EntityMergeGroup:
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.aliases) < 2:
            raise ValueError("entity merge group requires at least two aliases")


@dataclass(frozen=True, slots=True)
class EntitySummary:
    entity_id: UUID
    aliases: tuple[str, ...]
    fact_count: int


@dataclass(frozen=True, slots=True)
class EntityFact:
    id: UUID
    text: str
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime
    source_role: str
    invalidated_at: datetime | None


@dataclass(frozen=True, slots=True)
class EntityFactPage:
    items: tuple[EntityFact, ...]
    has_more: bool
