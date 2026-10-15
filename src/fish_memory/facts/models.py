from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from fish_memory.facts.embedding import build_fact_embedding_text
from fish_memory.models import Message, Role


class FactStructuredSemantics(BaseModel):
    model_config = ConfigDict(frozen=True)

    what: str = Field(min_length=1)
    where: tuple[str, ...]
    who: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    text: str
    occurred_start: datetime | None
    occurred_end: datetime | None
    entities: tuple[str, ...]
    what: str | None = None
    where: tuple[str, ...] = field(default_factory=tuple)
    who: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structured(self) -> FactStructuredSemantics:
        """Return the durable structured semantics for this fact."""
        return FactStructuredSemantics(
            what=self.what or self.text,
            where=self.where,
            who=self.who,
        )

    @property
    def embedding_text(self) -> str:
        return build_fact_embedding_text(
            text=self.text,
            what=self.what,
            where=self.where,
            who=self.who,
            entities=self.entities,
        )


@dataclass(frozen=True, slots=True)
class FactExtractionContext:
    """A bounded adjacent turn used only to resolve the current message."""

    role: Role
    content: str
    external_event: str | None = None


@dataclass(frozen=True, slots=True)
class PendingFactMessage:
    message: Message
    target_sequence: int
    fact_extraction_attempt_count: int = 0
    previous_opposite_message: Message | None = None
    external_event: str | None = None


@dataclass(frozen=True, slots=True)
class BackgroundTaskProgress:
    task_name: str
    through_sequence: int
    target_sequence: int
    pending: int
    last_error: str | None
    updated_at: datetime | None
