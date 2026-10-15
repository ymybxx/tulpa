from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fish_memory.answer.history_tokens import estimate_text_tokens
from fish_memory.clock import format_prompt_timestamp

CHUNK_SECONDS = 600
HISTORY_MAX_AGE = timedelta(hours=5)
SUMMARY_TOKEN_LIMIT = 2_000


@dataclass(frozen=True, slots=True)
class HistoryUnit:
    source_id: UUID
    label: str
    content: str
    started_at: datetime
    ended_at: datetime
    sequence: int = 0
    # Source-change notification metadata, never rendered into model History.
    triggers_decision: bool = field(default=True, compare=False, repr=False)
    interrupts_observation: bool = field(default=True, compare=False, repr=False)
    # The result of a search, recall or task TA itself started, or a message
    # sent in for the user: TA answers it even while 主动陪伴 is off.
    reports_work: bool = field(default=False, compare=False, repr=False)

    def render(self) -> str:
        return f"{self.label}: " + self.content.strip().replace("\n", "\n  ")


@dataclass(frozen=True, slots=True)
class HistoryChunk:
    id: UUID
    started_at: datetime
    ended_at: datetime
    revision: int = 0
    summary: str = ""
    units: tuple[HistoryUnit, ...] = ()
    unit_count: int = 0
    content_revision: int = 0

    def render(self, exclude: frozenset[UUID] = frozenset()) -> str:
        parts = [f"summary: {self.summary}"] if self.summary else []
        parts.extend(
            unit.render()
            for unit in sorted(
                self.units,
                key=lambda unit: (unit.started_at, unit.sequence, str(unit.source_id)),
            )
            if unit.source_id not in exclude
        )
        body = "\n".join(parts)
        if not body:
            return ""
        return f"{format_time(self.started_at)} - {format_time(self.ended_at)}\n{body}"

    @property
    def token_count(self) -> int:
        return estimate_text_tokens(self.render())


def format_time(value: datetime) -> str:
    return format_prompt_timestamp(value, sep=" ")


def chunk_bounds(value: datetime) -> tuple[datetime, datetime]:
    timestamp = int(value.timestamp())
    start = datetime.fromtimestamp(timestamp - timestamp % CHUNK_SECONDS, UTC)
    return start, start + timedelta(seconds=CHUNK_SECONDS)


def render_history(
    chunks: tuple[HistoryChunk, ...], *, exclude: frozenset[UUID] = frozenset()
) -> str:
    return "\n\n".join(text for chunk in chunks if (text := chunk.render(exclude)))


def select_prefix(
    chunks: tuple[HistoryChunk, ...], *, retain_tokens: int
) -> tuple[HistoryChunk, ...]:
    """Whole chunks, including an oversized newest chunk and previous summaries."""
    remaining = estimate_text_tokens(render_history(chunks))
    selected = []
    for chunk in chunks:
        if remaining < retain_tokens:
            break
        selected.append(chunk)
        remaining = estimate_text_tokens(render_history(chunks[len(selected) :]))
    return tuple(selected)
