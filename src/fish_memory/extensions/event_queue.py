"""FIFO admission; only adjacent requests with the same definition may merge."""

from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import Any
from uuid import UUID

from fish_memory.extensions.models import EventDefinition, Instance

MAX_PENDING_EVENTS = 256
MAX_MERGED_EVENTS = 32
MAX_MERGED_DATA_CHARS = 16_000


@dataclass(frozen=True, slots=True)
class PendingEvent:
    instance: Instance
    definition: EventDefinition
    data: dict[str, Any]
    saved: dict[str, Any]
    data_chars: int

    @property
    def event_id(self) -> UUID:
        return UUID(self.saved["event_id"])

    @property
    def observation_id(self) -> UUID:
        return UUID(self.saved["observation_id"])


def next_batch(queue: deque[PendingEvent]) -> tuple[PendingEvent, ...]:
    first = queue[0]
    batch = [first]
    size = first.data_chars
    if first.definition.mergeable:
        for pending in islice(queue, 1, MAX_MERGED_EVENTS):
            if (
                pending.instance is not first.instance
                or pending.definition != first.definition
                or size + pending.data_chars > MAX_MERGED_DATA_CHARS
            ):
                break
            batch.append(pending)
            size += pending.data_chars
    return tuple(batch)
