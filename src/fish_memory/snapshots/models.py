from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from fish_memory.models import Role
from fish_memory.performance.models import JsonValue


@dataclass(frozen=True, slots=True)
class SnapshotItemState:
    """The category-free, server-owned state of one normalized snapshot item."""

    id: int
    summary: str
    support_fact_ids: tuple[UUID, ...]
    lineage_fact_ids: tuple[UUID, ...]
    last_supported_at: datetime
    item_version: int

    def __post_init__(self) -> None:
        if self.id < 1:
            raise ValueError("stable item id must be positive")
        _validate_summary(self.summary)
        _validate_fact_ids(self.support_fact_ids, "current support")
        _validate_fact_ids(self.lineage_fact_ids, "lineage")
        if not set(self.support_fact_ids).issubset(self.lineage_fact_ids):
            raise ValueError("current support facts must be a lineage subset")
        if self.last_supported_at.tzinfo is None:
            raise ValueError("last supported time must be timezone-aware")
        if self.item_version < 1:
            raise ValueError("item version must be positive")


@dataclass(frozen=True, slots=True)
class PreCreateOperation:
    summary: str
    source_fact_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_summary(self.summary)
        _validate_indexes(self.source_fact_indexes, "source fact")


@dataclass(frozen=True, slots=True)
class PreUpdateMarker:
    item_index: int
    source_fact_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_item_index(self.item_index)
        _validate_indexes(self.source_fact_indexes, "source fact")


@dataclass(frozen=True, slots=True)
class SnapshotPreUpdate:
    creates: tuple[PreCreateOperation, ...]
    updates: tuple[PreUpdateMarker, ...]

    def __post_init__(self) -> None:
        item_indexes = tuple(marker.item_index for marker in self.updates)
        if len(item_indexes) != len(set(item_indexes)):
            raise ValueError("duplicate pre-update item index")

        assigned_indexes = tuple(
            index
            for operation in self.creates
            for index in operation.source_fact_indexes
        ) + tuple(
            index for marker in self.updates for index in marker.source_fact_indexes
        )
        if len(assigned_indexes) != len(set(assigned_indexes)):
            raise ValueError("source fact index is already assigned")


@dataclass(frozen=True, slots=True)
class SnapshotPreUpdateResult:
    pre_update: SnapshotPreUpdate
    request_payload: JsonValue
    response_payload: JsonValue


@dataclass(frozen=True, slots=True)
class ConfirmedUpdate:
    basis: str
    item_index: int
    new_support_indexes: tuple[int, ...]
    remove_support_indexes: tuple[int, ...]
    summary: str

    def __post_init__(self) -> None:
        _validate_basis(self.basis)
        _validate_item_index(self.item_index)
        _validate_indexes(
            self.new_support_indexes,
            "new support",
            require_nonempty=False,
        )
        _validate_indexes(
            self.remove_support_indexes,
            "remove support",
            require_nonempty=False,
        )
        _validate_summary(self.summary)


@dataclass(frozen=True, slots=True)
class ConfirmedCreate:
    basis: str
    new_support_indexes: tuple[int, ...]
    summary: str

    def __post_init__(self) -> None:
        _validate_basis(self.basis)
        _validate_indexes(self.new_support_indexes, "new support")
        _validate_summary(self.summary)


@dataclass(frozen=True, slots=True)
class ConfirmedIgnore:
    """A Static-only confirmation that consumes facts without mutating memory."""

    basis: str
    new_support_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_basis(self.basis)
        _validate_indexes(self.new_support_indexes, "new support")


@dataclass(frozen=True, slots=True)
class SnapshotItemMutation:
    """A fully resolved create or update ready for normalized persistence."""

    action: Literal["create", "update"]
    item_id: int | None
    basis: str | None
    summary: str
    support_fact_ids: tuple[UUID, ...]
    lineage_fact_ids: tuple[UUID, ...]
    new_support_fact_ids: tuple[UUID, ...]
    removed_support_fact_ids: tuple[UUID, ...]
    last_supported_at: datetime

    def __post_init__(self) -> None:
        if self.action not in ("create", "update"):
            raise ValueError("mutation action must be create or update")
        if (self.action == "create") != (self.item_id is None):
            raise ValueError("create mutations have no item id; updates require one")
        if self.item_id is not None and self.item_id < 1:
            raise ValueError("stable item id must be positive")
        if self.basis is not None:
            _validate_basis(self.basis)
        _validate_summary(self.summary)
        _validate_fact_ids(self.support_fact_ids, "current support")
        _validate_fact_ids(self.lineage_fact_ids, "lineage")
        if not set(self.support_fact_ids).issubset(self.lineage_fact_ids):
            raise ValueError("current support facts must be a lineage subset")
        _validate_fact_ids(
            self.new_support_fact_ids,
            "new support",
            require_nonempty=self.action == "create",
        )
        _validate_fact_ids(
            self.removed_support_fact_ids,
            "removed support",
            require_nonempty=False,
        )
        if not set(self.new_support_fact_ids).issubset(self.support_fact_ids):
            raise ValueError("new support facts must support the resulting item")
        if set(self.removed_support_fact_ids) & set(self.support_fact_ids):
            raise ValueError(
                "removed support facts must not support the resulting item"
            )
        if self.last_supported_at.tzinfo is None:
            raise ValueError("last supported time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class NormalizedDynamicSnapshot:
    """A normalized Dynamic Snapshot backed by stable item rows."""

    items: tuple[SnapshotItemState, ...]
    through_fact_sequence: int
    version: int

    def __post_init__(self) -> None:
        _validate_snapshot_state(self.through_fact_sequence, self.version)

    def sorted_items(self) -> tuple[SnapshotItemState, ...]:
        return tuple(
            sorted(
                self.items,
                key=lambda item: (item.last_supported_at, item.id),
            )
        )

    def active_items(
        self,
        now: datetime,
        max_age: timedelta,
    ) -> tuple[SnapshotItemState, ...]:
        if now.tzinfo is None:
            raise ValueError("current time must be timezone-aware")
        return tuple(
            item
            for item in self.sorted_items()
            if now - item.last_supported_at <= max_age
        )


@dataclass(frozen=True, slots=True)
class NormalizedStaticProfile:
    """A normalized Static Profile backed by stable item rows."""

    items: tuple[SnapshotItemState, ...]
    through_fact_sequence: int
    version: int

    def __post_init__(self) -> None:
        _validate_snapshot_state(self.through_fact_sequence, self.version)

    def sorted_items(self) -> tuple[SnapshotItemState, ...]:
        return tuple(
            sorted(
                self.items,
                key=lambda item: (item.last_supported_at, item.id),
            )
        )

    def active_items(self) -> tuple[SnapshotItemState, ...]:
        return self.sorted_items()


@dataclass(frozen=True, slots=True)
class NormalizedSnapshotOverview:
    snapshot_type: str
    version: int
    updated_at: datetime
    through_fact_sequence: int
    items: tuple[SnapshotItemState, ...]


@dataclass(frozen=True, slots=True)
class SnapshotFact:
    id: UUID
    sequence: int
    fish_id: UUID
    source_message_id: UUID
    source_role: Role
    text: str
    occurred_start: datetime | None
    occurred_end: datetime | None
    mentioned_at: datetime
    entities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PendingNormalizedSnapshotBatch:
    fish_id: UUID
    session_id: UUID
    facts: tuple[SnapshotFact, ...]
    snapshot: NormalizedDynamicSnapshot
    target_fact_sequence: int
    trigger_message_id: UUID


@dataclass(frozen=True, slots=True)
class PendingNormalizedStaticProfileBatch:
    fish_id: UUID
    facts: tuple[SnapshotFact, ...]
    profile: NormalizedStaticProfile
    target_fact_sequence: int
    trigger_message_id: UUID


def _validate_snapshot_state(through_fact_sequence: int, version: int) -> None:
    if through_fact_sequence < 0:
        raise ValueError("through fact sequence must not be negative")
    if version < 0:
        raise ValueError("snapshot version must not be negative")


def _validate_summary(summary: str) -> None:
    if not summary.strip():
        raise ValueError("snapshot summary must not be blank")


def _validate_basis(basis: str) -> None:
    if not basis.strip():
        raise ValueError("mutation basis must not be blank")
    if len(basis) > 100:
        raise ValueError("mutation basis must be at most 100 characters")


def _validate_item_index(item_index: int) -> None:
    if isinstance(item_index, bool) or item_index < 0:
        raise ValueError("item index must be non-negative")


def _validate_indexes(
    indexes: tuple[int, ...],
    name: str,
    *,
    require_nonempty: bool = True,
) -> None:
    if require_nonempty and not indexes:
        raise ValueError(f"{name} indexes must not be empty")
    if any(isinstance(index, bool) or index < 0 for index in indexes):
        raise ValueError(f"{name} indexes must be non-negative")
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"{name} indexes must not contain duplicates")


def _validate_fact_ids(
    fact_ids: tuple[UUID, ...],
    name: str,
    *,
    require_nonempty: bool = True,
) -> None:
    if require_nonempty and not fact_ids:
        raise ValueError(f"{name} facts must not be empty")
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError(f"{name} facts must not contain duplicates")
