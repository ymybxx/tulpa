from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from fish_memory.snapshots.models import SnapshotItemState


@dataclass(frozen=True, slots=True)
class SnapshotItemEmbeddingState:
    item_id: int
    item_version: int
    text_hash: str
    provider: str
    model: str
    dimensions: int

    def __post_init__(self) -> None:
        _validate_item_version(self.item_id, self.item_version)
        _validate_hash(self.text_hash)
        _validate_space(self.provider, self.model, self.dimensions)


@dataclass(frozen=True, slots=True)
class SnapshotItemEmbeddingProjection:
    item_id: int
    item_version: int
    session_id: UUID | None
    snapshot_type: str
    snapshot_text: str
    text_hash: str
    last_supported_at: datetime
    vector: tuple[float, ...] | None

    def __post_init__(self) -> None:
        _validate_item_version(self.item_id, self.item_version)
        _validate_scope(self.snapshot_type, self.session_id)
        if not self.snapshot_text.strip():
            raise ValueError("snapshot item projection text must not be blank")
        _validate_hash(self.text_hash)
        if self.last_supported_at.tzinfo is None:
            raise ValueError("snapshot item projection time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class PendingSnapshotItemProjectionBatch:
    fish_id: UUID
    snapshot_id: int
    session_id: UUID | None
    snapshot_type: str
    snapshot_version: int
    items: tuple[SnapshotItemState, ...]
    embedding_states: tuple[SnapshotItemEmbeddingState, ...]

    def __post_init__(self) -> None:
        if self.snapshot_id < 1:
            raise ValueError("snapshot id must be positive")
        if self.snapshot_version < 1:
            raise ValueError("snapshot version must be positive")
        _validate_scope(self.snapshot_type, self.session_id)
        item_ids = tuple(item.id for item in self.items)
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("duplicate pending snapshot item")
        state_ids = tuple(state.item_id for state in self.embedding_states)
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("duplicate snapshot item embedding state")
        if not set(state_ids).issubset(item_ids):
            raise ValueError("embedding state is not current")


def _validate_scope(snapshot_type: str, session_id: UUID | None) -> None:
    if not (
        (snapshot_type == "dynamic" and session_id is not None)
        or (snapshot_type == "static" and session_id is None)
    ):
        raise ValueError("invalid snapshot item projection scope")


def _validate_item_version(item_id: int, item_version: int) -> None:
    if item_id < 1:
        raise ValueError("snapshot item id must be positive")
    if item_version < 1:
        raise ValueError("snapshot item version must be positive")


def _validate_hash(value: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("snapshot item text hash must be lowercase SHA-256")


def _validate_space(provider: str, model: str, dimensions: int) -> None:
    if not provider.strip() or not model.strip():
        raise ValueError("snapshot item embedding diagnostics must not be blank")
    if not 1 <= dimensions <= 2000:
        raise ValueError("snapshot item embedding dimensions are invalid")
