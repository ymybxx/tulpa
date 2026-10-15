from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.relations import models
from fish_memory.snapshots.models import SnapshotItemState

FISH_ID = UUID(int=1)
SESSION_ID = UUID(int=2)
FACT_ID = UUID(int=3)
NOW = datetime(2026, 8, 27, tzinfo=UTC)


def current_item(item_id: int = 7) -> SnapshotItemState:
    return SnapshotItemState(
        id=item_id,
        summary="用户需要取回西装。",
        support_fact_ids=(FACT_ID,),
        lineage_fact_ids=(FACT_ID,),
        last_supported_at=NOW,
        item_version=3,
    )


def test_pending_projection_batch_allows_progress_only_work() -> None:
    batch = models.PendingSnapshotItemProjectionBatch(
        fish_id=FISH_ID,
        snapshot_id=11,
        session_id=SESSION_ID,
        snapshot_type="dynamic",
        snapshot_version=4,
        items=(),
        embedding_states=(),
    )

    assert batch.snapshot_version == 4
    assert batch.items == ()


def test_pending_projection_batch_rejects_duplicate_item_ids() -> None:
    with pytest.raises(ValueError, match="duplicate pending snapshot item"):
        models.PendingSnapshotItemProjectionBatch(
            fish_id=FISH_ID,
            snapshot_id=11,
            session_id=SESSION_ID,
            snapshot_type="dynamic",
            snapshot_version=4,
            items=(current_item(), current_item()),
            embedding_states=(),
        )


@pytest.mark.parametrize(
    ("snapshot_type", "session_id"),
    (("dynamic", None), ("static", SESSION_ID)),
)
def test_pending_projection_batch_rejects_invalid_scope(
    snapshot_type: str,
    session_id: UUID | None,
) -> None:
    with pytest.raises(ValueError, match="snapshot item projection scope"):
        models.PendingSnapshotItemProjectionBatch(
            fish_id=FISH_ID,
            snapshot_id=11,
            session_id=session_id,
            snapshot_type=snapshot_type,
            snapshot_version=4,
            items=(),
            embedding_states=(),
        )


def test_embedding_state_must_belong_to_a_current_item() -> None:
    state = models.SnapshotItemEmbeddingState(
        item_id=99,
        item_version=2,
        text_hash="a" * 64,
        provider="diagnostic-provider",
        model="diagnostic-model",
        dimensions=3,
    )

    with pytest.raises(ValueError, match="embedding state is not current"):
        models.PendingSnapshotItemProjectionBatch(
            fish_id=FISH_ID,
            snapshot_id=11,
            session_id=SESSION_ID,
            snapshot_type="dynamic",
            snapshot_version=4,
            items=(current_item(),),
            embedding_states=(state,),
        )
