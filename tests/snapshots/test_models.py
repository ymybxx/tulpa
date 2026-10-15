from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from fish_memory.snapshots.models import (
    ConfirmedCreate,
    ConfirmedIgnore,
    ConfirmedUpdate,
    NormalizedDynamicSnapshot,
    NormalizedStaticProfile,
    PreCreateOperation,
    PreUpdateMarker,
    SnapshotItemMutation,
    SnapshotItemState,
    SnapshotPreUpdate,
)


def test_snapshot_item_state_requires_a_stable_id_and_valid_support_lineage() -> None:
    fact_one = UUID(int=101)
    fact_two = UUID(int=102)
    state = SnapshotItemState(
        id=7,
        summary="用户正在归还旧靴。",
        support_fact_ids=(fact_one,),
        lineage_fact_ids=(fact_one, fact_two),
        last_supported_at=datetime(2026, 7, 24, tzinfo=UTC),
        item_version=2,
    )

    assert state.id == 7

    with pytest.raises(ValueError, match="stable item id"):
        SnapshotItemState(
            id=0,
            summary="用户正在归还旧靴。",
            support_fact_ids=(fact_one,),
            lineage_fact_ids=(fact_one,),
            last_supported_at=datetime(2026, 7, 24, tzinfo=UTC),
            item_version=1,
        )
    with pytest.raises(ValueError, match="current support"):
        SnapshotItemState(
            id=7,
            summary="用户正在归还旧靴。",
            support_fact_ids=(),
            lineage_fact_ids=(fact_one,),
            last_supported_at=datetime(2026, 7, 24, tzinfo=UTC),
            item_version=1,
        )
    with pytest.raises(ValueError, match="lineage"):
        SnapshotItemState(
            id=7,
            summary="用户正在归还旧靴。",
            support_fact_ids=(fact_one,),
            lineage_fact_ids=(),
            last_supported_at=datetime(2026, 7, 24, tzinfo=UTC),
            item_version=1,
        )
    with pytest.raises(ValueError, match="subset"):
        SnapshotItemState(
            id=7,
            summary="用户正在归还旧靴。",
            support_fact_ids=(fact_one,),
            lineage_fact_ids=(fact_two,),
            last_supported_at=datetime(2026, 7, 24, tzinfo=UTC),
            item_version=1,
        )


def test_pre_update_operations_validate_indexes_and_unique_assignments() -> None:
    pre_update = SnapshotPreUpdate(
        creates=(
            PreCreateOperation(
                summary="用户预约了取货。",
                source_fact_indexes=(0,),
            ),
        ),
        updates=(PreUpdateMarker(item_index=1, source_fact_indexes=(2,)),),
    )

    assert pre_update.updates[0].item_index == 1

    with pytest.raises(ValueError, match="non-negative"):
        PreUpdateMarker(item_index=-1, source_fact_indexes=(0,))
    with pytest.raises(ValueError, match="duplicate"):
        PreCreateOperation(summary="用户预约了取货。", source_fact_indexes=(0, 0))
    with pytest.raises(ValueError, match="already assigned"):
        SnapshotPreUpdate(
            creates=(
                PreCreateOperation(
                    summary="用户预约了取货。",
                    source_fact_indexes=(0,),
                ),
            ),
            updates=(PreUpdateMarker(item_index=1, source_fact_indexes=(0,)),),
        )


def test_confirmed_operations_validate_basis_and_local_indexes() -> None:
    assert (
        ConfirmedUpdate(
            basis="旧事项已完成。",
            item_index=0,
            new_support_indexes=(0,),
            remove_support_indexes=(1,),
            summary="用户已经完成旧事项。",
        ).summary
        == "用户已经完成旧事项。"
    )

    with pytest.raises(ValueError, match="100"):
        ConfirmedUpdate(
            basis="a" * 101,
            item_index=0,
            new_support_indexes=(0,),
            remove_support_indexes=(),
            summary="用户已经完成旧事项。",
        )
    with pytest.raises(ValueError, match="non-negative"):
        ConfirmedCreate(
            basis="独立主题",
            new_support_indexes=(-1,),
            summary="新事项",
        )
    with pytest.raises(ValueError, match="must not be empty"):
        ConfirmedIgnore(
            basis="短期事实",
            new_support_indexes=(),
        )


def test_confirmed_update_and_update_mutation_allow_removing_support_only() -> None:
    fact_one = UUID(int=101)
    fact_two = UUID(int=102)
    update = ConfirmedUpdate(
        basis="旧证据已不再支持当前事项。",
        item_index=0,
        new_support_indexes=(),
        remove_support_indexes=(1,),
        summary="用户仍在归还旧靴。",
    )
    mutation = SnapshotItemMutation(
        action="update",
        item_id=7,
        basis=update.basis,
        summary=update.summary,
        support_fact_ids=(fact_one,),
        lineage_fact_ids=(fact_one, fact_two),
        new_support_fact_ids=(),
        removed_support_fact_ids=(fact_two,),
        last_supported_at=datetime(2026, 7, 24, tzinfo=UTC),
    )

    assert update.new_support_indexes == ()
    assert mutation.new_support_fact_ids == ()


def test_confirmed_create_forbids_item_and_remove_indexes() -> None:
    with pytest.raises(TypeError):
        ConfirmedCreate(
            basis="独立主题",
            new_support_indexes=(0,),
            summary="新事项",
            item_index=3,
        )

    with pytest.raises(TypeError):
        ConfirmedCreate(
            basis="独立主题",
            new_support_indexes=(0,),
            summary="新事项",
            remove_support_indexes=(0,),
        )


def test_new_snapshot_models_do_not_carry_legacy_category_fields() -> None:
    for model in (
        SnapshotItemState,
        SnapshotPreUpdate,
        PreCreateOperation,
        PreUpdateMarker,
        ConfirmedUpdate,
        ConfirmedCreate,
        ConfirmedIgnore,
        SnapshotItemMutation,
    ):
        assert "category" not in model.__dataclass_fields__


def make_state(
    item_id: int,
    summary: str,
    supported_at: datetime,
) -> SnapshotItemState:
    fact_id = UUID(int=item_id)
    return SnapshotItemState(
        id=item_id,
        summary=summary,
        support_fact_ids=(fact_id,),
        lineage_fact_ids=(fact_id,),
        last_supported_at=supported_at,
        item_version=1,
    )


def test_normalized_aggregates_order_equal_times_by_stable_item_id() -> None:
    same_time = datetime(2026, 7, 24, tzinfo=UTC)
    old = make_state(9, "old", datetime(2026, 7, 20, tzinfo=UTC))
    higher_id = make_state(11, "higher", same_time)
    lower_id = make_state(10, "lower", same_time)

    dynamic = NormalizedDynamicSnapshot(
        items=(higher_id, old, lower_id),
        through_fact_sequence=4,
        version=2,
    )
    static = NormalizedStaticProfile(
        items=(higher_id, old, lower_id),
        through_fact_sequence=4,
        version=2,
    )

    assert dynamic.sorted_items() == (old, lower_id, higher_id)
    assert static.active_items() == (old, lower_id, higher_id)


def test_only_dynamic_aggregate_applies_working_memory_max_age() -> None:
    now = datetime(2026, 7, 24, tzinfo=UTC)
    recent = make_state(1, "recent", now - timedelta(days=2))
    old = make_state(2, "old", now - timedelta(days=8))
    dynamic = NormalizedDynamicSnapshot(
        items=(old, recent),
        through_fact_sequence=4,
        version=2,
    )
    static = NormalizedStaticProfile(
        items=(old, recent),
        through_fact_sequence=4,
        version=2,
    )

    assert dynamic.active_items(now, timedelta(days=7)) == (recent,)
    assert static.active_items() == (old, recent)
