from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.models import Role
from fish_memory.snapshots.models import (
    ConfirmedCreate,
    ConfirmedIgnore,
    ConfirmedUpdate,
    PreUpdateMarker,
    SnapshotFact,
    SnapshotItemState,
    SnapshotPreUpdate,
)
from fish_memory.snapshots.operations import apply_confirmed_operations

NOW = datetime(2026, 7, 24, tzinfo=UTC)


def make_fact(number: int, text: str) -> SnapshotFact:
    return SnapshotFact(
        id=UUID(int=number),
        sequence=number,
        fish_id=UUID(int=20),
        source_message_id=UUID(int=30 + number),
        source_role=Role.USER,
        text=text,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW,
        entities=(),
    )


def make_item(*fact_ids: UUID) -> SnapshotItemState:
    return SnapshotItemState(
        id=7,
        summary="用户正在归还旧靴。",
        support_fact_ids=fact_ids,
        lineage_fact_ids=fact_ids,
        last_supported_at=NOW,
        item_version=3,
    )


def test_apply_confirmed_operations_updates_support_and_retains_lineage() -> None:
    old, removed, added = (
        make_fact(1, "用户正在归还旧靴。"),
        make_fact(2, "用户不再需要这双靴子。"),
        make_fact(3, "用户已归还旧靴。"),
    )
    mutations = apply_confirmed_operations(
        items=(make_item(old.id, removed.id),),
        new_facts=(added,),
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
        ),
        confirmed_operations=(
            ConfirmedUpdate(
                basis="旧事项已完成。",
                item_index=0,
                new_support_indexes=(0,),
                remove_support_indexes=(1,),
                summary="用户已经完成旧靴退回事项。",
            ),
        ),
    )

    mutation = mutations[0]
    assert mutation.action == "update"
    assert mutation.item_id == 7
    assert mutation.support_fact_ids == (old.id, added.id)
    assert mutation.lineage_fact_ids == (old.id, removed.id, added.id)
    assert mutation.removed_support_fact_ids == (removed.id,)
    assert mutation.new_support_fact_ids == (added.id,)


def test_apply_confirmed_operations_allows_replacing_all_old_support() -> None:
    old = make_fact(1, "用户正在归还旧靴。")
    added = make_fact(2, "用户已归还旧靴。")

    mutation = apply_confirmed_operations(
        items=(make_item(old.id),),
        new_facts=(added,),
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
        ),
        confirmed_operations=(
            ConfirmedUpdate(
                basis="旧事项已完成。",
                item_index=0,
                new_support_indexes=(0,),
                remove_support_indexes=(0,),
                summary="用户已经完成旧靴退回事项。",
            ),
        ),
    )[0]

    assert mutation.support_fact_ids == (added.id,)
    assert mutation.lineage_fact_ids == (old.id, added.id)


def test_apply_confirmed_operations_allows_removing_only_some_old_support() -> None:
    retained = make_fact(1, "用户仍在归还旧靴。")
    removed = make_fact(2, "用户已不再需要这双旧靴。")
    assigned = make_fact(3, "用户的取货计划尚未确定。")

    mutation = apply_confirmed_operations(
        items=(make_item(retained.id, removed.id),),
        new_facts=(assigned,),
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
        ),
        confirmed_operations=(
            ConfirmedUpdate(
                basis="仅一条旧证据不再支持该事项。",
                item_index=0,
                new_support_indexes=(),
                remove_support_indexes=(1,),
                summary="用户仍在归还旧靴。",
            ),
        ),
    )[0]

    assert mutation.support_fact_ids == (retained.id,)
    assert mutation.lineage_fact_ids == (retained.id, removed.id)
    assert mutation.new_support_fact_ids == ()
    assert mutation.last_supported_at == NOW


def test_apply_confirmed_operations_updates_before_appending_split_create() -> None:
    old = make_fact(1, "用户正在归还旧靴。")
    updated = make_fact(2, "用户已归还旧靴。")
    separate = make_fact(3, "用户正在计划取货。")

    mutations = apply_confirmed_operations(
        items=(make_item(old.id),),
        new_facts=(updated, separate),
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0, 1)),),
        ),
        confirmed_operations=(
            ConfirmedCreate(
                basis="出现独立的新事项。",
                new_support_indexes=(1,),
                summary="用户正在计划取货。",
            ),
            ConfirmedUpdate(
                basis="旧事项已完成。",
                item_index=0,
                new_support_indexes=(0,),
                remove_support_indexes=(0,),
                summary="用户已经完成旧靴退回事项。",
            ),
        ),
    )

    assert [mutation.action for mutation in mutations] == ["update", "create"]
    assert mutations[1].item_id is None
    assert mutations[1].support_fact_ids == (separate.id,)


def test_apply_confirmed_operations_allows_create_only_for_wrong_marker() -> None:
    old = make_fact(1, "用户正在归还旧靴。")
    separate = make_fact(2, "用户正在计划取货。")

    mutations = apply_confirmed_operations(
        items=(make_item(old.id),),
        new_facts=(separate,),
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
        ),
        confirmed_operations=(
            ConfirmedCreate(
                basis="不是原事项的一部分。",
                new_support_indexes=(0,),
                summary="用户正在计划取货。",
            ),
        ),
    )

    assert len(mutations) == 1
    assert mutations[0].action == "create"


def test_apply_confirmed_operations_consumes_static_ignore_without_mutation() -> None:
    old = make_fact(1, "用户长期使用中文。")
    temporary = make_fact(2, "用户这次要求使用英文。")

    mutations = apply_confirmed_operations(
        items=(make_item(old.id),),
        new_facts=(temporary,),
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
        ),
        confirmed_operations=(
            ConfirmedIgnore(
                basis="单次要求不改变长期偏好。",
                new_support_indexes=(0,),
            ),
        ),
        allow_static_ignores=True,
    )

    assert mutations == ()


def test_apply_confirmed_operations_rejects_fact_reused_by_ignore() -> None:
    old = make_fact(1, "用户长期使用中文。")
    temporary = make_fact(2, "用户这次要求使用英文。")

    with pytest.raises(ValueError, match="more than once"):
        apply_confirmed_operations(
            items=(make_item(old.id),),
            new_facts=(temporary,),
            pre_update=SnapshotPreUpdate(
                creates=(),
                updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
            ),
            confirmed_operations=(
                ConfirmedIgnore(
                    basis="临时要求。",
                    new_support_indexes=(0,),
                ),
                ConfirmedCreate(
                    basis="错误地重复使用。",
                    new_support_indexes=(0,),
                    summary="错误条目。",
                ),
            ),
            allow_static_ignores=True,
        )


def test_apply_confirmed_operations_rejects_static_ignore_by_default() -> None:
    old = make_fact(1, "用户长期使用中文。")
    temporary = make_fact(2, "用户这次要求使用英文。")

    with pytest.raises(ValueError, match="ignore operations are static-only"):
        apply_confirmed_operations(
            items=(make_item(old.id),),
            new_facts=(temporary,),
            pre_update=SnapshotPreUpdate(
                creates=(),
                updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
            ),
            confirmed_operations=(
                ConfirmedIgnore(
                    basis="临时要求。",
                    new_support_indexes=(0,),
                ),
            ),
        )


def test_apply_confirmed_operations_rejects_duplicate_new_fact_globally() -> None:
    old = make_fact(1, "用户正在归还旧靴。")
    added = make_fact(2, "用户已归还旧靴。")

    with pytest.raises(ValueError, match="more than once"):
        apply_confirmed_operations(
            items=(make_item(old.id),),
            new_facts=(added,),
            pre_update=SnapshotPreUpdate(
                creates=(),
                updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
            ),
            confirmed_operations=(
                ConfirmedUpdate(
                    basis="旧事项已完成。",
                    item_index=0,
                    new_support_indexes=(0,),
                    remove_support_indexes=(0,),
                    summary="用户已经完成旧靴退回事项。",
                ),
                ConfirmedCreate(
                    basis="出现独立的新事项。",
                    new_support_indexes=(0,),
                    summary="用户正在计划取货。",
                ),
            ),
        )


def test_apply_confirmed_operations_rejects_unknown_pre_update_item_index() -> None:
    added = make_fact(1, "用户已归还旧靴。")

    with pytest.raises(ValueError, match="outside its local item collection"):
        apply_confirmed_operations(
            items=(),
            new_facts=(added,),
            pre_update=SnapshotPreUpdate(
                creates=(),
                updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
            ),
            confirmed_operations=(
                ConfirmedCreate(
                    basis="出现独立的新事项。",
                    new_support_indexes=(0,),
                    summary="用户正在计划取货。",
                ),
            ),
        )


def test_apply_confirmed_operations_rejects_empty_marker_confirmation() -> None:
    old = make_fact(1, "用户正在归还旧靴。")
    assigned = make_fact(2, "用户已归还旧靴。")

    with pytest.raises(ValueError, match="must not be empty"):
        apply_confirmed_operations(
            items=(make_item(old.id),),
            new_facts=(assigned,),
            pre_update=SnapshotPreUpdate(
                creates=(),
                updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
            ),
            confirmed_operations={0: ()},
        )


def test_apply_confirmed_operations_does_not_mutate_fact_validity() -> None:
    old = make_fact(1, "用户正在归还旧靴。")
    added = make_fact(2, "用户已归还旧靴。")

    apply_confirmed_operations(
        items=(make_item(old.id),),
        new_facts=(added,),
        pre_update=SnapshotPreUpdate(
            creates=(),
            updates=(PreUpdateMarker(item_index=0, source_fact_indexes=(0,)),),
        ),
        confirmed_operations=(
            ConfirmedUpdate(
                basis="旧事项已完成。",
                item_index=0,
                new_support_indexes=(0,),
                remove_support_indexes=(0,),
                summary="用户已经完成旧靴退回事项。",
            ),
        ),
    )

    assert old == make_fact(1, "用户正在归还旧靴。")
