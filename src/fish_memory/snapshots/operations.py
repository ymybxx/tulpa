"""Pure validation and application of normalized snapshot item operations."""

from collections.abc import Mapping, Sequence
from typing import TypeVar
from uuid import UUID

from fish_memory.snapshots.models import (
    ConfirmedCreate,
    ConfirmedIgnore,
    ConfirmedUpdate,
    SnapshotFact,
    SnapshotItemMutation,
    SnapshotItemState,
    SnapshotPreUpdate,
)

ConfirmedOperation = ConfirmedUpdate | ConfirmedCreate | ConfirmedIgnore
_Value = TypeVar("_Value")


def apply_confirmed_operations(
    *,
    items: Sequence[SnapshotItemState],
    new_facts: Sequence[SnapshotFact],
    pre_update: SnapshotPreUpdate,
    confirmed_operations: (
        Sequence[ConfirmedOperation] | Mapping[int, Sequence[ConfirmedOperation]]
    ),
    allow_static_ignores: bool = False,
) -> tuple[SnapshotItemMutation, ...]:
    """Resolve local LLM indexes into immutable persistence mutations.

    A sequence of confirmed operations is accepted when exactly one pre-update
    marker exists.  Callers with several markers must provide a mapping keyed
    by the original marker's ``item_index`` so each confirmation keeps its own
    local new- and old-fact index spaces.
    """

    _validate_item_ids(items)
    _validate_new_facts(new_facts)
    marker_operations = _group_confirmed_operations(
        pre_update,
        confirmed_operations,
    )

    mutations: list[SnapshotItemMutation] = []
    used_new_fact_ids: set[UUID] = set()

    for operation in pre_update.creates:
        selected_facts = _select_indexes(new_facts, operation.source_fact_indexes)
        _claim_new_facts(selected_facts, used_new_fact_ids)
        mutations.append(
            _create_mutation(
                basis=None,
                summary=operation.summary,
                selected_facts=selected_facts,
            )
        )

    updates: list[SnapshotItemMutation] = []
    confirmed_creates: list[SnapshotItemMutation] = []
    updated_item_indexes: set[int] = set()
    for marker in pre_update.updates:
        local_new_facts = _select_indexes(new_facts, marker.source_fact_indexes)
        old_item = _select_indexes(
            items,
            (marker.item_index,),
            collection_name="item",
        )[0]
        for operation in marker_operations[marker.item_index]:
            if isinstance(operation, ConfirmedIgnore):
                if not allow_static_ignores:
                    raise ValueError("ignore operations are static-only")
                selected_facts = _select_indexes(
                    local_new_facts,
                    operation.new_support_indexes,
                )
                _claim_new_facts(selected_facts, used_new_fact_ids)
            elif isinstance(operation, ConfirmedUpdate):
                if operation.item_index != marker.item_index:
                    raise ValueError("confirmed update targets the wrong item index")
                if marker.item_index in updated_item_indexes:
                    raise ValueError("item index has more than one confirmed update")
                selected_facts = _select_indexes(
                    local_new_facts,
                    operation.new_support_indexes,
                )
                removed_fact_ids = _select_indexes(
                    old_item.support_fact_ids,
                    operation.remove_support_indexes,
                )
                _claim_new_facts(selected_facts, used_new_fact_ids)
                updates.append(
                    _update_mutation(
                        old_item,
                        operation,
                        selected_facts,
                        removed_fact_ids,
                    )
                )
                updated_item_indexes.add(marker.item_index)
            else:
                selected_facts = _select_indexes(
                    local_new_facts,
                    operation.new_support_indexes,
                )
                _claim_new_facts(selected_facts, used_new_fact_ids)
                confirmed_creates.append(
                    _create_mutation(
                        basis=operation.basis,
                        summary=operation.summary,
                        selected_facts=selected_facts,
                    )
                )

    # The database transaction updates existing IDs before it inserts rows for
    # creates.  Mirroring that order keeps this pure planning result explicit.
    return tuple(updates + mutations + confirmed_creates)


def _group_confirmed_operations(
    pre_update: SnapshotPreUpdate,
    confirmed_operations: (
        Sequence[ConfirmedOperation] | Mapping[int, Sequence[ConfirmedOperation]]
    ),
) -> dict[int, Sequence[ConfirmedOperation]]:
    marker_indexes = {marker.item_index for marker in pre_update.updates}
    if isinstance(confirmed_operations, Mapping):
        if set(confirmed_operations) != marker_indexes:
            raise ValueError("confirmed operations must cover each pre-update marker")
        grouped_operations = dict(confirmed_operations)
        _validate_nonempty_confirmations(grouped_operations)
        return grouped_operations
    if not pre_update.updates:
        if confirmed_operations:
            raise ValueError("confirmed operations require a pre-update marker")
        return {}
    if len(pre_update.updates) != 1:
        raise ValueError("multiple pre-update markers require grouped confirmations")
    grouped_operations = {pre_update.updates[0].item_index: confirmed_operations}
    _validate_nonempty_confirmations(grouped_operations)
    return grouped_operations


def _update_mutation(
    old_item: SnapshotItemState,
    operation: ConfirmedUpdate,
    selected_facts: tuple[SnapshotFact, ...],
    removed_fact_ids: tuple[UUID, ...],
) -> SnapshotItemMutation:
    removed_ids = set(removed_fact_ids)
    selected_ids = tuple(fact.id for fact in selected_facts)
    support_fact_ids = _append_distinct(
        tuple(
            fact_id
            for fact_id in old_item.support_fact_ids
            if fact_id not in removed_ids
        ),
        selected_ids,
    )
    lineage_fact_ids = _append_distinct(old_item.lineage_fact_ids, selected_ids)
    last_supported_at = (
        max(fact.mentioned_at for fact in selected_facts)
        if selected_facts
        else old_item.last_supported_at
    )
    return SnapshotItemMutation(
        action="update",
        item_id=old_item.id,
        basis=operation.basis,
        summary=operation.summary,
        support_fact_ids=support_fact_ids,
        lineage_fact_ids=lineage_fact_ids,
        new_support_fact_ids=selected_ids,
        removed_support_fact_ids=tuple(removed_fact_ids),
        last_supported_at=last_supported_at,
    )


def _create_mutation(
    *,
    basis: str | None,
    summary: str,
    selected_facts: tuple[SnapshotFact, ...],
) -> SnapshotItemMutation:
    selected_ids = tuple(fact.id for fact in selected_facts)
    return SnapshotItemMutation(
        action="create",
        item_id=None,
        basis=basis,
        summary=summary,
        support_fact_ids=selected_ids,
        lineage_fact_ids=selected_ids,
        new_support_fact_ids=selected_ids,
        removed_support_fact_ids=(),
        last_supported_at=max(fact.mentioned_at for fact in selected_facts),
    )


def _select_indexes(
    values: Sequence[_Value],
    indexes: Sequence[int],
    *,
    collection_name: str = "fact",
) -> tuple[_Value, ...]:
    try:
        return tuple(values[index] for index in indexes)
    except IndexError as error:
        raise ValueError(
            f"operation index is outside its local {collection_name} collection"
        ) from error


def _claim_new_facts(
    facts: Sequence[SnapshotFact],
    used_new_fact_ids: set[UUID],
) -> None:
    fact_ids = {fact.id for fact in facts}
    if used_new_fact_ids & fact_ids:
        raise ValueError("new fact may not be used more than once")
    used_new_fact_ids.update(fact_ids)


def _append_distinct(
    existing: Sequence[_Value],
    additions: Sequence[_Value],
) -> tuple[_Value, ...]:
    result = list(existing)
    known = set(existing)
    for value in additions:
        if value not in known:
            result.append(value)
            known.add(value)
    return tuple(result)


def _validate_item_ids(items: Sequence[SnapshotItemState]) -> None:
    item_ids = tuple(item.id for item in items)
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("snapshot items must have unique stable item ids")


def _validate_new_facts(new_facts: Sequence[SnapshotFact]) -> None:
    fact_ids = tuple(fact.id for fact in new_facts)
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("new facts must have unique ids")


def _validate_nonempty_confirmations(
    grouped_operations: Mapping[int, Sequence[ConfirmedOperation]],
) -> None:
    if any(not operations for operations in grouped_operations.values()):
        raise ValueError("confirmed operations for each marker must not be empty")
