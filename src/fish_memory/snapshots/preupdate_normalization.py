"""Narrow provider-output normalization shared by Snapshot pre-updaters."""

import json
from dataclasses import dataclass

from fish_memory.performance.models import JsonValue


@dataclass(frozen=True, slots=True)
class NormalizedPreUpdateOutput:
    parsed: bool
    payload: object
    dropped_operations: tuple[dict[str, JsonValue], ...]
    normalizations: tuple[dict[str, JsonValue], ...]


def normalize_preupdate_output(
    content: str,
    *,
    item_count: int,
    fact_count: int,
) -> NormalizedPreUpdateOutput:
    """Apply only fail-closed, evidence-safe normalization to valid operations."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return NormalizedPreUpdateOutput(False, content, (), ())
    if not isinstance(payload, dict):
        return NormalizedPreUpdateOutput(True, payload, (), ())

    normalized = dict(payload)
    dropped: list[dict[str, JsonValue]] = []
    normalizations: list[dict[str, JsonValue]] = []
    original_indexes: dict[str, list[int]] = {}
    for collection_name, kind in (("creates", "create"), ("updates", "update")):
        raw_operations = payload.get(collection_name)
        if not isinstance(raw_operations, list):
            continue
        kept: list[object] = []
        kept_original_indexes: list[int] = []
        for index, operation in enumerate(raw_operations):
            if _is_evidence_free_operation(
                operation,
                kind=kind,
                item_count=item_count,
            ):
                dropped.append(
                    {
                        "kind": kind,
                        "index": index,
                        "reason": "empty_source_fact_indexes",
                    }
                )
                continue
            normalized_operation, diagnostic = _deduplicate_source_indexes(
                operation,
                kind=kind,
                index=index,
                item_count=item_count,
                fact_count=fact_count,
            )
            kept.append(normalized_operation)
            kept_original_indexes.append(index)
            if diagnostic is not None:
                normalizations.append(diagnostic)
        normalized[collection_name] = kept
        original_indexes[collection_name] = kept_original_indexes

    owners: dict[int, list[tuple[str, str, int]]] = {}
    for collection_name, kind in (("creates", "create"), ("updates", "update")):
        operations = normalized.get(collection_name)
        if not isinstance(operations, list):
            continue
        indexes = original_indexes.get(collection_name, [])
        for operation, original_index in zip(operations, indexes, strict=True):
            source_indexes = _valid_source_indexes(
                operation,
                kind=kind,
                item_count=item_count,
                fact_count=fact_count,
            )
            if source_indexes is None:
                continue
            for source_index in source_indexes:
                owners.setdefault(source_index, []).append(
                    (collection_name, kind, original_index)
                )

    ambiguous_indexes = {
        source_index
        for source_index, assignments in owners.items()
        if len(assignments) > 1
    }
    if ambiguous_indexes:
        for collection_name, kind in (
            ("creates", "create"),
            ("updates", "update"),
        ):
            operations = normalized.get(collection_name)
            if not isinstance(operations, list):
                continue
            kept = []
            kept_original_indexes = []
            indexes = original_indexes.get(collection_name, [])
            for operation, original_index in zip(operations, indexes, strict=True):
                source_indexes = _valid_source_indexes(
                    operation,
                    kind=kind,
                    item_count=item_count,
                    fact_count=fact_count,
                )
                conflicts = (
                    []
                    if source_indexes is None
                    else [
                        index for index in source_indexes if index in ambiguous_indexes
                    ]
                )
                if conflicts:
                    dropped.append(
                        {
                            "kind": kind,
                            "index": original_index,
                            "reason": "ambiguous_source_fact_indexes",
                            "sourceFactIndexes": conflicts,
                        }
                    )
                    continue
                kept.append(operation)
                kept_original_indexes.append(original_index)
            normalized[collection_name] = kept
            original_indexes[collection_name] = kept_original_indexes
    return NormalizedPreUpdateOutput(
        True,
        normalized,
        tuple(dropped),
        tuple(normalizations),
    )


def _is_evidence_free_operation(
    operation: object,
    *,
    kind: str,
    item_count: int,
) -> bool:
    if not _has_valid_operation_shape(
        operation,
        kind=kind,
        item_count=item_count,
    ):
        return False
    assert isinstance(operation, dict)
    value = operation.get("sourceFactIndexes")
    if not isinstance(value, list) or value:
        return False
    return True


def _deduplicate_source_indexes(
    operation: object,
    *,
    kind: str,
    index: int,
    item_count: int,
    fact_count: int,
) -> tuple[object, dict[str, JsonValue] | None]:
    indexes = _valid_source_indexes(
        operation,
        kind=kind,
        item_count=item_count,
        fact_count=fact_count,
    )
    if indexes is None:
        return operation, None
    assert isinstance(operation, dict)
    deduplicated = list(dict.fromkeys(indexes))
    if deduplicated == indexes:
        return operation, None
    normalized = {**operation, "sourceFactIndexes": deduplicated}
    return normalized, {
        "kind": kind,
        "index": index,
        "field": "sourceFactIndexes",
        "original": indexes,
        "normalized": deduplicated,
        "reason": "duplicate_source_fact_indexes",
    }


def _valid_source_indexes(
    operation: object,
    *,
    kind: str,
    item_count: int,
    fact_count: int,
) -> list[int] | None:
    if not _has_valid_operation_shape(
        operation,
        kind=kind,
        item_count=item_count,
    ):
        return None
    assert isinstance(operation, dict)
    value = operation.get("sourceFactIndexes")
    if not isinstance(value, list) or not value:
        return None
    indexes: list[int] = []
    for raw_index in value:
        if type(raw_index) is not int or not 0 <= raw_index < fact_count:
            return None
        indexes.append(raw_index)
    return indexes


def _has_valid_operation_shape(
    operation: object,
    *,
    kind: str,
    item_count: int,
) -> bool:
    if not isinstance(operation, dict):
        return False
    if kind == "create":
        if set(operation) != {"summary", "sourceFactIndexes"}:
            return False
        summary = operation.get("summary")
        return isinstance(summary, str) and bool(summary.strip())
    if set(operation) != {"itemIndex", "sourceFactIndexes"}:
        return False
    item_index = operation.get("itemIndex")
    return type(item_index) is int and 0 <= item_index < item_count
