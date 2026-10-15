from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias
from uuid import UUID

from fish_memory.entities.normalization import normalize_entity_alias


@dataclass(frozen=True, slots=True)
class EntityAliases:
    entity_id: UUID
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.aliases:
            raise ValueError("mask entity requires aliases")


MaskObjectId: TypeAlias = UUID | int


@dataclass(frozen=True, slots=True)
class MaskEmbedding:
    object_id: MaskObjectId
    entity_ids: tuple[UUID, ...]
    signature_hash: str
    source_text_hash: str
    masked_text: str
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MaskPlan:
    signature: tuple[UUID, ...]
    signature_hash: str
    masked_query: str
    masked_key: str


@dataclass(frozen=True, slots=True)
class SingleMaskSource:
    kind: str
    object_id: MaskObjectId
    fish_id: UUID
    source_text: str
    source_text_hash: str
    entity: EntityAliases

    def __post_init__(self) -> None:
        if self.kind not in {"fact", "relation"}:
            raise ValueError("single mask source kind is invalid")


def entity_signature(entity_ids: tuple[UUID, ...]) -> tuple[UUID, ...]:
    return tuple(sorted(set(entity_ids), key=lambda value: value.int))


def signature_hash(signature: tuple[UUID, ...]) -> str:
    normalized = entity_signature(signature)
    if not normalized:
        raise ValueError("mask signature must not be empty")
    digest = hashlib.sha256()
    for entity_id in normalized:
        digest.update(entity_id.bytes)
    return digest.hexdigest()


def mask_text(text: str, entities: tuple[EntityAliases, ...]) -> str | None:
    if not text or not entities:
        return None
    normalized_text, source_spans = _normalized_text_with_source_spans(text)
    aliases_by_entity: dict[UUID, set[str]] = {}
    for entity in entities:
        aliases_by_entity.setdefault(entity.entity_id, set()).update(
            alias
            for value in entity.aliases
            if (alias := normalize_entity_alias(value))
        )
    if not aliases_by_entity or any(
        not aliases for aliases in aliases_by_entity.values()
    ):
        return None

    spans_by_entity: dict[UUID, tuple[tuple[int, int], ...]] = {}
    for entity_id, aliases in aliases_by_entity.items():
        matches: list[tuple[int, int]] = []
        for alias in sorted(aliases, key=lambda value: (-len(value), value)):
            for start, end in _alias_occurrences(normalized_text, alias):
                original_start = source_spans[start][0]
                original_end = source_spans[end - 1][1]
                matches.append((original_start, original_end))
        merged = _merge_spans(matches)
        if not merged:
            return None
        spans_by_entity[entity_id] = merged

    entity_items = list(spans_by_entity.items())
    for index, (_entity_id, spans) in enumerate(entity_items):
        for _other_id, other_spans in entity_items[index + 1 :]:
            if any(
                left_start < right_end and right_start < left_end
                for left_start, left_end in spans
                for right_start, right_end in other_spans
            ):
                return None

    removed = [False] * len(text)
    for spans in spans_by_entity.values():
        for start, end in spans:
            for index in range(start, end):
                removed[index] = True
    masked = "".join(
        character for index, character in enumerate(text) if not removed[index]
    )
    normalized_masked = " ".join(masked.split())
    return normalized_masked or None


def build_mask_plan(
    query: str,
    key: str,
    query_entities: tuple[EntityAliases, ...],
    *,
    allowed_entity_ids: Sequence[UUID] | None = None,
) -> MaskPlan | None:
    allowed = None if allowed_entity_ids is None else set(allowed_entity_ids)
    selected = tuple(
        entity
        for entity in query_entities
        if (allowed is None or entity.entity_id in allowed)
        and mask_text(query, (entity,)) is not None
        and mask_text(key, (entity,)) is not None
    )
    if not selected:
        return None
    signature = entity_signature(tuple(entity.entity_id for entity in selected))
    masked_query = mask_text(query, selected)
    masked_key = mask_text(key, selected)
    if masked_query is None or masked_key is None:
        return None
    return MaskPlan(
        signature=signature,
        signature_hash=signature_hash(signature),
        masked_query=masked_query,
        masked_key=masked_key,
    )


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine vectors must have equal positive dimensions")
    dot = math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(math.fsum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("cosine vectors must be non-zero")
    return dot / (left_norm * right_norm)


def _normalized_text_with_source_spans(
    text: str,
) -> tuple[str, tuple[tuple[int, int], ...]]:
    characters: list[str] = []
    spans: list[tuple[int, int]] = []
    for source_index, character in enumerate(text):
        normalized = unicodedata.normalize("NFKC", character).casefold()
        for item in normalized:
            if item.isspace():
                if characters and characters[-1] == " ":
                    start, _end = spans[-1]
                    spans[-1] = (start, source_index + 1)
                    continue
                item = " "
            characters.append(item)
            spans.append((source_index, source_index + 1))
    return "".join(characters), tuple(spans)


def _alias_occurrences(text: str, alias: str) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    start = 0
    while True:
        index = text.find(alias, start)
        if index < 0:
            return tuple(result)
        end = index + len(alias)
        if _has_valid_boundaries(text, alias, index, end):
            result.append((index, end))
        start = index + 1


def _has_valid_boundaries(text: str, alias: str, start: int, end: int) -> bool:
    if alias[0].isascii() and alias[0].isalnum():
        if start > 0 and text[start - 1].isascii() and text[start - 1].isalnum():
            return False
    if alias[-1].isascii() and alias[-1].isalnum():
        if end < len(text) and text[end].isascii() and text[end].isalnum():
            return False
    return True


def _merge_spans(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not spans:
        return ()
    ordered = sorted(set(spans))
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)
