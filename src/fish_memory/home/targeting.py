from __future__ import annotations

import re
from collections.abc import Sequence

from fish_memory.home.catalog import (
    normalize_home_alias,
    requested_home_domains,
)
from fish_memory.home.models import HomeCandidate, HomeSetTarget

_ALL_MARKERS = ("所有", "全部", "全都", "都")
_COUNT_RE = re.compile(
    r"(?P<count>\d+|[一二两三四五六七八九十]+)\s*"
    r"(?:个|台|盏|扇|组|套|只|路|条)"
)
_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def has_explicit_set_request(text: str) -> bool:
    return _has_all_marker(text) or bool(explicit_counts(text))


def validate_set_target(
    text: str,
    target: HomeSetTarget,
    candidates: Sequence[HomeCandidate],
) -> None:
    if target.selection != "all":
        raise ValueError("set selection must be all")
    counts = explicit_counts(text)
    if target.count is None:
        if not _has_all_marker(text):
            raise ValueError("set target requires explicit set or quantity language")
    elif target.count < 1 or target.count not in counts:
        raise ValueError("set target requires explicit set or quantity language")
    _validate_explicit_filters(text, target, candidates)


def explicit_counts(text: str) -> frozenset[int]:
    values: set[int] = set()
    for match in _COUNT_RE.finditer(text):
        if value := _parse_count(match.group("count")):
            values.add(value)
    return frozenset(values)


def _has_all_marker(text: str) -> bool:
    return any(marker in text for marker in _ALL_MARKERS)


def _validate_explicit_filters(
    text: str,
    target: HomeSetTarget,
    candidates: Sequence[HomeCandidate],
) -> None:
    requested_domains = requested_home_domains(text)
    if requested_domains and target.domain not in requested_domains:
        raise ValueError("set target drops an explicit area or domain")
    normalized_text = normalize_home_alias(text)
    requested_areas = {
        normalized_area
        for candidate in candidates
        if (normalized_area := normalize_home_alias(candidate.entity.area_name or ""))
        and normalized_area in normalized_text
    }
    if (
        requested_areas
        and normalize_home_alias(target.area or "") not in requested_areas
    ):
        raise ValueError("set target drops an explicit area or domain")


def _parse_count(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        tens, ones = value.split("十", 1)
        tens_value = _CHINESE_DIGITS.get(tens, 1) if tens else 1
        ones_value = _CHINESE_DIGITS.get(ones, 0) if ones else 0
        return tens_value * 10 + ones_value
    return _CHINESE_DIGITS.get(value)
