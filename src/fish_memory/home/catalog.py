from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from fish_memory.home.adapters import adapter_for_domain, supported_domains
from fish_memory.home.models import (
    HomeAreaFacet,
    HomeCandidate,
    HomeCatalogSnapshot,
    HomeEntity,
)

_NOISE_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)

_GENERIC_DOMAIN_MARKERS = (
    ("灯光", "light"),
    ("灯", "light"),
    ("开关", "switch"),
    ("空调", "climate"),
    ("电视", "media_player"),
    ("播放器", "media_player"),
    ("音箱", "media_player"),
    ("窗帘", "cover"),
    ("风扇", "fan"),
    ("门锁", "lock"),
    ("扫地机", "vacuum"),
    ("吸尘器", "vacuum"),
    ("light", "light"),
    ("switch", "switch"),
    ("climate", "climate"),
    ("tv", "media_player"),
    ("speaker", "media_player"),
    ("fan", "fan"),
)


def normalize_home_alias(value: str) -> str:
    return _NOISE_RE.sub("", value.casefold())


def requested_home_domains(query: str) -> frozenset[str]:
    normalized_query = normalize_home_alias(query)
    return frozenset(
        domain
        for marker, domain in _GENERIC_DOMAIN_MARKERS
        if marker in normalized_query
    )


def group_candidates_by_device(
    candidates: Sequence[HomeCandidate],
) -> tuple[tuple[HomeCandidate, ...], ...]:
    """Rank physical devices by their best endpoint without dropping endpoints."""
    grouped: dict[str, list[HomeCandidate]] = defaultdict(list)
    for candidate in candidates:
        # HA integrations may expose dozens of entities for one device. Entities
        # without registry metadata cannot be grouped reliably, so keep them apart.
        device_key = candidate.entity.device_id or (
            f"entity:{candidate.entity.entity_id}"
        )
        grouped[device_key].append(candidate)

    ranked_groups: list[tuple[int, str, tuple[HomeCandidate, ...]]] = []
    for device_key, members in grouped.items():
        ranked_members = tuple(
            sorted(
                members,
                key=lambda item: (-item.score, item.entity.entity_id),
            )
        )
        ranked_groups.append((ranked_members[0].score, device_key, ranked_members))
    ranked_groups.sort(key=lambda item: (-item[0], item[1]))
    return tuple(group for _score, _key, group in ranked_groups)


def limit_candidates_by_device(
    candidates: Sequence[HomeCandidate],
    *,
    limit: int,
) -> tuple[HomeCandidate, ...]:
    """Limit physical device groups while retaining every matched group endpoint."""
    if limit < 1:
        return ()
    return tuple(
        candidate
        for group in group_candidates_by_device(candidates)[:limit]
        for candidate in group
    )


class HomeEntityCatalog:
    def __init__(self, snapshot: HomeCatalogSnapshot) -> None:
        self._snapshot = snapshot

    @property
    def snapshot(self) -> HomeCatalogSnapshot:
        return self._snapshot

    @classmethod
    def empty(cls) -> HomeEntityCatalog:
        return cls(HomeCatalogSnapshot(entities=MappingProxyType({}), ready=False))

    @classmethod
    def from_payloads(
        cls,
        *,
        states: Sequence[Mapping[str, Any]],
        entity_registry: Sequence[Mapping[str, Any]],
        device_registry: Sequence[Mapping[str, Any]],
        area_registry: Sequence[Mapping[str, Any]],
        services: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> HomeEntityCatalog:
        registry_by_entity = {
            str(item.get("entity_id")): item
            for item in entity_registry
            if item.get("entity_id")
        }
        devices = {
            str(item.get("id")): item for item in device_registry if item.get("id")
        }
        areas = {
            str(item.get("area_id")): item
            for item in area_registry
            if item.get("area_id")
        }
        domains = supported_domains()
        entities: dict[str, HomeEntity] = {}
        for state in states:
            entity_id = str(state.get("entity_id", ""))
            if "." not in entity_id:
                continue
            domain = entity_id.split(".", 1)[0]
            if domain not in domains:
                continue
            registry = registry_by_entity.get(entity_id, {})
            device_id = _optional_text(registry.get("device_id"))
            device = devices.get(device_id or "", {})
            area_id = _optional_text(registry.get("area_id")) or _optional_text(
                device.get("area_id")
            )
            area = areas.get(area_id or "", {})
            attributes = dict(state.get("attributes") or {})
            raw_aliases = (
                attributes.get("friendly_name"),
                registry.get("name"),
                registry.get("original_name"),
                device.get("name_by_user"),
                device.get("name"),
                area.get("name"),
                entity_id.split(".", 1)[1].replace("_", " "),
            )
            aliases = tuple(
                dict.fromkeys(
                    text for value in raw_aliases if (text := _optional_text(value))
                )
            )
            domain_services = services.get(domain, {})
            schemas = MappingProxyType(
                {
                    name: frozenset((definition.get("fields") or {}).keys())
                    for name, definition in domain_services.items()
                }
            )
            operations = adapter_for_domain(domain).operations(attributes, schemas)
            entities[entity_id] = HomeEntity(
                entity_id=entity_id,
                domain=domain,
                state=str(state.get("state", "unknown")),
                attributes=MappingProxyType(attributes),
                aliases=aliases,
                area_id=area_id,
                area_name=_optional_text(area.get("name")),
                device_id=device_id,
                device_name=_optional_text(
                    device.get("name_by_user") or device.get("name")
                ),
                operations=operations,
                service_schemas=schemas,
            )
        return cls(HomeCatalogSnapshot(entities=MappingProxyType(entities), ready=True))

    def get(self, entity_id: str) -> HomeEntity | None:
        return self._snapshot.entities.get(entity_id)

    def available_entities(self) -> tuple[HomeEntity, ...]:
        return tuple(
            entity
            for entity in self._snapshot.entities.values()
            if _entity_is_available(entity)
        )

    def area_facets(self) -> tuple[HomeAreaFacet, ...]:
        by_id: dict[str, HomeAreaFacet] = {}
        for entity in self._snapshot.entities.values():
            if entity.area_id and entity.area_name:
                by_id[entity.area_id] = HomeAreaFacet(
                    area_id=entity.area_id,
                    name=entity.area_name,
                )
        return tuple(
            sorted(
                by_id.values(),
                key=lambda area: (area.name.casefold(), area.area_id),
            )
        )

    def exact(self, alias: str) -> tuple[HomeEntity, ...]:
        normalized = normalize_home_alias(alias)
        return tuple(
            entity
            for entity in self._snapshot.entities.values()
            if any(normalize_home_alias(item) == normalized for item in entity.aliases)
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        available_only: bool = False,
    ) -> tuple[HomeCandidate, ...]:
        normalized_query = normalize_home_alias(query)
        requested_domains = requested_home_domains(query)
        ranked: list[HomeCandidate] = []
        for entity in self._snapshot.entities.values():
            if available_only and not _entity_is_available(entity):
                continue
            matched: list[str] = []
            score = 0
            normalized_area = normalize_home_alias(entity.area_name or "")
            for alias in entity.aliases:
                normalized_alias = normalize_home_alias(alias)
                # An area describes where an entity is, not what it is. Scoring it
                # as a device alias lets every helper in a room outrank the device.
                if not normalized_alias or normalized_alias == normalized_area:
                    continue
                if normalized_alias == normalized_query:
                    alias_score = 1000
                elif normalized_alias in normalized_query:
                    alias_score = 500 + len(normalized_alias)
                elif normalized_query in normalized_alias:
                    alias_score = 200 + len(normalized_query)
                else:
                    continue
                matched.append(alias)
                score = max(score, alias_score)
            if entity.domain in requested_domains:
                matched.append(entity.aliases[0])
                score = max(score, 100)
            elif any(
                marker in normalize_home_alias(alias)
                for marker, _domain in _GENERIC_DOMAIN_MARKERS
                if marker in normalized_query
                for alias in entity.aliases
            ):
                matched.append(entity.aliases[0])
                score = max(score, 100)
            if matched:
                ranked.append(
                    HomeCandidate(
                        entity=entity,
                        score=score,
                        matched_aliases=tuple(dict.fromkeys(matched)),
                    )
                )
        ranked.sort(key=lambda item: (-item.score, item.entity.entity_id))
        return tuple(ranked[:limit])


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _entity_is_available(entity: HomeEntity) -> bool:
    if entity.state == "unavailable":
        return False
    # Stateless buttons report unknown even when they are available to press.
    return entity.state != "unknown" or entity.domain == "button"
