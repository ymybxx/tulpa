"""Immutable Home preparation data, independent of runs and device writes."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from fish_memory.home.adapters import adapter_for_domain
from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.models import (
    ExcludedHomeTarget,
    HomeAction,
    HomeCatalogSnapshot,
    HomeControlPlan,
    HomeEntity,
    HomeIntentType,
    HomeQueryPlan,
    HomeRunOutcome,
)


@dataclass(frozen=True, slots=True)
class PreparedHomeRequest:
    """A private plan; only ``start_prepared`` may persist or execute it."""

    intent: HomeIntentType
    plan: HomeControlPlan | HomeQueryPlan | None = None
    actions: tuple[HomeAction, ...] = ()
    excluded: tuple[ExcludedHomeTarget, ...] = ()
    satisfied: tuple[HomeAction, ...] = ()
    outcome: HomeRunOutcome | None = None


def freeze_catalog(catalog: HomeEntityCatalog) -> HomeEntityCatalog:
    """Detach registry/state data before any planning await can change it."""

    snapshot = catalog.snapshot
    return HomeEntityCatalog(
        HomeCatalogSnapshot(
            ready=snapshot.ready,
            entities=MappingProxyType(
                {
                    entity_id: replace(
                        entity,
                        attributes=MappingProxyType(
                            {
                                key: _freeze_value(value)
                                for key, value in entity.attributes.items()
                            }
                        ),
                        service_schemas=MappingProxyType(dict(entity.service_schemas)),
                    )
                    for entity_id, entity in snapshot.entities.items()
                }
            ),
        )
    )


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def thaw_home_value(value: object) -> object:
    """Build ordinary JSON containers only at the public persistence boundary."""

    if isinstance(value, Mapping):
        return {key: thaw_home_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_home_value(item) for item in value]
    return value


class CatalogControlEvaluator:
    """Read-only adapter checks against the same snapshot as scope planning."""

    def __init__(self, catalog: HomeEntityCatalog) -> None:
        self._catalog = catalog

    def _entity(self, action: HomeAction) -> HomeEntity:
        entity = self._catalog.get(action.entity_id)
        if entity is None or action.operation not in entity.operations:
            raise ValueError("prepared Home action is outside the catalog capabilities")
        return entity

    def action_is_applicable(self, action: HomeAction) -> bool:
        entity = self._entity(action)
        return adapter_for_domain(entity.domain).action_is_applicable(action, entity)

    def target_is_satisfied(self, action: HomeAction) -> bool:
        entity = self._entity(action)
        return adapter_for_domain(entity.domain).target_is_satisfied(action, entity)
