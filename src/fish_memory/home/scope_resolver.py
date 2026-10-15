from __future__ import annotations

from typing import Protocol

from fish_memory.home.adapters import adapter_for_domain
from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.models import (
    ExcludedHomeTarget,
    HomeAction,
    HomeCandidate,
    HomeControlPlan,
    HomeEntity,
    HomeOperation,
    HomeQueryCategory,
    HomeQueryPlan,
    HomeQueryState,
    HomeScopePlan,
    HomeStateFilter,
    ResolvedControlScope,
    ResolvedQueryScope,
)

_QUERY_EXCLUDED_DOMAINS = frozenset({"button", "switch", "scene", "script"})


class ControlEvaluator(Protocol):
    def action_is_applicable(self, action: HomeAction) -> bool: ...
    def target_is_satisfied(self, action: HomeAction) -> bool: ...


class HomeScopeResolver:
    def __init__(self, catalog: HomeEntityCatalog) -> None:
        self._catalog = catalog

    def resolve_query(self, plan: HomeQueryPlan) -> ResolvedQueryScope:
        entities: list[HomeQueryState] = []
        for entity in self._scoped_entities(plan.scope):
            if (
                entity.domain in _QUERY_EXCLUDED_DOMAINS
                or HomeOperation.QUERY not in entity.operations
            ):
                continue
            category = adapter_for_domain(entity.domain).query_category(entity)
            if _matches_state_filter(category, plan.state_filter):
                entities.append(HomeQueryState(entity=entity, category=category))
        return ResolvedQueryScope(entities=tuple(entities))

    def resolve_control(
        self,
        plan: HomeControlPlan,
        evaluator: ControlEvaluator,
    ) -> ResolvedControlScope:
        eligible: list[HomeCandidate] = []
        excluded: list[ExcludedHomeTarget] = []
        satisfied: list[HomeAction] = []
        for entity in self._scoped_entities(plan.scope):
            if plan.operation not in entity.operations:
                excluded.append(ExcludedHomeTarget(entity, "operation_unsupported"))
                continue
            adapter = adapter_for_domain(entity.domain)
            if reason := adapter.control_health(entity, plan.operation):
                excluded.append(ExcludedHomeTarget(entity, reason))
                continue
            action = HomeAction(entity.entity_id, plan.operation, plan.value)
            if not evaluator.action_is_applicable(action):
                excluded.append(ExcludedHomeTarget(entity, "inapplicable"))
                continue
            if plan.value not in {
                "increase",
                "decrease",
            } and evaluator.target_is_satisfied(action):
                satisfied.append(action)
                continue
            eligible.append(HomeCandidate(entity=entity, score=0, matched_aliases=()))
        return ResolvedControlScope(
            eligible=tuple(eligible),
            excluded=tuple(excluded),
            satisfied=tuple(satisfied),
        )

    def _scoped_entities(self, plan: HomeScopePlan) -> tuple[HomeEntity, ...]:
        area_ids = {
            self._catalog.area_facets()[index].area_id for index in plan.area_indexes
        }
        return tuple(
            entity
            for entity in sorted(
                self._catalog.snapshot.entities.values(),
                key=lambda item: item.entity_id,
            )
            if (not plan.domains or entity.domain in plan.domains)
            and (not area_ids or entity.area_id in area_ids)
        )


def _matches_state_filter(
    category: HomeQueryCategory,
    state_filter: HomeStateFilter,
) -> bool:
    if state_filter is HomeStateFilter.ALL:
        return True
    if state_filter is HomeStateFilter.INTERESTING:
        return category in {
            HomeQueryCategory.ACTIVE,
            HomeQueryCategory.INTERESTING,
            HomeQueryCategory.ABNORMAL,
        }
    return category.value == state_filter.value
