import importlib

from fish_memory.home import models
from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.executor import HomeActionExecutor
from fish_memory.home.models import (
    HomeControlPlan,
    HomeOperation,
    HomeQueryCategory,
    HomeQueryDetail,
    HomeQueryPlan,
    HomeScopeKind,
    HomeScopePlan,
    HomeSelection,
    HomeStateFilter,
)


def test_scope_resolution_types_are_public_home_models() -> None:
    assert hasattr(models, "HomeQueryCategory")
    assert hasattr(models, "HomeQueryState")
    assert hasattr(models, "ExcludedHomeTarget")
    assert hasattr(models, "ResolvedQueryScope")
    assert hasattr(models, "ResolvedControlScope")


class FakeClient:
    def __init__(self, catalog: HomeEntityCatalog) -> None:
        self.catalog = catalog


def resolver_class():
    module = importlib.import_module("fish_memory.home.scope_resolver")
    return module.HomeScopeResolver


def area_catalog() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.direct",
                "state": "on",
                "attributes": {"friendly_name": "主灯"},
            },
            {
                "entity_id": "light.misnamed",
                "state": "on",
                "attributes": {"friendly_name": "客厅灯"},
            },
        ],
        entity_registry=[
            {
                "entity_id": "light.direct",
                "device_id": "device-direct",
                "area_id": "living",
            },
            {
                "entity_id": "light.misnamed",
                "device_id": "device-misnamed",
            },
        ],
        device_registry=[
            {"id": "device-direct", "name": "主灯", "area_id": "bedroom"},
            {
                "id": "device-misnamed",
                "name": "客厅灯",
                "area_id": "bedroom",
            },
        ],
        area_registry=[
            {"area_id": "living", "name": "客厅"},
            {"area_id": "bedroom", "name": "卧室"},
        ],
        services={"light": {"turn_on": {"fields": {}}, "turn_off": {"fields": {}}}},
    )


def mixed_catalog() -> HomeEntityCatalog:
    states = [
        ("light.online", "off", "在线灯"),
        ("light.already_on", "on", "已开灯"),
        ("light.offline", "unavailable", "离线灯"),
        ("light.unknown", "unknown", "未知灯"),
        ("climate.offline", "unavailable", "离线空调"),
        ("switch.plug", "on", "智能插座"),
        ("button.scene", "unknown", "回家按钮"),
        ("lock.front", "unlocked", "前门锁"),
    ]
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": entity_id,
                "state": state,
                "attributes": {"friendly_name": name},
            }
            for entity_id, state, name in states
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "light": {"turn_on": {"fields": {}}, "turn_off": {"fields": {}}},
            "climate": {"turn_on": {"fields": {}}, "turn_off": {"fields": {}}},
            "switch": {"turn_on": {"fields": {}}, "turn_off": {"fields": {}}},
            "button": {"press": {"fields": {}}},
            "lock": {"lock": {"fields": {}}, "unlock": {"fields": {}}},
        },
    )


def scope(
    *,
    kind: HomeScopeKind = HomeScopeKind.HOME,
    area_indexes: tuple[int, ...] = (),
    domains: tuple[str, ...] = (),
    selection: HomeSelection = HomeSelection.ALL,
) -> HomeScopePlan:
    return HomeScopePlan(kind, area_indexes, domains, selection, None)


def test_area_scope_uses_registry_assignment_not_friendly_name() -> None:
    catalog = area_catalog()
    areas = catalog.area_facets()
    living_index = next(
        index for index, area in enumerate(areas) if area.area_id == "living"
    )
    plan = HomeQueryPlan(
        scope=scope(kind=HomeScopeKind.AREA, area_indexes=(living_index,)),
        state_filter=HomeStateFilter.ALL,
        detail=HomeQueryDetail.FULL,
    )

    resolved = resolver_class()(catalog).resolve_query(plan)

    assert [item.entity.entity_id for item in resolved.entities] == ["light.direct"]


def test_specific_scope_keeps_optional_area_constraint() -> None:
    catalog = area_catalog()
    areas = catalog.area_facets()
    living_index = next(
        index for index, area in enumerate(areas) if area.area_id == "living"
    )
    plan = HomeQueryPlan(
        scope=scope(
            kind=HomeScopeKind.SPECIFIC,
            area_indexes=(living_index,),
            selection=HomeSelection.SPECIFIC,
        ),
        state_filter=HomeStateFilter.ALL,
        detail=HomeQueryDetail.FULL,
    )

    resolved = resolver_class()(catalog).resolve_query(plan)

    assert [item.entity.entity_id for item in resolved.entities] == ["light.direct"]


def test_query_view_keeps_abnormal_state_and_excludes_switch_and_button() -> None:
    resolved = resolver_class()(mixed_catalog()).resolve_query(
        HomeQueryPlan(
            scope=scope(),
            state_filter=HomeStateFilter.ALL,
            detail=HomeQueryDetail.FULL,
        )
    )

    categories = {item.entity.entity_id: item.category for item in resolved.entities}
    assert categories["light.already_on"] is HomeQueryCategory.ACTIVE
    assert categories["climate.offline"] is HomeQueryCategory.ABNORMAL
    assert "switch.plug" not in categories
    assert "button.scene" not in categories


def test_interesting_query_omits_ordinary_inactive_devices() -> None:
    resolved = resolver_class()(mixed_catalog()).resolve_query(
        HomeQueryPlan(
            scope=scope(),
            state_filter=HomeStateFilter.INTERESTING,
            detail=HomeQueryDetail.SUMMARY,
        )
    )

    assert {item.entity.entity_id for item in resolved.entities} == {
        "light.already_on",
        "light.offline",
        "light.unknown",
        "climate.offline",
        "lock.front",
    }


def test_control_view_excludes_unhealthy_and_records_satisfied_targets() -> None:
    catalog = mixed_catalog()
    resolver = resolver_class()(catalog)
    executor = HomeActionExecutor(FakeClient(catalog))

    resolved = resolver.resolve_control(
        HomeControlPlan(
            scope=scope(domains=("light",)),
            operation=HomeOperation.TURN_ON,
        ),
        executor,
    )

    assert [item.entity.entity_id for item in resolved.eligible] == ["light.online"]
    assert {item.entity.entity_id: item.reason for item in resolved.excluded} == {
        "light.offline": "unavailable",
        "light.unknown": "unknown",
    }
    assert [item.entity_id for item in resolved.satisfied] == ["light.already_on"]


def test_stateless_unknown_button_remains_eligible_for_activate() -> None:
    catalog = mixed_catalog()
    resolver = resolver_class()(catalog)

    resolved = resolver.resolve_control(
        HomeControlPlan(
            scope=scope(domains=("button",)),
            operation=HomeOperation.ACTIVATE,
        ),
        HomeActionExecutor(FakeClient(catalog)),
    )

    assert [item.entity.entity_id for item in resolved.eligible] == ["button.scene"]
    assert resolved.excluded == ()
