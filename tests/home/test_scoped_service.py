from uuid import UUID

import pytest

from fish_memory.home import models, service
from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.models import (
    HomeAction,
    HomeActionStatus,
    HomeControlPlan,
    HomeIntentType,
    HomeOperation,
    HomeQueryDetail,
    HomeQueryPlan,
    HomeScopeKind,
    HomeScopePlan,
    HomeSelection,
    HomeStateFilter,
)
from fish_memory.home.service import HomeAutomationService

FISH_ID = UUID(int=1)
SESSION_ID = UUID(int=2)
MESSAGE_ID = UUID(int=3)


def test_scoped_service_exposes_typed_run_outcome_and_handle() -> None:
    assert hasattr(models, "HomeRunOutcome")
    assert hasattr(service, "HomeRunHandle")
    assert hasattr(service.HomeAutomationService, "start")


class FakeStore:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.updated: list[dict[str, object]] = []

    async def create_run(self, **kwargs):
        self.created.append(kwargs)
        return type("Run", (), {"id": UUID(int=4)})()

    async def update_run(self, _run_id, **kwargs):
        self.updated.append(kwargs)


class FakeHistoryStore:
    async def load_history(self, *_args, **_kwargs):
        return []


class LegacyRouter:
    device_limit = 40

    async def route(self, *_args, **_kwargs):
        raise AssertionError("scoped flow must not call the legacy router")


class FakePlanner:
    def __init__(self, plan) -> None:
        self.planned = plan
        self.calls: list[tuple[object, ...]] = []

    async def plan(self, *args, **kwargs):
        self.calls.append((*args, kwargs))
        return self.planned


class FakeSelector:
    def __init__(self, selected_index: int | None = 0) -> None:
        self.selected_index = selected_index
        self.calls = []

    async def select(
        self,
        text,
        candidates,
        *,
        operation,
        value,
        history=(),
        tracker=None,
    ):
        self.calls.append((text, candidates, operation, value, history, tracker))
        if self.selected_index is None:
            return ()
        entity = candidates[self.selected_index].entity
        return (HomeAction(entity.entity_id, operation, value),)


class FakeClient:
    def __init__(self) -> None:
        self.catalog = HomeEntityCatalog.from_payloads(
            states=[
                {
                    "entity_id": "light.one",
                    "state": "on",
                    "attributes": {"friendly_name": "灯一"},
                },
                {
                    "entity_id": "light.two",
                    "state": "on",
                    "attributes": {"friendly_name": "灯二"},
                },
                {
                    "entity_id": "light.offline",
                    "state": "unavailable",
                    "attributes": {"friendly_name": "离线灯"},
                },
                {
                    "entity_id": "switch.plug",
                    "state": "on",
                    "attributes": {"friendly_name": "智能插座"},
                },
            ],
            entity_registry=[],
            device_registry=[],
            area_registry=[],
            services={
                "light": {
                    "turn_on": {"fields": {}},
                    "turn_off": {"fields": {}},
                },
                "switch": {
                    "turn_on": {"fields": {}},
                    "turn_off": {"fields": {}},
                },
            },
        )
        self.calls = []

    async def call_service(self, domain, service, *, service_data=None, target=None):
        self.calls.append((domain, service, service_data, target))
        return {"ok": True}

    async def wait_for_state(self, _entity_id, expected_states, *, timeout_seconds):
        del timeout_seconds
        return next(iter(expected_states))

    async def wait_for_cover_movement(self, *_args, **_kwargs):
        return None


def scope(selection: HomeSelection, *, count: int | None = None) -> HomeScopePlan:
    return HomeScopePlan(
        kind=HomeScopeKind.HOME,
        area_indexes=(),
        domains=("light",),
        selection=selection,
        count=count,
    )


def make_service(plan, *, selector: FakeSelector | None = None):
    client = FakeClient()
    store = FakeStore()
    selected = selector or FakeSelector()
    value = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=LegacyRouter(),
        scope_planner=FakePlanner(plan),
        selector=selected,
        performance_store=None,
    )
    return value, client, store, selected


@pytest.mark.asyncio
async def test_batch_control_expands_locally_without_selector() -> None:
    plan = HomeControlPlan(
        scope=scope(HomeSelection.ALL),
        operation=HomeOperation.TURN_OFF,
    )
    home, client, _store, selector = make_service(plan)

    outcome = await home.start(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "关闭所有灯",
        10,
        intent=HomeIntentType.CONTROL,
    ).result()

    assert outcome.status is HomeActionStatus.PARTIAL
    assert selector.calls == []
    assert [call[3]["entity_id"] for call in client.calls] == [
        "light.one",
        "light.two",
    ]
    assert [item.entity.entity_id for item in outcome.excluded] == ["light.offline"]


@pytest.mark.asyncio
async def test_precise_control_selects_only_eligible_candidates() -> None:
    plan = HomeControlPlan(
        scope=scope(HomeSelection.SPECIFIC),
        operation=HomeOperation.TURN_OFF,
    )
    home, client, _store, selector = make_service(plan)

    outcome = await home.start(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "关闭灯一",
        10,
        intent=HomeIntentType.CONTROL,
    ).result()

    assert outcome.status is HomeActionStatus.SUCCEEDED
    assert len(selector.calls) == 1
    assert [item.entity.entity_id for item in selector.calls[0][1]] == [
        "light.one",
        "light.two",
    ]
    assert [call[3]["entity_id"] for call in client.calls] == ["light.one"]


@pytest.mark.asyncio
async def test_query_reads_catalog_without_service_or_selector_call() -> None:
    plan = HomeQueryPlan(
        scope=scope(HomeSelection.ALL),
        state_filter=HomeStateFilter.INTERESTING,
        detail=HomeQueryDetail.SUMMARY,
    )
    home, client, _store, selector = make_service(plan)

    outcome = await home.start(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "哪些灯开着或离线？",
        10,
        intent=HomeIntentType.QUERY,
    ).result()

    assert outcome.status is HomeActionStatus.SUCCEEDED
    assert client.calls == []
    assert selector.calls == []
    assert outcome.query is not None
    assert [item.entity.entity_id for item in outcome.query.entities] == [
        "light.offline",
        "light.one",
        "light.two",
    ]


@pytest.mark.asyncio
async def test_count_control_fails_before_partial_execution() -> None:
    plan = HomeControlPlan(
        scope=scope(HomeSelection.COUNT, count=3),
        operation=HomeOperation.TURN_OFF,
    )
    home, client, _store, selector = make_service(plan)

    outcome = await home.start(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "关闭三盏灯",
        10,
        intent=HomeIntentType.CONTROL,
    ).result()

    assert outcome.status is HomeActionStatus.FAILED
    assert outcome.error == "可操作设备数量不足"
    assert client.calls == []
    assert selector.calls == []


@pytest.mark.asyncio
async def test_precise_control_with_no_selected_target_is_ambiguous() -> None:
    plan = HomeControlPlan(
        scope=scope(HomeSelection.SPECIFIC),
        operation=HomeOperation.TURN_OFF,
    )
    home, client, _store, selector = make_service(
        plan,
        selector=FakeSelector(selected_index=None),
    )

    outcome = await home.start(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "关掉那一个",
        10,
        intent=HomeIntentType.CONTROL,
    ).result()

    assert outcome.status is HomeActionStatus.AMBIGUOUS
    assert outcome.error == "无法唯一确定设备"
    assert client.calls == []
    assert len(selector.calls) == 1
