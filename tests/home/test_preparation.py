import asyncio
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.models import (
    HomeAction,
    HomeActionStatus,
    HomeControlPlan,
    HomeIntentType,
    HomeOperation,
    HomeQueryDetail,
    HomeQueryPlan,
    HomeSelection,
    HomeStateFilter,
)
from fish_memory.performance.tracker import MessagePerformanceTracker
from tests.home.test_scoped_service import (
    FISH_ID,
    MESSAGE_ID,
    SESSION_ID,
    FakePlanner,
    FakeSelector,
    make_service,
    scope,
)


class ForbiddenHistory:
    async def load_history(self, *_args, **_kwargs):
        raise AssertionError("prepared requests must not reload history")


class PerformanceStore:
    def __init__(self):
        self.calls = []

    async def insert_steps(self, *args):
        self.calls.append(args)


def query_plan():
    return HomeQueryPlan(
        scope=scope(HomeSelection.ALL),
        state_filter=HomeStateFilter.ALL,
        detail=HomeQueryDetail.FULL,
    )


def catalog(
    *, entity_id="light.one", state="on", brightness=128, nested=None, controllable=True
):
    attributes = {"friendly_name": "灯一", "brightness": brightness}
    if nested is not None:
        attributes["nested"] = nested
    result = HomeEntityCatalog.from_payloads(
        states=[{"entity_id": entity_id, "state": state, "attributes": attributes}],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "light": {
                "turn_on": {"fields": {"brightness_pct": {}}},
                "turn_off": {"fields": {}},
            }
        }
        if controllable
        else {},
    )
    if not controllable:
        return HomeEntityCatalog(
            replace(
                result.snapshot,
                entities={
                    key: replace(entity, operations=frozenset({HomeOperation.QUERY}))
                    for key, entity in result.snapshot.entities.items()
                },
            )
        )
    return result


@pytest.mark.asyncio
async def test_prepare_freezes_query_before_planning_and_commit_uses_that_result():
    home, client, store, selector = make_service(query_plan())
    nested = {"values": [1, 2]}
    client.catalog = catalog(nested=nested)
    home._history_store = ForbiddenHistory()
    performance = PerformanceStore()
    home._performance_store = performance
    entered, release = asyncio.Event(), asyncio.Event()

    class BlockingPlanner(FakePlanner):
        async def plan(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return await super().plan(*args, **kwargs)

    planner = BlockingPlanner(query_plan())
    home._scope_planner = planner
    history = [{"role": "user", "content": "原来的上下文"}]
    preparing = asyncio.create_task(
        home.prepare(
            "灯是什么状态",
            intent=HomeIntentType.QUERY,
            history=history,
            tracker=MessagePerformanceTracker(),
        )
    )
    await entered.wait()
    nested["values"].append(3)
    history[0]["content"] = "已改变的上下文"
    client.catalog = HomeEntityCatalog.empty()
    release.set()
    prepared = await preparing

    assert store.created == store.updated == client.calls == performance.calls == []
    assert planner.calls[0][-1]["history"][0]["content"] == "原来的上下文"
    assert prepared.outcome is not None and prepared.outcome.query is not None
    entity = prepared.outcome.query.entities[0].entity
    assert entity.state == "on"
    assert entity.attributes["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        entity.attributes["nested"]["changed"] = True
    with pytest.raises(FrozenInstanceError):
        prepared.actions = ()

    home._scope_planner = home._selector = None
    outcome = await home.start_prepared(
        FISH_ID, SESSION_ID, MESSAGE_ID, prepared
    ).result()
    assert outcome is prepared.outcome
    assert len(planner.calls) == 1 and selector.calls == [] and client.calls == []
    assert len(store.created) == 1 and len(store.updated) == 1
    result = store.updated[0]["result"]
    assert result["items"][0]["state"] == "on"
    assert result["items"][0]["attributes"]["nested"]["values"] == [1, 2]
    json.dumps(result)
    assert len(performance.calls) == 1


@pytest.mark.asyncio
async def test_control_prepares_once_and_only_executes_selected_frozen_action():
    plan = HomeControlPlan(scope(HomeSelection.SPECIFIC), HomeOperation.TURN_OFF)
    home, client, store, selector = make_service(plan)
    home._history_store = ForbiddenHistory()
    planner = home._scope_planner
    prepared = await home.prepare(
        "关闭灯一",
        intent=HomeIntentType.CONTROL,
        history=(),
        tracker=MessagePerformanceTracker(),
    )
    assert prepared.outcome is None
    assert prepared.actions == (HomeAction("light.one", HomeOperation.TURN_OFF),)
    await home.drain()
    assert store.created == store.updated == client.calls == []

    # The committed path must work without either LLM routing component.
    home._scope_planner = home._selector = None
    result = await home.start_prepared(
        FISH_ID, SESSION_ID, MESSAGE_ID, prepared
    ).result()
    assert result.status is HomeActionStatus.SUCCEEDED
    assert len(planner.calls) == len(selector.calls) == 1
    assert [call[3]["entity_id"] for call in client.calls] == ["light.one"]
    assert [update["status"] for update in store.updated] == [
        HomeActionStatus.EXECUTING,
        HomeActionStatus.SUCCEEDED,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ["planner", "selector"])
async def test_cancelled_preparation_never_creates_run_or_executes(blocked_stage):
    plan = HomeControlPlan(scope(HomeSelection.SPECIFIC), HomeOperation.TURN_OFF)
    home, client, store, _selector = make_service(plan)
    entered = asyncio.Event()

    class Blocked:
        async def plan(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

        select = plan

    if blocked_stage == "planner":
        home._scope_planner = Blocked()
    else:
        home._selector = Blocked()
    preparing = asyncio.create_task(
        home.prepare(
            "关闭灯一",
            intent=HomeIntentType.CONTROL,
            history=(),
            tracker=MessagePerformanceTracker(),
        )
    )
    await entered.wait()
    preparing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await preparing
    await home.drain()
    assert store.created == store.updated == client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unready", "ambiguous", "count", "planner"])
async def test_prepared_failure_remains_private_until_committed(failure):
    plan = HomeControlPlan(
        scope(HomeSelection.COUNT, count=3)
        if failure == "count"
        else scope(HomeSelection.SPECIFIC),
        HomeOperation.TURN_OFF,
    )
    home, client, store, _selector = make_service(
        plan,
        selector=FakeSelector(selected_index=None) if failure == "ambiguous" else None,
    )
    if failure == "unready":
        client.catalog = HomeEntityCatalog.empty()
    elif failure == "planner":

        class FailingPlanner:
            async def plan(self, *_args, **_kwargs):
                raise ValueError("invalid scope")

        home._scope_planner = FailingPlanner()
    prepared = await home.prepare(
        "关闭灯",
        intent=HomeIntentType.CONTROL,
        history=(),
        tracker=MessagePerformanceTracker(),
    )
    assert prepared.outcome is not None
    assert prepared.outcome.status is (
        HomeActionStatus.AMBIGUOUS
        if failure == "ambiguous"
        else HomeActionStatus.FAILED
    )
    assert store.created == store.updated == client.calls == []
    result = await home.start_prepared(
        FISH_ID, SESSION_ID, MESSAGE_ID, prepared
    ).result()
    assert result is prepared.outcome and client.calls == []
    assert len(store.created) == 1 and len(store.updated) == 1


@pytest.mark.asyncio
async def test_already_satisfied_control_is_prepared_without_execution():
    plan = HomeControlPlan(scope(HomeSelection.ALL), HomeOperation.TURN_OFF)
    home, client, store, _selector = make_service(plan)
    client.catalog = catalog(state="off")
    prepared = await home.prepare(
        "关闭所有灯",
        intent=HomeIntentType.CONTROL,
        history=(),
        tracker=MessagePerformanceTracker(),
    )
    assert prepared.actions == ()
    assert prepared.outcome is not None
    assert prepared.outcome.status is HomeActionStatus.SUCCEEDED
    assert prepared.satisfied == (HomeAction("light.one", HomeOperation.TURN_OFF),)
    assert store.created == store.updated == client.calls == []
    assert (
        await home.start_prepared(FISH_ID, SESSION_ID, MESSAGE_ID, prepared).result()
        is prepared.outcome
    )
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["removed", "unsupported", "unavailable"])
async def test_control_commit_rechecks_planned_target_without_substitution(change):
    plan = HomeControlPlan(scope(HomeSelection.SPECIFIC), HomeOperation.TURN_OFF)
    home, client, store, selector = make_service(plan)
    prepared = await home.prepare(
        "关闭灯一",
        intent=HomeIntentType.CONTROL,
        history=(),
        tracker=MessagePerformanceTracker(),
    )
    client.catalog = catalog(
        entity_id="light.replacement" if change == "removed" else "light.one",
        state="unavailable" if change == "unavailable" else "on",
        controllable=change != "unsupported",
    )
    outcome = await home.start_prepared(
        FISH_ID, SESSION_ID, MESSAGE_ID, prepared
    ).result()
    assert outcome.status is HomeActionStatus.FAILED
    assert client.calls == [] and len(selector.calls) == 1
    assert store.updated[-1]["status"] is HomeActionStatus.FAILED


@pytest.mark.asyncio
async def test_specific_relative_brightness_is_frozen_before_commit():
    plan = HomeControlPlan(
        scope(HomeSelection.SPECIFIC), HomeOperation.SET_BRIGHTNESS, "increase"
    )
    home, client, _store, _selector = make_service(plan)
    client.catalog = catalog(brightness=128)
    prepared = await home.prepare(
        "灯一调亮一点",
        intent=HomeIntentType.CONTROL,
        history=(),
        tracker=MessagePerformanceTracker(),
    )
    assert prepared.actions == (
        HomeAction("light.one", HomeOperation.SET_BRIGHTNESS, 60),
    )
    client.catalog = catalog(brightness=200)
    outcome = await home.start_prepared(
        FISH_ID, SESSION_ID, MESSAGE_ID, prepared
    ).result()
    assert outcome.status is HomeActionStatus.SUCCEEDED
    assert client.calls[0][2] == {"brightness_pct": 60}
