import importlib
import json

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.home import models
from fish_memory.home.models import (
    HomeAreaFacet,
    HomeIntentType,
    HomeOperation,
    HomeQueryDetail,
    HomeScopeKind,
    HomeSelection,
    HomeStateFilter,
)
from fish_memory.llm.client import JsonCompletion


def test_scope_planning_types_are_public_home_models() -> None:
    assert hasattr(models, "HomeAreaFacet")
    assert hasattr(models, "HomeScopePlan")
    assert hasattr(models, "HomeControlPlan")
    assert hasattr(models, "HomeQueryPlan")


class FakeJsonLlm:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[ChatMessage]] = []

    async def complete_json(self, messages: list[ChatMessage]) -> JsonCompletion:
        self.calls.append(messages)
        return JsonCompletion(content=self.content, input_tokens=10, output_tokens=5)

    def json_request_payload(self, messages: list[ChatMessage]) -> object:
        return {"messages": messages}


def planner_class():
    module = importlib.import_module("fish_memory.home.scope_planner")
    return module.HomeScopePlanner


@pytest.mark.asyncio
async def test_query_scope_plan_uses_request_local_area_index() -> None:
    llm = FakeJsonLlm(
        json.dumps(
            {
                "scope": "area",
                "areaIndexes": [0],
                "domains": ["light"],
                "selection": "all",
                "count": None,
                "stateFilter": "active",
                "detail": "summary",
            }
        )
    )

    plan = await planner_class()(llm).plan(
        "客厅有哪些灯开着？",
        HomeIntentType.QUERY,
        areas=(HomeAreaFacet("living", "客厅"),),
    )

    assert plan.scope.kind is HomeScopeKind.AREA
    assert plan.scope.area_indexes == (0,)
    assert plan.scope.domains == ("light",)
    assert plan.scope.selection is HomeSelection.ALL
    assert plan.state_filter is HomeStateFilter.ACTIVE
    assert plan.detail is HomeQueryDetail.SUMMARY
    assert (
        '{"scope":"area","areaIndexes":[0],"domains":["light"],'
        '"selection":"all","count":null,"stateFilter":"active",'
        '"detail":"summary"}' in llm.calls[0][0]["content"]
    )
    assert (
        "json" in "\n".join(message["content"] for message in llm.calls[0]).casefold()
    )
    prompt = llm.calls[0][0]["content"]
    assert "泛指设备且没有设备类型时 domains 必须为空数组" in prompt
    assert "离线、不可用、故障或异常必须使用 abnormal" in prompt


@pytest.mark.asyncio
async def test_control_scope_plan_parses_operation_without_entity_ids() -> None:
    llm = FakeJsonLlm(
        json.dumps(
            {
                "scope": "specific",
                "areaIndexes": [],
                "domains": ["light"],
                "selection": "specific",
                "count": None,
                "operation": "turn_off",
                "value": None,
            }
        )
    )

    plan = await planner_class()(llm).plan(
        "关掉电视后面的灯",
        HomeIntentType.CONTROL,
        areas=(),
    )

    assert plan.scope.kind is HomeScopeKind.SPECIFIC
    assert plan.scope.selection is HomeSelection.SPECIFIC
    assert plan.operation is HomeOperation.TURN_OFF
    assert plan.value is None
    assert "只有明确的全部、所有或 every/all 才能使用 all" in llm.calls[0][0]["content"]
    assert (
        '{"scope":"area","areaIndexes":[0],"domains":["light"],'
        '"selection":"specific","count":null,"operation":"turn_off",'
        '"value":null}' in llm.calls[0][0]["content"]
    )


@pytest.mark.asyncio
async def test_control_scope_plan_rejects_model_supplied_entity_id() -> None:
    llm = FakeJsonLlm(
        json.dumps(
            {
                "scope": "specific",
                "areaIndexes": [],
                "domains": ["light"],
                "selection": "specific",
                "count": None,
                "operation": "turn_off",
                "value": None,
                "entityId": "light.fake",
            }
        )
    )

    with pytest.raises(ValueError, match="unknown fields"):
        await planner_class()(llm).plan(
            "关掉那盏灯",
            HomeIntentType.CONTROL,
            areas=(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            {
                "scope": "area",
                "areaIndexes": [1],
                "domains": ["light"],
                "selection": "all",
                "count": None,
                "stateFilter": "active",
                "detail": "summary",
            },
            "area index",
        ),
        (
            {
                "scope": "home",
                "areaIndexes": [],
                "domains": ["fake"],
                "selection": "all",
                "count": None,
                "stateFilter": "active",
                "detail": "summary",
            },
            "domain",
        ),
        (
            {
                "scope": "home",
                "areaIndexes": [],
                "domains": ["light"],
                "selection": "count",
                "count": 0,
                "operation": "turn_on",
                "value": None,
            },
            "positive count",
        ),
    ],
)
async def test_scope_plan_rejects_invalid_structural_values(
    payload: dict[str, object],
    error: str,
) -> None:
    intent = (
        HomeIntentType.QUERY if "stateFilter" in payload else HomeIntentType.CONTROL
    )
    llm = FakeJsonLlm(json.dumps(payload))

    with pytest.raises(ValueError, match=error):
        await planner_class()(llm).plan(
            "request",
            intent,
            areas=(HomeAreaFacet("living", "客厅"),),
        )
