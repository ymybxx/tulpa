import importlib
import json
from types import MappingProxyType

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.home.models import (
    HomeAction,
    HomeCandidate,
    HomeEntity,
    HomeOperation,
)
from fish_memory.llm.client import JsonCompletion


class FakeJsonLlm:
    def __init__(self, content: object) -> None:
        self.content = content
        self.calls: list[list[ChatMessage]] = []

    async def complete_json(self, messages: list[ChatMessage]) -> JsonCompletion:
        self.calls.append(messages)
        return JsonCompletion(content=json.dumps(self.content))

    def json_request_payload(self, messages: list[ChatMessage]) -> object:
        return {"messages": messages}


def selector_class():
    module = importlib.import_module("fish_memory.home.selector")
    return module.HomeDeviceSelector


def candidate(
    entity_id: str,
    name: str,
    *,
    area: str = "客厅",
    device_id: str | None = None,
) -> HomeCandidate:
    entity = HomeEntity(
        entity_id=entity_id,
        domain="light",
        state="on",
        attributes=MappingProxyType({}),
        aliases=(name,),
        area_name=area,
        device_id=device_id,
        device_name=name,
        operations=frozenset({HomeOperation.QUERY, HomeOperation.TURN_OFF}),
    )
    return HomeCandidate(entity=entity, score=0, matched_aliases=())


@pytest.mark.asyncio
async def test_selector_returns_actions_from_request_local_indexes() -> None:
    llm = FakeJsonLlm({"candidateIndexes": [1]})
    candidates = (
        candidate("light.main", "主灯"),
        candidate("light.tv_back", "电视后灯带"),
    )

    actions = await selector_class()(llm).select(
        "关闭电视后面的灯",
        candidates,
        operation=HomeOperation.TURN_OFF,
        value=None,
    )

    assert actions == (HomeAction("light.tv_back", HomeOperation.TURN_OFF),)
    payload = json.loads(llm.calls[0][-1]["content"])
    assert "light.main" not in llm.calls[0][-1]["content"]
    assert "light.tv_back" not in llm.calls[0][-1]["content"]
    assert payload["operation"] == "turn_off"
    assert [entity["candidateIndex"] for entity in payload["entities"]] == [0, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({"candidateIndexes": [2]}, "outside"),
        ({"candidateIndexes": [0, 0]}, "duplicate"),
        ({"candidateIndexes": [True]}, "integer"),
        ({"candidateIndexes": [0], "operation": "turn_on"}, "only candidateIndexes"),
    ],
)
async def test_selector_rejects_invalid_or_authority_expanding_output(
    response: object,
    error: str,
) -> None:
    llm = FakeJsonLlm(response)

    with pytest.raises(ValueError, match=error):
        await selector_class()(llm).select(
            "关灯",
            (candidate("light.main", "主灯"),),
            operation=HomeOperation.TURN_OFF,
            value=None,
        )


@pytest.mark.asyncio
async def test_selector_returns_empty_when_model_cannot_identify_target() -> None:
    llm = FakeJsonLlm({"candidateIndexes": []})

    actions = await selector_class()(llm).select(
        "关掉那一个",
        (candidate("light.main", "主灯"),),
        operation=HomeOperation.TURN_OFF,
        value=None,
    )

    assert actions == ()


@pytest.mark.asyncio
async def test_selector_rejects_candidate_that_lacks_immutable_operation() -> None:
    llm = FakeJsonLlm({"candidateIndexes": [0]})
    unsupported = candidate("light.main", "主灯").entity
    unsupported = HomeEntity(
        entity_id=unsupported.entity_id,
        domain=unsupported.domain,
        state=unsupported.state,
        attributes=unsupported.attributes,
        aliases=unsupported.aliases,
        area_name=unsupported.area_name,
        device_name=unsupported.device_name,
        operations=frozenset({HomeOperation.QUERY}),
    )

    with pytest.raises(ValueError, match="does not support"):
        await selector_class()(llm).select(
            "关灯",
            (HomeCandidate(unsupported, 0, ()),),
            operation=HomeOperation.TURN_OFF,
            value=None,
        )
