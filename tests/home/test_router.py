import json

import pytest

from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.models import HomeAction, HomeOperation
from fish_memory.home.router import HomeRouter
from fish_memory.llm.client import JsonCompletion
from fish_memory.performance.tracker import MessagePerformanceTracker


class FakeLlm:
    def __init__(self, content: dict[str, object]) -> None:
        self.content = content
        self.messages = []

    async def complete_json(self, messages):
        self.messages.append(messages)
        return JsonCompletion(content=json.dumps(self.content))

    def json_request_payload(self, messages):
        return {"messages": messages}


def _entities_command(
    *candidate_indexes: int,
    operation: str,
    value: object = None,
) -> dict[str, object]:
    return {
        "target": {
            "kind": "entities",
            "candidateIndexes": list(candidate_indexes),
        },
        "operation": operation,
        "value": value,
    }


def _catalog() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.left",
                "state": "off",
                "attributes": {"friendly_name": "左床头灯"},
            },
            {
                "entity_id": "light.right",
                "state": "off",
                "attributes": {"friendly_name": "右床头灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {}}}},
    )


def _catalog_with_unavailable_light() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.available_1",
                "state": "off",
                "attributes": {"friendly_name": "可用灯一"},
            },
            {
                "entity_id": "light.available_2",
                "state": "off",
                "attributes": {"friendly_name": "可用灯二"},
            },
            {
                "entity_id": "light.offline",
                "state": "unavailable",
                "attributes": {"friendly_name": "离线灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {}}}},
    )


def _television_catalog() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "media_player.tv",
                "state": "idle",
                "attributes": {
                    "friendly_name": "客厅电视",
                    "supported_features": 256,
                },
            },
            {
                "entity_id": "button.tv_turn_on",
                "state": "unknown",
                "attributes": {"friendly_name": "客厅电视 BLE本地控制 开机"},
            },
        ],
        entity_registry=[
            {"entity_id": "media_player.tv", "device_id": "device-tv"},
            {"entity_id": "button.tv_turn_on", "device_id": "device-tv"},
        ],
        device_registry=[
            {"id": "device-tv", "name": "客厅电视", "area_id": "living-room"}
        ],
        area_registry=[{"area_id": "living-room", "name": "客厅"}],
        services={
            "media_player": {
                "turn_on": {"fields": {}},
                "turn_off": {"fields": {}},
            },
            "button": {"press": {"fields": {}}},
        },
    )


@pytest.mark.asyncio
async def test_router_uses_candidate_indexes_not_entity_ids() -> None:
    llm = FakeLlm(
        {
            "commands": [
                {
                    "target": {"kind": "entities", "candidateIndexes": [1]},
                    "operation": "turn_on",
                    "value": None,
                }
            ]
        }
    )

    candidates = _catalog().search(
        "打开左床头灯和右床头灯", limit=40, available_only=True
    )
    decision = await HomeRouter(llm).route("打开左床头灯和右床头灯", candidates)

    assert decision.commands[0].entity_id == "light.right"
    assert decision.commands[0].operation is HomeOperation.TURN_ON
    prompt = "\n".join(message["content"] for message in llm.messages[0])
    assert "candidateIndex" in prompt
    assert "access_token" not in prompt


@pytest.mark.asyncio
async def test_router_skips_selected_media_player_without_requested_operation() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "media_player.bedroom_speaker",
                "state": "playing",
                "attributes": {
                    "friendly_name": "小爱音箱Play 音箱",
                    "supported_features": 4096,
                },
            },
            {
                "entity_id": "media_player.living_room_speaker",
                "state": "idle",
                "attributes": {
                    "friendly_name": "小爱音箱Pro 音箱",
                    "supported_features": 0,
                },
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"media_player": {"media_stop": {"fields": {}}}},
    )
    candidates = catalog.search("关闭小爱音箱", limit=40, available_only=True)
    indexes = {
        candidate.entity.entity_id: index for index, candidate in enumerate(candidates)
    }
    llm = FakeLlm(
        {
            "commands": [
                _entities_command(
                    indexes["media_player.bedroom_speaker"],
                    indexes["media_player.living_room_speaker"],
                    operation="stop",
                )
            ]
        }
    )

    decision = await HomeRouter(llm).route("关闭小爱音箱", candidates)

    assert decision.commands == (
        HomeAction(
            "media_player.bedroom_speaker",
            HomeOperation.STOP,
        ),
    )
    assert decision.skipped_commands == (
        HomeAction(
            "media_player.living_room_speaker",
            HomeOperation.STOP,
        ),
    )


@pytest.mark.asyncio
async def test_router_returns_no_actions_for_unsupported_entities() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "media_player.speaker",
                "state": "playing",
                "attributes": {
                    "friendly_name": "音箱",
                    "supported_features": 0,
                },
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"media_player": {"media_stop": {"fields": {}}}},
    )
    candidates = catalog.search("停止音箱", limit=40, available_only=True)
    llm = FakeLlm({"commands": [_entities_command(0, operation="stop")]})

    decision = await HomeRouter(llm).route("停止音箱", candidates)

    assert decision.commands == ()
    assert decision.skipped_commands == (
        HomeAction("media_player.speaker", HomeOperation.STOP),
    )


@pytest.mark.asyncio
async def test_router_rejects_set_without_explicit_set_language() -> None:
    llm = FakeLlm(
        {
            "commands": [
                {
                    "target": {
                        "kind": "set",
                        "selection": "all",
                        "area": "客厅",
                        "domain": "light",
                        "count": None,
                    },
                    "operation": "turn_off",
                    "value": None,
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="explicit set or quantity"):
        await HomeRouter(llm).route(
            "关闭客厅的灯",
            _catalog().search("关闭客厅的灯", limit=40, available_only=True),
        )


@pytest.mark.asyncio
async def test_router_returns_set_target_without_enumerating_entities() -> None:
    llm = FakeLlm(
        {
            "commands": [
                {
                    "target": {
                        "kind": "set",
                        "selection": "all",
                        "area": None,
                        "domain": "light",
                        "count": 2,
                    },
                    "operation": "turn_on",
                    "value": None,
                }
            ]
        }
    )

    decision = await HomeRouter(llm).route(
        "随便打开两个灯",
        _catalog().search("随便打开两个灯", limit=40, available_only=True),
    )

    assert len(decision.commands) == 1
    command = decision.commands[0]
    assert not isinstance(command, HomeAction)
    assert command.target.domain == "light"
    assert command.target.count == 2
    assert command.operation is HomeOperation.TURN_ON


@pytest.mark.asyncio
async def test_router_rejects_set_count_not_supported_by_current_request() -> None:
    llm = FakeLlm(
        {
            "commands": [
                {
                    "target": {
                        "kind": "set",
                        "selection": "all",
                        "area": None,
                        "domain": "light",
                        "count": 3,
                    },
                    "operation": "turn_off",
                    "value": None,
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="explicit set or quantity"):
        await HomeRouter(llm).route(
            "随便关闭两个灯",
            _catalog().search("随便关闭两个灯", limit=40, available_only=True),
        )


@pytest.mark.asyncio
async def test_router_rejects_target_fields_from_the_other_kind() -> None:
    llm = FakeLlm(
        {
            "commands": [
                {
                    "target": {
                        "kind": "entities",
                        "candidateIndexes": [0],
                        "selection": "all",
                    },
                    "operation": "turn_off",
                    "value": None,
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="entities target contains set fields"):
        await HomeRouter(llm).route(
            "关闭左床头灯",
            _catalog().search("关闭左床头灯", limit=40, available_only=True),
        )


@pytest.mark.asyncio
async def test_router_rejects_set_that_drops_explicit_area_or_domain() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "客厅灯"},
            }
        ],
        entity_registry=[{"entity_id": "light.living_room", "area_id": "living"}],
        device_registry=[],
        area_registry=[{"area_id": "living", "name": "客厅"}],
        services={"light": {"turn_off": {"fields": {}}}},
    )
    llm = FakeLlm(
        {
            "commands": [
                {
                    "target": {
                        "kind": "set",
                        "selection": "all",
                        "area": None,
                        "domain": None,
                        "count": None,
                    },
                    "operation": "turn_off",
                    "value": None,
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="drops an explicit area or domain"):
        await HomeRouter(llm).route(
            "关闭客厅所有灯",
            catalog.search("关闭客厅所有灯", limit=40, available_only=True),
        )


@pytest.mark.asyncio
async def test_router_allows_multiple_explicit_entity_indexes() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.living_left",
                "state": "on",
                "attributes": {"friendly_name": "客厅左灯"},
            },
            {
                "entity_id": "light.living_right",
                "state": "on",
                "attributes": {"friendly_name": "客厅右灯"},
            },
        ],
        entity_registry=[
            {"entity_id": "light.living_left", "area_id": "living"},
            {"entity_id": "light.living_right", "area_id": "living"},
        ],
        device_registry=[],
        area_registry=[{"area_id": "living", "name": "客厅"}],
        services={"light": {"turn_off": {"fields": {}}}},
    )
    llm = FakeLlm({"commands": [_entities_command(0, 1, operation="turn_off")]})

    decision = await HomeRouter(llm).route(
        "关闭客厅的灯",
        catalog.search("关闭客厅的灯", limit=40, available_only=True),
    )

    assert decision.commands == (
        HomeAction("light.living_left", HomeOperation.TURN_OFF),
        HomeAction("light.living_right", HomeOperation.TURN_OFF),
    )


@pytest.mark.asyncio
async def test_router_records_prompt_and_decision_for_message_debug() -> None:
    response = {"commands": [_entities_command(0, operation="turn_on")]}
    llm = FakeLlm(response)
    tracker = MessagePerformanceTracker()
    candidates = _catalog().search("打开左床头灯", limit=40, available_only=True)

    await HomeRouter(llm).route("打开左床头灯", candidates, tracker=tracker)

    step = tracker.snapshot()[0]
    assert step.name == "llm.home_router"
    assert step.input_payload == {"messages": llm.messages[0]}
    expected_content = json.dumps(response)
    assert step.output_payload == {
        "content": expected_content,
        "input_tokens": None,
        "output_tokens": None,
    }


@pytest.mark.asyncio
async def test_router_allows_ordered_operations_for_one_entity() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "climate.living_room",
                "state": "off",
                "attributes": {"friendly_name": "客厅空调"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "climate": {
                "turn_on": {"fields": {}},
                "set_temperature": {"fields": {"temperature": {}}},
            }
        },
    )
    llm = FakeLlm(
        {
            "commands": [
                _entities_command(0, operation="turn_on"),
                _entities_command(0, operation="set_temperature", value=24),
            ]
        }
    )

    decision = await HomeRouter(llm).route(
        "打开客厅空调并调到24度",
        catalog.search("打开客厅空调并调到24度", limit=40, available_only=True),
    )

    assert decision.commands == (
        HomeAction("climate.living_room", HomeOperation.TURN_ON),
        HomeAction("climate.living_room", HomeOperation.SET_TEMPERATURE, 24),
    )


@pytest.mark.asyncio
async def test_router_rejects_conflicting_values_for_same_entity_operation() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "climate.living_room",
                "state": "heat",
                "attributes": {"friendly_name": "客厅空调"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"climate": {"set_temperature": {"fields": {"temperature": {}}}}},
    )
    llm = FakeLlm(
        {
            "commands": [
                _entities_command(0, operation="set_temperature", value=24),
                _entities_command(0, operation="set_temperature", value=26),
            ]
        }
    )

    with pytest.raises(ValueError, match="conflicting values"):
        await HomeRouter(llm).route(
            "把客厅空调调到24度",
            catalog.search("把客厅空调调到24度", limit=40, available_only=True),
        )


@pytest.mark.asyncio
async def test_router_uses_history_only_to_resolve_current_command() -> None:
    llm = FakeLlm({"commands": [_entities_command(1, operation="turn_on")]})
    candidates = _catalog().search(
        "打开一个卧室灯 再打开一个", limit=40, available_only=True
    )

    decision = await HomeRouter(llm).route(
        "再打开一个",
        candidates,
        history=({"role": "user", "content": "打开一个卧室灯"},),
    )

    assert decision.commands == (HomeAction("light.right", HomeOperation.TURN_ON),)
    payload = json.loads(llm.messages[0][1]["content"])
    assert payload["content"] == "再打开一个"
    assert payload["history"] == [{"role": "user", "content": "打开一个卧室灯"}]


@pytest.mark.asyncio
async def test_router_can_request_warm_light_for_capable_light() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.desk",
                "state": "on",
                "attributes": {
                    "friendly_name": "书桌灯1 灯光",
                    "supported_color_modes": ["color_temp"],
                    "min_color_temp_kelvin": 3000,
                    "max_color_temp_kelvin": 6400,
                },
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {"color_temp_kelvin": {}}}}},
    )
    llm = FakeLlm(
        {
            "commands": [
                _entities_command(0, operation="set_color_temperature", value="warm")
            ]
        }
    )

    candidates = catalog.search("书桌灯1调成暖光", limit=40, available_only=True)
    decision = await HomeRouter(llm).route("书桌灯1调成暖光", candidates)

    assert decision.commands == (
        HomeAction(
            "light.desk",
            HomeOperation.SET_COLOR_TEMPERATURE,
            "warm",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "value", "expected"),
    [
        ("月亮亮度调低", None, 40),
        ("月亮亮度调低", "decrease", 40),
        ("月亮亮度调高", "increase", 60),
        ("月亮亮度调低到30%", 30, 30),
    ],
)
async def test_router_resolves_relative_brightness_from_current_state(
    text: str,
    value: object,
    expected: int,
) -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.moon",
                "state": "on",
                "attributes": {"friendly_name": "月亮 灯光", "brightness": 128},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {"brightness_pct": {}}}}},
    )
    llm = FakeLlm(
        {"commands": [_entities_command(0, operation="set_brightness", value=value)]}
    )

    candidates = catalog.search("月亮", limit=40, available_only=True)
    decision = await HomeRouter(llm).route(text, candidates)

    assert decision.commands == (
        HomeAction("light.moon", HomeOperation.SET_BRIGHTNESS, expected),
    )
    assert "只选择 state 为 on 的灯" in llm.messages[0][0]["content"]


@pytest.mark.asyncio
async def test_router_rejects_invalid_candidate_index() -> None:
    llm = FakeLlm({"commands": [_entities_command(9, operation="turn_on")]})

    with pytest.raises(ValueError, match="candidateIndex"):
        candidates = _catalog().search(
            "打开左床头灯和右床头灯", limit=40, available_only=True
        )
        await HomeRouter(llm).route("打开左床头灯和右床头灯", candidates)


@pytest.mark.asyncio
async def test_router_collapses_duplicate_actions() -> None:
    llm = FakeLlm(
        {
            "commands": [
                _entities_command(0, operation="turn_on"),
                _entities_command(0, operation="turn_on"),
            ]
        }
    )

    candidates = _catalog().search("打开左床头灯", limit=40, available_only=True)
    decision = await HomeRouter(llm).route("打开左床头灯", candidates)

    assert len(decision.commands) == 1


@pytest.mark.asyncio
async def test_router_rejects_conflicting_actions_for_one_entity() -> None:
    llm = FakeLlm(
        {
            "commands": [
                _entities_command(0, operation="turn_on"),
                _entities_command(0, operation="turn_off"),
            ]
        }
    )

    with pytest.raises(ValueError, match="conflicting"):
        candidates = _catalog().search(
            "打开再关闭左床头灯", limit=40, available_only=True
        )
        await HomeRouter(llm).route("打开再关闭左床头灯", candidates)


@pytest.mark.asyncio
async def test_router_receives_generic_online_device_candidates_only() -> None:
    llm = FakeLlm(
        {
            "commands": [
                {
                    "target": {
                        "kind": "set",
                        "selection": "all",
                        "area": None,
                        "domain": "light",
                        "count": 2,
                    },
                    "operation": "turn_on",
                    "value": None,
                }
            ]
        }
    )

    catalog = _catalog_with_unavailable_light()
    candidates = catalog.search("随便开两个灯", limit=40, available_only=True)
    decision = await HomeRouter(llm).route("随便开两个灯", candidates)

    assert len(decision.commands) == 1
    prompt = "\n".join(message["content"] for message in llm.messages[0])
    assert "可用灯一" in prompt
    assert "可用灯二" in prompt
    assert "离线灯" not in prompt


@pytest.mark.asyncio
async def test_router_can_select_related_power_button_for_television() -> None:
    llm = FakeLlm({"commands": [_entities_command(0, operation="activate")]})

    catalog = _television_catalog()
    candidates = catalog.search("打开电视机", limit=40, available_only=True)
    decision = await HomeRouter(llm).route("打开电视机", candidates)

    assert decision.commands == (
        HomeAction("button.tv_turn_on", HomeOperation.ACTIVATE),
    )
    payload = json.loads(llm.messages[0][1]["content"])
    assert payload["devices"] == [
        {
            "name": "客厅电视",
            "area": "客厅",
            "entities": [
                {
                    "candidateIndex": 0,
                    "name": "客厅电视 BLE本地控制 开机",
                    "domain": "button",
                    "state": "unknown",
                    "allowedOperations": ["activate"],
                },
                {
                    "candidateIndex": 1,
                    "name": "客厅电视",
                    "domain": "media_player",
                    "state": "idle",
                    "allowedOperations": ["query", "turn_off"],
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_router_keeps_same_operation_buttons_as_distinct_endpoints() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "button.play_radio",
                "state": "unknown",
                "attributes": {"friendly_name": "小爱音箱Play 智能音箱 播放电台"},
            },
            {
                "entity_id": "button.stop_alarm",
                "state": "unknown",
                "attributes": {"friendly_name": "小爱音箱Play 闹钟 停止闹钟"},
            },
            {
                "entity_id": "button.wake",
                "state": "unknown",
                "attributes": {"friendly_name": "小爱音箱Play 智能音箱 唤醒"},
            },
        ],
        entity_registry=[
            {"entity_id": entity_id, "device_id": "device-speaker"}
            for entity_id in (
                "button.play_radio",
                "button.stop_alarm",
                "button.wake",
            )
        ],
        device_registry=[{"id": "device-speaker", "name": "小爱音箱Play"}],
        area_registry=[],
        services={"button": {"press": {"fields": {}}}},
    )
    candidates = catalog.search("小爱音箱Play", limit=40, available_only=True)
    stop_index = next(
        index
        for index, candidate in enumerate(candidates)
        if candidate.entity.entity_id == "button.stop_alarm"
    )
    llm = FakeLlm({"commands": [_entities_command(stop_index, operation="activate")]})

    decision = await HomeRouter(llm).route("停止小爱音箱的闹钟", candidates)

    assert decision.commands == (
        HomeAction("button.stop_alarm", HomeOperation.ACTIVATE),
    )
    payload = json.loads(llm.messages[0][1]["content"])
    assert len(payload["devices"]) == 1
    assert {entity["name"] for entity in payload["devices"][0]["entities"]} == {
        "小爱音箱Play 智能音箱 播放电台",
        "小爱音箱Play 闹钟 停止闹钟",
        "小爱音箱Play 智能音箱 唤醒",
    }
    assert all(
        entity["allowedOperations"] == ["activate"]
        for entity in payload["devices"][0]["entities"]
    )


@pytest.mark.asyncio
async def test_router_command_limit_is_independent_from_device_limit() -> None:
    llm = FakeLlm(
        {
            "commands": [
                _entities_command(0, operation="turn_on"),
                _entities_command(1, operation="turn_on"),
            ]
        }
    )
    candidates = _catalog().search(
        "打开左床头灯和右床头灯", limit=40, available_only=True
    )

    with pytest.raises(ValueError, match="command count limit"):
        await HomeRouter(llm, device_limit=1, command_limit=1).route(
            "打开左床头灯和右床头灯", candidates
        )


@pytest.mark.asyncio
async def test_router_enforces_device_limit_when_called_directly() -> None:
    llm = FakeLlm({"commands": []})
    candidates = _catalog().search(
        "打开左床头灯和右床头灯", limit=40, available_only=True
    )

    await HomeRouter(llm, device_limit=1).route("打开左床头灯和右床头灯", candidates)

    payload = json.loads(llm.messages[0][1]["content"])
    assert len(payload["devices"]) == 1
    assert "deviceIndex" not in payload["devices"][0]
    assert len(payload["devices"][0]["entities"]) == 1
