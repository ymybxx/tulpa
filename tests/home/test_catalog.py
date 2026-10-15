from fish_memory.home.catalog import (
    HomeEntityCatalog,
    group_candidates_by_device,
    limit_candidates_by_device,
    normalize_home_alias,
    requested_home_domains,
)
from fish_memory.home.models import HomeOperation


def test_normalize_home_alias_removes_noise_without_translating_names() -> None:
    assert normalize_home_alias(" 鞋柜灯  灯光 ") == "鞋柜灯灯光"
    assert normalize_home_alias("Living-Room Light") == "livingroomlight"


def test_catalog_indexes_generic_home_assistant_metadata() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.entry",
                "state": "off",
                "attributes": {
                    "friendly_name": "鞋柜灯  灯光",
                    "supported_color_modes": ["brightness"],
                },
            },
            {
                "entity_id": "switch.entry",
                "state": "on",
                "attributes": {"friendly_name": "鞋柜灯插座"},
            },
        ],
        entity_registry=[
            {
                "entity_id": "light.entry",
                "name": "鞋柜灯",
                "original_name": "Entrance Light",
                "device_id": "device-1",
                "area_id": "entry",
            }
        ],
        device_registry=[{"id": "device-1", "name": "门口照明", "area_id": "entry"}],
        area_registry=[{"area_id": "entry", "name": "玄关"}],
        services={
            "light": {
                "turn_on": {"fields": {"brightness_pct": {}}},
                "turn_off": {"fields": {}},
            }
        },
    )

    candidates = catalog.search("把玄关鞋柜灯调亮", limit=5)

    assert candidates[0].entity.entity_id == "light.entry"
    assert candidates[0].entity.area_name == "玄关"
    assert "turn_on" in candidates[0].entity.operations
    assert "set_brightness" in candidates[0].entity.operations
    assert candidates[0].entity.service_schemas["turn_on"] == frozenset(
        {"brightness_pct"}
    )


def test_catalog_exposes_color_temperature_only_for_capable_lights() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.desk",
                "state": "on",
                "attributes": {
                    "friendly_name": "书桌灯",
                    "supported_color_modes": ["color_temp"],
                    "min_color_temp_kelvin": 3000,
                    "max_color_temp_kelvin": 6400,
                },
            },
            {
                "entity_id": "light.basic",
                "state": "on",
                "attributes": {
                    "friendly_name": "基础灯",
                    "supported_color_modes": ["brightness"],
                },
            },
            {
                "entity_id": "light.incomplete",
                "state": "on",
                "attributes": {
                    "friendly_name": "缺少范围的灯",
                    "supported_color_modes": ["color_temp"],
                },
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "light": {
                "turn_on": {"fields": {"color_temp_kelvin": {}, "brightness_pct": {}}}
            }
        },
    )

    desk = catalog.get("light.desk")
    basic = catalog.get("light.basic")
    incomplete = catalog.get("light.incomplete")

    assert desk is not None
    assert basic is not None
    assert incomplete is not None
    assert HomeOperation.SET_COLOR_TEMPERATURE in desk.operations
    assert HomeOperation.SET_COLOR_TEMPERATURE not in basic.operations
    assert HomeOperation.SET_COLOR_TEMPERATURE not in incomplete.operations


def test_catalog_keeps_ambiguous_aliases_as_multiple_candidates() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.left",
                "state": "off",
                "attributes": {"friendly_name": "床头灯"},
            },
            {
                "entity_id": "light.right",
                "state": "off",
                "attributes": {"friendly_name": "床头灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={},
    )

    matches = catalog.exact("床头灯")

    assert {item.entity_id for item in matches} == {
        "light.left",
        "light.right",
    }


def test_catalog_broadly_matches_device_type_and_excludes_unavailable() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.first",
                "state": "off",
                "attributes": {"friendly_name": "玄关顶灯"},
            },
            {
                "entity_id": "light.second",
                "state": "on",
                "attributes": {"friendly_name": "卧室台灯"},
            },
            {
                "entity_id": "light.offline",
                "state": "unavailable",
                "attributes": {"friendly_name": "离线灯"},
            },
            {
                "entity_id": "switch.other",
                "state": "off",
                "attributes": {"friendly_name": "加湿器"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {}}}},
    )

    candidates = catalog.search(
        "随便开两个灯",
        limit=10,
        available_only=True,
    )

    assert [item.entity.entity_id for item in candidates] == [
        "light.first",
        "light.second",
    ]


def test_area_alias_alone_does_not_fill_device_candidates() -> None:
    button_states = [
        {
            "entity_id": f"button.living_room_{index:02d}",
            "state": "unknown",
            "attributes": {"friendly_name": f"客厅辅助按钮 {index}"},
        }
        for index in range(45)
    ]
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            *button_states,
            {
                "entity_id": "light.tv_strip",
                "state": "on",
                "attributes": {
                    "friendly_name": "电视灯带 灯",
                    "brightness": 255,
                    "supported_color_modes": ["brightness"],
                },
            },
        ],
        entity_registry=[
            {
                "entity_id": state["entity_id"],
                "area_id": "living_room",
            }
            for state in [*button_states, {"entity_id": "light.tv_strip"}]
        ],
        device_registry=[],
        area_registry=[{"area_id": "living_room", "name": "客厅"}],
        services={
            "button": {"press": {"fields": {}}},
            "light": {"turn_on": {"fields": {"brightness_pct": {}}}},
        },
    )

    candidates = catalog.search(
        "客厅灯调最暗",
        limit=40,
        available_only=True,
    )

    assert [candidate.entity.entity_id for candidate in candidates] == [
        "light.tv_strip"
    ]
    assert candidates[0].entity.state == "on"


def test_requested_home_domains_recognizes_explicit_device_type() -> None:
    assert requested_home_domains("关闭空调") == frozenset({"climate"})


def test_catalog_derives_operations_for_supported_domains() -> None:
    states = [
        {
            "entity_id": f"{domain}.sample",
            "state": "off",
            "attributes": {"friendly_name": domain},
        }
        for domain in (
            "switch",
            "climate",
            "media_player",
            "cover",
            "fan",
            "lock",
            "vacuum",
            "scene",
            "script",
        )
    ]
    catalog = HomeEntityCatalog.from_payloads(
        states=states,
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={},
    )

    assert "turn_on" in catalog.get("switch.sample").operations
    assert "set_temperature" in catalog.get("climate.sample").operations
    assert "play" in catalog.get("media_player.sample").operations
    assert "open" in catalog.get("cover.sample").operations
    assert "set_percentage" in catalog.get("fan.sample").operations
    assert "unlock" in catalog.get("lock.sample").operations
    assert "start" in catalog.get("vacuum.sample").operations
    assert "activate" in catalog.get("scene.sample").operations
    assert "activate" in catalog.get("script.sample").operations


def test_catalog_exposes_climate_power_operations_from_available_services() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "climate.living_room",
                "state": "cool",
                "attributes": {"friendly_name": "客厅空调"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "climate": {
                "turn_on": {"fields": {}},
                "turn_off": {"fields": {}},
                "set_temperature": {"fields": {"temperature": {}}},
            }
        },
    )

    operations = catalog.get("climate.living_room").operations

    assert HomeOperation.TURN_ON in operations
    assert HomeOperation.TURN_OFF in operations


def test_catalog_does_not_expose_missing_climate_power_services() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "climate.living_room",
                "state": "cool",
                "attributes": {"friendly_name": "客厅空调"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"climate": {"set_temperature": {"fields": {}}}},
    )

    operations = catalog.get("climate.living_room").operations

    assert HomeOperation.TURN_ON not in operations
    assert HomeOperation.TURN_OFF not in operations


def test_catalog_uses_entity_features_and_includes_related_action_button() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "media_player.tv",
                "state": "idle",
                "attributes": {
                    "friendly_name": "客厅电视",
                    # Supports turn_off but not turn_on.
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
        device_registry=[{"id": "device-tv", "name": "客厅电视"}],
        area_registry=[],
        services={
            "media_player": {
                "turn_on": {"fields": {}},
                "turn_off": {"fields": {}},
            },
            "button": {"press": {"fields": {}}},
        },
    )

    media_player = catalog.get("media_player.tv")
    candidates = catalog.search("打开电视机", limit=10, available_only=True)

    assert HomeOperation.TURN_ON not in media_player.operations
    assert HomeOperation.TURN_OFF in media_player.operations
    assert [candidate.entity.entity_id for candidate in candidates] == [
        "button.tv_turn_on",
        "media_player.tv",
    ]
    assert candidates[0].entity.operations == frozenset({HomeOperation.ACTIVATE})


def test_device_limit_keeps_all_endpoints_for_selected_devices() -> None:
    action_count = 45
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            *[
                {
                    "entity_id": f"button.action_{index:02d}",
                    "state": "unknown",
                    "attributes": {
                        "friendly_name": f"测试设备动作{index}",
                    },
                }
                for index in range(action_count)
            ],
            {
                "entity_id": "light.target",
                "state": "off",
                "attributes": {"friendly_name": "测试设备"},
            },
        ],
        entity_registry=[
            *[
                {
                    "entity_id": f"button.action_{index:02d}",
                    "device_id": "device-actions",
                }
                for index in range(action_count)
            ],
            {"entity_id": "light.target", "device_id": "device-target"},
        ],
        device_registry=[
            {"id": "device-actions", "name": "动作设备"},
            {"id": "device-target", "name": "目标设备"},
        ],
        area_registry=[],
        services={
            "button": {"press": {"fields": {}}},
            "light": {"turn_on": {"fields": {}}},
        },
    )

    candidates = catalog.search("测试设备", limit=100, available_only=True)
    limited = limit_candidates_by_device(candidates, limit=2)
    groups = group_candidates_by_device(limited)

    assert len(limited) == action_count + 1
    assert len(groups) == 2
    assert groups[0][0].entity.entity_id == "light.target"
    assert len(groups[1]) == action_count


def test_entities_without_device_id_count_as_independent_devices() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": f"button.action_{index}",
                "state": "unknown",
                "attributes": {"friendly_name": f"测试设备动作{index}"},
            }
            for index in range(3)
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"button": {"press": {"fields": {}}}},
    )

    candidates = catalog.search("测试设备", limit=10, available_only=True)
    limited = limit_candidates_by_device(candidates, limit=2)

    assert len(limited) == 2
    assert len(group_candidates_by_device(limited)) == 2
