from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.models import HomeOperation
from fish_memory.home.parser import LocalHomeCommandParser


def _catalog(*names: str) -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": f"light.light_{index}",
                "state": "off",
                "attributes": {"friendly_name": name, "brightness": 128},
            }
            for index, name in enumerate(names)
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {"brightness_pct": {}}}}},
    )


def _cover_catalog() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "button.bedroom_curtain_identify",
                "state": "unknown",
                "attributes": {"friendly_name": "窗帘 设备确认"},
            },
            {
                "entity_id": "cover.bedroom",
                "state": "closed",
                "attributes": {"friendly_name": "主卧窗帘"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "cover": {
                "open_cover": {"fields": {}},
                "close_cover": {"fields": {}},
            },
            "button": {"press": {"fields": {}}},
        },
    )


def test_parser_routes_unique_explicit_control_without_llm() -> None:
    decision = LocalHomeCommandParser().parse("把鞋柜灯调亮到40%", _catalog("鞋柜灯"))

    assert decision is not None
    assert decision.commands[0].entity_id == "light.light_0"
    assert decision.commands[0].operation is HomeOperation.SET_BRIGHTNESS
    assert decision.commands[0].value == 40


def test_parser_routes_state_query() -> None:
    decision = LocalHomeCommandParser().parse("鞋柜灯开着吗？", _catalog("鞋柜灯"))

    assert decision is not None
    assert decision.commands[0].operation is HomeOperation.QUERY


def test_parser_maps_open_and_close_to_cover_operations() -> None:
    parser = LocalHomeCommandParser()

    opened = parser.parse("打开窗帘", _cover_catalog())
    closed = parser.parse("关闭窗帘", _cover_catalog())

    assert opened is not None
    assert opened.commands[0].operation is HomeOperation.OPEN
    assert closed is not None
    assert closed.commands[0].operation is HomeOperation.CLOSE


def test_parser_rejects_ambiguous_or_non_imperative_text() -> None:
    parser = LocalHomeCommandParser()

    assert parser.parse("打开床头灯", _catalog("床头灯", "床头灯")) is None
    assert parser.parse("我没有让你打开鞋柜灯", _catalog("鞋柜灯")) is None
    assert parser.parse("如果打开鞋柜灯会怎样", _catalog("鞋柜灯")) is None
    assert parser.parse("他说‘打开鞋柜灯’", _catalog("鞋柜灯")) is None


def test_parser_does_not_treat_entity_name_containing_bie_as_negation() -> None:
    decision = LocalHomeCommandParser().parse("打开别墅灯", _catalog("别墅灯"))

    assert decision is not None
    assert decision.commands[0].operation is HomeOperation.TURN_ON


def test_parser_maps_relative_brightness_from_current_state() -> None:
    parser = LocalHomeCommandParser()

    brighter = parser.parse("把鞋柜灯调亮一点", _catalog("鞋柜灯"))
    dimmer = parser.parse("把鞋柜灯调暗一点", _catalog("鞋柜灯"))

    assert brighter is not None
    assert brighter.commands[0].operation is HomeOperation.SET_BRIGHTNESS
    assert brighter.commands[0].value == 60
    assert dimmer is not None
    assert dimmer.commands[0].value == 40


def test_parser_maps_relative_brightness_high_and_low_wording() -> None:
    parser = LocalHomeCommandParser()

    brighter = parser.parse("把鞋柜灯亮度调高", _catalog("鞋柜灯"))
    dimmer = parser.parse("鞋柜灯亮度调低", _catalog("鞋柜灯"))

    assert brighter is not None
    assert brighter.commands[0].value == 60
    assert dimmer is not None
    assert dimmer.commands[0].operation is HomeOperation.SET_BRIGHTNESS
    assert dimmer.commands[0].value == 40


def test_parser_leaves_multi_device_scope_to_llm_router() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.on",
                "state": "on",
                "attributes": {"friendly_name": "开着的灯"},
            },
            {
                "entity_id": "light.off",
                "state": "off",
                "attributes": {"friendly_name": "关着的灯"},
            },
            {
                "entity_id": "light.unavailable",
                "state": "unavailable",
                "attributes": {"friendly_name": "离线灯"},
            },
            {
                "entity_id": "light.unknown",
                "state": "unknown",
                "attributes": {"friendly_name": "未知灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_off": {"fields": {}}}},
    )

    decision = LocalHomeCommandParser().parse("关闭所有灯", catalog)

    assert decision is None


def test_parser_infers_candidate_action_without_resolving_target() -> None:
    catalog = _catalog("左床头灯", "右床头灯")
    parser = LocalHomeCommandParser()

    actions = [
        parser.infer_action("关闭所有灯", candidate.entity)
        for candidate in catalog.search("关闭所有灯", limit=10)
    ]

    assert [action.operation for action in actions if action is not None] == [
        HomeOperation.TURN_OFF,
        HomeOperation.TURN_OFF,
    ]


def test_parser_does_not_infer_one_target_state_from_conflicting_actions() -> None:
    entity = _catalog("鞋柜灯").get("light.light_0")
    assert entity is not None

    action = LocalHomeCommandParser().infer_action("先打开鞋柜灯再关闭", entity)

    assert action is None
