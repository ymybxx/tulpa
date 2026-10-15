from typing import Any

import pytest

from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.client import HomeActionUncertainError, HomeAssistantError
from fish_memory.home.executor import HomeActionExecutor
from fish_memory.home.models import HomeAction, HomeOperation


class FakeClient:
    def __init__(
        self,
        catalog: HomeEntityCatalog,
        *,
        observed_state: str | None = "on",
        observed_cover_state: str | None = "open",
    ) -> None:
        self.catalog = catalog
        self.observed_state = observed_state
        self.observed_cover_state = observed_cover_state
        self.calls: list[tuple[str, str, dict[str, object], dict[str, object]]] = []
        self.waits: list[tuple[str, frozenset[str], float]] = []
        self.cover_waits: list[tuple[str, float, int, float]] = []

    async def call_service(
        self,
        domain: str,
        service: str,
        *,
        service_data: dict[str, object] | None = None,
        target: dict[str, object] | None = None,
    ) -> Any:
        self.calls.append((domain, service, service_data or {}, target or {}))
        return {"ok": True}

    async def wait_for_state(
        self,
        entity_id: str,
        expected_states: frozenset[str],
        *,
        timeout_seconds: float,
    ) -> str | None:
        self.waits.append((entity_id, expected_states, timeout_seconds))
        return self.observed_state

    async def wait_for_cover_movement(
        self,
        entity_id: str,
        initial_position: float,
        target_position: int,
        *,
        timeout_seconds: float,
    ) -> str | None:
        self.cover_waits.append(
            (entity_id, initial_position, target_position, timeout_seconds)
        )
        return self.observed_cover_state


class StopRejectingClient(FakeClient):
    async def call_service(
        self,
        domain: str,
        service: str,
        *,
        service_data: dict[str, object] | None = None,
        target: dict[str, object] | None = None,
    ) -> Any:
        self.calls.append((domain, service, service_data or {}, target or {}))
        if service == "media_stop":
            raise HomeAssistantError("provider rejected media_stop")
        return {"ok": True}


class UncertainStopClient(StopRejectingClient):
    async def call_service(
        self,
        domain: str,
        service: str,
        *,
        service_data: dict[str, object] | None = None,
        target: dict[str, object] | None = None,
    ) -> Any:
        self.calls.append((domain, service, service_data or {}, target or {}))
        raise HomeActionUncertainError("connection lost after dispatch")


def _catalog() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.entry",
                "state": "off",
                "attributes": {"friendly_name": "玄关灯"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {"brightness_pct": {}}}}},
    )


def _media_catalog() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "media_player.living_room",
                "state": "playing",
                "attributes": {"friendly_name": "客厅音箱"},
            },
            {
                "entity_id": "cover.bedroom",
                "state": "opening",
                "attributes": {"friendly_name": "卧室窗帘"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "media_player": {"media_stop": {"fields": {}}},
            "cover": {"stop_cover": {"fields": {}}},
        },
    )


def _powered_light_catalog(*, brightness: int) -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.entry",
                "state": "on",
                "attributes": {
                    "friendly_name": "玄关灯",
                    "brightness": brightness,
                },
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {"brightness_pct": {}}}}},
    )


def _idle_media_catalog() -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "media_player.living_room_tv",
                "state": "idle",
                "attributes": {"friendly_name": "客厅电视"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"media_player": {"turn_on": {"fields": {}}}},
    )


def _positioned_cover_catalog(*, state: str, position: int) -> HomeEntityCatalog:
    return HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "cover.bedroom",
                "state": state,
                "attributes": {
                    "friendly_name": "卧室窗帘",
                    "current_position": position,
                },
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "cover": {
                "open_cover": {"fields": {}},
                "close_cover": {"fields": {}},
            }
        },
    )


@pytest.mark.asyncio
async def test_executor_maps_brightness_to_validated_service() -> None:
    client = FakeClient(_catalog())
    executor = HomeActionExecutor(client)

    result = await executor.execute(
        HomeAction("light.entry", HomeOperation.SET_BRIGHTNESS, 40)
    )

    assert result.accepted is True
    assert result.service_domain == "light"
    assert result.service_name == "turn_on"
    assert client.calls == [
        (
            "light",
            "turn_on",
            {"brightness_pct": 40},
            {"entity_id": "light.entry"},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected_kelvin"),
    [("warm", 3000), ("neutral", 4700), ("cool", 6400), (4200, 4200)],
)
async def test_executor_maps_color_temperature_to_light_range(
    value: str | int,
    expected_kelvin: int,
) -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.desk",
                "state": "on",
                "attributes": {
                    "friendly_name": "书桌灯",
                    "supported_color_modes": ["color_temp"],
                    "color_temp_kelvin": 5000,
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
    client = FakeClient(catalog)

    result = await HomeActionExecutor(client).execute(
        HomeAction("light.desk", HomeOperation.SET_COLOR_TEMPERATURE, value)
    )

    assert result.accepted is True
    assert result.service_name == "turn_on"
    assert client.calls == [
        (
            "light",
            "turn_on",
            {"color_temp_kelvin": expected_kelvin},
            {"entity_id": "light.desk"},
        )
    ]


@pytest.mark.asyncio
async def test_executor_supports_legacy_mired_color_temperature() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.legacy",
                "state": "on",
                "attributes": {
                    "friendly_name": "旧版灯",
                    "supported_color_modes": ["color_temp"],
                    "min_mireds": 153,
                    "max_mireds": 500,
                },
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {"color_temp": {}}}}},
    )
    client = FakeClient(catalog)

    await HomeActionExecutor(client).execute(
        HomeAction(
            "light.legacy",
            HomeOperation.SET_COLOR_TEMPERATURE,
            "warm",
        )
    )

    assert client.calls[0][2] == {"color_temp": 500}


@pytest.mark.asyncio
async def test_executor_does_not_treat_power_state_as_brightness_target() -> None:
    client = FakeClient(_powered_light_catalog(brightness=128))

    result = await HomeActionExecutor(client).execute(
        HomeAction("light.entry", HomeOperation.SET_BRIGHTNESS, 40)
    )

    assert result.accepted is True
    assert result.service_name == "turn_on"
    assert client.calls[0][2] == {"brightness_pct": 40}


@pytest.mark.asyncio
async def test_executor_skips_parameter_action_already_at_target() -> None:
    client = FakeClient(_powered_light_catalog(brightness=102))

    result = await HomeActionExecutor(client).execute(
        HomeAction("light.entry", HomeOperation.SET_BRIGHTNESS, 40)
    )

    assert result.accepted is True
    assert result.confirmed is True
    assert result.state == "on"
    assert client.calls == []


@pytest.mark.asyncio
async def test_executor_skips_state_action_already_at_target() -> None:
    client = FakeClient(_catalog())

    result = await HomeActionExecutor(client).execute(
        HomeAction("light.entry", HomeOperation.TURN_OFF)
    )

    assert result.accepted is True
    assert result.confirmed is True
    assert result.state == "off"
    assert client.calls == []


@pytest.mark.asyncio
async def test_executor_opens_cover_that_home_assistant_only_reports_as_open() -> None:
    client = FakeClient(
        _positioned_cover_catalog(state="open", position=1),
        observed_cover_state="open",
    )

    result = await HomeActionExecutor(client).execute(
        HomeAction("cover.bedroom", HomeOperation.OPEN)
    )

    assert result.accepted is True
    assert result.confirmed is True
    assert result.service_name == "open_cover"
    assert client.calls == [
        (
            "cover",
            "open_cover",
            {},
            {"entity_id": "cover.bedroom"},
        )
    ]
    assert client.cover_waits == [("cover.bedroom", 1.0, 100, 3.0)]
    assert client.waits == []


@pytest.mark.asyncio
async def test_executor_skips_cover_only_when_position_reaches_target() -> None:
    client = FakeClient(_positioned_cover_catalog(state="open", position=100))

    result = await HomeActionExecutor(client).execute(
        HomeAction("cover.bedroom", HomeOperation.OPEN)
    )

    assert result.confirmed is True
    assert client.calls == []
    assert client.cover_waits == []


@pytest.mark.asyncio
async def test_executor_uses_native_climate_turn_off_service() -> None:
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
        services={"climate": {"turn_off": {"fields": {}}}},
    )
    client = FakeClient(catalog, observed_state="off")

    result = await HomeActionExecutor(client).execute(
        HomeAction("climate.living_room", HomeOperation.TURN_OFF)
    )

    assert result.accepted is True
    assert result.confirmed is True
    assert result.service_name == "turn_off"
    assert client.calls == [
        (
            "climate",
            "turn_off",
            {},
            {"entity_id": "climate.living_room"},
        )
    ]


@pytest.mark.asyncio
async def test_executor_treats_active_media_player_as_already_on() -> None:
    client = FakeClient(_media_catalog())

    result = await HomeActionExecutor(client).execute(
        HomeAction("media_player.living_room", HomeOperation.TURN_ON)
    )

    assert result.accepted is True
    assert result.confirmed is True
    assert result.state == "playing"
    assert client.calls == []


@pytest.mark.asyncio
async def test_executor_turns_on_idle_media_player_because_power_is_uncertain() -> None:
    client = FakeClient(_idle_media_catalog(), observed_state="idle")

    result = await HomeActionExecutor(client).execute(
        HomeAction("media_player.living_room_tv", HomeOperation.TURN_ON)
    )

    assert result.accepted is True
    assert result.service_name == "turn_on"
    assert client.calls == [
        (
            "media_player",
            "turn_on",
            {},
            {"entity_id": "media_player.living_room_tv"},
        )
    ]


@pytest.mark.asyncio
async def test_executor_presses_stateless_action_button() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "button.tv_turn_on",
                "state": "unknown",
                "attributes": {"friendly_name": "客厅电视 开机"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"button": {"press": {"fields": {}}}},
    )
    client = FakeClient(catalog, observed_state=None)

    result = await HomeActionExecutor(client).execute(
        HomeAction("button.tv_turn_on", HomeOperation.ACTIVATE)
    )

    assert result.accepted is True
    assert result.service_name == "press"
    assert client.calls == [("button", "press", {}, {"entity_id": "button.tv_turn_on"})]


@pytest.mark.asyncio
async def test_executor_queries_cached_state_without_service_call() -> None:
    client = FakeClient(_catalog())

    result = await HomeActionExecutor(client).execute(
        HomeAction("light.entry", HomeOperation.QUERY)
    )

    assert result.state == "off"
    assert result.service_domain is None
    assert result.service_name is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_executor_rejects_unsupported_operation() -> None:
    client = FakeClient(_catalog())

    with pytest.raises(ValueError, match="does not support"):
        await HomeActionExecutor(client).execute(
            HomeAction("light.entry", HomeOperation.UNLOCK)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entity_id", "expected_service"),
    (
        ("media_player.living_room", "media_stop"),
        ("cover.bedroom", "stop_cover"),
    ),
)
async def test_executor_maps_stop_service_by_entity_domain(
    entity_id: str,
    expected_service: str,
) -> None:
    client = FakeClient(_media_catalog())

    await HomeActionExecutor(client).execute(HomeAction(entity_id, HomeOperation.STOP))

    assert client.calls[0][1] == expected_service


@pytest.mark.asyncio
async def test_executor_falls_back_to_pause_when_media_stop_is_rejected() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "media_player.speaker",
                "state": "playing",
                "attributes": {
                    "friendly_name": "音箱",
                    "supported_features": 4097,
                },
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "media_player": {
                "media_stop": {"fields": {}},
                "media_pause": {"fields": {}},
            }
        },
    )
    client = StopRejectingClient(catalog, observed_state="paused")

    result = await HomeActionExecutor(client).execute(
        HomeAction("media_player.speaker", HomeOperation.STOP)
    )

    assert [call[1] for call in client.calls] == ["media_stop", "media_pause"]
    assert result.accepted is True
    assert result.confirmed is False
    assert result.state is None
    assert result.service_name == "media_pause"


@pytest.mark.asyncio
async def test_executor_does_not_retry_uncertain_media_stop() -> None:
    client = UncertainStopClient(_media_catalog())

    with pytest.raises(HomeActionUncertainError):
        await HomeActionExecutor(client).execute(
            HomeAction("media_player.living_room", HomeOperation.STOP)
        )

    assert [call[1] for call in client.calls] == ["media_stop"]


@pytest.mark.asyncio
async def test_executor_rejects_unavailable_entity_without_dispatch() -> None:
    catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.entry",
                "state": "unavailable",
                "attributes": {"friendly_name": "玄关灯"},
            }
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_off": {"fields": {}}}},
    )
    client = FakeClient(catalog)

    result = await HomeActionExecutor(client).execute(
        HomeAction("light.entry", HomeOperation.TURN_OFF)
    )

    assert result.accepted is False
    assert result.confirmed is False
    assert result.state == "unavailable"
    assert result.error == "设备当前不可用"
    assert client.calls == []


@pytest.mark.asyncio
async def test_executor_accepts_unconfirmed_target_state_without_error() -> None:
    client = FakeClient(_catalog(), observed_state=None)

    result = await HomeActionExecutor(client, confirmation_timeout=0.1).execute(
        HomeAction("light.entry", HomeOperation.TURN_ON)
    )

    assert result.accepted is True
    assert result.confirmed is False
    assert result.error is None
    assert client.waits == [("light.entry", frozenset({"on"}), 0.1)]
