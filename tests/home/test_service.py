import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.home.catalog import HomeEntityCatalog
from fish_memory.home.models import (
    HomeAction,
    HomeActionStatus,
    HomeOperation,
    HomeRouteDecision,
    HomeSetCommand,
    HomeSetTarget,
)
from fish_memory.home.service import HomeAutomationService
from fish_memory.models import Message, MessageStatus, Role

FISH_ID = UUID(int=1)
SESSION_ID = UUID(int=2)
MESSAGE_ID = UUID(int=3)
MESSAGE_SEQUENCE = 10


class FakeStore:
    def __init__(self) -> None:
        self.created = []
        self.updated = []

    async def create_run(self, **kwargs):
        self.created.append(kwargs)
        return type("Run", (), {"id": UUID(int=4)})()

    async def update_run(self, run_id, **kwargs):
        self.updated.append((run_id, kwargs))


class FakeHistoryStore:
    def __init__(self, messages: list[Message] | None = None) -> None:
        self.messages = messages or []
        self.calls = []

    async def load_history(
        self,
        fish_id,
        session_id,
        *,
        before_sequence,
        limit=None,
    ):
        self.calls.append((fish_id, session_id, before_sequence, limit))
        return self.messages


class FakeClient:
    def __init__(self, *, ready: bool = True) -> None:
        self.catalog = HomeEntityCatalog.from_payloads(
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
            services={"light": {"turn_on": {"fields": {}}}},
        )
        if not ready:
            self.catalog = HomeEntityCatalog.empty()
        self.calls = []

    async def call_service(self, domain, service, *, service_data=None, target=None):
        self.calls.append((domain, service, service_data, target))
        return {"ok": True}

    async def wait_for_state(self, entity_id, expected_states, *, timeout_seconds):
        del entity_id, timeout_seconds
        return next(iter(expected_states))

    async def wait_for_cover_movement(
        self,
        entity_id,
        initial_position,
        target_position,
        *,
        timeout_seconds,
    ):
        del entity_id, initial_position, target_position, timeout_seconds
        return "open"


class BlockingClient(FakeClient):
    async def call_service(self, domain, service, *, service_data=None, target=None):
        del domain, service, service_data, target
        await asyncio.Event().wait()


class ConcurrentClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.started: set[str] = set()
        self.both_started = asyncio.Event()

    async def call_service(self, domain, service, *, service_data=None, target=None):
        entity_id = target["entity_id"]
        self.started.add(entity_id)
        if len(self.started) == 2:
            self.both_started.set()
        await self.both_started.wait()
        return await super().call_service(
            domain,
            service,
            service_data=service_data,
            target=target,
        )


class UnconfirmedClient(FakeClient):
    async def wait_for_state(self, entity_id, expected_states, *, timeout_seconds):
        del entity_id, expected_states, timeout_seconds
        return None


class FailingRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        raise AssertionError("local fast path should skip LLM")


class CountedRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        assert text == "随便开两个灯"
        assert history == ()
        return HomeRouteDecision(
            commands=(
                HomeSetCommand(
                    target=HomeSetTarget(
                        selection="all",
                        area=None,
                        domain="light",
                        count=2,
                    ),
                    operation=HomeOperation.TURN_ON,
                ),
            ),
            source="llm",
        )


class AllLightsRouter:
    device_limit = 40

    def __init__(self) -> None:
        self.candidate_entity_ids = []

    async def route(self, text, candidates, *, history=(), tracker=None):
        assert text == "关闭所有灯"
        assert history == ()
        self.candidate_entity_ids = [
            candidate.entity.entity_id for candidate in candidates
        ]
        return HomeRouteDecision(
            commands=(
                HomeSetCommand(
                    target=HomeSetTarget(
                        selection="all",
                        area=None,
                        domain="light",
                        count=None,
                    ),
                    operation=HomeOperation.TURN_OFF,
                ),
            ),
            source="llm",
        )


class DimAllLightsRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        del candidates, history, tracker
        assert text == "把所有灯调暗一点"
        return HomeRouteDecision(
            commands=(
                HomeSetCommand(
                    target=HomeSetTarget(
                        selection="all",
                        area=None,
                        domain="light",
                        count=None,
                    ),
                    operation=HomeOperation.SET_BRIGHTNESS,
                    value="decrease",
                ),
            ),
            source="llm",
        )


class DimmestLivingRoomLightsRouter:
    device_limit = 40

    def __init__(self) -> None:
        self.candidate_entity_ids = []

    async def route(self, text, candidates, *, history=(), tracker=None):
        del history, tracker
        assert text == "把客厅灯调到最暗"
        self.candidate_entity_ids = [
            candidate.entity.entity_id for candidate in candidates
        ]
        return HomeRouteDecision(
            commands=(
                *(
                    HomeAction(
                        candidate.entity.entity_id,
                        HomeOperation.SET_BRIGHTNESS,
                        0,
                    )
                    for candidate in candidates
                ),
            ),
            source="llm",
        )


class LivingRoomLightsRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        del candidates, history, tracker
        assert text == "关闭客厅所有灯"
        return HomeRouteDecision(
            commands=(
                HomeSetCommand(
                    target=HomeSetTarget(
                        selection="all",
                        area="客厅",
                        domain="light",
                        count=None,
                    ),
                    operation=HomeOperation.TURN_OFF,
                ),
            ),
            source="llm",
        )


class AllDevicesRouter:
    device_limit = 40

    def __init__(self) -> None:
        self.candidate_entity_ids = []

    async def route(self, text, candidates, *, history=(), tracker=None):
        del history, tracker
        assert text == "关闭所有设备"
        self.candidate_entity_ids = [
            candidate.entity.entity_id for candidate in candidates
        ]
        return HomeRouteDecision(
            commands=(
                HomeSetCommand(
                    target=HomeSetTarget(
                        selection="all",
                        area=None,
                        domain=None,
                        count=None,
                    ),
                    operation=HomeOperation.TURN_OFF,
                ),
            ),
            source="llm",
        )


class MultiActionRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        del text, candidates, history
        return HomeRouteDecision(
            commands=(
                HomeAction("light.entry", HomeOperation.TURN_ON),
                HomeAction("light.kitchen", HomeOperation.TURN_ON),
            ),
            source="llm",
        )


class PartialSpeakerRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        del candidates, history, tracker
        assert text == "关闭小爱音箱"
        return HomeRouteDecision(
            commands=(
                HomeAction(
                    "media_player.bedroom_speaker",
                    HomeOperation.STOP,
                ),
            ),
            skipped_commands=(
                HomeAction(
                    "media_player.living_room_speaker",
                    HomeOperation.STOP,
                ),
            ),
            source="llm",
        )


class ClimateSetupRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        del text, candidates, history, tracker
        return HomeRouteDecision(
            commands=(
                HomeAction("climate.living_room", HomeOperation.TURN_ON),
                HomeAction(
                    "climate.living_room",
                    HomeOperation.SET_TEMPERATURE,
                    24,
                ),
            ),
            source="llm",
        )


class ContextRouter:
    device_limit = 40

    def __init__(self) -> None:
        self.calls = []

    async def route(self, text, candidates, *, history=(), tracker=None):
        self.calls.append((text, candidates, history))
        return HomeRouteDecision(
            commands=(HomeAction("light.bedroom_2", HomeOperation.TURN_ON),),
            source="llm",
        )


class ProbeRouter:
    device_limit = 40

    def __init__(self) -> None:
        self.calls = []

    async def route(self, text, candidates, *, history=(), tracker=None):
        self.calls.append((text, candidates, history))
        return HomeRouteDecision(commands=(), source="llm")


class TwoDeviceProbeRouter(ProbeRouter):
    device_limit = 2


class RejectingRouter:
    device_limit = 40

    async def route(self, text, candidates, *, history=(), tracker=None):
        del text, candidates, history, tracker
        raise ValueError("invalid router output")


@pytest.mark.asyncio
async def test_service_executes_fast_path_in_background() -> None:
    store = FakeStore()
    history_store = FakeHistoryStore()
    client = FakeClient()
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=history_store,
        router=FailingRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "打开玄关灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert client.calls[0][0:2] == ("light", "turn_on")
    assert store.updated[-1][1]["status"] is HomeActionStatus.SUCCEEDED
    assert history_store.calls == []


@pytest.mark.asyncio
async def test_service_skips_non_home_message_without_persistence() -> None:
    store = FakeStore()
    service = HomeAutomationService(
        client=FakeClient(),
        store=store,
        history_store=FakeHistoryStore(),
        router=FailingRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "今天天气好吗", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert store.created == []


@pytest.mark.asyncio
async def test_service_routes_short_adjust_command() -> None:
    router = ProbeRouter()
    history_store = FakeHistoryStore(
        [
            Message(
                id=UUID(int=13),
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=9,
                role=Role.USER,
                content="把玄关灯调亮一点",
                status=MessageStatus.COMPLETED,
                created_at=datetime(2026, 8, 13, tzinfo=UTC),
            )
        ]
    )
    service = HomeAutomationService(
        client=FakeClient(),
        store=FakeStore(),
        history_store=history_store,
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "再调一下", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert len(router.calls) == 1
    assert router.calls[0][0] == "再调一下"


@pytest.mark.asyncio
async def test_service_persists_failure_when_catalog_is_not_ready() -> None:
    store = FakeStore()
    service = HomeAutomationService(
        client=FakeClient(ready=False),
        store=store,
        history_store=FakeHistoryStore(),
        router=FailingRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "打开玄关灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert len(store.created) == 1
    assert store.updated[-1][1]["status"] is HomeActionStatus.FAILED
    assert "未连接" in store.updated[-1][1]["error"]


@pytest.mark.asyncio
async def test_service_treats_rejected_router_output_as_ambiguous() -> None:
    store = FakeStore()
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.bedroom_left",
                "state": "on",
                "attributes": {"friendly_name": "左筒灯"},
            },
            {
                "entity_id": "light.bedroom_right",
                "state": "on",
                "attributes": {"friendly_name": "右筒灯"},
            },
        ],
        entity_registry=[
            {"entity_id": "light.bedroom_left", "area_id": "bedroom"},
            {"entity_id": "light.bedroom_right", "area_id": "bedroom"},
        ],
        device_registry=[],
        area_registry=[{"area_id": "bedroom", "name": "主卧"}],
        services={"light": {"turn_off": {"fields": {}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=RejectingRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭主卧灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert store.updated[-1][1] == {
        "status": HomeActionStatus.AMBIGUOUS,
        "error": "无法唯一确定设备或操作",
    }
    assert client.calls == []


@pytest.mark.asyncio
async def test_service_executes_every_routed_action() -> None:
    store = FakeStore()
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.entry",
                "state": "off",
                "attributes": {"friendly_name": "玄关灯"},
            },
            {
                "entity_id": "light.kitchen",
                "state": "off",
                "attributes": {"friendly_name": "厨房灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=MultiActionRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "打开玄关灯和厨房灯",
        MESSAGE_SEQUENCE,
    )
    await service.drain()

    assert [call[0:2] for call in client.calls] == [
        ("light", "turn_on"),
        ("light", "turn_on"),
    ]
    assert store.updated[-1][1]["status"] is HomeActionStatus.SUCCEEDED
    assert len(store.updated[-1][1]["action"]["actions"]) == 2


@pytest.mark.asyncio
async def test_service_marks_supported_subset_as_partial_success() -> None:
    store = FakeStore()
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
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
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=PartialSpeakerRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "关闭小爱音箱",
        MESSAGE_SEQUENCE,
    )
    await service.drain()

    update = store.updated[-1][1]
    assert client.calls == [
        (
            "media_player",
            "media_stop",
            {},
            {"entity_id": "media_player.bedroom_speaker"},
        )
    ]
    assert update["status"] is HomeActionStatus.PARTIAL
    assert update["accepted"] is True
    assert update["result"]["skipped"] == [
        {
            "entityId": "media_player.living_room_speaker",
            "operation": "stop",
            "value": None,
            "reason": "operation_not_supported",
        }
    ]


@pytest.mark.asyncio
async def test_service_executes_ordered_operations_for_one_climate_entity() -> None:
    store = FakeStore()
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
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
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=ClimateSetupRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "请处理客厅空调",
        MESSAGE_SEQUENCE,
    )
    await service.drain()

    assert client.calls == [
        ("climate", "turn_on", {}, {"entity_id": "climate.living_room"}),
        (
            "climate",
            "set_temperature",
            {"temperature": 24.0},
            {"entity_id": "climate.living_room"},
        ),
    ]
    assert store.updated[-1][1]["status"] is HomeActionStatus.SUCCEEDED
    assert [item["operation"] for item in store.updated[-1][1]["result"]["items"]] == [
        "turn_on",
        "set_temperature",
    ]


@pytest.mark.asyncio
async def test_service_times_out_a_stalled_home_assistant_write() -> None:
    store = FakeStore()
    service = HomeAutomationService(
        client=BlockingClient(),
        store=store,
        history_store=FakeHistoryStore(),
        router=FailingRouter(),
        performance_store=None,
        timeout_seconds=0.01,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "打开玄关灯", MESSAGE_SEQUENCE
    )
    await asyncio.wait_for(service.drain(), timeout=0.2)

    assert store.updated[-1][1]["status"] is HomeActionStatus.FAILED
    assert "超时" in store.updated[-1][1]["error"]


@pytest.mark.asyncio
async def test_service_marks_accepted_but_unconfirmed_action_as_succeeded() -> None:
    store = FakeStore()
    service = HomeAutomationService(
        client=UnconfirmedClient(),
        store=store,
        history_store=FakeHistoryStore(),
        router=FailingRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "打开玄关灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    update = store.updated[-1][1]
    assert update["status"] is HomeActionStatus.SUCCEEDED
    assert update["accepted"] is True
    assert update["confirmed"] is False
    assert update["error"] is None


@pytest.mark.asyncio
async def test_service_marks_unavailable_device_as_failed() -> None:
    store = FakeStore()
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
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
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=FailingRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭玄关灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert store.updated[-1][1]["status"] is HomeActionStatus.FAILED
    assert store.updated[-1][1]["accepted"] is False
    assert "不可用" in store.updated[-1][1]["error"]


@pytest.mark.asyncio
async def test_bulk_command_only_executes_available_devices_needing_change() -> None:
    store = FakeStore()
    client = FakeClient()
    router = AllLightsRouter()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.offline",
                "state": "unavailable",
                "attributes": {"friendly_name": "离线灯"},
            },
            {
                "entity_id": "light.online",
                "state": "on",
                "attributes": {"friendly_name": "在线灯"},
            },
            {
                "entity_id": "light.already_off",
                "state": "off",
                "attributes": {"friendly_name": "已关闭灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_off": {"fields": {}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭所有灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert router.candidate_entity_ids == ["light.online"]
    assert client.calls == [("light", "turn_off", {}, {"entity_id": "light.online"})]
    assert store.updated[-1][1]["status"] is HomeActionStatus.SUCCEEDED
    assert store.updated[-1][1]["action"] == {
        "entityId": "light.online",
        "operation": "turn_off",
        "value": None,
    }
    assert store.updated[-1][1]["confirmed"] is True
    assert store.updated[-1][1]["error"] is None


@pytest.mark.asyncio
async def test_set_relative_brightness_resolves_for_each_light() -> None:
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.first",
                "state": "on",
                "attributes": {"friendly_name": "灯一", "brightness": 128},
            },
            {
                "entity_id": "light.second",
                "state": "on",
                "attributes": {"friendly_name": "灯二", "brightness": 204},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {"brightness_pct": {}}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=FakeHistoryStore(),
        router=DimAllLightsRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "把所有灯调暗一点",
        MESSAGE_SEQUENCE,
    )
    await service.drain()

    assert [call[2] for call in client.calls] == [
        {"brightness_pct": 40},
        {"brightness_pct": 70},
    ]


@pytest.mark.asyncio
async def test_brightness_execution_skips_lights_that_are_off() -> None:
    client = FakeClient()
    router = DimmestLivingRoomLightsRouter()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.living_room_off_1",
                "state": "off",
                "attributes": {"friendly_name": "客厅灯一", "brightness": 128},
            },
            {
                "entity_id": "light.living_room_off_2",
                "state": "off",
                "attributes": {"friendly_name": "客厅灯二", "brightness": 128},
            },
            {
                "entity_id": "light.living_room_on",
                "state": "on",
                "attributes": {"friendly_name": "客厅灯三", "brightness": 128},
            },
        ],
        entity_registry=[
            {"entity_id": entity_id, "area_id": "living_room"}
            for entity_id in (
                "light.living_room_off_1",
                "light.living_room_off_2",
                "light.living_room_on",
            )
        ],
        device_registry=[],
        area_registry=[{"area_id": "living_room", "name": "客厅"}],
        services={"light": {"turn_on": {"fields": {"brightness_pct": {}}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=FakeHistoryStore(),
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "把客厅灯调到最暗",
        MESSAGE_SEQUENCE,
    )
    await service.drain()

    assert router.candidate_entity_ids == [
        "light.living_room_off_1",
        "light.living_room_off_2",
        "light.living_room_on",
    ]
    assert client.calls == [
        (
            "light",
            "turn_on",
            {"brightness_pct": 0},
            {"entity_id": "light.living_room_on"},
        )
    ]


@pytest.mark.asyncio
async def test_different_entities_execute_concurrently() -> None:
    client = ConcurrentClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.entry",
                "state": "off",
                "attributes": {"friendly_name": "玄关灯"},
            },
            {
                "entity_id": "light.kitchen",
                "state": "off",
                "attributes": {"friendly_name": "厨房灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=FakeHistoryStore(),
        router=MultiActionRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        "打开玄关灯和厨房灯",
        MESSAGE_SEQUENCE,
    )
    await asyncio.wait_for(service.drain(), timeout=1.0)

    assert client.started == {"light.entry", "light.kitchen"}


@pytest.mark.asyncio
async def test_actionable_filter_runs_before_router_device_limit() -> None:
    store = FakeStore()
    client = FakeClient()
    router = AllLightsRouter()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            *[
                {
                    "entity_id": f"light.off_{index:02d}",
                    "state": "off",
                    "attributes": {"friendly_name": f"已关闭灯{index}"},
                }
                for index in range(45)
            ],
            {
                "entity_id": "light.zzz_online",
                "state": "on",
                "attributes": {"friendly_name": "开启灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_off": {"fields": {}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭所有灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert router.candidate_entity_ids == ["light.zzz_online"]


@pytest.mark.asyncio
async def test_router_device_limit_does_not_drop_same_device_endpoints() -> None:
    action_count = 45
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            *[
                {
                    "entity_id": f"button.action_{index:02d}",
                    "state": "unknown",
                    "attributes": {"friendly_name": f"执行测试设备动作{index}"},
                }
                for index in range(action_count)
            ],
            {
                "entity_id": "light.target",
                "state": "off",
                "attributes": {"friendly_name": "执行测试设备"},
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
    router = TwoDeviceProbeRouter()
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=FakeHistoryStore(),
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "执行测试设备", MESSAGE_SEQUENCE
    )
    await service.drain()

    _, candidates, _ = router.calls[0]
    assert len(candidates) == action_count + 1
    assert {candidate.entity.device_id for candidate in candidates} == {
        "device-actions",
        "device-target",
    }


@pytest.mark.asyncio
async def test_set_expansion_is_not_limited_by_router_device_limit() -> None:
    store = FakeStore()
    client = FakeClient()
    router = AllLightsRouter()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": f"light.test_{index:02d}",
                "state": "on",
                "attributes": {"friendly_name": f"测试灯{index}"},
            }
            for index in range(45)
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_off": {"fields": {}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭所有灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert len(router.candidate_entity_ids) == 40
    assert len(client.calls) == 45


@pytest.mark.asyncio
async def test_set_expansion_applies_explicit_area_and_domain_filters() -> None:
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "客厅灯"},
            },
            {
                "entity_id": "light.bedroom",
                "state": "on",
                "attributes": {"friendly_name": "卧室灯"},
            },
            {
                "entity_id": "switch.living_room",
                "state": "on",
                "attributes": {"friendly_name": "客厅插座"},
            },
        ],
        entity_registry=[
            {"entity_id": "light.living_room", "area_id": "living"},
            {"entity_id": "light.bedroom", "area_id": "bedroom"},
            {"entity_id": "switch.living_room", "area_id": "living"},
        ],
        device_registry=[],
        area_registry=[
            {"area_id": "living", "name": "客厅"},
            {"area_id": "bedroom", "name": "卧室"},
        ],
        services={
            "light": {"turn_off": {"fields": {}}},
            "switch": {"turn_off": {"fields": {}}},
        },
    )
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=FakeHistoryStore(),
        router=LivingRoomLightsRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭客厅所有灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert client.calls == [
        ("light", "turn_off", {}, {"entity_id": "light.living_room"})
    ]


@pytest.mark.asyncio
async def test_all_devices_set_uses_complete_catalog_instead_of_history_match() -> None:
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.living_room",
                "state": "on",
                "attributes": {"friendly_name": "客厅灯"},
            },
            {
                "entity_id": "switch.bedroom",
                "state": "on",
                "attributes": {"friendly_name": "卧室插座"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "light": {"turn_off": {"fields": {}}},
            "switch": {"turn_off": {"fields": {}}},
        },
    )
    history_store = FakeHistoryStore(
        [
            Message(
                id=UUID(int=31),
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=9,
                role=Role.USER,
                content="客厅灯",
                status=MessageStatus.COMPLETED,
                created_at=datetime(2026, 8, 13, tzinfo=UTC),
            )
        ]
    )
    router = AllDevicesRouter()
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=history_store,
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭所有设备", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert router.candidate_entity_ids == [
        "light.living_room",
        "switch.bedroom",
    ]
    assert [call[0] for call in client.calls] == ["light", "switch"]


@pytest.mark.asyncio
async def test_counted_command_is_not_silently_filtered() -> None:
    store = FakeStore()
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": f"light.test_{index}",
                "state": "off",
                "attributes": {"friendly_name": f"测试灯{index}"},
            }
            for index in range(3)
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {}}}},
    )
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=FakeHistoryStore(),
        router=CountedRouter(),
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "随便开两个灯", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert len(client.calls) == 2
    assert store.updated[-1][1]["status"] is HomeActionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_follow_up_command_uses_recent_same_session_history() -> None:
    store = FakeStore()
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "light.bedroom_1",
                "state": "on",
                "attributes": {"friendly_name": "卧室左灯"},
            },
            {
                "entity_id": "light.bedroom_2",
                "state": "off",
                "attributes": {"friendly_name": "卧室右灯"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={"light": {"turn_on": {"fields": {}}}},
    )
    history_store = FakeHistoryStore(
        [
            Message(
                id=UUID(int=11),
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=8,
                role=Role.USER,
                content="打开一个卧室灯",
                status=MessageStatus.COMPLETED,
                created_at=datetime(2026, 8, 13, tzinfo=UTC),
            ),
            Message(
                id=UUID(int=12),
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=9,
                role=Role.ASSISTANT,
                content="已经打开一个卧室灯。",
                status=MessageStatus.COMPLETED,
                created_at=datetime(2026, 8, 13, tzinfo=UTC),
            ),
        ]
    )
    router = ContextRouter()
    service = HomeAutomationService(
        client=client,
        store=store,
        history_store=history_store,
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "再打开一个", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert history_store.calls == [(FISH_ID, SESSION_ID, MESSAGE_SEQUENCE, 6)]
    text, candidates, history = router.calls[0]
    assert text == "再打开一个"
    assert {candidate.entity.entity_id for candidate in candidates} == {
        "light.bedroom_2"
    }
    assert history == (
        {"role": "user", "content": "打开一个卧室灯"},
        {"role": "assistant", "content": "已经打开一个卧室灯。"},
    )
    assert client.calls == [("light", "turn_on", {}, {"entity_id": "light.bedroom_2"})]


@pytest.mark.asyncio
async def test_complete_command_uses_current_text_and_only_actionable_domain() -> None:
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "climate.bedroom",
                "state": "cool",
                "attributes": {"friendly_name": "卧室空调"},
            },
            {
                "entity_id": "climate.living_room",
                "state": "off",
                "attributes": {"friendly_name": "客厅空调"},
            },
            {
                "entity_id": "button.ac_temperature_up",
                "state": "unknown",
                "attributes": {"friendly_name": "空调调高一度"},
            },
            {
                "entity_id": "media_player.speaker",
                "state": "playing",
                "attributes": {"friendly_name": "客厅音箱"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "climate": {
                "turn_on": {"fields": {}},
                "turn_off": {"fields": {}},
            },
            "button": {"press": {"fields": {}}},
            "media_player": {"media_stop": {"fields": {}}},
        },
    )
    history_store = FakeHistoryStore(
        [
            Message(
                id=UUID(int=21),
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=8,
                role=Role.USER,
                content="打开电视灯带",
                status=MessageStatus.COMPLETED,
                created_at=datetime(2026, 8, 13, tzinfo=UTC),
            ),
            Message(
                id=UUID(int=22),
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=9,
                role=Role.ASSISTANT,
                content="正在打开电视灯带。",
                status=MessageStatus.COMPLETED,
                created_at=datetime(2026, 8, 13, tzinfo=UTC),
            ),
        ]
    )
    router = ProbeRouter()
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=history_store,
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭空调", MESSAGE_SEQUENCE
    )
    await service.drain()

    _, candidates, history = router.calls[0]
    assert [candidate.entity.entity_id for candidate in candidates] == [
        "climate.bedroom"
    ]
    assert history == (
        {"role": "user", "content": "打开电视灯带"},
        {"role": "assistant", "content": "正在打开电视灯带。"},
    )


@pytest.mark.asyncio
async def test_explicit_operation_falls_back_to_compatible_nonstandard_domain() -> None:
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
        states=[
            {
                "entity_id": "switch.ac_power",
                "state": "on",
                "attributes": {"friendly_name": "空调电源"},
            },
            {
                "entity_id": "button.ac_toggle",
                "state": "unknown",
                "attributes": {"friendly_name": "空调开关状态切换"},
            },
        ],
        entity_registry=[],
        device_registry=[],
        area_registry=[],
        services={
            "switch": {"turn_off": {"fields": {}}},
            "button": {"press": {"fields": {}}},
        },
    )
    router = ProbeRouter()
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=FakeHistoryStore(),
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "关闭空调", MESSAGE_SEQUENCE
    )
    await service.drain()

    assert router.calls == []
    assert client.calls == [
        ("switch", "turn_off", {}, {"entity_id": "switch.ac_power"})
    ]


@pytest.mark.asyncio
async def test_standard_domain_on_another_device_does_not_hide_power_button() -> None:
    client = FakeClient()
    client.catalog = HomeEntityCatalog.from_payloads(
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
            {
                "entity_id": "media_player.speaker",
                "state": "off",
                "attributes": {
                    "friendly_name": "客厅音箱",
                    "supported_features": 128,
                },
            },
            {
                "entity_id": "media_player.speaker_2",
                "state": "off",
                "attributes": {
                    "friendly_name": "卧室音箱",
                    "supported_features": 128,
                },
            },
        ],
        entity_registry=[
            {"entity_id": "media_player.tv", "device_id": "device-tv"},
            {"entity_id": "button.tv_turn_on", "device_id": "device-tv"},
            {"entity_id": "media_player.speaker", "device_id": "device-speaker"},
            {
                "entity_id": "media_player.speaker_2",
                "device_id": "device-speaker-2",
            },
        ],
        device_registry=[
            {"id": "device-tv", "name": "客厅电视"},
            {"id": "device-speaker", "name": "客厅音箱"},
            {"id": "device-speaker-2", "name": "卧室音箱"},
        ],
        area_registry=[],
        services={
            "button": {"press": {"fields": {}}},
            "media_player": {
                "turn_on": {"fields": {}},
                "turn_off": {"fields": {}},
            },
        },
    )
    router = ProbeRouter()
    service = HomeAutomationService(
        client=client,
        store=FakeStore(),
        history_store=FakeHistoryStore(),
        router=router,
        performance_store=None,
    )

    service.notify_user_message(
        FISH_ID, SESSION_ID, MESSAGE_ID, "打开电视机", MESSAGE_SEQUENCE
    )
    await service.drain()

    _, candidates, _ = router.calls[0]
    assert "button.tv_turn_on" in {
        candidate.entity.entity_id for candidate in candidates
    }
