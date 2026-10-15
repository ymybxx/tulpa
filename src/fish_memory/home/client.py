from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol, cast

import websockets

from fish_memory.home.catalog import HomeEntityCatalog

logger = logging.getLogger(__name__)


class HomeAssistantError(RuntimeError):
    pass


class HomeAssistantUnavailableError(HomeAssistantError):
    pass


class HomeActionUncertainError(HomeAssistantError):
    pass


class HomeSocket(Protocol):
    async def send(self, value: str) -> None: ...

    async def recv(self) -> str: ...

    async def close(self) -> None: ...


Connect = Callable[[str], Awaitable[HomeSocket]]


async def _default_connect(url: str) -> HomeSocket:
    return cast(HomeSocket, await websockets.connect(url, ping_interval=20))


class HomeAssistantClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        connect: Connect = _default_connect,
        # A desktop may stay away from its Home Assistant host for hours. Keep
        # quick recovery for short outages, then back off so an unavailable
        # optional integration does not wake the app and network every 5s.
        reconnect_delays: Sequence[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    ) -> None:
        self._url = _websocket_url(base_url)
        self._token = token
        self._connect = connect
        self._reconnect_delays = tuple(reconnect_delays) or (1.0,)
        self._socket: HomeSocket | None = None
        self._runner: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._state_changed = asyncio.Event()
        self._catalog = HomeEntityCatalog.empty()
        self._states: list[Mapping[str, Any]] = []
        self._entity_registry: list[Mapping[str, Any]] = []
        self._device_registry: list[Mapping[str, Any]] = []
        self._area_registry: list[Mapping[str, Any]] = []
        self._services: Mapping[str, Mapping[str, Mapping[str, Any]]] = {}

    @property
    def catalog(self) -> HomeEntityCatalog:
        return self._catalog

    async def start(self) -> None:
        if self._runner is None:
            self._runner = asyncio.create_task(
                self._run(), name="home-assistant-client"
            )

    async def wait_ready(self) -> None:
        await self._ready.wait()

    async def close(self) -> None:
        self._closed = True
        self._ready.clear()
        if self._socket is not None:
            await self._socket.close()
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None
        self._fail_pending(HomeAssistantUnavailableError("client closed"))

    async def call_service(
        self,
        domain: str,
        service: str,
        *,
        service_data: Mapping[str, object] | None = None,
        target: Mapping[str, object] | None = None,
    ) -> Any:
        if not self._ready.is_set() or self._socket is None:
            raise HomeAssistantUnavailableError("Home Assistant is not ready")
        try:
            # Many state-changing HA services reject return_response=true.
            return await self._request(
                {
                    "type": "call_service",
                    "domain": domain,
                    "service": service,
                    "service_data": dict(service_data or {}),
                    "target": dict(target or {}),
                }
            )
        except HomeAssistantUnavailableError as error:
            raise HomeActionUncertainError(
                "Home Assistant disconnected after action dispatch"
            ) from error

    async def wait_for_state(
        self,
        entity_id: str,
        expected_states: frozenset[str],
        *,
        timeout_seconds: float,
    ) -> str | None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            # Capture the current event before reading the catalog so a state
            # update between the read and wait cannot be missed.
            changed = self._state_changed
            entity = self._catalog.get(entity_id)
            if entity is not None and entity.state in expected_states:
                return entity.state
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(changed.wait(), timeout=remaining)
            except TimeoutError:
                return None

    async def wait_for_cover_movement(
        self,
        entity_id: str,
        initial_position: float,
        target_position: int,
        *,
        timeout_seconds: float,
    ) -> str | None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while True:
            changed = self._state_changed
            entity = self._catalog.get(entity_id)
            if entity is not None:
                position = entity.attributes.get("current_position")
                if (
                    not isinstance(position, bool)
                    and isinstance(position, (int, float))
                    and _moved_toward_target(
                        float(position),
                        initial_position,
                        target_position,
                    )
                ):
                    return entity.state
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(changed.wait(), timeout=remaining)
            except TimeoutError:
                return None

    async def _run(self) -> None:
        attempt = 0
        while not self._closed:
            try:
                socket = await self._connect(self._url)
                self._socket = socket
                await self._authenticate(socket)
                receive_task = asyncio.create_task(self._receive_loop(socket))
                try:
                    await self._load_catalog()
                    await self._request(
                        {"type": "subscribe_events", "event_type": "state_changed"}
                    )
                    self._ready.set()
                    attempt = 0
                    await receive_task
                finally:
                    receive_task.cancel()
                    await asyncio.gather(receive_task, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._ready.clear()
                self._fail_pending(HomeAssistantUnavailableError(str(error)))
                if self._closed:
                    return
                delay = self._reconnect_delays[
                    min(attempt, len(self._reconnect_delays) - 1)
                ]
                attempt += 1
                logger.warning(
                    "Home Assistant connection lost; retrying in %.1fs", delay
                )
                await asyncio.sleep(delay)

    async def _authenticate(self, socket: HomeSocket) -> None:
        required = json.loads(await socket.recv())
        if required.get("type") != "auth_required":
            raise HomeAssistantError("unexpected Home Assistant auth handshake")
        await socket.send(json.dumps({"type": "auth", "access_token": self._token}))
        result = json.loads(await socket.recv())
        if result.get("type") != "auth_ok":
            raise HomeAssistantError("Home Assistant authentication failed")

    async def _load_catalog(self) -> None:
        states, entities, devices, areas, services = await asyncio.gather(
            self._request({"type": "get_states"}),
            self._request({"type": "config/entity_registry/list"}),
            self._request({"type": "config/device_registry/list"}),
            self._request({"type": "config/area_registry/list"}),
            self._request({"type": "get_services"}),
        )
        self._states = list(states or [])
        self._entity_registry = list(entities or [])
        self._device_registry = list(devices or [])
        self._area_registry = list(areas or [])
        self._services = services or {}
        self._rebuild_catalog()

    async def _request(self, payload: Mapping[str, object]) -> Any:
        socket = self._socket
        if socket is None:
            raise HomeAssistantUnavailableError("Home Assistant is disconnected")
        self._request_id += 1
        request_id = self._request_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await socket.send(json.dumps({"id": request_id, **payload}))
            return await future
        except HomeAssistantError:
            if not future.done():
                future.cancel()
            raise
        except Exception as error:
            if not future.done():
                future.cancel()
            raise HomeAssistantUnavailableError(str(error)) from error
        finally:
            self._pending.pop(request_id, None)

    async def _receive_loop(self, socket: HomeSocket) -> None:
        while not self._closed:
            message = json.loads(await socket.recv())
            if message.get("type") == "result":
                request_id = message.get("id")
                future = self._pending.get(request_id)
                if future is None or future.done():
                    continue
                if message.get("success"):
                    future.set_result(message.get("result"))
                else:
                    future.set_exception(
                        HomeAssistantError(
                            str(message.get("error") or "request failed")
                        )
                    )
            elif message.get("type") == "event":
                self._apply_event(message.get("event") or {})

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        if event.get("event_type") != "state_changed":
            return
        data = event.get("data") or {}
        entity_id = data.get("entity_id")
        new_state = data.get("new_state")
        if not entity_id:
            return
        self._states = [
            state for state in self._states if state.get("entity_id") != entity_id
        ]
        if isinstance(new_state, Mapping):
            self._states.append(new_state)
        self._rebuild_catalog()
        changed = self._state_changed
        self._state_changed = asyncio.Event()
        changed.set()

    def _rebuild_catalog(self) -> None:
        self._catalog = HomeEntityCatalog.from_payloads(
            states=self._states,
            entity_registry=self._entity_registry,
            device_registry=self._device_registry,
            area_registry=self._area_registry,
            services=self._services,
        )

    def _fail_pending(self, error: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)


def _websocket_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.startswith("https://"):
        value = "wss://" + value.removeprefix("https://")
    elif value.startswith("http://"):
        value = "ws://" + value.removeprefix("http://")
    return value + "/api/websocket"


def _moved_toward_target(
    position: float,
    initial_position: float,
    target_position: int,
) -> bool:
    if target_position > initial_position:
        return position > initial_position
    return position < initial_position
