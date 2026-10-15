from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .tasks import Tasks

if TYPE_CHECKING:
    from .plugin import Event

logger = logging.getLogger(__name__)
MAX_FRAME_BYTES = 65_536
Result = TypeVar("Result", bound=BaseModel)


class PluginError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class Pet:
    """Scoped connection. Receipt confirms storage, not speech completion."""

    def __init__(
        self,
        url: str,
        token: str,
        definitions: list[dict[str, Any]],
        *,
        activity: dict[str, Any] | None = None,
        inputs: list[dict[str, Any]] | None = None,
    ) -> None:
        self._url, self._token, self._definitions = url, token, definitions
        self._activity = activity
        self._inputs = inputs or []
        self._input_requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=8)
        self._socket: Any = None
        self._ready = asyncio.Event()
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._notifications: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)
        self._slots = asyncio.Semaphore(16)
        self._task: asyncio.Task[None] | None = None
        self._state: dict[str, Any] = {"connected": False, "can_respond": False}
        self._closed = False
        self._fatal: PluginError | None = None
        self._send_lock = asyncio.Lock()
        self.capabilities: dict[str, Any] = {}
        self.tasks = Tasks(self)

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    async def __aenter__(self) -> Pet:
        self._task = asyncio.create_task(self._connect(), name="fish-plugin-socket")
        try:
            await self.wait_ready()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def wait_ready(self, wait_seconds: float = 10) -> None:
        async with asyncio.timeout(wait_seconds):
            await self._ready.wait()
        if self._fatal is not None:
            raise self._fatal
        if self._closed:
            raise PluginError("closed", "Plugin is stopped")

    async def _connect(self) -> None:
        delay = 0.2
        while not self._closed:
            try:
                async with connect(
                    self._url,
                    additional_headers={"Authorization": f"Bearer {self._token}"},
                    compression=None,
                    max_size=MAX_FRAME_BYTES,
                    max_queue=16,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=5,
                    close_timeout=2,
                ) as socket:
                    self._socket = socket
                    reader = asyncio.create_task(self._read(socket))
                    try:
                        registration = await self._request(
                            "register",
                            {
                                "events": self._definitions,
                                "activity": self._activity,
                                "inputs": self._inputs,
                            },
                            wait_ready=False,
                        )
                        self.capabilities = registration.get("capabilities", {})
                        self._ready.set()
                        delay = 0.2
                        await reader
                    finally:
                        reader.cancel()
                        await asyncio.gather(reader, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except PluginError as error:
                self._fatal = error
                return
            except Exception as error:
                code = getattr(getattr(error, "rcvd", None), "code", None)
                status = getattr(getattr(error, "response", None), "status_code", None)
                if code in {4401, 4403, 4406, 4409} or status in {401, 403}:
                    self._fatal = PluginError(
                        "connection_rejected",
                        "App rejected this plugin lease; start it again in the App",
                    )
                    return
                logger.warning(
                    "Fish connection lost (%s); reconnecting", type(error).__name__
                )
            finally:
                self._socket = None
                self._ready.clear()
                self._state = {"connected": False, "can_respond": False}
                for future in tuple(self._pending.values()):
                    if not future.done():
                        future.set_exception(
                            PluginError(
                                "disconnected",
                                "Connection lost; query result with event_id",
                            )
                        )
                if self._fatal is not None:
                    self._ready.set()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5)

    async def _read(self, socket: Any) -> None:
        async for raw in socket:
            value = json.loads(raw)
            if "id" in value:
                future = self._pending.get(value["id"])
                if future is not None and not future.done():
                    if "error" in value:
                        future.set_exception(PluginError(**value["error"]))
                    else:
                        future.set_result(value.get("result"))
            elif "method" in value:
                if value["method"] == "input":
                    # Keep manual inputs separate from lossy state notifications.
                    if self._input_requests.full():
                        raise PluginError("input_overflow", "Too many pending inputs")
                    self._input_requests.put_nowait(value["params"])
                    continue
                if value["method"] == "capabilities":
                    self.capabilities = dict(value["params"])
                if value["method"] == "state":
                    self._state = value["params"]
                if (
                    value["method"] == "disconnect"
                    and value["params"].get("reason") == "stopped"
                ):
                    raise PluginError("stopped", "Stopped by the App")
                if self._notifications.full():
                    self._notifications.get_nowait()
                self._notifications.put_nowait(value)

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        wait_ready: bool = True,
        response_wait_seconds: float = 10,
    ) -> Any:
        if wait_ready:
            await self.wait_ready()
        async with self._slots:
            identifier = uuid4().hex
            payload = json.dumps(
                {"id": identifier, "method": method, "params": params},
                ensure_ascii=False,
                allow_nan=False,
            )
            if len(payload.encode()) > MAX_FRAME_BYTES:
                raise PluginError(
                    "frame_too_large",
                    "Split or select input before pushing (64 KiB limit)",
                )
            future = asyncio.get_running_loop().create_future()
            self._pending[identifier] = future
            try:
                async with self._send_lock:
                    if self._socket is None:
                        raise PluginError("disconnected", "App connection unavailable")
                    await self._socket.send(payload)
                async with asyncio.timeout(response_wait_seconds):
                    return await future
            except ConnectionClosed as error:
                raise PluginError("disconnected", "App connection closed") from error
            finally:
                self._pending.pop(identifier, None)
                if not future.done():
                    future.cancel()

    async def push(
        self,
        event: Event[Any],
        data: BaseModel | dict[str, Any],
        *,
        event_id: UUID | str | None = None,
    ) -> dict[str, Any]:
        validated = event.data.model_validate(data)
        payload = {
            "event_id": str(event_id or uuid4()),
            "type": event.type,
            "data": validated.model_dump(mode="json"),
        }
        try:
            return await self._request("push", payload)
        except (PluginError, TimeoutError) as error:
            if isinstance(error, PluginError) and error.code != "disconnected":
                raise
            # One bounded retry with the same ID after an uncertain ACK.
            await self.wait_ready()
            return await self._request("push", payload)

    async def context(self, *fields: str) -> dict[str, Any]:
        return await self._request(
            "context", {"fields": list(fields)} if fields else {}
        )

    async def reason(
        self,
        prompt: str,
        result: type[Result],
        *,
        data: dict[str, Any] | None = None,
        fields: tuple[str, ...] = ("personality", "history", "memory", "perception"),
        input_text: str | None = None,
    ) -> Result:
        """Ask the configured reasoning model; return validated data, never speech.

        One in-flight call per instance. No automatic retry or model keys in the
        plugin. The App records a scoped trace and cancellation stops inference.
        """
        if input_text is not None and data:
            raise ValueError("Use input_text or data, not both")
        request_id = str(uuid4())
        try:
            response = await self._request(
                "reason",
                {
                    "prompt": prompt,
                    "schema": result.model_json_schema(),
                    "data": data or {},
                    **({"input_text": input_text} if input_text is not None else {}),
                    "fields": list(fields),
                    "request_id": request_id,
                },
                response_wait_seconds=50,
            )
            return result.model_validate(response["value"])
        except asyncio.CancelledError:
            with suppress(Exception):
                async with asyncio.timeout(2):
                    await self._request("cancel_reason", {"request_id": request_id})
            raise

    async def result(self, event_id: UUID | str) -> dict[str, Any]:
        return await self._request("result", {"event_id": str(event_id)})

    async def refresh_state(self) -> dict[str, Any]:
        self._state = await self._request("state", {})
        return self.state

    async def notifications(self) -> AsyncIterator[dict[str, Any]]:
        """Live hints; query result(event_id) to recover missed updates."""
        while not self._closed:
            yield await self._notifications.get()

    async def input_requests(self) -> AsyncIterator[dict[str, Any]]:
        while not self._closed:
            yield await self._input_requests.get()

    async def input_result(
        self, request_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._request(
            "input_result", {"request_id": request_id, "result": result}
        )

    async def close(self) -> None:
        self._closed = True
        self._ready.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def wait_closed(self) -> None:
        if self._task is not None:
            await asyncio.shield(self._task)
        if self._fatal is not None:
            raise self._fatal
