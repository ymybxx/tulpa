"""Instance-scoped protocol; management remains protected by App authentication."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import anyio
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from fish_memory.background_messages.routes import _require_owned_session
from fish_memory.extensions.models import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    CreateInstance,
    ExtensionError,
)

extension_router = APIRouter(tags=["extensions"])


@extension_router.get("/internal/extensions/manager", include_in_schema=False)
async def managed_plugins(request: Request) -> dict[str, Any]:
    await request.app.state.extension_service.sync_session()
    return request.app.state.plugin_manager.listing()


@extension_router.post("/internal/extensions/manager/{action}", include_in_schema=False)
async def manage_plugin(action: str, body: dict[str, Any], request: Request) -> Any:
    manager = request.app.state.plugin_manager
    url = str(request.base_url).rstrip("/")
    try:
        if action == "load":
            return (
                await manager.load(
                    Path(body["path"]), development=bool(body.get("development"))
                )
            ).public()
        if action == "builtin":
            if manager.assets is not None:
                path = next((manager.assets / "plugins").glob("*.fish-plugin"), None)
                if path is None:
                    raise ValueError("直播插件安装包缺失")
                return (await manager.load(path, development=False)).public()
            path = (await asyncio.to_thread(Path(__file__).resolve)).parents[
                3
            ] / "plugins/live"
            return (await manager.load(path, development=True)).public()
        if action == "start":
            return (
                await manager.start(
                    UUID(body["id"]),
                    body.get("config", {}),
                    url,
                )
            ).public()
        if action == "stop":
            await manager.stop(UUID(body["id"]))
            return {"stopped": True}
        if action == "input":
            identifier = UUID(body["id"])
            instance = manager.service.instances.get(identifier)
            if identifier not in manager.items or instance is None:
                raise ValueError("请先启动插件")
            return await manager.service.inputs.send(
                instance, body["input"], manager.service.emit
            )
        if action in {"reviews", "review"}:
            identifier = UUID(body["id"])
            if identifier not in manager.items or manager.reviews is None:
                raise ValueError("插件运行评估不可用")
            if action == "reviews":
                return await manager.reviews.history(
                    identifier, cursor=body.get("cursor")
                )
            return manager.reviews.request(identifier)
        if action == "development":
            await manager.development(
                bool(body["enabled"]),
                url,
            )
            return {"enabled": bool(body["enabled"])}
        raise ValueError("Unknown plugin action")
    except (ExtensionError, ValueError, KeyError, OSError) as error:
        raise HTTPException(400, str(error)) from error


@extension_router.post("/internal/extensions/dev/{command}", include_in_schema=False)
async def development_command(
    command: str, body: dict[str, Any], request: Request
) -> Any:
    try:
        return await request.app.state.plugin_manager.dev_command(
            command, body, str(request.base_url).rstrip("/")
        )
    except (ValueError, KeyError, OSError) as error:
        raise HTTPException(400, str(error)) from error


@extension_router.post("/internal/extensions/instances", include_in_schema=False)
async def create_instance(body: CreateInstance, request: Request) -> dict[str, Any]:
    await _require_owned_session(request, body.fish_id, body.session_id)
    item, token = await request.app.state.extension_service.create(body)
    return {"instance_id": str(item.id), "token": token, "protocol": PROTOCOL_VERSION}


@extension_router.delete(
    "/internal/extensions/instances/{instance_id}", include_in_schema=False
)
async def stop_instance(instance_id: UUID, request: Request) -> dict[str, bool]:
    await request.app.state.extension_service.stop(instance_id)
    return {"stopped": True}


@extension_router.get(
    "/internal/extensions/instances/{instance_id}", include_in_schema=False
)
async def diagnostics(instance_id: UUID, request: Request) -> dict[str, Any]:
    try:
        return request.app.state.extension_service.diagnostics(instance_id)
    except ExtensionError as error:
        raise HTTPException(404, error.message) from error


@extension_router.post(
    "/internal/extensions/instances/{instance_id}/events", include_in_schema=False
)
async def inject_event(
    instance_id: UUID, body: dict[str, Any], request: Request
) -> Any:
    service = request.app.state.extension_service
    instance = service.instances.get(instance_id)
    if instance is None:
        raise HTTPException(404, "Instance not found")
    try:
        return await service.request(instance, "push", body)
    except (ExtensionError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@extension_router.websocket("/v1/extensions/ws")
async def plugin_socket(socket: WebSocket) -> None:
    service = socket.app.state.extension_service
    item = service.authenticate(list(socket.scope["headers"]))
    if item is None:
        await socket.close(4401)
        return
    if socket.query_params.get("protocol") != str(PROTOCOL_VERSION):
        await socket.close(4406, "Unsupported plugin protocol")
        return
    # Native SDKs do not send an Origin. Web pages cannot borrow a local SDK lease.
    if socket.headers.get("origin"):
        await socket.close(4403)
        return
    try:
        queue = service.attach(item)
    except ExtensionError as error:
        await socket.close(4409, error.code)
        return
    await socket.accept()
    send_lock = asyncio.Lock()
    reasoning_requests: set[asyncio.Task[None]] = set()

    async def send(value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
        if len(encoded.encode()) > MAX_FRAME_BYTES:
            encoded = json.dumps(
                {
                    "id": value.get("id"),
                    "error": {
                        "code": "response_too_large",
                        "message": "Request fewer context fields",
                    },
                }
            )
        async with send_lock:
            await socket.send_text(encoded)

    async def notifications() -> None:
        while True:
            value = await queue.get()
            await send(value)
            if value["method"] == "disconnect":
                await socket.close(1013, value["params"]["reason"])
                return

    async def reason(identifier: str, params: dict[str, Any]) -> None:
        try:
            result = await service.request(item, "reason", params)
            await send({"id": identifier, "result": result})
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                try:
                    await send(
                        {
                            "id": identifier,
                            "error": {
                                "code": "reasoning_cancelled",
                                "message": "Reasoning cancelled",
                            },
                        }
                    )
                except Exception:
                    pass
        except (ExtensionError, ValueError) as error:
            await send(
                {
                    "id": identifier,
                    "error": {
                        "code": error.code
                        if isinstance(error, ExtensionError)
                        else "invalid_request",
                        "message": error.message
                        if isinstance(error, ExtensionError)
                        else str(error)[:1000],
                    },
                }
            )
        except Exception:
            await send(
                {
                    "id": identifier,
                    "error": {
                        "code": "internal_error",
                        "message": "Reasoning failed; inspect App diagnostics",
                    },
                }
            )

    async def requests() -> None:
        while True:
            raw = await socket.receive_text()
            if len(raw.encode()) > MAX_FRAME_BYTES:
                await socket.close(1009)
                return
            identifier = None
            try:
                value = json.loads(raw, parse_constant=lambda v: _invalid_number(v))
                if not isinstance(value, dict):
                    raise ValueError("Request must be an object")
                identifier = value.get("id")
                if not isinstance(identifier, str) or not 0 < len(identifier) <= 64:
                    raise ValueError("Request needs a string id (1 to 64 characters)")
                if not isinstance(value.get("method"), str) or not isinstance(
                    value.get("params", {}), dict
                ):
                    raise ValueError("Invalid method or params")
                if value["method"] == "reason":
                    # A long inference must not block push, state, or cancellation
                    # messages on the same plugin connection.
                    if reasoning_requests:
                        raise ExtensionError(
                            "reasoning_busy", "Reasoning is already running"
                        )
                    task = asyncio.create_task(
                        reason(identifier, value.get("params", {}))
                    )
                    reasoning_requests.add(task)
                    task.add_done_callback(reasoning_requests.discard)
                    continue
                result = await service.request(
                    item, value["method"], value.get("params", {})
                )
                await send({"id": identifier, "result": result})
            except ExtensionError as error:
                await send(
                    {
                        "id": identifier,
                        "error": {"code": error.code, "message": error.message},
                    }
                )
            except (ValueError, ValidationError, KeyError) as error:
                await send(
                    {
                        "id": identifier,
                        "error": {
                            "code": "invalid_request",
                            "message": str(error)[:1000],
                        },
                    }
                )
            except Exception:
                # Do not expose DB connection strings, paths or model keys to plugins.
                await send(
                    {
                        "id": identifier,
                        "error": {
                            "code": "internal_error",
                            "message": "Request failed; inspect App diagnostics",
                        },
                    }
                )

    tasks = [asyncio.create_task(requests()), asyncio.create_task(notifications())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                task.result()
            except (WebSocketDisconnect, RuntimeError):
                pass
    except asyncio.CancelledError:
        pass
    finally:
        with anyio.CancelScope(shield=True):
            pending = [*tasks, *reasoning_requests]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await service.detach(item, queue)


def _invalid_number(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


@extension_router.get(
    "/internal/extensions/{fish_id}/expressions/{message_id}/events",
    include_in_schema=False,
)
async def expression_events(
    fish_id: UUID,
    message_id: UUID,
    request: Request,
    session_id: Annotated[UUID, Query(alias="sessionId")],
    offset: Annotated[int, Query(alias="afterOffset", ge=0)] = 0,
) -> StreamingResponse:
    await _require_owned_session(request, fish_id, session_id)
    stream = request.app.state.extension_service.streams.get(message_id)
    if stream is None or (stream.intent.fish_id, stream.intent.session_id) != (
        fish_id,
        session_id,
    ):
        raise HTTPException(404, "Expression not found")
    if offset > len(stream.text):
        raise HTTPException(409, "Expression cursor exceeds available text")

    async def events() -> AsyncIterator[str]:
        async for event in stream.events(offset):
            if await request.is_disconnected():
                return
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
