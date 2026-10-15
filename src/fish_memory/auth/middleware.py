from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from fish_memory.auth.service import ApiAuthService

AsgiMessage = dict[str, Any]
AsgiReceive = Callable[[], Awaitable[AsgiMessage]]
AsgiSend = Callable[[AsgiMessage], Awaitable[None]]
AsgiApp = Callable[[dict[str, Any], AsgiReceive, AsgiSend], Awaitable[None]]

PUBLIC_EXACT_PATHS = {
    "/health",
    "/ready",
    "/v1/auth/status",
    "/v1/auth/unlock",
}


class ApiAuthMiddleware:
    def __init__(
        self,
        app: AsgiApp,
        *,
        auth_service: ApiAuthService,
        scoped_auth: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self.app = app
        self.auth_service = auth_service
        self.scoped_auth = scoped_auth

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: AsgiReceive,
        send: AsgiSend,
    ) -> None:
        scope_type = scope.get("type")
        if scope_type not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path", ""))
        if _is_public(path):
            await self.app(scope, receive, send)
            return
        headers = list(scope.get("headers", []))
        if self.scoped_auth is not None and self.scoped_auth(scope):
            await self.app(scope, receive, send)
            return
        if self.auth_service.authenticate_headers(headers) is not None:
            await self.app(scope, receive, send)
            return
        if scope_type == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        await _send_unauthorized(send)


def _is_public(path: str) -> bool:
    return path in PUBLIC_EXACT_PATHS


async def _send_unauthorized(send: AsgiSend) -> None:
    body = json.dumps(
        {
            "error": {
                "code": "unauthorized",
                "message": "authentication required",
            }
        },
        separators=(",", ":"),
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
