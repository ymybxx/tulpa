from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.service import ApiAuthService


def build_app(tmp_path: Path) -> FastAPI:
    credentials = ApiKeyCredentialStore(
        api_key="fm_test_secret",
        key_file=tmp_path / "api-key",
    )
    auth = ApiAuthService(credentials)
    app = FastAPI()
    app.state.auth_service = auth
    app.add_middleware(ApiAuthMiddleware, auth_service=auth)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/private")
    async def private() -> dict[str, str]:
        return {"value": "private"}

    @app.websocket("/v1/socket")
    async def socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("ready")
        await websocket.close()

    return app


def test_health_is_public_but_private_api_requires_bearer(tmp_path: Path) -> None:
    with TestClient(build_app(tmp_path)) as client:
        health = client.get("/health")
        unauthorized = client.get("/v1/private")
        authorized = client.get(
            "/v1/private",
            headers={"Authorization": "Bearer fm_test_secret"},
        )

    assert health.status_code == 200
    assert unauthorized.status_code == 401
    assert unauthorized.json() == {
        "error": {"code": "unauthorized", "message": "authentication required"}
    }
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert authorized.status_code == 200


def test_websocket_requires_bearer_or_browser_session(tmp_path: Path) -> None:
    app = build_app(tmp_path)
    auth = app.state.auth_service
    session = auth.create_browser_session()

    with TestClient(app) as client:
        with client.websocket_connect(
            "/v1/socket",
            headers={"Authorization": "Bearer fm_test_secret"},
        ) as websocket:
            assert websocket.receive_text() == "ready"
        client.cookies.set(auth.cookie_name, session)
        with client.websocket_connect("/v1/socket") as websocket:
            assert websocket.receive_text() == "ready"

        try:
            with TestClient(app).websocket_connect("/v1/socket"):
                raise AssertionError("anonymous WebSocket must not connect")
        except Exception as error:
            assert getattr(error, "code", None) == 4401
