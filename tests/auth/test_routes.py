from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.middleware import ApiAuthMiddleware
from fish_memory.auth.routes import auth_router
from fish_memory.auth.service import ApiAuthService


def build_app(tmp_path: Path, *, api_key: str | None = None) -> FastAPI:
    credentials = ApiKeyCredentialStore(api_key=api_key, key_file=tmp_path / "api-key")
    auth = ApiAuthService(credentials)
    app = FastAPI()
    app.state.auth_service = auth
    app.add_middleware(ApiAuthMiddleware, auth_service=auth)
    app.include_router(auth_router)

    @app.get("/v1/private")
    async def private() -> dict[str, str]:
        return {"value": "private"}

    return app


def test_unlock_sets_http_only_strict_cookie(tmp_path: Path) -> None:
    app = build_app(tmp_path)
    key = app.state.auth_service.credentials.current_key()

    with TestClient(app) as client:
        invalid = client.post("/v1/auth/unlock", json={"apiKey": "wrong"})
        valid = client.post("/v1/auth/unlock", json={"apiKey": key})
        status = client.get("/v1/auth/status")

    assert invalid.status_code == 401
    assert valid.status_code == 204
    cookie = valid.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie
    assert status.json()["authenticated"] is True


def test_rotate_invalidates_old_bearer_and_replaces_browser_session(
    tmp_path: Path,
) -> None:
    app = build_app(tmp_path)
    original = app.state.auth_service.credentials.current_key()

    with TestClient(app) as client:
        client.post("/v1/auth/unlock", json={"apiKey": original})
        rotated = client.post("/v1/auth/rotate")
        replacement = rotated.json()["apiKey"]
        status = client.get("/v1/auth/status")
        old_bearer = TestClient(app).get(
            "/v1/private",
            headers={"Authorization": f"Bearer {original}"},
        )

    assert replacement.startswith("fm_")
    assert replacement != original
    assert status.json()["authenticated"] is True
    assert old_bearer.status_code == 401


def test_environment_backed_key_cannot_be_rotated_in_browser(tmp_path: Path) -> None:
    app = build_app(tmp_path, api_key="fm_environment_key")

    with TestClient(app) as client:
        response = client.post(
            "/v1/auth/rotate",
            headers={"Authorization": "Bearer fm_environment_key"},
        )

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "environment_key_not_rotatable",
            "message": (
                "environment-backed API keys must be rotated by changing "
                "FISH_MEMORY_API_KEY and restarting fish-memory"
            ),
        }
    }
