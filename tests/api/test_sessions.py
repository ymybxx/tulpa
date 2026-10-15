from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.api.routes import catalog_router

from .fakes import DEFAULT_SESSION_ID, FakeSessionStore

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")


def build_test_app(store: FakeSessionStore) -> FastAPI:
    app = FastAPI()
    app.state.session_store = store
    app.include_router(catalog_router)
    return app


@pytest.mark.asyncio
async def test_lists_sessions_newest_first() -> None:
    store = FakeSessionStore()
    newest = await store.create(FISH_ID)
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/v1/fishes/{FISH_ID}/sessions")

    assert response.status_code == 200
    assert [item["sessionId"] for item in response.json()["items"]] == [
        str(newest.id),
        str(DEFAULT_SESSION_ID),
    ]


@pytest.mark.asyncio
async def test_creates_session_for_fish() -> None:
    store = FakeSessionStore()
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/v1/fishes/{FISH_ID}/sessions")

    assert response.status_code == 201
    assert response.json()["sessionId"] != str(DEFAULT_SESSION_ID)


@pytest.mark.asyncio
async def test_create_session_rejects_unknown_fish() -> None:
    store = FakeSessionStore()
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(f"/v1/fishes/{UUID(int=999)}/sessions")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "fish_not_found"
