from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.api.routes import internal_router
from fish_memory.facts.models import BackgroundTaskProgress

from .fakes import DEFAULT_SESSION_ID, FakeFactStore, FakeSessionStore

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")


def build_test_app(store: FakeFactStore) -> FastAPI:
    app = FastAPI()
    app.state.fact_store = store
    app.state.session_store = FakeSessionStore()
    app.include_router(internal_router)
    return app


@pytest.mark.asyncio
async def test_returns_fact_extraction_progress() -> None:
    store = FakeFactStore(
        BackgroundTaskProgress(
            task_name="fact_extraction",
            through_sequence=120,
            target_sequence=123,
            pending=3,
            last_error=None,
            updated_at=datetime(2026, 7, 23, 8, 20, tzinfo=UTC),
        )
    )
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(f"/internal/fishes/{FISH_ID}/background-progress")

    assert response.status_code == 200
    assert response.json() == {
        "tasks": [
            {
                "name": "fact_extraction",
                "throughSequence": 120,
                "targetSequence": 123,
                "pending": 3,
                "lastError": None,
                "updatedAt": "2026-07-23T08:20:00Z",
            }
        ]
    }
    assert store.progress_calls == [(FISH_ID, DEFAULT_SESSION_ID)]


@pytest.mark.asyncio
async def test_returns_zero_progress_for_untouched_or_one_message_fish() -> None:
    store = FakeFactStore(
        BackgroundTaskProgress(
            task_name="fact_extraction",
            through_sequence=0,
            target_sequence=0,
            pending=0,
            last_error=None,
            updated_at=None,
        )
    )
    transport = httpx.ASGITransport(app=build_test_app(store))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(f"/internal/fishes/{FISH_ID}/background-progress")

    assert response.status_code == 200
    assert response.json() == {
        "tasks": [
            {
                "name": "fact_extraction",
                "throughSequence": 0,
                "targetSequence": 0,
                "pending": 0,
                "lastError": None,
                "updatedAt": None,
            }
        ]
    }
