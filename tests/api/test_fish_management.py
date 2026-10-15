import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.api.routes import catalog_router, internal_router, router
from fish_memory.entities.models import EntityFact, EntityFactPage, EntitySummary
from fish_memory.pet.hub import PetEventHub
from fish_memory.snapshots.models import (
    NormalizedSnapshotOverview,
    SnapshotItemState,
)
from fish_memory.soul import SoulProfile
from fish_memory.tts.voice_config import FishVoiceConfig, VoiceKind

FISH_ID = UUID(int=1)
ENTITY_ID = UUID(int=2)
FACT_ID = UUID(int=3)
NOW = datetime(2026, 8, 1, tzinfo=UTC)


class FakeFishStore:
    def __init__(self) -> None:
        self.profile = SoulProfile.from_values("侦探", "沉着、简洁", "小麦")
        self.created_profile: SoulProfile | None = None
        self.updated: list[tuple[UUID, SoulProfile]] = []
        self.assistant_id: UUID | None = None
        self.voice_config: FishVoiceConfig | None = None

    async def create(
        self,
        *,
        role: str | None = None,
        setting: str | None = None,
        name: str | None = None,
        background: str | None = None,
    ) -> UUID:
        self.created_profile = SoulProfile.from_values(role, setting, name, background)
        return FISH_ID

    async def exists(self, fish_id: UUID) -> bool:
        return fish_id == FISH_ID

    async def get_created_at(self, fish_id: UUID) -> None:
        return None

    async def get_profile(self, fish_id: UUID) -> SoulProfile | None:
        assert fish_id == FISH_ID
        return self.profile

    async def update_profile(self, fish_id: UUID, profile: SoulProfile) -> bool:
        if fish_id != FISH_ID:
            return False
        self.updated.append((fish_id, profile))
        self.profile = profile
        return True

    async def get_voice_config(self, fish_id: UUID) -> FishVoiceConfig | None:
        assert fish_id == FISH_ID
        return self.voice_config

    async def update_voice_config(
        self,
        fish_id: UUID,
        config: FishVoiceConfig | None,
    ) -> bool:
        if fish_id != FISH_ID:
            return False
        self.voice_config = config
        return True

    async def get_assistant(self) -> UUID | None:
        return self.assistant_id

    async def is_assistant(self, fish_id: UUID) -> bool:
        return self.assistant_id == fish_id

    async def bind_assistant(self, fish_id: UUID) -> bool:
        if not await self.exists(fish_id):
            return False
        self.assistant_id = fish_id
        return True

    async def unbind_assistant(self) -> bool:
        was_bound = self.assistant_id is not None
        self.assistant_id = None
        return was_bound


class FakeSnapshotStore:
    async def list_normalized_current_overviews(
        self,
        fish_id: UUID,
    ) -> list[NormalizedSnapshotOverview]:
        assert fish_id == FISH_ID
        return [
            NormalizedSnapshotOverview(
                snapshot_type="dynamic",
                version=2,
                updated_at=NOW,
                through_fact_sequence=9,
                items=(
                    SnapshotItemState(
                        id=17,
                        summary="用户正在开发",
                        support_fact_ids=(FACT_ID,),
                        lineage_fact_ids=(FACT_ID, UUID(int=4), UUID(int=5)),
                        last_supported_at=NOW - timedelta(days=365),
                        item_version=4,
                    ),
                ),
            )
        ]


class FakeEntityStore:
    async def list_summaries(
        self,
        fish_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> list[EntitySummary]:
        assert fish_id == FISH_ID
        assert (limit, offset, search) == (100, 0, None)
        return [EntitySummary(ENTITY_ID, ("杰洛特",), 1)]

    async def list_facts(
        self,
        fish_id: UUID,
        entity_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> EntityFactPage:
        assert (fish_id, entity_id) == (FISH_ID, ENTITY_ID)
        assert (limit, offset) == (50, 0)
        return EntityFactPage(
            items=(
                EntityFact(
                    id=FACT_ID,
                    text="用户喜欢游戏",
                    occurred_start=None,
                    occurred_end=None,
                    mentioned_at=NOW,
                    source_role="user",
                    invalidated_at=None,
                ),
            ),
            has_more=False,
        )


def build_test_app(
    fish_store: FakeFishStore | None = None,
) -> tuple[FastAPI, FakeFishStore]:
    actual_fish_store = fish_store or FakeFishStore()
    app = FastAPI()
    app.state.fish_store = actual_fish_store
    app.state.snapshot_store = FakeSnapshotStore()
    app.state.entity_store = FakeEntityStore()
    app.state.settings = type("Settings", (), {"system_prompt": "global soul"})()
    app.include_router(catalog_router)
    app.include_router(router)
    app.include_router(internal_router)
    return app, actual_fish_store


@pytest.mark.asyncio
async def test_creates_fish_with_profile() -> None:
    app, fish_store = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/fishes",
            json={
                "role": "  侦探  ",
                "setting": "  细心  ",
                "name": "  小麦  ",
            },
        )

    assert response.status_code == 201
    assert response.json() == {"fishId": str(FISH_ID)}
    assert fish_store.created_profile == SoulProfile.from_values("侦探", "细心", "小麦")


@pytest.mark.asyncio
async def test_desktop_mode_binds_first_created_fish_as_assistant() -> None:
    app, fish_store = build_test_app()
    app.state.pet_event_hub = hub = PetEventHub()
    _, events = hub.subscribe(control_only=True)
    app.state.settings.desktop_mode = True
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/fishes", json={})

    assert response.status_code == 201
    assert fish_store.assistant_id == FISH_ID
    assert json.loads(events.get_nowait().value)["type"] == "pet.assistant.changed"


@pytest.mark.asyncio
async def test_desktop_mode_keeps_existing_assistant_when_creating_fish() -> None:
    app, fish_store = build_test_app()
    app.state.pet_event_hub = hub = PetEventHub()
    _, events = hub.subscribe(control_only=True)
    app.state.settings.desktop_mode = True
    existing_assistant = UUID(int=2)
    fish_store.assistant_id = existing_assistant
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/fishes", json={})

    assert response.status_code == 201
    assert fish_store.assistant_id == existing_assistant
    assert events.empty()


@pytest.mark.asyncio
async def test_returns_management_overview_and_entity_facts() -> None:
    app, _ = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        overview = await client.get(f"/internal/fishes/{FISH_ID}/management")
        facts = await client.get(
            f"/internal/fishes/{FISH_ID}/entities/{ENTITY_ID}/facts"
        )

    assert overview.status_code == 200
    body = overview.json()
    assert body["fishId"] == str(FISH_ID)
    assert body["isAssistant"] is False
    assert body["voiceConfig"] is None
    assert "agentTargetId" not in body
    assert body["soulProfile"] == {
        "role": "侦探",
        "setting": "沉着、简洁",
        "name": "小麦",
        "background": "",
    }
    assert "身份与关系：侦探" in body["effectiveSoul"]
    assert body["dynamicSnapshot"]["items"] == [
        {
            "id": 17,
            "text": "用户正在开发",
            "lastSupportedAt": "2025-08-01T00:00:00Z",
            "sourceFactCount": 1,
        }
    ]
    assert body["entities"] == [
        {"entityId": str(ENTITY_ID), "aliases": ["杰洛特"], "factCount": 1}
    ]
    assert facts.status_code == 200
    assert facts.json()["items"][0]["factId"] == str(FACT_ID)


@pytest.mark.asyncio
async def test_updates_profile() -> None:
    app, fish_store = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch(
            f"/v1/fishes/{FISH_ID}/soul",
            json={"role": "作家", "setting": "温和", "name": "墨墨"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "role": "作家",
        "setting": "温和",
        "name": "墨墨",
        "background": "",
    }
    assert fish_store.updated == [
        (FISH_ID, SoulProfile.from_values("作家", "温和", "墨墨"))
    ]


@pytest.mark.asyncio
async def test_gets_profile_without_management_data() -> None:
    app, _ = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/v1/fishes/{FISH_ID}/soul")

    assert response.status_code == 200
    assert response.json() == {
        "role": "侦探",
        "setting": "沉着、简洁",
        "name": "小麦",
        "background": "",
    }


@pytest.mark.asyncio
async def test_updates_fish_voice() -> None:
    app, fish_store = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch(
            f"/v1/fishes/{FISH_ID}/voice",
            json={
                "model": "qwen-audio-3.0-tts-flash",
                "voice": "longanhuan_v3.6",
                "kind": "preset",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "model": "qwen-audio-3.0-tts-flash",
        "voice": "longanhuan_v3.6",
        "kind": "preset",
    }
    assert fish_store.voice_config == FishVoiceConfig(
        model="qwen-audio-3.0-tts-flash",
        voice="longanhuan_v3.6",
        kind=VoiceKind.PRESET,
    )


@pytest.mark.asyncio
async def test_management_returns_not_found_for_unknown_fish() -> None:
    app, _ = build_test_app()
    transport = httpx.ASGITransport(app=app)
    unknown = UUID(int=99)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/internal/fishes/{unknown}/management")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "fish_not_found"


@pytest.mark.asyncio
async def test_get_assistant_returns_current_binding() -> None:
    app, fish_store = build_test_app()
    fish_store.assistant_id = FISH_ID
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/fishes/assistant")

    assert response.status_code == 200
    assert response.json() == {"fishId": str(FISH_ID)}


@pytest.mark.asyncio
async def test_get_assistant_returns_null_when_unbound() -> None:
    app, _ = build_test_app()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/fishes/assistant")

    assert response.status_code == 200
    assert response.json() == {"fishId": None}


@pytest.mark.asyncio
async def test_put_assistant_binds_selected_fish() -> None:
    app, fish_store = build_test_app()
    app.state.pet_event_hub = hub = PetEventHub()
    _, events = hub.subscribe(control_only=True)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(f"/v1/fishes/{FISH_ID}/assistant")

    assert response.status_code == 200
    assert response.json() == {"fishId": str(FISH_ID)}
    assert fish_store.assistant_id == FISH_ID
    assert json.loads(events.get_nowait().value)["type"] == "pet.assistant.changed"


@pytest.mark.asyncio
async def test_put_assistant_returns_not_found_for_inactive_fish() -> None:
    app, _ = build_test_app()
    app.state.pet_event_hub = hub = PetEventHub()
    _, events = hub.subscribe(control_only=True)
    transport = httpx.ASGITransport(app=app)
    unknown = UUID(int=99)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(f"/v1/fishes/{unknown}/assistant")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "fish_not_found"
    assert events.empty()


@pytest.mark.asyncio
async def test_delete_assistant_unbinds_without_deleting_fish() -> None:
    app, fish_store = build_test_app()
    app.state.pet_event_hub = hub = PetEventHub()
    _, events = hub.subscribe(control_only=True)
    fish_store.assistant_id = FISH_ID
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/v1/fishes/assistant")

    assert response.status_code == 204
    assert fish_store.assistant_id is None
    assert json.loads(events.get_nowait().value)["type"] == "pet.assistant.changed"
