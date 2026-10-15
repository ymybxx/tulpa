import json
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.fishes import FishStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.soul import DEFAULT_ROLE, DEFAULT_SETTING, SoulProfile
from fish_memory.tts.voice_config import FishVoiceConfig, VoiceKind

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
EMBEDDING_SPACE = EmbeddingSpace(
    provider="local",
    model="intfloat/multilingual-e5-small",
    dimensions=384,
)


class FakePool:
    def __init__(self) -> None:
        self.fetchrow_results: list[dict[str, object] | None] = []
        self.fetchval_results: list[object] = []
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchrow(
        self,
        query: str,
        *args: Any,
    ) -> dict[str, object] | None:
        self.calls.append((" ".join(query.split()), args))
        return self.fetchrow_results.pop(0)

    async def fetchval(self, query: str, *args: Any) -> object:
        self.calls.append((" ".join(query.split()), args))
        return self.fetchval_results.pop(0)

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((" ".join(query.split()), args))
        return "OK"

    def acquire(self) -> "FakePoolContext":
        return FakePoolContext(self)

    def transaction(self) -> "FakeTransaction":
        return FakeTransaction()


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class FakePoolContext:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool

    async def __aenter__(self) -> FakePool:
        return self.pool

    async def __aexit__(self, *args: object) -> None:
        return None


def store_with(pool: FakePool) -> FishStore:
    return FishStore(cast(asyncpg.Pool, pool), embedding_space=EMBEDDING_SPACE)


@pytest.mark.asyncio
async def test_create_fish_normalizes_profile_values() -> None:
    pool = FakePool()
    pool.fetchrow_results.append({"id": FISH_ID})

    fish_id = await store_with(pool).create(
        role="  侦探  ",
        setting="  细心、克制  ",
        name="  小麦  ",
    )

    assert fish_id == FISH_ID
    query, args = pool.calls[0]
    assert "INSERT INTO fishes" in query
    assert "snapshot_storage_version" not in query
    assert isinstance(args[0], UUID)
    assert args[1:] == (
        "侦探",
        "细心、克制",
        "小麦",
        "local",
        "intfloat/multilingual-e5-small",
        384,
        None,
    )


@pytest.mark.asyncio
async def test_get_profile_uses_defaults_for_missing_values() -> None:
    pool = FakePool()
    pool.fetchrow_results.extend(
        [{"role": None, "setting": "自然", "name": "", "background": None}, None]
    )
    store = store_with(pool)

    assert await store.get_profile(FISH_ID) == SoulProfile(
        role=DEFAULT_ROLE,
        setting="自然",
        name="",
    )
    assert await store.get_profile(FISH_ID) is None


@pytest.mark.asyncio
async def test_update_profile_returns_whether_active_fish_exists() -> None:
    pool = FakePool()
    pool.fetchrow_results.extend([{"id": FISH_ID}, None])
    store = store_with(pool)

    profile = SoulProfile.from_values("  侦探  ", "  直接  ", "  小麦  ")
    assert await store.update_profile(FISH_ID, profile) is True
    assert (
        await store.update_profile(
            FISH_ID,
            SoulProfile.from_values(None, None, None),
        )
        is False
    )
    assert pool.calls[0][1] == (FISH_ID, "侦探", "直接", "小麦", None)
    assert pool.calls[1][1] == (FISH_ID, DEFAULT_ROLE, DEFAULT_SETTING, None, None)


@pytest.mark.asyncio
async def test_get_assistant_returns_database_binding() -> None:
    pool = FakePool()
    pool.fetchval_results.append(FISH_ID)

    assert await store_with(pool).get_assistant() == FISH_ID
    query, args = pool.calls[0]
    assert "is_assistant" in query
    assert "deleted_at IS NULL" in query
    assert args == ()


@pytest.mark.asyncio
async def test_bind_assistant_updates_one_active_fish_transactionally() -> None:
    pool = FakePool()
    pool.fetchrow_results.append({"id": FISH_ID})

    assert await store_with(pool).bind_assistant(FISH_ID) is True

    queries = [query for query, _ in pool.calls]
    assert "pg_advisory_xact_lock" in queries[0]
    assert "deleted_at IS NULL" in queries[1]
    assert any("SET is_assistant = FALSE" in query for query in queries)
    assert any("SET is_assistant = TRUE" in query for query in queries)


@pytest.mark.asyncio
async def test_unbind_assistant_clears_current_binding() -> None:
    pool = FakePool()
    pool.fetchrow_results.append({"id": FISH_ID})

    assert await store_with(pool).unbind_assistant() is True
    query, args = pool.calls[0]
    assert "SET is_assistant = FALSE" in query
    assert args == ()


@pytest.mark.asyncio
async def test_get_and_update_voice_config() -> None:
    pool = FakePool()
    pool.fetchrow_results.extend(
        [
            {
                "voice_config": {
                    "model": "qwen-audio-3.0-tts-flash",
                    "voice": "custom-voice",
                    "kind": "custom",
                }
            },
            {"id": FISH_ID},
        ]
    )
    store = store_with(pool)

    assert await store.get_voice_config(FISH_ID) == FishVoiceConfig(
        model="qwen-audio-3.0-tts-flash",
        voice="custom-voice",
        kind=VoiceKind.CUSTOM,
    )
    assert await store.update_voice_config(
        FISH_ID,
        FishVoiceConfig(
            model="qwen-audio-3.0-tts-flash",
            voice="custom-voice",
            kind=VoiceKind.CUSTOM,
        ),
    )

    select_query, select_args = pool.calls[0]
    update_query, update_args = pool.calls[1]
    assert "voice_config" in select_query
    assert select_args == (FISH_ID,)
    assert "SET voice_config" in update_query
    assert update_args[0] == FISH_ID
    assert json.loads(cast(str, update_args[1])) == {
        "model": "qwen-audio-3.0-tts-flash",
        "voice": "custom-voice",
        "kind": "custom",
    }
