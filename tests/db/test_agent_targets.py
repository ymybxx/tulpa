from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db import agent_targets as targets_module


class RecordingPool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchrow_result: object | None = None

    async def fetchrow(self, query: str, *args: object) -> object | None:
        self.calls.append((" ".join(query.split()), args))
        return self.fetchrow_result


def store_with(pool: RecordingPool) -> Any:
    return targets_module.TargetStore(cast(asyncpg.Pool, pool))


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_type", ["Hermes", "bad slug", "-bad", "x" * 65])
async def test_upsert_rejects_invalid_connector_slug_before_database_write(
    connector_type: str,
) -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="connector type"):
        await store_with(pool).upsert_target(
            connector_type,
            "http://localhost:9000",
            "default",
            "",
            "hermes-key",
            {},
            "Hermes",
        )

    assert pool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        [],
        {"api_key": "not allowed"},
        {"nested": {"Authorization": "not allowed"}},
        {"payload": "界" * 6_000},
    ],
)
async def test_upsert_rejects_unsafe_connector_config_before_database_write(
    config: object,
) -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="connector_config"):
        await store_with(pool).upsert_target(
            "fake.valid",
            "http://localhost:9000",
            "default",
            "",
            None,
            config,
            "Fake",
        )

    assert pool.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_key",
    [
        "secret",
        "client_secret",
        "clientSecret",
        "access_token",
        "accessToken",
        "refresh_token",
        "refreshToken",
        "secret_key",
        "secretKey",
        "api_key",
        "apiKey",
        "api-key",
        "x-api-key",
        "authorization",
        "Bearer",
    ],
)
async def test_upsert_rejects_all_normalized_credential_keys(
    secret_key: str,
) -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="credential-like"):
        await store_with(pool).upsert_target(
            "fake.valid",
            "http://localhost:9000",
            "default",
            "",
            None,
            {"nested": {secret_key: "must not persist"}},
            "Fake",
        )

    assert pool.calls == []


@pytest.mark.asyncio
async def test_upsert_allows_usage_token_counter_keys() -> None:
    pool = RecordingPool()

    with pytest.raises(RuntimeError, match="did not return"):
        await store_with(pool).upsert_target(
            "fake.valid",
            "http://localhost:9000",
            "default",
            "",
            None,
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "reasoning_tokens": 30,
            },
            "Fake",
        )

    assert len(pool.calls) == 1


@pytest.mark.asyncio
async def test_upsert_rejects_oversized_external_fields_before_database_write() -> None:
    pool = RecordingPool()

    with pytest.raises(ValueError, match="external_key"):
        await store_with(pool).upsert_target(
            "hermes",
            "http://localhost:9000",
            "x" * 513,
            "",
            None,
            None,
            "Hermes",
        )

    assert pool.calls == []


@pytest.mark.asyncio
async def test_get_fish_target_reads_only_the_explicit_override() -> None:
    pool = RecordingPool()
    fish_id = UUID(int=1)

    assert await store_with(pool).get_fish_target(fish_id) is None

    query, args = pool.calls[0]
    assert args == (fish_id,)
    assert "target.id = fish.agent_target_id" in query
    assert "COALESCE" not in query


@pytest.mark.asyncio
async def test_fish_exists_distinguishes_target_update_race() -> None:
    pool = RecordingPool()
    fish_id = UUID(int=1)
    pool.fetchrow_result = {"id": fish_id}

    assert await store_with(pool).fish_exists(fish_id) is True

    query, args = pool.calls[0]
    assert args == (fish_id,)
    assert "FROM fishes" in query
    assert "agent_targets" not in query


@pytest.mark.asyncio
async def test_fish_exists_returns_false_for_missing_or_deleted_fish() -> None:
    pool = RecordingPool()

    assert await store_with(pool).fish_exists(UUID(int=1)) is False
