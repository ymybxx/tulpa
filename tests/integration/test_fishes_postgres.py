from typing import cast

import asyncpg
import pytest

from fish_memory.db.embedding_spaces import (
    EmbeddingSpaceMismatchError,
    FishEmbeddingSpaceStore,
)
from fish_memory.db.fishes import FishStore
from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.soul import SoulProfile

EMBEDDING_SPACE = EmbeddingSpace(
    provider="local",
    model="intfloat/multilingual-e5-small",
    dimensions=384,
)


@pytest.mark.asyncio
async def test_fish_store_persists_profile(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE)

    fish_id = await store.create(
        role="  侦探  ",
        setting="  简洁回答  ",
        name="  小麦  ",
    )

    storage_version = await postgres_pool.fetchval(
        "SELECT snapshot_storage_version FROM fishes WHERE id = $1",
        fish_id,
    )

    assert await store.exists(fish_id) is True
    assert storage_version == 2
    assert await store.get_profile(fish_id) == SoulProfile(
        role="侦探",
        setting="简洁回答",
        name="小麦",
    )
    assert (
        await store.update_profile(
            fish_id,
            SoulProfile.from_values(None, None, None),
        )
        is True
    )
    assert await store.get_profile(fish_id) == SoulProfile.from_values(
        None,
        None,
        None,
    )


@pytest.mark.asyncio
async def test_fish_store_persists_single_assistant_binding(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    first = await store.create(name="first")
    second = await store.create(name="second")

    assert await store.get_assistant() is None
    assert await store.bind_assistant(first) is True
    assert await store.get_assistant() == first
    assert await store.bind_assistant(second) is True
    assert await store.get_assistant() == second
    assert await store.is_assistant(first) is False
    assert await store.is_assistant(second) is True
    assert await store.unbind_assistant() is True
    assert await store.get_assistant() is None


@pytest.mark.asyncio
async def test_deleted_fish_cannot_become_assistant(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    store = FishStore(postgres_pool, embedding_space=EMBEDDING_SPACE)
    fish_id = await store.create()
    await postgres_pool.execute(
        "UPDATE fishes SET deleted_at = now() WHERE id = $1",
        fish_id,
    )

    assert await store.bind_assistant(fish_id) is False


@pytest.mark.asyncio
async def test_fishes_can_use_different_embedding_dimensions_but_each_is_locked(
    postgres_pool: asyncpg.Pool,
) -> None:
    await run_migrations(cast(MigrationPool, postgres_pool))
    space_384 = EMBEDDING_SPACE
    space_768 = EmbeddingSpace(
        provider="openai-compatible",
        model="remote-multilingual-768",
        dimensions=768,
    )
    fish_384 = await FishStore(
        postgres_pool,
        embedding_space=space_384,
    ).create(name="local")
    fish_768 = await FishStore(
        postgres_pool,
        embedding_space=space_768,
    ).create(name="remote")

    rows = await postgres_pool.fetch(
        """
        SELECT id, embedding_provider, embedding_model, embedding_dimensions
        FROM fishes
        WHERE id = ANY($1::uuid[])
        """,
        [fish_384, fish_768],
    )
    spaces = {
        row["id"]: EmbeddingSpace(
            provider=row["embedding_provider"],
            model=row["embedding_model"],
            dimensions=row["embedding_dimensions"],
        )
        for row in rows
    }
    assert spaces == {fish_384: space_384, fish_768: space_768}

    binding_store = FishEmbeddingSpaceStore(postgres_pool)
    await binding_store.ensure_compatible(fish_384, space_384)
    await binding_store.ensure_compatible(fish_768, space_768)
    with pytest.raises(EmbeddingSpaceMismatchError) as error:
        await binding_store.ensure_compatible(fish_384, space_768)

    assert error.value.existing == space_384
    assert error.value.requested == space_768
    assert "cannot switch embedding provider, model, or dimensions" in str(error.value)
