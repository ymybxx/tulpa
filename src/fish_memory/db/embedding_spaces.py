from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

import asyncpg

from fish_memory.embeddings.client import EmbeddingSpace


class EmbeddingSpaceMismatchError(RuntimeError):
    def __init__(
        self,
        fish_id: UUID,
        existing: EmbeddingSpace,
        requested: EmbeddingSpace,
    ) -> None:
        self.fish_id = fish_id
        self.existing = existing
        self.requested = requested
        super().__init__(
            f"Fish {fish_id} is bound to embedding space "
            f"{existing.provider}/{existing.model}/{existing.dimensions}, but the "
            f"service is configured for "
            f"{requested.provider}/{requested.model}/{requested.dimensions}. "
            "An existing Fish cannot switch embedding provider, model, or dimensions "
            "without rebuilding all of its embeddings."
        )


class FishEmbeddingSpaceNotFoundError(LookupError):
    pass


class FishEmbeddingSpaceStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def ensure_compatible(
        self,
        fish_id: UUID,
        requested: EmbeddingSpace,
    ) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT
                        embedding_provider,
                        embedding_model,
                        embedding_dimensions
                    FROM fishes
                    WHERE id = $1
                      AND deleted_at IS NULL
                    FOR UPDATE
                    """,
                    fish_id,
                )
                if row is None:
                    raise FishEmbeddingSpaceNotFoundError(str(fish_id))

                existing = _optional_space(row)
                if existing is None:
                    profiles = await connection.fetch(
                        """
                        SELECT DISTINCT provider, model, dimensions
                        FROM fact_embeddings
                        WHERE fish_id = $1
                        LIMIT 2
                        """,
                        fish_id,
                    )
                    if len(profiles) > 1:
                        raise RuntimeError(
                            f"Fish {fish_id} contains multiple embedding spaces"
                        )
                    existing = (
                        _required_fact_space(profiles[0]) if profiles else requested
                    )
                    await connection.execute(
                        """
                        UPDATE fishes
                        SET embedding_provider = $2,
                            embedding_model = $3,
                            embedding_dimensions = $4
                        WHERE id = $1
                        """,
                        fish_id,
                        existing.provider,
                        existing.model,
                        existing.dimensions,
                    )

                if existing != requested:
                    raise EmbeddingSpaceMismatchError(fish_id, existing, requested)


async def ensure_embedding_hnsw_index(
    pool: asyncpg.Pool,
    dimensions: int,
) -> None:
    if not 1 <= dimensions <= 2000:
        raise ValueError("embedding dimensions must be between 1 and 2000")
    queries = (
        f"""
            CREATE INDEX IF NOT EXISTS idx_fact_embeddings_hnsw_{dimensions}
            ON fact_embeddings
            USING hnsw ((embedding::vector({dimensions})) vector_cosine_ops)
            WHERE dimensions = {dimensions}
        """,
        f"""
            CREATE INDEX IF NOT EXISTS
                idx_memory_snapshot_item_embeddings_hnsw_{dimensions}
            ON memory_snapshot_item_embeddings
            USING hnsw ((embedding::vector({dimensions})) vector_cosine_ops)
            WHERE dimensions = {dimensions}
        """,
    )
    async with pool.acquire() as connection:
        for query in queries:
            await connection.execute(query)


def _optional_space(row: Mapping[str, object]) -> EmbeddingSpace | None:
    provider = row["embedding_provider"]
    model = row["embedding_model"]
    dimensions = row["embedding_dimensions"]
    if provider is None and model is None and dimensions is None:
        return None
    if not isinstance(provider, str) or not isinstance(model, str):
        raise RuntimeError("Fish has an incomplete embedding space")
    if not isinstance(dimensions, int):
        raise RuntimeError("Fish has an incomplete embedding space")
    return EmbeddingSpace(provider, model, dimensions)


def _required_fact_space(row: Mapping[str, object]) -> EmbeddingSpace:
    provider = row["provider"]
    model = row["model"]
    dimensions = row["dimensions"]
    if not isinstance(provider, str) or not isinstance(model, str):
        raise RuntimeError("Fact embedding has an invalid space")
    if not isinstance(dimensions, int):
        raise RuntimeError("Fact embedding has an invalid space")
    return EmbeddingSpace(provider, model, dimensions)
