import hashlib
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

import asyncpg

from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.relations.models import (
    PendingSnapshotItemProjectionBatch,
    SnapshotItemEmbeddingProjection,
    SnapshotItemEmbeddingState,
)
from fish_memory.snapshots.models import SnapshotItemState

_TASK_NAMES = {
    "dynamic": "dynamic_relations",
    "static": "static_relations",
}


class _ProjectionConflict(RuntimeError):
    pass


class SnapshotItemRelationStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        snapshot_type: str,
        background_owner: UUID | None = None,
    ) -> None:
        if snapshot_type not in _TASK_NAMES:
            raise ValueError("invalid relation snapshot type")
        self._pool = pool
        self._snapshot_type = snapshot_type
        self._task_name = _TASK_NAMES[snapshot_type]
        self._background_owner = background_owner

    async def list_pending_batches(
        self,
        limit: int,
        dimensions: int,
    ) -> list[PendingSnapshotItemProjectionBatch]:
        if limit < 1:
            raise ValueError("relation candidate limit must be positive")
        if not 1 <= dimensions <= 2000:
            raise ValueError("relation embedding dimensions are invalid")
        if self._snapshot_type == "dynamic":
            rows = await self._list_pending_dynamic_records(limit, dimensions)
        else:
            rows = await self._list_pending_static_records(limit, dimensions)
        return _batches_from_records(rows)

    async def _list_pending_dynamic_records(
        self,
        limit: int,
        dimensions: int,
    ) -> Sequence[Mapping[str, object]]:
        return await self._pool.fetch(
            """
            WITH candidate_snapshot AS (
                SELECT
                    snapshot.id AS snapshot_id,
                    snapshot.fish_id,
                    snapshot.session_id,
                    snapshot.snapshot_type,
                    snapshot.version AS snapshot_version,
                    progress.updated_at AS progress_updated_at
                FROM memory_snapshots AS snapshot
                JOIN fish_sessions AS session
                  ON session.id = snapshot.session_id
                 AND session.fish_id = snapshot.fish_id
                JOIN fishes AS fish ON fish.id = snapshot.fish_id
                LEFT JOIN session_background_task_progress AS progress
                  ON progress.session_id = session.id
                 AND progress.task_name = $2
                WHERE snapshot.snapshot_type = $3
                  AND snapshot.session_id IS NOT NULL
                  AND fish.deleted_at IS NULL
                  AND fish.snapshot_storage_version = 2
                  AND fish.background_owner IS NOT DISTINCT FROM $4
                  AND (
                      COALESCE(progress.through_sequence, 0) < snapshot.version
                      OR EXISTS (
                          SELECT 1
                          FROM memory_snapshot_items AS item
                          LEFT JOIN memory_snapshot_item_embeddings AS embedding
                            ON embedding.snapshot_item_id = item.id
                           AND embedding.fish_id = snapshot.fish_id
                           AND embedding.session_id IS NOT DISTINCT FROM
                               snapshot.session_id
                           AND embedding.snapshot_type = snapshot.snapshot_type
                          WHERE item.snapshot_id = snapshot.id
                            AND (
                                embedding.snapshot_item_id IS NULL
                                OR embedding.item_version <> item.item_version
                                OR embedding.dimensions <> $5
                            )
                      )
                  )
                ORDER BY progress.updated_at NULLS FIRST, snapshot.id
                LIMIT $1
            )
            SELECT
                candidate.fish_id,
                candidate.snapshot_id,
                candidate.session_id,
                candidate.snapshot_type,
                candidate.snapshot_version,
                item.id AS item_id,
                item.summary,
                item.support_fact_ids,
                item.lineage_fact_ids,
                item.last_supported_at,
                item.item_version,
                embedding.snapshot_item_id AS embedding_item_id,
                embedding.item_version AS embedding_item_version,
                embedding.text_hash AS embedding_text_hash,
                embedding.provider AS embedding_provider,
                embedding.model AS embedding_model,
                embedding.dimensions AS embedding_dimensions
            FROM candidate_snapshot AS candidate
            LEFT JOIN memory_snapshot_items AS item
              ON item.snapshot_id = candidate.snapshot_id
            LEFT JOIN memory_snapshot_item_embeddings AS embedding
              ON embedding.snapshot_item_id = item.id
             AND embedding.fish_id = candidate.fish_id
             AND embedding.session_id IS NOT DISTINCT FROM candidate.session_id
             AND embedding.snapshot_type = candidate.snapshot_type
            ORDER BY candidate.snapshot_id, item.id
            """,
            limit,
            self._task_name,
            self._snapshot_type,
            self._background_owner,
            dimensions,
        )

    async def _list_pending_static_records(
        self,
        limit: int,
        dimensions: int,
    ) -> Sequence[Mapping[str, object]]:
        return await self._pool.fetch(
            """
            WITH candidate_snapshot AS (
                SELECT
                    snapshot.id AS snapshot_id,
                    snapshot.fish_id,
                    snapshot.session_id,
                    snapshot.snapshot_type,
                    snapshot.version AS snapshot_version,
                    progress.updated_at AS progress_updated_at
                FROM memory_snapshots AS snapshot
                JOIN fishes AS fish ON fish.id = snapshot.fish_id
                LEFT JOIN background_task_progress AS progress
                  ON progress.fish_id = fish.id
                 AND progress.task_name = $2
                WHERE snapshot.snapshot_type = $3
                  AND snapshot.session_id IS NULL
                  AND fish.deleted_at IS NULL
                  AND fish.snapshot_storage_version = 2
                  AND fish.background_owner IS NOT DISTINCT FROM $4
                  AND (
                      COALESCE(progress.through_sequence, 0) < snapshot.version
                      OR EXISTS (
                          SELECT 1
                          FROM memory_snapshot_items AS item
                          LEFT JOIN memory_snapshot_item_embeddings AS embedding
                            ON embedding.snapshot_item_id = item.id
                           AND embedding.fish_id = snapshot.fish_id
                           AND embedding.session_id IS NOT DISTINCT FROM
                               snapshot.session_id
                           AND embedding.snapshot_type = snapshot.snapshot_type
                          WHERE item.snapshot_id = snapshot.id
                            AND (
                                embedding.snapshot_item_id IS NULL
                                OR embedding.item_version <> item.item_version
                                OR embedding.dimensions <> $5
                            )
                      )
                  )
                ORDER BY progress.updated_at NULLS FIRST, snapshot.id
                LIMIT $1
            )
            SELECT
                candidate.fish_id,
                candidate.snapshot_id,
                candidate.session_id,
                candidate.snapshot_type,
                candidate.snapshot_version,
                item.id AS item_id,
                item.summary,
                item.support_fact_ids,
                item.lineage_fact_ids,
                item.last_supported_at,
                item.item_version,
                embedding.snapshot_item_id AS embedding_item_id,
                embedding.item_version AS embedding_item_version,
                embedding.text_hash AS embedding_text_hash,
                embedding.provider AS embedding_provider,
                embedding.model AS embedding_model,
                embedding.dimensions AS embedding_dimensions
            FROM candidate_snapshot AS candidate
            LEFT JOIN memory_snapshot_items AS item
              ON item.snapshot_id = candidate.snapshot_id
            LEFT JOIN memory_snapshot_item_embeddings AS embedding
              ON embedding.snapshot_item_id = item.id
             AND embedding.fish_id = candidate.fish_id
             AND embedding.session_id IS NOT DISTINCT FROM candidate.session_id
             AND embedding.snapshot_type = candidate.snapshot_type
            ORDER BY candidate.snapshot_id, item.id
            """,
            limit,
            self._task_name,
            self._snapshot_type,
            self._background_owner,
            dimensions,
        )

    async def persist_batch(
        self,
        batch: PendingSnapshotItemProjectionBatch,
        projections: Sequence[SnapshotItemEmbeddingProjection],
        embedding_space: EmbeddingSpace,
    ) -> bool:
        if batch.snapshot_type != self._snapshot_type:
            raise ValueError("relation batch snapshot type does not match store")
        if not _projections_match_batch(batch, projections):
            return False

        expected_versions = {item.id: item.item_version for item in batch.items}
        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    header = await connection.fetchrow(
                        """
                        SELECT snapshot.id, snapshot.version
                        FROM memory_snapshots AS snapshot
                        JOIN fishes AS fish ON fish.id = snapshot.fish_id
                        WHERE snapshot.id = $1
                          AND snapshot.fish_id = $2
                          AND snapshot.session_id IS NOT DISTINCT FROM $3
                          AND snapshot.snapshot_type = $4
                          AND snapshot.version = $5
                          AND fish.deleted_at IS NULL
                          AND fish.snapshot_storage_version = 2
                          AND fish.background_owner IS NOT DISTINCT FROM $6
                          AND (
                              $3::uuid IS NULL
                              OR EXISTS (
                                  SELECT 1
                                  FROM fish_sessions AS session
                                  WHERE session.id = $3
                                    AND session.fish_id = snapshot.fish_id
                              )
                          )
                        FOR UPDATE OF snapshot
                        """,
                        batch.snapshot_id,
                        batch.fish_id,
                        batch.session_id,
                        batch.snapshot_type,
                        batch.snapshot_version,
                        self._background_owner,
                    )
                    if header is None:
                        raise _ProjectionConflict

                    locked_items = await connection.fetch(
                        """
                        SELECT id, item_version
                        FROM memory_snapshot_items
                        WHERE snapshot_id = $1
                        ORDER BY id
                        FOR UPDATE
                        """,
                        batch.snapshot_id,
                    )
                    locked_versions = {
                        cast(int, row["id"]): cast(int, row["item_version"])
                        for row in locked_items
                    }
                    if locked_versions != expected_versions:
                        raise _ProjectionConflict

                    for projection in projections:
                        if projection.vector is None:
                            updated = await connection.execute(
                                """
                                UPDATE memory_snapshot_item_embeddings
                                SET item_version = $2,
                                    fish_id = $3,
                                    session_id = $4,
                                    snapshot_type = $5,
                                    snapshot_text = $6,
                                    text_hash = $7,
                                    last_supported_at = $8,
                                    dimensions = $9,
                                    updated_at = date_trunc('second', now())
                                WHERE snapshot_item_id = $1
                                  AND text_hash = $7
                                  AND dimensions = $9
                                """,
                                projection.item_id,
                                projection.item_version,
                                batch.fish_id,
                                batch.session_id,
                                batch.snapshot_type,
                                projection.snapshot_text,
                                projection.text_hash,
                                projection.last_supported_at,
                                embedding_space.dimensions,
                            )
                            if updated != "UPDATE 1":
                                raise _ProjectionConflict
                            continue

                        await connection.execute(
                            """
                            DELETE FROM memory_snapshot_item_mask_embeddings
                            WHERE snapshot_item_id = $1
                            """,
                            projection.item_id,
                        )
                        await connection.execute(
                            """
                            INSERT INTO memory_snapshot_item_embeddings (
                                snapshot_item_id,
                                item_version,
                                fish_id,
                                session_id,
                                snapshot_type,
                                snapshot_text,
                                text_hash,
                                last_supported_at,
                                provider,
                                model,
                                dimensions,
                                embedding
                            )
                            VALUES (
                                $1, $2, $3, $4, $5, $6,
                                $7, $8, $9, $10, $11, $12::vector
                            )
                            ON CONFLICT (snapshot_item_id) DO UPDATE
                            SET item_version = EXCLUDED.item_version,
                                fish_id = EXCLUDED.fish_id,
                                session_id = EXCLUDED.session_id,
                                snapshot_type = EXCLUDED.snapshot_type,
                                snapshot_text = EXCLUDED.snapshot_text,
                                text_hash = EXCLUDED.text_hash,
                                last_supported_at = EXCLUDED.last_supported_at,
                                provider = EXCLUDED.provider,
                                model = EXCLUDED.model,
                                dimensions = EXCLUDED.dimensions,
                                embedding = EXCLUDED.embedding,
                                updated_at = date_trunc('second', now())
                            """,
                            projection.item_id,
                            projection.item_version,
                            batch.fish_id,
                            batch.session_id,
                            batch.snapshot_type,
                            projection.snapshot_text,
                            projection.text_hash,
                            projection.last_supported_at,
                            embedding_space.provider,
                            embedding_space.model,
                            embedding_space.dimensions,
                            _vector_literal(
                                projection.vector,
                                dimensions=embedding_space.dimensions,
                            ),
                        )

                    complete = await connection.fetchval(
                        """
                        SELECT NOT EXISTS (
                            SELECT 1
                            FROM memory_snapshot_items AS item
                            LEFT JOIN memory_snapshot_item_embeddings AS embedding
                              ON embedding.snapshot_item_id = item.id
                             AND embedding.fish_id = $3
                             AND embedding.session_id IS NOT DISTINCT FROM $4
                             AND embedding.snapshot_type = $5
                            WHERE item.snapshot_id = $1
                              AND (
                                  embedding.snapshot_item_id IS NULL
                                  OR embedding.item_version <> item.item_version
                                  OR embedding.dimensions <> $2
                              )
                        )
                        """,
                        batch.snapshot_id,
                        embedding_space.dimensions,
                        batch.fish_id,
                        batch.session_id,
                        batch.snapshot_type,
                    )
                    if complete is True:
                        await self._persist_progress(connection, batch)
            return True
        except _ProjectionConflict:
            return False

    async def record_error(
        self,
        batch: PendingSnapshotItemProjectionBatch,
        error: str,
    ) -> None:
        async with self._pool.acquire() as connection:
            if batch.session_id is not None:
                await connection.execute(
                    """
                    INSERT INTO session_background_task_progress (
                        session_id, task_name, through_sequence, last_error
                    )
                    VALUES ($1, $2, 0, $3)
                    ON CONFLICT (session_id, task_name) DO UPDATE
                    SET last_error = EXCLUDED.last_error,
                        updated_at = now()
                    """,
                    batch.session_id,
                    self._task_name,
                    error,
                )
                return
            await connection.execute(
                """
                INSERT INTO background_task_progress (
                    fish_id, task_name, through_sequence, last_error
                )
                VALUES ($1, $2, 0, $3)
                ON CONFLICT (fish_id, task_name) DO UPDATE
                SET last_error = EXCLUDED.last_error,
                    updated_at = now()
                """,
                batch.fish_id,
                self._task_name,
                error,
            )

    async def _persist_progress(
        self,
        connection: asyncpg.Connection,
        batch: PendingSnapshotItemProjectionBatch,
    ) -> None:
        if batch.session_id is not None:
            await connection.execute(
                """
                INSERT INTO session_background_task_progress (
                    session_id, task_name, through_sequence, last_error
                )
                VALUES ($1, $2, $3, NULL)
                ON CONFLICT (session_id, task_name) DO UPDATE
                SET through_sequence = GREATEST(
                        session_background_task_progress.through_sequence,
                        EXCLUDED.through_sequence
                    ),
                    last_error = NULL,
                    updated_at = now()
                """,
                batch.session_id,
                self._task_name,
                batch.snapshot_version,
            )
            return
        await connection.execute(
            """
            INSERT INTO background_task_progress (
                fish_id, task_name, through_sequence, last_error
            )
            VALUES ($1, $2, $3, NULL)
            ON CONFLICT (fish_id, task_name) DO UPDATE
            SET through_sequence = GREATEST(
                    background_task_progress.through_sequence,
                    EXCLUDED.through_sequence
                ),
                last_error = NULL,
                updated_at = now()
            """,
            batch.fish_id,
            self._task_name,
            batch.snapshot_version,
        )


def _batches_from_records(
    records: Sequence[Mapping[str, object]],
) -> list[PendingSnapshotItemProjectionBatch]:
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for record in records:
        snapshot_id = cast(int, record["snapshot_id"])
        grouped.setdefault(snapshot_id, []).append(record)

    batches: list[PendingSnapshotItemProjectionBatch] = []
    for snapshot_id, rows in grouped.items():
        first = rows[0]
        items: list[SnapshotItemState] = []
        states: list[SnapshotItemEmbeddingState] = []
        for row in rows:
            raw_item_id = row["item_id"]
            if raw_item_id is None:
                continue
            item_id = cast(int, raw_item_id)
            items.append(
                SnapshotItemState(
                    id=item_id,
                    summary=cast(str, row["summary"]),
                    support_fact_ids=_uuid_tuple(row["support_fact_ids"]),
                    lineage_fact_ids=_uuid_tuple(row["lineage_fact_ids"]),
                    last_supported_at=cast(datetime, row["last_supported_at"]),
                    item_version=cast(int, row["item_version"]),
                )
            )
            if row["embedding_item_id"] is not None:
                states.append(
                    SnapshotItemEmbeddingState(
                        item_id=cast(int, row["embedding_item_id"]),
                        item_version=cast(int, row["embedding_item_version"]),
                        text_hash=cast(str, row["embedding_text_hash"]),
                        provider=cast(str, row["embedding_provider"]),
                        model=cast(str, row["embedding_model"]),
                        dimensions=cast(int, row["embedding_dimensions"]),
                    )
                )
        batches.append(
            PendingSnapshotItemProjectionBatch(
                fish_id=cast(UUID, first["fish_id"]),
                snapshot_id=snapshot_id,
                session_id=cast(UUID | None, first["session_id"]),
                snapshot_type=cast(str, first["snapshot_type"]),
                snapshot_version=cast(int, first["snapshot_version"]),
                items=tuple(items),
                embedding_states=tuple(states),
            )
        )
    return batches


def _projections_match_batch(
    batch: PendingSnapshotItemProjectionBatch,
    projections: Sequence[SnapshotItemEmbeddingProjection],
) -> bool:
    item_by_id = {item.id: item for item in batch.items}
    projection_ids = tuple(projection.item_id for projection in projections)
    if len(projection_ids) != len(set(projection_ids)):
        return False
    if not set(projection_ids).issubset(item_by_id):
        return False
    for projection in projections:
        item = item_by_id[projection.item_id]
        if (
            projection.item_version != item.item_version
            or projection.session_id != batch.session_id
            or projection.snapshot_type != batch.snapshot_type
            or projection.snapshot_text != item.summary.strip()
            or projection.last_supported_at != item.last_supported_at
            or projection.text_hash
            != hashlib.sha256(item.summary.strip().encode()).hexdigest()
        ):
            return False
    return True


def _uuid_tuple(raw_values: object) -> tuple[UUID, ...]:
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
        raise ValueError("snapshot item fact ids must be an array")
    return tuple(cast(UUID, value) for value in raw_values)


def _vector_literal(
    vector: Sequence[float],
    *,
    dimensions: int,
) -> str:
    if len(vector) != dimensions:
        raise ValueError("relation embedding dimensions do not match")
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("relation embedding contains a non-finite value")
    return "[" + ",".join(str(value) for value in values) + "]"
