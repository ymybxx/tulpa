import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

import asyncpg
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, RootModel

from fish_memory.answer.context import ChatMessage
from fish_memory.db.fact_frontier import fact_frontier_ctes
from fish_memory.models import Role
from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    NormalizedSnapshotOverview,
    NormalizedStaticProfile,
    PendingNormalizedSnapshotBatch,
    PendingNormalizedStaticProfileBatch,
    SnapshotFact,
    SnapshotItemMutation,
    SnapshotItemState,
)


class _NormalizedSnapshotItemPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    summary: str
    support_fact_ids: list[UUID] = Field(alias="supportFactIds")
    lineage_fact_ids: list[UUID] = Field(alias="lineageFactIds")
    last_supported_at: AwareDatetime = Field(alias="lastSupportedAt")
    item_version: int = Field(alias="itemVersion")


class _NormalizedSnapshotItemsPayload(RootModel[list[_NormalizedSnapshotItemPayload]]):
    pass


class _NormalizedSnapshotConflict(RuntimeError):
    pass


class SnapshotStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        background_owner: UUID | None = None,
    ) -> None:
        self._pool = pool
        self._background_owner = background_owner

    async def list_pending_normalized_batches(
        self,
        limit: int,
        batch_size: int,
    ) -> list[PendingNormalizedSnapshotBatch]:
        rows = await self._list_pending_normalized_dynamic_records(
            limit,
            batch_size,
        )
        grouped = _group_records_by_session(rows)
        return [
            _normalized_batch_from_records(fish_id, session_id, records)
            for (fish_id, session_id), records in grouped.items()
        ]

    async def list_pending_normalized_static_batches(
        self,
        limit: int,
        batch_size: int,
    ) -> list[PendingNormalizedStaticProfileBatch]:
        rows = await self._list_pending_normalized_static_records(
            limit,
            batch_size,
        )
        grouped = _group_records_by_fish(rows)
        return [
            _normalized_static_batch_from_records(fish_id, records)
            for fish_id, records in grouped.items()
        ]

    async def load_support_facts(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> tuple[SnapshotFact, ...]:
        requested_ids = tuple(fact_ids)
        if not requested_ids:
            return ()
        rows = await self._pool.fetch(
            """
            WITH requested(fact_id, ordinality) AS (
                SELECT *
                FROM unnest($2::uuid[]) WITH ORDINALITY
            )
            SELECT
                fact.fish_id,
                fact.id AS fact_id,
                fact.sequence AS fact_sequence,
                fact.source_message_id,
                source_message.role AS source_role,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                fact.entities
            FROM requested
            JOIN facts AS fact
              ON fact.id = requested.fact_id
             AND fact.fish_id = $1
            JOIN messages AS source_message
              ON source_message.id = fact.source_message_id
            ORDER BY requested.ordinality
            """,
            fish_id,
            list(requested_ids),
        )
        facts = tuple(_fact_from_record(row) for row in rows)
        if tuple(fact.id for fact in facts) != requested_ids:
            raise ValueError("support facts do not belong to the fish or are missing")
        return facts

    async def load_recent_turns_for_fish(
        self,
        fish_id: UUID,
        through_message_id: UUID,
        limit: int = 5,
    ) -> tuple[ChatMessage, ...]:
        if limit < 1:
            raise ValueError("recent turn limit must be positive")
        anchor = await self._pool.fetchrow(
            """
            SELECT message.session_id, message.sequence
            FROM messages AS message
            JOIN fish_sessions AS session
              ON session.id = message.session_id
             AND session.fish_id = $1
            WHERE message.id = $2
            """,
            fish_id,
            through_message_id,
        )
        if anchor is None:
            raise ValueError("through message does not belong to the fish")
        return await self._load_recent_turn_window(
            cast(UUID, anchor["session_id"]),
            cast(int, anchor["sequence"]),
            limit,
        )

    async def _load_recent_turn_window(
        self,
        session_id: UUID,
        anchor_sequence: int,
        limit: int,
    ) -> tuple[ChatMessage, ...]:
        rows = await self._pool.fetch(
            """
            WITH recent_user_turns AS (
                SELECT
                    message.sequence,
                    row_number() OVER (ORDER BY message.sequence DESC) AS turn_number
                FROM messages AS message
                WHERE message.session_id = $1
                  AND message.sequence <= $2
                  AND message.role = 'user'
            ),
            window_start AS (
                SELECT min(sequence) AS sequence
                FROM recent_user_turns
                WHERE turn_number <= $3
            )
            SELECT
                message.role,
                CASE
                    WHEN message.role = 'assistant'
                     AND message.consumed_offset IS NOT NULL
                     AND message.consumed_offset <> -1
                    THEN LEFT(message.content, message.consumed_offset)
                    ELSE message.content
                END AS visible_content
            FROM messages AS message
            CROSS JOIN window_start
            WHERE message.session_id = $1
              AND message.sequence <= $2
              AND message.sequence >= window_start.sequence
            ORDER BY message.sequence
            """,
            session_id,
            anchor_sequence,
            limit,
        )
        return tuple(
            cast(
                ChatMessage,
                {
                    "role": cast(str, row["role"]),
                    "content": cast(str, row["visible_content"]),
                },
            )
            for row in rows
        )

    async def _list_pending_normalized_dynamic_records(
        self,
        limit: int,
        batch_size: int,
    ) -> Sequence[Mapping[str, object]]:
        return await self._pool.fetch(
            f"""
            WITH {fact_frontier_ctes(owner_parameter="$3")},
            candidate_session AS (
                SELECT
                    stable_target.session_id,
                    fish.id AS fish_id,
                    COALESCE(progress.through_sequence, 0) AS through_sequence
                FROM fishes AS fish
                JOIN bounds AS stable_target
                    ON stable_target.fish_id = fish.id
                LEFT JOIN session_background_task_progress AS fact_progress
                    ON fact_progress.session_id = stable_target.session_id
                   AND fact_progress.task_name = 'fact_extraction'
                LEFT JOIN session_background_task_progress AS progress
                    ON progress.session_id = stable_target.session_id
                   AND progress.task_name = 'dynamic_snapshot'
                WHERE fish.deleted_at IS NULL
                  AND fish.snapshot_storage_version = 2
                  AND COALESCE(fact_progress.through_sequence, 0)
                      >= stable_target.target_sequence
                  AND EXISTS (
                      SELECT 1
                      FROM facts AS pending_fact
                      WHERE pending_fact.session_id = stable_target.session_id
                        AND pending_fact.sequence
                            > COALESCE(progress.through_sequence, 0)
                  )
                ORDER BY progress.updated_at NULLS FIRST, stable_target.session_id
                LIMIT $1
            )
            SELECT
                candidate.fish_id,
                candidate.session_id,
                fact.id AS fact_id,
                fact.sequence AS fact_sequence,
                fact.source_message_id,
                source_message.role AS source_role,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                fact.entities,
                snapshot.through_fact_sequence AS snapshot_through_sequence,
                snapshot.version AS snapshot_version,
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'id', item.id,
                                'summary', item.summary,
                                'supportFactIds', item.support_fact_ids,
                                'lineageFactIds', item.lineage_fact_ids,
                                'lastSupportedAt', item.last_supported_at,
                                'itemVersion', item.item_version
                            )
                            ORDER BY item.last_supported_at, item.id
                        )
                        FROM memory_snapshot_items AS item
                        WHERE item.snapshot_id = snapshot.id
                    ),
                    '[]'::jsonb
                ) AS snapshot_items
            FROM candidate_session AS candidate
            JOIN LATERAL (
                SELECT selected.*
                FROM facts AS selected
                WHERE selected.session_id = candidate.session_id
                  AND selected.sequence > candidate.through_sequence
                ORDER BY selected.sequence
                LIMIT $2
            ) AS fact ON TRUE
            JOIN messages AS source_message
                ON source_message.id = fact.source_message_id
            LEFT JOIN memory_snapshots AS snapshot
                ON snapshot.session_id = candidate.session_id
               AND snapshot.snapshot_type = 'dynamic'
            ORDER BY candidate.session_id, fact.sequence
            """,
            limit,
            batch_size,
            self._background_owner,
        )

    async def _list_pending_normalized_static_records(
        self,
        limit: int,
        batch_size: int,
    ) -> Sequence[Mapping[str, object]]:
        return await self._pool.fetch(
            f"""
            WITH {fact_frontier_ctes(owner_parameter="$3")},
            ready_fish AS (
                SELECT target.fish_id
                FROM bounds AS target
                LEFT JOIN session_background_task_progress AS fact_progress
                    ON fact_progress.session_id = target.session_id
                   AND fact_progress.task_name = 'fact_extraction'
                GROUP BY target.fish_id
                HAVING bool_and(
                    COALESCE(fact_progress.through_sequence, 0)
                        >= target.target_sequence
                )
            ),
            candidate_fish AS (
                SELECT
                    fish.id AS fish_id,
                    COALESCE(progress.through_sequence, 0) AS through_sequence
                FROM fishes AS fish
                JOIN ready_fish AS ready ON ready.fish_id = fish.id
                LEFT JOIN background_task_progress AS progress
                    ON progress.fish_id = fish.id
                   AND progress.task_name = 'static_profile'
                WHERE fish.deleted_at IS NULL
                  AND fish.snapshot_storage_version = 2
                  AND EXISTS (
                      SELECT 1
                      FROM facts AS pending_fact
                      WHERE pending_fact.fish_id = fish.id
                        AND pending_fact.sequence
                            > COALESCE(progress.through_sequence, 0)
                  )
                ORDER BY progress.updated_at NULLS FIRST, fish.id
                LIMIT $1
            )
            SELECT
                candidate.fish_id,
                fact.id AS fact_id,
                fact.sequence AS fact_sequence,
                fact.source_message_id,
                source_message.role AS source_role,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                fact.entities,
                snapshot.through_fact_sequence AS snapshot_through_sequence,
                snapshot.version AS snapshot_version,
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'id', item.id,
                                'summary', item.summary,
                                'supportFactIds', item.support_fact_ids,
                                'lineageFactIds', item.lineage_fact_ids,
                                'lastSupportedAt', item.last_supported_at,
                                'itemVersion', item.item_version
                            )
                            ORDER BY item.last_supported_at, item.id
                        )
                        FROM memory_snapshot_items AS item
                        WHERE item.snapshot_id = snapshot.id
                    ),
                    '[]'::jsonb
                ) AS snapshot_items
            FROM candidate_fish AS candidate
            JOIN LATERAL (
                SELECT selected.*
                FROM facts AS selected
                WHERE selected.fish_id = candidate.fish_id
                  AND selected.sequence > candidate.through_sequence
                ORDER BY selected.sequence
                LIMIT $2
            ) AS fact ON TRUE
            JOIN messages AS source_message
                ON source_message.id = fact.source_message_id
            LEFT JOIN memory_snapshots AS snapshot
                ON snapshot.fish_id = candidate.fish_id
               AND snapshot.snapshot_type = 'static'
               AND snapshot.session_id IS NULL
            ORDER BY candidate.fish_id, fact.sequence
            """,
            limit,
            batch_size,
            self._background_owner,
        )

    async def persist_mutations(
        self,
        batch: PendingNormalizedSnapshotBatch | PendingNormalizedStaticProfileBatch,
        mutations: Sequence[SnapshotItemMutation],
    ) -> bool:
        if isinstance(batch, PendingNormalizedSnapshotBatch):
            session_id: UUID | None = batch.session_id
            snapshot_type = "dynamic"
            task_name = "dynamic_snapshot"
            aggregate = batch.snapshot
        elif isinstance(batch, PendingNormalizedStaticProfileBatch):
            session_id = None
            snapshot_type = "static"
            task_name = "static_profile"
            aggregate = batch.profile
        else:
            raise TypeError("normalized persistence requires a normalized batch")

        expected_items = {item.id: item for item in aggregate.items}
        if len(expected_items) != len(aggregate.items):
            return False
        updates = tuple(
            mutation for mutation in mutations if mutation.action == "update"
        )
        creates = tuple(
            mutation for mutation in mutations if mutation.action == "create"
        )
        update_ids = tuple(
            mutation.item_id for mutation in updates if mutation.item_id is not None
        )
        if len(update_ids) != len(updates) or len(update_ids) != len(set(update_ids)):
            return False
        if any(item_id not in expected_items for item_id in update_ids):
            return False
        if aggregate.version == 0 and updates:
            return False

        try:
            async with self._pool.acquire() as connection:
                async with connection.transaction():
                    header = await self._advance_normalized_header(
                        connection,
                        fish_id=batch.fish_id,
                        session_id=session_id,
                        snapshot_type=snapshot_type,
                        target_fact_sequence=batch.target_fact_sequence,
                        expected_version=aggregate.version,
                        expected_through_fact_sequence=(
                            aggregate.through_fact_sequence
                        ),
                        has_mutations=bool(mutations),
                    )
                    if header is None:
                        raise _NormalizedSnapshotConflict
                    snapshot_id = cast(int, header["id"])
                    snapshot_version = cast(int, header["version"])

                    if updates:
                        locked_rows = await connection.fetch(
                            """
                            SELECT id, item_version
                            FROM memory_snapshot_items
                            WHERE snapshot_id = $1
                              AND id = ANY($2::bigint[])
                            ORDER BY id
                            FOR UPDATE
                            """,
                            snapshot_id,
                            list(update_ids),
                        )
                        locked_versions = {
                            cast(int, row["id"]): cast(int, row["item_version"])
                            for row in locked_rows
                        }
                        if locked_versions != {
                            item_id: expected_items[item_id].item_version
                            for item_id in update_ids
                        }:
                            raise _NormalizedSnapshotConflict

                    if not mutations:
                        await self._advance_normalized_progress(
                            connection,
                            fish_id=batch.fish_id,
                            session_id=session_id,
                            task_name=task_name,
                            target_fact_sequence=batch.target_fact_sequence,
                        )
                        return True

                    revision = await connection.fetchrow(
                        """
                        INSERT INTO memory_snapshot_revisions (
                            snapshot_id,
                            version,
                            through_fact_sequence
                        )
                        VALUES ($1, $2, $3)
                        RETURNING id
                        """,
                        snapshot_id,
                        snapshot_version,
                        batch.target_fact_sequence,
                    )
                    if revision is None:
                        raise RuntimeError("database did not return snapshot revision")
                    revision_id = cast(int, revision["id"])

                    applied: list[tuple[SnapshotItemMutation, int, int]] = []
                    for mutation in updates:
                        item_id = cast(int, mutation.item_id)
                        expected_item_version = expected_items[item_id].item_version
                        row = await connection.fetchrow(
                            """
                            UPDATE memory_snapshot_items
                            SET summary = $4,
                                support_fact_ids = $5,
                                lineage_fact_ids = $6,
                                last_supported_at = $7,
                                item_version = item_version + 1,
                                updated_at = now()
                            WHERE id = $1
                              AND snapshot_id = $2
                              AND item_version = $3
                            RETURNING id, item_version
                            """,
                            item_id,
                            snapshot_id,
                            expected_item_version,
                            mutation.summary,
                            list(mutation.support_fact_ids),
                            list(mutation.lineage_fact_ids),
                            mutation.last_supported_at,
                        )
                        if row is None:
                            raise _NormalizedSnapshotConflict
                        applied.append(
                            (
                                mutation,
                                cast(int, row["id"]),
                                cast(int, row["item_version"]),
                            )
                        )

                    for mutation in creates:
                        row = await connection.fetchrow(
                            """
                            INSERT INTO memory_snapshot_items (
                                snapshot_id,
                                summary,
                                support_fact_ids,
                                lineage_fact_ids,
                                last_supported_at
                            )
                            VALUES ($1, $2, $3, $4, $5)
                            RETURNING id, item_version
                            """,
                            snapshot_id,
                            mutation.summary,
                            list(mutation.support_fact_ids),
                            list(mutation.lineage_fact_ids),
                            mutation.last_supported_at,
                        )
                        if row is None:
                            raise RuntimeError("database did not return snapshot item")
                        applied.append(
                            (
                                mutation,
                                cast(int, row["id"]),
                                cast(int, row["item_version"]),
                            )
                        )

                    for mutation, item_id, item_version in applied:
                        await connection.execute(
                            """
                            INSERT INTO memory_snapshot_item_versions (
                                snapshot_item_id,
                                snapshot_revision_id,
                                snapshot_version,
                                item_version,
                                action,
                                basis,
                                summary,
                                support_fact_ids,
                                lineage_fact_ids,
                                new_support_fact_ids,
                                removed_support_fact_ids,
                                last_supported_at
                            )
                            VALUES (
                                $1, $2, $3, $4, $5, $6,
                                $7, $8, $9, $10, $11, $12
                            )
                            """,
                            item_id,
                            revision_id,
                            snapshot_version,
                            item_version,
                            mutation.action,
                            mutation.basis,
                            mutation.summary,
                            list(mutation.support_fact_ids),
                            list(mutation.lineage_fact_ids),
                            list(mutation.new_support_fact_ids),
                            list(mutation.removed_support_fact_ids),
                            mutation.last_supported_at,
                        )

                    await self._advance_normalized_progress(
                        connection,
                        fish_id=batch.fish_id,
                        session_id=session_id,
                        task_name=task_name,
                        target_fact_sequence=batch.target_fact_sequence,
                    )
                    return True
        except _NormalizedSnapshotConflict:
            return False

    async def _advance_normalized_header(
        self,
        connection: asyncpg.Connection,
        *,
        fish_id: UUID,
        session_id: UUID | None,
        snapshot_type: str,
        target_fact_sequence: int,
        expected_version: int,
        expected_through_fact_sequence: int,
        has_mutations: bool,
    ) -> asyncpg.Record | Mapping[str, object] | None:
        if expected_version == 0:
            if session_id is not None:
                return await connection.fetchrow(
                    """
                    INSERT INTO memory_snapshots (
                        fish_id,
                        session_id,
                        snapshot_type,
                        content,
                        through_fact_sequence
                    )
                    SELECT fish.id, session.id, $3, NULL, $4
                    FROM fishes AS fish
                    JOIN fish_sessions AS session
                      ON session.id = $2
                     AND session.fish_id = fish.id
                    WHERE fish.id = $1
                      AND fish.deleted_at IS NULL
                      AND fish.snapshot_storage_version = 2
                      AND fish.background_owner IS NOT DISTINCT FROM $5
                    ON CONFLICT DO NOTHING
                    RETURNING id, version
                    """,
                    fish_id,
                    session_id,
                    snapshot_type,
                    target_fact_sequence,
                    self._background_owner,
                )
            return await connection.fetchrow(
                """
                INSERT INTO memory_snapshots (
                    fish_id,
                    session_id,
                    snapshot_type,
                    content,
                    through_fact_sequence
                )
                SELECT fish.id, NULL, $2, NULL, $3
                FROM fishes AS fish
                WHERE fish.id = $1
                  AND fish.deleted_at IS NULL
                  AND fish.snapshot_storage_version = 2
                  AND fish.background_owner IS NOT DISTINCT FROM $4
                ON CONFLICT DO NOTHING
                RETURNING id, version
                """,
                fish_id,
                snapshot_type,
                target_fact_sequence,
                self._background_owner,
            )
        if has_mutations:
            return await connection.fetchrow(
                """
                UPDATE memory_snapshots AS snapshot
                SET through_fact_sequence = $4,
                    version = snapshot.version + 1,
                    updated_at = now()
                FROM fishes AS fish
                WHERE snapshot.fish_id = $1
                  AND snapshot.session_id IS NOT DISTINCT FROM $2
                  AND snapshot.snapshot_type = $3
                  AND snapshot.version = $5
                  AND snapshot.through_fact_sequence = $6
                  AND fish.id = snapshot.fish_id
                  AND fish.deleted_at IS NULL
                  AND fish.snapshot_storage_version = 2
                  AND fish.background_owner IS NOT DISTINCT FROM $7
                  AND (
                      $2::uuid IS NULL
                      OR EXISTS (
                          SELECT 1
                          FROM fish_sessions AS session
                          WHERE session.id = $2
                            AND session.fish_id = fish.id
                      )
                  )
                RETURNING snapshot.id, snapshot.version
                """,
                fish_id,
                session_id,
                snapshot_type,
                target_fact_sequence,
                expected_version,
                expected_through_fact_sequence,
                self._background_owner,
            )
        return await connection.fetchrow(
            """
            UPDATE memory_snapshots AS snapshot
            SET through_fact_sequence = $4,
                updated_at = now()
            FROM fishes AS fish
            WHERE snapshot.fish_id = $1
              AND snapshot.session_id IS NOT DISTINCT FROM $2
              AND snapshot.snapshot_type = $3
              AND snapshot.version = $5
              AND snapshot.through_fact_sequence = $6
              AND fish.id = snapshot.fish_id
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
              AND fish.background_owner IS NOT DISTINCT FROM $7
              AND (
                  $2::uuid IS NULL
                  OR EXISTS (
                      SELECT 1
                      FROM fish_sessions AS session
                      WHERE session.id = $2
                        AND session.fish_id = fish.id
                  )
              )
            RETURNING snapshot.id, snapshot.version
            """,
            fish_id,
            session_id,
            snapshot_type,
            target_fact_sequence,
            expected_version,
            expected_through_fact_sequence,
            self._background_owner,
        )

    async def _advance_normalized_progress(
        self,
        connection: asyncpg.Connection,
        *,
        fish_id: UUID,
        session_id: UUID | None,
        task_name: str,
        target_fact_sequence: int,
    ) -> None:
        if session_id is None:
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
                fish_id,
                task_name,
                target_fact_sequence,
            )
            return
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
            session_id,
            task_name,
            target_fact_sequence,
        )

    async def load_normalized_dynamic_snapshot(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> NormalizedDynamicSnapshot | None:
        row = await self._pool.fetchrow(
            """
            SELECT
                snapshot.through_fact_sequence,
                snapshot.version,
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'id', item.id,
                                'summary', item.summary,
                                'supportFactIds', item.support_fact_ids,
                                'lineageFactIds', item.lineage_fact_ids,
                                'lastSupportedAt', item.last_supported_at,
                                'itemVersion', item.item_version
                            )
                            ORDER BY item.last_supported_at, item.id
                        )
                        FROM memory_snapshot_items AS item
                        WHERE item.snapshot_id = snapshot.id
                    ),
                    '[]'::jsonb
                ) AS snapshot_items
            FROM memory_snapshots AS snapshot
            JOIN fishes AS fish ON fish.id = snapshot.fish_id
            WHERE snapshot.fish_id = $1
              AND snapshot.session_id = $2
              AND snapshot.snapshot_type = 'dynamic'
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
            """,
            fish_id,
            session_id,
        )
        if row is None:
            return None
        return NormalizedDynamicSnapshot(
            items=_normalized_items_from_value(row["snapshot_items"]),
            through_fact_sequence=cast(int, row["through_fact_sequence"]),
            version=cast(int, row["version"]),
        )

    async def load_normalized_static_profile(
        self,
        fish_id: UUID,
    ) -> NormalizedStaticProfile | None:
        row = await self._pool.fetchrow(
            """
            SELECT
                snapshot.through_fact_sequence,
                snapshot.version,
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'id', item.id,
                                'summary', item.summary,
                                'supportFactIds', item.support_fact_ids,
                                'lineageFactIds', item.lineage_fact_ids,
                                'lastSupportedAt', item.last_supported_at,
                                'itemVersion', item.item_version
                            )
                            ORDER BY item.last_supported_at, item.id
                        )
                        FROM memory_snapshot_items AS item
                        WHERE item.snapshot_id = snapshot.id
                    ),
                    '[]'::jsonb
                ) AS snapshot_items
            FROM memory_snapshots AS snapshot
            JOIN fishes AS fish ON fish.id = snapshot.fish_id
            WHERE snapshot.fish_id = $1
              AND snapshot.session_id IS NULL
              AND snapshot.snapshot_type = 'static'
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
            """,
            fish_id,
        )
        if row is None:
            return None
        return NormalizedStaticProfile(
            items=_normalized_items_from_value(row["snapshot_items"]),
            through_fact_sequence=cast(int, row["through_fact_sequence"]),
            version=cast(int, row["version"]),
        )

    async def list_normalized_current_overviews(
        self,
        fish_id: UUID,
    ) -> list[NormalizedSnapshotOverview]:
        rows = await self._pool.fetch(
            """
            SELECT
                snapshot.snapshot_type,
                snapshot.through_fact_sequence,
                snapshot.version,
                snapshot.updated_at,
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'id', item.id,
                                'summary', item.summary,
                                'supportFactIds', item.support_fact_ids,
                                'lineageFactIds', item.lineage_fact_ids,
                                'lastSupportedAt', item.last_supported_at,
                                'itemVersion', item.item_version
                            )
                            ORDER BY item.last_supported_at, item.id
                        )
                        FROM memory_snapshot_items AS item
                        WHERE item.snapshot_id = snapshot.id
                    ),
                    '[]'::jsonb
                ) AS snapshot_items
            FROM memory_snapshots AS snapshot
            JOIN fishes AS fish ON fish.id = snapshot.fish_id
            WHERE snapshot.fish_id = $1
              AND snapshot.snapshot_type IN ('dynamic', 'static')
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
            ORDER BY snapshot.snapshot_type, snapshot.session_id NULLS FIRST
            """,
            fish_id,
        )
        return [
            NormalizedSnapshotOverview(
                snapshot_type=cast(str, row["snapshot_type"]),
                version=cast(int, row["version"]),
                updated_at=cast(datetime, row["updated_at"]),
                through_fact_sequence=cast(int, row["through_fact_sequence"]),
                items=_normalized_items_from_value(row["snapshot_items"]),
            )
            for row in rows
        ]

    async def record_error(
        self,
        fish_id: UUID,
        session_id: UUID,
        error: str,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO session_background_task_progress (
                session_id, task_name, last_error
            )
            SELECT $2, 'dynamic_snapshot', $3
            FROM fish_sessions
            WHERE id = $2 AND fish_id = $1
            ON CONFLICT (session_id, task_name) DO UPDATE
            SET last_error = EXCLUDED.last_error,
                updated_at = now()
            """,
            fish_id,
            session_id,
            error,
        )

    async def record_static_error(self, fish_id: UUID, error: str) -> None:
        await self._record_error(fish_id, "static_profile", error)

    async def _record_error(
        self,
        fish_id: UUID,
        task_name: str,
        error: str,
    ) -> None:
        await self._pool.execute(
            """
            INSERT INTO background_task_progress (
                fish_id,
                task_name,
                last_error
            )
            VALUES ($1, $2, $3)
            ON CONFLICT (fish_id, task_name) DO UPDATE
            SET last_error = EXCLUDED.last_error,
                updated_at = now()
            """,
            fish_id,
            task_name,
            error,
        )


def _normalized_batch_from_records(
    fish_id: UUID,
    session_id: UUID,
    records: Sequence[Mapping[str, object]],
) -> PendingNormalizedSnapshotBatch:
    if not records:
        raise ValueError("normalized snapshot batch requires facts")
    facts = tuple(_fact_from_record(record) for record in records)
    first = records[0]
    snapshot_version = cast(int | None, first["snapshot_version"])
    snapshot = (
        NormalizedDynamicSnapshot(items=(), through_fact_sequence=0, version=0)
        if snapshot_version is None
        else NormalizedDynamicSnapshot(
            items=_normalized_items_from_value(first["snapshot_items"]),
            through_fact_sequence=cast(int, first["snapshot_through_sequence"]),
            version=snapshot_version,
        )
    )
    return PendingNormalizedSnapshotBatch(
        fish_id=fish_id,
        session_id=session_id,
        facts=facts,
        snapshot=snapshot,
        target_fact_sequence=facts[-1].sequence,
        trigger_message_id=facts[-1].source_message_id,
    )


def _normalized_static_batch_from_records(
    fish_id: UUID,
    records: Sequence[Mapping[str, object]],
) -> PendingNormalizedStaticProfileBatch:
    if not records:
        raise ValueError("normalized static profile batch requires facts")
    facts = tuple(_fact_from_record(record) for record in records)
    first = records[0]
    snapshot_version = cast(int | None, first["snapshot_version"])
    profile = (
        NormalizedStaticProfile(items=(), through_fact_sequence=0, version=0)
        if snapshot_version is None
        else NormalizedStaticProfile(
            items=_normalized_items_from_value(first["snapshot_items"]),
            through_fact_sequence=cast(int, first["snapshot_through_sequence"]),
            version=snapshot_version,
        )
    )
    return PendingNormalizedStaticProfileBatch(
        fish_id=fish_id,
        facts=facts,
        profile=profile,
        target_fact_sequence=facts[-1].sequence,
        trigger_message_id=facts[-1].source_message_id,
    )


def _group_records_by_fish(
    rows: Sequence[Mapping[str, object]],
) -> dict[UUID, list[Mapping[str, object]]]:
    grouped: dict[UUID, list[Mapping[str, object]]] = {}
    for row in rows:
        fish_id = cast(UUID, row["fish_id"])
        grouped.setdefault(fish_id, []).append(row)
    return grouped


def _group_records_by_session(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[UUID, UUID], list[Mapping[str, object]]]:
    grouped: dict[tuple[UUID, UUID], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (cast(UUID, row["fish_id"]), cast(UUID, row["session_id"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _fact_from_record(record: Mapping[str, object]) -> SnapshotFact:
    raw_entities = record["entities"]
    entities = (
        json.loads(raw_entities) if isinstance(raw_entities, str) else raw_entities
    )
    if not isinstance(entities, list):
        raise ValueError("fact entities must be a JSON array")
    return SnapshotFact(
        id=cast(UUID, record["fact_id"]),
        sequence=cast(int, record["fact_sequence"]),
        fish_id=cast(UUID, record["fish_id"]),
        source_message_id=cast(UUID, record["source_message_id"]),
        source_role=Role(cast(str, record["source_role"])),
        text=cast(str, record["text"]),
        occurred_start=cast(datetime | None, record["occurred_start"]),
        occurred_end=cast(datetime | None, record["occurred_end"]),
        mentioned_at=cast(datetime, record["mentioned_at"]),
        entities=tuple(str(item) for item in entities),
    )


def _normalized_items_from_value(
    raw_items: object,
) -> tuple[SnapshotItemState, ...]:
    payload = (
        _NormalizedSnapshotItemsPayload.model_validate_json(raw_items)
        if isinstance(raw_items, str)
        else _NormalizedSnapshotItemsPayload.model_validate(raw_items)
    )
    return tuple(
        SnapshotItemState(
            id=item.id,
            summary=item.summary,
            support_fact_ids=tuple(item.support_fact_ids),
            lineage_fact_ids=tuple(item.lineage_fact_ids),
            last_supported_at=item.last_supported_at,
            item_version=item.item_version,
        )
        for item in payload.root
    )
