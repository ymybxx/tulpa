"""Forgetting removes a memory for good, including everything derived from it.

Invalidated facts still reach recall as past records, so forgetting deletes
instead. A snapshot summary may paraphrase any fact it was built from, so every
item whose lineage touches a forgotten fact goes too; so do relation groups,
whose embeddings were computed from their members' text. Updaters only read
facts past their cursor, so the rest of those items' facts do not regrow them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from uuid import UUID

import asyncpg


@dataclass(frozen=True, slots=True)
class ForgetPlan:
    """What forgetting one memory removes."""

    fact_count: int
    item_count: int


class MemoryForgetStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def preview(
        self,
        fish_id: UUID,
        *,
        item_id: int | None = None,
        fact_id: UUID | None = None,
    ) -> ForgetPlan | None:
        async with self._pool.acquire() as connection:
            facts = await _target_facts(connection, fish_id, item_id, fact_id)
            if facts is None:
                return None
            items = await _affected_items(connection, fish_id, facts, item_id)
            return ForgetPlan(len(facts), len(items))

    async def forget(
        self,
        fish_id: UUID,
        *,
        item_id: int | None = None,
        fact_id: UUID | None = None,
    ) -> ForgetPlan | None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                # Serialize with snapshot updaters: they commit against the
                # snapshot version this bumps, so a concurrent batch retries.
                await connection.execute(
                    "SELECT id FROM memory_snapshots WHERE fish_id = $1 FOR UPDATE",
                    fish_id,
                )
                facts = await _target_facts(connection, fish_id, item_id, fact_id)
                if facts is None:
                    return None
                items = await _affected_items(connection, fish_id, facts, item_id)
                snapshots = {snapshot for _item, snapshot in items}
                await connection.execute(
                    "DELETE FROM memory_snapshot_items WHERE id = ANY($1::bigint[])",
                    [item for item, _snapshot in items],
                )
                if snapshots:
                    await connection.execute(
                        """
                        UPDATE memory_snapshots
                        SET version = version + 1,
                            updated_at = now()
                        WHERE id = ANY($1::bigint[])
                        """,
                        list(snapshots),
                    )
                await connection.execute(
                    """
                    DELETE FROM fact_relation_groups
                    WHERE fish_id = $1
                      AND fact_ids && $2::uuid[]
                    """,
                    fish_id,
                    facts,
                )
                await connection.execute(
                    "DELETE FROM facts WHERE fish_id = $1 AND id = ANY($2::uuid[])",
                    fish_id,
                    facts,
                )
                return ForgetPlan(len(facts), len(items))


async def _target_facts(
    connection: asyncpg.Connection,
    fish_id: UUID,
    item_id: int | None,
    fact_id: UUID | None,
) -> list[UUID] | None:
    if (item_id is None) == (fact_id is None):
        raise ValueError("forget exactly one snapshot item or one fact")
    if fact_id is not None:
        row = await connection.fetchrow(
            "SELECT id FROM facts WHERE fish_id = $1 AND id = $2",
            fish_id,
            fact_id,
        )
        return None if row is None else [cast(UUID, row["id"])]
    row = await connection.fetchrow(
        """
        SELECT item.lineage_fact_ids
        FROM memory_snapshot_items AS item
        JOIN memory_snapshots AS snapshot ON snapshot.id = item.snapshot_id
        WHERE snapshot.fish_id = $1
          AND item.id = $2
        """,
        fish_id,
        item_id,
    )
    if row is None:
        return None
    rows = await connection.fetch(
        "SELECT id FROM facts WHERE fish_id = $1 AND id = ANY($2::uuid[])",
        fish_id,
        list(row["lineage_fact_ids"]),
    )
    return [cast(UUID, fact["id"]) for fact in rows]


async def _affected_items(
    connection: asyncpg.Connection,
    fish_id: UUID,
    facts: list[UUID],
    item_id: int | None,
) -> list[tuple[int, int]]:
    rows = await connection.fetch(
        """
        SELECT item.id, item.snapshot_id
        FROM memory_snapshot_items AS item
        JOIN memory_snapshots AS snapshot ON snapshot.id = item.snapshot_id
        WHERE snapshot.fish_id = $1
          AND (item.lineage_fact_ids && $2::uuid[] OR item.id = $3)
        ORDER BY item.id
        """,
        fish_id,
        facts,
        item_id,
    )
    return [(cast(int, row["id"]), cast(int, row["snapshot_id"])) for row in rows]
