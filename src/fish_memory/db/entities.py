from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

import asyncpg

from fish_memory.entities.models import (
    EntityCandidate,
    EntityContextMessage,
    EntityFact,
    EntityFactPage,
    EntityMergeGroup,
    EntitySummary,
    SnapshotEntityContext,
)
from fish_memory.entities.normalization import normalize_entity_alias
from fish_memory.models import Role

ENTITY_RECENT_CONTEXT_LIMIT = 20
ENTITY_RECENT_CONTEXT_MAX_CHARS = 12_000


async def lock_entity_index(
    connection: asyncpg.Connection,
    fish_id: UUID,
) -> None:
    await connection.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1::uuid::text, 0))",
        fish_id,
    )


async def replace_fact_entities(
    connection: asyncpg.Connection,
    fish_id: UUID,
    fact_id: UUID,
    aliases: Sequence[str],
) -> None:
    await connection.execute(
        "DELETE FROM fact_entities WHERE fact_id = $1",
        fact_id,
    )
    normalized_aliases = tuple(
        dict.fromkeys(
            alias for value in aliases if (alias := normalize_entity_alias(value))
        )
    )
    if not normalized_aliases:
        return
    # A data-modifying CTE's inserted rows are visible through RETURNING, while
    # the sibling table scan sees pre-existing aliases. Their union resolves
    # every alias without one database round trip per entity.
    await connection.execute(
        """
        WITH input_aliases AS (
            SELECT DISTINCT unnest($3::text[]) AS alias
        ),
        inserted AS (
            INSERT INTO entity_aliases (fish_id, alias, entity_id)
            SELECT $1, alias, gen_random_uuid()
            FROM input_aliases
            ON CONFLICT (fish_id, alias) DO NOTHING
            RETURNING alias, entity_id
        ),
        resolved AS (
            SELECT alias, entity_id
            FROM inserted
            UNION ALL
            SELECT existing.alias, existing.entity_id
            FROM entity_aliases AS existing
            JOIN input_aliases AS input
              ON input.alias = existing.alias
            WHERE existing.fish_id = $1
              AND NOT EXISTS (
                  SELECT 1
                  FROM inserted
                  WHERE inserted.alias = existing.alias
              )
        )
        INSERT INTO fact_entities (entity_id, fact_id)
        SELECT entity_id, $2
        FROM resolved
        ON CONFLICT DO NOTHING
        """,
        fish_id,
        fact_id,
        list(normalized_aliases),
    )


class EntityStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def load_context(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
    ) -> SnapshotEntityContext:
        ordered_fact_ids = tuple(dict.fromkeys(fact_ids))
        if not ordered_fact_ids:
            return SnapshotEntityContext()
        rows = await self._pool.fetch(
            """
            SELECT
                link.fact_id,
                link.entity_id,
                alias.alias,
                fact.text,
                message.role AS source_role
            FROM fact_entities AS link
            JOIN facts AS fact
              ON fact.id = link.fact_id
             AND fact.fish_id = $1
            JOIN messages AS message
              ON message.id = fact.source_message_id
            JOIN entity_aliases AS alias
              ON alias.entity_id = link.entity_id
             AND alias.fish_id = $1
            WHERE link.fact_id = ANY($2::uuid[])
            ORDER BY
                array_position($2::uuid[], link.fact_id),
                link.entity_id,
                alias.alias
            """,
            fish_id,
            list(ordered_fact_ids),
        )

        entity_order: list[UUID] = []
        aliases_by_entity: dict[UUID, list[str]] = {}
        entities_by_fact: dict[UUID, list[UUID]] = {}
        fact_source_roles: dict[UUID, Role] = {}
        fact_texts: dict[UUID, str] = {}
        for row in rows:
            fact_id = cast(UUID, row["fact_id"])
            entity_id = cast(UUID, row["entity_id"])
            alias = cast(str, row["alias"])
            fact_source_roles[fact_id] = Role(cast(str, row["source_role"]))
            fact_texts[fact_id] = cast(str, row["text"])
            if entity_id not in aliases_by_entity:
                entity_order.append(entity_id)
                aliases_by_entity[entity_id] = []
            if alias not in aliases_by_entity[entity_id]:
                aliases_by_entity[entity_id].append(alias)
            fact_entities = entities_by_fact.setdefault(fact_id, [])
            if entity_id not in fact_entities:
                fact_entities.append(entity_id)

        indexes = {entity_id: index for index, entity_id in enumerate(entity_order)}
        recent_rows = await self._pool.fetch(
            """
            WITH target AS (
                SELECT max(message.sequence) AS target_sequence
                FROM facts AS fact
                JOIN messages AS message
                  ON message.id = fact.source_message_id
                WHERE fact.fish_id = $1
                  AND fact.id = ANY($2::uuid[])
            )
            SELECT
                message.role,
                CASE
                    WHEN message.role = 'assistant'
                         AND message.consumed_offset IS NOT NULL
                         AND message.consumed_offset <> -1
                    THEN LEFT(message.content, message.consumed_offset)
                    ELSE message.content
                END AS content
            FROM messages AS message
            CROSS JOIN target
            WHERE message.fish_id = $1
              AND message.sequence <= target.target_sequence
              AND (
                  message.role = 'user'
                  OR message.status IN ('completed', 'interrupted', 'failed')
              )
            ORDER BY message.sequence DESC
            LIMIT $3
            """,
            fish_id,
            list(ordered_fact_ids),
            ENTITY_RECENT_CONTEXT_LIMIT,
        )
        recent_context = _recent_entity_context(recent_rows)
        return SnapshotEntityContext(
            candidates=tuple(
                EntityCandidate(
                    id=entity_id,
                    aliases=tuple(aliases_by_entity[entity_id]),
                )
                for entity_id in entity_order
            ),
            fact_entity_indexes={
                fact_id: tuple(indexes[entity_id] for entity_id in entity_ids)
                for fact_id, entity_ids in entities_by_fact.items()
            },
            fact_source_roles=fact_source_roles,
            fact_texts=fact_texts,
            recent_context=recent_context,
        )

    async def list_summaries(
        self,
        fish_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> list[EntitySummary]:
        if limit < 1 or offset < 0:
            raise ValueError("invalid entity pagination")
        rows = await self._pool.fetch(
            """
            SELECT
                aliases.entity_id,
                array_agg(DISTINCT aliases.alias ORDER BY aliases.alias) AS aliases,
                count(DISTINCT fact.id)::bigint AS fact_count
            FROM entity_aliases AS aliases
            LEFT JOIN fact_entities AS links
              ON links.entity_id = aliases.entity_id
            LEFT JOIN facts AS fact
              ON fact.id = links.fact_id
             AND fact.fish_id = aliases.fish_id
            WHERE aliases.fish_id = $1
              AND ($4::text IS NULL OR aliases.alias ILIKE '%' || $4 || '%')
            GROUP BY aliases.entity_id
            ORDER BY min(aliases.created_at), aliases.entity_id
            LIMIT $2 OFFSET $3
            """,
            fish_id,
            limit,
            offset,
            search,
        )
        return [
            EntitySummary(
                entity_id=cast(UUID, row["entity_id"]),
                aliases=tuple(
                    dict.fromkeys(
                        str(alias) for alias in cast(Sequence[object], row["aliases"])
                    )
                ),
                fact_count=cast(int, row["fact_count"]),
            )
            for row in rows
        ]

    async def list_facts(
        self,
        fish_id: UUID,
        entity_id: UUID,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> EntityFactPage:
        if limit < 1 or offset < 0:
            raise ValueError("invalid fact pagination")
        rows = await self._pool.fetch(
            """
            SELECT
                fact.id,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                message.role AS source_role,
                fact.invalidated_at
            FROM entity_aliases AS aliases
            JOIN fact_entities AS links
              ON links.entity_id = aliases.entity_id
            JOIN facts AS fact
              ON fact.id = links.fact_id
             AND fact.fish_id = aliases.fish_id
            JOIN messages AS message
              ON message.id = fact.source_message_id
            WHERE aliases.fish_id = $1
              AND aliases.entity_id = $2
            GROUP BY
                fact.id,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                message.role,
                fact.invalidated_at,
                fact.sequence
            ORDER BY fact.sequence DESC
            LIMIT $3 OFFSET $4
            """,
            fish_id,
            entity_id,
            limit + 1,
            offset,
        )
        has_more = len(rows) > limit
        return EntityFactPage(
            items=tuple(
                EntityFact(
                    id=cast(UUID, row["id"]),
                    text=cast(str, row["text"]),
                    occurred_start=cast(datetime | None, row["occurred_start"]),
                    occurred_end=cast(datetime | None, row["occurred_end"]),
                    mentioned_at=cast(datetime, row["mentioned_at"]),
                    source_role=cast(str, row["source_role"]),
                    invalidated_at=cast(datetime | None, row["invalidated_at"]),
                )
                for row in rows[:limit]
            ),
            has_more=has_more,
        )

    async def merge_groups(
        self,
        fish_id: UUID,
        groups: Sequence[EntityMergeGroup],
    ) -> None:
        if not groups:
            return
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await lock_entity_index(connection, fish_id)
                for group in groups:
                    await self._merge_group(connection, fish_id, group)

    async def _merge_group(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID,
        group: EntityMergeGroup,
    ) -> None:
        rows = await connection.fetch(
            """
            SELECT alias, entity_id
            FROM entity_aliases
            WHERE fish_id = $1
              AND alias = ANY($2::text[])
            ORDER BY alias
            """,
            fish_id,
            list(group.aliases),
        )
        if {cast(str, row["alias"]) for row in rows} != set(group.aliases):
            return
        entity_ids = sorted(
            {cast(UUID, row["entity_id"]) for row in rows},
            key=lambda value: value.int,
        )
        if len(entity_ids) < 2:
            return
        survivor = entity_ids[0]
        merged = entity_ids[1:]
        await connection.execute(
            """
            INSERT INTO fact_entities (entity_id, fact_id)
            SELECT $1, fact_id
            FROM fact_entities
            WHERE entity_id = ANY($2::uuid[])
            ON CONFLICT DO NOTHING
            """,
            survivor,
            merged,
        )

        await connection.execute(
            "DELETE FROM fact_entities WHERE entity_id = ANY($1::uuid[])",
            merged,
        )

        await connection.execute(
            """
            UPDATE entity_aliases
            SET entity_id = $2
            WHERE fish_id = $1
              AND entity_id = ANY($3::uuid[])
            """,
            fish_id,
            survivor,
            merged,
        )


def _recent_entity_context(
    rows: Sequence[Mapping[str, object]],
) -> tuple[EntityContextMessage, ...]:
    selected: list[EntityContextMessage] = []
    total_chars = 0
    for row in rows:
        content = str(row["content"] or "")
        if not content:
            continue
        if selected and total_chars + len(content) > ENTITY_RECENT_CONTEXT_MAX_CHARS:
            break
        selected.append(EntityContextMessage(Role(str(row["role"])), content))
        total_chars += len(content)
    selected.reverse()
    return tuple(selected)
