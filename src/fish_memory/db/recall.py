from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID

import asyncpg

from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.entities.normalization import normalize_entity_alias
from fish_memory.models import Role
from fish_memory.recall.masking import EntityAliases, MaskEmbedding, SingleMaskSource
from fish_memory.recall.models import (
    RecallCandidate,
    RecallFactHead,
    RecallOrder,
    RecallRelationHead,
    RecallSourceRole,
    RecallTimeField,
    RecallTimeFilter,
    RecallTimeRange,
)


class RecallStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        background_owner: UUID | None = None,
    ) -> None:
        self._pool = pool
        self._background_owner = background_owner

    async def search_fact_head(
        self,
        fish_id: UUID,
        dimensions: int,
        vector: Sequence[float],
        *,
        source_role: RecallSourceRole = RecallSourceRole.EITHER,
        time_range: RecallTimeRange | None,
        limit: int,
        use_hnsw: bool,
    ) -> list[RecallFactHead]:
        if limit < 1:
            return []
        vector_value = _vector_literal(vector, dimensions=dimensions)
        distance = (
            f"embedding.embedding::vector({dimensions}) <=> $3::vector({dimensions})"
        )
        conditions = [
            "embedding.fish_id = $1",
            "embedding.dimensions = $2",
            "fact.fish_id = $1",
            "fact.invalidated_at IS NULL",
        ]
        args: list[object] = [fish_id, dimensions, vector_value]
        conditions.extend(
            _range_conditions("fact", time_range, args),
        )
        _append_source_role_condition(
            conditions,
            args,
            source_role,
            source_alias="source",
        )
        args.append(limit)
        where_clause = "\n                      AND ".join(conditions)
        query = f"""
            WITH fact_head AS MATERIALIZED (
                SELECT
                    fact.id,
                    fact.id AS fact_id,
                    fact.source_message_id,
                    source.role AS source_role,
                    fact.text,
                    fact.occurred_start,
                    fact.occurred_end,
                    fact.mentioned_at,
                    fact.entities,
                    fact.invalidated_at,
                    embedding.embedding_text,
                    embedding.text_hash,
                    ARRAY(
                        SELECT link.entity_id
                        FROM fact_entities AS link
                        WHERE link.fact_id = fact.id
                        ORDER BY link.entity_id
                    ) AS entity_ids,
                    1 - ({distance}) AS semantic_score
                FROM fact_embeddings AS embedding
                JOIN facts AS fact ON fact.id = embedding.fact_id
                JOIN messages AS source ON source.id = fact.source_message_id
                WHERE {where_clause}
                ORDER BY {distance}, fact.id
                LIMIT ${len(args)}
            )
            SELECT
                fact.*,
                ARRAY(
                    SELECT item.id
                    FROM memory_snapshot_items AS item
                    JOIN memory_snapshots AS snapshot
                      ON snapshot.id = item.snapshot_id
                    JOIN memory_snapshot_item_embeddings AS relation_embedding
                      ON relation_embedding.snapshot_item_id = item.id
                     AND relation_embedding.item_version = item.item_version
                    WHERE snapshot.fish_id = $1
                      AND relation_embedding.fish_id = $1
                      AND relation_embedding.dimensions = $2
                      AND relation_embedding.session_id IS NOT DISTINCT FROM
                          snapshot.session_id
                      AND relation_embedding.snapshot_type = snapshot.snapshot_type
                      AND item.support_fact_ids @> ARRAY[fact.id]::uuid[]
                    ORDER BY item.id
                ) AS relation_item_ids
            FROM fact_head AS fact
            ORDER BY fact.semantic_score DESC, fact.id
        """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SET LOCAL hnsw.iterative_scan = strict_order"
                    if use_hnsw
                    else "SET LOCAL enable_indexscan = off"
                )
                rows = await connection.fetch(query, *args)
        return [_fact_head_from_record(row) for row in rows]

    async def load_fact_keys(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
        dimensions: int,
        vector: Sequence[float],
        *,
        time_range: RecallTimeRange | None,
    ) -> list[RecallFactHead]:
        ordered_ids = tuple(dict.fromkeys(fact_ids))
        if not ordered_ids:
            return []
        vector_value = _vector_literal(vector, dimensions=dimensions)
        distance = (
            f"embedding.embedding::vector({dimensions}) <=> $4::vector({dimensions})"
        )
        conditions = [
            "fact.fish_id = $1",
            "fact.id = ANY($2::uuid[])",
            "embedding.fish_id = $1",
            "embedding.dimensions = $3",
            "fact.invalidated_at IS NULL",
        ]
        args: list[object] = [fish_id, list(ordered_ids), dimensions, vector_value]
        conditions.extend(_range_conditions("fact", time_range, args))
        where_clause = "\n              AND ".join(conditions)
        rows = await self._pool.fetch(
            f"""
            SELECT
                fact.id AS fact_id,
                fact.source_message_id,
                source.role AS source_role,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                fact.entities,
                fact.invalidated_at,
                embedding.embedding_text,
                embedding.text_hash,
                ARRAY(
                    SELECT link.entity_id
                    FROM fact_entities AS link
                    WHERE link.fact_id = fact.id
                    ORDER BY link.entity_id
                ) AS entity_ids,
                '{{}}'::bigint[] AS relation_item_ids,
                1 - ({distance}) AS semantic_score
            FROM facts AS fact
            JOIN messages AS source ON source.id = fact.source_message_id
            JOIN fact_embeddings AS embedding ON embedding.fact_id = fact.id
            WHERE {where_clause}
            ORDER BY array_position($2::uuid[], fact.id)
            """,
            *args,
        )
        return [_fact_head_from_record(row) for row in rows]

    async def count_time_scoped_facts(
        self,
        fish_id: UUID,
        dimensions: int,
        time_range: RecallTimeRange,
        *,
        source_role: RecallSourceRole = RecallSourceRole.EITHER,
        stop_after: int,
    ) -> int:
        if stop_after < 1:
            raise ValueError("time scoped fact count limit must be positive")
        conditions = [
            "fact.fish_id = $1",
            "embedding.fish_id = $1",
            "embedding.dimensions = $2",
            "fact.invalidated_at IS NULL",
        ]
        args: list[object] = [fish_id, dimensions]
        conditions.extend(_range_conditions("fact", time_range, args))
        _append_source_role_condition(
            conditions,
            args,
            source_role,
            source_alias="source",
        )
        args.append(stop_after + 1)
        where_clause = "\n                  AND ".join(conditions)
        rows = await self._pool.fetch(
            f"""
            SELECT count(*)::bigint AS fact_count
            FROM (
                SELECT 1
                FROM facts AS fact
                JOIN fact_embeddings AS embedding
                  ON embedding.fact_id = fact.id
                JOIN messages AS source ON source.id = fact.source_message_id
                WHERE {where_clause}
                LIMIT ${len(args)}
            ) AS scoped
            """,
            *args,
        )
        if not rows:
            raise RuntimeError("time scoped fact count returned no row")
        return cast(int, rows[0]["fact_count"])

    async def load_fact_mask_embeddings(
        self,
        fish_id: UUID,
        dimensions: int,
        fact_ids: Sequence[UUID],
    ) -> list[MaskEmbedding]:
        ordered_ids = tuple(dict.fromkeys(fact_ids))
        if not ordered_ids:
            return []
        rows = await self._pool.fetch(
            """
            SELECT
                fact_id AS object_id,
                entity_ids,
                signature_hash,
                source_text_hash,
                masked_text,
                embedding::text AS vector_text
            FROM fact_mask_embeddings
            WHERE fish_id = $1
              AND dimensions = $2
              AND fact_id = ANY($3::uuid[])
            ORDER BY fact_id, signature_hash
            """,
            fish_id,
            dimensions,
            list(ordered_ids),
        )
        return [_mask_embedding_from_record(row) for row in rows]

    async def list_fact_single_mask_sources(
        self,
        dimensions: int,
        limit: int,
        *,
        offset: int = 0,
    ) -> list[SingleMaskSource]:
        rows = await self._pool.fetch(
            """
            SELECT
                embedding.fact_id AS object_id,
                embedding.fish_id,
                embedding.embedding_text AS source_text,
                embedding.text_hash AS source_text_hash,
                link.entity_id,
                aliases.aliases
            FROM fact_embeddings AS embedding
            JOIN facts AS fact ON fact.id = embedding.fact_id
            JOIN fishes AS fish ON fish.id = embedding.fish_id
            JOIN fact_entities AS link ON link.fact_id = fact.id
            JOIN LATERAL (
                SELECT array_agg(alias.alias ORDER BY alias.alias) AS aliases
                FROM entity_aliases AS alias
                WHERE alias.fish_id = embedding.fish_id
                  AND alias.entity_id = link.entity_id
            ) AS aliases ON TRUE
            WHERE embedding.dimensions = $1
              AND embedding.embedding_text IS NOT NULL
              AND embedding.text_hash IS NOT NULL
              AND fact.invalidated_at IS NULL
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
              AND fish.background_owner IS NOT DISTINCT FROM $3
              AND EXISTS (
                  SELECT 1
                  FROM unnest(aliases.aliases) AS candidate(alias)
                  WHERE strpos(lower(embedding.embedding_text), candidate.alias) > 0
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM fact_mask_embeddings AS existing
                  WHERE existing.fact_id = embedding.fact_id
                    AND existing.dimensions = embedding.dimensions
                    AND existing.source_text_hash = embedding.text_hash
                    AND existing.entity_ids = ARRAY[link.entity_id]::uuid[]
              )
            ORDER BY embedding.fact_id, link.entity_id
            LIMIT $2
            OFFSET $4
            """,
            dimensions,
            limit,
            self._background_owner,
            offset,
        )
        return [_single_mask_source_from_record("fact", row) for row in rows]

    async def list_relation_single_mask_sources(
        self,
        dimensions: int,
        limit: int,
        *,
        offset: int = 0,
    ) -> list[SingleMaskSource]:
        rows = await self._pool.fetch(
            """
            SELECT
                relation_embedding.snapshot_item_id AS object_id,
                relation_embedding.fish_id,
                relation_embedding.snapshot_text AS source_text,
                relation_embedding.text_hash AS source_text_hash,
                entity.entity_id,
                aliases.aliases
            FROM memory_snapshot_item_embeddings AS relation_embedding
            JOIN fishes AS fish ON fish.id = relation_embedding.fish_id
            JOIN memory_snapshot_items AS item
              ON item.id = relation_embedding.snapshot_item_id
             AND relation_embedding.item_version = item.item_version
            JOIN memory_snapshots AS snapshot
              ON snapshot.id = item.snapshot_id
             AND snapshot.fish_id = relation_embedding.fish_id
             AND relation_embedding.session_id IS NOT DISTINCT FROM snapshot.session_id
             AND relation_embedding.snapshot_type = snapshot.snapshot_type
            JOIN LATERAL (
                SELECT DISTINCT link.entity_id
                FROM fact_entities AS link
                JOIN facts AS fact ON fact.id = link.fact_id
                WHERE link.fact_id = ANY(item.support_fact_ids)
                  AND fact.fish_id = relation_embedding.fish_id
                  AND fact.invalidated_at IS NULL
            ) AS entity ON TRUE
            JOIN LATERAL (
                SELECT array_agg(alias.alias ORDER BY alias.alias) AS aliases
                FROM entity_aliases AS alias
                WHERE alias.fish_id = relation_embedding.fish_id
                  AND alias.entity_id = entity.entity_id
            ) AS aliases ON TRUE
            WHERE relation_embedding.dimensions = $1
              AND fish.deleted_at IS NULL
              AND fish.snapshot_storage_version = 2
              AND fish.background_owner IS NOT DISTINCT FROM $3
              AND EXISTS (
                  SELECT 1
                  FROM unnest(aliases.aliases) AS candidate(alias)
                  WHERE strpos(lower(relation_embedding.snapshot_text), candidate.alias)
                        > 0
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM memory_snapshot_item_mask_embeddings AS existing
                  WHERE existing.snapshot_item_id =
                        relation_embedding.snapshot_item_id
                    AND existing.dimensions = relation_embedding.dimensions
                    AND existing.source_text_hash = relation_embedding.text_hash
                    AND existing.entity_ids = ARRAY[entity.entity_id]::uuid[]
              )
            ORDER BY relation_embedding.snapshot_item_id, entity.entity_id
            LIMIT $2
            OFFSET $4
            """,
            dimensions,
            limit,
            self._background_owner,
            offset,
        )
        return [_single_mask_source_from_record("relation", row) for row in rows]

    async def resolve_query_entities(
        self,
        fish_id: UUID,
        query: str,
    ) -> list[EntityAliases]:
        batches = await self.resolve_query_entities_batch(fish_id, (query,))
        return list(batches[0])

    async def resolve_query_entities_batch(
        self,
        fish_id: UUID,
        queries: Sequence[str],
    ) -> tuple[tuple[EntityAliases, ...], ...]:
        normalized_queries = [normalize_entity_alias(query) for query in queries]
        if not normalized_queries:
            return ()
        rows = await self._pool.fetch(
            """
            WITH inputs AS (
                SELECT ordinality - 1 AS query_index, query
                FROM unnest($2::text[]) WITH ORDINALITY AS value(query, ordinality)
            ), matched AS (
                SELECT DISTINCT input.query_index, alias.entity_id
                FROM inputs AS input
                JOIN entity_aliases AS alias
                  ON alias.fish_id = $1
                 AND strpos(input.query, alias.alias) > 0
            )
            SELECT
                matched.query_index,
                alias.entity_id,
                array_agg(alias.alias ORDER BY alias.alias) AS aliases
            FROM matched
            JOIN entity_aliases AS alias
              ON alias.fish_id = $1
             AND alias.entity_id = matched.entity_id
            GROUP BY matched.query_index, alias.entity_id
            ORDER BY matched.query_index, alias.entity_id
            """,
            fish_id,
            normalized_queries,
        )
        result: list[list[EntityAliases]] = [[] for _ in normalized_queries]
        for row in rows:
            query_index = cast(int, row["query_index"])
            if query_index < 0 or query_index >= len(result):
                raise ValueError("query entity row has an invalid query index")
            raw_aliases = row["aliases"]
            if not isinstance(raw_aliases, Sequence) or isinstance(
                raw_aliases,
                (str, bytes),
            ):
                raise ValueError("query entity aliases must be an array")
            result[query_index].append(
                EntityAliases(
                    entity_id=cast(UUID, row["entity_id"]),
                    aliases=tuple(str(value) for value in raw_aliases),
                )
            )
        return tuple(tuple(entities) for entities in result)

    async def load_relation_mask_embeddings(
        self,
        fish_id: UUID,
        dimensions: int,
        relation_item_ids: Sequence[int],
    ) -> list[MaskEmbedding]:
        ordered_ids = tuple(dict.fromkeys(relation_item_ids))
        if not ordered_ids:
            return []
        rows = await self._pool.fetch(
            """
            SELECT
                mask.snapshot_item_id AS object_id,
                mask.entity_ids,
                mask.signature_hash,
                mask.source_text_hash,
                mask.masked_text,
                mask.embedding::text AS vector_text
            FROM memory_snapshot_item_mask_embeddings AS mask
            JOIN memory_snapshot_item_embeddings AS relation_embedding
              ON relation_embedding.snapshot_item_id = mask.snapshot_item_id
             AND relation_embedding.fish_id = mask.fish_id
             AND relation_embedding.dimensions = mask.dimensions
             AND mask.source_text_hash = relation_embedding.text_hash
            JOIN memory_snapshot_items AS item
              ON item.id = relation_embedding.snapshot_item_id
             AND relation_embedding.item_version = item.item_version
            JOIN memory_snapshots AS snapshot
              ON snapshot.id = item.snapshot_id
             AND snapshot.fish_id = relation_embedding.fish_id
             AND relation_embedding.session_id IS NOT DISTINCT FROM snapshot.session_id
             AND relation_embedding.snapshot_type = snapshot.snapshot_type
            WHERE snapshot.fish_id = $1
              AND mask.dimensions = $2
              AND mask.snapshot_item_id = ANY($3::bigint[])
            ORDER BY mask.snapshot_item_id, mask.signature_hash
            """,
            fish_id,
            dimensions,
            list(ordered_ids),
        )
        return [_mask_embedding_from_record(row) for row in rows]

    async def upsert_fact_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None:
        if not embeddings:
            return
        rows: list[tuple[object, ...]] = []
        for embedding in embeddings:
            if not isinstance(embedding.object_id, UUID):
                raise ValueError("fact mask object id must be a UUID")
            rows.append(_mask_embedding_row(fish_id, space, embedding))
        await self._pool.executemany(
            """
            INSERT INTO fact_mask_embeddings (
                fact_id,
                fish_id,
                entity_ids,
                signature_hash,
                source_text_hash,
                masked_text,
                provider,
                model,
                dimensions,
                embedding
            )
            VALUES ($1, $2, $3::uuid[], $4, $5, $6, $7, $8, $9, $10::vector)
            ON CONFLICT (fact_id, signature_hash, dimensions) DO UPDATE
            SET fish_id = EXCLUDED.fish_id,
                entity_ids = EXCLUDED.entity_ids,
                source_text_hash = EXCLUDED.source_text_hash,
                masked_text = EXCLUDED.masked_text,
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                embedding = EXCLUDED.embedding,
                updated_at = date_trunc('second', now())
            """,
            rows,
        )

    async def upsert_relation_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None:
        if not embeddings:
            return
        rows: list[tuple[object, ...]] = []
        for embedding in embeddings:
            if not isinstance(embedding.object_id, int):
                raise ValueError("relation mask object id must be an integer")
            rows.append(_mask_embedding_row(fish_id, space, embedding))
        await self._pool.executemany(
            """
            WITH current_projection AS MATERIALIZED (
                SELECT item.id, relation_embedding.snapshot_item_id
                FROM memory_snapshot_item_embeddings AS relation_embedding
                JOIN memory_snapshot_items AS item
                  ON item.id = relation_embedding.snapshot_item_id
                 AND relation_embedding.item_version = item.item_version
                JOIN memory_snapshots AS snapshot
                  ON snapshot.id = item.snapshot_id
                 AND snapshot.fish_id = relation_embedding.fish_id
                 AND relation_embedding.session_id IS NOT DISTINCT FROM
                     snapshot.session_id
                 AND relation_embedding.snapshot_type = snapshot.snapshot_type
                WHERE relation_embedding.snapshot_item_id = $1
                  AND snapshot.fish_id = $2
                  AND relation_embedding.fish_id = $2
                  AND relation_embedding.text_hash = $5
                  AND relation_embedding.dimensions = $9
                FOR SHARE OF relation_embedding
            )
            INSERT INTO memory_snapshot_item_mask_embeddings (
                snapshot_item_id,
                fish_id,
                entity_ids,
                signature_hash,
                source_text_hash,
                masked_text,
                provider,
                model,
                dimensions,
                embedding
            )
            SELECT
                $1, $2, $3::uuid[], $4, $5,
                $6, $7, $8, $9, $10::vector
            FROM current_projection
            ON CONFLICT (snapshot_item_id, signature_hash, dimensions) DO UPDATE
            SET fish_id = EXCLUDED.fish_id,
                entity_ids = EXCLUDED.entity_ids,
                source_text_hash = EXCLUDED.source_text_hash,
                masked_text = EXCLUDED.masked_text,
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                embedding = EXCLUDED.embedding,
                updated_at = date_trunc('second', now())
            """,
            rows,
        )

    async def search_relation_head(
        self,
        fish_id: UUID,
        dimensions: int,
        vector: Sequence[float],
        *,
        source_role: RecallSourceRole = RecallSourceRole.EITHER,
        time_range: RecallTimeRange | None,
        limit: int,
        use_hnsw: bool,
    ) -> list[RecallRelationHead]:
        if limit < 1:
            return []
        vector_value = _vector_literal(vector, dimensions=dimensions)
        distance = (
            f"relation_embedding.embedding::vector({dimensions}) "
            f"<=> $3::vector({dimensions})"
        )
        member_conditions = [
            "member.id = ANY(item.support_fact_ids)",
            "member.fish_id = $1",
            "member.invalidated_at IS NULL",
        ]
        args: list[object] = [fish_id, dimensions, vector_value]
        member_conditions.extend(
            _range_conditions("member", time_range, args),
        )
        source_role_conditions: list[str] = []
        _append_source_role_condition(
            source_role_conditions,
            args,
            source_role,
            source_alias="source_message",
        )
        args.append(limit)
        member_where = "\n                          AND ".join(member_conditions)
        source_role_where = "\n                  AND ".join(
            [
                "source_member.id = ANY(item.support_fact_ids)",
                "source_member.fish_id = $1",
                "source_member.invalidated_at IS NULL",
                *source_role_conditions,
            ]
        )
        source_role_predicate = (
            ""
            if source_role is RecallSourceRole.EITHER
            else f"""
              AND EXISTS (
                  SELECT 1
                  FROM facts AS source_member
                  JOIN messages AS source_message
                    ON source_message.id = source_member.source_message_id
                  WHERE {source_role_where}
              )"""
        )
        query = f"""
            SELECT
                item.id AS snapshot_item_id,
                relation_embedding.snapshot_type,
                audit.snapshot_version,
                relation_embedding.snapshot_text,
                relation_embedding.text_hash,
                relation_embedding.last_supported_at,
                item.support_fact_ids,
                item.support_count,
                1 - ({distance}) AS semantic_score
            FROM memory_snapshot_item_embeddings AS relation_embedding
            JOIN memory_snapshot_items AS item
              ON item.id = relation_embedding.snapshot_item_id
             AND relation_embedding.item_version = item.item_version
            JOIN memory_snapshots AS snapshot
              ON snapshot.id = item.snapshot_id
             AND snapshot.fish_id = relation_embedding.fish_id
             AND relation_embedding.session_id IS NOT DISTINCT FROM snapshot.session_id
             AND relation_embedding.snapshot_type = snapshot.snapshot_type
            JOIN memory_snapshot_item_versions AS audit
              ON audit.snapshot_item_id = item.id
             AND audit.item_version = item.item_version
            WHERE snapshot.fish_id = $1
              AND relation_embedding.fish_id = $1
              AND relation_embedding.dimensions = $2
              AND (
                  SELECT count(DISTINCT member.id)::integer
                  FROM facts AS member
                  JOIN fact_embeddings AS member_embedding
                    ON member_embedding.fact_id = member.id
                   AND member_embedding.fish_id = $1
                   AND member_embedding.dimensions = $2
                  WHERE {member_where}
              ) = item.support_count
              {source_role_predicate}
            ORDER BY {distance}, item.id
            LIMIT ${len(args)}
        """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SET LOCAL hnsw.iterative_scan = strict_order"
                    if use_hnsw
                    else "SET LOCAL enable_indexscan = off"
                )
                rows = await connection.fetch(query, *args)
        return [_relation_head_from_record(row) for row in rows]

    async def load_relation_item_keys(
        self,
        fish_id: UUID,
        relation_item_ids: Sequence[int],
        dimensions: int,
        vector: Sequence[float],
        *,
        time_range: RecallTimeRange | None,
    ) -> list[RecallRelationHead]:
        ordered_ids = tuple(dict.fromkeys(relation_item_ids))
        if not ordered_ids:
            return []
        vector_value = _vector_literal(vector, dimensions=dimensions)
        distance = (
            f"relation_embedding.embedding::vector({dimensions}) "
            f"<=> $4::vector({dimensions})"
        )
        member_conditions = [
            "member.id = ANY(item.support_fact_ids)",
            "member.fish_id = $1",
            "member.invalidated_at IS NULL",
        ]
        args: list[object] = [
            fish_id,
            list(ordered_ids),
            dimensions,
            vector_value,
        ]
        member_conditions.extend(
            _range_conditions("member", time_range, args),
        )
        member_where = "\n                      AND ".join(member_conditions)
        rows = await self._pool.fetch(
            f"""
            SELECT
                item.id AS snapshot_item_id,
                relation_embedding.snapshot_type,
                audit.snapshot_version,
                relation_embedding.snapshot_text,
                relation_embedding.text_hash,
                relation_embedding.last_supported_at,
                item.support_fact_ids,
                item.support_count,
                1 - ({distance}) AS semantic_score
            FROM memory_snapshot_item_embeddings AS relation_embedding
            JOIN memory_snapshot_items AS item
              ON item.id = relation_embedding.snapshot_item_id
             AND relation_embedding.item_version = item.item_version
            JOIN memory_snapshots AS snapshot
              ON snapshot.id = item.snapshot_id
             AND snapshot.fish_id = relation_embedding.fish_id
             AND relation_embedding.session_id IS NOT DISTINCT FROM snapshot.session_id
             AND relation_embedding.snapshot_type = snapshot.snapshot_type
            JOIN memory_snapshot_item_versions AS audit
              ON audit.snapshot_item_id = item.id
             AND audit.item_version = item.item_version
            WHERE snapshot.fish_id = $1
              AND relation_embedding.fish_id = $1
              AND item.id = ANY($2::bigint[])
              AND relation_embedding.dimensions = $3
              AND EXISTS (
                  SELECT 1
                  FROM facts AS member
                  WHERE {member_where}
              )
            ORDER BY array_position($2::bigint[], item.id)
            """,
            *args,
        )
        return [_relation_head_from_record(row) for row in rows]

    async def semantic_search(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        vector: Sequence[float],
        *,
        limit: int,
        time_filter: RecallTimeFilter | None = None,
        order: RecallOrder = RecallOrder.RELEVANCE,
    ) -> list[RecallCandidate]:
        if time_filter is not None and time_filter.field is RecallTimeField.EITHER:
            raise ValueError("either time field must be split before database search")
        vector_value = _vector_literal(vector, dimensions=space.dimensions)
        distance = (
            f"embedding.embedding::vector({space.dimensions}) "
            f"<=> $5::vector({space.dimensions})"
        )
        conditions: list[str] = [
            "embedding.fish_id = $1",
            "embedding.provider = $2",
            "embedding.model = $3",
            "embedding.dimensions = $4",
        ]
        args: list[object] = [
            fish_id,
            space.provider,
            space.model,
            space.dimensions,
            vector_value,
        ]
        time_column: str | None = None
        if time_filter is not None and time_filter.field is not None:
            if time_filter.field is RecallTimeField.OCCURRED:
                conditions.append("fact.occurred_start IS NOT NULL")
                time_column = "fact.occurred_start"
                if time_filter.start is not None:
                    args.append(time_filter.start)
                    conditions.append(
                        "COALESCE(fact.occurred_end, fact.occurred_start) "
                        f">= ${len(args)}"
                    )
                if time_filter.end is not None:
                    args.append(time_filter.end)
                    conditions.append(f"fact.occurred_start <= ${len(args)}")
            else:
                time_column = "fact.mentioned_at"
                if time_filter.start is not None:
                    args.append(time_filter.start)
                    conditions.append(f"fact.mentioned_at >= ${len(args)}")
                if time_filter.end is not None:
                    args.append(time_filter.end)
                    conditions.append(f"fact.mentioned_at <= ${len(args)}")

        where_clause = "\n                      AND ".join(conditions)
        if order is RecallOrder.RELEVANCE:
            args.append(limit)
            query = f"""
                SELECT
                    fact.id AS fact_id,
                    fact.source_message_id,
                    source.role AS source_role,
                    fact.text,
                    fact.occurred_start,
                    fact.occurred_end,
                    fact.mentioned_at,
                    fact.entities,
                    fact.invalidated_at,
                    1 - (
                        {distance}
                    ) AS semantic_score
                FROM fact_embeddings AS embedding
                JOIN facts AS fact ON fact.id = embedding.fact_id
                JOIN messages AS source
                    ON source.id = fact.source_message_id
                WHERE {where_clause}
                ORDER BY {distance}, fact.id
                LIMIT ${len(args)}
            """
        else:
            if time_column is None:
                raise ValueError("chronological search requires a time field")
            direction = "ASC" if order is RecallOrder.EARLIEST else "DESC"
            args.append(limit)
            query = f"""
                SELECT
                    fact.id AS fact_id,
                    fact.source_message_id,
                    source.role AS source_role,
                    fact.text,
                    fact.occurred_start,
                    fact.occurred_end,
                    fact.mentioned_at,
                    fact.entities,
                    fact.invalidated_at,
                    1 - (
                        {distance}
                    ) AS semantic_score
                FROM fact_embeddings AS embedding
                JOIN facts AS fact ON fact.id = embedding.fact_id
                JOIN messages AS source
                    ON source.id = fact.source_message_id
                WHERE {where_clause}
                ORDER BY {time_column} {direction},
                         {distance},
                         fact.id
                LIMIT ${len(args)}
            """
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SET LOCAL hnsw.iterative_scan = strict_order")
                rows = await connection.fetch(query, *args)
        return [_candidate_from_record(row) for row in rows]

    async def entity_search(
        self,
        fish_id: UUID,
        aliases: Sequence[str],
        space: EmbeddingSpace,
        vector: Sequence[float],
        *,
        time_filter: RecallTimeFilter | None = None,
        order: RecallOrder = RecallOrder.RELEVANCE,
        limit: int,
    ) -> list[RecallCandidate]:
        if not aliases:
            return []
        if time_filter is not None and time_filter.field is RecallTimeField.EITHER:
            raise ValueError("either time field must be split before database search")
        conditions = ["fact.fish_id = $1", "source.role = 'user'"]
        args: list[object] = [
            fish_id,
            list(aliases),
            space.provider,
            space.model,
            space.dimensions,
            _vector_literal(vector, dimensions=space.dimensions),
        ]
        distance = (
            f"embedding.embedding::vector({space.dimensions}) "
            f"<=> $6::vector({space.dimensions})"
        )
        time_column: str | None = None
        if time_filter is not None and time_filter.field is not None:
            if time_filter.field is RecallTimeField.OCCURRED:
                conditions.append("fact.occurred_start IS NOT NULL")
                time_column = "fact.occurred_start"
                if time_filter.start is not None:
                    args.append(time_filter.start)
                    conditions.append(
                        "COALESCE(fact.occurred_end, fact.occurred_start) "
                        f">= ${len(args)}"
                    )
                if time_filter.end is not None:
                    args.append(time_filter.end)
                    conditions.append(f"fact.occurred_start <= ${len(args)}")
            else:
                time_column = "fact.mentioned_at"
                if time_filter.start is not None:
                    args.append(time_filter.start)
                    conditions.append(f"fact.mentioned_at >= ${len(args)}")
                if time_filter.end is not None:
                    args.append(time_filter.end)
                    conditions.append(f"fact.mentioned_at <= ${len(args)}")
        if order is RecallOrder.RELEVANCE:
            order_clause = "semantic_score DESC NULLS LAST, fact.id"
        else:
            if time_column is None:
                raise ValueError("chronological search requires a time field")
            direction = "ASC" if order is RecallOrder.EARLIEST else "DESC"
            order_clause = (
                f"{time_column} {direction}, semantic_score DESC NULLS LAST, fact.id"
            )
        args.append(limit)
        where_clause = "\n              AND ".join(conditions)
        rows = await self._pool.fetch(
            f"""
            WITH matched_facts AS (
                SELECT DISTINCT link.fact_id
                FROM entity_aliases AS alias
                JOIN fact_entities AS link
                  ON link.entity_id = alias.entity_id
                WHERE alias.fish_id = $1
                  AND alias.alias = ANY($2::text[])
            )
            SELECT
                fact.id AS fact_id,
                fact.source_message_id,
                source.role AS source_role,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                fact.entities,
                fact.invalidated_at,
                1 - ({distance}) AS semantic_score
            FROM matched_facts AS matched
            JOIN facts AS fact
              ON fact.id = matched.fact_id
            JOIN messages AS source
              ON source.id = fact.source_message_id
            LEFT JOIN fact_embeddings AS embedding
              ON embedding.fact_id = fact.id
             AND embedding.provider = $3
             AND embedding.model = $4
             AND embedding.dimensions = $5
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT ${len(args)}
            """,
            *args,
        )
        return [_candidate_from_record(row) for row in rows]

    async def time_search(
        self,
        fish_id: UUID,
        time_filter: RecallTimeFilter,
        order: RecallOrder,
        *,
        limit: int,
    ) -> list[RecallCandidate]:
        if time_filter.field is None:
            return []
        if time_filter.field is RecallTimeField.EITHER:
            raise ValueError("either time field must be split before database search")

        conditions = ["fact.fish_id = $1"]
        args: list[object] = [fish_id]
        if time_filter.field is RecallTimeField.OCCURRED:
            conditions.append("fact.occurred_start IS NOT NULL")
            order_column = "fact.occurred_start"
            if time_filter.start is not None:
                args.append(time_filter.start)
                conditions.append(
                    f"COALESCE(fact.occurred_end, fact.occurred_start) >= ${len(args)}"
                )
            if time_filter.end is not None:
                args.append(time_filter.end)
                conditions.append(f"fact.occurred_start <= ${len(args)}")
        else:
            order_column = "fact.mentioned_at"
            if time_filter.start is not None:
                args.append(time_filter.start)
                conditions.append(f"fact.mentioned_at >= ${len(args)}")
            if time_filter.end is not None:
                args.append(time_filter.end)
                conditions.append(f"fact.mentioned_at <= ${len(args)}")

        direction = "ASC" if order is RecallOrder.EARLIEST else "DESC"
        args.append(limit)
        where_clause = "\n              AND ".join(conditions)
        rows = await self._pool.fetch(
            f"""
            SELECT
                fact.id AS fact_id,
                fact.source_message_id,
                source.role AS source_role,
                fact.text,
                fact.occurred_start,
                fact.occurred_end,
                fact.mentioned_at,
                fact.entities,
                fact.invalidated_at,
                NULL::double precision AS semantic_score
            FROM facts AS fact
            JOIN messages AS source ON source.id = fact.source_message_id
            WHERE {where_clause}
            ORDER BY {order_column} {direction}, fact.id
            LIMIT ${len(args)}
            """,
            *args,
        )
        return [_candidate_from_record(row) for row in rows]


def _range_conditions(
    alias: str,
    time_range: RecallTimeRange | None,
    args: list[object],
) -> list[str]:
    if time_range is None or not time_range.has_bounds:
        return []
    occurred = [f"{alias}.occurred_start IS NOT NULL"]
    mentioned: list[str] = []
    if time_range.start is not None:
        args.append(time_range.start)
        occurred.append(
            f"COALESCE({alias}.occurred_end, {alias}.occurred_start) >= ${len(args)}"
        )
        args.append(time_range.start)
        mentioned.append(f"{alias}.mentioned_at >= ${len(args)}")
    if time_range.end is not None:
        args.append(time_range.end)
        occurred.append(f"{alias}.occurred_start <= ${len(args)}")
        args.append(time_range.end)
        mentioned.append(f"{alias}.mentioned_at <= ${len(args)}")
    return ["((" + " AND ".join(occurred) + ") OR (" + " AND ".join(mentioned) + "))"]


def _time_conditions(
    alias: str,
    time_filter: RecallTimeFilter | None,
    args: list[object],
) -> list[str]:
    if time_filter is None or time_filter.field is None:
        return []

    def for_field(field: RecallTimeField) -> list[str]:
        conditions: list[str] = []
        if field is RecallTimeField.OCCURRED:
            conditions.append(f"{alias}.occurred_start IS NOT NULL")
            if time_filter.start is not None:
                args.append(time_filter.start)
                conditions.append(
                    f"COALESCE({alias}.occurred_end, {alias}.occurred_start) "
                    f">= ${len(args)}"
                )
            if time_filter.end is not None:
                args.append(time_filter.end)
                conditions.append(f"{alias}.occurred_start <= ${len(args)}")
            return conditions

        if time_filter.start is not None:
            args.append(time_filter.start)
            conditions.append(f"{alias}.mentioned_at >= ${len(args)}")
        if time_filter.end is not None:
            args.append(time_filter.end)
            conditions.append(f"{alias}.mentioned_at <= ${len(args)}")
        return conditions

    if time_filter.field is not RecallTimeField.EITHER:
        return for_field(time_filter.field)
    initial_arg_count = len(args)
    occurred = for_field(RecallTimeField.OCCURRED)
    mentioned = for_field(RecallTimeField.MENTIONED)
    if not occurred or not mentioned:
        del args[initial_arg_count:]
        return []
    return ["((" + " AND ".join(occurred) + ") OR (" + " AND ".join(mentioned) + "))"]


def _append_source_role_condition(
    conditions: list[str],
    args: list[object],
    source_role: RecallSourceRole,
    *,
    source_alias: str,
) -> None:
    if source_role is RecallSourceRole.EITHER:
        return
    args.append(source_role.value)
    conditions.append(f"{source_alias}.role = ${len(args)}")


def _fact_head_from_record(record: Mapping[str, object]) -> RecallFactHead:
    raw_relation_ids = record.get("relation_item_ids", [])
    if not isinstance(raw_relation_ids, Sequence) or isinstance(
        raw_relation_ids,
        (str, bytes),
    ):
        raise ValueError("fact head relation item ids must be an array")
    embedding_text = record["embedding_text"]
    text_hash = record["text_hash"]
    raw_entity_ids = record.get("entity_ids", [])
    if not isinstance(raw_entity_ids, Sequence) or isinstance(
        raw_entity_ids,
        (str, bytes),
    ):
        raise ValueError("fact head entity ids must be an array")
    return RecallFactHead(
        fact=_candidate_from_record(record),
        relation_item_ids=tuple(cast(int, value) for value in raw_relation_ids),
        embedding_text=(embedding_text if isinstance(embedding_text, str) else None),
        text_hash=text_hash if isinstance(text_hash, str) else None,
        entity_ids=tuple(cast(UUID, value) for value in raw_entity_ids),
    )


def _relation_head_from_record(record: Mapping[str, object]) -> RecallRelationHead:
    raw_fact_ids = record["support_fact_ids"]
    if not isinstance(raw_fact_ids, Sequence) or isinstance(
        raw_fact_ids,
        (str, bytes),
    ):
        raise ValueError("relation head support fact ids must be an array")
    return RecallRelationHead(
        id=cast(int, record["snapshot_item_id"]),
        snapshot_type=cast(str, record["snapshot_type"]),
        snapshot_version=cast(int, record["snapshot_version"]),
        snapshot_text=cast(str, record["snapshot_text"]),
        text_hash=cast(str, record["text_hash"]),
        last_supported_at=cast(datetime, record["last_supported_at"]),
        support_fact_ids=tuple(cast(UUID, value) for value in raw_fact_ids),
        support_count=cast(int, record["support_count"]),
        semantic_score=float(cast(float, record["semantic_score"])),
    )


def _mask_embedding_from_record(record: Mapping[str, object]) -> MaskEmbedding:
    raw_entity_ids = record["entity_ids"]
    if not isinstance(raw_entity_ids, Sequence) or isinstance(
        raw_entity_ids,
        (str, bytes),
    ):
        raise ValueError("mask embedding entity ids must be an array")
    vector_text = record["vector_text"]
    if not isinstance(vector_text, str):
        raise ValueError("mask embedding vector must be text")
    stripped = vector_text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError("mask embedding vector text is invalid")
    values = stripped[1:-1]
    vector = () if not values else tuple(float(value) for value in values.split(","))
    return MaskEmbedding(
        object_id=cast(UUID | int, record["object_id"]),
        entity_ids=tuple(cast(UUID, value) for value in raw_entity_ids),
        signature_hash=cast(str, record["signature_hash"]),
        source_text_hash=cast(str, record["source_text_hash"]),
        masked_text=cast(str, record["masked_text"]),
        vector=vector,
    )


def _single_mask_source_from_record(
    kind: str,
    record: Mapping[str, object],
) -> SingleMaskSource:
    raw_aliases = record["aliases"]
    if not isinstance(raw_aliases, Sequence) or isinstance(
        raw_aliases,
        (str, bytes),
    ):
        raise ValueError("single mask source aliases must be an array")
    return SingleMaskSource(
        kind=kind,
        object_id=cast(UUID | int, record["object_id"]),
        fish_id=cast(UUID, record["fish_id"]),
        source_text=cast(str, record["source_text"]),
        source_text_hash=cast(str, record["source_text_hash"]),
        entity=EntityAliases(
            entity_id=cast(UUID, record["entity_id"]),
            aliases=tuple(str(value) for value in raw_aliases),
        ),
    )


def _mask_embedding_row(
    fish_id: UUID,
    space: EmbeddingSpace,
    embedding: MaskEmbedding,
) -> tuple[object, ...]:
    return (
        embedding.object_id,
        fish_id,
        list(embedding.entity_ids),
        embedding.signature_hash,
        embedding.source_text_hash,
        embedding.masked_text,
        space.provider,
        space.model,
        space.dimensions,
        _vector_literal(embedding.vector, dimensions=space.dimensions),
    )


def _candidate_from_record(
    record: Mapping[str, object],
) -> RecallCandidate:
    raw_entities = record["entities"]
    entities = (
        json.loads(raw_entities) if isinstance(raw_entities, str) else raw_entities
    )
    if not isinstance(entities, list) or not all(
        isinstance(entity, str) for entity in entities
    ):
        raise ValueError("fact entities must be a string array")
    semantic_score = record["semantic_score"]
    return RecallCandidate(
        id=cast(UUID, record["fact_id"]),
        source_message_id=cast(UUID, record["source_message_id"]),
        source_role=Role(cast(str, record["source_role"])),
        text=cast(str, record["text"]),
        occurred_start=cast(datetime | None, record["occurred_start"]),
        occurred_end=cast(datetime | None, record["occurred_end"]),
        mentioned_at=cast(datetime, record["mentioned_at"]),
        entities=tuple(cast(list[str], entities)),
        semantic_score=(
            None if semantic_score is None else float(cast(float, semantic_score))
        ),
        invalidated_at=cast(datetime | None, record["invalidated_at"]),
    )


def _vector_literal(vector: Sequence[float], *, dimensions: int) -> str:
    if len(vector) != dimensions:
        raise ValueError(
            f"embedding dimension {len(vector)} does not match expected {dimensions}"
        )
    values = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding values must be finite")
    return "[" + ",".join(str(value) for value in values) + "]"
