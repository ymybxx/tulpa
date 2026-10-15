from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from time import perf_counter_ns
from typing import Protocol, cast
from uuid import UUID

from fish_memory.answer.context import ChatMessage
from fish_memory.embeddings.client import (
    EmbeddingResult,
    EmbeddingSpace,
)
from fish_memory.performance.models import JsonValue
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.recall.masking import (
    EntityAliases,
    MaskEmbedding,
    MaskPlan,
    build_mask_plan,
    cosine_similarity,
)
from fish_memory.recall.models import (
    PlannedRecall,
    RecallCandidate,
    RecalledFact,
    RecallExecutionTrace,
    RecallFactHead,
    RecallQueryPlan,
    RecallRelationHead,
    RecallRelationItem,
    RecallRelationUnit,
    RecallResult,
    RecallSourceRole,
    RecallTimeRange,
)
from fish_memory.recall.ranking import dominant_gap_size

_MEMORY_UNIT_HEAD_CAPACITY = 50
_EXACT_FACT_LIMIT_AT_384_DIMENSIONS = 200_000


def _select_final_memory_units(
    selected: list[tuple[float, float, int, object]],
    *,
    fact_head: Sequence[RecallFactHead],
    mask_champion_relation_ids: set[int],
    mask_champion_fact_ids: set[UUID],
) -> list[tuple[float, float, int, object]]:
    if not selected:
        return []

    gap_size = dominant_gap_size(tuple(item[0] for item in selected))
    has_dominant_gap = gap_size < len(selected)
    relations_by_id: dict[int, RecallRelationUnit] = {}
    standalone_by_id: dict[UUID, RecalledFact] = {}
    relations_by_member_fact_id: dict[UUID, list[RecallRelationUnit]] = {}
    for _score, _secondary, _stable, item in selected:
        if isinstance(item, RecallRelationUnit):
            relations_by_id.setdefault(item.id, item)
            for fact in item.facts:
                relations_by_member_fact_id.setdefault(fact.id, []).append(item)
        else:
            fact = cast(RecalledFact, item)
            standalone_by_id.setdefault(fact.id, fact)

    initial = selected[:gap_size] if has_dominant_gap or len(selected) < 3 else selected
    retained_ids = {
        cast(RecallRelationUnit | RecalledFact, item[3]).id for item in initial
    }
    prefix_relation_ids = {
        item[3].id
        for item in selected[:gap_size]
        if isinstance(item[3], RecallRelationUnit)
    }
    primary_relation_ids = (
        prefix_relation_ids
        if has_dominant_gap
        else (set(relations_by_id) if len(selected) < 3 else set())
    )

    selected_mask_relation_ids = mask_champion_relation_ids.intersection(
        relations_by_id
    )
    retained_ids.update(selected_mask_relation_ids)
    primary_relation_ids.update(selected_mask_relation_ids)

    for fact_id in mask_champion_fact_ids:
        containing_relations = relations_by_member_fact_id.get(fact_id, ())
        if containing_relations:
            relation_ids = {unit.id for unit in containing_relations}
            retained_ids.update(relation_ids)
            primary_relation_ids.update(relation_ids)
        elif fact_id in standalone_by_id:
            retained_ids.add(fact_id)

    if relations_by_id:
        best_relation = max(
            relations_by_id.values(),
            key=lambda unit: (unit.relation_text_score, -unit.id),
        )
        retained_ids.add(best_relation.id)
        primary_relation_ids.add(best_relation.id)

    if fact_head:
        first_fact_id = fact_head[0].fact.id
        containing_relations = relations_by_member_fact_id.get(first_fact_id, ())
        if containing_relations:
            relation_ids = {unit.id for unit in containing_relations}
            retained_ids.update(relation_ids)
            primary_relation_ids.update(relation_ids)
        elif first_fact_id in standalone_by_id:
            retained_ids.add(first_fact_id)

    retained_candidates = [
        item
        for item in selected
        if cast(RecallRelationUnit | RecalledFact, item[3]).id in retained_ids
    ]
    should_truncate = (
        has_dominant_gap and len(retained_candidates) * 2 > _MEMORY_UNIT_HEAD_CAPACITY
    )
    candidates = retained_candidates if should_truncate else selected
    return [
        (
            score,
            secondary,
            stable,
            replace(item, primary=item.id in primary_relation_ids)
            if isinstance(item, RecallRelationUnit)
            else item,
        )
        for score, secondary, stable, item in candidates
    ]


class RecallEmbeddingError(RuntimeError):
    pass


class RecallPlannerProtocol(Protocol):
    async def plan(
        self,
        content: str,
        current_time: datetime,
        *,
        conversation_context: Sequence[ChatMessage] = (),
    ) -> PlannedRecall: ...


class RecallEmbedderProtocol(Protocol):
    @property
    def space(self) -> EmbeddingSpace: ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...

    async def embed_query(self, texts: Sequence[str]) -> EmbeddingResult: ...


class EmbeddingSpaceStoreProtocol(Protocol):
    async def ensure_compatible(
        self,
        fish_id: UUID,
        requested: EmbeddingSpace,
    ) -> None: ...


class RecallStoreProtocol(Protocol):
    async def resolve_query_entities(
        self,
        fish_id: UUID,
        query: str,
    ) -> list[EntityAliases]: ...

    async def load_fact_mask_embeddings(
        self,
        fish_id: UUID,
        dimensions: int,
        fact_ids: Sequence[UUID],
    ) -> list[MaskEmbedding]: ...

    async def load_relation_mask_embeddings(
        self,
        fish_id: UUID,
        dimensions: int,
        relation_item_ids: Sequence[int],
    ) -> list[MaskEmbedding]: ...

    async def upsert_fact_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None: ...

    async def upsert_relation_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None: ...

    async def count_time_scoped_facts(
        self,
        fish_id: UUID,
        dimensions: int,
        time_range: RecallTimeRange,
        *,
        source_role: RecallSourceRole = RecallSourceRole.EITHER,
        stop_after: int,
    ) -> int: ...

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
    ) -> list[RecallFactHead]: ...

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
    ) -> list[RecallRelationHead]: ...

    async def load_relation_item_keys(
        self,
        fish_id: UUID,
        relation_item_ids: Sequence[int],
        dimensions: int,
        vector: Sequence[float],
        *,
        time_range: RecallTimeRange | None,
    ) -> list[RecallRelationHead]: ...

    async def load_fact_keys(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
        dimensions: int,
        vector: Sequence[float],
        *,
        time_range: RecallTimeRange | None,
    ) -> list[RecallFactHead]: ...


class FactRecallService:
    def __init__(
        self,
        store: RecallStoreProtocol,
        planner: RecallPlannerProtocol,
        embedder: RecallEmbedderProtocol,
        embedding_space_store: EmbeddingSpaceStoreProtocol,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        clock: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self._store = store
        self._planner = planner
        self._embedder = embedder
        self._embedding_space = embedder.space
        self._embedding_space_store = embedding_space_store
        self._now = now
        self._clock = clock

    async def recall(
        self,
        fish_id: UUID,
        content: str,
        depth: int,
        *,
        planning_content: str | None = None,
        entity_hints: Sequence[str] = (),
        conversation_context: Sequence[ChatMessage] = (),
        tracker: MessagePerformanceTracker | None = None,
    ) -> RecallResult:
        if type(depth) is not int or depth not in {0, 1}:
            raise ValueError("recall depth must be 0 or 1")
        started_ns = self._clock()
        durations = _empty_durations()
        if depth == 0:
            durations["total"] = _elapsed_ms(started_ns, self._clock())
            return RecallResult(
                facts=(),
                trace=RecallExecutionTrace(
                    depth=0,
                    degraded=False,
                    plan=None,
                    plan_error=None,
                    duration_ms=durations,
                ),
            )

        await self._embedding_space_store.ensure_compatible(
            fish_id,
            self._embedding_space,
        )

        plan: RecallQueryPlan | None = None
        planned: PlannedRecall | None = None
        degraded = False
        plan_error: str | None = None
        query = content
        planner_query = content if planning_content is None else planning_content
        if depth == 1:
            planning_started_ns = self._clock()
            tracker_started_ns = _tracker_now(tracker)
            planning_error: str | None = None
            try:
                planned = await self._planner.plan(
                    planner_query,
                    self._now(),
                    conversation_context=conversation_context,
                )
                plan = planned.plan.model_copy(
                    update={"semantic_query": content},
                )
                planned = replace(planned, plan=plan)
            except Exception as exception:
                degraded = True
                plan_error = type(exception).__name__
                planning_error = plan_error
            finally:
                _record_tracker_step(
                    tracker,
                    "recall.plan",
                    tracker_started_ns,
                    input_payload={
                        "content": planner_query,
                        "semanticQuery": content,
                        "depth": depth,
                        "conversationContext": [
                            dict(message) for message in conversation_context
                        ],
                    },
                    output_payload=(
                        None
                        if planned is None
                        else {
                            "request": planned.request_payload,
                            "response": planned.response_payload,
                        }
                    ),
                    error=planning_error,
                )
            durations["planning"] = _elapsed_ms(
                planning_started_ns,
                self._clock(),
            )

        vector = await self._embed_query(query, durations, tracker)
        return await self._recall_relation_first(
            fish_id,
            query,
            depth,
            vector,
            plan,
            planned,
            degraded,
            plan_error,
            durations,
            started_ns,
            tracker,
        )

    async def _recall_relation_first(
        self,
        fish_id: UUID,
        query: str,
        depth: int,
        vector: Sequence[float],
        plan: RecallQueryPlan | None,
        planned: PlannedRecall | None,
        degraded: bool,
        plan_error: str | None,
        durations: dict[str, float],
        started_ns: int,
        tracker: MessagePerformanceTracker | None,
    ) -> RecallResult:
        source_role = (
            RecallSourceRole.EITHER if plan is None or degraded else plan.source_role
        )
        time_range = (
            plan.time
            if plan is not None and not degraded and plan.time.has_bounds
            else None
        )
        dimensions = self._embedding_space.dimensions
        exact_limit = max(
            1,
            _EXACT_FACT_LIMIT_AT_384_DIMENSIONS * 384 // dimensions,
        )
        count_task = (
            None
            if time_range is None
            else asyncio.create_task(
                self._store.count_time_scoped_facts(
                    fish_id,
                    dimensions,
                    time_range,
                    source_role=source_role,
                    stop_after=exact_limit,
                )
            )
        )
        if count_task is None:
            scoped_fact_count: int | None = None
            use_hnsw = True
        else:
            (count_result,) = await asyncio.gather(count_task, return_exceptions=True)
            if isinstance(count_result, asyncio.CancelledError):
                raise count_result
            scoped_fact_count = (
                None if isinstance(count_result, BaseException) else count_result
            )
            use_hnsw = scoped_fact_count is None or scoped_fact_count > exact_limit

        fact_task = asyncio.create_task(
            self._store.search_fact_head(
                fish_id,
                dimensions,
                vector,
                source_role=source_role,
                time_range=time_range,
                limit=_MEMORY_UNIT_HEAD_CAPACITY,
                use_hnsw=use_hnsw,
            )
        )
        relation_task = asyncio.create_task(
            self._store.search_relation_head(
                fish_id,
                dimensions,
                vector,
                source_role=source_role,
                time_range=time_range,
                limit=_MEMORY_UNIT_HEAD_CAPACITY,
                use_hnsw=use_hnsw,
            )
        )
        fact_result, relation_result = await asyncio.gather(
            fact_task,
            relation_task,
            return_exceptions=True,
        )
        if isinstance(fact_result, asyncio.CancelledError):
            raise fact_result
        if isinstance(relation_result, asyncio.CancelledError):
            raise relation_result
        fact_head = [] if isinstance(fact_result, BaseException) else fact_result
        direct_relation_head = (
            [] if isinstance(relation_result, BaseException) else relation_result
        )

        relations_by_id = {relation.id: relation for relation in direct_relation_head}
        anchored_relation_ids = tuple(
            dict.fromkeys(
                relation_id
                for fact in fact_head
                for relation_id in fact.relation_item_ids
                if relation_id not in relations_by_id
            )
        )
        if anchored_relation_ids:
            try:
                loaded = await self._store.load_relation_item_keys(
                    fish_id,
                    anchored_relation_ids,
                    dimensions,
                    vector,
                    time_range=time_range,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                loaded = []
            relations_by_id.update((relation.id, relation) for relation in loaded)

        all_fact_ids = tuple(
            dict.fromkeys(
                [
                    *(fact.fact.id for fact in fact_head),
                    *(
                        fact_id
                        for relation in relations_by_id.values()
                        for fact_id in relation.support_fact_ids
                    ),
                ]
            )
        )
        try:
            loaded_facts = await self._store.load_fact_keys(
                fish_id,
                all_fact_ids,
                dimensions,
                vector,
                time_range=time_range,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            loaded_facts = []
        facts_by_id = {fact.fact.id: fact for fact in loaded_facts}
        for fact in fact_head:
            facts_by_id.setdefault(fact.fact.id, fact)

        direct_rank = {fact.fact.id: index for index, fact in enumerate(fact_head)}
        relation_units: list[RecallRelationUnit] = []
        valid_relation_fact_ids: set[UUID] = set()
        degraded_fact_ids: set[UUID] = set()
        for relation in relations_by_id.values():
            member_keys = tuple(
                facts_by_id[fact_id]
                for fact_id in relation.support_fact_ids
                if fact_id in facts_by_id
            )
            if not member_keys:
                continue
            if len(member_keys) != relation.support_count:
                degraded_fact_ids.update(
                    member.fact.id
                    for member in member_keys
                    if source_role is RecallSourceRole.EITHER
                    or member.fact.source_role.value == source_role.value
                )
                continue
            anchor_fact_ids = tuple(
                sorted(
                    (
                        fact.fact.id
                        for fact in fact_head
                        if relation.id in fact.relation_item_ids
                        and fact.fact.id in {member.fact.id for member in member_keys}
                    ),
                    key=direct_rank.__getitem__,
                )
            )
            unit_score = max(
                relation.semantic_score,
                *(_candidate_semantic_score(member.fact) for member in member_keys),
            )
            recalled_members = tuple(
                _recalled_fact(
                    member.fact,
                    matched_by=(
                        ("semantic", "relation")
                        if member.fact.id in anchor_fact_ids
                        else ("relation",)
                    ),
                    rank_score=unit_score,
                )
                for member in member_keys
            )
            relation_units.append(
                RecallRelationUnit(
                    id=relation.id,
                    snapshot_type=relation.snapshot_type,
                    snapshot_version=relation.snapshot_version,
                    snapshot_text=relation.snapshot_text,
                    text_hash=relation.text_hash,
                    last_supported_at=relation.last_supported_at,
                    facts=recalled_members,
                    anchor_fact_ids=anchor_fact_ids,
                    relation_text_score=relation.semantic_score,
                    unit_score=unit_score,
                    summary_eligible=True,
                )
            )
            valid_relation_fact_ids.update(member.fact.id for member in member_keys)

        standalone_keys: dict[UUID, RecallFactHead] = {
            fact.fact.id: fact for fact in fact_head
        }
        standalone_keys.update(
            (fact_id, facts_by_id[fact_id])
            for fact_id in degraded_fact_ids
            if fact_id in facts_by_id
        )
        standalone = [
            _recalled_fact(
                fact.fact,
                matched_by=("semantic",),
                rank_score=_candidate_semantic_score(fact.fact),
            )
            for fact in standalone_keys.values()
            if fact.fact.id not in valid_relation_fact_ids
        ]
        ranked_units: list[tuple[float, float, int, object]] = [
            (
                unit.unit_score,
                unit.relation_text_score,
                -unit.id,
                unit,
            )
            for unit in relation_units
        ]
        ranked_units.extend(
            (
                _candidate_semantic_score(fact),
                _candidate_semantic_score(fact),
                -fact.id.int,
                fact,
            )
            for fact in standalone
        )
        ranked_units.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        selected = ranked_units[:_MEMORY_UNIT_HEAD_CAPACITY]
        if fact_head:
            first_fact_head = fact_head[0]
            first_fact_id = first_fact_head.fact.id
            first_fact_is_represented = any(
                (
                    isinstance(item, RecallRelationUnit)
                    and any(fact.id == first_fact_id for fact in item.facts)
                )
                or (isinstance(item, RecalledFact) and item.id == first_fact_id)
                for _score, _secondary, _stable, item in selected
            )
            if not first_fact_is_represented:
                first_fact = next(
                    (fact for fact in standalone if fact.id == first_fact_id),
                    None,
                )
                if first_fact is None:
                    first_fact = _recalled_fact(
                        first_fact_head.fact,
                        matched_by=("semantic",),
                        rank_score=_candidate_semantic_score(first_fact_head.fact),
                    )
                first_fact_score = _candidate_semantic_score(first_fact)
                forced = (
                    first_fact_score,
                    first_fact_score,
                    -first_fact.id.int,
                    first_fact,
                )
                if len(selected) < _MEMORY_UNIT_HEAD_CAPACITY:
                    selected.append(forced)
                else:
                    selected[-1] = forced
                selected.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        try:
            (
                selected,
                mask_champion_relation_ids,
                mask_champion_fact_ids,
            ) = await self._rerank_relation_first_masks(
                fish_id,
                query,
                selected,
                facts_by_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            mask_champion_relation_ids = set()
            mask_champion_fact_ids = set()
        selected = _select_final_memory_units(
            selected,
            fact_head=fact_head,
            mask_champion_relation_ids=mask_champion_relation_ids,
            mask_champion_fact_ids=mask_champion_fact_ids,
        )
        selected_relation_units = tuple(
            item[3] for item in selected if isinstance(item[3], RecallRelationUnit)
        )

        selected_facts: list[RecalledFact] = []
        seen_fact_ids: set[UUID] = set()
        for _score, _secondary, _stable, item in selected:
            facts = item.facts if isinstance(item, RecallRelationUnit) else (item,)
            for fact in cast(Sequence[RecalledFact], facts):
                if fact.id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact.id)
                selected_facts.append(fact)

        durations["total"] = _elapsed_ms(started_ns, self._clock())
        relation_head_by_id = relations_by_id
        return RecallResult(
            facts=tuple(selected_facts),
            trace=RecallExecutionTrace(
                depth=depth,
                degraded=degraded,
                plan=plan,
                plan_error=plan_error,
                duration_ms=durations,
                planner_request_payload=(
                    None if planned is None else planned.request_payload
                ),
                planner_response_payload=(
                    None if planned is None else planned.response_payload
                ),
            ),
            direct_fact_ids=tuple(fact.fact.id for fact in fact_head),
            relation_items=tuple(
                RecallRelationItem(
                    id=unit.id,
                    snapshot_type=unit.snapshot_type,
                    support_fact_ids=relation_head_by_id[unit.id].support_fact_ids,
                    snapshot_version=unit.snapshot_version,
                )
                for unit in selected_relation_units
            ),
            relation_units=selected_relation_units,
            memory_unit_order=tuple(
                cast(RecallRelationUnit | RecalledFact, item[3]).id for item in selected
            ),
        )

    async def _rerank_relation_first_masks(
        self,
        fish_id: UUID,
        query: str,
        selected: list[tuple[float, float, int, object]],
        facts_by_id: dict[UUID, RecallFactHead],
    ) -> tuple[
        list[tuple[float, float, int, object]],
        set[int],
        set[UUID],
    ]:
        query_entities = tuple(await self._store.resolve_query_entities(fish_id, query))
        if not query_entities or not selected:
            return selected, set(), set()

        plans: dict[
            tuple[str, UUID | int],
            tuple[MaskPlan, str],
        ] = {}

        def add_fact_plan(fact_id: UUID) -> None:
            key = facts_by_id.get(fact_id)
            if (
                key is None
                or key.embedding_text is None
                or key.text_hash is None
                or not key.entity_ids
            ):
                return
            plan = build_mask_plan(
                query,
                key.embedding_text,
                query_entities,
                allowed_entity_ids=key.entity_ids,
            )
            if plan is not None:
                plans[("fact", fact_id)] = (plan, key.text_hash)

        for _score, _secondary, _stable, item in selected:
            if isinstance(item, RecallRelationUnit):
                allowed_entity_ids = tuple(
                    dict.fromkeys(
                        entity_id
                        for fact in item.facts
                        for entity_id in facts_by_id[fact.id].entity_ids
                    )
                )
                relation_plan = build_mask_plan(
                    query,
                    item.snapshot_text,
                    query_entities,
                    allowed_entity_ids=allowed_entity_ids,
                )
                if relation_plan is not None:
                    plans[("relation", item.id)] = (
                        relation_plan,
                        item.text_hash,
                    )
                for fact in item.facts:
                    add_fact_plan(fact.id)
            else:
                add_fact_plan(cast(RecalledFact, item).id)
        if not plans:
            return selected, set(), set()

        fact_ids = tuple(
            cast(UUID, object_id) for kind, object_id in plans if kind == "fact"
        )
        relation_ids = tuple(
            cast(int, object_id) for kind, object_id in plans if kind == "relation"
        )
        fact_masks_task = asyncio.create_task(
            self._store.load_fact_mask_embeddings(
                fish_id,
                self._embedding_space.dimensions,
                fact_ids,
            )
        )
        relation_masks_task = asyncio.create_task(
            self._store.load_relation_mask_embeddings(
                fish_id,
                self._embedding_space.dimensions,
                relation_ids,
            )
        )
        stored_fact_masks, stored_relation_masks = await asyncio.gather(
            fact_masks_task,
            relation_masks_task,
        )
        stored_by_key: dict[
            tuple[str, UUID | int, str, str],
            MaskEmbedding,
        ] = {}
        for kind, rows in (
            ("fact", stored_fact_masks),
            ("relation", stored_relation_masks),
        ):
            for row in rows:
                stored_by_key[
                    (
                        kind,
                        row.object_id,
                        row.signature_hash,
                        row.source_text_hash,
                    )
                ] = row

        query_text_by_signature: dict[str, str] = {}
        for plan, _source_hash in plans.values():
            existing = query_text_by_signature.setdefault(
                plan.signature_hash,
                plan.masked_query,
            )
            if existing != plan.masked_query:
                raise ValueError("one mask signature produced multiple query texts")
        signature_order = tuple(query_text_by_signature)
        query_embedding = await self._embedder.embed_query(
            [query_text_by_signature[value] for value in signature_order]
        )
        if len(query_embedding.vectors) != len(signature_order):
            raise ValueError("mask query embedding count does not match input count")
        query_vectors = dict(zip(signature_order, query_embedding.vectors, strict=True))

        vectors_by_key: dict[tuple[str, UUID | int], Sequence[float]] = {}
        missing_by_text: dict[str, list[tuple[str, UUID | int]]] = {}
        for object_key, (plan, source_hash) in plans.items():
            stored = stored_by_key.get(
                (
                    object_key[0],
                    object_key[1],
                    plan.signature_hash,
                    source_hash,
                )
            )
            if stored is not None and stored.entity_ids == plan.signature:
                vectors_by_key[object_key] = stored.vector
                continue
            missing_by_text.setdefault(plan.masked_key, []).append(object_key)

        new_fact_masks: list[MaskEmbedding] = []
        new_relation_masks: list[MaskEmbedding] = []
        if missing_by_text:
            missing_texts = tuple(missing_by_text)
            key_embedding = await self._embedder.embed(missing_texts)
            if len(key_embedding.vectors) != len(missing_texts):
                raise ValueError("mask key embedding count does not match input count")
            for masked_text, vector_value in zip(
                missing_texts,
                key_embedding.vectors,
                strict=True,
            ):
                for object_key in missing_by_text[masked_text]:
                    plan, source_hash = plans[object_key]
                    record = MaskEmbedding(
                        object_id=object_key[1],
                        entity_ids=plan.signature,
                        signature_hash=plan.signature_hash,
                        source_text_hash=source_hash,
                        masked_text=plan.masked_key,
                        vector=tuple(vector_value),
                    )
                    vectors_by_key[object_key] = record.vector
                    if object_key[0] == "fact":
                        new_fact_masks.append(record)
                    else:
                        new_relation_masks.append(record)

        mask_scores: dict[tuple[str, UUID | int], float] = {}
        for object_key, (plan, _source_hash) in plans.items():
            key_vector = vectors_by_key.get(object_key)
            if key_vector is None:
                continue
            mask_scores[object_key] = cosine_similarity(
                query_vectors[plan.signature_hash],
                key_vector,
            )
        champion_relations: dict[str, tuple[float, int]] = {}
        champion_facts: dict[str, tuple[float, UUID]] = {}
        for object_key, score in mask_scores.items():
            plan, _source_hash = plans[object_key]
            if object_key[0] == "relation":
                relation_id = cast(int, object_key[1])
                current = champion_relations.get(plan.signature_hash)
                if current is None or (score, -relation_id) > (
                    current[0],
                    -current[1],
                ):
                    champion_relations[plan.signature_hash] = (
                        score,
                        relation_id,
                    )
            else:
                fact_id = cast(UUID, object_key[1])
                current = champion_facts.get(plan.signature_hash)
                if current is None or (score, -fact_id.int) > (
                    current[0],
                    -current[1].int,
                ):
                    champion_facts[plan.signature_hash] = (score, fact_id)

        persist_tasks: list[asyncio.Task[None]] = []
        if new_fact_masks:
            persist_tasks.append(
                asyncio.create_task(
                    self._store.upsert_fact_mask_embeddings(
                        fish_id,
                        self._embedding_space,
                        new_fact_masks,
                    )
                )
            )
        if new_relation_masks:
            persist_tasks.append(
                asyncio.create_task(
                    self._store.upsert_relation_mask_embeddings(
                        fish_id,
                        self._embedding_space,
                        new_relation_masks,
                    )
                )
            )
        if persist_tasks:
            try:
                await asyncio.gather(*persist_tasks)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

        reranked: list[tuple[float, float, int, object]] = []
        for _score, _secondary, stable, item in selected:
            if isinstance(item, RecallRelationUnit):
                relation_score = max(
                    item.relation_text_score,
                    mask_scores.get(("relation", item.id), float("-inf")),
                )
                member_score = max(
                    max(
                        _candidate_semantic_score(facts_by_id[fact.id].fact),
                        mask_scores.get(("fact", fact.id), float("-inf")),
                    )
                    for fact in item.facts
                )
                unit_score = max(relation_score, member_score)
                matched_members = tuple(
                    replace(fact, rank_score=unit_score) for fact in item.facts
                )
                updated: object = replace(
                    item,
                    facts=matched_members,
                    relation_text_score=relation_score,
                    unit_score=unit_score,
                )
                reranked.append((unit_score, relation_score, stable, updated))
                continue
            fact = cast(RecalledFact, item)
            score = max(
                _candidate_semantic_score(facts_by_id[fact.id].fact),
                mask_scores.get(("fact", fact.id), float("-inf")),
            )
            reranked.append((score, score, stable, replace(fact, rank_score=score)))
        reranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        return (
            reranked,
            {value[1] for value in champion_relations.values()},
            {value[1] for value in champion_facts.values()},
        )

    async def _embed_query(
        self,
        query: str,
        durations: dict[str, float],
        tracker: MessagePerformanceTracker | None,
    ) -> Sequence[float]:
        embedding_started_ns = self._clock()
        tracker_started_ns = _tracker_now(tracker)
        embedding: EmbeddingResult | None = None
        embedding_error: str | None = None
        try:
            embedding = await self._embedder.embed_query([query])
        except Exception as exception:
            embedding_error = type(exception).__name__
            raise RecallEmbeddingError("embedding unavailable") from exception
        finally:
            _record_tracker_step(
                tracker,
                "recall.embedding",
                tracker_started_ns,
                input_payload={
                    "provider": self._embedding_space.provider,
                    "model": self._embedding_space.model,
                    "dimensions": self._embedding_space.dimensions,
                    "texts": [query],
                },
                output_payload=(
                    None
                    if embedding is None
                    else {
                        "vector_count": len(embedding.vectors),
                        "input_tokens": embedding.input_tokens,
                    }
                ),
                error=embedding_error,
            )
        durations["embedding"] = _elapsed_ms(
            embedding_started_ns,
            self._clock(),
        )
        if len(embedding.vectors) != 1:
            raise RecallEmbeddingError("embedding unavailable")
        return embedding.vectors[0]


def _tracker_now(tracker: MessagePerformanceTracker | None) -> int | None:
    return None if tracker is None else tracker.now_ns()


def _record_tracker_step(
    tracker: MessagePerformanceTracker | None,
    name: str,
    started_ns: int | None,
    *,
    input_payload: JsonValue = None,
    output_payload: JsonValue = None,
    error: str | None = None,
) -> None:
    if tracker is None or started_ns is None:
        return
    tracker.record(
        name,
        started_ns=started_ns,
        finished_ns=tracker.now_ns(),
        input_payload=input_payload,
        output_payload=output_payload,
        error=error,
    )


def _empty_durations() -> dict[str, float]:
    return {
        "planning": 0.0,
        "embedding": 0.0,
        "semanticSearch": 0.0,
        "entitySearch": 0.0,
        "temporalSearch": 0.0,
        "merge": 0.0,
        "relationSearch": 0.0,
        "relatedFactLoad": 0.0,
        "total": 0.0,
    }


def _elapsed_ms(started_ns: int, finished_ns: int) -> float:
    return max(0.0, (finished_ns - started_ns) / 1_000_000)


def _candidate_semantic_score(
    candidate: RecallCandidate | RecalledFact,
) -> float:
    return (
        float("-inf") if candidate.semantic_score is None else candidate.semantic_score
    )


def _recalled_fact(
    candidate: RecallCandidate,
    *,
    matched_by: tuple[str, ...],
    rank_score: float | None,
) -> RecalledFact:
    return RecalledFact(
        id=candidate.id,
        source_message_id=candidate.source_message_id,
        source_role=candidate.source_role,
        text=candidate.text,
        occurred_start=candidate.occurred_start,
        occurred_end=candidate.occurred_end,
        mentioned_at=candidate.mentioned_at,
        entities=candidate.entities,
        matched_by=matched_by,
        semantic_score=candidate.semantic_score,
        rank_score=rank_score,
        invalidated_at=candidate.invalidated_at,
    )
