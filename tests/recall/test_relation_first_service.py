import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.llm.client import JsonCompletion
from fish_memory.models import Role
from fish_memory.performance.models import JsonValue
from fish_memory.recall.masking import (
    EntityAliases,
    MaskEmbedding,
    entity_signature,
    signature_hash,
)
from fish_memory.recall.models import (
    PlannedRecall,
    RecallCandidate,
    RecallFactHead,
    RecallQueryPlan,
    RecallRelationHead,
    RecallSourceRole,
    RecallTimeRange,
)
from fish_memory.recall.planner import RecallPlanner
from fish_memory.recall.service import FactRecallService, RecallStoreProtocol

FISH_ID = UUID(int=1)
A = UUID(int=10)
B = UUID(int=11)
C = UUID(int=12)
NOW = datetime(2026, 8, 26, tzinfo=UTC)
SPACE = EmbeddingSpace("local", "test", 384)
VECTOR = (0.25,) * 384
MASK_VECTOR = (1.0,) + (0.0,) * 383
ENTITY_ID = UUID(int=50)


def candidate(
    value: UUID,
    score: float,
    text: str,
    source_role: Role = Role.USER,
) -> RecallCandidate:
    return RecallCandidate(
        id=value,
        source_message_id=UUID(int=100 + value.int),
        source_role=source_role,
        text=text,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW,
        entities=(),
        semantic_score=score,
        invalidated_at=None,
    )


def fact_head(
    value: UUID,
    score: float,
    text: str,
    relation_item_ids: tuple[int, ...],
    entity_ids: tuple[UUID, ...] = (),
    source_role: Role = Role.USER,
) -> RecallFactHead:
    return RecallFactHead(
        fact=candidate(value, score, text, source_role),
        relation_item_ids=relation_item_ids,
        embedding_text=text,
        text_hash=f"{value.int:064x}",
        entity_ids=entity_ids,
    )


def relation_head(score: float) -> RecallRelationHead:
    return RecallRelationHead(
        id=7,
        snapshot_type="dynamic",
        snapshot_version=3,
        snapshot_text="西装待取、新靴待取、旧靴待退",
        text_hash="a" * 64,
        last_supported_at=NOW,
        support_fact_ids=(A, B),
        support_count=2,
        semantic_score=score,
    )


class FakeEmbedder:
    space = SPACE

    def __init__(self) -> None:
        self.query_calls: list[tuple[str, ...]] = []
        self.key_calls: list[tuple[str, ...]] = []

    async def embed_query(self, texts: Sequence[str]) -> EmbeddingResult:
        self.query_calls.append(tuple(texts))
        vectors = tuple(
            MASK_VECTOR if text.startswith("以前") else VECTOR for text in texts
        )
        return EmbeddingResult(vectors, 4)

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.key_calls.append(tuple(texts))
        return EmbeddingResult(tuple(MASK_VECTOR for _text in texts), 4)


class FakePlanner:
    def __init__(
        self,
        time_range: RecallTimeRange | None = None,
        source_role: RecallSourceRole = RecallSourceRole.EITHER,
    ) -> None:
        self.time_range = time_range or RecallTimeRange(
            start=None,
            end=None,
        )
        self.source_role = source_role
        self.failure: Exception | None = None
        self.calls = 0
        self.contents: list[str] = []

    async def plan(
        self,
        content: str,
        current_time: datetime,
        *,
        conversation_context: Sequence[ChatMessage] = (),
    ) -> PlannedRecall:
        del current_time, conversation_context
        self.calls += 1
        self.contents.append(content)
        if self.failure is not None:
            raise self.failure
        return PlannedRecall(
            plan=RecallQueryPlan(
                semanticQuery="planner must not replace the query",
                sourceRole=self.source_role,
                time=self.time_range,
            ),
            request_payload={},
            response_payload={},
        )


class MalformedTimeJsonClient:
    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        del messages
        return JsonCompletion(
            '{"sourceRole":"either","entityHints":[],"time":{"start":0,"end":null}}'
        )

    def json_request_payload(self, messages: Sequence[ChatMessage]) -> JsonValue:
        return {"messages": [dict(message) for message in messages]}


class FakeSpaceStore:
    def __init__(self) -> None:
        self.calls = 0
        self.failure: Exception | None = None

    async def ensure_compatible(
        self,
        fish_id: UUID,
        requested: EmbeddingSpace,
    ) -> None:
        assert (fish_id, requested) == (FISH_ID, SPACE)
        self.calls += 1
        if self.failure is not None:
            raise self.failure


class FakeStore:
    def __init__(self) -> None:
        self.fact_heads = [
            fact_head(A, 0.88, "需要取西装", (7,)),
            fact_head(B, 0.86, "需要取新靴并退旧靴", (7,)),
            fact_head(C, 0.80, "喜欢银剑", ()),
        ]
        self.hydrated_fact_heads: list[RecallFactHead] | None = None
        self.loaded_relation = relation_head(0.70)
        self.loaded_relations: list[RecallRelationHead] | None = None
        self.fact_search_calls: list[tuple[int, bool, RecallSourceRole]] = []
        self.relation_search_calls: list[tuple[int, bool, RecallSourceRole]] = []
        self.loaded_relation_item_ids: list[tuple[int, ...]] = []
        self.query_entities: list[EntityAliases] = []
        self.persisted_fact_masks: list[MaskEmbedding] = []
        self.persisted_relation_masks: list[MaskEmbedding] = []
        self.direct_relations: list[RecallRelationHead] = []
        self.fact_search_error: Exception | None = None
        self.relation_search_error: Exception | None = None
        self.time_range: RecallTimeRange | None = None
        self.time_count = 0
        self.count_calls: list[tuple[int, RecallSourceRole]] = []
        self.relation_load_error: Exception | None = None
        self.time_count_error: BaseException | None = None
        self.fact_head_started: asyncio.Event | None = None
        self.relation_head_started: asyncio.Event | None = None
        self.head_search_release: asyncio.Event | None = None

    async def search_fact_head(
        self,
        fish_id: UUID,
        dimensions: int,
        vector: Sequence[float],
        *,
        source_role: RecallSourceRole,
        time_range: RecallTimeRange | None,
        limit: int,
        use_hnsw: bool,
    ) -> list[RecallFactHead]:
        if self.fact_head_started is not None:
            self.fact_head_started.set()
            assert self.head_search_release is not None
            await self.head_search_release.wait()
        assert (fish_id, dimensions, tuple(vector), time_range) == (
            FISH_ID,
            384,
            VECTOR,
            self.time_range,
        )
        self.fact_search_calls.append((limit, use_hnsw, source_role))
        if self.fact_search_error is not None:
            raise self.fact_search_error
        return self.fact_heads

    async def search_relation_head(
        self,
        fish_id: UUID,
        dimensions: int,
        vector: Sequence[float],
        *,
        source_role: RecallSourceRole,
        time_range: RecallTimeRange | None,
        limit: int,
        use_hnsw: bool,
    ) -> list[RecallRelationHead]:
        if self.relation_head_started is not None:
            self.relation_head_started.set()
            assert self.head_search_release is not None
            await self.head_search_release.wait()
        assert (fish_id, dimensions, tuple(vector), time_range) == (
            FISH_ID,
            384,
            VECTOR,
            self.time_range,
        )
        self.relation_search_calls.append((limit, use_hnsw, source_role))
        if self.relation_search_error is not None:
            raise self.relation_search_error
        return self.direct_relations

    async def load_relation_item_keys(
        self,
        fish_id: UUID,
        relation_item_ids: Sequence[int],
        dimensions: int,
        vector: Sequence[float],
        *,
        time_range: RecallTimeRange | None,
    ) -> list[RecallRelationHead]:
        assert (fish_id, dimensions, tuple(vector), time_range) == (
            FISH_ID,
            384,
            VECTOR,
            self.time_range,
        )
        self.loaded_relation_item_ids.append(tuple(relation_item_ids))
        if self.relation_load_error is not None:
            raise self.relation_load_error
        return (
            [self.loaded_relation]
            if self.loaded_relations is None
            else self.loaded_relations
        )

    async def count_time_scoped_facts(
        self,
        fish_id: UUID,
        dimensions: int,
        time_range: RecallTimeRange,
        *,
        source_role: RecallSourceRole,
        stop_after: int,
    ) -> int:
        assert (fish_id, dimensions, time_range) == (
            FISH_ID,
            384,
            self.time_range,
        )
        self.count_calls.append((stop_after, source_role))
        if self.time_count_error is not None:
            raise self.time_count_error
        return self.time_count

    async def load_fact_keys(
        self,
        fish_id: UUID,
        fact_ids: Sequence[UUID],
        dimensions: int,
        vector: Sequence[float],
        *,
        time_range: RecallTimeRange | None,
    ) -> list[RecallFactHead]:
        assert (fish_id, dimensions, tuple(vector), time_range) == (
            FISH_ID,
            384,
            VECTOR,
            self.time_range,
        )
        requested = set(fact_ids)
        rows = (
            self.fact_heads
            if self.hydrated_fact_heads is None
            else self.hydrated_fact_heads
        )
        return [
            replace(item, relation_item_ids=())
            for item in rows
            if item.fact.id in requested
        ]

    async def resolve_query_entities(
        self,
        fish_id: UUID,
        query: str,
    ) -> list[EntityAliases]:
        del fish_id, query
        return self.query_entities

    async def load_fact_mask_embeddings(
        self,
        fish_id: UUID,
        dimensions: int,
        fact_ids: Sequence[UUID],
    ) -> list[MaskEmbedding]:
        del fish_id, dimensions
        requested = set(fact_ids)
        return [row for row in self.persisted_fact_masks if row.object_id in requested]

    async def load_relation_mask_embeddings(
        self,
        fish_id: UUID,
        dimensions: int,
        relation_item_ids: Sequence[int],
    ) -> list[MaskEmbedding]:
        del fish_id, dimensions
        requested = set(relation_item_ids)
        return [
            row for row in self.persisted_relation_masks if row.object_id in requested
        ]

    async def upsert_fact_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None:
        assert (fish_id, space) == (FISH_ID, SPACE)
        self.persisted_fact_masks.extend(embeddings)

    async def upsert_relation_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None:
        assert (fish_id, space) == (FISH_ID, SPACE)
        self.persisted_relation_masks.extend(embeddings)


@pytest.mark.asyncio
async def test_relation_first_recall_uses_fact_keys_to_pull_relation_unit() -> None:
    store = FakeStore()
    embedder = FakeEmbedder()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(
        FISH_ID,
        "How many clothing items need pickup or return?",
        1,
    )

    assert embedder.query_calls == [("How many clothing items need pickup or return?",)]
    assert store.count_calls == []
    assert store.relation_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.fact_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.loaded_relation_item_ids == [(7,)]
    assert len(result.relation_units) == 1
    assert result.relation_units[0].id == 7
    assert result.relation_units[0].unit_score == pytest.approx(0.88)
    assert result.relation_units[0].anchor_fact_ids == (A, B)
    assert [fact.id for fact in result.facts] == [A, B, C]


@pytest.mark.asyncio
async def test_planner_uses_original_content_while_embedding_uses_generated_query() -> (
    None
):
    planner = FakePlanner()
    embedder = FakeEmbedder()
    service = FactRecallService(
        FakeStore(),  # type: ignore[arg-type]
        planner,
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(
        FISH_ID,
        "用户上周（2026年8月19日至25日）的锻炼时长",
        1,
        planning_content="How long did I exercise last week?",
    )

    assert planner.contents == ["How long did I exercise last week?"]
    assert embedder.query_calls == [("用户上周（2026年8月19日至25日）的锻炼时长",)]
    assert result.trace.plan is not None
    assert result.trace.plan.semantic_query == (
        "用户上周（2026年8月19日至25日）的锻炼时长"
    )


@pytest.mark.asyncio
async def test_fact_and_relation_heads_start_before_either_is_released() -> None:
    store = FakeStore()
    store.fact_head_started = asyncio.Event()
    store.relation_head_started = asyncio.Event()
    store.head_search_release = asyncio.Event()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    recall_task = asyncio.create_task(service.recall(FISH_ID, "query", 1))
    try:
        await asyncio.wait_for(
            asyncio.gather(
                store.fact_head_started.wait(),
                store.relation_head_started.wait(),
            ),
            timeout=1.0,
        )
    except TimeoutError:
        store.head_search_release.set()
        await asyncio.wait_for(recall_task, timeout=1.0)
        raise

    store.head_search_release.set()
    await asyncio.wait_for(recall_task, timeout=1.0)

    assert store.fact_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.relation_search_calls == [(50, True, RecallSourceRole.EITHER)]


@pytest.mark.asyncio
async def test_recall_rejects_depth_two() -> None:
    service = FactRecallService(
        cast(RecallStoreProtocol, FakeStore()),
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    with pytest.raises(ValueError, match="recall depth must be 0 or 1"):
        await service.recall(FISH_ID, "query", 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [True, False, 0.0, 1.0])
async def test_recall_rejects_non_strict_integer_depth(depth: object) -> None:
    service = FactRecallService(
        cast(RecallStoreProtocol, FakeStore()),
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    with pytest.raises(ValueError, match="recall depth must be 0 or 1"):
        await service.recall(FISH_ID, "query", cast(int, depth))


@pytest.mark.asyncio
async def test_depth_zero_skips_planner_embedding_and_store() -> None:
    store = FakeStore()
    planner = FakePlanner()
    embedder = FakeEmbedder()
    space_store = FakeSpaceStore()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        planner,
        embedder,
        space_store,
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "query", 0)

    assert result.facts == ()
    assert planner.calls == 0
    assert embedder.query_calls == []
    assert space_store.calls == 0
    assert store.fact_search_calls == []


@pytest.mark.asyncio
async def test_embedding_space_failure_stops_before_planning() -> None:
    store = FakeStore()
    planner = FakePlanner()
    embedder = FakeEmbedder()
    space_store = FakeSpaceStore()
    space_store.failure = RuntimeError("embedding space mismatch")
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        planner,
        embedder,
        space_store,
        now=lambda: NOW,
    )

    with pytest.raises(RuntimeError, match="embedding space mismatch"):
        await service.recall(FISH_ID, "query", 1)

    assert planner.calls == 0
    assert embedder.query_calls == []


@pytest.mark.asyncio
async def test_planner_failure_degrades_to_unscoped_full_query() -> None:
    store = FakeStore()
    planner = FakePlanner()
    planner.failure = RuntimeError("planner unavailable")
    embedder = FakeEmbedder()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        planner,
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "original query", 1)

    assert result.trace.degraded is True
    assert result.trace.plan_error == "RuntimeError"
    assert embedder.query_calls == [("original query",)]
    assert store.fact_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.relation_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.count_calls == []


@pytest.mark.asyncio
async def test_malformed_planner_time_degrades_to_unscoped_full_query() -> None:
    store = FakeStore()
    embedder = FakeEmbedder()
    service = FactRecallService(
        cast(RecallStoreProtocol, store),
        RecallPlanner(MalformedTimeJsonClient()),
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "original query", 1)

    assert result.trace.degraded is True
    assert result.trace.plan is None
    assert result.trace.plan_error == "ValidationError"
    assert embedder.query_calls == [("original query",)]
    assert store.count_calls == []
    assert store.fact_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.relation_search_calls == [(50, True, RecallSourceRole.EITHER)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_role",
    [
        RecallSourceRole.USER,
        RecallSourceRole.ASSISTANT,
        RecallSourceRole.EITHER,
    ],
)
async def test_plan_source_role_scopes_parallel_heads_and_time_count(
    source_role: RecallSourceRole,
) -> None:
    time_range = RecallTimeRange(start=NOW, end=None)
    store = FakeStore()
    store.time_range = time_range
    store.time_count = 100
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(time_range, source_role=source_role),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    await service.recall(FISH_ID, "source scoped query", 1)

    assert store.count_calls == [(200_000, source_role)]
    assert store.fact_search_calls == [(50, False, source_role)]
    assert store.relation_search_calls == [(50, False, source_role)]


@pytest.mark.asyncio
async def test_bounded_range_keeps_generic_mention_only_luxury_and_twins_facts() -> (
    None
):
    """Catches the service narrowing mention-only facts to occurrence time."""
    time_range = RecallTimeRange(
        start=datetime(2026, 4, 1, tzinfo=UTC),
        end=datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC),
    )
    store = FakeStore()
    store.time_range = time_range
    store.time_count = 2
    store.fact_heads = [
        fact_head(A, 0.92, "用户购买奢华晚礼服花费800美元", ()),
        fact_head(B, 0.91, "Ava 和 Lily 出生于四月", ()),
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(time_range, source_role=RecallSourceRole.USER),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "最近的奢侈品和新生儿", 1)

    assert [fact.id for fact in result.facts] == [A, B]
    assert result.memory_unit_order == (A, B)
    assert store.fact_search_calls == [(50, False, RecallSourceRole.USER)]
    assert store.relation_search_calls == [(50, False, RecallSourceRole.USER)]


@pytest.mark.asyncio
async def test_source_qualified_mixed_relation_keeps_all_hydrated_members() -> None:
    store = FakeStore()
    store.fact_heads = [
        fact_head(A, 0.88, "用户说要取西装", (7,), source_role=Role.USER),
    ]
    store.hydrated_fact_heads = [
        fact_head(A, 0.88, "用户说要取西装", (), source_role=Role.USER),
        fact_head(B, 0.86, "助手提醒新靴待取", (7,), source_role=Role.ASSISTANT),
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(source_role=RecallSourceRole.USER),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "用户需要取什么？", 1)

    assert store.relation_search_calls == [(50, True, RecallSourceRole.USER)]
    assert [fact.id for fact in result.relation_units[0].facts] == [A, B]


@pytest.mark.asyncio
async def test_underfilled_source_scope_does_not_fetch_the_other_source() -> None:
    store = FakeStore()
    store.fact_heads = [fact_head(A, 0.88, "用户事实", ())]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(source_role=RecallSourceRole.USER),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "用户说过什么？", 1)

    assert [fact.id for fact in result.facts] == [A]
    assert store.count_calls == []
    assert store.fact_search_calls == [(50, True, RecallSourceRole.USER)]
    assert store.relation_search_calls == [(50, True, RecallSourceRole.USER)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source_role", "fact_head_rows", "expected_fact_ids"),
    [
        (
            RecallSourceRole.USER,
            [fact_head(A, 0.88, "用户事实", (7,), source_role=Role.USER)],
            [A],
        ),
        (
            RecallSourceRole.ASSISTANT,
            [fact_head(B, 0.86, "助手事实", (7,), source_role=Role.ASSISTANT)],
            [B],
        ),
        (
            RecallSourceRole.EITHER,
            [fact_head(A, 0.88, "用户事实", (7,), source_role=Role.USER)],
            [A, B],
        ),
    ],
)
async def test_incomplete_relation_degradation_respects_source_scope(
    source_role: RecallSourceRole,
    fact_head_rows: list[RecallFactHead],
    expected_fact_ids: list[UUID],
) -> None:
    store = FakeStore()
    store.fact_heads = fact_head_rows
    store.hydrated_fact_heads = [
        fact_head(A, 0.88, "用户事实", (), source_role=Role.USER),
        fact_head(B, 0.86, "助手事实", (), source_role=Role.ASSISTANT),
    ]
    store.loaded_relation = replace(
        store.loaded_relation,
        support_fact_ids=(A, B, C),
        support_count=3,
    )
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(source_role=source_role),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "source-scoped incomplete relation", 1)

    assert result.relation_units == ()
    assert [fact.id for fact in result.facts] == expected_fact_ids


@pytest.mark.asyncio
async def test_relation_first_recall_reranks_and_persists_dual_sided_mask() -> None:
    store = FakeStore()
    store.fact_heads = [
        fact_head(
            A,
            0.70,
            "杰洛特以前使用过银剑",
            (7,),
            entity_ids=(ENTITY_ID,),
        ),
        fact_head(B, 0.69, "银剑被收在仓库", (7,)),
        fact_head(C, 0.80, "希尔以前使用过银剑", ()),
    ]
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    embedder = FakeEmbedder()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert result.relation_units[0].unit_score == pytest.approx(1.0)
    assert [fact.id for fact in result.facts] == [A, B, C]
    assert embedder.query_calls == [
        ("白狼以前用过什么剑",),
        ("以前用过什么剑",),
    ]
    assert embedder.key_calls == [("以前使用过银剑",)]
    assert len(store.persisted_fact_masks) == 1
    persisted = store.persisted_fact_masks[0]
    assert persisted.object_id == A
    assert persisted.entity_ids == (ENTITY_ID,)
    assert persisted.masked_text == "以前使用过银剑"
    assert store.persisted_relation_masks == []


@pytest.mark.asyncio
async def test_persisted_multi_mask_avoids_reembedding_the_key() -> None:
    store = FakeStore()
    store.fact_heads = [
        fact_head(
            A,
            0.70,
            "杰洛特以前使用过银剑",
            (),
            entity_ids=(ENTITY_ID,),
        )
    ]
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    embedder = FakeEmbedder()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    await service.recall(FISH_ID, "白狼以前用过什么剑", 1)
    await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert embedder.key_calls == [("以前使用过银剑",)]
    assert len(store.persisted_fact_masks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scoped_count", "expected_hnsw"),
    [(200_000, False), (200_001, True)],
)
async def test_time_scope_switches_exact_at_the_dimension_scaled_boundary(
    scoped_count: int,
    expected_hnsw: bool,
) -> None:
    time_range = RecallTimeRange(start=NOW, end=None)
    store = FakeStore()
    store.time_range = time_range
    store.time_count = scoped_count
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(time_range),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    await service.recall(FISH_ID, "那时候发生了什么？", 1)

    assert store.count_calls == [(200_000, RecallSourceRole.EITHER)]
    assert store.fact_search_calls == [(50, expected_hnsw, RecallSourceRole.EITHER)]
    assert store.relation_search_calls == [(50, expected_hnsw, RecallSourceRole.EITHER)]


@pytest.mark.asyncio
async def test_relation_head_failure_still_uses_fact_to_relation_mapping() -> None:
    store = FakeStore()
    store.relation_search_error = RuntimeError("relation hnsw unavailable")
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "clothing", 1)

    assert [unit.id for unit in result.relation_units] == [7]
    assert store.loaded_relation_item_ids == [(7,)]


@pytest.mark.asyncio
async def test_fact_head_failure_still_returns_direct_relation_head() -> None:
    store = FakeStore()
    store.fact_search_error = RuntimeError("fact hnsw unavailable")
    store.direct_relations = [store.loaded_relation]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "clothing", 1)

    assert [unit.id for unit in result.relation_units] == [7]
    assert [fact.id for fact in result.facts] == [A, B]


@pytest.mark.asyncio
async def test_truncated_time_count_keeps_fixed_fact_head_limit() -> None:
    time_range = RecallTimeRange(start=NOW, end=None)
    store = FakeStore()
    store.time_range = time_range
    store.time_count = 200_001
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(time_range),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    await service.recall(FISH_ID, "那时候发生了什么？", 1)

    assert store.count_calls == [(200_000, RecallSourceRole.EITHER)]
    assert store.fact_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.relation_search_calls == [(50, True, RecallSourceRole.EITHER)]


@pytest.mark.asyncio
async def test_fact_anchored_relation_load_failure_falls_back_to_facts() -> None:
    store = FakeStore()
    store.relation_load_error = RuntimeError("relation load unavailable")
    store.fact_heads = [
        fact_head(A, 0.88, "需要取西装", (7,)),
        fact_head(B, 0.87, "需要取新靴并退旧靴", (7,)),
        fact_head(C, 0.86, "喜欢银剑", ()),
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "clothing", 1)

    assert result.relation_units == ()
    assert [fact.id for fact in result.facts] == [A, B, C]


@pytest.mark.asyncio
async def test_time_count_failure_uses_conservative_hnsw_defaults() -> None:
    time_range = RecallTimeRange(start=NOW, end=None)
    store = FakeStore()
    store.time_range = time_range
    store.time_count_error = RuntimeError("time count unavailable")
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(time_range),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "那时候发生了什么？", 1)

    assert result.facts
    assert store.fact_search_calls == [(50, True, RecallSourceRole.EITHER)]
    assert store.relation_search_calls == [(50, True, RecallSourceRole.EITHER)]


@pytest.mark.asyncio
async def test_time_count_cancellation_propagates_before_heads_start() -> None:
    time_range = RecallTimeRange(start=NOW, end=None)
    store = FakeStore()
    store.time_range = time_range
    store.time_count_error = asyncio.CancelledError()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(time_range),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            service.recall(FISH_ID, "bounded query", 1),
            timeout=1.0,
        )

    assert store.count_calls == [(200_000, RecallSourceRole.EITHER)]
    assert store.fact_search_calls == []
    assert store.relation_search_calls == []


@pytest.mark.asyncio
async def test_underfilled_direct_relation_degrades_eligible_member_to_fact() -> None:
    store = FakeStore()
    store.fact_search_error = RuntimeError("fact head unavailable")
    store.direct_relations = [store.loaded_relation]
    store.fact_heads = [fact_head(A, 0.74, "需要取西装", (7,))]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "clothing", 1)

    assert result.relation_units == ()
    assert [fact.id for fact in result.facts] == [A]


@pytest.mark.asyncio
async def test_complete_single_support_item_remains_a_relation_unit() -> None:
    store = FakeStore()
    store.fact_heads = [fact_head(A, 0.74, "需要取西装", (7,))]
    store.loaded_relation = replace(
        store.loaded_relation,
        support_fact_ids=(A,),
        support_count=1,
        semantic_score=0.91,
    )
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "clothing", 1)

    assert len(result.relation_units) == 1
    assert result.relation_units[0].summary_eligible is True
    assert result.relation_units[0].unit_score == pytest.approx(0.91)
    assert [fact.id for fact in result.relation_units[0].facts] == [A]
    assert result.memory_unit_order == (7,)


@pytest.mark.asyncio
async def test_partial_multi_support_item_never_uses_summary_score() -> None:
    store = FakeStore()
    store.fact_heads = [
        fact_head(A, 0.74, "需要取西装", (7,)),
        fact_head(B, 0.72, "需要取新靴", (7,)),
    ]
    store.loaded_relation = replace(
        store.loaded_relation,
        support_fact_ids=(A, B, C),
        support_count=3,
        semantic_score=0.99,
    )
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "clothing", 1)

    assert result.relation_units == ()
    assert result.memory_unit_order == (A, B)
    assert [fact.rank_score for fact in result.facts] == [0.74, 0.72]


@pytest.mark.asyncio
async def test_relation_with_zero_eligible_support_is_dropped() -> None:
    store = FakeStore()
    store.fact_heads = []
    store.direct_relations = [store.loaded_relation]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "clothing", 1)

    assert result.relation_units == ()
    assert result.facts == ()
    assert result.memory_unit_order == ()


@pytest.mark.asyncio
async def test_relation_mask_ignores_query_entity_absent_from_eligible_members() -> (
    None
):
    store = FakeStore()
    store.loaded_relation = replace(
        store.loaded_relation,
        snapshot_text="杰洛特以前使用过银剑",
    )
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    signature = entity_signature((ENTITY_ID,))
    store.persisted_relation_masks = [
        MaskEmbedding(
            object_id=7,
            entity_ids=signature,
            signature_hash=signature_hash(signature),
            source_text_hash="a" * 64,
            masked_text="以前使用过银剑",
            vector=MASK_VECTOR,
        )
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert len(store.persisted_relation_masks) == 1
    assert result.relation_units[0].unit_score == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_relation_mask_uses_entity_union_from_eligible_members() -> None:
    store = FakeStore()
    store.fact_heads = [
        fact_head(A, 0.70, "银剑已使用", (7,), entity_ids=(ENTITY_ID,)),
        fact_head(B, 0.69, "银剑在仓库", (7,)),
    ]
    store.loaded_relation = replace(
        store.loaded_relation,
        snapshot_text="杰洛特以前使用过银剑",
    )
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    embedder = FakeEmbedder()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert len(store.persisted_relation_masks) == 1
    assert store.persisted_relation_masks[0].object_id == 7
    assert store.persisted_relation_masks[0].entity_ids == (ENTITY_ID,)
    assert store.persisted_relation_masks[0].masked_text == "以前使用过银剑"


@pytest.mark.asyncio
async def test_current_hash_relation_mask_is_reused_after_support_only_update() -> None:
    store = FakeStore()
    store.fact_heads = [
        fact_head(A, 0.70, "银剑已使用", (7,), entity_ids=(ENTITY_ID,)),
        fact_head(B, 0.69, "银剑在仓库", (7,)),
    ]
    store.loaded_relation = replace(
        store.loaded_relation,
        snapshot_text="杰洛特以前使用过银剑",
    )
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    signature = entity_signature((ENTITY_ID,))
    store.persisted_relation_masks = [
        MaskEmbedding(
            object_id=7,
            entity_ids=signature,
            signature_hash=signature_hash(signature),
            source_text_hash="a" * 64,
            masked_text="以前使用过银剑",
            vector=MASK_VECTOR,
        )
    ]
    embedder = FakeEmbedder()
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        embedder,
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert result.relation_units[0].unit_score == pytest.approx(1.0)
    assert embedder.key_calls == []
    assert len(store.persisted_relation_masks) == 1


@pytest.mark.asyncio
async def test_full_heads_are_capped_before_mask_reranking() -> None:
    store = FakeStore()
    store.fact_heads = [
        fact_head(UUID(int=1_000 + index), 1.0 - index / 100, str(index), ())
        for index in range(51)
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "query", 1)

    assert len(result.memory_unit_order) == 50
    assert len(result.facts) == 50


@pytest.mark.asyncio
async def test_fact_head_rank_one_survives_below_provisional_relation_cap() -> None:
    store = FakeStore()
    support_fact_ids = [UUID(int=3_000 + index) for index in range(50)]
    store.fact_heads = [
        fact_head(A, 0.90, "rank one fact", (201,)),
        fact_head(B, 0.89, "rank two fact", (200,)),
    ]
    store.hydrated_fact_heads = [
        fact_head(A, 0.90, "rank one fact", ()),
        fact_head(B, 0.89, "rank two fact", ()),
        *[
            fact_head(fact_id, 0.20, f"direct support {index}", ())
            for index, fact_id in enumerate(support_fact_ids)
        ],
    ]
    store.direct_relations = [
        replace(
            relation_head(0.99 - index / 1_000),
            id=100 + index,
            text_hash=f"{100 + index:064x}",
            support_fact_ids=(fact_id,),
            support_count=1,
        )
        for index, fact_id in enumerate(support_fact_ids)
    ]
    store.loaded_relations = [
        replace(
            relation_head(0.935),
            id=200,
            text_hash=f"{200:064x}",
            support_fact_ids=(B,),
            support_count=1,
        ),
        replace(
            relation_head(0.80),
            id=201,
            text_hash=f"{201:064x}",
            support_fact_ids=(A,),
            support_count=1,
        ),
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    first = await service.recall(FISH_ID, "query", 1)
    second = await service.recall(FISH_ID, "query", 1)

    expected_order = (*range(100, 149), A)
    assert A in first.memory_unit_order
    assert A in {fact.id for fact in first.facts}
    assert len(first.memory_unit_order) <= 50
    assert first.memory_unit_order == expected_order
    assert second.memory_unit_order == expected_order


@pytest.mark.parametrize(
    ("prefix_size", "expected_size"),
    ((1, 50), (25, 50), (26, 26)),
)
@pytest.mark.asyncio
async def test_dominant_gap_only_truncates_when_more_than_half_head_remains(
    prefix_size: int,
    expected_size: int,
) -> None:
    store = FakeStore()
    fact_ids = [UUID(int=2_000 + index) for index in range(50)]
    scores = (
        *(0.99 - index / 10_000 for index in range(prefix_size)),
        *(0.60 - index / 10_000 for index in range(50 - prefix_size)),
    )
    store.fact_heads = [
        fact_head(fact_id, score, f"fact {index}", ())
        for index, (fact_id, score) in enumerate(zip(fact_ids, scores, strict=True))
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "query", 1)

    assert result.memory_unit_order == tuple(fact_ids[:expected_size])
    assert [fact.id for fact in result.facts] == fact_ids[:expected_size]


@pytest.mark.asyncio
async def test_gap_gate_counts_mask_champion_in_final_retained_units() -> None:
    store = FakeStore()
    prefix_ids = [UUID(int=2_100 + index) for index in range(25)]
    tail_ids = [UUID(int=2_200 + index) for index in range(24)]
    store.fact_heads = [
        *[
            fact_head(fact_id, 0.99 - index / 10_000, f"prefix {index}", ())
            for index, fact_id in enumerate(prefix_ids)
        ],
        fact_head(
            A,
            0.60,
            "杰洛特以前使用过银剑",
            (),
            entity_ids=(ENTITY_ID,),
        ),
        *[
            fact_head(fact_id, 0.59 - index / 10_000, f"tail {index}", ())
            for index, fact_id in enumerate(tail_ids)
        ],
    ]
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    signature = entity_signature((ENTITY_ID,))
    store.persisted_fact_masks = [
        MaskEmbedding(
            object_id=A,
            entity_ids=signature,
            signature_hash=signature_hash(signature),
            source_text_hash=f"{A.int:064x}",
            masked_text="以前使用过银剑",
            vector=(1.0, 1.0) + (0.0,) * 382,
        )
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert result.memory_unit_order == (*prefix_ids, A)


@pytest.mark.asyncio
async def test_sub_half_gap_keeps_all_candidates_with_relation_mask_champion() -> None:
    store = FakeStore()
    top_one = UUID(int=2_100)
    top_two = UUID(int=2_101)
    tail_one = UUID(int=2_102)
    tail_two = UUID(int=2_103)
    store.fact_heads = [
        fact_head(top_one, 0.96, "top one", ()),
        fact_head(top_two, 0.94, "top two", ()),
        fact_head(A, 0.61, "relation support", (), entity_ids=(ENTITY_ID,)),
        fact_head(tail_one, 0.60, "tail one", ()),
        fact_head(tail_two, 0.59, "tail two", ()),
    ]
    store.direct_relations = [
        replace(
            relation_head(0.61),
            snapshot_text="杰洛特以前使用过银剑",
            support_fact_ids=(A,),
            support_count=1,
        ),
        replace(
            relation_head(0.72),
            id=8,
            text_hash="b" * 64,
            support_fact_ids=(tail_one,),
            support_count=1,
        ),
    ]
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    signature = entity_signature((ENTITY_ID,))
    store.persisted_relation_masks = [
        MaskEmbedding(
            object_id=7,
            entity_ids=signature,
            signature_hash=signature_hash(signature),
            source_text_hash="a" * 64,
            masked_text="以前使用过银剑",
            vector=(1.0, 1.0) + (0.0,) * 382,
        )
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert result.memory_unit_order == (top_one, top_two, 8, 7, tail_two)
    assert [unit.id for unit in result.relation_units] == [8, 7]
    assert [unit.primary for unit in result.relation_units] == [True, True]


@pytest.mark.asyncio
async def test_sub_half_gap_keeps_all_candidates_with_standalone_mask_champion() -> (
    None
):
    store = FakeStore()
    top_one = UUID(int=2_200)
    top_two = UUID(int=2_201)
    tail_one = UUID(int=2_202)
    tail_two = UUID(int=2_203)
    store.fact_heads = [
        fact_head(top_one, 0.96, "top one", ()),
        fact_head(top_two, 0.94, "top two", ()),
        fact_head(
            A,
            0.61,
            "杰洛特以前使用过银剑",
            (),
            entity_ids=(ENTITY_ID,),
        ),
        fact_head(tail_one, 0.60, "tail one", ()),
        fact_head(tail_two, 0.59, "tail two", ()),
    ]
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    signature = entity_signature((ENTITY_ID,))
    store.persisted_fact_masks = [
        MaskEmbedding(
            object_id=A,
            entity_ids=signature,
            signature_hash=signature_hash(signature),
            source_text_hash=f"{A.int:064x}",
            masked_text="以前使用过银剑",
            vector=(1.0, 1.0) + (0.0,) * 382,
        )
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert result.memory_unit_order == (top_one, top_two, A, tail_one, tail_two)
    assert [fact.id for fact in result.facts] == [
        top_one,
        top_two,
        A,
        tail_one,
        tail_two,
    ]


@pytest.mark.asyncio
async def test_sub_half_gap_keeps_all_candidates_with_member_mask_champion() -> None:
    store = FakeStore()
    top_one = UUID(int=2_300)
    top_two = UUID(int=2_301)
    tail_one = UUID(int=2_302)
    tail_two = UUID(int=2_303)
    store.fact_heads = [
        fact_head(top_one, 0.96, "top one", ()),
        fact_head(top_two, 0.94, "top two", ()),
        fact_head(
            A,
            0.61,
            "杰洛特以前使用过银剑",
            (),
            entity_ids=(ENTITY_ID,),
        ),
        fact_head(tail_one, 0.60, "tail one", ()),
        fact_head(tail_two, 0.59, "tail two", ()),
    ]
    store.direct_relations = [
        replace(
            relation_head(0.61),
            snapshot_text="银剑的使用记录",
            support_fact_ids=(A,),
            support_count=1,
        ),
        replace(
            relation_head(0.72),
            id=8,
            text_hash="b" * 64,
            support_fact_ids=(tail_one,),
            support_count=1,
        ),
    ]
    store.query_entities = [EntityAliases(ENTITY_ID, ("杰洛特", "白狼"))]
    signature = entity_signature((ENTITY_ID,))
    store.persisted_fact_masks = [
        MaskEmbedding(
            object_id=A,
            entity_ids=signature,
            signature_hash=signature_hash(signature),
            source_text_hash=f"{A.int:064x}",
            masked_text="以前使用过银剑",
            vector=(1.0, 1.0) + (0.0,) * 382,
        )
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "白狼以前用过什么剑", 1)

    assert result.memory_unit_order == (top_one, top_two, 8, 7, tail_two)
    assert [unit.id for unit in result.relation_units] == [8, 7]
    assert [unit.primary for unit in result.relation_units] == [True, True]


@pytest.mark.asyncio
async def test_sub_half_gap_keeps_all_candidates_with_best_relation_champion() -> None:
    store = FakeStore()
    top_one = UUID(int=2_400)
    top_two = UUID(int=2_401)
    tail = UUID(int=2_402)
    store.fact_heads = [
        fact_head(top_one, 0.96, "top one", ()),
        fact_head(top_two, 0.94, "top two", ()),
        fact_head(B, 0.72, "higher unit support", ()),
        fact_head(A, 0.61, "best relation support", ()),
        fact_head(tail, 0.59, "tail", ()),
    ]
    store.direct_relations = [
        replace(
            relation_head(0.70),
            support_fact_ids=(A,),
            support_count=1,
        ),
        replace(
            relation_head(0.60),
            id=8,
            text_hash="b" * 64,
            support_fact_ids=(B,),
            support_count=1,
        ),
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "query", 1)

    assert result.memory_unit_order == (top_one, top_two, 8, 7, tail)
    assert [unit.id for unit in result.relation_units] == [8, 7]
    assert [unit.primary for unit in result.relation_units] == [False, True]


@pytest.mark.asyncio
async def test_sub_half_gap_keeps_all_candidates_with_fact_head_champions() -> None:
    store = FakeStore()
    tail = UUID(int=2_502)
    store.fact_heads = [
        fact_head(A, 0.90, "rank one shared support", (7, 8)),
        fact_head(B, 0.895, "higher relation support one", (9,)),
        fact_head(C, 0.89, "higher relation support two", (10,)),
        fact_head(tail, 0.885, "irrelevant tail", ()),
    ]
    store.direct_relations = [
        replace(
            relation_head(0.70),
            support_fact_ids=(A,),
            support_count=1,
        ),
        replace(
            relation_head(0.69),
            id=8,
            text_hash="b" * 64,
            support_fact_ids=(A,),
            support_count=1,
        ),
        replace(
            relation_head(0.99),
            id=9,
            text_hash="c" * 64,
            support_fact_ids=(B,),
            support_count=1,
        ),
        replace(
            relation_head(0.98),
            id=10,
            text_hash="d" * 64,
            support_fact_ids=(C,),
            support_count=1,
        ),
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "query", 1)

    assert result.memory_unit_order == (9, 10, 7, 8, tail)
    assert [unit.id for unit in result.relation_units] == [9, 10, 7, 8]
    assert [unit.primary for unit in result.relation_units] == [
        True,
        True,
        True,
        True,
    ]


@pytest.mark.asyncio
async def test_no_gap_keeps_candidates_but_reserves_tail_relations() -> None:
    store = FakeStore()
    fact_ids = (UUID(int=2_600), UUID(int=2_601), UUID(int=2_602))
    scores = (0.90, 0.89, 0.88)
    store.fact_heads = []
    store.hydrated_fact_heads = [
        fact_head(fact_id, score, f"support {index}", ())
        for index, (fact_id, score) in enumerate(zip(fact_ids, scores, strict=True))
    ]
    store.direct_relations = [
        replace(
            relation_head(score),
            id=7 + index,
            text_hash=f"{7 + index:064x}",
            support_fact_ids=(fact_id,),
            support_count=1,
        )
        for index, (fact_id, score) in enumerate(zip(fact_ids, scores, strict=True))
    ]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "query", 1)

    assert len(result.relation_units) == 3
    assert [unit.primary for unit in result.relation_units] == [True, False, False]


@pytest.mark.asyncio
async def test_underfilled_relation_member_is_not_also_a_standalone_unit() -> None:
    store = FakeStore()
    store.fact_search_error = RuntimeError("fact head unavailable")
    store.fact_heads = [
        fact_head(A, 0.90, "shared", (7, 8)),
        fact_head(C, 0.80, "complete", (8,)),
    ]
    underfilled = replace(
        store.loaded_relation,
        id=7,
        support_fact_ids=(A, B),
        support_count=2,
    )
    complete = replace(
        store.loaded_relation,
        id=8,
        support_fact_ids=(A, C),
        support_count=2,
    )
    store.direct_relations = [underfilled, complete]
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "shared", 1)

    assert [unit.id for unit in result.relation_units] == [8]
    assert result.memory_unit_order == (8,)
    assert [fact.id for fact in result.facts] == [A, C]


@pytest.mark.asyncio
async def test_time_range_keeps_semantic_order_without_chronological_reordering() -> (
    None
):
    time_range = RecallTimeRange(
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=None,
    )
    store = FakeStore()
    early = fact_head(A, 0.90, "early", ())
    late = fact_head(C, 0.80, "late", ())
    early = replace(
        early,
        fact=replace(early.fact, mentioned_at=datetime(2021, 1, 1, tzinfo=UTC)),
    )
    late = replace(
        late,
        fact=replace(late.fact, mentioned_at=datetime(2022, 1, 1, tzinfo=UTC)),
    )
    store.fact_heads = [early, late]
    store.time_range = time_range
    store.time_count = 2
    service = FactRecallService(
        store,  # type: ignore[arg-type]
        FakePlanner(time_range),
        FakeEmbedder(),
        FakeSpaceStore(),
        now=lambda: NOW,
    )

    result = await service.recall(FISH_ID, "最新发生了什么？", 1)

    assert [fact.id for fact in result.facts] == [A, C]
