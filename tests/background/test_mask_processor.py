from collections.abc import Sequence
from uuid import UUID

import pytest

from fish_memory.background.mask_processor import MaskEmbeddingProcessor
from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.recall.masking import EntityAliases, MaskEmbedding, SingleMaskSource

FISH_ID = UUID(int=1)
FACT_ID = UUID(int=2)
ENTITY_ID = UUID(int=3)
SPACE = EmbeddingSpace("local", "test", 3)


class FakeEmbedder:
    space = SPACE

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            tuple((1.0, 0.0, 0.0) for _text in texts),
            len(texts),
        )


class FakeStore:
    def __init__(self) -> None:
        entity = EntityAliases(ENTITY_ID, ("杰洛特", "白狼", "Geralt"))
        self.fact_sources = [
            SingleMaskSource(
                kind="fact",
                object_id=FACT_ID,
                fish_id=FISH_ID,
                source_text="杰洛特以前使用过银剑",
                source_text_hash="a" * 64,
                entity=entity,
            )
        ]
        self.relation_sources = [
            SingleMaskSource(
                kind="relation",
                object_id=7,
                fish_id=FISH_ID,
                source_text="白狼以前使用过银剑",
                source_text_hash="b" * 64,
                entity=entity,
            )
        ]
        self.fact_rows: list[MaskEmbedding] = []
        self.relation_rows: list[MaskEmbedding] = []
        self.fact_offsets: list[int] = []
        self.relation_offsets: list[int] = []

    async def list_fact_single_mask_sources(
        self,
        dimensions: int,
        limit: int,
        *,
        offset: int = 0,
    ) -> list[SingleMaskSource]:
        assert dimensions == 3
        assert limit > 0
        self.fact_offsets.append(offset)
        return self.fact_sources[offset : offset + limit]

    async def list_relation_single_mask_sources(
        self,
        dimensions: int,
        limit: int,
        *,
        offset: int = 0,
    ) -> list[SingleMaskSource]:
        assert dimensions == 3
        assert limit > 0
        self.relation_offsets.append(offset)
        return self.relation_sources[offset : offset + limit]

    async def upsert_fact_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None:
        assert (fish_id, space) == (FISH_ID, SPACE)
        self.fact_rows.extend(embeddings)
        persisted = {
            (row.object_id, row.entity_ids[0], row.source_text_hash)
            for row in embeddings
        }
        self.fact_sources = [
            source
            for source in self.fact_sources
            if (
                source.object_id,
                source.entity.entity_id,
                source.source_text_hash,
            )
            not in persisted
        ]

    async def upsert_relation_mask_embeddings(
        self,
        fish_id: UUID,
        space: EmbeddingSpace,
        embeddings: Sequence[MaskEmbedding],
    ) -> None:
        assert (fish_id, space) == (FISH_ID, SPACE)
        self.relation_rows.extend(embeddings)
        persisted = {
            (row.object_id, row.entity_ids[0], row.source_text_hash)
            for row in embeddings
        }
        self.relation_sources = [
            source
            for source in self.relation_sources
            if (
                source.object_id,
                source.entity.entity_id,
                source.source_text_hash,
            )
            not in persisted
        ]


@pytest.mark.asyncio
async def test_generates_one_single_mask_per_entity_and_deduplicates_texts() -> None:
    store = FakeStore()
    embedder = FakeEmbedder()
    processor = MaskEmbeddingProcessor(store, embedder, batch_size=100)

    assert await processor.process_once() is True

    assert embedder.calls == [("以前使用过银剑",)]
    assert len(store.fact_rows) == 1
    assert len(store.relation_rows) == 1
    assert store.fact_rows[0].entity_ids == (ENTITY_ID,)
    assert store.relation_rows[0].entity_ids == (ENTITY_ID,)
    assert store.fact_rows[0].masked_text == "以前使用过银剑"
    assert store.relation_rows[0].masked_text == "以前使用过银剑"


@pytest.mark.asyncio
async def test_returns_false_without_missing_single_masks() -> None:
    store = FakeStore()
    store.fact_sources = []
    store.relation_sources = []

    processor = MaskEmbeddingProcessor(store, FakeEmbedder(), batch_size=100)

    assert await processor.process_once() is False


@pytest.mark.asyncio
async def test_unmaskable_sources_do_not_cause_tight_loop_or_starvation() -> None:
    store = FakeStore()
    unmaskable_entity = EntityAliases(ENTITY_ID, ("he",))
    maskable_entity = EntityAliases(UUID(int=4), ("杰洛特",))
    store.fact_sources = [
        SingleMaskSource(
            kind="fact",
            object_id=FACT_ID,
            fish_id=FISH_ID,
            source_text="the sword",
            source_text_hash="c" * 64,
            entity=unmaskable_entity,
        ),
        SingleMaskSource(
            kind="fact",
            object_id=UUID(int=5),
            fish_id=FISH_ID,
            source_text="杰洛特以前使用过银剑",
            source_text_hash="d" * 64,
            entity=maskable_entity,
        ),
    ]
    store.relation_sources = []
    embedder = FakeEmbedder()
    processor = MaskEmbeddingProcessor(store, embedder, batch_size=1)

    assert await processor.process_once() is True
    assert await processor.process_once() is True
    assert await processor.process_once() is False

    assert store.fact_offsets == [0, 1, 1]
    assert embedder.calls == [("以前使用过银剑",)]
    assert [row.object_id for row in store.fact_rows] == [UUID(int=5)]
