from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from fish_memory.embeddings.client import EmbeddingResult, EmbeddingSpace
from fish_memory.recall.masking import (
    MaskEmbedding,
    SingleMaskSource,
    entity_signature,
    mask_text,
    signature_hash,
)


class MaskEmbeddingStoreProtocol(Protocol):
    async def list_fact_single_mask_sources(
        self,
        dimensions: int,
        limit: int,
        *,
        offset: int = 0,
    ) -> list[SingleMaskSource]: ...

    async def list_relation_single_mask_sources(
        self,
        dimensions: int,
        limit: int,
        *,
        offset: int = 0,
    ) -> list[SingleMaskSource]: ...

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


class MaskEmbedderProtocol(Protocol):
    @property
    def space(self) -> EmbeddingSpace: ...

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


class MaskEmbeddingProcessor:
    def __init__(
        self,
        store: MaskEmbeddingStoreProtocol,
        embedder: MaskEmbedderProtocol,
        *,
        batch_size: int,
        embedding_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("mask embedding batch size must be positive")
        self._store = store
        self._embedder = embedder
        self._space = embedder.space
        self._batch_size = batch_size
        self._embedding_semaphore = embedding_semaphore
        self._fact_offset = 0
        self._relation_offset = 0

    async def process_once(self) -> bool:
        fact_task = asyncio.create_task(
            self._store.list_fact_single_mask_sources(
                self._space.dimensions,
                self._batch_size,
                offset=self._fact_offset,
            )
        )
        relation_task = asyncio.create_task(
            self._store.list_relation_single_mask_sources(
                self._space.dimensions,
                self._batch_size,
                offset=self._relation_offset,
            )
        )
        fact_sources, relation_sources = await asyncio.gather(
            fact_task,
            relation_task,
        )
        sources = [*fact_sources, *relation_sources]
        if not sources:
            self._fact_offset = 0
            self._relation_offset = 0
            return False

        masked_by_source: dict[int, str] = {}
        sources_by_text: dict[str, list[int]] = {}
        for index, source in enumerate(sources):
            masked = mask_text(source.source_text, (source.entity,))
            if masked is None:
                continue
            masked_by_source[index] = masked
            sources_by_text.setdefault(masked, []).append(index)
        maskable_kinds = {sources[index].kind for index in masked_by_source}
        advanced = False
        if fact_sources and "fact" not in maskable_kinds:
            self._fact_offset += len(fact_sources)
            advanced = True
        elif not fact_sources:
            self._fact_offset = 0
        if relation_sources and "relation" not in maskable_kinds:
            self._relation_offset += len(relation_sources)
            advanced = True
        elif not relation_sources:
            self._relation_offset = 0
        if not sources_by_text:
            return advanced

        texts = tuple(sources_by_text)
        if self._embedding_semaphore is None:
            embedded = await self._embedder.embed(texts)
        else:
            async with self._embedding_semaphore:
                embedded = await self._embedder.embed(texts)
        if len(embedded.vectors) != len(texts):
            raise ValueError("single mask embedding count does not match input count")

        records_by_source: dict[int, MaskEmbedding] = {}
        for text, vector in zip(texts, embedded.vectors, strict=True):
            for index in sources_by_text[text]:
                source = sources[index]
                signature = entity_signature((source.entity.entity_id,))
                records_by_source[index] = MaskEmbedding(
                    object_id=source.object_id,
                    entity_ids=signature,
                    signature_hash=signature_hash(signature),
                    source_text_hash=source.source_text_hash,
                    masked_text=masked_by_source[index],
                    vector=tuple(vector),
                )

        tasks: list[asyncio.Task[None]] = []
        for kind in ("fact", "relation"):
            fish_ids = {
                source.fish_id
                for index, source in enumerate(sources)
                if source.kind == kind and index in records_by_source
            }
            for fish_id in fish_ids:
                records = [
                    records_by_source[index]
                    for index, source in enumerate(sources)
                    if source.kind == kind
                    and source.fish_id == fish_id
                    and index in records_by_source
                ]
                method = (
                    self._store.upsert_fact_mask_embeddings
                    if kind == "fact"
                    else self._store.upsert_relation_mask_embeddings
                )
                tasks.append(asyncio.create_task(method(fish_id, self._space, records)))
        if tasks:
            await asyncio.gather(*tasks)
        return True
