from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from fish_memory.api.routes import router
from fish_memory.db.embedding_spaces import EmbeddingSpaceMismatchError
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.models import Role
from fish_memory.recall.models import (
    RecalledFact,
    RecallExecutionTrace,
    RecallQueryPlan,
    RecallResult,
    RecallSourceRole,
    RecallTimeRange,
)
from fish_memory.recall.service import RecallEmbeddingError

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
FACT_ID = UUID("00000000-0000-0000-0000-000000000002")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000003")
NOW = datetime(2026, 7, 26, tzinfo=UTC)


class FakeFishStore:
    def __init__(self, exists: bool = True) -> None:
        self.fish_exists = exists

    async def exists(self, fish_id: UUID) -> bool:
        assert fish_id == FISH_ID
        return self.fish_exists


class FakeRecall:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, str, int]] = []
        self.failure: Exception | None = None
        self.degraded = False

    async def recall(
        self,
        fish_id: UUID,
        content: str,
        depth: int,
    ) -> RecallResult:
        self.calls.append((fish_id, content, depth))
        if self.failure is not None:
            raise self.failure
        return RecallResult(
            facts=(
                RecalledFact(
                    id=FACT_ID,
                    source_message_id=MESSAGE_ID,
                    source_role=Role.USER,
                    text="该用户不想继续玩游戏",
                    occurred_start=None,
                    occurred_end=None,
                    mentioned_at=NOW,
                    entities=("游戏",),
                    matched_by=("semantic",),
                    semantic_score=0.83,
                    rank_score=None,
                ),
            ),
            trace=RecallExecutionTrace(
                depth=depth,
                degraded=self.degraded,
                plan=RecallQueryPlan(
                    semanticQuery="杰洛特改名为大呆",
                    sourceRole=RecallSourceRole.EITHER,
                    entityHints=["杰洛特", "大呆"],
                    time=RecallTimeRange(start=None, end=None),
                ),
                plan_error=None,
                duration_ms={
                    "planning": 0.0,
                    "embedding": 1.0,
                    "semanticSearch": 2.0,
                    "timeSearch": 0.0,
                    "merge": 0.0,
                    "total": 3.0,
                },
            ),
        )


def build_test_app(
    recall: FakeRecall,
    *,
    fish_exists: bool = True,
) -> FastAPI:
    app = FastAPI()
    app.state.fish_store = FakeFishStore(fish_exists)
    app.state.recall_service = recall
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_returns_recalled_facts_and_trace() -> None:
    recall = FakeRecall()
    transport = httpx.ASGITransport(app=build_test_app(recall))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/memories/recall",
            json={
                "query": "我上次不开心是为什么？",
                "debug": True,
            },
        )

    assert response.status_code == 200
    assert recall.calls == [(FISH_ID, "我上次不开心是为什么？", 1)]
    payload = response.json()
    assert "strategyRequested" not in payload
    assert "strategyUsed" not in payload
    assert payload["degraded"] is False
    assert payload["facts"][0] == {
        "id": str(FACT_ID),
        "sourceMessageId": str(MESSAGE_ID),
        "sourceRole": "user",
        "text": "该用户不想继续玩游戏",
        "mentionedAt": "2026-07-26T00:00:00Z",
        "entities": ["游戏"],
        "matchedBy": ["semantic"],
        "semanticScore": 0.83,
    }
    assert payload["trace"]["depth"] == 1
    assert payload["trace"]["plan"] == {
        "semanticQuery": "杰洛特改名为大呆",
        "sourceRole": "either",
        "entityHints": ["杰洛特", "大呆"],
        "time": {},
    }
    assert payload["trace"]["durationMs"]["semanticSearch"] == 2.0


@pytest.mark.asyncio
async def test_omits_debug_trace() -> None:
    recall = FakeRecall()
    transport = httpx.ASGITransport(app=build_test_app(recall))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/memories/recall",
            json={"query": "用户喜欢什么游戏"},
        )

    assert recall.calls == [(FISH_ID, "用户喜欢什么游戏", 1)]
    assert "trace" not in response.json()


@pytest.mark.asyncio
async def test_reports_degraded_recall() -> None:
    recall = FakeRecall()
    recall.degraded = True
    transport = httpx.ASGITransport(app=build_test_app(recall))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/memories/recall",
            json={"query": "用户喜欢什么游戏", "debug": True},
        )

    assert response.json()["degraded"] is True
    assert response.json()["trace"]["degraded"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": " "},
        {"query": "x" * 6001},
    ],
)
async def test_rejects_invalid_request(body: dict[str, object]) -> None:
    transport = httpx.ASGITransport(app=build_test_app(FakeRecall()))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(f"/v1/fishes/{FISH_ID}/memories/recall", json=body)

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_rejects_strategy_as_an_extra_field() -> None:
    transport = httpx.ASGITransport(app=build_test_app(FakeRecall()))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/memories/recall",
            json={"query": "query", "strategy": "fast"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_returns_not_found_for_missing_or_deleted_fish() -> None:
    recall = FakeRecall()
    transport = httpx.ASGITransport(app=build_test_app(recall, fish_exists=False))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/memories/recall",
            json={"query": "query"},
        )

    assert response.status_code == 404
    assert recall.calls == []


@pytest.mark.asyncio
async def test_returns_bad_gateway_when_embedding_fails() -> None:
    recall = FakeRecall()
    recall.failure = RecallEmbeddingError("embedding unavailable")
    transport = httpx.ASGITransport(app=build_test_app(recall))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/memories/recall",
            json={"query": "query"},
        )

    assert response.status_code == 502


@pytest.mark.asyncio
async def test_returns_conflict_when_fish_embedding_space_differs() -> None:
    recall = FakeRecall()
    recall.failure = EmbeddingSpaceMismatchError(
        FISH_ID,
        EmbeddingSpace("openai-compatible", "old-model", 1024),
        EmbeddingSpace("local", "intfloat/multilingual-e5-small", 384),
    )
    transport = httpx.ASGITransport(app=build_test_app(recall))

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"/v1/fishes/{FISH_ID}/memories/recall",
            json={"query": "query"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "embedding_space_mismatch"
