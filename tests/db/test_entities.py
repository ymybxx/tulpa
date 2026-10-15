from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.entities import (
    EntityStore,
    lock_entity_index,
    replace_fact_entities,
)
from fish_memory.entities.models import (
    EntityContextMessage,
    EntityFact,
    EntityFactPage,
    EntityMergeGroup,
    EntitySummary,
)
from fish_memory.models import Role

FISH_ID = UUID(int=1)
FACT_ID = UUID(int=2)
NOW = datetime(2026, 7, 24, tzinfo=UTC)


class FakeContext:
    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_results: list[list[dict[str, object]]] = []
        self.fetchrow_results: list[dict[str, object] | None] = []

    def _record(self, query: str, args: tuple[Any, ...]) -> None:
        self.calls.append((" ".join(query.split()), args))

    async def execute(self, query: str, *args: Any) -> str:
        self._record(query, args)
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        self._record(query, args)
        return self.fetch_results.pop(0)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, object] | None:
        self._record(query, args)
        return self.fetchrow_results.pop(0)

    def transaction(self) -> FakeContext:
        return FakeContext(None)


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeContext:
        return FakeContext(self.connection)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        return await self.connection.fetch(query, *args)


def store_with(connection: FakeConnection) -> EntityStore:
    return EntityStore(cast(asyncpg.Pool, FakePool(connection)))


@pytest.mark.asyncio
async def test_replaces_fact_entities_with_normalized_unique_aliases() -> None:
    connection = FakeConnection()

    await lock_entity_index(cast(asyncpg.Connection, connection), FISH_ID)
    await replace_fact_entities(
        cast(asyncpg.Connection, connection),
        FISH_ID,
        FACT_ID,
        ("  ＧerAlt  ", "geralt", " 杰洛特 ", ""),
    )

    assert "pg_advisory_xact_lock" in connection.calls[0][0]
    assert "DELETE FROM fact_entities" in connection.calls[1][0]
    assert len(connection.calls) == 3
    query, args = connection.calls[2]
    assert "INSERT INTO entity_aliases" in query
    assert "INSERT INTO fact_entities" in query
    assert args == (FISH_ID, FACT_ID, ["geralt", "杰洛特"])


@pytest.mark.asyncio
async def test_loads_stable_candidates_and_fact_indexes() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            {
                "fact_id": UUID(int=20),
                "entity_id": UUID(int=30),
                "alias": "a",
                "text": "fact a",
                "source_role": "user",
            },
            {
                "fact_id": UUID(int=20),
                "entity_id": UUID(int=31),
                "alias": "b",
                "text": "fact a",
                "source_role": "user",
            },
            {
                "fact_id": UUID(int=20),
                "entity_id": UUID(int=31),
                "alias": "bee",
                "text": "fact a",
                "source_role": "user",
            },
            {
                "fact_id": UUID(int=21),
                "entity_id": UUID(int=31),
                "alias": "b",
                "text": "assistant fact b",
                "source_role": "assistant",
            },
            {
                "fact_id": UUID(int=21),
                "entity_id": UUID(int=31),
                "alias": "bee",
                "text": "assistant fact b",
                "source_role": "assistant",
            },
        ]
    )
    connection.fetch_results.append([])

    context = await store_with(connection).load_context(
        FISH_ID,
        (UUID(int=20), UUID(int=21)),
    )

    assert "fact.invalidated_at IS NULL" not in connection.calls[0][0]
    assert [(candidate.id, candidate.aliases) for candidate in context.candidates] == [
        (UUID(int=30), ("a",)),
        (UUID(int=31), ("b", "bee")),
    ]
    assert context.fact_entity_indexes == {
        UUID(int=20): (0, 1),
        UUID(int=21): (1,),
    }
    assert context.fact_source_roles == {
        UUID(int=20): Role.USER,
        UUID(int=21): Role.ASSISTANT,
    }
    assert context.fact_texts == {
        UUID(int=20): "fact a",
        UUID(int=21): "assistant fact b",
    }
    assert context.recent_context == ()


@pytest.mark.asyncio
async def test_loads_recent_visible_messages_for_entity_resolution() -> None:
    connection = FakeConnection()
    connection.fetch_results.extend(
        [
            [],
            [
                {"role": "assistant", "content": "助手：西湖的英文是 West Lake。"},
                {"role": "user", "content": "用户：西湖的英文怎么说？"},
            ],
        ]
    )

    context = await store_with(connection).load_context(FISH_ID, (FACT_ID,))

    assert context.candidates == ()
    assert context.recent_context == (
        EntityContextMessage(Role.USER, "用户：西湖的英文怎么说？"),
        EntityContextMessage(Role.ASSISTANT, "助手：西湖的英文是 West Lake。"),
    )
    assert connection.calls[1][1] == (FISH_ID, [FACT_ID], 20)


@pytest.mark.asyncio
async def test_merges_aliases_and_fact_links_idempotently() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            {"alias": "a", "entity_id": UUID(int=40)},
            {"alias": "b", "entity_id": UUID(int=39)},
        ]
    )

    await store_with(connection).merge_groups(
        FISH_ID,
        (EntityMergeGroup(("a", "b")),),
    )

    assert any("pg_advisory_xact_lock" in query for query, _ in connection.calls)
    copy_call = next(
        call for call in connection.calls if "INSERT INTO fact_entities" in call[0]
    )
    assert copy_call[1] == (UUID(int=39), [UUID(int=40)])
    assert any(
        "DELETE FROM fact_entities" in query and args == ([UUID(int=40)],)
        for query, args in connection.calls
    )
    assert any(
        "UPDATE entity_aliases" in query
        and args == (FISH_ID, UUID(int=39), [UUID(int=40)])
        for query, args in connection.calls
    )


@pytest.mark.asyncio
async def test_lists_entity_summaries_and_paginates_linked_facts() -> None:
    connection = FakeConnection()
    connection.fetch_results.extend(
        [
            [
                {
                    "entity_id": UUID(int=30),
                    "aliases": ["geralt", "geralt", "杰洛特", "杰洛特"],
                    "fact_count": 2,
                }
            ],
            [
                {
                    "id": UUID(int=40),
                    "text": "该用户去了公园",
                    "occurred_start": None,
                    "occurred_end": None,
                    "mentioned_at": NOW,
                    "source_role": "user",
                    "invalidated_at": None,
                },
                {
                    "id": UUID(int=41),
                    "text": "该用户回家了",
                    "occurred_start": None,
                    "occurred_end": None,
                    "mentioned_at": NOW,
                    "source_role": "assistant",
                    "invalidated_at": NOW,
                },
            ],
        ]
    )

    store = store_with(connection)
    summaries = await store.list_summaries(FISH_ID)
    page = await store.list_facts(FISH_ID, UUID(int=30), limit=1)

    assert summaries == [
        EntitySummary(
            entity_id=UUID(int=30), aliases=("geralt", "杰洛特"), fact_count=2
        )
    ]
    assert page == EntityFactPage(
        items=(
            EntityFact(
                id=UUID(int=40),
                text="该用户去了公园",
                occurred_start=None,
                occurred_end=None,
                mentioned_at=NOW,
                source_role="user",
                invalidated_at=None,
            ),
        ),
        has_more=True,
    )
    assert "LIMIT $2" in connection.calls[0][0]
    assert "LIMIT $3" in connection.calls[1][0]
