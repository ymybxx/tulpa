import hashlib
import json
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.facts import FactStore
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.facts.models import (
    BackgroundTaskProgress,
    ExtractedFact,
    PendingFactMessage,
)
from fish_memory.models import Message, MessageStatus, Role

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
SECOND_FISH_ID = UUID("00000000-0000-0000-0000-000000000002")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000003")
SECOND_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000004")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000006")
SECOND_SESSION_ID = UUID("00000000-0000-0000-0000-000000000007")
NOW = datetime(2026, 7, 23, tzinfo=UTC)
FACT_ID = UUID("00000000-0000-0000-0000-000000000005")
SECOND_FACT_ID = UUID("00000000-0000-0000-0000-000000000008")
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_SPACE = EmbeddingSpace(
    provider="openai-compatible",
    model=EMBEDDING_MODEL,
    dimensions=1024,
)
VECTOR = (0.25,) * 1024


def message_record(
    *,
    fish_id: UUID = FISH_ID,
    session_id: UUID = SESSION_ID,
    message_id: UUID = MESSAGE_ID,
    sequence: int = 7,
    role: str = "user",
    content: str = "主人搬到了上海",
    status: str = "initialized",
    target_sequence: int = 9,
) -> dict[str, object]:
    return {
        "id": message_id,
        "fish_id": fish_id,
        "session_id": session_id,
        "sequence": sequence,
        "role": role,
        "content": content,
        "status": status,
        "created_at": NOW,
        "context_through_sequence": None,
        "consumed_offset": None,
        "finished_at": None,
        "fact_extraction_attempt_count": 0,
        "context_id": None,
        "context_fish_id": None,
        "context_session_id": None,
        "context_sequence": None,
        "context_role": None,
        "context_content": None,
        "context_status": None,
        "context_created_at": None,
        "context_context_through_sequence": None,
        "context_consumed_offset": None,
        "context_finished_at": None,
        "target_sequence": target_sequence,
    }


class FakeContext:
    def __init__(self, value: Any, events: list[str], name: str) -> None:
        self._value = value
        self._events = events
        self._name = name

    async def __aenter__(self) -> Any:
        self._events.append(f"{self._name}:enter")
        return self._value

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._events.append(f"{self._name}:exit")


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_results: list[list[dict[str, object]]] = []
        self.fetchrow_results: list[dict[str, object] | None] = []
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []
        self.events: list[str] = []

    def _record(self, query: str, args: tuple[Any, ...]) -> None:
        normalized = " ".join(query.split())
        self.calls.append((normalized, args))
        self.events.append(normalized.split()[0])

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        self._record(query, args)
        return self.fetch_results.pop(0)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, object] | None:
        self._record(query, args)
        return self.fetchrow_results.pop(0)

    async def execute(self, query: str, *args: Any) -> str:
        self._record(query, args)
        return "OK"

    async def executemany(
        self,
        query: str,
        args: list[tuple[object, ...]],
    ) -> None:
        normalized = " ".join(query.split())
        self.executemany_calls.append((normalized, args))
        self.events.append("INSERT")

    def transaction(self) -> FakeContext:
        return FakeContext(None, self.events, "transaction")


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def acquire(self) -> FakeContext:
        return FakeContext(self._connection, self._connection.events, "acquire")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        return await self._connection.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, object] | None:
        return await self._connection.fetchrow(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return await self._connection.execute(query, *args)

    async def executemany(
        self,
        query: str,
        args: list[tuple[object, ...]],
    ) -> None:
        await self._connection.executemany(query, args)


def store_with(connection: FakeConnection) -> FactStore:
    return FactStore(cast(asyncpg.Pool, FakePool(connection)))


@pytest.mark.asyncio
async def test_lists_pending_message_batches_per_active_fish() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            message_record(),
            message_record(
                message_id=UUID("00000000-0000-0000-0000-000000000099"),
                sequence=8,
                content="主人买了一辆车",
                target_sequence=9,
            ),
            message_record(
                fish_id=SECOND_FISH_ID,
                session_id=SECOND_SESSION_ID,
                message_id=SECOND_MESSAGE_ID,
                sequence=4,
                role="assistant",
                content="仍在生成但不是最新消息",
                status="generating",
                target_sequence=6,
            ),
        ]
    )
    store = store_with(connection)

    pending = await store.list_pending_messages(limit=10, batch_size=2)

    assert pending == [
        PendingFactMessage(
            message=Message(
                id=MESSAGE_ID,
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=7,
                role=Role.USER,
                content="主人搬到了上海",
                status=MessageStatus.INITIALIZED,
                created_at=NOW,
            ),
            target_sequence=9,
        ),
        PendingFactMessage(
            message=Message(
                id=UUID("00000000-0000-0000-0000-000000000099"),
                fish_id=FISH_ID,
                session_id=SESSION_ID,
                sequence=8,
                role=Role.USER,
                content="主人买了一辆车",
                status=MessageStatus.INITIALIZED,
                created_at=NOW,
            ),
            target_sequence=9,
        ),
        PendingFactMessage(
            message=Message(
                id=SECOND_MESSAGE_ID,
                fish_id=SECOND_FISH_ID,
                session_id=SECOND_SESSION_ID,
                sequence=4,
                role=Role.ASSISTANT,
                content="仍在生成但不是最新消息",
                status=MessageStatus.GENERATING,
                created_at=NOW,
            ),
            target_sequence=6,
        ),
    ]
    query, args = connection.calls[0]
    assert "JOIN messages" in query
    assert "fish.deleted_at IS NULL" in query
    assert "fish.snapshot_storage_version = 2" in query
    assert "max(message.sequence) AS latest_sequence" in query
    assert "AS waiting_sequence" in query
    assert "message.sequence < latest.waiting_sequence" in query
    assert "COALESCE(max(message.sequence), 0) AS target_sequence" in query
    assert "message.sequence < latest.latest_sequence" in query
    assert "candidate_session.target_sequence AS target_sequence" in query
    assert "LEFT JOIN session_background_task_progress" in query
    assert "progress.task_name = 'fact_extraction'" in query
    assert "JOIN LATERAL" in query
    assert "message.sequence > COALESCE(progress.through_sequence, 0)" in query
    assert "message.sequence <= bounds.target_sequence" in query
    assert "message.fact_extraction_status = 'pending'" in query
    assert "ORDER BY message.sequence" in query
    assert "LIMIT $2" in query
    assert "ORDER BY progress.updated_at NULLS FIRST, bounds.session_id" in query
    assert "ORDER BY candidate_session.progress_updated_at NULLS FIRST" in query
    assert "visible_message_content" not in query
    assert args == (10, 2, None)


@pytest.mark.asyncio
async def test_pending_message_includes_previous_opposite_role_context() -> None:
    connection = FakeConnection()
    record = message_record(role="assistant", sequence=8)
    record.update(
        {
            "context_id": SECOND_MESSAGE_ID,
            "context_fish_id": FISH_ID,
            "context_session_id": SESSION_ID,
            "context_sequence": 7,
            "context_role": "user",
            "context_content": "你刚才在哪里兑换了优惠券？",
            "context_status": "initialized",
            "context_created_at": NOW,
            "context_context_through_sequence": None,
            "context_consumed_offset": None,
            "context_finished_at": None,
        }
    )
    connection.fetch_results.append([record])
    store = store_with(connection)

    pending = await store.list_pending_messages(limit=1, batch_size=1)

    assert pending[0].previous_opposite_message == Message(
        id=SECOND_MESSAGE_ID,
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        sequence=7,
        role=Role.USER,
        content="你刚才在哪里兑换了优惠券？",
        status=MessageStatus.INITIALIZED,
        created_at=NOW,
    )
    query, _ = connection.calls[0]
    assert "LEFT JOIN LATERAL" in query
    assert "message.role <> candidate.role" in query


@pytest.mark.asyncio
async def test_persists_replacement_facts_and_watermark_in_one_transaction() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append({"id": FACT_ID, "fact_index": 0})
    store = store_with(connection)
    pending = PendingFactMessage(
        message=Message(
            id=MESSAGE_ID,
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            sequence=7,
            role=Role.USER,
            content="主人搬到了上海",
            status=MessageStatus.INITIALIZED,
            created_at=NOW,
        ),
        target_sequence=9,
    )
    fact = ExtractedFact(
        text="主人于2026年7月23日搬到上海",
        occurred_start=datetime(2026, 7, 23, tzinfo=UTC),
        occurred_end=datetime(2026, 7, 23, 23, 59, 59, tzinfo=UTC),
        entities=("主人", "上海"),
        what="搬到上海",
        where=("上海",),
        who=("主人",),
    )

    await store.persist_extraction(
        pending,
        [fact],
        [VECTOR],
        EMBEDDING_SPACE,
    )

    assert any("pg_advisory_xact_lock" in query for query, _ in connection.calls)
    alias_calls = [
        call for call in connection.calls if "INSERT INTO entity_aliases" in call[0]
    ]
    assert len(alias_calls) == 1
    assert "INSERT INTO fact_entities" in alias_calls[0][0]
    assert alias_calls[0][1] == (FISH_ID, FACT_ID, ["主人", "上海"])
    message_query, message_args = connection.calls[0]
    assert "UPDATE messages" in message_query
    assert "fact_extraction_status = 'completed'" in message_query
    assert "fact_extraction_error = NULL" in message_query
    assert message_args == (MESSAGE_ID,)
    delete_query, delete_args = connection.calls[1]
    assert "DELETE FROM facts" in delete_query
    assert "source_message_id = $1" in delete_query
    assert "fact_index >= $2" in delete_query
    assert delete_args == (MESSAGE_ID, 1)

    insert_query, insert_args = connection.calls[2]
    assert "INSERT INTO facts" in insert_query
    assert "session_id" in insert_query
    assert "$10::jsonb" in insert_query
    assert "$11::jsonb" in insert_query
    assert "ON CONFLICT (source_message_id, fact_index) DO UPDATE" in insert_query
    for field in (
        "fish_id",
        "session_id",
        "text",
        "occurred_start",
        "occurred_end",
        "mentioned_at",
        "entities",
        "structured",
    ):
        assert f"{field} = EXCLUDED.{field}" in insert_query
    assert insert_args[1:9] == (
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        0,
        fact.text,
        fact.occurred_start,
        fact.occurred_end,
        NOW,
    )
    assert json.loads(cast(str, insert_args[9])) == ["主人", "上海"]
    assert json.loads(cast(str, insert_args[10])) == {
        "what": "搬到上海",
        "where": ["上海"],
        "who": ["主人"],
    }
    assert "RETURNING id, fact_index" in insert_query

    embedding_query, embedding_rows = connection.executemany_calls[0]
    assert "INSERT INTO fact_embeddings" in embedding_query
    assert "ON CONFLICT (fact_id) DO UPDATE" in embedding_query
    assert "embedding_text" in embedding_query
    assert "text_hash" in embedding_query
    assert embedding_rows == [
        (
            FACT_ID,
            FISH_ID,
            "openai-compatible",
            EMBEDDING_MODEL,
            1024,
            fact.embedding_text,
            hashlib.sha256(fact.embedding_text.encode()).hexdigest(),
            "[" + ",".join(["0.25"] * 1024) + "]",
        )
    ]

    progress_query, progress_args = next(
        call
        for call in connection.calls
        if "INSERT INTO session_background_task_progress" in call[0]
    )
    assert "INSERT INTO session_background_task_progress" in progress_query
    assert "'fact_extraction'" in progress_query
    assert "ON CONFLICT (session_id, task_name) DO UPDATE" in progress_query
    assert "last_error = NULL" in progress_query
    assert "fact_extraction_status = 'waiting_consumption'" in progress_query
    assert "SELECT min(sequence) - 1" in progress_query
    assert progress_args == (SESSION_ID, 7)


@pytest.mark.asyncio
async def test_persists_each_fact_with_its_own_embedding_source_text() -> None:
    """Catches a batch reusing the final Fact's key text for every embedding."""
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            {"id": FACT_ID, "fact_index": 0},
            {"id": SECOND_FACT_ID, "fact_index": 1},
        ]
    )
    store = store_with(connection)
    pending = PendingFactMessage(
        message=Message(
            id=MESSAGE_ID,
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            sequence=7,
            role=Role.USER,
            content="两条不同的事实",
            status=MessageStatus.INITIALIZED,
            created_at=NOW,
        ),
        target_sequence=9,
    )
    first = ExtractedFact(
        text="用户仍需领取一件物品",
        occurred_start=None,
        occurred_end=None,
        entities=("第一件物品",),
        what="仍需领取第一件物品",
    )
    second = ExtractedFact(
        text="用户曾在会议上使用另一件物品",
        occurred_start=None,
        occurred_end=None,
        entities=("第二件物品",),
        what="曾在会议上使用第二件物品",
    )
    second_vector = (0.5,) * 1024

    await store.persist_extraction(
        pending,
        [first, second],
        [VECTOR, second_vector],
        EMBEDDING_SPACE,
    )

    _query, rows = connection.executemany_calls[0]
    assert [(row[0], row[5], row[6], row[7]) for row in rows] == [
        (
            FACT_ID,
            first.embedding_text,
            hashlib.sha256(first.embedding_text.encode()).hexdigest(),
            "[" + ",".join(["0.25"] * 1024) + "]",
        ),
        (
            SECOND_FACT_ID,
            second.embedding_text,
            hashlib.sha256(second.embedding_text.encode()).hexdigest(),
            "[" + ",".join(["0.5"] * 1024) + "]",
        ),
    ]


@pytest.mark.asyncio
async def test_does_not_index_entities_from_assistant_facts() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append({"id": FACT_ID, "fact_index": 0})
    store = store_with(connection)
    pending = PendingFactMessage(
        message=Message(
            id=MESSAGE_ID,
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            sequence=7,
            role=Role.ASSISTANT,
            content="助手介绍了一本书",
            status=MessageStatus.INITIALIZED,
            created_at=NOW,
        ),
        target_sequence=9,
    )
    fact = ExtractedFact(
        text="助手介绍了《某本书》",
        occurred_start=None,
        occurred_end=None,
        entities=("《某本书》",),
    )

    await store.persist_extraction(
        pending,
        [fact],
        [VECTOR],
        EMBEDDING_SPACE,
    )

    assert not any(
        "INSERT INTO entity_aliases" in query for query, _ in connection.calls
    )
    assert any("DELETE FROM fact_entities" in query for query, _ in connection.calls)


@pytest.mark.asyncio
async def test_rejects_mismatched_fact_and_vector_counts_before_writing() -> None:
    connection = FakeConnection()
    store = store_with(connection)
    pending = PendingFactMessage(
        message=Message(
            id=MESSAGE_ID,
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            sequence=7,
            role=Role.USER,
            content="source",
            status=MessageStatus.INITIALIZED,
            created_at=NOW,
        ),
        target_sequence=9,
    )
    fact = ExtractedFact(
        text="one",
        occurred_start=None,
        occurred_end=None,
        entities=(),
    )

    with pytest.raises(ValueError, match="count"):
        await store.persist_extraction(
            pending,
            [fact],
            [],
            EMBEDDING_SPACE,
        )

    assert connection.events == []


@pytest.mark.asyncio
async def test_terminal_failure_marks_message_and_advances_watermark() -> None:
    connection = FakeConnection()
    store = store_with(connection)
    pending = PendingFactMessage(
        message=Message(
            id=MESSAGE_ID,
            fish_id=FISH_ID,
            session_id=SESSION_ID,
            sequence=7,
            role=Role.USER,
            content="source",
            status=MessageStatus.INITIALIZED,
            created_at=NOW,
        ),
        target_sequence=9,
    )

    await store.record_failure(pending, "invalid response", terminal=True)

    message_query, message_args = connection.calls[0]
    assert "UPDATE messages" in message_query
    assert (
        "fact_extraction_attempt_count = fact_extraction_attempt_count + 1"
        in message_query
    )
    assert "fact_extraction_status = CASE" in message_query
    assert message_args == (MESSAGE_ID, "invalid response", True)

    query, args = connection.calls[1]
    assert "INSERT INTO session_background_task_progress" in query
    assert "'fact_extraction'" in query
    assert "through_sequence = GREATEST" in query
    assert "last_error = NULL" in query
    assert "updated_at = now()" in query
    assert args == (SESSION_ID, 7)


@pytest.mark.asyncio
async def test_get_progress_returns_non_latest_target_and_pending_count() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "task_name": "fact_extraction",
            "through_sequence": 7,
            "target_sequence": 9,
            "pending": 2,
            "last_error": None,
            "updated_at": NOW,
        }
    )
    store = store_with(connection)

    progress = await store.get_progress(FISH_ID, SESSION_ID)

    assert progress == BackgroundTaskProgress(
        task_name="fact_extraction",
        through_sequence=7,
        target_sequence=9,
        pending=2,
        last_error=None,
        updated_at=NOW,
    )
    query, args = connection.calls[0]
    assert "max(message.sequence)" in query
    assert "message.sequence < bounds.max_sequence" in query
    assert "AS waiting_sequence" in query
    assert "message.sequence < bounds.waiting_sequence" in query
    assert "message.sequence > COALESCE(progress.through_sequence, 0)" in query
    assert "message.sequence <= target.target_sequence" in query
    assert "session_background_task_progress" in query
    assert args == (FISH_ID, SESSION_ID)
