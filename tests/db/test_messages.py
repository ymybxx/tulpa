from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db import messages as messages_module
from fish_memory.db.embedding_spaces import EmbeddingSpaceMismatchError
from fish_memory.db.messages import (
    LegacyFishReadOnlyError,
    MessagesStore,
)
from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.models import MessageStatus, Role

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000002")
SESSION_ID = UUID("00000000-0000-0000-0000-000000000003")
SECOND_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000004")
SECOND_FISH_ID = UUID("00000000-0000-0000-0000-000000000005")
SECOND_SESSION_ID = UUID("00000000-0000-0000-0000-000000000006")
NOW = datetime(2026, 7, 19, tzinfo=UTC)
EMBEDDING_SPACE = EmbeddingSpace(
    provider="local",
    model="intfloat/multilingual-e5-small",
    dimensions=384,
)


def fish_record(
    *,
    deleted_at: datetime | None = None,
    snapshot_storage_version: int = 2,
) -> dict[str, object]:
    return {
        "deleted_at": deleted_at,
        "snapshot_storage_version": snapshot_storage_version,
        "embedding_provider": EMBEDDING_SPACE.provider,
        "embedding_model": EMBEDDING_SPACE.model,
        "embedding_dimensions": EMBEDDING_SPACE.dimensions,
    }


def record(
    *,
    message_id: UUID = MESSAGE_ID,
    fish_id: UUID = FISH_ID,
    session_id: UUID = SESSION_ID,
    sequence: int = 1,
    role: str = "user",
    content: str = "hello",
    status: str = "initialized",
    context_through_sequence: int | None = None,
    consumed_offset: int | None = None,
    finished_at: datetime | None = None,
    history_token_count: int = 0,
    is_assistant: bool = False,
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
        "context_through_sequence": context_through_sequence,
        "consumed_offset": consumed_offset,
        "finished_at": finished_at,
        "history_token_count": history_token_count,
        "is_assistant": is_assistant,
    }


class FakeContext:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def __aenter__(self) -> Any:
        return self.value

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
        self.fetchrow_results: list[dict[str, object] | None] = []
        self.fetch_results: list[list[dict[str, object]]] = []
        self.fetchval_results: list[Any] = []

    def _record(self, query: str, args: tuple[Any, ...]) -> None:
        self.calls.append((" ".join(query.split()), args))

    async def fetchrow(self, query: str, *args: Any) -> dict[str, object] | None:
        self._record(query, args)
        if not self.fetchrow_results and "history_window" in query:
            return {
                "history_window_start_sequence": 1,
                "history_window_token_count": 0,
            }
        return self.fetchrow_results.pop(0)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        self._record(query, args)
        return self.fetch_results.pop(0)

    async def fetchval(self, query: str, *args: Any) -> Any:
        self._record(query, args)
        return self.fetchval_results.pop(0)

    async def execute(self, query: str, *args: Any) -> str:
        self._record(query, args)
        return "OK"

    async def executemany(
        self,
        query: str,
        args: list[tuple[Any, ...]],
    ) -> None:
        self._record(query, (args,))

    def transaction(self) -> FakeContext:
        return FakeContext(None)


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeContext:
        return FakeContext(self.connection)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, object] | None:
        return await self.connection.fetchrow(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        return await self.connection.fetch(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self.connection.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return await self.connection.execute(query, *args)


def store_with(connection: FakeConnection) -> MessagesStore:
    return MessagesStore(
        cast(asyncpg.Pool, FakePool(connection)),
        embedding_space=EMBEDDING_SPACE,
    )


@pytest.mark.asyncio
async def test_first_user_message_creates_a_session_and_returns_its_id() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            fish_record(),
            None,
            {
                "id": SESSION_ID,
                "fish_id": FISH_ID,
                "created_at": NOW,
                "last_message_at": None,
                "history_window_start_sequence": 0,
                "history_window_token_count": 0,
            },
            record(),
        ]
    )
    store = store_with(connection)

    message = await store.insert_user_message(FISH_ID, "hello", created_at=NOW)

    queries = [query for query, _args in connection.calls]
    assert any("FROM fish_sessions" in query for query in queries)
    assert any("INSERT INTO fish_sessions" in query for query in queries)
    insert_query = next(query for query in queries if "INSERT INTO messages" in query)
    assert "session_id" in insert_query
    assert message.session_id == SESSION_ID


@pytest.mark.asyncio
async def test_inserts_user_message_and_returns_typed_model() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [fish_record(), {"id": SESSION_ID}, record(content=" original content ")]
    )
    store = store_with(connection)

    message = await store.insert_user_message(FISH_ID, " original content ")

    queries = [query for query, _ in connection.calls]
    assert "INSERT INTO fishes" in queries[0]
    assert "ON CONFLICT DO NOTHING" in queries[0]
    assert "embedding_dimensions" in queries[0]
    assert connection.calls[0][1] == (
        FISH_ID,
        "local",
        "intfloat/multilingual-e5-small",
        384,
    )
    assert "SELECT" in queries[1] and "deleted_at" in queries[1]
    assert "snapshot_storage_version" in queries[1]
    assert "FOR UPDATE" in queries[1]
    assert "snapshot_storage_version" not in queries[0]
    query, args = connection.calls[3]
    assert "INSERT INTO messages" in query
    assert args[1:4] == (FISH_ID, SESSION_ID, " original content ")
    assert args[-1] > 0
    assert message.id == MESSAGE_ID
    assert message.role is Role.USER
    assert message.status is MessageStatus.INITIALIZED


@pytest.mark.asyncio
async def test_inserts_user_message_at_explicit_time() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [fish_record(), {"id": SESSION_ID}, record(content="timestamped")]
    )
    store = store_with(connection)
    created_at = datetime(2023, 5, 20, 2, 21, tzinfo=UTC)

    await store.insert_user_message(
        FISH_ID,
        "timestamped",
        created_at=created_at,
    )

    query, args = connection.calls[3]
    assert "created_at" in query
    assert args[1:5] == (FISH_ID, SESSION_ID, "timestamped", created_at)
    assert args[-1] > 0


@pytest.mark.asyncio
async def test_reusing_explicit_user_id_skips_insert_only_side_effects() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            fish_record(),
            {"id": SESSION_ID},
            None,
            record(),
        ]
    )
    store = store_with(connection)

    message = await store.insert_user_message(
        FISH_ID,
        "hello",
        session_id=SESSION_ID,
        message_id=MESSAGE_ID,
    )

    queries = [query for query, _args in connection.calls]
    insert_query, insert_args = next(
        (query, args)
        for query, args in connection.calls
        if "INSERT INTO messages" in query
    )
    assert "ON CONFLICT (id) DO NOTHING" in insert_query
    assert insert_args[0] == MESSAGE_ID
    assert message.id == MESSAGE_ID
    assert all("UPDATE fish_sessions" not in query for query in queries)
    assert all("history_window" not in query for query in queries)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing",
    [
        record(fish_id=SECOND_FISH_ID),
        record(session_id=SECOND_SESSION_ID),
        record(role="assistant"),
        record(content="different"),
        record(status="generating"),
    ],
)
async def test_reusing_explicit_user_id_with_different_identity_raises(
    existing: dict[str, object],
) -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [fish_record(), {"id": SESSION_ID}, None, existing]
    )
    store = store_with(connection)

    with pytest.raises(
        messages_module.MessageIdentityConflictError, match=str(MESSAGE_ID)
    ):
        await store.insert_user_message(
            FISH_ID,
            "hello",
            session_id=SESSION_ID,
            message_id=MESSAGE_ID,
        )


@pytest.mark.asyncio
async def test_reusing_explicit_answer_id_with_different_context_raises() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            None,
            record(
                role="assistant",
                content="",
                status="generating",
                context_through_sequence=11,
                consumed_offset=-1,
            ),
        ]
    )
    store = store_with(connection)

    with pytest.raises(
        messages_module.MessageIdentityConflictError, match=str(MESSAGE_ID)
    ):
        await store.insert_generating_answer(
            FISH_ID,
            SESSION_ID,
            12,
            message_id=MESSAGE_ID,
        )


@pytest.mark.asyncio
async def test_rejects_user_message_for_deleted_fish() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(fish_record(deleted_at=NOW))
    store = store_with(connection)

    with pytest.raises(messages_module.FishDeletedError):
        await store.insert_user_message(FISH_ID, "hello")

    assert all("INSERT INTO messages" not in query for query, _ in connection.calls)


@pytest.mark.asyncio
async def test_rejects_user_message_for_legacy_fish_before_creating_session() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(fish_record(snapshot_storage_version=1))
    store = store_with(connection)

    with pytest.raises(LegacyFishReadOnlyError, match=str(FISH_ID)):
        await store.insert_user_message(FISH_ID, "hello")

    queries = [query for query, _ in connection.calls]
    assert all("fish_sessions" not in query for query in queries)
    assert all("INSERT INTO messages" not in query for query in queries)


@pytest.mark.asyncio
async def test_deleted_fish_takes_precedence_over_legacy_read_only() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        fish_record(deleted_at=NOW, snapshot_storage_version=1)
    )
    store = store_with(connection)

    with pytest.raises(messages_module.FishDeletedError):
        await store.insert_user_message(FISH_ID, "hello")


@pytest.mark.asyncio
async def test_rejects_switching_an_existing_fish_embedding_space() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "deleted_at": None,
            "snapshot_storage_version": 2,
            "embedding_provider": "openai-compatible",
            "embedding_model": "text-embedding-v4",
            "embedding_dimensions": 1024,
        }
    )
    store = store_with(connection)

    with pytest.raises(EmbeddingSpaceMismatchError, match="cannot switch"):
        await store.insert_user_message(FISH_ID, "hello")

    assert all("INSERT INTO messages" not in query for query, _ in connection.calls)


@pytest.mark.asyncio
async def test_soft_deletes_only_active_fish() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append({"id": FISH_ID})
    store = store_with(connection)

    deleted = await store.soft_delete_fish(FISH_ID)

    query, args = connection.calls[0]
    assert "UPDATE fishes" in query
    assert "deleted_at IS NULL" in query
    assert args == (FISH_ID,)
    assert deleted is True


@pytest.mark.asyncio
async def test_soft_delete_returns_false_when_fish_is_not_active() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(None)
    store = store_with(connection)

    assert await store.soft_delete_fish(FISH_ID) is False


@pytest.mark.asyncio
async def test_inserts_generating_answer_with_default_progress() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        record(
            role="assistant",
            content="",
            status="generating",
            context_through_sequence=12,
            consumed_offset=-1,
        )
    )
    store = store_with(connection)

    message = await store.insert_generating_answer(
        FISH_ID,
        SESSION_ID,
        12,
        message_id=MESSAGE_ID,
    )

    insert_query, args = connection.calls[0]
    update_query, update_args = connection.calls[1]
    assert "INSERT INTO messages" in insert_query
    assert "ON CONFLICT (id) DO NOTHING" in insert_query
    assert args[0] == MESSAGE_ID
    assert args[1:] == (FISH_ID, SESSION_ID, 12, -1)
    assert "UPDATE fish_sessions" in update_query
    assert "last_message_at" in update_query
    assert update_args == (SESSION_ID, NOW)
    assert message.session_id == SESSION_ID
    assert message.status is MessageStatus.GENERATING
    assert message.consumed_offset == -1


@pytest.mark.asyncio
async def test_agent_message_starts_completed_behind_consumption_barrier() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        record(
            role="assistant",
            content="agent result",
            status="completed",
            context_through_sequence=12,
            consumed_offset=0,
            history_token_count=0,
        )
    )
    store = store_with(connection)

    message = await store.insert_agent_message_on_connection(
        cast(asyncpg.Connection, connection),
        FISH_ID,
        SESSION_ID,
        12,
        "agent result",
        message_id=MESSAGE_ID,
        created_at=NOW,
    )

    query, args = connection.calls[0]
    assert "'completed'" in query
    assert "'waiting_consumption'" in query
    assert "history_token_count" in query
    assert args == (MESSAGE_ID, FISH_ID, SESSION_ID, "agent result", NOW, 12)
    assert message.status is MessageStatus.COMPLETED
    assert message.consumed_offset == 0
    assert message.history_token_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", "   "])
async def test_agent_message_rejects_blank_content_before_database(
    content: str,
) -> None:
    connection = FakeConnection()
    store = store_with(connection)

    with pytest.raises(ValueError, match="must not be blank"):
        await store.insert_agent_message_on_connection(
            cast(asyncpg.Connection, connection),
            FISH_ID,
            SESSION_ID,
            12,
            content,
            message_id=MESSAGE_ID,
            created_at=NOW,
        )

    assert connection.calls == []


@pytest.mark.asyncio
async def test_partial_agent_consumption_does_not_release_history_or_fact_barrier() -> (
    None
):
    connection = FakeConnection()
    waiting = record(
        role="assistant",
        content="agent result",
        status="completed",
        consumed_offset=0,
        history_token_count=0,
    )
    waiting["fact_extraction_status"] = "waiting_consumption"
    partial = dict(waiting)
    partial["consumed_offset"] = 4
    connection.fetchrow_results.extend([waiting, partial])
    store = store_with(connection)

    message = await store.finalize_agent_message_consumption(
        MESSAGE_ID, FISH_ID, 4, False
    )

    assert message is not None and message.consumed_offset == 4
    update_query = connection.calls[1][0]
    assert "SET consumed_offset = GREATEST" in update_query
    assert "SET fact_extraction_status" not in update_query
    assert all("history_window" not in query for query, _ in connection.calls)


@pytest.mark.asyncio
async def test_full_agent_consumption_releases_once_and_rebuilds_history() -> None:
    connection = FakeConnection()
    waiting = record(
        role="assistant",
        content="agent result",
        status="completed",
        consumed_offset=0,
        history_token_count=0,
    )
    waiting["fact_extraction_status"] = "waiting_consumption"
    released = dict(waiting)
    released["consumed_offset"] = -1
    released["history_token_count"] = 15
    released["fact_extraction_status"] = "pending"
    connection.fetchrow_results.extend([waiting, released])
    connection.fetch_results.append([released])
    store = store_with(connection)

    message = await store.finalize_agent_message_consumption(
        MESSAGE_ID, FISH_ID, -1, False
    )

    assert message is not None and message.history_token_count == 15
    transition_query = connection.calls[1][0]
    assert "fact_extraction_status = 'pending'" in transition_query
    rebuild_query = next(
        query for query, _ in connection.calls if "ORDER BY sequence" in query
    )
    assert "fact_extraction_status <> 'waiting_consumption'" in rebuild_query


@pytest.mark.asyncio
async def test_loads_latest_valid_history_in_ascending_order() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            record(sequence=3, role="assistant", status="completed"),
            record(sequence=1),
        ]
    )
    store = store_with(connection)

    history = await store.load_history(
        FISH_ID,
        SESSION_ID,
        before_sequence=5,
        limit=500,
    )

    query, args = next(
        (query, args) for query, args in connection.calls if "FROM messages" in query
    )
    assert "session_id = $1" in query
    assert "status IN ('completed', 'interrupted')" in query
    assert "deleted_at IS NULL" in query
    assert "ORDER BY sequence DESC" in query
    assert args == (SESSION_ID, FISH_ID, 5, 1, 500)
    assert [message.sequence for message in history] == [1, 3]


@pytest.mark.asyncio
async def test_loads_agent_history_only_before_its_scoped_source_message() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            {"sequence": 5},
            {
                "history_window_start_sequence": 1,
                "history_window_token_count": 0,
            },
        ]
    )
    connection.fetch_results.append(
        [
            record(sequence=4, role="assistant", status="completed"),
            record(sequence=1),
        ]
    )
    store = store_with(connection)

    history = await store.load_history_before_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        limit=12,
    )

    source_query, source_args = connection.calls[0]
    assert "id = $1" in source_query
    assert "fish_id = $2" in source_query
    assert "session_id = $3" in source_query
    assert "role = 'user'" in source_query
    assert source_args == (MESSAGE_ID, FISH_ID, SESSION_ID)
    history_query, history_args = next(
        (query, args) for query, args in connection.calls if "sequence < $3" in query
    )
    assert "fact_extraction_status <> 'waiting_consumption'" in history_query
    assert history_args == (SESSION_ID, FISH_ID, 5, 1, 12)
    assert history is not None
    assert [message.sequence for message in history] == [1, 4]


@pytest.mark.asyncio
async def test_refuses_agent_history_when_source_is_not_in_the_run_session() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(None)

    history = await store_with(connection).load_history_before_message(
        FISH_ID,
        SESSION_ID,
        MESSAGE_ID,
        limit=12,
    )

    assert history is None
    assert len(connection.calls) == 1


@pytest.mark.asyncio
async def test_loads_agent_history_before_task_root_source_not_child_source() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            {"sequence": 5},
            {
                "history_window_start_sequence": 1,
                "history_window_token_count": 0,
            },
        ]
    )
    connection.fetch_results.append([record(sequence=1)])
    task_id = UUID("00000000-0000-0000-0000-000000000099")

    history = await store_with(connection).load_history_before_task_root_source(
        FISH_ID,
        SESSION_ID,
        task_id,
        limit=12,
    )

    root_query, root_args = connection.calls[0]
    assert "root.previous_run_id IS NULL" in root_query
    assert "root.task_id = $1" in root_query
    assert "source.session_id = $3" in root_query
    assert "another_root.previous_run_id IS NULL" in root_query
    assert root_args == (task_id, FISH_ID, SESSION_ID)
    assert history is not None
    assert [message.sequence for message in history] == [1]


@pytest.mark.asyncio
async def test_readonly_history_calculates_uninitialized_window_without_writes() -> (
    None
):
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "history_window_start_sequence": 0,
            "history_window_token_count": 0,
        }
    )
    connection.fetch_results.append(
        [
            record(sequence=1, history_token_count=12),
            record(sequence=2, history_token_count=12),
            record(sequence=3, history_token_count=12),
        ]
    )
    store = MessagesStore(
        cast(asyncpg.Pool, FakePool(connection)),
        embedding_space=EMBEDDING_SPACE,
        history_token_retain=20,
    )

    history = await store.load_history_readonly(
        FISH_ID,
        SESSION_ID,
        before_sequence=5,
        limit=500,
    )

    queries = [query for query, _args in connection.calls]
    assert all("UPDATE" not in query for query in queries)
    assert all("FOR UPDATE" not in query for query in queries)
    assert [message.sequence for message in history] == [3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("window_start", "window_total"),
    [(0, 0), (1, 24)],
    ids=["uninitialized-window", "initialized-window-rebuild"],
)
async def test_readonly_history_applies_prospective_user_to_retained_window(
    window_start: int,
    window_total: int,
) -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "history_window_start_sequence": window_start,
            "history_window_token_count": window_total,
        }
    )
    connection.fetch_results.append(
        [
            record(sequence=1, history_token_count=12),
            record(sequence=2, history_token_count=12),
        ]
    )
    store = MessagesStore(
        cast(asyncpg.Pool, FakePool(connection)),
        embedding_space=EMBEDDING_SPACE,
        history_token_limit=30,
        history_token_retain=20,
    )

    history = await store.load_history_readonly(
        FISH_ID,
        SESSION_ID,
        before_sequence=5,
        limit=500,
        appended_history_token_count=16,
    )

    queries = [query for query, _args in connection.calls]
    assert all("UPDATE" not in query for query in queries)
    assert [message.sequence for message in history] == []


@pytest.mark.asyncio
async def test_completion_becomes_interrupted_if_a_new_user_message_exists() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "context_through_sequence": 12,
            "consumed_offset": -1,
            "sequence": 13,
            "session_id": SESSION_ID,
        }
    )
    connection.fetchval_results.append(True)
    store = store_with(connection)

    status = await store.finish_answer(
        MESSAGE_ID,
        FISH_ID,
        content="partial",
        requested_status=MessageStatus.COMPLETED,
    )

    update_query, update_args = next(
        (query, args) for query, args in connection.calls if "UPDATE messages" in query
    )
    assert "UPDATE messages" in update_query
    assert update_args[3] == MessageStatus.INTERRUPTED
    assert "consumed_offset" not in update_query
    assert status is MessageStatus.INTERRUPTED


@pytest.mark.asyncio
async def test_terminal_progress_update_is_atomic_and_scoped_to_fish() -> None:
    connection = FakeConnection()
    target = record(
        role="assistant",
        content="0123456789",
        status="completed",
        consumed_offset=0,
    )
    target["fact_extraction_status"] = "pending"
    updated = dict(target)
    updated["consumed_offset"] = 9
    connection.fetchrow_results.extend([target, updated])
    store = store_with(connection)

    offset = await store.update_consumed_offset(MESSAGE_ID, FISH_ID, 9)

    lock_query, lock_args = connection.calls[0]
    query, args = connection.calls[1]
    assert "FOR UPDATE" in lock_query
    assert lock_args == (MESSAGE_ID, FISH_ID)
    assert "GREATEST(consumed_offset, $3)" in query
    assert "fish_id = $2" in query
    assert "deleted_at IS NULL" in lock_query
    assert args == (MESSAGE_ID, FISH_ID, 9)
    assert offset == 9


@pytest.mark.asyncio
async def test_progress_update_keeps_interrupted_offset_frozen() -> None:
    connection = FakeConnection()
    interrupted = record(
        role="assistant",
        content="0123456789",
        status="interrupted",
        consumed_offset=4,
    )
    interrupted["fact_extraction_status"] = "pending"
    connection.fetchrow_results.append(interrupted)
    store = store_with(connection)

    offset = await store.update_consumed_offset(MESSAGE_ID, FISH_ID, 9)

    assert len(connection.calls) == 1
    assert "FOR UPDATE" in connection.calls[0][0]
    assert offset == 4


@pytest.mark.asyncio
async def test_interrupt_terminal_answer_replaces_completed_playback_offset() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "content": "一二三四五",
            "consumed_offset": 5,
            "sequence": 3,
            "status": "completed",
            "session_id": SESSION_ID,
        }
    )
    connection.fetch_results.append([])
    store = store_with(connection)

    offset = await store.interrupt_terminal_answer(MESSAGE_ID, FISH_ID, 2)

    update_query, update_args = next(
        (query, args)
        for query, args in connection.calls
        if "UPDATE messages" in query and "status = 'interrupted'" in query
    )
    assert "consumed_offset = $3" in update_query
    assert update_args[:3] == (MESSAGE_ID, FISH_ID, 2)
    assert offset == 2


@pytest.mark.asyncio
async def test_rejects_terminal_progress_beyond_content_length() -> None:
    connection = FakeConnection()
    target = record(
        role="assistant",
        content="short",
        status="completed",
        consumed_offset=0,
    )
    target["fact_extraction_status"] = "pending"
    connection.fetchrow_results.append(target)
    store = store_with(connection)

    with pytest.raises(ValueError, match="content length"):
        await store.update_consumed_offset(MESSAGE_ID, FISH_ID, 10)


@pytest.mark.asyncio
async def test_progress_returns_none_when_answer_does_not_exist() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(None)
    store = store_with(connection)

    offset = await store.update_consumed_offset(MESSAGE_ID, FISH_ID, 1)

    assert offset is None


@pytest.mark.asyncio
async def test_marks_stale_generating_answers_failed_on_startup() -> None:
    connection = FakeConnection()
    connection.fetchval_results.append(2)
    store = store_with(connection)

    recovered = await store.fail_stale_generating_answers()

    query, _ = connection.calls[0]
    assert "status = 'generating'" in query
    assert "status = 'failed'" in query
    assert recovered == 2


@pytest.mark.asyncio
async def test_reads_latest_user_sequence_for_one_fish() -> None:
    connection = FakeConnection()
    connection.fetchval_results.append(42)
    store = store_with(connection)

    sequence = await store.get_latest_user_sequence(FISH_ID, SESSION_ID)

    query, args = connection.calls[0]
    assert "session_id = $2" in query
    assert "role = 'user'" in query
    assert "max(sequence)" in query
    assert "deleted_at IS NULL" in query
    assert args == (FISH_ID, SESSION_ID)
    assert sequence == 42


@pytest.mark.asyncio
async def test_lists_fishes_by_recent_visible_message() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            {
                "fish_id": FISH_ID,
                "last_message": "最近一条",
                "last_active_at": NOW,
                "message_count": 3,
                "is_assistant": True,
            }
        ]
    )
    store = store_with(connection)

    fishes = await store.list_fishes()

    query, args = connection.calls[0]
    assert "row_number() OVER" in query
    assert "JOIN fishes" in query
    assert "deleted_at IS NULL" in query
    assert "message.status IN" in query
    assert "'completed', 'interrupted', 'failed'" in query
    assert "deleted_at IS NULL" in query
    assert "ORDER BY summary.last_active_at DESC NULLS LAST" in query
    assert args == ()
    assert fishes[0].fish_id == FISH_ID
    assert fishes[0].last_message == "最近一条"
    assert fishes[0].last_active_at == NOW
    assert fishes[0].message_count == 3
    assert fishes[0].is_assistant is True


@pytest.mark.asyncio
async def test_soft_delete_fish_clears_assistant_binding() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append({"id": FISH_ID})
    store = store_with(connection)

    assert await store.soft_delete_fish(FISH_ID) is True

    query, args = connection.calls[0]
    assert "is_assistant = FALSE" in query
    assert args == (FISH_ID,)


@pytest.mark.asyncio
async def test_lists_recent_visible_messages_in_ascending_order() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            record(sequence=2, content="second"),
            record(sequence=4, role="assistant", content="fourth", status="failed"),
        ]
    )
    store = store_with(connection)

    messages = await store.list_messages(FISH_ID, SESSION_ID, limit=2)

    query, args = connection.calls[0]
    assert "role = 'user'" in query
    assert "status IN ('completed', 'interrupted', 'failed')" in query
    assert "ORDER BY sequence DESC" in query
    assert "ORDER BY sequence" in query
    assert "session_id = $2" in query
    assert args == (FISH_ID, SESSION_ID, 2)
    assert [message.sequence for message in messages] == [2, 4]
