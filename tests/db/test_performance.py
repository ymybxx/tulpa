import json
from datetime import UTC, datetime
from types import TracebackType
from typing import cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.performance import PerformanceStore
from fish_memory.performance.models import PerformanceStep

MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000002")
FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
NOW = datetime(2026, 7, 21, tzinfo=UTC)


class FakeConnection:
    def __init__(self) -> None:
        self.executemany_calls: list[tuple[str, list[tuple[object, ...]]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetchval_results: list[object] = []
        self.fetch_calls: list[tuple[str, tuple[object, ...]]] = []
        self.fetch_results: list[list[dict[str, object]]] = []

    async def executemany(
        self,
        query: str,
        args: list[tuple[object, ...]],
    ) -> None:
        self.executemany_calls.append((" ".join(query.split()), args))

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        self.fetch_calls.append((" ".join(query.split()), args))
        return self.fetch_results.pop(0)

    async def fetchval(self, query: str, *args: object) -> object:
        self.fetchval_calls.append((" ".join(query.split()), args))
        return self.fetchval_results.pop(0)

    async def execute(self, query: str, *args: object) -> None:
        self.execute_calls.append((" ".join(query.split()), args))

    def transaction(self) -> "FakeTransaction":
        return FakeTransaction()


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self._connection)

    async def fetch(self, query: str, *args: object) -> list[dict[str, object]]:
        return await self._connection.fetch(query, *args)


def performance_store_with(connection: FakeConnection) -> PerformanceStore:
    return PerformanceStore(cast(asyncpg.Pool, FakePool(connection)))


@pytest.mark.asyncio
async def test_inserts_all_steps_in_one_executemany_call() -> None:
    connection = FakeConnection()
    store = performance_store_with(connection)
    steps = (
        PerformanceStep(
            name="history.load",
            index=0,
            started_offset_us=100,
            duration_us=250,
        ),
        PerformanceStep(
            name="llm.answer",
            index=0,
            started_offset_us=400,
            duration_us=900,
            input_payload={"model": "qwen"},
            output_payload={"content": "你好"},
        ),
    )

    await store.insert_steps(MESSAGE_ID, steps)

    assert len(connection.executemany_calls) == 1
    query, rows = connection.executemany_calls[0]
    assert "INSERT INTO message_performance_steps" in query
    assert rows[0][0] == MESSAGE_ID
    assert rows[0][5] is None
    assert rows[0][6] is None
    assert json.loads(cast(str, rows[1][5])) == {"model": "qwen"}
    assert json.loads(cast(str, rows[1][6])) == {"content": "你好"}


@pytest.mark.asyncio
async def test_empty_insert_does_not_acquire_connection() -> None:
    connection = FakeConnection()
    store = performance_store_with(connection)

    await store.insert_steps(MESSAGE_ID, ())

    assert connection.executemany_calls == []


@pytest.mark.asyncio
async def test_lists_steps_in_timeline_order() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            {
                "step_name": "llm.answer",
                "step_index": 0,
                "started_offset_us": 400,
                "duration_us": 900,
                "input_payload": '{"model":"qwen"}',
                "output_payload": {"content": "你好"},
                "error": None,
                "created_at": NOW,
            }
        ]
    )

    steps = await performance_store_with(connection).list_steps(MESSAGE_ID)

    assert steps == [
        PerformanceStep(
            name="llm.answer",
            index=0,
            started_offset_us=400,
            duration_us=900,
            input_payload={"model": "qwen"},
            output_payload={"content": "你好"},
        )
    ]
    query, args = connection.fetch_calls[0]
    assert "ORDER BY started_offset_us, id" in query
    assert args == (MESSAGE_ID,)


@pytest.mark.asyncio
async def test_lists_message_steps_only_for_the_requested_fish() -> None:
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            {
                "step_name": "llm.dynamic_snapshot",
                "step_index": 0,
                "started_offset_us": 500,
                "duration_us": 1_200,
                "input_payload": {"messages": [{"role": "system", "content": "完整"}]},
                "output_payload": {"content": '{"creates":[]}'},
                "error": None,
            }
        ]
    )

    steps = await performance_store_with(connection).list_message_steps(
        FISH_ID,
        MESSAGE_ID,
    )

    assert steps[0].name == "llm.dynamic_snapshot"
    assert steps[0].input_payload == {
        "messages": [{"role": "system", "content": "完整"}]
    }
    query, args = connection.fetch_calls[0]
    assert "JOIN messages AS message" in query
    assert "message.fish_id = $2" in query
    assert args == (MESSAGE_ID, FISH_ID)


@pytest.mark.asyncio
async def test_replaces_only_pet_voice_metrics_for_one_owned_message() -> None:
    connection = FakeConnection()
    connection.fetchval_results.append(True)
    metrics = {
        "status": "completed",
        "nativeMs": {"endpointToPlayback": 1165.2},
    }

    stored = await performance_store_with(connection).replace_pet_voice_metrics(
        FISH_ID,
        MESSAGE_ID,
        metrics,
    )

    assert stored is True
    assert connection.fetchval_calls[0][1] == (MESSAGE_ID, FISH_ID)
    delete_query, delete_args = connection.execute_calls[0]
    insert_query, insert_args = connection.execute_calls[1]
    assert "step_name = 'voice.pet'" in delete_query
    assert "tts.client" not in delete_query
    assert delete_args == (MESSAGE_ID,)
    assert "VALUES ($1, 'voice.pet', 0, 0, $2, $3::jsonb)" in insert_query
    assert insert_args[:2] == (MESSAGE_ID, 1_165_200)
    assert json.loads(cast(str, insert_args[2])) == metrics


@pytest.mark.asyncio
async def test_missing_pet_voice_message_is_not_written() -> None:
    connection = FakeConnection()
    connection.fetchval_results.append(False)

    stored = await performance_store_with(connection).replace_pet_voice_metrics(
        FISH_ID,
        MESSAGE_ID,
        {"status": "failed", "nativeMs": {"endpointToPlayback": None}},
    )

    assert stored is False
    assert connection.execute_calls == []
