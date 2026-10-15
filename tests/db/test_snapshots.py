from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import asyncpg
import pytest

from fish_memory.db.snapshots import SnapshotStore
from fish_memory.models import Role
from fish_memory.snapshots.models import (
    NormalizedDynamicSnapshot,
    NormalizedSnapshotOverview,
    NormalizedStaticProfile,
    PendingNormalizedSnapshotBatch,
    PendingNormalizedStaticProfileBatch,
    SnapshotFact,
    SnapshotItemMutation,
    SnapshotItemState,
)

FISH_ID = UUID(int=1)
MESSAGE_ID = UUID(int=2)
FACT_ID = UUID(int=3)
SESSION_ID = UUID(int=4)
SECOND_FACT_ID = UUID(int=5)
THIRD_FACT_ID = UUID(int=6)
NOW = datetime(2026, 7, 24, tzinfo=UTC)


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
        self.events: list[str] = []

    def _record(self, query: str, args: tuple[Any, ...]) -> str:
        normalized = " ".join(query.split())
        self.calls.append((normalized, args))
        self.events.append(normalized.split()[0])
        return normalized

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        self._record(query, args)
        return self.fetch_results.pop(0)

    async def fetchrow(
        self,
        query: str,
        *args: Any,
    ) -> dict[str, object] | None:
        self._record(query, args)
        return self.fetchrow_results.pop(0)

    async def execute(self, query: str, *args: Any) -> str:
        self._record(query, args)
        return "OK"

    def transaction(self) -> FakeContext:
        return FakeContext(None, self.events, "transaction")


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeContext:
        return FakeContext(self.connection, self.connection.events, "acquire")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, object]]:
        return await self.connection.fetch(query, *args)

    async def fetchrow(
        self,
        query: str,
        *args: Any,
    ) -> dict[str, object] | None:
        return await self.connection.fetchrow(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return await self.connection.execute(query, *args)


def store_with(connection: FakeConnection) -> SnapshotStore:
    return SnapshotStore(cast(asyncpg.Pool, FakePool(connection)))


def fact_record() -> dict[str, object]:
    return {
        "fish_id": FISH_ID,
        "session_id": SESSION_ID,
        "fact_id": FACT_ID,
        "fact_sequence": 7,
        "source_message_id": MESSAGE_ID,
        "source_role": "user",
        "text": "正在实现快照",
        "occurred_start": None,
        "occurred_end": None,
        "mentioned_at": NOW,
        "entities": ["快照"],
        "snapshot_through_sequence": 6,
        "snapshot_version": 2,
    }


@pytest.mark.asyncio
async def test_load_support_facts_uses_one_ordered_fish_scoped_query() -> None:
    """Catches per-item queries, reordered UUIDs, or invalidated-Fact filtering."""
    connection = FakeConnection()
    second = {
        **fact_record(),
        "fact_id": SECOND_FACT_ID,
        "fact_sequence": 8,
        "text": "第二条支持",
        "entities": ["第二"],
        "invalidated_at": NOW,
    }
    first = {
        **fact_record(),
        "text": "第一条支持",
        "entities": ["第一"],
        "invalidated_at": None,
    }
    connection.fetch_results.append([second, first])

    facts = await store_with(connection).load_support_facts(
        FISH_ID,
        (SECOND_FACT_ID, FACT_ID),
    )

    assert [fact.id for fact in facts] == [SECOND_FACT_ID, FACT_ID]
    assert [fact.text for fact in facts] == ["第二条支持", "第一条支持"]
    assert len(connection.calls) == 1
    query, args = connection.calls[0]
    assert "WITH ORDINALITY" in query
    assert "fact.fish_id = $1" in query
    assert "ORDER BY requested.ordinality" in query
    assert "invalidated_at" not in query
    assert args == (FISH_ID, [SECOND_FACT_ID, FACT_ID])


@pytest.mark.asyncio
async def test_load_support_facts_fails_closed_when_any_fact_is_missing() -> None:
    """Catches a partial evidence set reaching the confirmer."""
    connection = FakeConnection()
    connection.fetch_results.append([fact_record()])

    with pytest.raises(ValueError, match="support facts do not belong"):
        await store_with(connection).load_support_facts(
            FISH_ID,
            (FACT_ID, SECOND_FACT_ID),
        )


@pytest.mark.asyncio
async def test_load_recent_turns_for_fish_resolves_anchor_session() -> None:
    """Catches Static confirmation reusing a conversation from another Session."""
    connection = FakeConnection()
    connection.fetchrow_results.append({"session_id": SESSION_ID, "sequence": 12})
    connection.fetch_results.append(
        [
            {"role": "user", "visible_content": "same-session question"},
            {"role": "assistant", "visible_content": "same-session answer"},
        ]
    )

    turns = await store_with(connection).load_recent_turns_for_fish(
        FISH_ID,
        MESSAGE_ID,
    )

    assert turns == (
        {"role": "user", "content": "same-session question"},
        {"role": "assistant", "content": "same-session answer"},
    )
    anchor_query, anchor_args = connection.calls[0]
    assert "JOIN fish_sessions AS session" in anchor_query
    assert "session.fish_id = $1" in anchor_query
    assert "message.id = $2" in anchor_query
    assert anchor_args == (FISH_ID, MESSAGE_ID)
    assert connection.calls[1][1] == (SESSION_ID, 12, 5)


@pytest.mark.asyncio
async def test_load_recent_turns_for_fish_returns_five_user_led_turns() -> None:
    """Catches a context loader that leaks later/cross-session turns or raw output."""
    connection = FakeConnection()
    connection.fetchrow_results.append({"session_id": SESSION_ID, "sequence": 12})
    connection.fetch_results.append(
        [
            {"role": "user", "visible_content": "turn 2"},
            {"role": "assistant", "visible_content": "reply 2"},
            {"role": "user", "visible_content": "turn 3"},
            {"role": "assistant", "visible_content": "reply 3"},
            {"role": "user", "visible_content": "turn 4"},
            {"role": "assistant", "visible_content": "reply 4"},
            {"role": "user", "visible_content": "turn 5"},
            {"role": "assistant", "visible_content": "reply 5"},
            {"role": "user", "visible_content": "turn 6"},
            {"role": "assistant", "visible_content": "truncated reply"},
        ]
    )

    turns = await store_with(connection).load_recent_turns_for_fish(
        FISH_ID,
        MESSAGE_ID,
    )

    assert turns == (
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "content": "reply 3"},
        {"role": "user", "content": "turn 4"},
        {"role": "assistant", "content": "reply 4"},
        {"role": "user", "content": "turn 5"},
        {"role": "assistant", "content": "reply 5"},
        {"role": "user", "content": "turn 6"},
        {"role": "assistant", "content": "truncated reply"},
    )
    anchor_query, anchor_args = connection.calls[0]
    assert "session.fish_id = $1" in anchor_query
    assert "message.id = $2" in anchor_query
    assert anchor_args == (FISH_ID, MESSAGE_ID)
    window_query, window_args = connection.calls[1]
    assert "row_number()" in window_query
    assert "message.role = 'user'" in window_query
    assert "message.sequence <= $2" in window_query
    assert "message.session_id = $1" in window_query
    assert "LEFT(message.content, message.consumed_offset)" in window_query
    assert "ORDER BY message.sequence" in window_query
    assert window_args == (SESSION_ID, 12, 5)


@pytest.mark.asyncio
async def test_load_recent_turns_for_fish_rejects_foreign_anchor() -> None:
    """Catches a caller using an assigned Fact from another Fish as context."""
    connection = FakeConnection()
    connection.fetchrow_results.append(None)

    with pytest.raises(ValueError, match="does not belong"):
        await store_with(connection).load_recent_turns_for_fish(FISH_ID, MESSAGE_ID)

    assert len(connection.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_load_recent_turns_for_fish_rejects_nonpositive_limit(limit: int) -> None:
    """Catches a zero or negative turn limit silently widening the context window."""
    connection = FakeConnection()

    with pytest.raises(ValueError, match="limit must be positive"):
        await store_with(connection).load_recent_turns_for_fish(
            FISH_ID,
            MESSAGE_ID,
            limit=limit,
        )

    assert connection.calls == []


@pytest.mark.asyncio
async def test_load_recent_turns_for_fish_honors_a_single_turn_limit() -> None:
    """Catches a loader that treats an explicit one-turn limit as the default five."""
    connection = FakeConnection()
    connection.fetchrow_results.append({"session_id": SESSION_ID, "sequence": 8})
    connection.fetch_results.append(
        [
            {"role": "user", "visible_content": "latest turn"},
            {"role": "assistant", "visible_content": "latest reply"},
        ]
    )

    turns = await store_with(connection).load_recent_turns_for_fish(
        FISH_ID,
        MESSAGE_ID,
        limit=1,
    )

    assert turns == (
        {"role": "user", "content": "latest turn"},
        {"role": "assistant", "content": "latest reply"},
    )
    assert connection.calls[1][1] == (SESSION_ID, 8, 1)


def normalized_item(
    *,
    item_id: int = 41,
    item_version: int = 3,
    support_fact_ids: tuple[UUID, ...] = (FACT_ID,),
    lineage_fact_ids: tuple[UUID, ...] = (FACT_ID,),
    summary: str = "旧状态",
    last_supported_at: datetime = NOW,
) -> SnapshotItemState:
    return SnapshotItemState(
        id=item_id,
        summary=summary,
        support_fact_ids=support_fact_ids,
        lineage_fact_ids=lineage_fact_ids,
        last_supported_at=last_supported_at,
        item_version=item_version,
    )


def normalized_dynamic_batch(
    *,
    version: int = 2,
    items: tuple[SnapshotItemState, ...] | None = None,
    through_fact_sequence: int | None = None,
    target_fact_sequence: int = 7,
) -> PendingNormalizedSnapshotBatch:
    state_items = (normalized_item(),) if items is None else items
    effective_through_sequence = (
        (6 if version else 0)
        if through_fact_sequence is None
        else through_fact_sequence
    )
    return PendingNormalizedSnapshotBatch(
        fish_id=FISH_ID,
        session_id=SESSION_ID,
        facts=(
            SnapshotFact(
                id=SECOND_FACT_ID,
                sequence=target_fact_sequence,
                fish_id=FISH_ID,
                source_message_id=MESSAGE_ID,
                source_role=Role.USER,
                text="新状态",
                occurred_start=None,
                occurred_end=None,
                mentioned_at=NOW,
                entities=("状态",),
            ),
        ),
        snapshot=NormalizedDynamicSnapshot(
            items=state_items,
            through_fact_sequence=effective_through_sequence,
            version=version,
        ),
        target_fact_sequence=target_fact_sequence,
        trigger_message_id=MESSAGE_ID,
    )


def normalized_static_batch(
    *,
    version: int = 2,
    items: tuple[SnapshotItemState, ...] | None = None,
) -> PendingNormalizedStaticProfileBatch:
    state_items = (normalized_item(),) if items is None else items
    return PendingNormalizedStaticProfileBatch(
        fish_id=FISH_ID,
        facts=normalized_dynamic_batch(version=version, items=state_items).facts,
        profile=NormalizedStaticProfile(
            items=state_items,
            through_fact_sequence=6 if version else 0,
            version=version,
        ),
        target_fact_sequence=7,
        trigger_message_id=MESSAGE_ID,
    )


def update_mutation(
    *,
    item_id: int = 41,
    summary: str = "新状态",
    support_fact_ids: tuple[UUID, ...] = (SECOND_FACT_ID,),
    lineage_fact_ids: tuple[UUID, ...] = (FACT_ID, SECOND_FACT_ID),
    new_support_fact_ids: tuple[UUID, ...] = (SECOND_FACT_ID,),
    removed_support_fact_ids: tuple[UUID, ...] = (FACT_ID,),
) -> SnapshotItemMutation:
    return SnapshotItemMutation(
        action="update",
        item_id=item_id,
        basis="新事实替换旧状态",
        summary=summary,
        support_fact_ids=support_fact_ids,
        lineage_fact_ids=lineage_fact_ids,
        new_support_fact_ids=new_support_fact_ids,
        removed_support_fact_ids=removed_support_fact_ids,
        last_supported_at=NOW,
    )


def create_mutation(
    *,
    fact_id: UUID = SECOND_FACT_ID,
    summary: str = "新条目",
) -> SnapshotItemMutation:
    return SnapshotItemMutation(
        action="create",
        item_id=None,
        basis=None,
        summary=summary,
        support_fact_ids=(fact_id,),
        lineage_fact_ids=(fact_id,),
        new_support_fact_ids=(fact_id,),
        removed_support_fact_ids=(),
        last_supported_at=NOW,
    )


def normalized_item_payload(
    item: SnapshotItemState,
) -> dict[str, object]:
    return {
        "id": item.id,
        "summary": item.summary,
        "supportFactIds": [str(value) for value in item.support_fact_ids],
        "lineageFactIds": [str(value) for value in item.lineage_fact_ids],
        "lastSupportedAt": item.last_supported_at.isoformat(),
        "itemVersion": item.item_version,
    }


@pytest.mark.asyncio
async def test_lists_only_v2_pending_normalized_dynamic_batches() -> None:
    connection = FakeConnection()
    row = fact_record()
    row["snapshot_items"] = [normalized_item_payload(normalized_item())]
    row["fact_id"] = SECOND_FACT_ID
    row["text"] = "新状态"
    row["entities"] = ["状态"]
    connection.fetch_results.append([row])

    batches = await store_with(connection).list_pending_normalized_batches(10, 20)

    assert batches == [normalized_dynamic_batch()]
    query, args = connection.calls[0]
    assert "fish.snapshot_storage_version = 2" in query
    assert "fish.background_owner IS NOT DISTINCT FROM $3" in query
    assert "snapshot.snapshot_type = 'dynamic'" in query
    assert "ORDER BY item.last_supported_at, item.id" in query
    assert args == (10, 20, None)


@pytest.mark.asyncio
async def test_lists_only_v2_pending_normalized_static_batches() -> None:
    connection = FakeConnection()
    row = fact_record()
    row["snapshot_items"] = [normalized_item_payload(normalized_item())]
    row["fact_id"] = SECOND_FACT_ID
    row["text"] = "新状态"
    row["entities"] = ["状态"]
    connection.fetch_results.append([row])

    batches = await store_with(connection).list_pending_normalized_static_batches(
        10,
        20,
    )

    assert batches == [normalized_static_batch()]
    query, args = connection.calls[0]
    assert "fish.snapshot_storage_version = 2" in query
    assert "fish.background_owner IS NOT DISTINCT FROM $3" in query
    assert "snapshot.snapshot_type = 'static'" in query
    assert "snapshot.session_id IS NULL" in query
    assert args == (10, 20, None)


@pytest.mark.asyncio
async def test_persist_mutations_keeps_item_id_and_audits_support_changes() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            {"id": 9, "version": 3},
            {"id": 88},
            {"id": 41, "item_version": 4},
        ]
    )
    connection.fetch_results.append([{"id": 41, "item_version": 3}])

    persisted = await store_with(connection).persist_mutations(
        normalized_dynamic_batch(),
        (update_mutation(),),
    )

    assert persisted is True
    assert connection.events[0:3] == [
        "acquire:enter",
        "transaction:enter",
        "UPDATE",
    ]
    header_query, header_args = connection.calls[0]
    assert "snapshot.through_fact_sequence = $6" in header_query
    assert "FROM fish_sessions AS session" in header_query
    assert "session.fish_id = fish.id" in header_query
    assert header_args == (FISH_ID, SESSION_ID, "dynamic", 7, 2, 6, None)
    queries = [query for query, _ in connection.calls]
    assert all("memory_snapshot_versions" not in query for query in queries)
    assert all("invalidated_at" not in query for query in queries)
    assert all("UPDATE facts" not in query for query in queries)

    item_update = next(
        call
        for call in connection.calls
        if call[0].startswith("UPDATE memory_snapshot_items")
    )
    assert "snapshot_id = $2" in item_update[0]
    assert "item_version = $3" in item_update[0]
    assert item_update[1] == (
        41,
        9,
        3,
        "新状态",
        [SECOND_FACT_ID],
        [FACT_ID, SECOND_FACT_ID],
        NOW,
    )

    item_version = next(
        call
        for call in connection.calls
        if call[0].startswith("INSERT INTO memory_snapshot_item_versions")
    )
    assert item_version[1] == (
        41,
        88,
        3,
        4,
        "update",
        "新事实替换旧状态",
        "新状态",
        [SECOND_FACT_ID],
        [FACT_ID, SECOND_FACT_ID],
        [SECOND_FACT_ID],
        [FACT_ID],
        NOW,
    )
    assert connection.events[-3:] == [
        "INSERT",
        "transaction:exit",
        "acquire:exit",
    ]


@pytest.mark.asyncio
async def test_persist_mutations_applies_all_updates_before_creates() -> None:
    second_item = normalized_item(
        item_id=42,
        item_version=7,
        support_fact_ids=(SECOND_FACT_ID,),
        lineage_fact_ids=(SECOND_FACT_ID,),
        summary="第二旧状态",
    )
    batch = normalized_dynamic_batch(items=(normalized_item(), second_item))
    second_update = update_mutation(
        item_id=42,
        summary="第二新状态",
        support_fact_ids=(THIRD_FACT_ID,),
        lineage_fact_ids=(SECOND_FACT_ID, THIRD_FACT_ID),
        new_support_fact_ids=(THIRD_FACT_ID,),
        removed_support_fact_ids=(SECOND_FACT_ID,),
    )
    connection = FakeConnection()
    connection.fetchrow_results.extend(
        [
            {"id": 9, "version": 3},
            {"id": 88},
            {"id": 41, "item_version": 4},
            {"id": 42, "item_version": 8},
            {"id": 43, "item_version": 1},
        ]
    )
    connection.fetch_results.append(
        [
            {"id": 41, "item_version": 3},
            {"id": 42, "item_version": 7},
        ]
    )

    assert await store_with(connection).persist_mutations(
        batch,
        (create_mutation(), second_update, update_mutation()),
    )

    current_item_calls = [
        (index, query)
        for index, (query, _) in enumerate(connection.calls)
        if query.startswith("UPDATE memory_snapshot_items")
        or query.startswith("INSERT INTO memory_snapshot_items")
    ]
    assert [query.split()[0] for _, query in current_item_calls] == [
        "UPDATE",
        "UPDATE",
        "INSERT",
    ]
    version_calls = [
        args
        for query, args in connection.calls
        if query.startswith("INSERT INTO memory_snapshot_item_versions")
    ]
    assert len(version_calls) == 3
    assert {(args[0], args[4]) for args in version_calls} == {
        (41, "update"),
        (42, "update"),
        (43, "create"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [0, 2])
async def test_persist_mutations_advances_watermark_without_revision(
    version: int,
) -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {"id": 9, "version": 1 if version == 0 else version}
    )
    batch = normalized_dynamic_batch(version=version, items=())

    assert await store_with(connection).persist_mutations(batch, ()) is True

    queries = [query for query, _ in connection.calls]
    header_query = queries[0]
    if version == 0:
        assert header_query.startswith("INSERT INTO memory_snapshots")
        assert "NULL" in header_query
    else:
        assert header_query.startswith("UPDATE memory_snapshots")
        assert "version = version + 1" not in header_query
    assert not any("memory_snapshot_revisions" in query for query in queries)
    assert not any("memory_snapshot_items" in query for query in queries)
    assert connection.calls[-1][1] == (SESSION_ID, "dynamic_snapshot", 7)


@pytest.mark.asyncio
async def test_persist_mutations_returns_false_on_header_version_conflict() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(None)

    persisted = await store_with(connection).persist_mutations(
        normalized_dynamic_batch(),
        (update_mutation(),),
    )

    assert persisted is False
    assert len(connection.calls) == 1
    assert "session_background_task_progress" not in connection.calls[0][0]


@pytest.mark.asyncio
async def test_initial_dynamic_header_requires_session_owned_by_fish() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append(None)

    persisted = await store_with(connection).persist_mutations(
        normalized_dynamic_batch(version=0, items=()),
        (),
    )

    assert persisted is False
    query, args = connection.calls[0]
    assert "JOIN fish_sessions AS session" in query
    assert "session.fish_id = fish.id" in query
    assert args == (FISH_ID, SESSION_ID, "dynamic", 7, None)
    assert len(connection.calls) == 1


@pytest.mark.asyncio
async def test_persist_mutations_rejects_changed_item_version_atomically() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append({"id": 9, "version": 3})
    connection.fetch_results.append([{"id": 41, "item_version": 4}])

    persisted = await store_with(connection).persist_mutations(
        normalized_dynamic_batch(),
        (update_mutation(),),
    )

    assert persisted is False
    assert len(connection.calls) == 2
    assert "FOR UPDATE" in connection.calls[1][0]
    assert "memory_snapshot_revisions" not in connection.calls[-1][0]


@pytest.mark.asyncio
async def test_persist_mutations_uses_static_scope_and_progress() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.append({"id": 9, "version": 2})

    assert await store_with(connection).persist_mutations(
        normalized_static_batch(items=()),
        (),
    )

    header_query, header_args = connection.calls[0]
    assert "session_id IS NOT DISTINCT FROM $2" in header_query
    assert "fish.background_owner IS NOT DISTINCT FROM" in header_query
    assert header_args[:3] == (FISH_ID, None, "static")
    assert connection.calls[-1][1] == (FISH_ID, "static_profile", 7)


@pytest.mark.asyncio
async def test_loads_normalized_dynamic_snapshot_and_filters_max_age_in_domain() -> (
    None
):
    recent = normalized_item(item_id=42, item_version=1, summary="最近")
    old = normalized_item(
        item_id=41,
        item_version=2,
        summary="较早",
        last_supported_at=NOW - timedelta(days=31),
    )
    connection = FakeConnection()
    connection.fetchrow_results.append(
        {
            "through_fact_sequence": 7,
            "version": 3,
            "snapshot_items": [
                normalized_item_payload(old),
                normalized_item_payload(recent),
            ],
        }
    )

    snapshot = await store_with(connection).load_normalized_dynamic_snapshot(
        FISH_ID,
        SESSION_ID,
    )

    assert snapshot is not None
    assert snapshot == NormalizedDynamicSnapshot(
        items=(old, recent),
        through_fact_sequence=7,
        version=3,
    )
    assert snapshot.active_items(NOW, timedelta(days=30)) == (recent,)
    query, args = connection.calls[0]
    assert "fish.snapshot_storage_version = 2" in query
    assert "ORDER BY item.last_supported_at, item.id" in query
    assert "last_supported_at >=" not in query
    assert args == (FISH_ID, SESSION_ID)


@pytest.mark.asyncio
async def test_normalized_reads_return_no_aggregate_for_v1_fish() -> None:
    connection = FakeConnection()
    connection.fetchrow_results.extend([None, None])
    connection.fetch_results.append([])

    dynamic = await store_with(connection).load_normalized_dynamic_snapshot(
        FISH_ID,
        SESSION_ID,
    )
    static = await store_with(connection).load_normalized_static_profile(FISH_ID)
    overviews = await store_with(connection).list_normalized_current_overviews(FISH_ID)

    assert dynamic is None
    assert static is None
    assert overviews == []
    assert all(
        "fish.snapshot_storage_version = 2" in query for query, _ in connection.calls
    )


@pytest.mark.asyncio
async def test_lists_normalized_overviews_with_category_free_items() -> None:
    item = normalized_item(summary="规范化条目")
    connection = FakeConnection()
    connection.fetch_results.append(
        [
            {
                "snapshot_type": "static",
                "through_fact_sequence": 7,
                "version": 3,
                "updated_at": NOW,
                "snapshot_items": [normalized_item_payload(item)],
            }
        ]
    )

    overviews = await store_with(connection).list_normalized_current_overviews(FISH_ID)

    assert overviews == [
        NormalizedSnapshotOverview(
            snapshot_type="static",
            version=3,
            updated_at=NOW,
            through_fact_sequence=7,
            items=(item,),
        )
    ]
    query, args = connection.calls[0]
    assert "fish.snapshot_storage_version = 2" in query
    assert "ORDER BY item.last_supported_at, item.id" in query
    assert "last_supported_at >=" not in query
    assert args == (FISH_ID,)
